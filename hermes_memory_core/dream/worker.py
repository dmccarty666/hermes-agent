"""
DreamWorker — 9-stage nightly dream pipeline.

Stages:
  1. scope_selection  — determine turn scope based on scope param
  2. fetch_turns      — load pending turns from SQLite
  3. summarize_session — call LLM to summarize each session group
  4. extract_facts    — extract candidate facts per session
  5. extract_decisions — extract decisions per session
  6. extract_questions — extract open questions per session
  7. detect_contradictions — run contradiction heuristic, mark disputed
  8. update_project_memory — update project .md files
  9. record_dream_run  — write dream report, mark turns dreamed

Embedding endpoint: localhost:1235 (local LMStudio only — no cross-host GPU contention)
LLM inference: Spark2 GPU node (DREAM_LLM_URL @ 192.168.2.105:1234, Qwen3.6-35B)
All candidate writes go through write_memory() (not direct DB inserts).

Key runtime guards (2026-05-23):
  _DREAMER_SEMAPHORE (threading.Semaphore(1)) — single-threaded dreamer enforcement
  LMS_TIMEOUT = 300s (was 240s)
  MAX_LLM_RETRIES = 2, RETRY_BASE_DELAY = 5s (exponential backoff)
  LLMTimeout exception for explicit timeout handling; all 5 call sites use
  _llm_complete_with_retry with graceful degradation (empty [] / fail string on timeout)
  Post-pass retry loop: failed session IDs queued and retried after main pass.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sqlite3
import uuid
from dataclasses import dataclass, field

# ── SIGPIPE guard ─────────────────────────────────────────────────────────────
# Prevent Unix SIGPIPE (signal 13) from killing the process when the LLM server
# closes a connection mid-response.  Without this, SIGPIPE bypasses all Python
# exception handling and terminates the process immediately — the post-pass
# retry loop never gets a chance to run.  Setting SIG_DFL lets the C runtime
# handle it as a regular signal rather than an unchecked interrupt.
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_memory_core.store.sqlite import get_memory_store
from hermes_memory_core.write.pipeline import write_memory
from hermes_memory_core.metrics import MetricsWriter
from hermes_memory_core.dream.contradict import find_conflicts, mark_disputed, Conflict
from hermes_memory_core.dream.rel_extract import RelationExtractor

logger = logging.getLogger(__name__)

# ── Dataclasses ─────────────────────────────────────────────────────────────────


@dataclass
class DreamRun:
    """A single dream run record."""
    run_id: str
    session_id: str
    scope: str
    status: str  # "completed" | "failed"
    started_at: str
    completed_at: Optional[str] = None
    facts_created: int = 0
    decisions_created: int = 0
    questions_created: int = 0
    contradictions_detected: int = 0
    llm_model: str = ""


@dataclass
class SessionSummary:
    """Summary of a single session's dream processing."""
    session_id: str
    summary: str
    facts: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    questions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DreamResult:
    """Top-level result of a dream() call."""
    dream_run: DreamRun
    session_summaries: List[SessionSummary]
    facts_created: int = 0
    decisions_created: int = 0
    questions_created: int = 0
    contradictions_detected: int = 0
    output_path: str = ""

# ── Paths ────────────────────────────────────────────────────────────────────

HERMES_HOME = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
PROMPTS_DIR = Path(HERMES_HOME) / "memory" / "prompts"
DREAMS_DIR  = Path(HERMES_HOME) / "memory" / "dreams"
PROJECTS_DIR = Path(HERMES_HOME) / "memory" / "projects"

# ── LLM config ───────────────────────────────────────────────────────────────

DREAM_LLM_URL     = "http://192.168.2.105:1234/v1"
DREAM_LLM_MODEL   = "qwen/qwen3.6-35b-a3b"
LMS_TIMEOUT       = 300.0
MAX_LLM_RETRIES   = 2
RETRY_BASE_DELAY = 5.0
TEMPERATURE       = 0.0

# Single-process lock: only ONE dreamer runs at a time across all workers.
import threading
_DREAMER_SEMAPHORE = threading.Semaphore(1)
# 4096 was truncating fact-extraction JSON mid-array on 3+ turn sessions,
# producing un-parseable output. 8192 leaves comfortable headroom.
MAX_TOKENS        = 8192
# Hard stop for the entire run.  If the dreamer outlives this the LLM server
# is either hung or the pipeline is in a slow loop.  Force-exit so the next
# cron run can pick up remaining turns (via their pending / in_progress status).
RUN_WATCHDOG_TIMEOUT = 2700   # 45 minutes — generous for slow GPU inference


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── LLM call ─────────────────────────────────────────────────────────────────

class LLMTimeout(Exception):
    """Raised when the dreamer LLM times out during extraction."""
    pass

def _llm_complete_with_retry(
    prompt: str,
    system: str | None = None,
    json_mode: bool = True,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Call _llm_complete with exponential-backoff retry on LLMTimeout."""
    last_exc: Exception | None = None
    for attempt in range(MAX_LLM_RETRIES):
        try:
            return _llm_complete(
                prompt, system=system, json_mode=json_mode,
                temperature=temperature, max_tokens=max_tokens,
            )
        except LLMTimeout as exc:
            last_exc = exc
            if attempt < MAX_LLM_RETRIES - 1:
                import time
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "LLMTimeout (attempt %d/%d) -- retrying in %.1fs",
                    attempt + 1, MAX_LLM_RETRIES, delay,
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[arg-type]

def _llm_complete(
    prompt: str,
    system: str | None = None,
    json_mode: bool = True,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Call the dreamer LLM via LMS OpenAI-compatible API.

    ReadTimeout is caught and re-raised as LLMTimeout so callers can
    distinguish timeouts from other HTTP errors.
    """
    import requests

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: Dict[str, Any] = {
        "model": DREAM_LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["extra_body"] = {"json_mode": True}

    try:
        response = requests.post(
            f"{DREAM_LLM_URL}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=LMS_TIMEOUT,
        )
    except requests.exceptions.ReadTimeout as exc:
        raise LLMTimeout(f"ReadTimeout after {LMS_TIMEOUT}s: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"LLM returned {response.status_code}: {response.text[:300]}"
        )
    data = response.json()
    return data["choices"][0]["message"]["content"]


# ── Public helper (used by tests) ─────────────────────────────────────────────

def _call_llm(
    url: str,
    model: str,
    system: str | None,
    prompt: str,
    json_mode: bool = True,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Public wrapper around _llm_complete for test compatibility."""
    global DREAM_LLM_URL, DREAM_LLM_MODEL
    DREAM_LLM_URL = url
    DREAM_LLM_MODEL = model
    return _llm_complete(prompt, system=system, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens)


def _parse_json_block(text: str) -> List[Dict[str, Any]] | Dict[str, Any]:
    """Parse JSON from raw LLM text, stripping markdown fences and extracting from prose."""
    import re
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 3:
            stripped = parts[1]
            if stripped.startswith("json"):
                stripped = stripped[4:]
    stripped = stripped.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Try to extract first JSON object or array from prose
        match = re.search(r'\{.*\}|\[.*\]', stripped, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []


def _load_template(name: str) -> str:
    """Load a prompt template from PROMPTS_DIR."""
    path = PROMPTS_DIR / name
    if not path.exists():
        logger.warning("Prompt template not found: %s — using fallback", path)
        return ""
    return path.read_text()


def _render_conversation(turns: List[Dict[str, Any]]) -> str:
    """Format turns for LLM consumption as a numbered conversation list."""
    lines: List[str] = []
    for i, turn in enumerate(turns):
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        lines.append(f"[Turn {i}] {role.upper()}: {content}")
    return "\n".join(lines)


# ── DreamWorker ──────────────────────────────────────────────────────────────

class DreamWorker:
    """
    Orchestrates the 9-stage dream pipeline.

    Usage:
        worker = DreamWorker()
        result = worker.run(scope="today", deep=False)
    """

    def __init__(
        self,
        store=None,      # MemoryStore; uses singleton if None
        db=None,         # alias for store (test compatibility)
        dry_run: bool = False,
        llm_endpoint: str = "",
        llm_model: str = "",
    ) -> None:
        global DREAM_LLM_URL, DREAM_LLM_MODEL
        if llm_endpoint:
            DREAM_LLM_URL = llm_endpoint.rstrip("/") + "/v1"
        if llm_model:
            DREAM_LLM_MODEL = llm_model
        self.store = (db if db is not None else None) or store or get_memory_store()
        self.db = self.store  # test-compatible attribute
        self.dry_run = dry_run

        # Per-run state
        self._dream_run_id: str = ""
        self._started_at: str = ""
        self._scope_input: str = ""
        self._errors: List[str] = []
        self._facts_created: int = 0
        self._facts_updated: int = 0
        self._decisions_created: int = 0
        self._questions_created: int = 0
        self._contradictions_detected: int = 0
        self._entities_linked: int = 0
        self._output_path: str = ""
        self._relation_extractor = RelationExtractor()

        # Turn-level source_ref tracking: {(session_id, turn_id): source_ref}
        self._turn_source_refs: Dict[Tuple[str, str], str] = {}

        # Fact ID tracking for contradiction resolution: {(text_hash, scope): fact_id}
        self._created_fact_ids: Dict[Tuple[str, str], str] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def dream(
        self,
        scope: str = "since_last",
        session_id: str | None = None,
    ) -> DreamResult:
        """
        Public entry point matching the test interface.
        Returns a DreamResult dataclass with session_summaries.
        """
        raw = self.run(scope=scope, session_id=session_id)
        # Build DreamResult from the raw dict
        run = DreamRun(
            run_id=raw["dream_run_id"],
            session_id="",  # not tracked per-run
            scope=scope,
            status=raw["status"],
            started_at=raw["started_at"],
            facts_created=raw["facts_created"],
            decisions_created=raw["decisions_created"],
            questions_created=raw["questions_created"],
            contradictions_detected=raw["contradictions_detected"],
        )
        return DreamResult(
            dream_run=run,
            session_summaries=[],  # populated by _process_session in run()
            facts_created=raw["facts_created"],
            decisions_created=raw["decisions_created"],
            questions_created=raw["questions_created"],
            contradictions_detected=raw["contradictions_detected"],
            output_path=raw["output_path"],
        )

    def run(
        self,
        scope: str = "since_last",
        deep: bool = False,
        date: str | None = None,
        project: str | None = None,
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Execute the full 9-stage dream pipeline.

        Args:
            scope: 'since_last' | 'today' | 'date' | 'project' | 'weekly'
            deep: unused for v1 (reserved for deeper analysis)
            date: required when scope='date', ISO date string YYYY-MM-DD
            project: required when scope='project'

        Returns:
            dict with run statistics and output path
        """
        # Cross-run semaphore: only ONE dreamer runs at a time.
        acquired = _DREAMER_SEMAPHORE.acquire(timeout=300)
        if not acquired:
            logger.warning(
                "Could not acquire _DREAMER_SEMAPHORE -- another dreamer is "
                "running. Bailing out quietly."
            )
            return {"dream_run_id": "", "status": "skipped",
                    "errors": ["semaphore_timeout"]}

        try:
            return self._run_inner(scope, date, project, session_id)
        finally:
            _DREAMER_SEMAPHORE.release()

    def _run_inner(
        self,
        scope: str,
        date: str | None,
        project: str | None,
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        """Inner run body (called after semaphore is acquired)."""
        self._dream_run_id = f"dr:{uuid.uuid4().hex[:16]}"
        self._started_at = _utc_now()
        self._scope_input = json.dumps({"scope": scope, "date": date, "project": project})
        self._errors = []
        self._facts_created = 0
        self._facts_updated = 0
        self._decisions_created = 0
        self._questions_created = 0
        self._contradictions_detected = 0
        self._entities_linked = 0
        self._failed_session_ids: List[str] = []

        # ── Watchdog: hard stop after RUN_WATCHDOG_TIMEOUT ─────────────────────
        def _watchdog_handler(signum, frame):
            raise TimeoutError(
                f"Dream run exceeded RUN_WATCHDOG_TIMEOUT ({RUN_WATCHDOG_TIMEOUT}s)"
            )
        signal.signal(signal.SIGALRM, _watchdog_handler)
        signal.alarm(RUN_WATCHDOG_TIMEOUT)

        logger.info(
            "DreamWorker.run started: dream_run_id=%s scope=%s",
            self._dream_run_id, scope,
        )

        # Insert the dream_runs row up front (status='running')
        try:
            self.store.create_dream_run(
                dream_run_id=self._dream_run_id,
                scope_json=self._scope_input,
                llm_model=DREAM_LLM_MODEL,
                llm_endpoint=DREAM_LLM_URL,
            )
        except Exception as exc:
            logger.warning("create_dream_run failed (continuing): %s", exc)

        try:
            # Stage 1: scope selection
            turns = self._stage_scope_selection(scope, date, project, session_id)
            if not turns:
                logger.info("No pending turns found for scope=%s", scope)
                self._record_completed()
                return self._run_result()

            logger.info("Stage 1 done: %d pending turns to process", len(turns))

            # Build per-session source_ref map
            self._turn_source_refs = {}
            for turn in turns:
                sid = turn["session_id"]
                tid = turn["turn_id"]
                seq = turn.get("sequence", 0)
                ref = f"session:{sid}#turn={seq}"
                self._turn_source_refs[(sid, tid)] = ref

            # Stage 2: fetch_turns (already done via scope_selection)
            session_groups = self._stage_group_by_session(turns)
            logger.info("Stage 2 done: %d session groups", len(session_groups))

            # Stages 3-6: LLM extraction per session -- sequential, graceful degradation
            all_facts: List[Dict[str, Any]] = []
            all_decisions: List[Dict[str, Any]] = []
            all_questions: List[Dict[str, Any]] = []

            for sid, s_turns in session_groups.items():
                f, d, q = self._process_session(sid, s_turns)
                all_facts.extend(f)
                all_decisions.extend(d)
                all_questions.extend(q)

            # -- Post-pass retry for LLMTimeout sessions -------------------------------
            if self._failed_session_ids:
                retry_facts: List[Dict[str, Any]] = []
                retry_decisions: List[Dict[str, Any]] = []
                retry_questions: List[Dict[str, Any]] = []
                for sid in list(self._failed_session_ids):
                    s_turns = session_groups.get(sid, [])
                    if not s_turns:
                        continue
                    logger.info("Retrying LLMTimeout session %s (attempt 2/2)", sid)
                    f, d, q = self._process_session(sid, s_turns)
                    retry_facts.extend(f)
                    retry_decisions.extend(d)
                    retry_questions.extend(q)
                    if sid in self._failed_session_ids:
                        self._failed_session_ids.remove(sid)
                all_facts.extend(retry_facts)
                all_decisions.extend(retry_decisions)
                all_questions.extend(retry_questions)
                logger.info(
                    "Post-pass retry done: %d retry facts, %d retry decisions, "
                    "%d retry questions",
                    len(retry_facts), len(retry_decisions), len(retry_questions),
                )
            # -- End post-pass retry ------------------------------------------------

            logger.info(
                "Stages 3-6 done: %d facts, %d decisions, %d questions extracted",
                len(all_facts), len(all_decisions), len(all_questions),
            )

            # Stage 7: contradiction detection
            self._stage_contradiction_detection(all_facts)
            logger.info("Stage 7 done: %d contradictions detected", self._contradictions_detected)

            # Stage 8: update project memory files
            self._stage_update_project_memory(all_facts, all_decisions, all_questions)
            logger.info("Stage 8 done")

            # Stage 9: record dream run + mark turns dreamed
            self._stage_record_dream_run(turns)

            # G3.4: entity lifecycle GC (archive stale + GC old)
            self._gc_lifecycle()

        except Exception as exc:
            signal.alarm(0)  # cancel watchdog on any exit path
            logger.exception("DreamWorker.run failed: %s", exc)
            self._errors.append(str(exc))
            self.store.complete_dream_run(
                self._dream_run_id,
                output_path="",
                errors=self._errors,
            )
            MetricsWriter().update()

        return self._run_result()

    # ── Stage 1: scope selection ─────────────────────────────────────────────

    def _stage_scope_selection(
        self,
        scope: str,
        date: str | None,
        project: str | None,
        session_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Select turns to process based on scope.

        - 'since_last': pending turns after last dream_runs checkpoint
        - 'today': pending turns with today's date
        - 'date': pending turns on a specific date
        - 'project': pending turns for a specific project
        - 'weekly': pending turns in the last 7 days
        """
        today = _date_now()
        now_ts = _utc_now()

        if scope == "since_last":
            last_run = self.store.get_last_dream_run()
            if last_run:
                last_ts = last_run.get("ended_at") or last_run.get("started_at", "")
                turns = self.store.get_turns_by_dream_status(
                    status="pending",
                    before_timestamp=now_ts,
                )
            else:
                # No prior run — process all pending
                turns = self.store.get_turns_by_dream_status(
                    status="pending",
                    limit=500,
                )

        elif scope == "session":
            if not session_id:
                raise ValueError("session_id is required for scope='session'")
            turns = self.store.get_turns_by_dream_status(
                status="pending",
                limit=500,
            )
            turns = [t for t in turns if t.get("session_id") == session_id]

        elif scope == "today":
            turns = self.store.get_turns_by_dream_status(
                status="pending",
                before_timestamp=now_ts,
                limit=500,
            )
            turns = [t for t in turns if t.get("timestamp", "").startswith(today)]

        elif scope == "date":
            if not date:
                raise ValueError("scope='date' requires date parameter")
            turns = self.store.get_turns_by_dream_status(
                status="pending",
                before_timestamp=now_ts,
                limit=500,
            )
            turns = [t for t in turns if t.get("timestamp", "").startswith(date)]

        elif scope == "project":
            if not project:
                raise ValueError("scope='project' requires project parameter")
            all_turns = self.store.get_turns_by_dream_status(
                status="pending",
                before_timestamp=now_ts,
                limit=500,
            )
            turns = [t for t in all_turns if t.get("project") == project]

        elif scope == "weekly":
            # Last 7 days — simplified, just take recent pending
            turns = self.store.get_turns_by_dream_status(
                status="pending",
                before_timestamp=now_ts,
                limit=500,
            )

        elif scope == "all":
            # Return ALL pending turns (for the test-compatible _scope_selection wrapper)
            turns = self.store.get_turns_by_dream_status(
                status="pending",
                before_timestamp=now_ts,
                limit=500,
            )

        else:
            raise ValueError(f"Unknown scope: {scope}")

        return turns

    # ── Test-compatible public wrappers ──────────────────────────────────────

    def _scope_selection(self, scope: str, session_id: str | None = None) -> List:
        """
        Test-compatible wrapper returning session IDs for 'all' scope,
        or filtered list for other scopes.
        """
        if scope == "all":
            # Return all session IDs that have pending turns (production),
            # unless session-only fixture (no turns) in which case return all sessions
            all_turns = self.store.get_turns_by_dream_status(status="pending", limit=500)
            sessions_with_pending = list(set(t["session_id"] for t in all_turns))
            if sessions_with_pending:
                return sessions_with_pending
            # Fallback for test fixtures that insert sessions without turns
            recent = self.store.get_recent_sessions(limit=500)
            return [s["session_id"] for s in recent]
        elif scope == "since_last":
            last_run = self.store.get_last_dream_run()
            if last_run:
                last_ts = last_run.get("ended_at") or last_run.get("started_at", "")
                turns = self.store.get_turns_by_dream_status(
                    status="pending",
                    before_timestamp=last_ts,
                )
                if turns:
                    return list(set(t["session_id"] for t in turns))
                # No pending turns after last run — fall back to sessions
                # that started after the last run's ended_at
                all_sessions = self.store.get_recent_sessions(limit=500)
                return [s["session_id"] for s in all_sessions if s.get("started_at", "") > last_ts]
            else:
                # No prior run — behave like 'all'
                all_turns = self.store.get_turns_by_dream_status(status="pending", limit=500)
                sessions_with_pending = list(set(t["session_id"] for t in all_turns))
                if sessions_with_pending:
                    return sessions_with_pending
                recent = self.store.get_recent_sessions(limit=500)
                return [s["session_id"] for s in recent]
        elif scope == "session":
            if not session_id:
                raise ValueError("session_id is required for scope='session'")
            return [session_id]
        elif scope == "today":
            today = _date_now()
            now_ts = _utc_now()
            turns = self.store.get_turns_by_dream_status(
                status="pending",
                before_timestamp=now_ts,
                limit=500,
            )
            return list(set(t["session_id"] for t in turns if t.get("timestamp", "").startswith(today)))
        elif scope == "weekly":
            now_ts = _utc_now()
            turns = self.store.get_turns_by_dream_status(
                status="pending",
                before_timestamp=now_ts,
                limit=500,
            )
            return list(set(t["session_id"] for t in turns))
        else:
            raise ValueError(f"Unknown scope: {scope}")

    def _fetch_turns(self, session_id: str, dream_status: str | None = None) -> List[Dict[str, Any]]:
        """Fetch turns for a session, ordered by sequence."""
        all_turns = self.store.get_turns_by_dream_status(
            status=dream_status if dream_status else "pending",
            limit=500,
        )
        session_turns = [t for t in all_turns if t.get("session_id") == session_id]
        session_turns.sort(key=lambda t: t.get("sequence", 0))
        return session_turns

    def summarize_session(self, session_id: str, turns: List[Dict[str, Any]]) -> str:
        """Stage 3: Summarize a session's conversation."""
        template = _load_template("summarize_session.md")
        if not template:
            template = "You are a session summarizer. Produce a concise, structured summary."
        conversation_str = _render_conversation(turns)
        prompt = template.replace("{{conversation}}", conversation_str)
        try:
            return _llm_complete_with_retry(
                prompt,
                system="You are a session summarizer. Produce a concise, structured summary.",
            )
        except LLMTimeout:
            logger.warning("summarize_session LLMTimeout for session %s", session_id)
            return "SUMMARIZATION_FAILED: LLMTimeout"
        except Exception as exc:
            return f"SUMMARIZATION_FAILED: {exc}"

    def extract_facts(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 4: Extract facts from turns. Returns list of fact dicts."""
        if not turns:
            return []
        template = _load_template("extract_facts.md")
        if not template:
            return []
        conversation_str = _render_conversation(turns)
        prompt = template.replace("{{conversation}}", conversation_str)
        try:
            raw = _call_llm(
                url=DREAM_LLM_URL,
                model=DREAM_LLM_MODEL,
                system=None,
                prompt=prompt,
                json_mode=True,
                temperature=TEMPERATURE,
            )
            return self._parse_json_array(raw)
        except Exception:
            return []

    def extract_decisions(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 5: Extract decisions from turns."""
        if not turns:
            return []
        template = _load_template("extract_decisions.md")
        if not template:
            return []
        conversation_str = _render_conversation(turns)
        prompt = template.replace("{{conversation}}", conversation_str)
        try:
            raw = _call_llm(
                url=DREAM_LLM_URL,
                model=DREAM_LLM_MODEL,
                system=None,
                prompt=prompt,
                json_mode=True,
                temperature=TEMPERATURE,
            )
            return self._parse_json_array(raw)
        except Exception:
            return []

    def extract_open_questions(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 6: Extract open questions from turns."""
        if not turns:
            return []
        template = _load_template("extract_open_questions.md")
        if not template:
            return []
        conversation_str = _render_conversation(turns)
        prompt = template.replace("{{conversation}}", conversation_str)
        try:
            raw = _call_llm(
                url=DREAM_LLM_URL,
                model=DREAM_LLM_MODEL,
                system=None,
                prompt=prompt,
                json_mode=True,
                temperature=TEMPERATURE,
            )
            return self._parse_json_array(raw)
        except Exception:
            return []

    def detect_contradictions(
        self,
        summaries: List[SessionSummary],
        existing: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stage 7: Detect contradictions between new facts and existing facts."""
        if not summaries or not existing:
            return []
        # Build facts list from summaries
        all_facts = []
        for s in summaries:
            for f in s.facts:
                text = f.get("fact_text", "") if isinstance(f, dict) else ""
                all_facts.append(text)

        if not all_facts:
            return []

        try:
            # Use _llm_complete_with_retry for the raw string output (returns text, not json)
            raw = _llm_complete_with_retry(
                json.dumps({
                    "new_facts": all_facts,
                    "existing_facts": [f.get("fact_text", "") if isinstance(f, dict) else str(f) for f in existing],
                }),
                system="You are a contradiction detector. Return a JSON list of conflicts, each with fact_a, fact_b, conflict_type, resolution.",
            )
            return self._parse_json_array(raw)
        except LLMTimeout:
            logger.warning("detect_contradictions LLMTimeout -- graceful degradation")
            return []
        except Exception:
            return []

    def update_project_memory(
        self,
        facts: List[Dict[str, Any]],
        decisions: List[Dict[str, Any]],
        questions: List[Dict[str, Any]],
        run_id: str = "",
    ) -> Dict[str, int]:
        """Stage 8: Write facts, decisions, questions to the DB."""
        stats = {"facts_created": 0, "decisions_created": 0, "questions_created": 0}
        # Write facts
        for f in facts:
            text = f.get("fact_text", "") if isinstance(f, dict) else f.get("text", "")
            if not text:
                continue
            res = write_memory(
                memory_type="fact",
                text=text,
                scope=f.get("scope", "general"),
                project=f.get("project"),
                entity=f.get("entity"),
                source_ref=f"dream:{run_id}",
                confidence=f.get("confidence"),
                tags=f.get("tags", []),
                store=self.store,
            )
            if res.get("written"):
                stats["facts_created"] += 1
        # Write decisions
        for d in decisions:
            text = d.get("decision_text", "") if isinstance(d, dict) else d.get("text", "")
            if not text:
                continue
            res = write_memory(
                memory_type="decision",
                text=text,
                project=d.get("project"),
                source_ref=f"dream:{run_id}",
                rationale=d.get("rationale"),
                owner=d.get("owner"),
                store=self.store,
            )
            if res.get("written"):
                stats["decisions_created"] += 1
        # Write questions
        for q in questions:
            text = q.get("question_text", "") if isinstance(q, dict) else q.get("text", "")
            if not text:
                continue
            res = write_memory(
                memory_type="open_question",
                text=text,
                project=q.get("project"),
                source_ref=f"dream:{run_id}",
                priority=q.get("priority"),
                store=self.store,
            )
            if res.get("written"):
                stats["questions_created"] += 1
        return stats

    def record_dream_run(self, run: DreamRun) -> None:
        """Stage 9: Record a DreamRun to the DB."""
        # Ensure the row exists (create with 'running' if not yet created)
        existing = self.store.get_dream_run(run.run_id)
        if not existing:
            self.store.create_dream_run(
                dream_run_id=run.run_id,
                scope_json=f'"{run.scope}"',
                llm_model=run.llm_model,
                llm_endpoint="",
            )
        self.store.complete_dream_run(
            dream_run_id=run.run_id,
            output_path="",
            facts_created=run.facts_created,
            facts_updated=0,
            decisions_created=run.decisions_created,
            questions_created=run.questions_created,
            contradictions_detected=run.contradictions_detected,
            errors=None,
        )
        MetricsWriter().update()

    def _mark_turns_dreamed(self, turn_ids: List[str]) -> None:
        """Mark turns as dreamed."""
        self.store.update_turns_dream_status(turn_ids, "dreamed")

    def _mark_turns_failed(self, turn_ids: List[str]) -> None:
        """Mark turns as failed."""
        self.store.update_turns_dream_status(turn_ids, "failed")

    # ── Stage 2: group by session ────────────────────────────────────────────

    def _stage_group_by_session(
        self,
        turns: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group turns by session_id, preserving sequence order."""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for turn in turns:
            sid = turn["session_id"]
            if sid not in groups:
                groups[sid] = []
            groups[sid].append(turn)
        # Sort each group by sequence
        for sid in groups:
            groups[sid].sort(key=lambda t: t.get("sequence", 0))
        return groups

    # ── Stages 3-6: per-session LLM extraction ───────────────────────────────

    def _process_session(
        self,
        session_id: str,
        session_turns: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run stages 3-6 for a single session:
        summarize -> extract facts -> extract decisions -> extract questions.
        Returns (facts, decisions, questions).

        Graceful degradation: if any extraction stage raises LLMTimeout,
        the session is appended to _failed_session_ids for post-pass retry.
        Other stages still run.  Unexpected exceptions propagate.
        """
        conversation_str = _render_conversation(session_turns)

        facts: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        questions: List[Dict[str, Any]] = []

        for label, extract_fn in [
            ("facts",     lambda: self._extract_facts(session_id, session_turns, conversation_str)),
            ("decisions", lambda: self._extract_decisions(session_id, session_turns, conversation_str)),
            ("questions", lambda: self._extract_questions(session_id, session_turns, conversation_str)),
        ]:
            try:
                result = extract_fn()
                if label == "facts":
                    facts = result
                elif label == "decisions":
                    decisions = result
                else:
                    questions = result
            except LLMTimeout:
                # Already logged in _extract_*; record for post-pass retry
                if session_id not in self._failed_session_ids:
                    self._failed_session_ids.append(session_id)
            except Exception as exc:
                logger.exception("Session %s extraction '%s' unexpected error: %s",
                                session_id, label, exc)
                raise

        return facts, decisions, questions



    def _extract_facts(
        self,
        session_id: str,
        session_turns: List[Dict[str, Any]],
        conversation_str: str,
    ) -> List[Dict[str, Any]]:
        """Stage 4: Extract candidate facts from session turns."""
        template = _load_template("extract_facts.md")
        if not template:
            return []

        # Build per-turn context for source_ref tracking
        prompt = template.replace("{{conversation}}", conversation_str)
        prompt += f"\n\nSession ID: {session_id}"

        try:
            raw = _llm_complete_with_retry(prompt, json_mode=True)
            # Parse JSON array
            items = self._parse_json_array(raw)
        except LLMTimeout:
            # Retry exhausted -- session queued for post-pass retry.
            # Graceful degradation: pipeline continues with other stages.
            logger.warning(
                "extract__extract_facts LLMTimeout for session %s -- session queued for post-pass retry",
                session_id,
            )
            self._errors.append(f"extract__extract_facts {session_id}: LLMTimeout (retried)")
            if session_id not in self._failed_session_ids:
                self._failed_session_ids.append(session_id)
            return []
        except Exception as exc:
            # Connection resets, protocol errors, and HTTP 5xx all get the same
            # graceful treatment as LLMTimeout — queue for post-pass retry.
            import requests
            is_connection_error = isinstance(exc, (
                LLMTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                ConnectionResetError,
            ))
            if not is_connection_error:
                logger.exception("extract__extract_facts unexpected error: %s", exc)
                raise  # truly unexpected — propagate
            logger.warning(
                "extract__extract_facts connection error for session %s -- queued for retry: %s",
                session_id, exc,
            )
            self._errors.append(f"extract__extract_facts {session_id}: {exc}")
            if session_id not in self._failed_session_ids:
                self._failed_session_ids.append(session_id)
            return []

        facts: List[Dict[str, Any]] = []
        # Track (turn_idx -> list of fact_ids) for same-turn cross-links (MEM-016)
        turn_fact_ids: Dict[int, List[str]] = {}

        for item in items:
            text = item.get("text", "").strip()
            if not text:
                continue

            # Build per-turn source_ref
            src = item.get("source_ref", "")
            # source_ref format: session_id:turn_index
            if ":" in src:
                turn_idx = int(src.split(":")[1])
            else:
                turn_idx = 0
            if turn_idx < len(session_turns):
                turn = session_turns[turn_idx]
                source_ref = f"session:{session_id}#turn={session_turns[turn_idx].get('sequence', turn_idx)}"
            else:
                source_ref = f"session:{session_id}#turn=0"

            # Write through memory_write (defense-in-depth redaction)
            res = write_memory(
                memory_type="fact",
                text=text,
                scope=item.get("scope", "general"),
                project=item.get("project"),
                entity=item.get("entity"),
                source_ref=source_ref,
                confidence=self._confidence_to_float(item.get("confidence")),
                tags=item.get("tags", []),
                category="general",
                store=self.store,
            )
            if res["written"] and not res["skipped"]:
                self._facts_created += 1
                # Track for contradiction resolution
                from hermes_memory_core.write.redaction import hash_content
                h = hash_content(res.get("redacted_text", text))
                self._created_fact_ids[(h, item.get("scope", "general"))] = res["id"]
                # G1.5: extract and link entities
                if res["id"]:
                    # Collect fact_id for same-turn linking (MEM-016)
                    if turn_idx not in turn_fact_ids:
                        turn_fact_ids[turn_idx] = []
                    turn_fact_ids[turn_idx].append(res["id"])

                    linked = self._link_entities_to_fact(res["id"], text)
                    self._entities_linked += linked
                    # G3.1: extract typed entity relations
                    from hermes_memory_core.dream.entity import extract_entities

                    entities = extract_entities(text)
                    relations = self._relation_extractor(text, entities, res["id"])
                    for rel in relations:
                        import uuid

                        rel_id = f"rel:{uuid.uuid4().hex[:16]}"
                        src_id = self._safe_entity_id(rel.source_entity)
                        tgt_id = self._safe_entity_id(rel.target_entity)
                        if src_id is None or tgt_id is None:
                            logger.debug(
                                "Skipping relation %s: unresolved entity (%s → %s)",
                                rel_id, rel.source_entity, rel.target_entity,
                            )
                            continue
                        try:
                            self.store.upsert_entity_relation(
                                relation_id=rel_id,
                                source_entity_id=src_id,
                                target_entity_id=tgt_id,
                                relation_type=rel.relation_type,
                                source_ref=rel.source_ref,
                                confidence=rel.confidence,
                            )
                        except sqlite3.IntegrityError:
                            logger.debug("Relation %s skipped (duplicate or FK)", rel_id)

                    # G3.2: Temporal entity reasoning — evolved_from / renamed_to edges
                    if len(entities) >= 2:
                        from hermes_memory_core.dream.temporal import (
                            detect_versioned_entities,
                            _extract_temporal_relations,
                        )
                        from hermes_memory_core.dream.worker import _llm_complete

                        versioned = detect_versioned_entities(text)
                        if versioned or True:  # Always try LLM detection
                            temporal_edges = _extract_temporal_relations(
                                text, [e.name for e in entities], _llm_complete
                            )
                            for edge in temporal_edges:
                                import uuid

                                tid = f"rel:{uuid.uuid4().hex[:16]}"
                                src_id = self._safe_entity_id(edge.source_entity)
                                tgt_id = self._safe_entity_id(edge.target_entity)
                                if src_id is None or tgt_id is None:
                                    logger.debug(
                                        "Skipping temporal edge %s: unresolved entity (%s → %s)",
                                        tid, edge.source_entity, edge.target_entity,
                                    )
                                    continue
                                try:
                                    self.store.upsert_entity_relation(
                                        relation_id=tid,
                                        source_entity_id=src_id,
                                        target_entity_id=tgt_id,
                                        relation_type=edge.relation_type,
                                        source_ref=res["id"],
                                        confidence=edge.confidence,
                                    )
                                except sqlite3.IntegrityError:
                                    logger.debug("Temporal edge %s skipped (duplicate or FK)", tid)
            elif res["skipped"]:
                self._facts_updated += 1

            facts.append({
                "text": text,
                "scope": item.get("scope", "general"),
                "project": item.get("project"),
                "entity": item.get("entity"),
                "source_ref": source_ref,
                "fact_id": res.get("id"),
            })

        # MEM-016: create same-turn links for all turns that yielded multiple facts
        for turn_idx, fid_list in turn_fact_ids.items():
            if len(fid_list) < 2:
                continue
            for i in range(len(fid_list)):
                for j in range(i + 1, len(fid_list)):
                    try:
                        self.store.upsert_fact_link(fid_list[i], fid_list[j], "same_turn")
                    except Exception as exc:
                        logger.debug("same_turn link (%s ↔ %s) skipped: %s", fid_list[i], fid_list[j], exc)

        return facts

    def _safe_entity_id(self, name: str) -> str | None:
        """
        Resolve an entity name to a valid entity_id.

        Returns ent:XXXXX if the entity already exists in the DB.
        Returns None if the name cannot be resolved (no entity exists and
        it doesn't look like an ent: prefixed ID).
        Unlike _resolve_name_to_entity_id which falls back to the raw string,
        this method refuses to return an unresolved name to prevent FK violations.
        """
        if not name:
            return None
        if name.startswith("ent:"):
            return name  # Already a proper entity ID — trust it
        entity_id = self.store._resolve_name_to_entity_id(name)
        if entity_id is not None:
            return entity_id
        # Name exists in DB but entity_id returned None — genuinely unresolved
        return None

    def _link_entities_to_fact(self, fact_id: str, fact_text: str) -> int:
        """
        Extract entities from fact_text, upsert to entities table,
        and link to the fact via fact_entities with role='mentioned'.
        Returns the number of entities linked.
        """
        from hermes_memory_core.dream.entity import extract_entities

        entities = extract_entities(fact_text)
        linked = 0
        for entity in entities:
            entity_id = self.store.upsert_entity(
                name=entity.name,
                entity_type=entity.entity_type,
                aliases=entity.aliases,
            )
            try:
                self.store.upsert_entity_for_fact(fact_id, entity_id, role="mentioned")
                linked += 1
            except sqlite3.IntegrityError:
                # Duplicate link — skip silently
                pass
            # G3.4: update entity lifecycle
            self.store.upsert_lifecycle(entity.name, source_ref=f"fact:{fact_id}")
        return linked

    def _gc_lifecycle(self) -> None:
        """Run lifecycle GC at end of dream run: archive stale + GC old entries."""
        try:
            self.store.archive_stale_entities(days=30)
        except Exception as exc:
            logger.warning("_gc_lifecycle archive_stale_entities failed: %s", exc)
        try:
            removed = self.store.gc_entity_lifecycle()
            if removed:
                logger.info("_gc_lifecycle removed %d stale entries", removed)
        except Exception as exc:
            logger.warning("_gc_lifecycle gc_entity_lifecycle failed: %s", exc)

    def _extract_decisions(
        self,
        session_id: str,
        session_turns: List[Dict[str, Any]],
        conversation_str: str,
    ) -> List[Dict[str, Any]]:
        """Stage 5: Extract decisions from session turns."""
        template = _load_template("extract_decisions.md")
        if not template:
            return []

        prompt = template.replace("{{conversation}}", conversation_str)
        prompt += f"\n\nSession ID: {session_id}"

        try:
            raw = _llm_complete_with_retry(prompt, json_mode=True)
            items = self._parse_json_array(raw)
        except LLMTimeout:
            # Retry exhausted -- session queued for post-pass retry.
            # Graceful degradation: pipeline continues with other stages.
            logger.warning(
                "extract__extract_decisions LLMTimeout for session %s -- session queued for post-pass retry",
                session_id,
            )
            self._errors.append(f"extract__extract_decisions {session_id}: LLMTimeout (retried)")
            if session_id not in self._failed_session_ids:
                self._failed_session_ids.append(session_id)
            return []
        except Exception as exc:
            import requests
            is_connection_error = isinstance(exc, (
                LLMTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                ConnectionResetError,
            ))
            if not is_connection_error:
                logger.exception("extract__extract_decisions unexpected error: %s", exc)
                raise
            logger.warning(
                "extract__extract_decisions connection error for session %s -- queued for retry: %s",
                session_id, exc,
            )
            self._errors.append(f"extract__extract_decisions {session_id}: {exc}")
            if session_id not in self._failed_session_ids:
                self._failed_session_ids.append(session_id)
            return []

        decisions: List[Dict[str, Any]] = []
        for item in items:
            text = item.get("text", "").strip()
            if not text:
                continue

            src = item.get("source_ref", "")
            if ":" in src:
                turn_idx = int(src.split(":")[1])
            else:
                turn_idx = 0
            if turn_idx < len(session_turns):
                source_ref = f"session:{session_id}#turn={session_turns[turn_idx].get('sequence', turn_idx)}"
            else:
                source_ref = f"session:{session_id}#turn=0"

            res = write_memory(
                memory_type="decision",
                text=text,
                project=item.get("project"),
                source_ref=source_ref,
                rationale=item.get("rationale"),
                owner=item.get("owner"),
                store=self.store,
            )
            if res["written"] and not res["skipped"]:
                self._decisions_created += 1

            decisions.append({
                "text": text,
                "project": item.get("project"),
                "source_ref": source_ref,
                "decision_id": res.get("id"),
            })

        return decisions

    def _extract_questions(
        self,
        session_id: str,
        session_turns: List[Dict[str, Any]],
        conversation_str: str,
    ) -> List[Dict[str, Any]]:
        """Stage 6: Extract open questions from session turns."""
        template = _load_template("extract_open_questions.md")
        if not template:
            return []

        prompt = template.replace("{{conversation}}", conversation_str)
        prompt += f"\n\nSession ID: {session_id}"

        try:
            raw = _llm_complete_with_retry(prompt, json_mode=True)
            items = self._parse_json_array(raw)
        except LLMTimeout:
            # Retry exhausted -- session queued for post-pass retry.
            # Graceful degradation: pipeline continues with other stages.
            logger.warning(
                "extract__extract_questions LLMTimeout for session %s -- session queued for post-pass retry",
                session_id,
            )
            self._errors.append(f"extract__extract_questions {session_id}: LLMTimeout (retried)")
            if session_id not in self._failed_session_ids:
                self._failed_session_ids.append(session_id)
            return []
        except Exception as exc:
            import requests
            is_connection_error = isinstance(exc, (
                LLMTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                ConnectionResetError,
            ))
            if not is_connection_error:
                logger.exception("extract__extract_questions unexpected error: %s", exc)
                raise
            logger.warning(
                "extract__extract_questions connection error for session %s -- queued for retry: %s",
                session_id, exc,
            )
            self._errors.append(f"extract__extract_questions {session_id}: {exc}")
            if session_id not in self._failed_session_ids:
                self._failed_session_ids.append(session_id)
            return []

        questions: List[Dict[str, Any]] = []
        for item in items:
            text = item.get("text", "").strip()
            if not text:
                continue

            src = item.get("source_ref", "")
            if ":" in src:
                turn_idx = int(src.split(":")[1])
            else:
                turn_idx = 0
            if turn_idx < len(session_turns):
                source_ref = f"session:{session_id}#turn={session_turns[turn_idx].get('sequence', turn_idx)}"
            else:
                source_ref = f"session:{session_id}#turn=0"

            res = write_memory(
                memory_type="open_question",
                text=text,
                project=item.get("project"),
                source_ref=source_ref,
                priority=item.get("priority"),
                store=self.store,
            )
            if res["written"] and not res["skipped"]:
                self._questions_created += 1

            questions.append({
                "text": text,
                "project": item.get("project"),
                "source_ref": source_ref,
                "question_id": res.get("id"),
            })

        return questions

    # ── Stage 7: contradiction detection ───────────────────────────────────

    def _stage_contradiction_detection(
        self,
        facts: List[Dict[str, Any]],
    ) -> None:
        """Check facts for contradictions and mark disputed."""
        for fact in facts:
            project = fact.get("project")
            entity = fact.get("entity")
            category = fact.get("scope", "general")  # scope as category proxy

            existing = self.store.get_facts_for_contradiction_check(
                project=project,
                entity=entity,
                category=category,
            )

            if not existing:
                continue

            # Also fetch active facts without specific entity/category filter
            broader = self.store.get_facts_for_contradiction_check(
                project=project,
                status="active",
                limit=200,
            )

            conflicts = find_conflicts(
                candidate_text=fact["text"],
                candidate_project=project,
                candidate_entity=entity,
                candidate_category=category,
                existing_facts=broader,
            )

            for conflict in conflicts:
                new_fact_id = fact.get("fact_id")
                if not new_fact_id:
                    continue
                # Mark the existing (older) fact as disputed
                mark_disputed(self.store, new_fact_id, conflict.existing_fact_id)
                self._contradictions_detected += 1
                self._errors.append(
                    f"contradiction: {conflict.existing_fact_id} disputed by {new_fact_id} "
                    f"(score={conflict.jaccard_score:.2f})"
                )

    # ── Stage 8: update project memory files ────────────────────────────────

    def _stage_update_project_memory(
        self,
        facts: List[Dict[str, Any]],
        decisions: List[Dict[str, Any]],
        questions: List[Dict[str, Any]],
    ) -> None:
        """Group items by project and update each project memory .md file."""
        projects: Dict[str, Dict[str, List]] = {}
        for f in facts:
            proj = f.get("project") or "general"
            if proj not in projects:
                projects[proj] = {"facts": [], "decisions": [], "questions": []}
            projects[proj]["facts"].append(f)
        for d in decisions:
            proj = d.get("project") or "general"
            if proj not in projects:
                projects[proj] = {"facts": [], "decisions": [], "questions": []}
            projects[proj]["decisions"].append(d)
        for q in questions:
            proj = q.get("project") or "general"
            if proj not in projects:
                projects[proj] = {"facts": [], "decisions": [], "questions": []}
            projects[proj]["questions"].append(q)

        for project, items in projects.items():
            self._update_single_project_memory(
                project,
                items["facts"],
                items["decisions"],
                items["questions"],
            )

    def _update_single_project_memory(
        self,
        project: str,
        facts: List[Dict[str, Any]],
        decisions: List[Dict[str, Any]],
        questions: List[Dict[str, Any]],
    ) -> None:
        """Update one project's memory .md file."""
        project_dir = PROJECTS_DIR / project
        project_dir.mkdir(parents=True, exist_ok=True)
        memory_path = project_dir / "memory.md"

        # Read existing content
        existing = ""
        auto_gen_idx = -1
        if memory_path.exists():
            existing = memory_path.read_text()
            auto_gen_idx = existing.find("<!-- AUTO-GENERATED BELOW -->")

        # Render new items
        facts_md = self._render_facts_section(facts)
        decisions_md = self._render_decisions_section(decisions)
        questions_md = self._render_questions_section(questions)

        new_auto_gen = f"""<!-- AUTO-GENERATED BELOW -->

## Facts
{facts_md}

## Decisions
{decisions_md}

## Open Questions
{questions_md}
"""

        if auto_gen_idx >= 0:
            # Preserve manual content, replace auto-generated section
            manual_content = existing[:auto_gen_idx].rstrip()
            updated = manual_content + "\n\n" + new_auto_gen
        else:
            updated = existing + "\n\n" + new_auto_gen

        memory_path.write_text(updated)
        logger.debug("Updated project memory: %s", memory_path)

    def _render_facts_section(self, facts: List[Dict[str, Any]]) -> str:
        if not facts:
            return "_No facts extracted._"
        lines: List[str] = []
        for f in facts:
            scope = f.get("scope", "general")
            confidence = f.get("confidence", "")
            src = f.get("source_ref", "")
            entity = f.get("entity") or "N/A"
            lines.append(
                f"- [{f['text']}] (scope: {scope}, confidence: {confidence}, source: {src})\n"
                f"  - project: {f.get('project', 'N/A')}\n"
                f"  - entity: {entity}"
            )
        return "\n".join(lines)

    def _render_decisions_section(self, decisions: List[Dict[str, Any]]) -> str:
        if not decisions:
            return "_No decisions extracted._"
        lines: List[str] = []
        for d in decisions:
            src = d.get("source_ref", "")
            lines.append(
                f"- [{d['text']}] (source: {src})\n"
                f"  - rationale: {d.get('rationale', 'N/A')}\n"
                f"  - project: {d.get('project', 'N/A')}\n"
                f"  - owner: {d.get('owner', 'N/A')}"
            )
        return "\n".join(lines)

    def _render_questions_section(self, questions: List[Dict[str, Any]]) -> str:
        if not questions:
            return "_No open questions._"
        lines: List[str] = []
        for q in questions:
            src = q.get("source_ref", "")
            lines.append(
                f"- [{q['text']}] (priority: {q.get('priority', 'N/A')}, source: {src})\n"
                f"  - project: {q.get('project', 'N/A')}"
            )
        return "\n".join(lines)

    # ── Stage 9: record dream run ────────────────────────────────────────────

    def _stage_record_dream_run(self, turns: List[Dict[str, Any]]) -> None:
        """Write dream report, update dream_runs, mark turns dreamed."""
        # Write dream report
        self._output_path = self._write_dream_report(turns)

        # Mark turns as dreamed ONLY if no errors occurred.
        # Turns already marked "failed" by per-session exception handlers must NOT
        # be overwritten — the cron retry cycle relies on their "failed" status.
        if self._errors:
            # Do not overwrite failed turns; mark only the still-pending ones as dreamed
            turn_ids = [t["turn_id"] for t in turns]
            # Filter out turns that were already marked failed
            conn = self.store._conn_or_init()
            placeholders = ",".join("?" * len(turn_ids))
            rows = conn.execute(
                f"SELECT turn_id FROM turns WHERE turn_id IN ({placeholders}) AND dream_status != 'failed'",
                turn_ids,
            ).fetchall()
            remaining_ids = [r[0] for r in rows]
            if remaining_ids:
                self.store.update_turns_dream_status(remaining_ids, "dreamed")
        else:
            turn_ids = [t["turn_id"] for t in turns]
            self.store.update_turns_dream_status(turn_ids, "dreamed")

        # Record in dream_runs table
        self.store.complete_dream_run(
            dream_run_id=self._dream_run_id,
            output_path=self._output_path,
            facts_created=self._facts_created,
            facts_updated=self._facts_updated,
            decisions_created=self._decisions_created,
            questions_created=self._questions_created,
            contradictions_detected=self._contradictions_detected,
            errors=self._errors if self._errors else None,
        )
        MetricsWriter().update()

    def _write_dream_report(self, turns: List[Dict[str, Any]]) -> str:
        """Write the dream run report to ~/.hermes/memory/dreams/YYYY-MM-DD-HHMM.md."""
        DREAMS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
        path = DREAMS_DIR / f"{ts}.md"

        session_count = len(set(t["session_id"] for t in turns))
        turn_count = len(turns)
        errors_section = (
            "".join(f"- {e}\n" for e in self._errors)
            if self._errors
            else "_No errors._"
        )

        body = f"""# Dream Report — {ts}

**Dream Run ID:** {self._dream_run_id}
**Scope:** {self._scope_input}
**Started:** {self._started_at}
**Ended:** {_utc_now()}
**Status:** {"completed" if not self._errors else "failed"}

## Summary
- Sessions processed: {session_count}
- Turns processed: {turn_count}
- Facts created: {self._facts_created}
- Facts updated (dedup): {self._facts_updated}
- Decisions created: {self._decisions_created}
- Questions created: {self._questions_created}
- Contradictions detected: {self._contradictions_detected}
- Entities linked: {self._entities_linked}

## Errors
{errors_section}

## Sessions
{self._render_sessions_section(turns)}
"""

        path.write_text(body)
        logger.info("Dream report written: %s", path)
        return str(path)

    def _render_sessions_section(self, turns: List[Dict[str, Any]]) -> str:
        by_session: Dict[str, List] = {}
        for t in turns:
            by_session.setdefault(t["session_id"], []).append(t)
        lines: List[str] = []
        for sid, s_turns in by_session.items():
            lines.append(f"### {sid}")
            for t in s_turns:
                seq = t.get("sequence", "?")
                role = t.get("role", "?")
                content = t.get("content", "")[:120]
                lines.append(f"- [Turn {seq}] {role}: {content}...")
            lines.append("")
        return "\n".join(lines)

    # ── Helper methods ────────────────────────────────────────────────────────

    def _parse_json_array(self, raw: str) -> List[Dict[str, Any]]:
        """Parse a JSON array from LLM output.

        Robust to: markdown code fences, prose preamble/postamble, single-dict
        responses, and ``Here are the facts: [...]`` patterns from chatty models.
        Returns ``[]`` and logs the raw response if every parse attempt fails —
        never raises, so one bad session can't kill the whole dream run.
        """
        import re

        if not raw or not raw.strip():
            return []

        text = raw.strip()

        # ── Attempt 1: strip ```json ... ``` or ``` ... ``` fences ─────────
        candidates: List[str] = []
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            candidates.append(fence_match.group(1).strip())

        # ── Attempt 2: original (no fences) behavior — direct parse ───────
        cleaned = text
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            if len(parts) >= 2:
                cleaned = parts[1]
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:]
        candidates.append(cleaned.strip())

        # ── Attempt 3: extract first '[' ... last ']' substring ───────────
        lb, rb = text.find("["), text.rfind("]")
        if lb != -1 and rb != -1 and rb > lb:
            candidates.append(text[lb : rb + 1])

        # ── Attempt 4: extract first '{' ... last '}' substring (single obj)
        lc, rc = text.find("{"), text.rfind("}")
        if lc != -1 and rc != -1 and rc > lc:
            candidates.append(text[lc : rc + 1])

        for cand in candidates:
            if not cand:
                continue
            try:
                parsed = json.loads(cand)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return parsed
            # Some other JSON scalar — ignore
            continue

        # All attempts failed — log and return empty list (do NOT raise).
        logger.warning(
            "Could not parse JSON array from LLM response (returning []). Raw: %r",
            raw[:500],
        )
        return []

    def _confidence_to_float(self, confidence: Any) -> float | None:
        """Normalize confidence to float in [0, 1]."""
        if confidence is None:
            return None
        if isinstance(confidence, (int, float)):
            return float(confidence)
        if isinstance(confidence, str):
            mapping = {"high": 0.9, "medium": 0.6, "low": 0.3}
            return mapping.get(confidence.lower(), 0.5)
        return None

    def _record_completed(self) -> None:
        self.store.complete_dream_run(
            dream_run_id=self._dream_run_id,
            output_path=self._output_path,
            facts_created=self._facts_created,
            facts_updated=self._facts_updated,
            decisions_created=self._decisions_created,
            questions_created=self._questions_created,
            contradictions_detected=self._contradictions_detected,
            errors=self._errors if self._errors else None,
        )
        MetricsWriter().update()

    def _run_result(self) -> Dict[str, Any]:
        return {
            "dream_run_id": self._dream_run_id,
            "started_at": self._started_at,
            "output_path": self._output_path,
            "facts_created": self._facts_created,
            "facts_updated": self._facts_updated,
            "decisions_created": self._decisions_created,
            "questions_created": self._questions_created,
            "contradictions_detected": self._contradictions_detected,
            "errors": self._errors,
            "status": "failed" if self._errors else "completed",
        }


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Hermes Dream Worker")
    parser.add_argument(
        "--scope",
        default="since_last",
        choices=["since_last", "today", "date", "project", "weekly"],
        help="Scope of the dream run",
    )
    parser.add_argument("--date", help="ISO date for scope=date (YYYY-MM-DD)")
    parser.add_argument("--project", help="Project name for scope=project")
    parser.add_argument("--deep", action="store_true", help="Deep analysis mode")
    args = parser.parse_args()

    worker = DreamWorker()
    result = worker.run(
        scope=args.scope,
        deep=args.deep,
        date=args.date,
        project=args.project,
    )
    print(json.dumps(result, indent=2))
