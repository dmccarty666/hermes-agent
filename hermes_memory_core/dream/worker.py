# Copyright 2026 David McCarty. All rights reserved.
"""Dreamer worker for Hermes Local Memory.

Nightly (or session-end) LLM-driven extraction of:
  - facts
  - decisions
  - open_questions
  - contradictions

Implements the 9-stage pipeline from TDD §10:
  1. scope_selection   — determine which sessions to process
  2. fetch_turns       — load conversation turns for the scope
  3. summarize_session — LLM summary of each session
  4. extract_facts     — LLM fact extraction from summaries
  5. extract_decisions — LLM decision extraction from summaries
  6. extract_questions — LLM open-question extraction from summaries
  7. detect_contradictions — LLM contradiction detection across summaries
  8. update_project_memory — write facts/decisions/questions to SQLite
  9. record_dream_run  — record dream_runs audit entry
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_memory_core.store.sqlite import MemoryDB
from hermes_memory_core.dream.daily_memory import write_daily_memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DreamRun:
    """A single dreamer run."""

    run_id: str
    session_id: str
    scope: str  # "session", "project", "since_last", "all"
    status: str  # "running", "completed", "failed"
    started_at: str
    completed_at: Optional[str] = None
    facts_created: int = 0
    decisions_created: int = 0
    questions_created: int = 0
    contradictions_detected: int = 0
    errors_json: str = ""
    llm_model: str = ""
    input_scope_json: str = ""
    output_path: str = ""


@dataclass
class SessionSummary:
    """LLM-generated summary of a single session."""

    session_id: str
    summary: str
    facts: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    questions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DreamResult:
    """Result of a full dreamer run."""

    dream_run: DreamRun
    session_summaries: List[SessionSummary] = field(default_factory=list)
    facts: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    questions: List[Dict[str, Any]] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path.home() / ".hermes" / "memory" / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        name: Template filename (e.g. "summarize_session.md").

    Returns:
        Template content as a string.

    Raises:
        FileNotFoundError: If the template does not exist.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM client wrapper
# ---------------------------------------------------------------------------


def _call_llm(
    endpoint: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float = 120.0,
) -> str:
    """Call the local LMS / Spark2 inference server.

    Args:
        endpoint: Base URL of the inference server (e.g. "http://192.168.2.105:1234").
        model: Model identifier (e.g. "Qwen3.6-35B").
        system_prompt: System prompt for the chat completion.
        user_prompt: User message content.
        timeout: Request timeout in seconds.

    Returns:
        The assistant's response text.

    Raises:
        requests.exceptions.RequestException: If the request fails.
    """
    import requests

    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _parse_json_block(text: str) -> Any:
    """Extract and parse a JSON block from LLM response text.

    Handles cases where the LLM wraps JSON in markdown fences or
    returns extra prose around the JSON.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed JSON (dict, list, or scalar).

    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted.
    """
    import re

    # Try markdown code fence first
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    # Try to find JSON object/array in the text
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

    # Last resort: return the entire text as a single string value
    return {"_raw": text}


# ---------------------------------------------------------------------------
# DreamWorker — the 9-stage pipeline
# ---------------------------------------------------------------------------


class DreamWorker:
    """LLM-driven memory extraction worker.

    Implements the 9-stage dreamer pipeline:
      1. scope_selection
      2. fetch_turns
      3. summarize_session
      4. extract_facts
      5. extract_decisions
      6. extract_questions
      7. detect_contradictions
      8. update_project_memory
      9. record_dream_run

    Args:
        llm_endpoint: Base URL of the local inference server.
        llm_model: Model identifier to use for extraction.
        db: Optional MemoryDB instance (created lazily if not provided).
    """

    def __init__(
        self,
        llm_endpoint: str = "http://192.168.2.105:1234",
        llm_model: str = "Qwen3.6-35B",
        db: Optional[MemoryDB] = None,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self._db = db

    @property
    def db(self) -> MemoryDB:
        """Return the MemoryDB instance, creating one if needed."""
        if self._db is None:
            self._db = MemoryDB()
            self._db.initialize()
        return self._db

    # ------------------------------------------------------------------
    # Stage 1: scope_selection
    # ------------------------------------------------------------------

    def _scope_selection(
        self,
        scope: str = "since_last",
        session_id: Optional[str] = None,
    ) -> List[str]:
        """Determine which session IDs to process.

        Args:
            scope: One of "session", "project", "since_last", "all".
            session_id: Specific session_id when scope is "session".

        Returns:
            List of session IDs to process.

        Raises:
            ValueError: If scope is invalid.
        """
        conn = self.db._connect()
        try:
            if scope == "session":
                if not session_id:
                    raise ValueError(
                        "session_id is required when scope='session'"
                    )
                return [session_id]

            if scope == "all":
                rows = conn.execute(
                    "SELECT session_id FROM sessions ORDER BY started_at DESC"
                ).fetchall()
                return [r[0] for r in rows]

            if scope == "since_last":
                # Find the most recent completed dream run
                row = conn.execute(
                    "SELECT ended_at FROM dream_runs "
                    "WHERE status='completed' "
                    "ORDER BY ended_at DESC LIMIT 1"
                ).fetchone()
                if row and row[0]:
                    cutoff = row[0]
                    rows = conn.execute(
                        "SELECT session_id FROM sessions "
                        "WHERE started_at > ? "
                        "ORDER BY started_at DESC",
                        (cutoff,),
                    ).fetchall()
                    return [r[0] for r in rows]
                # No previous dream run — fall through to "all"

            # Default to "all" scope (covers "since_last" with no prior run)
            if scope in ("since_last", "all"):
                rows = conn.execute(
                    "SELECT session_id FROM sessions ORDER BY started_at DESC"
                ).fetchall()
                return [r[0] for r in rows]

            if scope == "project":
                raise NotImplementedError(
                    "project scope not yet implemented"
                )

            raise ValueError(f"Unknown scope: {scope}")
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Stage 2: fetch_turns
    # ------------------------------------------------------------------

    def _fetch_turns(
        self, session_id: str, dream_status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Load conversation turns for a session from SQLite.

        Args:
            session_id: Session to load turns for.
            dream_status: If provided, only return turns with this dream_status
                          (e.g. 'pending'). If None, return all turns.

        Returns:
            List of turn dicts (ordered by sequence).
        """
        conn = self.db._connect()
        try:
            if dream_status is not None:
                rows = conn.execute(
                    "SELECT turn_id, sequence, timestamp, role, content, dream_status "
                    "FROM turns WHERE session_id = ? AND dream_status = ? "
                    "ORDER BY sequence ASC",
                    (session_id, dream_status),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT turn_id, sequence, timestamp, role, content, dream_status "
                    "FROM turns WHERE session_id = ? ORDER BY sequence ASC",
                    (session_id,),
                ).fetchall()
            return [
                {
                    "turn_id": r[0],
                    "sequence": r[1],
                    "timestamp": r[2],
                    "role": r[3],
                    "content": r[4] or "",
                    "dream_status": r[5],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def _mark_turns_dreamed(self, turn_ids: List[str]) -> None:
        """Mark turns as dreamed in SQLite.

        Args:
            turn_ids: List of turn_id values to mark as 'dreamed'.
        """
        if not turn_ids:
            return
        conn = self.db._connect()
        try:
            placeholders = ",".join(["?"] * len(turn_ids))
            conn.execute(
                f"UPDATE turns SET dream_status = 'dreamed' WHERE turn_id IN ({placeholders})",
                turn_ids,
            )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.warning("_mark_turns_dreamed failed: %s", exc)
        finally:
            conn.close()

    def _mark_disputed_facts(
        self,
        contradictions: List[Dict[str, Any]],
        all_facts: List[Dict[str, Any]],
    ) -> None:
        """Mark disputed facts in SQLite using contradiction results.

        Iterates through LLM-returned contradictions and looks up the corresponding
        new fact IDs (from the _created_fact_ids mapping set by update_project_memory)
        and marks them as 'disputed' with supersedes_fact_id set.

        Args:
            contradictions: List of contradiction dicts from detect_contradictions.
            all_facts: List of fact dicts extracted in this dream run.
        """
        from hermes_memory_core.dream.contradict import mark_disputed

        created_ids = getattr(self, "_created_fact_ids", {})
        if not created_ids:
            return

        # Build a fact_text -> scope mapping for quick lookup
        fact_text_to_scope: Dict[str, str] = {}
        for f in all_facts:
            txt = f.get("fact_text", f.get("text", ""))
            scope = f.get("scope", "general")
            if txt:
                fact_text_to_scope[txt] = scope

        conn = self.db._connect()
        try:
            for conflict in contradictions:
                # LLM returns fact_a (new/candidate) and fact_b (existing)
                fact_a_text = conflict.get("fact_a", "")
                if not fact_a_text:
                    continue

                # Look up the new fact's ID from the created IDs mapping
                scope = fact_text_to_scope.get(fact_a_text, "general")
                candidate_key = (fact_a_text, scope)
                candidate_fact_id = created_ids.get(candidate_key)
                if not candidate_fact_id:
                    continue

                # Get the existing fact ID (from the DB, referenced by text in conflict)
                existing_text = conflict.get("fact_b", "")
                if not existing_text:
                    continue

                # Look up existing fact by text in the DB
                rows = conn.execute(
                    "SELECT fact_id FROM facts WHERE fact_text = ? LIMIT 1",
                    (existing_text,),
                ).fetchall()
                if not rows:
                    continue
                existing_fact_id = rows[0][0]

                # Mark the candidate (new) fact as disputed
                mark_disputed(conn, candidate_fact_id, existing_fact_id)
                logger.info(
                    "Marked fact %s as disputed (supersedes %s)",
                    candidate_fact_id,
                    existing_fact_id,
                )

            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.warning("_mark_disputed_facts failed: %s", exc)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Stage 3: summarize_session
    # ------------------------------------------------------------------

    def summarize_session(self, session_id: str, turns: List[Dict[str, Any]]) -> str:
        """Generate an LLM summary of a session's conversation.

        Args:
            session_id: Session identifier.
            turns: List of turn dicts (from _fetch_turns).

        Returns:
            Session summary text.
        """
        template = _load_prompt("summarize_session")
        turns_text = "\n".join(
            f"[{t['timestamp']}] {t['role']}: {t['content']}" for t in turns
        )
        user_prompt = template.replace("{SESSION_ID}", session_id).replace("{TURNS}", turns_text)

        try:
            response = _call_llm(
                self.llm_endpoint,
                self.llm_model,
                "You are a session summarizer. Produce a concise, structured summary.",
                user_prompt,
            )
            return response
        except Exception as exc:
            logger.warning("LLM summarize_session failed for %s: %s", session_id, exc)
            return f"SUMMARIZATION_FAILED: {exc}"

    # ------------------------------------------------------------------
    # Stage 4: extract_facts
    # ------------------------------------------------------------------

    def extract_facts(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract facts from conversation turns via LLM.

        Args:
            turns: List of turn dicts.

        Returns:
            List of fact dicts with keys: fact_text, project, scope, confidence, tags.
        """
        template = _load_prompt("extract_facts")
        turns_text = "\n".join(
            f"[{t['timestamp']}] {t['role']}: {t['content']}" for t in turns
        )
        user_prompt = template.replace("{TURNS}", turns_text)

        try:
            response = _call_llm(
                self.llm_endpoint,
                self.llm_model,
                "You are a fact extractor. Return structured facts.",
                user_prompt,
            )
            parsed = _parse_json_block(response)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                # Wrap single fact in a list
                return [parsed]
            return []
        except Exception as exc:
            logger.warning("LLM extract_facts failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Stage 5: extract_decisions
    # ------------------------------------------------------------------

    def extract_decisions(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract decisions from conversation turns via LLM.

        Args:
            turns: List of turn dicts.

        Returns:
            List of decision dicts with keys: decision_text, rationale, project, owner.
        """
        template = _load_prompt("extract_decisions")
        turns_text = "\n".join(
            f"[{t['timestamp']}] {t['role']}: {t['content']}" for t in turns
        )
        user_prompt = template.replace("{TURNS}", turns_text)

        try:
            response = _call_llm(
                self.llm_endpoint,
                self.llm_model,
                "You are a decision extractor. Return structured decisions.",
                user_prompt,
            )
            parsed = _parse_json_block(response)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
            return []
        except Exception as exc:
            logger.warning("LLM extract_decisions failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Stage 6: extract_open_questions
    # ------------------------------------------------------------------

    def extract_open_questions(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract open questions from conversation turns via LLM.

        Args:
            turns: List of turn dicts.

        Returns:
            List of question dicts with keys: question_text, priority, project.
        """
        template = _load_prompt("extract_open_questions")
        turns_text = "\n".join(
            f"[{t['timestamp']}] {t['role']}: {t['content']}" for t in turns
        )
        user_prompt = template.replace("{TURNS}", turns_text)

        try:
            response = _call_llm(
                self.llm_endpoint,
                self.llm_model,
                "You are an open-question extractor. Return structured questions.",
                user_prompt,
            )
            parsed = _parse_json_block(response)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
            return []
        except Exception as exc:
            logger.warning("LLM extract_open_questions failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Stage 7: detect_contradictions
    # ------------------------------------------------------------------

    def detect_contradictions(
        self,
        summaries: List[SessionSummary],
        existing_facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect contradictions between new facts and existing memory.

        Args:
            summaries: List of session summaries from this dream run.
            existing_facts: Previously stored facts to compare against.

        Returns:
            List of contradiction dicts with keys: fact_a, fact_b, conflict_type, resolution.
        """
        # Gather all new facts from summaries
        new_facts = []
        for summary in summaries:
            new_facts.extend(summary.facts)

        if not new_facts or not existing_facts:
            return []

        template = _load_prompt("detect_contradictions")
        facts_text = json.dumps(new_facts, indent=2, ensure_ascii=False)
        existing_text = json.dumps(existing_facts, indent=2, ensure_ascii=False)
        user_prompt = template.replace("{NEW_FACTS}", facts_text).replace(
            "{EXISTING_FACTS}", existing_text
        )

        try:
            response = _call_llm(
                self.llm_endpoint,
                self.llm_model,
                "You are a contradiction detector. Find conflicting facts.",
                user_prompt,
            )
            parsed = _parse_json_block(response)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
            return []
        except Exception as exc:
            logger.warning("LLM detect_contradictions failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Stage 8: update_project_memory
    # ------------------------------------------------------------------

    def update_project_memory(
        self,
        facts: List[Dict[str, Any]],
        decisions: List[Dict[str, Any]],
        questions: List[Dict[str, Any]],
        source_ref: str,
    ) -> Dict[str, int]:
        """Write extracted facts, decisions, and questions to SQLite.

        Args:
            facts: List of fact dicts.
            decisions: List of decision dicts.
            questions: List of question dicts.
            source_ref: Source reference for the dream run (e.g. "dream:{run_id}").

        Returns:
            Dict with counts: facts_created, decisions_created, questions_created.
            Also sets ._created_fact_ids mapping {(fact_text, scope): fact_id}
            for use by contradiction detection.
        """
        conn = self.db._connect()
        now = datetime.now(timezone.utc).isoformat()
        counts = {"facts_created": 0, "decisions_created": 0, "questions_created": 0}
        # Track created fact IDs for contradiction detection
        self._created_fact_ids: Dict[tuple, str] = {}

        try:
            # Write facts
            for fact in facts:
                fact_text = fact.get("fact_text", fact.get("text", ""))
                if not fact_text:
                    continue
                fact_id = f"fact_{uuid.uuid4().hex[:16]}"
                project = fact.get("project", "")
                scope = fact.get("scope", "general")
                confidence = fact.get("confidence", 0.5)
                tags = json.dumps(fact.get("tags", []))
                # Use per-turn source_refs if attached, otherwise fall back to run-level ref
                source_refs_json = json.dumps(fact.get("_source_refs", [source_ref]))
                # Generate a content hash for the fact
                content_hash = hashlib.sha256(fact_text.encode()).hexdigest()[:32]
                conn.execute(
                    """INSERT INTO facts
                       (fact_id, fact_text, content_hash, scope, project, confidence,
                        tags_json, source_refs_json, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        fact_id,
                        fact_text,
                        content_hash,
                        scope,
                        project,
                        confidence,
                        tags,
                        source_refs_json,
                        now,
                        now,
                    ),
                )
                counts["facts_created"] += 1
                # Track for contradiction detection
                self._created_fact_ids[(fact_text, scope)] = fact_id

            # Write decisions
            for dec in decisions:
                decision_text = dec.get("decision_text", dec.get("text", ""))
                if not decision_text:
                    continue
                decision_id = f"decision_{uuid.uuid4().hex[:16]}"
                rationale = dec.get("rationale", "")
                project = dec.get("project", "")
                owner = dec.get("owner", "")
                source_refs_json = json.dumps(dec.get("_source_refs", [source_ref]))
                conn.execute(
                    """INSERT INTO decisions
                       (decision_id, decision_text, rationale, project, owner,
                        status, source_refs_json, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?)""",
                    (
                        decision_id,
                        decision_text,
                        rationale,
                        project,
                        owner,
                        source_refs_json,
                        now,
                        now,
                    ),
                )
                counts["decisions_created"] += 1

            # Write open questions
            for q in questions:
                question_text = q.get("question_text", q.get("text", ""))
                if not question_text:
                    continue
                question_id = f"question_{uuid.uuid4().hex[:16]}"
                project = q.get("project", "")
                priority = q.get("priority", "medium")
                source_refs_json = json.dumps(q.get("_source_refs", [source_ref]))
                conn.execute(
                    """INSERT INTO open_questions
                       (question_id, question_text, project, priority,
                        status, source_refs_json, created_at, updated_at)
                       VALUES (?, ?, ?, ?, 'open', ?, ?, ?)""",
                    (
                        question_id,
                        question_text,
                        project,
                        priority,
                        source_refs_json,
                        now,
                        now,
                    ),
                )
                counts["questions_created"] += 1

            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error("update_project_memory failed: %s", exc)
            raise
        finally:
            conn.close()

        return counts

    # ------------------------------------------------------------------
    # Stage 9: record_dream_run
    # ------------------------------------------------------------------
    # Stage 8b: daily memory file
    # ------------------------------------------------------------------

    def _update_daily_memory(
        self,
        session_ids: List[str],
        facts: List[Dict[str, Any]],
        decisions: List[Dict[str, Any]],
        questions: List[Dict[str, Any]],
    ) -> None:
        """Write or update ~/.hermes/memories/YYYY-MM-DD.md.

        Groups sessions by date and calls write_daily_memory per date.

        Args:
            session_ids: Session IDs processed in this run.
            facts: Extracted facts (with source_ref).
            decisions: Extracted decisions (with source_ref).
            questions: Extracted questions (with source_ref).
        """
        try:
            from datetime import date
            conn = self.db._connect()
            try:
                # Get session metadata (date, title, project) for each session
                placeholders = ",".join(["?"] * len(session_ids))
                rows = conn.execute(
                    f"""SELECT session_id, title, project, started_at
                        FROM sessions WHERE session_id IN ({placeholders})""",
                    session_ids,
                ).fetchall()
            finally:
                conn.close()

            # Group by date
            by_date: Dict[str, List[Dict[str, Any]]] = {}
            for row in rows:
                sid, title, project, started_at = row
                if started_at:
                    d = date.fromisoformat(started_at[:10])
                else:
                    d = date.today()
                date_str = d.isoformat()
                if date_str not in by_date:
                    by_date[date_str] = []
                by_date[date_str].append({
                    "session_id": sid,
                    "title": title or "Untitled",
                    "project": project or "",
                })

            # Write one file per date
            for date_str, sessions in by_date.items():
                d = date.fromisoformat(date_str)
                write_daily_memory(
                    d,
                    sessions_processed=sessions,
                    facts=facts,
                    decisions=decisions,
                    questions=questions,
                )
        except Exception as exc:
            logger.warning("_update_daily_memory failed: %s — continuing", exc)
            # Non-fatal: daily file write failure should not fail the whole dream run

    # ------------------------------------------------------------------
    # Stage 9: record_dream_run
    # ------------------------------------------------------------------

    def record_dream_run(self, dream_run: DreamRun) -> None:
        """Record a dream run in the dream_runs audit table.

        Args:
            dream_run: DreamRun instance with final counts.
        """
        conn = self.db._connect()
        try:
            conn.execute(
                """INSERT INTO dream_runs
                   (dream_run_id, started_at, ended_at, status,
                    input_scope_json, output_path, facts_created,
                    facts_updated, decisions_created, questions_created,
                    contradictions_detected, errors_json, llm_model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    dream_run.run_id,
                    dream_run.started_at,
                    dream_run.completed_at or "",
                    dream_run.status,
                    dream_run.input_scope_json,
                    dream_run.output_path,
                    dream_run.facts_created,
                    0,  # facts_updated — not tracked in this version
                    dream_run.decisions_created,
                    dream_run.questions_created,
                    dream_run.contradictions_detected,
                    dream_run.errors_json,
                    dream_run.llm_model,
                ),
            )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error("record_dream_run failed: %s", exc)
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Main entry point: dream()
    # ------------------------------------------------------------------

    def dream(
        self,
        scope: str = "since_last",
        session_id: Optional[str] = None,
    ) -> DreamResult:
        """Run the full 9-stage dreamer pipeline.

        Args:
            scope: One of "session", "project", "since_last", "all".
            session_id: Specific session_id when scope is "session".

        Returns:
            DreamResult with all extracted data.

        Raises:
            ValueError: If scope is invalid.
        """
        run_id = f"dream_{uuid.uuid4().hex[:16]}"
        started_at = datetime.now(timezone.utc).isoformat()

        dream_run = DreamRun(
            run_id=run_id,
            session_id=session_id or "",
            scope=scope,
            status="running",
            started_at=started_at,
            llm_model=self.llm_model,
            input_scope_json=json.dumps({"scope": scope, "session_id": session_id}),
        )

        # Stage 1: scope selection
        session_ids = self._scope_selection(scope, session_id)
        if not session_ids:
            dream_run.status = "completed"
            dream_run.completed_at = datetime.now(timezone.utc).isoformat()
            self.record_dream_run(dream_run)
            return DreamResult(dream_run=dream_run)

        # Stage 2: fetch ONLY pending turns for each session (idempotency)
        all_turns: Dict[str, List[Dict[str, Any]]] = {}
        for sid in session_ids:
            turns = self._fetch_turns(sid, dream_status="pending")
            all_turns[sid] = turns

        # Collect all processed turn IDs for idempotency marking
        all_processed_turn_ids: List[str] = []

        # Stages 3-6: process each session
        summaries: List[SessionSummary] = []
        all_facts: List[Dict[str, Any]] = []
        all_decisions: List[Dict[str, Any]] = []
        all_questions: List[Dict[str, Any]] = []

        for sid in session_ids:
            turns = all_turns.get(sid, [])
            if not turns:
                continue

            # Track turn IDs for idempotency
            turn_ids_this_session = [t["turn_id"] for t in turns]
            all_processed_turn_ids.extend(turn_ids_this_session)

            # Per-turn source_refs: build a set of all turn sequences for this session
            # so we can include them in the session-level source_ref
            turn_refs = [f"session:{sid}#turn={t['sequence']}" for t in turns]

            # Stage 3: summarize
            summary_text = self.summarize_session(sid, turns)

            # Stage 4: extract facts
            facts = self.extract_facts(turns)
            for f in facts:
                # Attach per-session source_refs pointing to all turns in this session
                f["_source_refs"] = turn_refs
            all_facts.extend(facts)

            # Stage 5: extract decisions
            decisions = self.extract_decisions(turns)
            for d in decisions:
                d["_source_refs"] = turn_refs
            all_decisions.extend(decisions)

            # Stage 6: extract questions
            questions = self.extract_open_questions(turns)
            for q in questions:
                q["_source_refs"] = turn_refs
            all_questions.extend(questions)

            summaries.append(
                SessionSummary(
                    session_id=sid,
                    summary=summary_text,
                    facts=facts,
                    decisions=decisions,
                    questions=questions,
                )
            )

        # Mark all processed turns as 'dreamed' (idempotency)
        self._mark_turns_dreamed(all_processed_turn_ids)

        # Stage 7: detect contradictions (compare with existing memory)
        existing_facts = []
        try:
            conn = self.db._connect()
            try:
                rows = conn.execute(
                    "SELECT fact_id, fact_text, project, scope, confidence "
                    "FROM facts ORDER BY created_at DESC LIMIT 100"
                ).fetchall()
                existing_facts = [
                    {
                        "fact_id": r[0],
                        "fact_text": r[1],
                        "project": r[2],
                        "scope": r[3],
                        "confidence": r[4],
                    }
                    for r in rows
                ]
            finally:
                conn.close()
        except Exception:
            pass

        contradictions = self.detect_contradictions(summaries, existing_facts)

        # Stage 8: update project memory (uses run-level source_ref; per-turn refs attached via _source_refs)
        source_ref = f"dream:{run_id}"
        counts = self.update_project_memory(all_facts, all_decisions, all_questions, source_ref)

        # Wire up contradictory facts: mark disputed facts in DB
        # The contradictions list contains {fact_a, fact_b, ...} where fact_a/b are texts
        if contradictions:
            self._mark_disputed_facts(contradictions, all_facts)

        # Stage 8b: update daily memory file (~/.hermes/memories/YYYY-MM-DD.md)
        self._update_daily_memory(session_ids, all_facts, all_decisions, all_questions)

        # Finalize dream run
        dream_run.status = "completed"
        dream_run.completed_at = datetime.now(timezone.utc).isoformat()
        dream_run.facts_created = counts["facts_created"]
        dream_run.decisions_created = counts["decisions_created"]
        dream_run.questions_created = counts["questions_created"]
        dream_run.contradictions_detected = len(contradictions)

        # Stage 9: record dream run
        self.record_dream_run(dream_run)

        return DreamResult(
            dream_run=dream_run,
            session_summaries=summaries,
            facts=all_facts,
            decisions=all_decisions,
            questions=all_questions,
            contradictions=contradictions,
        )
