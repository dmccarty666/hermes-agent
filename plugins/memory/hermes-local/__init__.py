"""HermesLocalProvider — Hermes Local Memory plugin (Phase 4).

Implements the MemoryProvider ABC. Registers when:
  memory.provider = hermes-local

Provides tools (read-oriented schemas exposed to the model — write/update/feedback
are kept internal so the model doesn't accidentally bypass the dreamer pipeline):
  - memory_query           (Epic 4.1.1)
  - memory_get_source      (Epic 4.1.2)
  - memory_recent_context  (Epic 4.3.1)

The full handler registry in tools.py also dispatches:
  - memory_write           (Epic 4.4.1)
  - memory_update          (Epic 4.4.2)
  - memory_dream_now
  - fact_feedback

Hooks registered:
  - on_session_end    : capture turns
  - on_session_switch : narrative thread
  - on_pre_compress   : prefetch context
  - on_memory_write   : write orchestration
  - on_delegation     : delegation awareness
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (sibling-module lookup — directory has a hyphen in its name)
# ---------------------------------------------------------------------------

# Plugin loader registers submodules under the hyphenated name, e.g.
# "plugins.memory.hermes-local.tools". Python's "from x import y" can't
# parse a hyphen, so we resolve siblings via sys.modules / importlib.

_PLUGIN_PKG = "plugins.memory.hermes-local"


def _sibling(modname: str):
    """Return a loaded sibling submodule, importing it if needed."""
    fq = f"{_PLUGIN_PKG}.{modname}"
    mod = sys.modules.get(fq)
    if mod is not None:
        return mod
    try:
        return importlib.import_module(fq)
    except Exception:
        # Fall back to spec_from_file_location so this works even when the
        # plugin is loaded outside the normal plugin discovery path.
        here = Path(__file__).parent
        sub_path = here / f"{modname}.py"
        if not sub_path.exists():
            raise
        import importlib.util as _util
        spec = _util.spec_from_file_location(fq, str(sub_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {fq}")
        loaded = _util.module_from_spec(spec)
        sys.modules[fq] = loaded
        spec.loader.exec_module(loaded)
        return loaded


def _hermes_home() -> Path:
    """Active Hermes home — honors HERMES_HOME env var, falls back to ~/.hermes."""
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)
    return Path(os.path.expanduser("~/.hermes"))


def _read_provider_config_value() -> str:
    """Read memory.provider from the active config.yaml. Empty string if missing."""
    config_path = _hermes_home() / "config.yaml"
    if not config_path.exists():
        return ""
    try:
        import yaml
        with open(config_path, encoding="utf-8-sig") as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get("memory", {}) or {}).get("provider", "") or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> "HermesLocalProvider":
    """Register the hermes-local memory provider.

    Called by the plugin loader. Gating is done via config check in
    is_available() so this function always returns a provider instance —
    the agent activates it only when memory.provider == 'hermes-local'.
    """
    provider = HermesLocalProvider()
    if ctx is not None and hasattr(ctx, "register_memory_provider"):
        ctx.register_memory_provider(provider)
    logger.info("[hermes-local] Memory provider registered (activation gated on config).")
    return provider


# ---------------------------------------------------------------------------
# HermesLocalProvider
# ---------------------------------------------------------------------------

class HermesLocalProvider(MemoryProvider):
    """Local-first memory provider for Hermes.

    Manages lossless turn capture, hybrid retrieval, and dream extraction.
    Lazily imports heavy modules to keep startup fast.
    """

    def __init__(self) -> None:
        self._hooks: List[tuple[str, Callable]] = []
        self._memory_db = None
        self._session_id: str = ""
        self._hermes_home: Path = _hermes_home()
        self._turn_sequence: int = 0
        self._seq_lock: "threading.Lock" = threading.Lock()
        # Phase 5 (Epic 5.1.2): narrative thread injection
        self._agent_ref: Optional[Any] = None
        self._nt_first_turn_done: bool = False
        self._nt_prev_content: str = ""

    # ── ABC: name ───────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "hermes-local"

    # ── ABC: is_available ───────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True when memory.provider == 'hermes-local' AND SQLite schema is present.

        Honors HERMES_HOME so tests can point at a temp config.yaml.
        The schema check is a cheap sqlite_master lookup; if the DB file
        is missing entirely we still report available since the schema
        will be auto-created on first use.
        """
        provider = _read_provider_config_value()
        if provider != "hermes-local":
            # Env override is also accepted (test/dev convenience)
            if os.environ.get("HERMES_MEMORY_PROVIDER") != "hermes-local":
                return False

        # Schema presence: only enforce when the DB file already exists.
        # A missing DB is a fresh install — the store will lazily init it.
        db_path = self._hermes_home / "memory" / "index" / "memory.sqlite"
        if not db_path.exists():
            return True
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            try:
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='facts'"
                ).fetchone()
                return row is not None
            finally:
                conn.close()
        except Exception as e:
            logger.debug("[hermes-local] is_available schema check failed: %s", e)
            return False

    # ── ABC: initialize ─────────────────────────────────────────────────────

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize for a session.

        kwargs may include hermes_home, platform, agent_context,
        agent_identity, user_id, parent_session_id, agent_ref.
        """
        self._session_id = session_id or ""
        hh = kwargs.get("hermes_home")
        if hh:
            self._hermes_home = Path(hh)
        else:
            self._hermes_home = _hermes_home()
        self._turn_sequence = 0

        # Narrative thread agent_ref (ADR-001 Option A1)
        agent_ref = kwargs.get("agent_ref") or kwargs.get("agent_instance")
        if agent_ref is not None:
            self._agent_ref = agent_ref
            logger.info(
                "[hermes-local] initialize: A1 agent_ref captured (%s)",
                type(agent_ref).__name__,
            )
        else:
            self._agent_ref = None

        logger.info(
            "[hermes-local] Initialized for session=%s home=%s",
            session_id,
            self._hermes_home,
        )

    # ── ABC: system_prompt_block ────────────────────────────────────────────

    def system_prompt_block(self) -> str:
        """Dynamic memory stats string injected into the system prompt."""
        try:
            from hermes_memory_core.store.sqlite import get_memory_store
            store = get_memory_store()
            conn = store._conn_or_init()
            n_facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            n_decisions = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            n_questions = conn.execute(
                "SELECT COUNT(*) FROM open_questions WHERE status='open'"
            ).fetchone()[0]
        except Exception as e:
            logger.debug("[hermes-local] system_prompt_block stats failed: %s", e)
            return (
                "# Hermes Local Memory\n"
                "Active. Lossless turn capture + hybrid retrieval (FTS5 + Qdrant + HRR).\n"
                "Use memory_query to search, memory_recent_context for session warm-up, "
                "memory_get_source to resolve source_refs."
            )

        if n_facts == 0 and n_decisions == 0 and n_questions == 0:
            return (
                "# Hermes Local Memory\n"
                "Active. Empty store — turns are captured automatically and the dreamer "
                "will extract facts/decisions/questions in the background.\n"
                "Use memory_query to search, memory_recent_context for warm-up."
            )

        return (
            f"# Hermes Local Memory\n"
            f"Active. {n_facts} facts, {n_decisions} decisions, {n_questions} open questions.\n"
            f"Use memory_query to search (modes: hybrid/semantic/keyword/facts/decisions/recent), "
            f"memory_recent_context for session warm-up, memory_get_source to resolve refs."
        )

    # ── ABC: prefetch ───────────────────────────────────────────────────────

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant memory using hybrid search, falling back to FTS5.

        Returns a markdown-formatted snippet block or empty string.
        """
        if not query or not query.strip():
            return ""

        # Use MemoryDB (subclass of MemoryStore) so backends that need
        # _connect() work. fts5_search will auto-create one if we pass None,
        # but constructing here keeps the call site explicit.
        store = None
        try:
            from hermes_memory_core.store.sqlite import MemoryDB
            store = MemoryDB()
            store.initialize()
        except Exception as e:
            logger.debug("[hermes-local] prefetch: MemoryDB init failed: %s", e)
            store = None

        try:
            from hermes_memory_core.search import hybrid as _hybrid

            result = _hybrid.search(
                query, mode="hybrid", limit=5, memory_db=store,
                filters={"index_status": "indexed"},
            )
            hits = result.get("results", []) if isinstance(result, dict) else []
            if hits:
                return self._format_prefetch_hits(hits, source="hybrid")
        except Exception as e:
            logger.debug("[hermes-local] hybrid prefetch failed: %s — falling back to FTS5", e)

        # Fallback: FTS5 keyword search
        try:
            from hermes_memory_core.search.fts5 import fts5_search

            rows = fts5_search(
                query,
                filters={"index_status": "indexed"},
                table="turns",
                limit=5,
                memory_db=store,
            )
            if rows:
                return self._format_prefetch_hits(rows, source="fts5")
        except Exception as e:
            logger.debug("[hermes-local] FTS5 prefetch also failed: %s", e)

        return ""

    @staticmethod
    def _format_prefetch_hits(hits: list, *, source: str) -> str:
        lines = [f"## Hermes Local Memory ({source})"]
        for h in hits[:5]:
            content = (
                h.get("content")
                or h.get("snippet")
                or h.get("fact_text")
                or h.get("text")
                or ""
            )
            if not content:
                continue
            score = h.get("score") or h.get("rank") or 0
            try:
                score_f = float(score)
                lines.append(f"- [{score_f:.2f}] {content[:240]}")
            except Exception:
                lines.append(f"- {content[:240]}")
        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    # ── ABC: sync_turn ──────────────────────────────────────────────────────

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Persist a completed turn to SQLite + raw JSONL.

        Writes user + assistant rows into the turns table via MemoryStore.
        Best-effort: any write failure is logged but never raised — memory
        write must never break the conversation.
        """
        sid = session_id or self._session_id or "unknown"
        try:
            from hermes_memory_core.store.sqlite import get_memory_store
            store = get_memory_store()
            store.upsert_session({
                "session_id": sid,
                "agent": "hermes",
                "title": None,
                "project": "default",
                "started_at": _utc_iso(),
                "ended_at": None,
                "summary": None,
                "qmd_path": None,
                "raw_path": None,
                "source": "agent",
                "platform": os.environ.get("HERMES_PLATFORM", "cli"),
                "created_at": _utc_iso(),
                "updated_at": _utc_iso(),
            })

            for role, content in (("user", user_content), ("assistant", assistant_content)):
                if not content:
                    continue
                with self._seq_lock:
                    self._turn_sequence += 1
                    turn_id = f"{sid}#t={self._turn_sequence:06d}"
                content_hash = _content_hash(content)
                store.insert_turn_if_not_exists({
                    "turn_id": turn_id,
                    "session_id": sid,
                    "sequence": self._turn_sequence,
                    "timestamp": _utc_iso(),
                    "role": role,
                    "content": content,
                    "raw_content_hash": content_hash,
                    "content_hash": content_hash,
                    "project": "default",
                    "tags_json": None,
                    "tool_calls_json": None,
                    "attachments_json": None,
                    "metadata_json": None,
                    "parent_turn_id": None,
                    "index_status": "pending",
                    "dream_status": "pending",
                    "redaction_applied": 0,
                    "redaction_types_json": None,
                })
        except Exception as e:
            logger.warning("[hermes-local] sync_turn failed: %s", e)

    # ── ABC: get_tool_schemas ───────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return the read-side tool schemas exposed to the model."""
        try:
            tools = _sibling("tools")
        except Exception as e:
            logger.warning("[hermes-local] failed to import tools module: %s", e)
            return []

        return [
            tools.MEMORY_QUERY_SCHEMA,
            tools.MEMORY_GET_SOURCE_SCHEMA,
            tools.MEMORY_RECENT_CONTEXT_SCHEMA,
        ]

    # ── ABC: handle_tool_call ───────────────────────────────────────────────

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch to the tools module handler.

        Returns a JSON string (the tool result).
        """
        import json
        try:
            tools = _sibling("tools")
        except Exception as e:
            return json.dumps({"error": f"tools module unavailable: {e}"})

        try:
            result = tools.handle_hermes_local_tool_call(tool_name, args or {})
        except Exception as e:
            logger.exception("[hermes-local] handle_tool_call(%s) raised", tool_name)
            return json.dumps({"error": f"{tool_name} failed: {e}"})

        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, default=str)
        except Exception:
            return json.dumps({"error": "result not JSON-serialisable", "repr": repr(result)})

    # ── ABC: shutdown ───────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Close any open connections and flush pending writes."""
        # Close MemoryDB (init.py) if cached
        try:
            if self._memory_db is not None and hasattr(self._memory_db, "close"):
                self._memory_db.close()
        except Exception as e:
            logger.debug("[hermes-local] shutdown MemoryDB.close failed: %s", e)
        finally:
            self._memory_db = None

        # Close the MemoryStore singleton too
        try:
            from hermes_memory_core.store.sqlite import get_memory_store
            store = get_memory_store()
            if hasattr(store, "close"):
                store.close()
                logger.info("[hermes-local] MemoryStore connection closed.")
        except Exception as e:
            logger.debug("[hermes-local] shutdown MemoryStore.close failed: %s", e)

    # ── Optional hook: on_session_end ──────────────────────────────────────

    def on_session_end(self, messages) -> None:
        """Called when a session ends. Phase 1 (Epic 1.3.3): handled via sync_turn pipeline."""
        try:
            n = len(messages) if messages is not None else 0
        except TypeError:
            n = 0
        logger.debug("[hermes-local] on_session_end: %s (%d turns)", self._session_id, n)

    # ── Optional hook: on_session_switch (narrative thread) ────────────────

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        """Update internal session_id; inject prior narrative thread on resume."""
        logger.debug(
            "[hermes-local] on_session_switch: %s from %s reset=%s",
            new_session_id, parent_session_id, reset,
        )
        self._session_id = new_session_id or self._session_id
        self._turn_sequence = 0

        if not parent_session_id or reset:
            self._nt_prev_content = ""
            self._nt_first_turn_done = False
            return

        try:
            narrative = _sibling("narrative")
            focus, exchanges, turn_count = narrative._read_thread_file(parent_session_id)
        except Exception as e:
            logger.warning(
                "[hermes-local] on_session_switch: read_thread_file(%s) failed — %s",
                parent_session_id, e,
            )
            self._nt_prev_content = ""
            self._nt_first_turn_done = False
            return

        if not focus and not exchanges:
            self._nt_prev_content = ""
            self._nt_first_turn_done = False
            return

        lines = []
        if focus:
            lines.append(f"Current Focus: {focus}")
        if turn_count:
            lines.append(f"Turns This Session: {turn_count}")
        if exchanges:
            lines.append("Recent Exchanges:")
            for ex in exchanges[-5:]:
                if ex.get("user"):
                    lines.append(f"  User: {ex['user'][:300]}")
                if ex.get("ai"):
                    lines.append(f"  → AI: {ex['ai'][:300]}")

        self._nt_prev_content = "\n".join(lines)
        self._nt_first_turn_done = False
        self._inject_narrative_message()

    # ── Optional hook: on_pre_compress ─────────────────────────────────────

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Return text to preserve through compression. Currently empty."""
        logger.debug("[hermes-local] on_pre_compress: %d messages", len(messages or []))
        return ""

    # ── Optional hook: on_memory_write ─────────────────────────────────────

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory tool writes via the canonical pipeline."""
        if action != "add" or not content:
            return
        try:
            from hermes_memory_core.write.pipeline import write_memory
            scope = "user" if target == "user" else "general"
            write_memory(
                memory_type="fact",
                text=content,
                scope=scope,
                source_ref=(metadata or {}).get("source_ref") or "memory_tool",
            )
        except Exception as e:
            logger.debug("[hermes-local] on_memory_write mirror failed: %s", e)

    # ── Optional hook: on_delegation ───────────────────────────────────────

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        logger.debug("[hermes-local] on_delegation: child=%s task=%r", child_session_id, task[:80])

    # ── DB access ──────────────────────────────────────────────────────────

    @property
    def db(self):
        """Lazily init the shared MemoryDB instance."""
        if self._memory_db is None:
            from hermes_memory_core import get_memory_db
            self._memory_db = get_memory_db()
        return self._memory_db

    # ── Narrative thread injection (Phase 5) ───────────────────────────────

    MAX_INJECTION_CHARS: int = 4000

    def _get_agent_ref(self) -> Optional[Any]:
        """Three-tier agent_ref resolution (A1 stored → A2 stack walk → A3 None)."""
        import inspect as _inspect

        if self._agent_ref is not None:
            return self._agent_ref
        try:
            for frame_info in _inspect.stack():
                local_vars = frame_info.frame.f_locals
                for var_val in local_vars.values():
                    if (
                        hasattr(var_val, "_agent")
                        and hasattr(var_val, "on_session_switch")
                        and type(var_val).__name__ == "MemoryManager"
                    ):
                        agent = getattr(var_val, "_agent", None)
                        if agent is not None:
                            return agent
        except Exception as e:
            logger.debug("[hermes-local] _get_agent_ref A2 failed: %s", e)
        return None

    def _inject_narrative_message(self) -> None:
        """Inject prior-session thread as a user message at index 0 of conversation_history."""
        if not self._nt_prev_content or self._nt_first_turn_done:
            return
        agent = self._get_agent_ref()
        if agent is None:
            logger.debug("[hermes-local] _inject_narrative_message: no agent_ref available")
            return

        content = self._nt_prev_content
        if len(content) > self.MAX_INJECTION_CHARS:
            content = content[: self.MAX_INJECTION_CHARS] + "\n[…thread truncated]"

        injection = {
            "role": "user",
            "content": (
                "In our previous session we were working on the following context:\n\n"
                f"{content}\n\n"
                "Briefly note what you found above from the last session "
                "and ask if there's anything to continue."
            ),
            "metadata": {"source": "narrative_thread_injection"},
        }

        try:
            conv_history = getattr(agent, "conversation_history", None)
            if isinstance(conv_history, list):
                conv_history.insert(0, injection)
                self._nt_first_turn_done = True
                logger.info(
                    "[hermes-local] Injected narrative thread (len=%d, session=%s)",
                    len(content), getattr(agent, "session_id", "?"),
                )
        except Exception as e:
            logger.warning("[hermes-local] narrative injection failed: %s", e)

    # ── Diagnostics ────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        try:
            return {"provider": self.name, "db": self.db.health_check()}
        except Exception as e:
            return {"provider": self.name, "status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Module-level convenience for callers that don't have an instance."""
    return HermesLocalProvider().is_available()


_provider: Optional[HermesLocalProvider] = None


def activate() -> HermesLocalProvider:
    """Activate and return the singleton provider."""
    global _provider
    if _provider is None:
        _provider = HermesLocalProvider()
    return _provider


def get_provider() -> Optional[HermesLocalProvider]:
    return _provider


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _content_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
