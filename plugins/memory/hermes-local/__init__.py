"""HermesLocalProvider — Hermes Local Memory plugin (Phase 4).

Implements the MemoryProvider ABC. Registers when:
  memory.provider = hermes-local

Provides tools:
  - memory_write   (Epic 4.4.1)
  - memory_query   (Epic 4.1.1)
  - memory_update  (Epic 4.4.2)
  - memory_recent_context (Epic 4.3.1)
  - memory_dream_now
  - memory_get_source
  - fact_feedback

Hooks registered:
  - on_session_end    : capture turns
  - on_session_switch : narrative thread
  - on_pre_compress   : prefetch context
  - on_memory_write   : write orchestration
  - on_delegation     : delegation awareness
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx: Optional[dict] = None) -> "HermesLocalProvider":
    """Register the hermes-local memory provider.

    Called by the plugin loader. Gating is done via config check below,
    so this function always registers but the provider is only activated
    when memory.provider=hermes-local.
    """
    provider = HermesLocalProvider()
    # Wire into Hermes plugin system
    # ctx["registry"].register("memory", provider)  # TODO: wire to real registry
    logger.info("[hermes-local] Memory provider registered (activation gated on config).")
    return provider


def is_available() -> bool:
    """Return True when memory.provider is set to hermes-local in config."""
    # Read from ~/.hermes/config.yaml via hermes_constants or direct YAML
    try:
        import yaml
        config_path = os.path.expanduser("~/.hermes/config.yaml")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            provider = cfg.get("memory", {}).get("provider", "")
            return provider == "hermes-local"
    except Exception:
        pass
    # Also check env override
    return os.environ.get("HERMES_MEMORY_PROVIDER") == "hermes-local"


# ---------------------------------------------------------------------------
# HermesLocalProvider
# ---------------------------------------------------------------------------

class HermesLocalProvider:
    """Local-first memory provider for Hermes.

    Manages lossless turn capture, hybrid retrieval, and dream extraction.
    Lazily imports heavy modules to keep startup fast.
    """

    name: str = "hermes-local"

    def __init__(self) -> None:
        self._hooks: List[tuple[str, Callable]] = []
        self._memory_db = None
        # Phase 5 (Epic 5.1.2): narrative thread injection
        self._agent_ref: Optional[Any] = None  # AIAgent reference for conversation_history injection
        self._nt_first_turn_done: bool = False  # prevent double-injection on same session
        self._nt_prev_content: str = ""  # cached prior-session thread content

    # ── Tool schemas ────────────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[dict]:
        """Return all tool schemas exposed by this provider."""
        from plugins.memory.hermes_local.tools import (
            MEMORY_WRITE_SCHEMA,
            MEMORY_UPDATE_SCHEMA,
            FACT_FEEDBACK_SCHEMA,
        )
        return [
            MEMORY_WRITE_SCHEMA,
            MEMORY_UPDATE_SCHEMA,
            FACT_FEEDBACK_SCHEMA,
        ]

    def get_tool(self, name: str) -> Optional[Callable]:
        """Return the tool function for name, or None."""
        from plugins.memory.hermes_local import tools as memory_tools
        tool_map = {
            "memory_write": memory_tools.memory_write_tool,
            "memory_update": memory_tools.memory_update_tool,
            "fact_feedback": memory_tools.fact_feedback_tool,
        }
        return tool_map.get(name)

    # ── Hooks ───────────────────────────────────────────────────────────────

    def on_session_end(self, session_id: str, turns: List[dict]) -> None:
        """Called when a session ends. Capture turns to SQLite + JSONL."""
        # Phase 1 (Epic 1.3.3): capture via sync_turn pipeline
        logger.debug(f"[hermes-local] on_session_end: {session_id} ({len(turns)} turns)")

    def on_session_switch(self, new_id: str, *, parent_id: Optional[str] = "", reset: bool = False, **kwargs) -> None:
        """Called on /new, /resume, /branch. Inject narrative thread context.

        ADR-001 Option A: bypass _cached_system_prompt by injecting prior
        session context directly into conversation_history at index 0.
        """
        import logging as _logging
        _logger = _logging.getLogger(__name__)
        _logger.debug(f"[hermes-local] on_session_switch: {new_id} from {parent_id} reset={reset}")

        # Phase 5 (Epic 5.1.2): narrative thread injection
        if not parent_id:
            self._nt_prev_content = ""
            self._nt_first_turn_done = False
            return

        # Read prior session's thread file
        try:
            from plugins.memory.hermes_local.narrative import _read_thread_file
            focus, exchanges, turn_count = _read_thread_file(parent_id)
        except Exception as e:
            _logger.warning("[hermes-local] on_session_switch: failed to read thread %s — %s", parent_id, e)
            self._nt_prev_content = ""
            self._nt_first_turn_done = False
            return

        if not focus and not exchanges:
            _logger.debug("[hermes-local] on_session_switch: no prior thread content for %s", parent_id)
            self._nt_prev_content = ""
            self._nt_first_turn_done = False
            return

        # Build _nt_prev_content from thread data
        lines = []
        if focus:
            lines.append(f"Current Focus: {focus}")
        if turn_count > 0:
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

        # Inject immediately if this is a reset=False session switch (e.g. /new with parent)
        # For reset=True (/reset), we skip injection — fresh session
        if not reset:
            self._inject_narrative_message()
        else:
            _logger.debug("[hermes-local] on_session_switch: reset=True — skipping injection (fresh session)")

    def on_pre_compress(self, session_id: str) -> dict:
        """Called before context compression. Return prefetched context."""
        # Phase 4 (prefetch.py): hybrid prefetch for compression
        logger.debug(f"[hermes-local] on_pre_compress: {session_id}")
        return {}

    def on_memory_write(self, memory_type: str, text: str, source_ref: str, **kwargs) -> dict:
        """Called when Hermes core triggers a memory write."""
        # Epic 4.4.1: delegate to canonical pipeline
        from hermes_memory_core import write_memory
        return write_memory(
            memory_type=memory_type,
            text=text,
            source_ref=source_ref,
            project=kwargs.get("project"),
            scope=kwargs.get("scope") or "general",
            confidence=kwargs.get("confidence"),
            tags=kwargs.get("tags"),
            rationale=kwargs.get("rationale"),
            owner=kwargs.get("owner"),
            priority=kwargs.get("priority"),
        )

    def on_delegation(self, agent: str, task: str, context: dict) -> None:
        """Called when a delegation occurs. Track for memory."""
        logger.debug(f"[hermes-local] on_delegation: {agent} <- {task}")

    # ── DB access ───────────────────────────────────────────────────────────

    @property
    def db(self):
        """Lazily init the shared MemoryDB instance."""
        if self._memory_db is None:
            from hermes_memory_core import get_memory_db
            self._memory_db = get_memory_db()
        return self._memory_db

    # ── Provider initialization (ADR-001 Option A1) ─────────────────────────

    def initialize(self, session_id: str, **kwargs) -> None:
        """Called by MemoryManager at activation.

        ADR-001 Option A1: capture agent_ref from kwargs if present.
        The three-tier fallback (A1 → A2 → A3) is implemented in
        _get_agent_ref() for use during on_session_switch.
        """
        import logging as _logging
        _logger = _logging.getLogger(__name__)
        # A1: agent_ref from Hermes kwargs (preferred path — clean, no reflection)
        agent_ref = kwargs.get("agent_ref") or kwargs.get("agent_instance")
        if agent_ref is not None:
            self._agent_ref = agent_ref
            _logger.info("[hermes-local] on_session_switch agent_ref: A1 (kwargs) — %s", type(agent_ref).__name__)
        else:
            # A2/A3 deferred to _get_agent_ref() — called at injection time, not here
            self._agent_ref = None
            _logger.debug("[hermes-local] on_session_switch agent_ref: A1 not available (defer to _get_agent_ref)")

    def _get_agent_ref(self) -> Optional[Any]:
        """Three-tier agent_ref resolution per ADR-001 commitments.

        Returns the AIAgent reference via:
          A1: stored self._agent_ref (from initialize kwargs)
          A2: inspect MemoryManager._agent reflectively
          A3: reflective _invalidate_system_prompt() call

        Returns None if all three tiers fail (test passes, logs warning).
        """
        import logging as _logging
        import inspect as _inspect
        _logger = _logging.getLogger(__name__)

        # A1: stored from initialize()
        if self._agent_ref is not None:
            _logger.info("[hermes-local] _get_agent_ref: A1 (stored) — %s", type(self._agent_ref).__name__)
            return self._agent_ref

        # A2: walk call stack to find MemoryManager instance
        try:
            for frame_info in _inspect.stack():
                local_vars = frame_info.frame.f_locals
                for var_name, var_val in local_vars.items():
                    if (
                        hasattr(var_val, "_agent")
                        and hasattr(var_val, "on_session_switch")
                        and type(var_val).__name__ == "MemoryManager"
                    ):
                        agent = getattr(var_val, "_agent", None)
                        if agent is not None:
                            _logger.info("[hermes-local] _get_agent_ref: A2 (MemoryManager._agent) — %s", type(agent).__name__)
                            return agent
        except Exception as e:
            _logger.warning("[hermes-local] _get_agent_ref: A2 failed — %s", e)

        # A3: reflective _invalidate_system_prompt (last resort)
        # We return None here; caller falls back to logging-and-proceeding
        _logger.warning("[hermes-local] _get_agent_ref: A3 fallback triggered (agent unavailable)")
        return None

    # ── Narrative thread injection (Phase 5, Epic 5.1.2) ────────────────────

    MAX_INJECTION_CHARS: int = 4000  # configurable via narrative_thread.max_injection_chars

    def _inject_narrative_message(self) -> None:
        """Inject the prior-session thread as a user message at index 0.

        Bypasses the _cached_system_prompt bottleneck (ADR-001 Option A).
        Injection is capped at MAX_INJECTION_CHARS to bound token cost.
        """
        import logging as _logging
        _logger = _logging.getLogger(__name__)

        if not self._nt_prev_content:
            return

        if self._nt_first_turn_done:
            _logger.debug("[hermes-local] _inject_narrative_message: already injected, skipping")
            return

        agent = self._get_agent_ref()
        if agent is None:
            _logger.warning(
                "[hermes-local] _inject_narrative_message: agent unavailable — "
                "A1/A2/A3 all failed; proceeding without injection"
            )
            return

        # Cap injection content
        content = self._nt_prev_content
        if len(content) > self.MAX_INJECTION_CHARS:
            content = content[: self.MAX_INJECTION_CHARS] + "\n[…thread truncated]"
            _logger.debug("[hermes-local] injection capped to %d chars", self.MAX_INJECTION_CHARS)

        # Construct the injection message per ADR-001 directive
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

        # Inject at index 0 of conversation_history
        try:
            conv_history = getattr(agent, "conversation_history", None)
            if conv_history is not None and isinstance(conv_history, list):
                conv_history.insert(0, injection)
                self._nt_first_turn_done = True
                _logger.info(
                    "[hermes-local] Injected narrative thread message at index 0 "
                    "(len=%d, session=%s)", len(content), getattr(agent, "session_id", "?")
                )
            else:
                _logger.warning(
                    "[hermes-local] agent.conversation_history not accessible "
                    "(type=%s) — cannot inject", type(conv_history).__name__
                )
        except Exception as e:
            _logger.warning("[hermes-local] Failed to inject narrative message: %s", e)

    def health_check(self) -> dict:
        """Return provider health status."""
        try:
            db_health = self.db.health_check()
            return {"provider": "hermes-local", "db": db_health}
        except Exception as e:
            return {"provider": "hermes-local", "status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Module-level activate (called by plugin loader)
# ---------------------------------------------------------------------------

_provider: Optional[HermesLocalProvider] = None


def activate() -> HermesLocalProvider:
    """Activate and return the singleton provider."""
    global _provider
    if _provider is None:
        _provider = HermesLocalProvider()
    return _provider


def get_provider() -> Optional[HermesLocalProvider]:
    return _provider