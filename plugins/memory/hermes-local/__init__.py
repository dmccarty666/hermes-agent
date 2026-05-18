"""Hermes Local Memory — plugin entry point.

Plugin contract:
  - ``__init__.py`` exposes ``register(ctx)`` which calls ``ctx.register_memory_provider(...)``.
  - ``plugin.yaml`` provides metadata.
  - ``is_available()`` gates activation: returns True when ``memory.provider == 'hermes-local'``.
  - Tool schemas (get_tool_schemas) are empty for Phase 1 — tools land in Phase 2.

Config namespace: ``plugins.hermes-local-memory`` in ``~/.hermes/config.yaml``.
"""

from __future__ import annotations

import logging
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the HermesLocalProvider with the plugin system.

    Activation is gated on ``memory.provider == 'hermes-local'`` in config.
    """
    if not HermesLocalProvider().is_available():
        logger.debug("HermesLocalProvider not available — memory.provider != 'hermes-local'")
        return
    provider = HermesLocalProvider()
    ctx.register_memory_provider(provider)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    """Read ``plugins.hermes-local-memory`` block from ``$HERMES_HOME/config.yaml``."""
    try:
        from hermes_constants import get_hermes_home
        from hermes_cli.config import cfg_get
        config_path = Path(str(get_hermes_home())) / "config.yaml"
        if not config_path.exists():
            return {}
        import yaml
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "hermes-local-memory", default={}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class HermesLocalProvider(MemoryProvider):
    """Hermes Local Memory provider.

    Phase 1 (T-008): QMD exporter. Capture, indexing, search, and tools land
    in subsequent phases.
    """

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._session_id: str | None = None
        self._sync_seq: int = 0

    @property
    def name(self) -> str:
        return "hermes-local"

    def is_available(self) -> bool:
        """Return True when ``memory.provider`` is set to ``hermes-local``."""
        try:
            from hermes_cli.config import cfg_get
            from hermes_constants import get_hermes_home
            config_path = Path(str(get_hermes_home())) / "config.yaml"
            if not config_path.exists():
                return False
            import yaml
            with open(config_path, encoding="utf-8-sig") as f:
                all_config = yaml.safe_load(f) or {}
            provider = cfg_get(all_config, "memory", "provider", default="") or ""
            return provider == "hermes-local"
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # MemoryProvider implementation
    # -----------------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize for a session.

        Sets session_id and ensures the memory base dir exists.
        """
        self._session_id = session_id
        try:
            from hermes_constants import get_hermes_home
            base = Path(str(get_hermes_home())) / "memory"
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def system_prompt_block(self) -> str:
        """Phase 1: no system-prompt contribution yet."""
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Phase 1: prefetch not yet active."""
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Phase 1: prefetch not yet active."""
        pass

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """Capture a turn: redact → JSONL → SQLite sessions/turns/raw_events → audit.

        Args:
            user_content: Raw user message content.
            assistant_content: Raw assistant response content.
            session_id: The Hermes session identifier.

        Raises:
            RuntimeError: if capture fails after all retries (SQLite locked path).
        """
        if not session_id:
            logger.warning("sync_turn called with empty session_id — skipping")
            return

        from hermes_memory_core.write.pipeline import capture_event
        from datetime import datetime, timezone

        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            seq = self._sync_seq
            self._sync_seq += 1

            # Build the user turn event — all EventSchema required fields must be present
            user_event = {
                "event_id": f"evt_{uuid.uuid4().hex[:12]}",
                "session_id": session_id,
                "turn_id": f"turn_{uuid.uuid4().hex[:12]}",
                "timestamp": now_iso,
                "role": "user",
                "content": user_content,
                "agent": "hermes-local",
                "project": "default",
                "source": "cli",
                "tags": [],
                "attachments": [],
                "metadata": {},
                "sequence": seq,
                "content_hash": hashlib.sha256(
                    (session_id + str(seq) + now_iso + "user" + user_content + "hermes-local").encode()
                ).hexdigest(),
                "dream_status": "pending",
                "embedding_status": "pending",
                "index_status": "pending",
            }

            # Build the assistant turn event (same turn_id to group them)
            assistant_event = {
                "event_id": f"evt_{uuid.uuid4().hex[:12]}",
                "session_id": session_id,
                "turn_id": user_event["turn_id"],
                "timestamp": now_iso,
                "role": "assistant",
                "content": assistant_content,
                "agent": "hermes-local",
                "project": "default",
                "source": "cli",
                "tags": [],
                "attachments": [],
                "metadata": {},
                "sequence": seq,
                "content_hash": hashlib.sha256(
                    (session_id + str(seq) + now_iso + "assistant" + assistant_content + "hermes-local").encode()
                ).hexdigest(),
                "dream_status": "pending",
                "embedding_status": "pending",
                "index_status": "pending",
            }

            capture_event(user_event)
            capture_event(assistant_event)

        except Exception as exc:
            logger.error(
                "sync_turn capture failed for session_id=%s: %s",
                session_id,
                exc,
                exc_info=True,
            )
            raise

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Phase 2: delegate to hermes_memory_core/tools.py."""
        from hermes_memory_core.tools import get_tool_schemas as _get_schemas
        return _get_schemas()

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Phase 2: delegate to hermes_memory_core/tools.py."""
        from hermes_memory_core.tools import handle_tool_call as _dispatch
        return _dispatch(tool_name, args, **kwargs)

    def shutdown(self) -> None:
        """Phase 1: nothing to shut down."""
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Called when a session ends — export QMD for this session.

        Reads captured events from JSONL via FSStore and writes the QMD
        archival view. Safe to call even if no events were captured.
        """
        session_id = getattr(self, "_session_id", None)
        if not session_id:
            logger.warning(
                "on_session_end called but no session_id set — skipping QMD export"
            )
            return
        try:
            from hermes_memory_core.store.fs import FSStore

            store = FSStore()
            store.export_qmd(session_id)
            logger.info("QMD exported for session %s on session end", session_id)
        except Exception as exc:
            logger.error("Failed to export QMD for session %s: %s", session_id, exc)