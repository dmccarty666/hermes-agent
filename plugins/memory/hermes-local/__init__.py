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

    def sync_turn(self, user_content: str, assistant_content: str,
                 *, session_id: str = "") -> None:
        """Phase 1: capture not yet wired. Stub for compilation."""
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Phase 1: zero tools registered — tools land in Phase 2."""
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Phase 1: no tools to handle."""
        raise NotImplementedError(f"HermesLocalProvider does not handle tool {tool_name}")

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