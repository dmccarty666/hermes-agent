"""Narrative thread — Phase 5 (Epic 5.1.1, Story T-030).

Rolling SESSION-THREAD/{session_id}.md with 5-exchange window.
Ported from holographic plugin (holographic/__init__.py § Narrative Thread).

File format per TDD §9.1:
  - Markdown with YAML frontmatter (session_id, created_at, turn_count, last_tool_use)
  - Body: last 5 user+assistant exchange pairs
  - Rolling window: older exchanges trimmed on each write

Thread directory: ~/.hermes/SESSION-THREAD/ (per TDD §6.1 filesystem layout)
Retention: files older than 30 days may be cleaned up (configurable).
"""

from __future__ import annotations

import functools
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

# --------------------------------------------------------------------------:
# Constants
# --------------------------------------------------------------------------:

MAX_HISTORY: int = 5  # rolling window size
SNIPPET_LEN: int = 500  # max chars per exchange side (same as holographic)
RETENTION_DAYS: int = 30  # cleanup threshold

# --------------------------------------------------------------------------:
# Timestamp helper
# --------------------------------------------------------------------------:

def _timestamp_cdt() -> str:
    """Locale-aware CDT timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M") + " CDT"


# --------------------------------------------------------------------------:
# Thread file I/O
# --------------------------------------------------------------------------:

@functools.lru_cache(maxsize=1)
def _thread_dir() -> Path:
    """Return the SESSION-THREAD directory path (per TDD §6.1)."""
    return get_hermes_home() / "SESSION-THREAD"


def _reset_path_cache() -> None:
    """Clear the _thread_dir LRU cache so it picks up a new HERMES_HOME.

    Call this after monkeypatching HERMES_HOME in tests.
    """
    _thread_dir.cache_clear()


def _thread_path(session_id: str) -> Path:
    """Return the thread file path for a given session_id."""
    return _thread_dir() / f"{session_id}.md"


# --------------------------------------------------------------------------:
# _read_thread_file — parse existing SESSION-THREAD.md
# --------------------------------------------------------------------------:

def _read_thread_file(session_id: str) -> Tuple[str, List[Dict[str, str]], int]:
    """Read an existing SESSION-THREAD.md.

    Returns (focus, exchanges, turn_count):
      - focus: str — Current Focus line content
      - exchanges: List[Dict] — [{"time": "HH:MM", "user": "...", "ai": "..."}]
      - turn_count: int — Turns This Session from frontmatter

    If the file doesn't exist, returns ("", [], 0).
    """
    path = _thread_path(session_id)
    if not path.exists():
        return "", [], 0

    try:
        content = path.read_text("utf-8")
    except Exception:
        return "", [], 0

    focus = ""
    turn_count = 0
    exchanges: List[Dict[str, str]] = []
    current_exchange: Optional[Dict[str, str]] = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("_Auto-updated") or line.startswith("# SESSION"):
            continue
        if line.startswith("## Last Updated:") or line.startswith("## Turns This Session:"):
            # Extract turn_count value
            if "Turns This Session:" in line:
                try:
                    turn_count = int(line.split("Turns This Session:", 1)[-1].strip())
                except ValueError:
                    pass
            continue
        if line.startswith("## Session Started:") or line.startswith("## Tools Used Recently"):
            continue
        if line.startswith("## Current Focus"):
            focus = line.split("## Current Focus", 1)[-1].strip()
            continue
        if line.startswith("## Exchange History"):
            continue
        if line.startswith("- **"):
            # Parse: "- **HH:MM** User: ... → ..."
            try:
                remainder = line[2:].strip()  # drop leading "- "
                time_part, rest = remainder.split("**", 1)[1].split("**", 1)
                rest = rest.strip()
                if rest.startswith("User:"):
                    _, user_part = rest.split("User:", 1)
                    current_exchange = {"time": time_part, "user": user_part.strip(), "ai": ""}
                elif current_exchange is not None and rest.startswith("→"):
                    current_exchange["ai"] = rest[1:].strip()
                    exchanges.append(current_exchange)
                    current_exchange = None
            except Exception:
                pass
        elif current_exchange is not None and line.startswith("→"):
            current_exchange["ai"] = line[1:].strip()
            exchanges.append(current_exchange)
            current_exchange = None
        elif line.startswith("--"):
            continue

    return focus, exchanges, turn_count


# --------------------------------------------------------------------------:
# _write_thread — write SESSION-THREAD.md with rolling 5-exchange window
# --------------------------------------------------------------------------:

def _write_thread(session_id: str, turns: List[Dict[str, str]]) -> None:
    """Write the SESSION-THREAD.md file for session_id.

    turns: List[Dict] — each dict has "user" and "ai" keys (raw content strings).
            These are the full untruncated exchanges; we truncate on write.

    The file maintains a rolling MAX_HISTORY (5) exchange window.
    Older exchanges are trimmed on each write.
    """
    # Build exchanges list from turns
    exchanges: List[Dict[str, str]] = []
    for turn in turns:
        user = turn.get("user", "")[:SNIPPET_LEN]
        ai = turn.get("ai", "")[:SNIPPET_LEN]
        now = datetime.now(timezone.utc).strftime("%H:%M")
        exchanges.append({"time": now, "user": user, "ai": ai})

    # Rolling window — keep last MAX_HISTORY
    if len(exchanges) > MAX_HISTORY:
        exchanges = exchanges[-MAX_HISTORY:]

    # Build content
    content = _build_thread_content(
        focus="",
        exchanges=exchanges,
        turn_count=len(turns),
        session_start=_timestamp_cdt(),
        tools_used=[],
    )

    # Ensure directory exists
    thread_dir = _thread_dir()
    thread_dir.mkdir(parents=True, exist_ok=True)

    # Write file
    path = _thread_path(session_id)
    try:
        path.write_text(content, "utf-8")
    except Exception as e:
        # Non-fatal — don't crash the session end hook
        import logging
        logging.getLogger(__name__).warning(
            "[narrative] Failed to write thread file %s: %s", path, e
        )


# --------------------------------------------------------------------------:
# _build_thread_content — build the markdown content (same as holographic)
# --------------------------------------------------------------------------:

def _build_thread_content(
    focus: str,
    exchanges: List[Dict[str, str]],
    turn_count: int,
    session_start: str,
    tools_used: List[str],
) -> str:
    """Build the SESSION-THREAD.md markdown content."""
    tools_list = ", ".join(tools_used[-10:]) if tools_used else "none"
    focus_str = focus or "(system/heartbeat)"

    history_lines = ""
    for ex in exchanges:
        history_lines += f"- **{ex['time']}** User: {ex['user']}\n  → {ex['ai']}\n"

    return f"""# SESSION-THREAD.md — Working Memory
_Auto-updated by Narrative Thread after each exchange. Survives compaction._

## Last Updated: {_timestamp_cdt()}
## Turns This Session: {turn_count}
## Session Started: {session_start}

## Current Focus
{focus_str}

## Tools Used Recently
{tools_list}

## Exchange History
{history_lines or '_No exchanges yet this session._'}

---
_Write brief notes below this line if needed. They'll persist until next auto-update._
""".strip()


# --------------------------------------------------------------------------:
# _cleanup_old_threads — remove thread files older than RETENTION_DAYS
# --------------------------------------------------------------------------:

def _cleanup_old_threads(retention_days: int = RETENTION_DAYS) -> None:
    """Delete thread files older than retention_days."""
    import logging

    logger = logging.getLogger(__name__)
    thread_dir = _thread_dir()
    if not thread_dir.exists():
        return

    cutoff = datetime.now(timezone.utc).timestamp() - (retention_days * 86400)
    cleaned = 0
    for f in thread_dir.iterdir():
        if f.suffix == ".md" and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                cleaned += 1
            except Exception as e:
                logger.debug("[narrative] Failed to delete old thread %s: %s", f, e)
    if cleaned:
        logger.info("[narrative] Cleaned %d old thread files (retention=%dd)", cleaned, retention_days)