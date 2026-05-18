"""Filesystem store for Hermes Local Memory.

Manages:
  - Raw JSONL: ``~/.hermes/memory/raw/YYYY/YYYY-MM-DD/{session_id}.jsonl``
  - QMD session exports: ``~/.hermes/memory/qmd/{session_id}.md``
  - Daily digests: ``~/.hermes/memory/daily/YYYY-MM-DD.md``
  - Project memory: ``~/.hermes/memory/projects/{project}/memory.md``

Phase 1 (T-001): stub. JSONL append in story 1.3.2, QMD in story 1.3.4.
"""

import hashlib
import json
import logging
from collections import defaultdict
from datetime import date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

logger = logging.getLogger(__name__)

_MEMORY_BASE = "memory"  # under $HERMES_HOME


def _utc_date_str() -> str:
    """Return today's date as 'YYYY-MM-DD' in UTC."""
    return date.today().isoformat()


def _utc_year() -> str:
    """Return the 4-digit UTC year string."""
    return str(date.today().year)


class FSStore:
    """Filesystem-backed raw event store.

    Phase 1 (T-001): stub. Full implementation in stories 1.3.2 and 1.3.4.
    """

    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            from hermes_constants import get_hermes_home

            base_path = Path(str(get_hermes_home())) / _MEMORY_BASE
        self.base_path = Path(base_path)
        # Handle pool: session_id -> open file handle (appended to in append_event)
        self._handles: Dict[str, TextIO] = {}
        # Dedup cache: session_id -> set of content hashes already written
        self._seen_hashes: Dict[str, set[str]] = defaultdict(set)

    def _content_hash(self, event: Dict[str, Any]) -> str:
        """Compute SHA256 content_hash for dedup.

        Hash is computed from raw original fields (pre-redaction), matching
        the canonical field order used throughout the pipeline.
        """
        canonical = "".join(
            str(event.get(field, ""))
            for field in (
                "event_id",
                "session_id",
                "turn_id",
                "sequence",
                "timestamp",
                "role",
                "content",
                "agent",
            )
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _jsonl_path(self, session_id: str, date_str: Optional[str] = None) -> Path:
        """Return the JSONL path for a session on a given date.

        Path: raw/YYYY/YYYY-MM-DD/{session_id}.jsonl
        """
        if date_str is None:
            date_str = _utc_date_str()
        year = date_str[:4]  # e.g. "2026" from "2026-05-17"
        return self.base_path / "raw" / year / date_str / f"{session_id}.jsonl"

    def append_event(self, event: Dict[str, Any],
                      date_override: Optional[str] = None) -> str | bool:
        """Append a turn event to the session JSONL file.

        Args:
            event: The event dict. Must contain session_id, timestamp (or use date_override).
            date_override: Optional 'YYYY-MM-DD' string to override the date segment.
                          Useful for tests that need to simulate multi-day sessions.
                          Defaults to today's UTC date derived from timestamp[:10].

        Returns:
            The content hash (str) on success, or False if the event was a dedup hit.
        Raises:
            IOError: if the file cannot be written.
        """
        content_hash = self._content_hash(event)
        session_id = event["session_id"]
        if date_override:
            date_str = date_override
        else:
            date_str = event.get("timestamp", "")[:10]
            if not date_str or len(date_str) < 10:
                date_str = _utc_date_str()

        path = self._jsonl_path(session_id, date_str)

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Per-session dedup: skip if already written this session
        if session_id in self._seen_hashes and content_hash in self._seen_hashes[session_id]:
            logger.debug("Dedup skip for session_id=%s hash=%s", session_id, content_hash)
            return False

        # Compute content hash (pre-redaction) and stamp into event before writing
        event["hash"] = content_hash

        # Lazy-open handle for this session (reuse if already open)
        if session_id not in self._handles:
            self._handles[session_id] = open(path, "a", encoding="utf-8")
        elif self._handles[session_id].name != str(path):
            # Session moved to a new date path — close old, open new
            self._handles[session_id].close()
            self._handles[session_id] = open(path, "a", encoding="utf-8")

        # Write the JSON line
        line = json.dumps(event, ensure_ascii=False)
        self._handles[session_id].write(line + "\n")
        self._handles[session_id].flush()  # ensure written before returning

        # Track hash for dedup
        self._seen_hashes[session_id].add(content_hash)

        logger.debug("Appended event %s to %s (hash=%s)", event.get("event_id"), path, content_hash)
        return content_hash

    def read_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Read all events for a session from JSONL, across all date segments.

        Events are returned in sequence order (sorted by sequence field).
        Returns an empty list if no events are found.
        """
        events: List[Dict[str, Any]] = []
        raw_dir = self.base_path / "raw"

        if not raw_dir.exists():
            return events

        # Scan all year/date directories for this session's JSONL files
        for year_dir in raw_dir.iterdir():
            if not year_dir.is_dir():
                continue
            for date_dir in year_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                jsonl_file = date_dir / f"{session_id}.jsonl"
                if jsonl_file.exists():
                    with open(jsonl_file, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                logger.warning("Corrupt JSONL line in %s", jsonl_file)

        # Sort by sequence number
        events.sort(key=lambda e: e.get("sequence", 0))
        return events

    def close_all(self) -> None:
        """Close all open file handles in the pool."""
        for session_id, handle in list(self._handles.items()):
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()
        self._seen_hashes.clear()

    def _qmd_path(self, session_id: str) -> Path:
        """Return the QMD path for a session.

        Path: qmd/YYYY/YYYY-MM-DD/{session_id}.qmd
        """
        today = date.today()
        year = str(today.year)
        date_str = today.isoformat()
        return self.base_path / "qmd" / year / date_str / f"{session_id}.qmd"

    def _format_tool_call(self, tc: Dict[str, Any]) -> str:
        """Format a tool call dict as a brief one-liner summary.

        Example: "→ tool_use: memory_query(query='...')"
        """
        try:
            fname = tc.get("function", {}).get("name", tc.get("name", "?"))
            args = tc.get("function", {}).get("arguments", tc.get("arguments", {}))
            if isinstance(args, str):
                args = json.loads(args) if args else {}
            # Show up to 2 key args for brevity
            if isinstance(args, dict):
                parts = [f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:2]]
                args_str = ", ".join(parts)
            else:
                args_str = repr(args)[:40]
            return f"→ tool_use: {fname}({args_str})"
        except Exception:
            return f"→ tool_use: {tc.get('name', '?')}"

    def _build_qmd(self, events: List[Dict[str, Any]]) -> str:
        """Build QMD string from a list of session events."""
        if not events:
            return self._build_qmd_empty()

        # --- Frontmatter ---
        first = events[0]
        last = events[-1]
        session_id = first.get("session_id", "?")
        project = ""
        tags: List[str] = []
        source_refs: List[str] = []

        for ev in events:
            meta = ev.get("metadata", {}) or {}
            if meta.get("project"):
                project = meta["project"]
            tags_from_meta = meta.get("tags", [])
            if isinstance(tags_from_meta, list):
                tags.extend(t for t in tags_from_meta if t not in tags)
            # Build source refs per turn
            seq = ev.get("sequence", 0)
            source_refs.append(f"session:{session_id}#turn={seq}")

        # Collect from top-level fields too (events may store these at top level)
        if not project:
            project = first.get("project", "")
        if not tags:
            tags = first.get("tags", [])

        tags_str = ", ".join(repr(t) for t in tags) if tags else "[]"

        frontmatter_lines = [
            "---",
            f"session_id: {session_id}",
            f"project: {project}",
            f"started_at: {first.get('timestamp', '')}",
            f"ended_at: {last.get('timestamp', '')}",
            f"tags: [{tags_str}]",
            f"source_refs:",
        ]
        for ref in source_refs:
            frontmatter_lines.append(f"  - {ref}")
        frontmatter_lines.append("---")

        # --- Body ---
        lines: List[str] = [""]
        lines.append(f"# Session {session_id}")
        lines.append("")

        for ev in events:
            role = ev.get("role", "?")
            content = ev.get("content", "")
            tool_calls = ev.get("tool_calls", []) or []
            seq = ev.get("sequence", 0)
            turn_id = ev.get("turn_id", "")
            ts = ev.get("timestamp", "")

            # Turn heading with metadata
            lines.append(f"## turn {seq} [{role}] {ts}")
            lines.append("")

            # Content
            if content:
                for chunk in content.split("\n"):
                    lines.append(chunk)
            else:
                lines.append("_(empty)_")

            # Tool call summaries
            for tc in tool_calls:
                lines.append(self._format_tool_call(tc))

            lines.append("")

        return "\n".join(frontmatter_lines) + "\n" + "\n".join(lines)

    def _build_qmd_empty(self) -> str:
        """Build an empty-session QMD."""
        session_id = "unknown"
        return "\n".join([
            "---",
            f"session_id: {session_id}",
            f"project: ''",
            f"started_at: ''",
            f"ended_at: ''",
            "tags: []",
            "source_refs: []",
            "---",
            "",
            f"# Session {session_id}",
            "",
            "_(no turns)_",
            "",
        ])

    def export_qmd(self, session_id: str) -> Path:
        """Export a session to QMD format.

        Reads all events for the session from JSONL, builds QMD content
        with YAML frontmatter + chronological turn body, and writes to
        ``memory/qmd/YYYY/YYYY-MM-DD/{session_id}.qmd``.

        Re-export overwrites safely (no error, no duplication).

        Args:
            session_id: The session to export.

        Returns:
            The Path where the QMD file was written.
        """
        events = self.read_session(session_id)
        content = self._build_qmd(events)

        path = self._qmd_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info("Exported QMD for session %s -> %s", session_id, path)
        return path

    def append_daily(self, date_str: str, content: str) -> Path:
        """Append to a daily digest file."""
        raise NotImplementedError("FSStore.append_daily() — story 1.3.4")