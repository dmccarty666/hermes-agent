# Copyright 2026 David McCarty. All rights reserved.
"""Daily memory file writer for Hermes Local Memory.

Writes a human-readable `~/.hermes/memories/YYYY-MM-DD.md` aggregating:
  - sessions processed
  - topics discussed
  - facts extracted
  - decisions made
  - open questions

Re-run safe: manual content above `<!-- AUTO-GENERATED BELOW -->` is preserved;
auto-generated content below that marker is refreshed/merged on each run.

Called as stage 5 in the dreamer pipeline (TDD §10.1).
"""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

MEMORIES_DIR = os.path.expanduser("~/.hermes/memories")
AUTO_MARKER = "<!-- AUTO-GENERATED BELOW -->"


def _ensure_memories_dir() -> None:
    """Ensure the memories directory exists."""
    Path(MEMORIES_DIR).mkdir(parents=True, exist_ok=True)


def _date_file_path(d: date) -> Path:
    """Return the path to the daily memory file for a given date."""
    return Path(MEMORIES_DIR) / f"{d.isoformat()}.md"


def _read_existing(path: Path) -> tuple[str, str]:
    """Read an existing daily memory file.

    Returns:
        (before_marker, after_marker) — content above and below the
        AUTO-GENERATED marker. If marker absent, returns (full_content, "").
    """
    if not path.exists():
        return "", ""
    content = path.read_text(encoding="utf-8")
    marker_idx = content.find(AUTO_MARKER)
    if marker_idx == -1:
        return content, ""
    return content[:marker_idx], content[marker_idx + len(AUTO_MARKER):]


def _render_section(title: str, items: List[Dict[str, Any]], source_ref_attr: str = "source_ref") -> str:
    """Render a section as markdown with source_refs inline."""
    if not items:
        return f"## {title}\n\n_None recorded._\n"
    lines = [f"## {title}\n"]
    for item in items:
        content = item.get("content", item.get("text", ""))
        if not content:
            continue
        src = item.get(source_ref_attr, item.get("source", ""))
        ref_str = f" ← `{src}`" if src else ""
        lines.append(f"- {content}{ref_str}")
    return "\n".join(lines) + "\n"


def _render_auto_content(
    date_iso: str,
    sessions_processed: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
) -> str:
    """Render the auto-generated portion (everything below the marker)."""
    lines = [
        f"# Memory — {date_iso}\n",
        "<!-- AUTO-GENERATED BELOW -->\n",
    ]

    # Sessions
    if sessions_processed:
        lines.append("## Sessions Processed\n")
        for sess in sessions_processed:
            sid = sess.get("session_id", "unknown")
            title = sess.get("title", "Untitled")
            lines.append(f"- [{sid}] {title}")
        lines.append("\n")
    else:
        lines.append("## Sessions Processed\n\n_None processed._\n")

    # Topics (collected from facts/projects if available)
    topics = list(set(
        sess.get("project", "")
        for sess in sessions_processed
        if sess.get("project")
    ))
    if topics:
        lines.append("## Topics\n")
        for topic in sorted(topics):
            lines.append(f"- {topic}")
        lines.append("\n")
    else:
        lines.append("## Topics\n\n_None recorded._\n")

    # Facts
    lines.append(_render_section("Facts Extracted", facts, source_ref_attr="source_ref"))

    # Decisions
    lines.append(_render_section("Decisions Made", decisions, source_ref_attr="source_ref"))

    # Questions
    lines.append(_render_section("Open Questions", questions, source_ref_attr="source_ref"))

    return "\n".join(lines)


def write_daily_memory(
    d: date,
    sessions_processed: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
) -> Path:
    """Write or update the daily memory file for a given date.

    Manual content above the `<!-- AUTO-GENERATED BELOW -->` marker is preserved.
    The auto-generated portion below the marker is replaced on every call.

    Args:
        d: Date for the memory file.
        sessions_processed: List of session dicts with at least session_id and title.
        facts: List of fact dicts with 'content' and 'source_ref' keys.
        decisions: List of decision dicts with 'content' and 'source_ref' keys.
        questions: List of question dicts with 'content' and 'source_ref' keys.

    Returns:
        Path to the written file.
    """
    _ensure_memories_dir()
    path = _date_file_path(d)
    date_iso = d.isoformat()

    before, after = _read_existing(path)

    # Detect whether the marker existed in the original file.
    # We check the ORIGINAL file content (before + marker + after) to see if
    # the marker string appears in it naturally.
    original_content = before + (AUTO_MARKER + after if after else "")
    has_marker = AUTO_MARKER in original_content

    if not has_marker:
        # No marker found anywhere — replace the file entirely (backward compat
        # for existing files written before this feature existed).
        combined = _render_auto_content(
            date_iso=date_iso,
            sessions_processed=sessions_processed,
            facts=facts,
            decisions=decisions,
            questions=questions,
        )
    else:
        auto_content = _render_auto_content(
            date_iso=date_iso,
            sessions_processed=sessions_processed,
            facts=facts,
            decisions=decisions,
            questions=questions,
        )
        # Preserve content above the marker; replace everything below it
        if before.endswith("\n"):
            combined = before + "\n" + auto_content
        elif before:
            combined = before + "\n\n" + auto_content
        else:
            combined = auto_content

    path.write_text(combined, encoding="utf-8")
    return path