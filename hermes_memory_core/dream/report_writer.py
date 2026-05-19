# Copyright 2026 David McCarty. All rights reserved.
"""Dream report writer — stages 8-9 of the TDD §10.1 pipeline.

Writes a structured markdown report to ``~/.hermes/memory/dreams/YYYY-MM-DD-HHMM.md``
after each dream run completes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_memory_core.dream.worker import DreamResult, DreamRun

logger = logging.getLogger(__name__)

# Directory where dream reports land
_DREAMS_DIR = Path.home() / ".hermes" / "memory" / "dreams"


def _format_iso(iso_str: Optional[str]) -> str:
    """Parse an ISO timestamp and return a human-readable string."""
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        return iso_str


def _fmt_count(value: int) -> str:
    return str(value) if value else "0"


def write_dream_report(result: DreamResult, started_at: str, ended_at: str) -> Path:
    """Write a structured dream report to disk.

    Args:
        result:      DreamResult from DreamWorker.dream().
        started_at:  ISO timestamp when the run began.
        ended_at:    ISO timestamp when the run finished.

    Returns:
        Path to the written report file.
    """
    _DREAMS_DIR.mkdir(parents=True, exist_ok=True)

    dr = result.dream_run
    now_local = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")

    report_path = _DREAMS_DIR / f"{now_local}.md"

    # ── YAML frontmatter ──────────────────────────────────────────────────
    sessions_processed = len(result.session_summaries)
    turns_processed = sum(
        len(s.facts) + len(s.decisions) + len(s.questions) for s in result.session_summaries
    )

    frontmatter = {
        "dream_run": {
            "run_id": dr.run_id,
            "scope": dr.scope,
            "started_at": started_at,
            "ended_at": ended_at,
            "status": dr.status,
            "sessions_processed": sessions_processed,
            "turns_processed": turns_processed,
            "facts_extracted": dr.facts_created,
            "decisions_extracted": dr.decisions_created,
            "questions_raised": dr.questions_created,
            "contradictions_found": dr.contradictions_detected,
            "llm_model": dr.llm_model,
        }
    }

    # ── Markdown body ──────────────────────────────────────────────────────
    lines: List[str] = []
    lines.append("# Dream Report\n")

    # Metadata block
    lines.append(f"**Run ID:** `{dr.run_id}`")
    lines.append(f"**Scope:** `{dr.scope}`")
    lines.append(f"**Started:** {_format_iso(started_at)}")
    lines.append(f"**Ended:** {_format_iso(ended_at)}")
    lines.append(f"**LLM Model:** `{dr.llm_model}`")
    lines.append(f"**Status:** `{dr.status}`")
    lines.append("")

    # Summary counts
    lines.append("## Extraction Summary\n")
    lines.append(f"| Metric | Count |")
    lines.append("|--------|------:{|")
    lines.append(f"| Sessions processed | {sessions_processed} |")
    lines.append(f"| Turns processed | {turns_processed} |")
    lines.append(f"| Facts extracted | {_fmt_count(dr.facts_created)} |")
    lines.append(f"| Decisions extracted | {_fmt_count(dr.decisions_created)} |")
    lines.append(f"| Questions raised | {_fmt_count(dr.questions_created)} |")
    lines.append(f"| Contradictions detected | {_fmt_count(dr.contradictions_detected)} |")
    lines.append("")

    # Sessions
    if result.session_summaries:
        lines.append("## Sessions\n")
        for s in result.session_summaries:
            lines.append(f"### Session: `{s.session_id}`\n")
            lines.append(f"{s.summary}\n")
            lines.append("")

    # Contradictions
    if result.contradictions:
        lines.append("## Contradictions Detected\n")
        for i, c in enumerate(result.contradictions, 1):
            lines.append(f"### Contradiction {i}\n")
            for k, v in c.items():
                lines.append(f"- **{k}:** {v}")
            lines.append("")
    else:
        lines.append("## Contradictions Detected\n")
        lines.append("No contradictions detected.\n")

    # Source refs for key items
    source_refs: List[str] = []
    for fact in result.facts[:5]:
        ref = fact.get("source_refs_json", [])
        if isinstance(ref, list):
            source_refs.extend(ref)
        else:
            source_refs.append(f"dream:{dr.run_id}")
    for dec in result.decisions[:3]:
        ref = dec.get("source_refs_json", [])
        if isinstance(ref, list):
            source_refs.extend(ref)
        else:
            source_refs.append(f"dream:{dr.run_id}")

    if source_refs:
        unique_refs = list(dict.fromkeys(source_refs))
        lines.append("## Source References\n")
        for ref in unique_refs:
            lines.append(f"- `{ref}`")
        lines.append("")

    body = "\n".join(lines)

    # ── Assemble final file ─────────────────────────────────────────────────
    yaml_block = json.dumps(frontmatter, indent=2)
    # Convert JSON dict to YAML-like frontmatter (strip outer braces)
    fm_lines = [f"```yaml"]
    for line in yaml_block.splitlines()[1:-1]:  # skip { and }
        fm_lines.append(line)
    fm_lines.append("```")
    fm_text = "\n".join(fm_lines)

    report_path.write_text(
        f"---\n{fm_text}\n---\n\n{body}",
        encoding="utf-8",
    )

    logger.info("Dream report written to %s", report_path)
    return report_path