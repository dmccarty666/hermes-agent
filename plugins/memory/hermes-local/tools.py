"""memory_write tool for Hermes Local Memory.

Implements the Phase 4 canonical write path for facts/decisions/open_questions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from hermes_memory_core import write_memory, MemoryWriteInput

__all__ = ["memory_write_tool"]

MEMORY_WRITE_SCHEMA = {
    "name": "memory_write",
    "description": (
        "Write a durable memory: fact / decision / open_question. "
        "Source reference required. Redaction always runs unless force_no_redact=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["fact", "decision", "open_question"],
                "description": "Kind of memory to write.",
            },
            "text": {
                "type": "string",
                "description": "The memory text. Will be scanned for secrets before write.",
            },
            "project": {
                "type": "string",
                "description": "Optional project namespace.",
            },
            "scope": {
                "type": "string",
                "enum": ["user", "project", "general"],
                "description": "Scope for facts (default: general).",
            },
            "source_ref": {
                "type": "string",
                "description": (
                    "Reference to the source of this memory. "
                    "Format: session:{id}#turn={id} or similar."
                ),
            },
            "confidence": {
                "type": "number",
                "description": "Optional confidence score 0-1.",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags.",
            },
            "rationale": {
                "type": "string",
                "description": "Rationale for decisions.",
            },
            "owner": {
                "type": "string",
                "description": "Owner for decisions.",
            },
            "priority": {
                "type": "string",
                "description": "Priority for open questions (high/medium/low).",
            },
            "force_no_redact": {
                "type": "boolean",
                "default": False,
                "description": "Skip redaction scan (logs override, use sparingly).",
            },
        },
        "required": ["type", "text", "source_ref"],
    },
}


def memory_write_tool(
    type: str,
    text: str,
    source_ref: str,
    project: Optional[str] = None,
    scope: Optional[str] = None,
    confidence: Optional[float] = None,
    tags: Optional[str] = None,
    rationale: Optional[str] = None,
    owner: Optional[str] = None,
    priority: Optional[str] = None,
    force_no_redact: bool = False,
) -> dict:
    """Write a fact, decision, or open question to Hermes Local Memory.

    Parameters
    ----------
    type : str
        One of "fact", "decision", "open_question".
    text : str
        The memory text.
    source_ref : str
        Resolvable reference to the source content.
    project : str, optional
        Project namespace.
    scope : str, optional
        Scope for facts ("user", "project", "general").
    confidence : float, optional
        Confidence score 0-1.
    tags : str, optional
        Comma-separated tags string.
    rationale : str, optional
        Rationale (decisions only).
    owner : str, optional
        Owner (decisions only).
    priority : str, optional
        Priority (open_questions only).
    force_no_redact : bool
        Skip redaction scan.

    Returns
    -------
    dict
        write_memory result with written=True/False.
    """
    # Parse comma-separated tags
    tag_list: Optional[List[str]] = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    input_obj = MemoryWriteInput(
        memory_type=type,
        text=text,
        source_ref=source_ref,
        project=project,
        scope=scope,
        confidence=confidence,
        tags=tag_list,
        rationale=rationale,
        owner=owner,
        priority=priority,
        force_no_redact=force_no_redact,
    )

    result = write_memory(
        memory_type=type,
        text=text,
        source_ref=source_ref,
        project=project,
        scope=scope or "general",
        confidence=confidence,
        tags=json.dumps(tag_list) if tag_list else None,
        rationale=rationale,
        owner=owner,
        priority=priority,
    )

    # Build human-readable summary
    if result.get("written"):
        record = result.get("record", {})
        memory_id = record.get("fact_id") or record.get("decision_id") or record.get("question_id", "unknown")
        types_redacted = result.get("types_redacted", [])
        if types_redacted:
            summary = f"[REDACTED: {', '.join(types_redacted)}] — written as {type}."
        else:
            summary = f"Written as {type} (ID: {memory_id[:20]}...)."
    else:
        summary = f"Blocked: {result.get('reason', 'unknown')}."

    return {
        "ok": result.get("written", False),
        "summary": summary,
        "memory_type": type,
        "types_redacted": result.get("types_redacted", []),
        "conflicts_checked": result.get("conflicts_checked", False),
        "record": result.get("record", {}),
    }