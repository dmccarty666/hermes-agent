"""memory_write tool for Hermes Local Memory.

Implements the Phase 4 canonical write path for facts/decisions/open_questions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from hermes_memory_core import write_memory, update_memory, fact_feedback, MemoryWriteInput

__all__ = ["memory_write_tool", "memory_update_tool", "fact_feedback_tool"]

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

    Returns
    -------
    dict
        write_memory result with written=True/False.
    """
    # Parse comma-separated tags into a list
    tag_list: Optional[List[str]] = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    result = write_memory(
        memory_type=type,
        text=text,
        source_ref=source_ref,
        project=project,
        scope=scope or "general",
        confidence=confidence,
        tags=tag_list,
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


# -------------------------------------------------------------------------------------
# memory_update tool — Story 4.4.2
# -------------------------------------------------------------------------------------

MEMORY_UPDATE_SCHEMA = {
    "name": "memory_update",
    "description": (
        "Update an existing memory's content / trust / tags / status / category. "
        "Only supplied fields are updated; omitted fields stay unchanged. "
        "Trust (confidence) is stored for facts; decisions and open_questions store status/category/text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory ID to update (fact_*, decision_*, or question_*).",
            },
            "memory_type": {
                "type": "string",
                "enum": ["fact", "decision", "open_question"],
                "description": "The kind of memory being updated.",
            },
            "text": {
                "type": "string",
                "description": "New memory text (content replacement).",
            },
            "trust_delta": {
                "type": "number",
                "description": (
                    "Change to apply to confidence/trust score. "
                    "For facts: applied as absolute value to confidence column (0.0-1.0)."
                ),
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags to set.",
            },
            "status": {
                "type": "string",
                "enum": ["active", "superseded", "disputed", "archived"],
                "description": "New status for the memory.",
            },
            "category": {
                "type": "string",
                "description": "Category label for the memory.",
            },
        },
        "required": ["memory_id", "memory_type"],
    },
}


def memory_update_tool(
    memory_id: str,
    memory_type: str,
    text: Optional[str] = None,
    trust_delta: Optional[float] = None,
    tags: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
) -> dict:
    """Update an existing memory item (fact / decision / open_question).

    Parameters
    ----------
    memory_id : str
        The memory ID to update (fact_*, decision_*, or question_*).
    memory_type : str
        One of "fact", "decision", "open_question".
    text : str, optional
        New memory text.
    trust_delta : float, optional
        Change to apply to confidence/trust score.
    tags : str, optional
        Comma-separated tags string.
    status : str, optional
        One of "active", "superseded", "disputed", "archived".
    category : str, optional
        Category label.

    Returns
    -------
    dict
        update_memory result with updated=True/False.
    """
    tag_list: Optional[List[str]] = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    result = update_memory(
        memory_id=memory_id,
        memory_type=memory_type,
        text=text,
        trust_delta=trust_delta,
        tags=tag_list,
        status=status,
        category=category,
    )

    if result.get("updated"):
        changes = result.get("changes", {})
        summary = f"Updated {memory_type} {memory_id[:20]}... — changed: {', '.join(changes.keys()) or 'none'}."
    else:
        summary = f"Update failed: {result.get('reason', 'unknown')}."

    return {
        "ok": result.get("updated", False),
        "summary": summary,
        "memory_type": memory_type,
        "id": result.get("id"),
        "changes": result.get("changes", {}),
        "reason": result.get("reason"),
    }


# -------------------------------------------------------------------------------------
# fact_feedback tool — Story 4.4.2 (ported from holographic semantics)
# -------------------------------------------------------------------------------------

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a memory after using it. "
        "helpful = confidence += 0.05 (clamped to [0, 1]); "
        "unhelpful = confidence -= 0.10 (clamped to [0, 1])."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The fact ID to rate.",
            },
            "action": {
                "type": "string",
                "enum": ["helpful", "unhelpful"],
                "description": "Feedback action: helpful (+0.05 trust) or unhelpful (-0.10 trust).",
            },
        },
        "required": ["memory_id", "action"],
    },
}


def fact_feedback_tool(
    memory_id: str,
    action: str,
) -> dict:
    """Apply feedback to a fact, adjusting its confidence trust score.

    Ported from holographic semantics:
      - helpful  → confidence += 0.05, clamped to [0.0, 1.0]
      - unhelpful → confidence -= 0.10, clamped to [0.0, 1.0]

    Parameters
    ----------
    memory_id : str
        The fact ID to rate.
    action : str
        Either "helpful" or "unhelpful".

    Returns
    -------
    dict
        fact_feedback result with ok=True/False and confidence values.
    """
    result = fact_feedback(memory_id=memory_id, action=action)

    if result.get("ok"):
        old = result.get("old_confidence")
        new = result.get("new_confidence")
        delta = 0.05 if action == "helpful" else -0.10
        summary = (
            f"Feedback applied to {memory_id[:20]}...: "
            f"{action} (delta={delta:+.2f}), "
            f"confidence {old:.2f} → {new:.2f}."
        )
    else:
        summary = f"Feedback failed: {result.get('reason', 'unknown')}."

    return {
        "ok": result.get("ok", False),
        "summary": summary,
        "id": result.get("id"),
        "old_confidence": result.get("old_confidence"),
        "new_confidence": result.get("new_confidence"),
        "action": result.get("action"),
        "reason": result.get("reason"),
    }