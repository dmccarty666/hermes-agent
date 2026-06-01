"""
Canonical write pipeline for Hermes Local Memory.

All writes (facts, decisions, open_questions) MUST go through this module.
It handles:
- Redaction (defense-in-depth — already done at capture, but run again here)
- Content hashing (dedup via UNIQUE constraint)
- Routing to the correct SQLite table via MemoryStore
- Audit log for every write event

Exported as memory_core.write.pipeline.write_memory() for use throughout.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_memory_core.write.redaction import redact, hash_content

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

HERMES_HOME = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
PROMPTS_DIR = Path(HERMES_HOME) / "memory" / "prompts"

# ── LLM client ────────────────────────────────────────────────────────────────

DREAM_LLM_URL = "http://192.168.2.105:1234/v1"
DREAM_LLM_MODEL = "qwen3.6-35b-instruct"


def _llm_complete(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    json_mode: bool = True,
) -> str:
    """
    Call the dreamer LLM (Qwen3.6-35B) via LMS.
    Returns the raw response text.
    Raises on connection error or non-200 response.
    """
    import requests

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    prompt_len = len(prompt)
    logger.debug(
        "dream_llm request: model=%s prompt_chars=%d json_mode=%s",
        DREAM_LLM_MODEL,
        prompt_len,
        json_mode,
        extra={"dream_llm": True},
    )

    import requests

    payload: Dict[str, Any] = {
        "model": DREAM_LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["extra_body"] = {"json_mode": True}

    try:
        response = requests.post(
            f"{DREAM_LLM_URL}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120.0,
        )
        if response.status_code != 200:
            error_msg = f"LLM returned {response.status_code}: {response.text[:300]}"
            logger.debug(
                "dream_llm failure: type=HTTP_%s msg=%s",
                response.status_code,
                error_msg[:200],
                extra={"dream_llm": True},
            )
            raise RuntimeError(error_msg)
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        logger.debug(
            "dream_llm success: response_preview=%s",
            content[:200],
            extra={"dream_llm": True},
        )
        return content
    except Exception as exc:
        logger.debug(
            "dream_llm failure: type=%s msg=%s",
            type(exc).__name__,
            str(exc)[:200],
            extra={"dream_llm": True},
        )
        raise


# ── Prompt loading ─────────────────────────────────────────────────────────────

def _load_template(name: str) -> str:
    """Load a prompt template from ~/.hermes/memory/prompts/."""
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text()


def _render_template(template: str, **kwargs: Any) -> str:
    """Simple {{variable}} substitution for prompt templates."""
    result = template
    for key, value in kwargs.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, str(value))
    return result


# ── Fact write ────────────────────────────────────────────────────────────────

def write_memory(
    memory_type: str,          # "fact" | "decision" | "open_question"
    text: str,
    scope: str | None = None,
    project: str | None = None,
    entity: str | None = None,
    source_ref: str | None = None,
    confidence: float | None = None,
    tags: List[str] | None = None,
    rationale: str | None = None,
    owner: str | None = None,
    priority: str | None = None,
    category: str = "general",
    supersedes_fact_id: str | None = None,
    store=None,                # MemoryStore instance; injected for testing
    skip_redaction: bool = False,
) -> Dict[str, Any]:
    """
    Canonical write path for facts, decisions, and open_questions.

    All candidate writes from the dreamer go through here:
    1. Redaction (defense-in-depth)
    2. Content hash (dedup)
    3. Write to SQLite via MemoryStore
    4. Audit log

    Args:
        memory_type: "fact", "decision", or "open_question"
        text: raw text content
        scope: "user", "project", "general" (for facts)
        project: project name if relevant
        entity: entity name if relevant
        source_ref: session-level source reference
        confidence: 0.0-1.0 for facts
        tags: list of tag strings
        rationale: decision rationale
        owner: decision owner
        priority: question priority
        category: fact category
        supersedes_fact_id: link to superseded fact (contradiction case)
        store: MemoryStore instance (uses singleton if None)
        skip_redaction: bypass redaction (use only for already-redacted content)

    Returns:
        dict with keys: written (bool), id (str), type, skipped (bool),
        redaction_types (list), error (str or None)
    """
    # Import here to avoid circular dependency at module level
    if store is None:
        from hermes_memory_core.store.sqlite import get_memory_store
        store = get_memory_store()

    result: Dict[str, Any] = {
        "written": False,
        "id": None,
        "type": memory_type,
        "skipped": False,
        "redaction_types": [],
        "error": None,
    }

    # 1. Redaction (defense-in-depth)
    if skip_redaction:
        redacted_text = text
        redaction_types = []
    else:
        r = redact(text)
        redacted_text = r.redacted_text
        redaction_types = r.types_found

    result["redaction_types"] = redaction_types
    if redaction_types:
        logger.info(
            "Redaction applied to %s write: %s",
            memory_type,
            redaction_types,
        )

    content_hash = hash_content(redacted_text)
    source_refs = [source_ref] if source_ref else []
    source_refs_json = json.dumps(source_refs)
    tags_json = json.dumps(tags or [])

    try:
        if memory_type == "fact":
            fact_id = f"fact:{uuid.uuid4().hex[:16]}"
            fid, created = store.upsert_fact(
                fact_id=fact_id,
                fact_text=redacted_text,
                content_hash=content_hash,
                scope=scope or "general",
                source_refs_json=source_refs_json,
                project=project,
                entity=entity,
                category=category,
                confidence=confidence,
                tags_json=tags_json,
                supersedes_fact_id=supersedes_fact_id,
            )
            result["id"] = fid
            result["written"] = True
            result["skipped"] = not created
            # Audit log
            store.write_audit(
                actor="dreamer",
                action="dream_fact_write",
                target_kind="fact",
                target_id=fid,
                detail={
                    "created": created,
                    "project": project,
                    "redaction_types": redaction_types,
                    "supersedes_fact_id": supersedes_fact_id,
                },
                source_ref=source_ref,
            )
            if not created:
                logger.debug("Fact deduplicated by content_hash: %s", fact_id)

        elif memory_type == "decision":
            decision_id = f"dec:{uuid.uuid4().hex[:16]}"
            did, created = store.upsert_decision(
                decision_id=decision_id,
                decision_text=redacted_text,
                source_refs_json=source_refs_json,
                rationale=rationale,
                project=project,
                owner=owner,
            )
            result["id"] = did
            result["written"] = True
            result["skipped"] = not created
            store.write_audit(
                actor="dreamer",
                action="dream_decision_write",
                target_kind="decision",
                target_id=did,
                detail={
                    "created": created,
                    "project": project,
                    "redaction_types": redaction_types,
                },
                source_ref=source_ref,
            )

        elif memory_type == "open_question":
            question_id = f"q:{uuid.uuid4().hex[:16]}"
            qid, created = store.upsert_open_question(
                question_id=question_id,
                question_text=redacted_text,
                source_refs_json=source_refs_json,
                project=project,
                priority=priority,
            )
            result["id"] = qid
            result["written"] = True
            result["skipped"] = not created
            store.write_audit(
                actor="dreamer",
                action="dream_question_write",
                target_kind="open_question",
                target_id=qid,
                detail={
                    "created": created,
                    "project": project,
                    "redaction_types": redaction_types,
                },
                source_ref=source_ref,
            )
        else:
            result["error"] = f"Unknown memory_type: {memory_type}"

    except Exception as exc:
        logger.exception("write_memory failed for %s: %s", memory_type, exc)
        result["error"] = str(exc)
        result["written"] = False

    return result


def write_audit_log(
    actor: str,
    action: str,
    target_kind: str | None = None,
    target_id: str | None = None,
    detail: Dict[str, Any] | None = None,
    source_ref: str | None = None,
    store=None,
) -> None:
    """Convenience wrapper for audit log writes."""
    if store is None:
        from hermes_memory_core.store.sqlite import get_memory_store
        store = get_memory_store()
    store.write_audit(actor, action, target_kind, target_id, detail, source_ref)
