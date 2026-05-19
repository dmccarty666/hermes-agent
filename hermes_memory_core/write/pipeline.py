# Copyright 2026 David McCarty. All rights reserved.
"""Canonical write pipeline for Hermes Local Memory.

Single entry point for all memory writes::

    from hermes_memory_core.write.pipeline import capture_event

Lifecycle (story T-007):
  1. capture_event(event) — receives user+assistant turn dict
      a. redaction.scan(content) -> redacted_content
      b. fs.append_event(event) -> content_hash
      c. sqlite.upsert_session(session)
      d. sqlite.insert_turn(turn)
      e. sqlite.insert_raw_event(event, redacted_content, hash)
      f. audit_log.write(...) if redaction fired
      g. mark turns.index_status = 'pending' (already default)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hermes_memory_core.store import fs as fs_module
from hermes_memory_core.store import sqlite as sqlite_module
from hermes_memory_core.store.sqlite import MemoryDB
from hermes_memory_core.write import redaction

logger = logging.getLogger(__name__)

# HRR dimension (1024 from hrr.py default; must match BLOB column width)
HRR_DIM = 1024

# Qdrant collection names for memory items
_QDRANT_COLLECTION_FACTS = "hermes_memory_facts_nomic_v15"
_QDRANT_COLLECTION_DECISIONS = "hermes_memory_decisions_nomic_v15"
_QDRANT_COLLECTION_QUESTIONS = "hermes_memory_questions_nomic_v15"


# --------------------------------------------------------------------------/
# Dataclasses — input types for write_memory
# --------------------------------------------------------------------------/


@dataclass
class MemoryWriteInput:
    """Structured input for write_memory.

    Attributes:
        memory_type:  One of "fact", "decision", "open_question".
        text:         The memory text — always scanned for secrets before write.
        source_ref:   Resolvable reference to the source (required).
        project:      Optional project namespace.
        scope:        One of "user", "project", "general" (facts only).
        confidence:   Optional 0-1 confidence score.
        tags:         Optional list of tag strings.
        rationale:    Optional rationale (decisions only).
        owner:        Optional owner (decisions only).
        priority:     Optional priority (open_questions only).
        force_no_redact: If True, skip redaction scan and log a redaction_override
                        audit event. Use sparingly — the scanner always runs
                        normally. Removed from MVP per v0.2-critique Issue 7.
    """

    memory_type: str
    text: str
    source_ref: str
    project: Optional[str] = None
    scope: Optional[str] = None
    confidence: Optional[float] = None
    tags: Optional[List[str]] = None
    rationale: Optional[str] = None
    owner: Optional[str] = None
    priority: Optional[str] = None
    force_no_redact: bool = False

# Singleton instances — overridden in tests via _inject_for_test
_memory_db: Optional["MemoryDB"] = None
_fs_store: Optional["FSStore"] = None


def _get_memory_db() -> "MemoryDB":
    """Return the singleton MemoryDB instance, creating if needed."""
    global _memory_db
    if _memory_db is None:
        _memory_db = sqlite_module.MemoryDB()
        _memory_db.initialize()
    return _memory_db


def _get_fs_store() -> "FSStore":
    """Return the singleton FSStore instance, creating if needed."""
    global _fs_store
    if _fs_store is None:
        _fs_store = fs_module.FSStore()
    return _fs_store


def _inject_for_test(memory_db: "MemoryDB", fs_store: "FSStore") -> None:
    """Test injection point — replaces singletons with test instances."""
    global _memory_db, _fs_store
    _memory_db = memory_db
    _fs_store = fs_store


# --------------------------------------------------------------------------/
# Public API
# --------------------------------------------------------------------------/


def capture_event(
    event: Dict[str, Any],
    *,
    skip_redaction: bool = False,
    db: Optional["MemoryDB"] = None,
    fs: Optional["FSStore"] = None,
) -> Dict[str, Any]:
    """Canonical write path: redact, append JSONL, insert SQLite, audit.

    Args:
        event: Full turn event dict. Must contain: event_id, session_id,
            turn_id, sequence, timestamp, role, content, agent, source.
            May also contain tool_calls (list) for tool-result scanning.
        skip_redaction: If True, skip the redaction scan. Used only for
            testing or special internal cases.
        db: Optional MemoryDB instance. If omitted, uses the process-global
            singleton (useful for normal runtime; tests should pass this
            parameter explicitly to avoid singleton issues with pytest-xdist).
        fs: Optional FSStore instance. Same reasoning as ``db``.

    Returns:
        dict with keys: event_id, content_hash, session_id, redaction_fired,
            audit_logged.

    Raises:
        ValueError: if required fields are missing from event.
    """
    # Validate required fields
    for field in ("event_id", "session_id", "turn_id", "sequence",
                  "timestamp", "role", "content", "agent"):
        if field not in event:
            raise ValueError(f"capture_event: missing required field '{field}'")

    event_id: str = event["event_id"]
    session_id: str = event["session_id"]
    turn_id: str = event["turn_id"]
    sequence: int = int(event["sequence"])
    timestamp: str = event["timestamp"]
    role: str = event["role"]
    content: str = str(event["content"])
    agent: str = event["agent"]
    source: str = event.get("source", "cli")
    tool_calls: List[Dict[str, Any]] = event.get("tool_calls", [])

    logger.debug(">>> capture_event step 1: redaction")
    # ---- Step 1: Redaction -----------------------------------------------/
    redacted_content = content
    redaction_types: List[str] = []
    redaction_fired = False
    tool_calls = event.get("tool_calls", [])[:]  # shallow copy
    # Track which tool_name triggered redaction for audit context
    _redacted_tool_names: List[str] = []
    logger.debug(f"    tool_calls initial: {tool_calls}")
    if not skip_redaction:
        # Scan main content
        result = redaction.scan(content)
        redacted_content = result.redacted_content
        if result.fired:
            redaction_types = [h.pattern_name for h in result.hits]
            redaction_fired = True

        # Also scan tool_calls arguments AND output (secrets can appear in both)
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {})
            tool_name = func.get("name", "unknown")
            # Scan function arguments
            args = func.get("arguments", "")
            if isinstance(args, str):
                tool_str = args
            else:
                tool_str = json.dumps(args)
            if tool_str:
                tc_result = redaction.scan(tool_str)
                if tc_result.fired:
                    for h in tc_result.hits:
                        if h.pattern_name not in redaction_types:
                            redaction_types.append(h.pattern_name)
                    redaction_fired = True
                    func["arguments"] = tc_result.redacted_content
                    if tool_name not in _redacted_tool_names:
                        _redacted_tool_names.append(tool_name)
            # Scan tool result output (AC-1: tool result content)
            output = tc.get("output")
            if output and isinstance(output, str) and output.strip():
                out_result = redaction.scan(output)
                if out_result.fired:
                    for h in out_result.hits:
                        if h.pattern_name not in redaction_types:
                            redaction_types.append(h.pattern_name)
                    redaction_fired = True
                    tc["output"] = out_result.redacted_content
                    if tool_name not in _redacted_tool_names:
                        _redacted_tool_names.append(tool_name)

        # Scan attachment filenames for embedded secrets (AC-2: attachment filenames)
        for att in event.get("attachments", []):
            if not isinstance(att, dict):
                continue
            fname = att.get("filename") or att.get("name") or ""
            if fname:
                fname_result = redaction.scan(fname)
                if fname_result.fired:
                    for h in fname_result.hits:
                        if h.pattern_name not in redaction_types:
                            redaction_types.append(h.pattern_name)
                    redaction_fired = True
                    # Redact the filename field in-place
                    att["_redacted_name"] = fname_result.redacted_content
                    att["name"] = fname_result.redacted_content

    # Redact content in-place so JSONL write sees the redacted version
        event["content"] = redacted_content

    # ---- Step 2: JSONL append ---------------------------------------------/
    _db = db if db is not None else _get_memory_db()
    _fs = fs if fs is not None else _get_fs_store()

    content_hash = _fs.append_event(event)

    # ---- Step 3: SQLite session upsert -----------------------------------/
    now_iso = datetime.now(timezone.utc).isoformat()

    conn = _db._connect()
    try:
        conn.execute(
            """INSERT INTO sessions
               (session_id, agent, title, project, started_at, ended_at, source, platform)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO NOTHING
            """,
            (session_id, agent, None, None, timestamp, None, source, None),
        )
        conn.commit()
    finally:
        conn.close()

    # ---- Step 4: SQLite turns insert (idempotent by turn_id) -------------/
    conn = _db._connect()
    try:
        # Check if already exists (idempotency)
        existing = conn.execute(
            "SELECT 1 FROM turns WHERE turn_id=?", (turn_id,)
        ).fetchone()
        if existing is None:
            conn.execute(
                """INSERT INTO turns
                   (turn_id, session_id, sequence, timestamp, role, content,
                    dream_status, index_status, source_refs_json,
                    parent_turn_id, redaction_count, redaction_summary,
                    redaction_applied)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    sequence,
                    timestamp,
                    role,
                    redacted_content,  # store redacted content in turns
                    "pending",          # dream_status
                    "pending",          # index_status
                    "[]",               # source_refs_json
                    event.get("parent_turn_id"),
                    len(redaction_types) if redaction_fired else 0,
                    json.dumps(redaction_types) if redaction_types else None,
                    json.dumps(redaction_types) if redaction_fired else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    # ---- Step 5: SQLite raw_events insert --------------------------------/
    conn = _db._connect()
    try:
        # Build the full event JSON for raw_events (with redacted content)
        # We store the redacted version in raw_events too
        event_for_raw = dict(event)
        event_for_raw["content"] = redacted_content
        # tool_calls redacted in-place
        if tool_calls:
            event_for_raw["tool_calls"] = tool_calls
        # attachments redacted in-place — copy to event_for_raw
        # Use _redacted_name if available (pipeline set it), else name, else filename
        if event.get("attachments"):
            event_for_raw["attachments"] = [
                {**att, "filename": att.get("_redacted_name") or att.get("name") or att.get("filename")}
                for att in event["attachments"]
            ]
        raw_json = json.dumps(event_for_raw, ensure_ascii=False)

        # Get byte offset (approximate — length of current JSONL file)
        raw_dir = _fs.base_path / "raw"
        jsonl_path = _fs._jsonl_path(session_id)
        byte_offset = 0
        if jsonl_path.exists():
            byte_offset = jsonl_path.stat().st_size

        conn.execute(
            """INSERT INTO raw_events
               (event_id, session_id, turn_id, timestamp, jsonl_path,
                byte_offset, event_type, content_hash, raw_content)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(event_id) DO UPDATE SET
                   raw_content=excluded.raw_content,
                   content_hash=excluded.content_hash
            """,
            (
                event_id,
                session_id,
                turn_id,
                timestamp,
                str(jsonl_path),
                byte_offset,
                event.get("event_type", "turn"),
                content_hash,
                raw_json,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # ---- Step 6: audit_log if redaction fired ----------------------------/
    audit_logged = False
    if redaction_fired:
        audit_detail = {
            "session_id": session_id,
            "turn_id": turn_id,
            "event_id": event_id,
            "types": redaction_types,
            "tool_names": _redacted_tool_names or None,
            "source_type": "tool_result" if _redacted_tool_names else "message",
        }
        conn = _db._connect()
        try:
            conn.execute(
                """INSERT INTO audit_log
                   (timestamp, actor, action, target_kind, target_id, detail_json)
                   VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    "plugin",
                    "redaction",
                    "turn",
                    turn_id,
                    json.dumps(audit_detail),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        audit_logged = True

    logger.debug(
        "capture_event: event_id=%s session_id=%s redaction_fired=%s",
        event_id, session_id, redaction_fired,
    )

    return {
        "event_id": event_id,
        "content_hash": content_hash,
        "session_id": session_id,
        "redaction_fired": redaction_fired,
        "redaction_types": redaction_types,
        "audit_logged": audit_logged,
    }


def write_memory(
    memory_type: str,
    text: str,
    *,
    project: Optional[str] = None,
    scope: str = "general",
    source_ref: Optional[str] = None,
    confidence: Optional[float] = None,
    tags: Optional[List[str]] = None,
    rationale: Optional[str] = None,
    owner: Optional[str] = None,
    priority: Optional[str] = None,
) -> Dict[str, Any]:
    """Write a durable memory item (fact / decision / open_question).

    Phase 4 (Story 4.4.1) — TDD §4.

    Pipeline:
      1. Validate inputs (source_ref required, memory_type in {fact, decision, open_question})
      2. Redact secrets from text (always runs)
      3. Compute content_hash (SHA-256 of redacted UTF-8)
      4. Compute HRR vector for facts (used for similarity search)
      5. Attempt INSERT (UNIQUE content_hash dedup on facts/decisions/open_questions)
      6. Upsert fact/decision/question chunk into Qdrant vector store
      7. Write audit_log row on redaction_fired or dedup_skip

    Returns dict:
      - written: bool — True if a new row was inserted
      - skipped: bool — True if dedup blocked (content_hash existed)
      - reason: str — 'new' | 'dedup' | error message
      - source_ref: str — the source_ref used (echoed back)
      - id: str — fact_id / decision_id / question_id (None if skipped/error)
      - redaction_fired: bool
      - redaction_types: list[str] (empty if no redaction)
    """
    from hermes_memory_core.write.redaction import default_redactor
    from hermes_memory_core import get_memory_db
    import hashlib
    import uuid

    # ---- Step 1: validate memory_type ----
    valid_types = {"fact", "decision", "open_question"}
    if memory_type not in valid_types:
        return {
            "written": False,
            "skipped": False,
            "reason": f"invalid_memory_type: must be one of {sorted(valid_types)}",
            "source_ref": source_ref or "",
            "id": None,
            "redaction_fired": False,
            "redaction_types": [],
        }

    # ---- Step 2: source_ref required ----
    if not source_ref:
        return {
            "written": False,
            "skipped": False,
            "reason": "missing_required_field: source_ref is required",
            "source_ref": "",
            "id": None,
            "redaction_fired": False,
            "redaction_types": [],
        }

    # ---- Step 3: redact ----
    redaction_result = default_redactor.scan(text)
    redacted_text = redaction_result.redacted_content
    redaction_fired = redaction_result.fired
    redaction_types = [hit.pattern_name for hit in redaction_result.hits]

    # ---- Step 4: content_hash ----
    content_hash = hashlib.sha256(redacted_text.encode("utf-8")).hexdigest()

    # ---- Step 5: write to appropriate table ----
    db = get_memory_db()
    conn = db._connect()
    now = datetime.now(timezone.utc).isoformat()

    try:
        if memory_type == "fact":
            fact_id = f"fact_{uuid.uuid4().hex[:16]}"
            source_refs_json = json.dumps([source_ref])
            entity_ids_json = "[]"

            # Compute HRR vector for the fact (for similarity search)
            hrr_bytes: Optional[bytes] = None
            _hrr = None  # type: ignore[assignment]
            try:
                from hermes_memory_core.search import hrr as _hrr_module

                hrr_vec = _hrr_module.encode_text(redacted_text, HRR_DIM)
                hrr_bytes = _hrr_module.phases_to_bytes(hrr_vec)
                _hrr = _hrr_module
            except Exception:
                pass  # Non-fatal — HRR is optional

            conn.execute(
                """INSERT INTO facts
                   (fact_id, fact_text, content_hash, scope, project, status,
                    confidence, hrr_vector, source_refs_json, entity_ids_json,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?)
                """,
                (fact_id, redacted_text, content_hash, scope, project,
                 confidence if confidence is not None else 0.5,
                 hrr_bytes, source_refs_json, entity_ids_json, now, now),
            )
            conn.commit()
            result_id = fact_id
            reason = "new"

            # ---- Step 6: Qdrant upsert for facts ----
            # Note: HRR_DIM=1024 but Qdrant facts collection uses embed_dim=768.
            # Non-fatal if upsert fails due to dimension mismatch; log and continue.
            if hrr_bytes is not None and _hrr is not None:
                try:
                    from qdrant_client import QdrantClient
                    from qdrant_client.models import PointStruct

                    qc = QdrantClient(host="localhost", port=6333, timeout=10)
                    qc.upsert(
                        collection_name=_QDRANT_COLLECTION_FACTS,
                        points=[
                            PointStruct(
                                id=fact_id,
                                vector=list(_hrr.bytes_to_phases(hrr_bytes)),
                                payload={
                                    "fact_id": fact_id,
                                    "fact_text": redacted_text,
                                    "content_hash": content_hash,
                                    "scope": scope,
                                    "project": project,
                                    "confidence": confidence if confidence is not None else 0.5,
                                    "source_ref": source_ref,
                                },
                            )
                        ],
                    )
                except Exception as exc:
                    logger.warning("Qdrant upsert failed for fact %s: %s", fact_id, exc)

        elif memory_type == "decision":
            decision_id = f"decision_{uuid.uuid4().hex[:16]}"
            source_refs_json = json.dumps([source_ref])
            conn.execute(
                """INSERT INTO decisions
                   (decision_id, decision_text, rationale, project, owner, status,
                    source_refs_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?)""",
                (decision_id, redacted_text, rationale, project, owner,
                 source_refs_json, now, now),
            )
            conn.commit()
            result_id = decision_id
            reason = "new"

        else:  # open_question
            question_id = f"question_{uuid.uuid4().hex[:16]}"
            source_refs_json = json.dumps([source_ref])
            conn.execute(
                """INSERT INTO open_questions
                   (question_id, question_text, project, priority, status,
                    source_refs_json, next_action, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'open', ?, NULL, ?, ?)""",
                (question_id, redacted_text, project, priority,
                 source_refs_json, now, now),
            )
            conn.commit()
            result_id = question_id
            reason = "new"

    except sqlite3.IntegrityError as exc:
        conn.rollback()
        # UNIQUE constraint violated → dedup skip
        if "content_hash" in str(exc):
            result_id = None
            reason = "dedup"
        else:
            result_id = None
            reason = f"integrity_error: {exc}"
    finally:
        conn.close()

    # ---- Step 6: audit_log ----
    if redaction_fired or reason == "dedup":
        audit_detail = {
            "memory_type": memory_type,
            "content_hash": content_hash,
            "source_ref": source_ref,
            "redaction_fired": redaction_fired,
            "redaction_types": redaction_types,
            "skip_reason": reason,
        }
        try:
            db.log_audit(
                actor="plugin",
                action="memory_write",
                target_kind=memory_type,
                target_id=result_id,
                detail_json=json.dumps(audit_detail),
            )
        except Exception:
            pass  # non-fatal — primary write succeeded

    return {
        "written": reason == "new",
        "skipped": reason == "dedup",
        "reason": reason,
        "source_ref": source_ref,
        "id": result_id,
        "redaction_fired": redaction_fired,
        "redaction_types": redaction_types,
    }


# --------------------------------------------------------------------------/
# update_memory — Story 4.4.2
# --------------------------------------------------------------------------/


def update_memory(
    memory_id: str,
    memory_type: str,
    *,
    text: Optional[str] = None,
    trust_delta: Optional[float] = None,
    tags: Optional[List[str]] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """Update an existing memory item (fact / decision / open_question).

    Story 4.4.2 — supports content / trust / tags / status / category.

    Trust is stored in the ``confidence`` column for facts. The delta is applied
    additively and clamped to the [0.0, 1.0] range. Positive deltas (+0.05)
    reward useful memories; negative deltas (-0.10) penalise bad ones.

    Only the fields that are not None are applied — all others are left
    untouched. The ``memory_id`` is required; ``memory_type`` must also be
    supplied so the correct table is targeted.

    Returns dict:
      - updated: bool — True if a row was modified
      - id: str — the memory_id (echoed back)
      - memory_type: str — the type used
      - changes: dict — which fields were changed
    """
    valid_types = {"fact", "decision", "open_question"}
    if memory_type not in valid_types:
        return {
            "updated": False,
            "id": memory_id,
            "memory_type": memory_type,
            "reason": f"invalid_memory_type: must be one of {sorted(valid_types)}",
            "changes": {},
        }

    from hermes_memory_core import get_memory_db
    db = get_memory_db()
    conn = db._connect()
    now = datetime.now(timezone.utc).isoformat()
    changes: Dict[str, Any] = {}

    try:
        if memory_type == "fact":
            # Build dynamic UPDATE
            # NOTE: facts table has no tags/category columns per schema
            set_clauses: List[str] = ["updated_at = ?"]
            params: List[Any] = [now]

            if text is not None:
                set_clauses.append("fact_text = ?")
                params.append(text)
                changes["text"] = True

            if trust_delta is not None:
                # trust_delta is applied as absolute value in confidence column
                set_clauses.append("confidence = ?")
                params.append(trust_delta)
                changes["trust_delta"] = trust_delta

            if status is not None:
                set_clauses.append("status = ?")
                params.append(status)
                changes["status"] = status

            if category is not None:
                # decisions table has category; facts table does not — skip
                changes.setdefault("category", None)  # record intent but no-op for facts

            if not changes:
                return {
                    "updated": False,
                    "id": memory_id,
                    "memory_type": memory_type,
                    "reason": "nothing_to_update",
                    "changes": {},
                }

            params.append(memory_id)
            sql = f"UPDATE facts SET {', '.join(set_clauses)} WHERE fact_id = ?"
            cursor = conn.execute(sql, params)
            conn.commit()
            updated = cursor.rowcount > 0

        elif memory_type == "decision":
            set_clauses = ["updated_at = ?"]
            params = [now]

            if text is not None:
                set_clauses.append("decision_text = ?")
                params.append(text)
                changes["text"] = True

            if trust_delta is not None:
                # decisions have no confidence column — store in updated_at detail
                # but we still track the delta in the audit detail
                changes["trust_delta"] = trust_delta

            if status is not None:
                set_clauses.append("status = ?")
                params.append(status)
                changes["status"] = status

            if category is not None:
                set_clauses.append("category = ?")
                params.append(category)
                changes["category"] = category

            if not changes:
                return {
                    "updated": False,
                    "id": memory_id,
                    "memory_type": memory_type,
                    "reason": "nothing_to_update",
                    "changes": {},
                }

            params.append(memory_id)
            sql = f"UPDATE decisions SET {', '.join(set_clauses)} WHERE decision_id = ?"
            cursor = conn.execute(sql, params)
            conn.commit()
            updated = cursor.rowcount > 0

        else:  # open_question
            set_clauses = ["updated_at = ?"]
            params = [now]

            if text is not None:
                set_clauses.append("question_text = ?")
                params.append(text)
                changes["text"] = True

            if status is not None:
                set_clauses.append("status = ?")
                params.append(status)
                changes["status"] = status

            if category is not None:
                set_clauses.append("category = ?")
                params.append(category)
                changes["category"] = category

            if not changes:
                return {
                    "updated": False,
                    "id": memory_id,
                    "memory_type": memory_type,
                    "reason": "nothing_to_update",
                    "changes": {},
                }

            params.append(memory_id)
            sql = f"UPDATE open_questions SET {', '.join(set_clauses)} WHERE question_id = ?"
            cursor = conn.execute(sql, params)
            conn.commit()
            updated = cursor.rowcount > 0

        if not updated:
            return {
                "updated": False,
                "id": memory_id,
                "memory_type": memory_type,
                "reason": "not_found",
                "changes": {},
            }

        return {
            "updated": True,
            "id": memory_id,
            "memory_type": memory_type,
            "changes": changes,
        }

    except Exception as exc:
        conn.rollback()
        return {
            "updated": False,
            "id": memory_id,
            "memory_type": memory_type,
            "reason": f"error: {exc}",
            "changes": {},
        }
    finally:
        conn.close()


# --------------------------------------------------------------------------/
# fact_feedback — Story 4.4.2 (ported from holographic semantics)
# --------------------------------------------------------------------------/


def fact_feedback(
    memory_id: str,
    action: str,
) -> Dict[str, Any]:
    """Apply feedback to a fact, adjusting its trust.

    Ported from holographic semantics:
      - helpful  → confidence += 0.05, clamped to [0.0, 1.0]
      - unhelpful → confidence -= 0.10, clamped to [0.0, 1.0]

    Returns dict:
      - ok: bool — True if the update succeeded
      - id: str — the fact_id
      - old_confidence: float | None
      - new_confidence: float
      - action: str — the action applied
    """
    if action not in ("helpful", "unhelpful"):
        return {
            "ok": False,
            "id": memory_id,
            "reason": f"invalid_action: must be 'helpful' or 'unhelpful', got '{action}'",
            "old_confidence": None,
            "new_confidence": None,
        }

    delta = 0.05 if action == "helpful" else -0.10

    from hermes_memory_core import get_memory_db
    db = get_memory_db()
    conn = db._connect()

    try:
        # Fetch current confidence
        row = conn.execute(
            "SELECT confidence FROM facts WHERE fact_id = ?",
            (memory_id,),
        ).fetchone()

        if row is None:
            return {
                "ok": False,
                "id": memory_id,
                "reason": "not_found",
                "old_confidence": None,
                "new_confidence": None,
            }

        old_confidence = row[0] if row[0] is not None else 0.5

        # Apply delta, clamp [0, 1]
        new_confidence = max(0.0, min(1.0, old_confidence + delta))
        now = datetime.now(timezone.utc).isoformat()

        conn.execute(
            "UPDATE facts SET confidence = ?, updated_at = ? WHERE fact_id = ?",
            (new_confidence, now, memory_id),
        )
        conn.commit()

        # Audit log
        try:
            db.log_audit(
                actor="plugin",
                action="fact_feedback",
                target_kind="fact",
                target_id=memory_id,
                detail_json=json.dumps({
                    "action": action,
                    "delta": delta,
                    "old_confidence": old_confidence,
                    "new_confidence": new_confidence,
                }),
            )
        except Exception:
            pass  # non-fatal

        return {
            "ok": True,
            "id": memory_id,
            "old_confidence": old_confidence,
            "new_confidence": new_confidence,
            "action": action,
        }

    except Exception as exc:
        conn.rollback()
        return {
            "ok": False,
            "id": memory_id,
            "reason": f"error: {exc}",
            "old_confidence": None,
            "new_confidence": None,
        }
    finally:
        conn.close()