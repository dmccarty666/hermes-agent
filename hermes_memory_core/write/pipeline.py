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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_memory_core.store import fs as fs_module
from hermes_memory_core.store import sqlite as sqlite_module
from hermes_memory_core.write import redaction

logger = logging.getLogger(__name__)

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
) -> Dict[str, Any]:
    """Canonical write path: redact, append JSONL, insert SQLite, audit.

    Args:
        event: Full turn event dict. Must contain: event_id, session_id,
            turn_id, sequence, timestamp, role, content, agent, source.
            May also contain tool_calls (list) for tool-result scanning.
        skip_redaction: If True, skip the redaction scan. Used only for
            testing or special internal cases.

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

    # ---- Step 2: JSONL append ---------------------------------------------/
    db = _get_memory_db()
    fs = _get_fs_store()

    content_hash = fs.append_event(event)

    # ---- Step 3: SQLite session upsert -----------------------------------/
    now_iso = datetime.now(timezone.utc).isoformat()

    conn = db._connect()
    try:
        conn.execute(
            """INSERT INTO sessions
               (session_id, agent, title, project, started_at, ended_at, source, platform)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO NOTHING
            """,
            (session_id, agent, None, None, timestamp, None, source, None),
        )
    finally:
        conn.close()

    # ---- Step 4: SQLite turns insert (idempotent by turn_id) -------------/
    conn = db._connect()
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
    finally:
        conn.close()

    # ---- Step 5: SQLite raw_events insert --------------------------------/
    conn = db._connect()
    try:
        # Build the full event JSON for raw_events (with redacted content)
        # We store the redacted version in raw_events too
        event_for_raw = dict(event)
        event_for_raw["content"] = redacted_content
        # tool_calls redacted in-place
        if tool_calls:
            event_for_raw["tool_calls"] = tool_calls
        raw_json = json.dumps(event_for_raw, ensure_ascii=False)

        # Get byte offset (approximate — length of current JSONL file)
        raw_dir = fs.base_path / "raw"
        jsonl_path = fs._jsonl_path(session_id)
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
        conn = db._connect()
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
    confidence: float = 0.5,
    tags: Optional[str] = None,
    rationale: Optional[str] = None,
    owner: Optional[str] = None,
    priority: Optional[str] = None,
    skip_redaction: bool = False,
) -> Dict[str, Any]:
    """Write a durable memory item (fact / decision / open_question).

    Phase 1 (T-007): stub — raises NotImplementedError.
    """
    raise NotImplementedError("write_memory() — story T-009")