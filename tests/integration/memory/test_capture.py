"""Test capture_event pipeline (Story T-010 — tool result & attachment scanning).

AC-1: capture_event scans tool result content (not just arguments)
AC-2: capture_event scans attachment filenames
AC-3: audit_log records which tool_name triggered redaction
AC-4: duplicate event_id events are idempotent
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

from hermes_memory_core.store import fs as fs_module
from hermes_memory_core.store import sqlite as sqlite_module
from hermes_memory_core.write.pipeline import capture_event


# --------------------------------------------------------------------------
# Fixtures — fake secrets (clearly fake, no risk)
# Use properly-formatted keys that actually match the redaction patterns:
#   openai_key: sk- + 20+ chars
#   aws_access_key: AKIA + 16 uppercase alphanumeric chars
#   github_token: ghp_ + 36 chars
# --------------------------------------------------------------------------
FAKE = {
    "openai_key": "sk-test123456789012345678901234",
    "aws_access_key": "AKIAFAKEFAKEFAKEFAKE",
    "github_token": "ghp_fake123456789012345678901234567890ab",
    "card_visa": "4532015112830368",
    "ssn": "123-45-6789",
}


def _make_event(
    *,
    event_id: str = "evt_001",
    session_id: str = "sess_001",
    turn_id: str = "turn_001",
    sequence: int = 1,
    role: str = "user",
    content: str = "hello world",
    tool_calls: Optional[list] = None,
    attachments: Optional[list] = None,
    **extra,
):
    """Build a minimal valid event dict."""
    return {
        "event_id": event_id,
        "session_id": session_id,
        "turn_id": turn_id,
        "sequence": sequence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "content": content,
        "agent": "test",
        "source": "test",
        "event_type": "turn",
        **(dict(tool_calls=tool_calls) if tool_calls is not None else {}),
        **(dict(attachments=attachments) if attachments is not None else {}),
        **extra,
    }


@pytest.fixture
def memory_db(tmp_path):
    """Isolated MemoryDB pointing at a tmp SQLite file."""
    db = sqlite_module.MemoryDB(db_path=str(tmp_path / "memory.sqlite"))
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def fs_store(tmp_path):
    """FSStore pointing at a tmp events/ dir."""
    store = fs_module.FSStore()
    store.base_path = tmp_path / "events"
    store.base_path.mkdir(parents=True, exist_ok=True)
    yield store


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset module-level singletons before and after each test."""
    import hermes_memory_core.write.pipeline as p
    original_db = p._memory_db
    original_fs = p._fs_store
    p._memory_db = None
    p._fs_store = None
    yield
    p._memory_db = original_db
    p._fs_store = original_fs


# --------------------------------------------------------------------------
# AC-1: Tool result content is scanned and redacted
# --------------------------------------------------------------------------
class TestToolResultScanning:
    def test_tool_result_with_secret_is_redacted(self, memory_db, fs_store):
        """When tool output contains a secret, it must be redacted."""
        event = _make_event(
            content="running the model",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "create_completion",
                        "arguments": '{"model":"gpt-4","prompt":"hi"}',
                    },
                    "output": json.dumps({"result": "success", "api_key": FAKE["openai_key"]}),
                }
            ],
        )
        result = capture_event(event, db=memory_db, fs=fs_store)
        assert result["redaction_fired"] is True
        # Tool result is stored in raw_events (redacted), not in turns.content (message only)
        conn = memory_db._connect()
        try:
            raw_row = conn.execute(
                "SELECT raw_content FROM raw_events WHERE turn_id=?", ("turn_001",)
            ).fetchone()
            assert raw_row is not None
            assert FAKE["openai_key"] not in raw_row[0]
            assert "[REDACTED:openai_key]" in raw_row[0]
        finally:
            conn.close()

    def test_tool_result_secret_not_in_jsonl(self, memory_db, fs_store):
        """Raw secret must not appear in raw_events JSONL."""
        event = _make_event(
            content="running",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "run_sql",
                        "arguments": "{}",
                    },
                    "output": json.dumps({"rows": [], "token": FAKE["github_token"]}),
                }
            ],
        )
        capture_event(event, db=memory_db, fs=fs_store)
        jsonl_path = fs_store._jsonl_path("sess_001")
        raw = jsonl_path.read_text()
        assert FAKE["github_token"] not in raw
        assert "[REDACTED:github_token]" in raw

    def test_tool_result_both_args_and_output_redacted(self, memory_db, fs_store):
        """When both function args AND output contain secrets, both are redacted."""
        event = _make_event(
            content="call",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "aws_api",
                        "arguments": json.dumps({"key": FAKE["aws_access_key"]}),
                    },
                    "output": json.dumps({"status": "ok", "secret": FAKE["openai_key"]}),
                }
            ],
        )
        result = capture_event(event, db=memory_db, fs=fs_store)
        assert result["redaction_fired"] is True
        assert "aws_access_key" in result["redaction_types"]

    def test_tool_result_output_only_secret(self, memory_db, fs_store):
        """When ONLY output contains a secret (args are clean), output is still redacted."""
        event = _make_event(
            content="result",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {"name": "get_secret", "arguments": "{}"},
                    "output": json.dumps({"value": FAKE["openai_key"]}),
                }
            ],
        )
        result = capture_event(event, db=memory_db, fs=fs_store)
        assert result["redaction_fired"] is True
        # Tool result lives in raw_events (redacted), not turns.content
        conn = memory_db._connect()
        try:
            raw_row = conn.execute(
                "SELECT raw_content FROM raw_events WHERE turn_id=?", ("turn_001",)
            ).fetchone()
            assert raw_row is not None
            assert FAKE["openai_key"] not in raw_row[0]
        finally:
            conn.close()

    def test_tool_result_with_nested_json(self, memory_db, fs_store):
        """Tool result can be a JSON string with nested secrets."""
        event = _make_event(
            content="nested",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {"name": "bulk_export", "arguments": "{}"},
                    "output": json.dumps({
                        "records": [
                            {"name": "Alice", "ssn": FAKE["ssn"]},
                        ]
                    }),
                }
            ],
        )
        result = capture_event(event, db=memory_db, fs=fs_store)
        assert result["redaction_fired"] is True
        # Tool result (with nested secret) stored in raw_events
        conn = memory_db._connect()
        try:
            raw_row = conn.execute(
                "SELECT raw_content FROM raw_events WHERE turn_id=?", ("turn_001",)
            ).fetchone()
            assert raw_row is not None
            assert FAKE["ssn"] not in raw_row[0]
            assert "[REDACTED:ssn]" in raw_row[0]
        finally:
            conn.close()


# --------------------------------------------------------------------------
# AC-2: Attachment filenames are scanned
# --------------------------------------------------------------------------
class TestAttachmentFilenameScanning:
    def test_attachment_with_secret_in_filename(self, memory_db, fs_store):
        """Attachment filename containing a secret is redacted."""
        # Use /tmp/ prefix so sk- is preceded by / (non-word char → word boundary ✓)
        event = _make_event(
            content="uploading",
            attachments=[
                {"filename": f"/tmp/upload_{FAKE['openai_key']}.json", "url": "file:///tmp/x"},
            ],
        )
        result = capture_event(event, db=memory_db, fs=fs_store)
        assert result["redaction_fired"] is True
        # Attachment filename stored in raw_events (redacted via att["name"])
        conn = memory_db._connect()
        try:
            raw_row = conn.execute(
                "SELECT raw_content FROM raw_events WHERE turn_id=?", ("turn_001",)
            ).fetchone()
            assert raw_row is not None
            assert FAKE["openai_key"] not in raw_row[0]
            assert "[REDACTED:openai_key]" in raw_row[0]
        finally:
            conn.close()

    def test_attachment_name_field(self, memory_db, fs_store):
        """Attachments may use 'name' instead of 'filename'."""
        event = _make_event(
            content="uploading",
            attachments=[
                {"name": f"report_{FAKE['github_token']}.pdf", "url": "file:///tmp/x"},
            ],
        )
        result = capture_event(event, db=memory_db, fs=fs_store)
        assert result["redaction_fired"] is True
        conn = memory_db._connect()
        try:
            raw_row = conn.execute(
                "SELECT raw_content FROM raw_events WHERE turn_id=?", ("turn_001",)
            ).fetchone()
            assert raw_row is not None
            assert FAKE["github_token"] not in raw_row[0]
        finally:
            conn.close()

    def test_attachment_without_name_field_ignored(self, memory_db, fs_store):
        """Attachments with no name/filename field don't error."""
        event = _make_event(
            content="uploading",
            attachments=[{"url": "https://example.com/public.zip"}],
        )
        result = capture_event(event, db=memory_db, fs=fs_store)
        assert result["redaction_fired"] is False

    def test_multiple_attachments_all_scanned(self, memory_db, fs_store):
        """All attachments in the event are scanned."""
        event = _make_event(
            content="batch upload",
            attachments=[
                # /tmp/ prefix ensures sk- is preceded by non-word char (word boundary ✓)
                {"filename": f"/tmp/key_{FAKE['openai_key']}.txt"},
                {"filename": f"/tmp/token_{FAKE['github_token']}.txt"},
            ],
        )
        result = capture_event(event, db=memory_db, fs=fs_store)
        assert result["redaction_fired"] is True
        conn = memory_db._connect()
        try:
            raw_row = conn.execute(
                "SELECT raw_content FROM raw_events WHERE turn_id=?", ("turn_001",)
            ).fetchone()
            assert raw_row is not None
            assert FAKE["openai_key"] not in raw_row[0]
            assert FAKE["github_token"] not in raw_row[0]
        finally:
            conn.close()


# --------------------------------------------------------------------------
# AC-3: audit_log records which tool_name triggered redaction
# --------------------------------------------------------------------------
class TestAuditToolNameContext:
    def test_audit_log_includes_tool_names(self, memory_db, fs_store):
        """When redaction fires on tool result/args, audit_log includes tool_name."""
        event = _make_event(
            content="call",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "github_api",
                        "arguments": json.dumps({"token": FAKE["github_token"]}),
                    },
                    "output": '{"ok": true}',
                }
            ],
        )
        result = capture_event(event, db=memory_db, fs=fs_store)
        assert result["redaction_fired"] is True
        assert result["audit_logged"] is True

        conn = memory_db._connect()
        try:
            row = conn.execute(
                "SELECT detail_json FROM audit_log ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            assert row is not None
            detail = json.loads(row[0])
            assert "github_api" in detail.get("tool_names", [])
            assert detail.get("source_type") == "tool_result"
        finally:
            conn.close()

    def test_audit_log_multiple_tools(self, memory_db, fs_store):
        """When multiple tools fire redaction, all tool_names are listed."""
        event = _make_event(
            content="calls",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {"name": "tool_a", "arguments": json.dumps({"k": FAKE["aws_access_key"]})},
                    "output": None,
                },
                {
                    "id": "call_2",
                    "function": {"name": "tool_b", "arguments": "{}"},
                    "output": json.dumps({"key": FAKE["openai_key"]}),
                },
            ],
        )
        result = capture_event(event, db=memory_db, fs=fs_store)
        assert result["redaction_fired"] is True
        conn = memory_db._connect()
        try:
            row = conn.execute(
                "SELECT detail_json FROM audit_log ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            assert row is not None
            detail = json.loads(row[0])
            assert "tool_a" in detail.get("tool_names", [])
            assert "tool_b" in detail.get("tool_names", [])
        finally:
            conn.close()

    def test_audit_log_no_tool_name_for_message_content(self, memory_db, fs_store):
        """Redaction in plain message content (no tool) sets source_type='message'."""
        event = _make_event(
            content=f"use this key: {FAKE['openai_key']}",
            tool_calls=[],
        )
        result = capture_event(event, db=memory_db, fs=fs_store)
        assert result["redaction_fired"] is True
        conn = memory_db._connect()
        try:
            row = conn.execute(
                "SELECT detail_json FROM audit_log ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            assert row is not None
            detail = json.loads(row[0])
            assert detail.get("source_type") == "message"
            assert not detail.get("tool_names")
        finally:
            conn.close()


# --------------------------------------------------------------------------
# AC-4: Duplicate event_id events are idempotent
# --------------------------------------------------------------------------
class TestIdempotency:
    def test_duplicate_event_id_no_duplicate_turn_rows(self, memory_db, fs_store):
        """Calling capture_event twice with the same event_id creates one turn."""
        event = _make_event()
        capture_event(event, db=memory_db, fs=fs_store)
        capture_event(event, db=memory_db, fs=fs_store)
        conn = memory_db._connect()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM turns WHERE turn_id=?", ("turn_001",)
            ).fetchone()[0]
            assert count == 1, "Expected exactly 1 turn row for duplicate event_id"
        finally:
            conn.close()

    def test_duplicate_event_id_no_duplicate_raw_events(self, memory_db, fs_store):
        """Duplicate event_ids do not create duplicate raw_events rows."""
        event = _make_event()
        capture_event(event, db=memory_db, fs=fs_store)
        capture_event(event, db=memory_db, fs=fs_store)
        conn = memory_db._connect()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM raw_events WHERE event_id=?", ("evt_001",)
            ).fetchone()[0]
            assert count == 1
        finally:
            conn.close()

    def test_duplicate_event_id_preserves_first_result(self, memory_db, fs_store):
        """Duplicate capture returns success but doesn't modify existing data."""
        event = _make_event(content="original content")
        r1 = capture_event(event, db=memory_db, fs=fs_store)
        event["content"] = "modified content"
        r2 = capture_event(event, db=memory_db, fs=fs_store)
        assert r2["redaction_fired"] == r1["redaction_fired"]
        conn = memory_db._connect()
        try:
            row = conn.execute(
                "SELECT content FROM turns WHERE turn_id=?", ("turn_001",)
            ).fetchone()
            assert row is not None
            assert row[0] == "original content"
        finally:
            conn.close()

    def test_different_sequence_same_turn_id(self, memory_db, fs_store):
        """Same turn_id with different sequence: second upserts on turn_id (idempotent)."""
        event1 = _make_event(sequence=1)
        event2 = _make_event(sequence=2)
        r1 = capture_event(event1, db=memory_db, fs=fs_store)
        r2 = capture_event(event2, db=memory_db, fs=fs_store)
        assert r1["redaction_fired"] == r2["redaction_fired"]
        conn = memory_db._connect()
        try:
            rows = conn.execute(
                "SELECT sequence FROM turns WHERE turn_id=? ORDER BY sequence",
                ("turn_001",),
            ).fetchall()
            # Pipeline upserts on turn_id PRIMARY KEY — same turn_id twice stores first
            assert len(rows) == 1, f"Expected 1 row (idempotent upsert), got {len(rows)}"
            assert rows[0][0] == 1, "First sequence should be preserved"
        finally:
            conn.close()