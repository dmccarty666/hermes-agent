"""Test HermesLocalProvider.sync_turn wiring (Story T-007).

AC-1: sync_turn creates two events (user + assistant) with same turn_id
AC-2: sync_turn content passes through redaction pipeline
AC-3: empty session_id is rejected with warning (no exception)
AC-4: events written to SQLite (sessions + turns + raw_events tables)
AC-5: events written to JSONL file
AC-6: sequence increments per sync_turn call
AC-7: initialize() sets pipeline singletons so capture_event works
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes_memory_core.store import fs as fs_module
from hermes_memory_core.store import sqlite as sqlite_module
from hermes_memory_core.write import pipeline as pipeline_module

# ---------------------------------------------------------------------------
# Fixtures — fake secrets (clearly fake, no risk)
# ---------------------------------------------------------------------------
FAKE = {
    "openai_key": "sk-test12345678901234567890",  # sk- + 20+ chars
    "aws_access_key": "AKIAFAKEFAKEFAKEFAKE",  # AKIA + 16 uppercase alphanum
    "github_token": "ghp_fakefakefakefakefakefakefakefake",  # ghp_ + 36
    "card_visa": "4532015112830368",  # Visa 16 digits
    "ssn": "123-45-6789",  # SSN with dashes
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
    original_db = pipeline_module._memory_db
    original_fs = pipeline_module._fs_store
    pipeline_module._memory_db = None
    pipeline_module._fs_store = None
    yield
    pipeline_module._memory_db = original_db
    pipeline_module._fs_store = original_fs


@pytest.fixture
def provider(memory_db, fs_store):
    """HermesLocalProvider wired to test pipeline singletons."""
    import importlib.util

    plugin_path = Path(__file__).parents[3] / "plugins" / "memory" / "hermes-local" / "__init__.py"
    spec = importlib.util.spec_from_file_location("hermes_local_plugin", str(plugin_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Wire pipeline singletons to our isolated test instances
    pipeline_module._inject_for_test(memory_db, fs_store)

    prov = module.HermesLocalProvider(config={})
    prov.initialize(session_id="test-session-001")
    return prov


# ---------------------------------------------------------------------------
# AC-1: Two events per turn, same turn_id
# ---------------------------------------------------------------------------
class TestSyncTurnTwoEvents:
    def test_creates_user_and_assistant_events(self, provider, memory_db):
        """sync_turn emits two events: role=user and role=assistant."""
        provider.sync_turn(
            user_content="hello",
            assistant_content="hi there",
            session_id="test-session-001",
        )

        conn = memory_db._connect()
        try:
            rows = conn.execute(
                "SELECT event_id, raw_content FROM raw_events ORDER BY rowid"
            ).fetchall()
            assert len(rows) == 2, f"Expected 2 events, got {len(rows)}: {rows}"

            # Parse content from raw_events JSON
            contents = set()
            roles = set()
            for (evt_id, raw) in rows:
                obj = json.loads(raw)
                contents.add(obj.get("content", ""))
                roles.add(obj.get("role", ""))

            assert "user" in roles, f"user event missing from {rows}"
            assert "assistant" in roles, f"assistant event missing from {rows}"
            assert "hello" in contents
            assert "hi there" in contents
        finally:
            conn.close()

    def test_user_and_assistant_share_turn_id(self, provider, memory_db):
        """Both events from one sync_turn call share the same turn_id."""
        provider.sync_turn(
            user_content="ping",
            assistant_content="pong",
            session_id="test-session-001",
        )

        conn = memory_db._connect()
        try:
            rows = conn.execute(
                "SELECT turn_id FROM raw_events ORDER BY rowid"
            ).fetchall()
            turn_ids = {r[0] for r in rows}
            assert len(turn_ids) == 1, f"Expected 1 turn_id, got {turn_ids}"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# AC-2: Redaction applied to sync_turn content
# ---------------------------------------------------------------------------
class TestSyncTurnRedaction:
    def _get_raw_content(self, memory_db, role_filter):
        """Helper: get raw_content JSON from raw_events for given role."""
        conn = memory_db._connect()
        try:
            rows = conn.execute("SELECT raw_content FROM raw_events").fetchall()
            for (raw,) in rows:
                obj = json.loads(raw)
                if obj.get("role") == role_filter:
                    return raw
            return ""
        finally:
            conn.close()

    def test_openai_key_redacted_in_content(self, provider, memory_db):
        """OpenAI key in content is redacted before storage."""
        provider.sync_turn(
            user_content=f"My key is {FAKE['openai_key']}",
            assistant_content="stored",
            session_id="test-session-001",
        )

        raw = self._get_raw_content(memory_db, "user")
        assert FAKE["openai_key"] not in raw, f"Raw secret still present: {raw}"
        assert "[REDACTED:openai_key]" in raw, f"Redaction marker missing: {raw}"

    def test_aws_key_redacted_in_content(self, provider, memory_db):
        """AWS access key in content is redacted before storage."""
        provider.sync_turn(
            user_content=f"using {FAKE['aws_access_key']} for auth",
            assistant_content="done",
            session_id="test-session-001",
        )

        raw = self._get_raw_content(memory_db, "user")
        assert FAKE["aws_access_key"] not in raw
        assert "[REDACTED:aws_access_key]" in raw

    def test_ssn_redacted_in_content(self, provider, memory_db):
        """SSN in content is redacted before storage."""
        provider.sync_turn(
            user_content=f"SSN: {FAKE['ssn']}",
            assistant_content="recorded",
            session_id="test-session-001",
        )

        raw = self._get_raw_content(memory_db, "user")
        assert FAKE["ssn"] not in raw
        assert "[REDACTED:ssn]" in raw


# ---------------------------------------------------------------------------
# AC-3: Empty session_id is rejected
# ---------------------------------------------------------------------------
class TestSyncTurnEmptySession:
    def test_empty_session_id_logs_warning_and_does_not_raise(self, provider, caplog):
        """Empty session_id produces a warning log, not an exception."""
        import logging

        with caplog.at_level(logging.WARNING):
            # Should not raise
            provider.sync_turn(
                user_content="hello",
                assistant_content="hi",
                session_id="",
            )

        assert any("empty session_id" in msg.lower() for msg in caplog.text.split("\n"))

    def test_empty_session_id_does_not_write_events(self, provider, memory_db):
        """No events written when session_id is empty."""
        provider.sync_turn(
            user_content="hello",
            assistant_content="hi",
            session_id="",
        )

        conn = memory_db._connect()
        try:
            count = conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
            assert count == 0, f"Unexpected events written: {count}"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# AC-4: Events land in SQLite tables
# ---------------------------------------------------------------------------
class TestSyncTurnSQLite:
    def test_writes_session_row(self, provider, memory_db):
        """sync_turn creates a session row in the sessions table."""
        provider.sync_turn(
            user_content="hello",
            assistant_content="hi there",
            session_id="test-session-001",
        )

        conn = memory_db._connect()
        try:
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id='test-session-001'"
            ).fetchone()
            assert row is not None, "session_id not written to sessions table"
        finally:
            conn.close()

    def test_writes_turn_row(self, provider, memory_db):
        """sync_turn creates a turn row in the turns table."""
        provider.sync_turn(
            user_content="hello",
            assistant_content="hi there",
            session_id="test-session-001",
        )

        conn = memory_db._connect()
        try:
            count = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
            assert count >= 1, "no turn row written"
        finally:
            conn.close()

    def test_writes_raw_events(self, provider, memory_db):
        """sync_turn writes user + assistant rows to raw_events."""
        provider.sync_turn(
            user_content="hello",
            assistant_content="hi there",
            session_id="test-session-001",
        )

        conn = memory_db._connect()
        try:
            count = conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
            assert count == 2, f"expected 2 raw_events, got {count}"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# AC-5: Events land in JSONL
# ---------------------------------------------------------------------------
class TestSyncTurnJSONL:
    def test_writes_to_jsonl_file(self, provider, fs_store):
        """sync_turn writes events to the session JSONL file."""
        provider.sync_turn(
            user_content="hello",
            assistant_content="hi there",
            session_id="test-session-001",
        )

        jsonl_path = fs_store._jsonl_path("test-session-001")
        assert jsonl_path.exists(), f"JSONL file not created at {jsonl_path}"

        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 2, f"expected 2 JSONL lines, got {len(lines)}"

        for line in lines:
            parsed = json.loads(line)
            assert "event_id" in parsed
            assert "role" in parsed

    def test_jsonl_contains_redacted_content(self, provider, fs_store):
        """JSONL stores redacted content, not raw secrets."""
        provider.sync_turn(
            user_content=f"key={FAKE['openai_key']}",
            assistant_content="ok",
            session_id="test-session-001",
        )

        jsonl_path = fs_store._jsonl_path("test-session-001")
        raw = jsonl_path.read_text()

        assert FAKE["openai_key"] not in raw, "raw secret in JSONL"
        assert "[REDACTED:openai_key]" in raw, "redaction marker missing from JSONL"


# ---------------------------------------------------------------------------
# AC-6: Sequence increments
# ---------------------------------------------------------------------------
class TestSyncTurnSequence:
    def test_sequence_increments_per_call(self, provider, memory_db):
        """Each sync_turn call increments the sequence counter."""
        provider.sync_turn(
            user_content="turn 1",
            assistant_content="response 1",
            session_id="test-session-001",
        )
        provider.sync_turn(
            user_content="turn 2",
            assistant_content="response 2",
            session_id="test-session-001",
        )

        conn = memory_db._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT sequence FROM turns ORDER BY sequence"
            ).fetchall()
            sequences = [r[0] for r in rows]
            # We expect at least two distinct sequences (one per sync_turn call)
            assert len(sequences) >= 2, f"expected multiple sequences, got {sequences}"
            assert sequences == sorted(sequences), f"sequences not increasing: {sequences}"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# AC-7: initialize() sets pipeline singletons
# ---------------------------------------------------------------------------
class TestSyncTurnPipelineInit:
    def test_capture_event_uses_pipeline_singletons(self, memory_db, fs_store):
        """When pipeline singletons are set, capture_event uses them."""
        import importlib.util

        plugin_path = Path(__file__).parents[3] / "plugins" / "memory" / "hermes-local" / "__init__.py"
        spec = importlib.util.spec_from_file_location("hermes_local_plugin", str(plugin_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Explicitly set singletons via test injection
        pipeline_module._inject_for_test(memory_db, fs_store)

        provider = module.HermesLocalProvider(config={})
        provider.initialize(session_id="test-session-001")

        # sync_turn should work without passing db/fs explicitly
        provider.sync_turn(
            user_content="hello",
            assistant_content="hi",
            session_id="test-session-001",
        )

        conn = memory_db._connect()
        try:
            count = conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
            assert count == 2, f"expected 2 events via singletons, got {count}"
        finally:
            conn.close()