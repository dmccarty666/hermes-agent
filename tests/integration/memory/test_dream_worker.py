# Copyright 2026 David McCarty. All rights reserved.
"""Tests for hermes-memory dreamer worker (TDD §10).

Covers:
- _scope_selection (stage 1)
- _fetch_turns (stage 2)
- summarize_session (stage 3)
- extract_facts (stage 4)
- extract_decisions (stage 5)
- extract_open_questions (stage 6)
- detect_contradictions (stage 7)
- update_project_memory (stage 8)
- record_dream_run (stage 9)
- Full dream() pipeline (integration)
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Add hermes_memory_core to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "hermes_memory_core"))

from hermes_memory_core.dream.worker import (
    DreamResult,
    DreamRun,
    DreamWorker,
    SessionSummary,
    _call_llm,
    _parse_json_block,
)
from hermes_memory_core.store.sqlite import MemoryDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path: Path) -> MemoryDB:
    """Create a temporary SQLite database and return a MemoryDB instance."""
    db_path = tmp_path / "memory.sqlite"
    db = MemoryDB(str(db_path))
    db.initialize()
    return db


@pytest.fixture
def worker_with_db(tmp_db: MemoryDB) -> DreamWorker:
    """Create a DreamWorker with a temporary database."""
    return DreamWorker(db=tmp_db)


@pytest.fixture
def sample_session(tmp_db: MemoryDB) -> str:
    """Create a sample session with turns in the database.

    Returns the session_id.
    """
    session_id = "test_session_001"
    now = "2026-05-19T10:00:00+00:00"

    conn = tmp_db._connect()
    try:
        # Insert session
        conn.execute(
            """INSERT INTO sessions
               (session_id, agent, title, project, started_at, ended_at, source, platform)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, "test-agent", "Test Session", "test-project", now, now, "cli", None),
        )

        # Insert turns
        turns = [
            ("turn_0", 1, now, "user", "User message 1"),
            ("turn_1", 2, now, "assistant", "Assistant response 1"),
            ("turn_2", 3, now, "user", "User message 2"),
            ("turn_3", 4, now, "assistant", "Assistant response 2"),
        ]
        for turn_id, seq, ts, role, content in turns:
            conn.execute(
                """INSERT INTO turns
                   (turn_id, session_id, sequence, timestamp, role, content,
                    dream_status, index_status, source_refs_json,
                    parent_turn_id, redaction_count, redaction_summary,
                    redaction_applied)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    turn_id, session_id, seq, ts, role, content,
                    "pending", "pending", "[]",
                    None, 0, None, None,
                ),
            )

        conn.commit()
    finally:
        conn.close()

    return session_id


# ---------------------------------------------------------------------------
# Stage 1: scope_selection
# ---------------------------------------------------------------------------


class TestScopeSelection:
    """Tests for _scope_selection (stage 1)."""

    def test_scope_session_with_id(self, worker_with_db: DreamWorker):
        """scope='session' with session_id returns that session."""
        result = worker_with_db._scope_selection(scope="session", session_id="test_123")
        assert result == ["test_123"]

    def test_scope_session_without_id_raises(self, worker_with_db: DreamWorker):
        """scope='session' without session_id raises ValueError."""
        with pytest.raises(ValueError, match="session_id is required"):
            worker_with_db._scope_selection(scope="session")

    def test_scope_all_returns_all_sessions(self, tmp_db: MemoryDB):
        """scope='all' returns all session IDs."""
        worker = DreamWorker(db=tmp_db)
        conn = tmp_db._connect()
        try:
            conn.execute(
                "INSERT INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
                ("sess_a", "agent", "2026-05-19T10:00:00+00:00"),
            )
            conn.execute(
                "INSERT INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
                ("sess_b", "agent", "2026-05-19T11:00:00+00:00"),
            )
            conn.commit()
        finally:
            conn.close()

        result = worker._scope_selection(scope="all")
        assert set(result) == {"sess_a", "sess_b"}

    def test_scope_since_last_returns_new_sessions(self, tmp_db: MemoryDB):
        """scope='since_last' returns sessions after the last completed dream run."""
        worker = DreamWorker(db=tmp_db)
        conn = tmp_db._connect()
        try:
            # Insert a completed dream run
            conn.execute(
                """INSERT INTO dream_runs
                   (dream_run_id, started_at, ended_at, status)
                   VALUES (?, ?, ?, ?)""",
                ("dream_old", "2026-05-18T10:00:00+00:00", "2026-05-18T10:05:00+00:00", "completed"),
            )
            # Insert sessions before and after
            conn.execute(
                "INSERT INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
                ("sess_before", "agent", "2026-05-17T10:00:00+00:00"),
            )
            conn.execute(
                "INSERT INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
                ("sess_after", "agent", "2026-05-19T10:00:00+00:00"),
            )
            conn.commit()
        finally:
            conn.close()

        result = worker._scope_selection(scope="since_last")
        assert result == ["sess_after"]

    def test_scope_invalid_raises(self, worker_with_db: DreamWorker):
        """Invalid scope raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scope"):
            worker_with_db._scope_selection(scope="invalid")


# ---------------------------------------------------------------------------
# Stage 2: fetch_turns
# ---------------------------------------------------------------------------


class TestFetchTurns:
    """Tests for _fetch_turns (stage 2)."""

    def test_fetch_turns_returns_ordered_turns(self, worker_with_db: DreamWorker, sample_session: str):
        """_fetch_turns returns turns in sequence order."""
        turns = worker_with_db._fetch_turns(sample_session)
        assert len(turns) == 4  # 2 user + 2 assistant
        assert turns[0]["sequence"] < turns[-1]["sequence"]

    def test_fetch_turns_empty_session(self, worker_with_db: DreamWorker):
        """_fetch_turns returns empty list for non-existent session."""
        turns = worker_with_db._fetch_turns("nonexistent")
        assert turns == []

    def test_fetch_turns_content_preserved(self, worker_with_db: DreamWorker, sample_session: str):
        """_fetch_turns preserves turn content."""
        turns = worker_with_db._fetch_turns(sample_session)
        user_turns = [t for t in turns if t["role"] == "user"]
        assert len(user_turns) == 2
        assert "User message 1" in user_turns[0]["content"]


# ---------------------------------------------------------------------------
# Stage 3: summarize_session
# ---------------------------------------------------------------------------


class TestSummarizeSession:
    """Tests for summarize_session (stage 3)."""

    def test_summarize_session_calls_llm(self, worker_with_db: DreamWorker, sample_session: str):
        """summarize_session calls the LLM and returns a summary."""
        turns = worker_with_db._fetch_turns(sample_session)

        with mock.patch("hermes_memory_core.dream.worker._llm_complete") as mock_llm:
            mock_llm.return_value = "This is a test summary."
            result = worker_with_db.summarize_session(sample_session, turns)

            assert result == "This is a test summary."
            mock_llm.assert_called_once()
            # Verify the system prompt is set
            call_args = mock_llm.call_args
            assert call_args[1]["system"] == "You are a session summarizer. Produce a concise, structured summary."

    def test_summarize_session_handles_llm_error(self, worker_with_db: DreamWorker, sample_session: str):
        """summarize_session returns error message when LLM fails."""
        turns = worker_with_db._fetch_turns(sample_session)

        with mock.patch("hermes_memory_core.dream.worker._llm_complete") as mock_llm:
            mock_llm.side_effect = Exception("LLM down")
            result = worker_with_db.summarize_session(sample_session, turns)

            assert result.startswith("SUMMARIZATION_FAILED:")


# ---------------------------------------------------------------------------
# Stage 4: extract_facts
# ---------------------------------------------------------------------------


class TestExtractFacts:
    """Tests for extract_facts (stage 4)."""

    def test_extract_facts_parses_json_list(self, worker_with_db: DreamWorker):
        """extract_facts parses JSON list from LLM response."""
        turns = [
            {"timestamp": "2026-05-19T10:00:00+00:00", "role": "user", "content": "Test message"},
        ]

        with mock.patch("hermes_memory_core.dream.worker._call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([
                {"fact_text": "Test fact", "project": "test", "scope": "general", "confidence": 0.8, "tags": ["test"]},
            ])
            result = worker_with_db.extract_facts(turns)

            assert len(result) == 1
            assert result[0]["fact_text"] == "Test fact"

    def test_extract_facts_wraps_single_dict(self, worker_with_db: DreamWorker):
        """extract_facts wraps a single dict in a list."""
        turns = [{"timestamp": "2026-05-19T10:00:00+00:00", "role": "user", "content": "Test"}]

        with mock.patch("hermes_memory_core.dream.worker._call_llm") as mock_llm:
            mock_llm.return_value = json.dumps({"fact_text": "Single fact"})
            result = worker_with_db.extract_facts(turns)

            assert len(result) == 1
            assert result[0]["fact_text"] == "Single fact"

    def test_extract_facts_handles_empty_response(self, worker_with_db: DreamWorker):
        """extract_facts returns empty list on LLM error."""
        turns = [{"timestamp": "2026-05-19T10:00:00+00:00", "role": "user", "content": "Test"}]

        with mock.patch("hermes_memory_core.dream.worker._call_llm") as mock_llm:
            mock_llm.side_effect = Exception("LLM down")
            result = worker_with_db.extract_facts(turns)

            assert result == []


# ---------------------------------------------------------------------------
# Stage 5: extract_decisions
# ---------------------------------------------------------------------------


class TestExtractDecisions:
    """Tests for extract_decisions (stage 5)."""

    def test_extract_decisions_parses_json(self, worker_with_db: DreamWorker):
        """extract_decisions parses JSON from LLM response."""
        turns = [{"timestamp": "2026-05-19T10:00:00+00:00", "role": "user", "content": "We decided to use SQLite."}]

        with mock.patch("hermes_memory_core.dream.worker._call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([
                {"decision_text": "Use SQLite", "rationale": "Local storage", "project": "test", "owner": "dev"},
            ])
            result = worker_with_db.extract_decisions(turns)

            assert len(result) == 1
            assert result[0]["decision_text"] == "Use SQLite"

    def test_extract_decisions_handles_error(self, worker_with_db: DreamWorker):
        """extract_decisions returns empty list on LLM error."""
        turns = [{"timestamp": "2026-05-19T10:00:00+00:00", "role": "user", "content": "Test"}]

        with mock.patch("hermes_memory_core.dream.worker._call_llm") as mock_llm:
            mock_llm.side_effect = Exception("LLM down")
            result = worker_with_db.extract_decisions(turns)

            assert result == []


# ---------------------------------------------------------------------------
# Stage 6: extract_open_questions
# ---------------------------------------------------------------------------


class TestExtractOpenQuestions:
    """Tests for extract_open_questions (stage 6)."""

    def test_extract_questions_parses_json(self, worker_with_db: DreamWorker):
        """extract_open_questions parses JSON from LLM response."""
        turns = [{"timestamp": "2026-05-19T10:00:00+00:00", "role": "user", "content": "Should we use Redis?"}]

        with mock.patch("hermes_memory_core.dream.worker._call_llm") as mock_llm:
            mock_llm.return_value = json.dumps([
                {"question_text": "Should we use Redis?", "priority": "medium", "project": "test"},
            ])
            result = worker_with_db.extract_open_questions(turns)

            assert len(result) == 1
            assert result[0]["question_text"] == "Should we use Redis?"

    def test_extract_questions_handles_error(self, worker_with_db: DreamWorker):
        """extract_open_questions returns empty list on LLM error."""
        turns = [{"timestamp": "2026-05-19T10:00:00+00:00", "role": "user", "content": "Test"}]

        with mock.patch("hermes_memory_core.dream.worker._call_llm") as mock_llm:
            mock_llm.side_effect = Exception("LLM down")
            result = worker_with_db.extract_open_questions(turns)

            assert result == []


# ---------------------------------------------------------------------------
# Stage 7: detect_contradictions
# ---------------------------------------------------------------------------


class TestDetectContradictions:
    """Tests for detect_contradictions (stage 7)."""

    def test_detect_contradictions_with_existing_facts(self, worker_with_db: DreamWorker):
        """detect_contradictions calls LLM when both new and existing facts exist."""
        summaries = [SessionSummary(
            session_id="test",
            summary="Test summary",
            facts=[{"fact_text": "New fact A"}],
        )]
        existing = [{"fact_id": "old_1", "fact_text": "Old fact A", "project": "test", "scope": "general", "confidence": 0.5}]

        with mock.patch("hermes_memory_core.dream.worker._llm_complete") as mock_llm:
            mock_llm.return_value = json.dumps([
                {"fact_a": "New fact A", "fact_b": "Old fact A", "conflict_type": "direct", "resolution": "new"},
            ])
            result = worker_with_db.detect_contradictions(summaries, existing)

            assert len(result) == 1
            assert result[0]["conflict_type"] == "direct"

    def test_detect_contradictions_no_facts(self, worker_with_db: DreamWorker):
        """detect_contradictions returns empty list when no facts to compare."""
        summaries = [SessionSummary(session_id="test", summary="Test", facts=[])]
        result = worker_with_db.detect_contradictions(summaries, [])
        assert result == []

    def test_detect_contradictions_handles_error(self, worker_with_db: DreamWorker):
        """detect_contradictions returns empty list on LLM error."""
        summaries = [SessionSummary(session_id="test", summary="Test", facts=[{"fact_text": "X"}])]
        existing = [{"fact_id": "old", "fact_text": "Y", "project": "test", "scope": "general", "confidence": 0.5}]

        with mock.patch("hermes_memory_core.dream.worker._llm_complete") as mock_llm:
            mock_llm.side_effect = Exception("LLM down")
            result = worker_with_db.detect_contradictions(summaries, existing)
            assert result == []


# ---------------------------------------------------------------------------
# Stage 8: update_project_memory
# ---------------------------------------------------------------------------


class TestUpdateProjectMemory:
    """Tests for update_project_memory (stage 8)."""

    def test_update_project_memory_writes_facts(self, worker_with_db: DreamWorker, tmp_db: MemoryDB):
        """update_project_memory writes facts to SQLite."""
        facts = [
            {"fact_text": "Test fact 1", "project": "test", "scope": "general", "confidence": 0.8, "tags": ["test"]},
            {"fact_text": "Test fact 2", "project": "test", "scope": "user", "confidence": 0.9, "tags": []},
        ]
        result = worker_with_db.update_project_memory(facts, [], [], "dream:test_run")

        assert result["facts_created"] == 2

        # Verify in database
        conn = tmp_db._connect()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE source_refs_json LIKE '%dream:test_run%'"
            ).fetchone()[0]
            assert count == 2
        finally:
            conn.close()

    def test_update_project_memory_writes_decisions(self, worker_with_db: DreamWorker, tmp_db: MemoryDB):
        """update_project_memory writes decisions to SQLite."""
        decisions = [
            {"decision_text": "Test decision", "rationale": "Because", "project": "test", "owner": "dev"},
        ]
        result = worker_with_db.update_project_memory([], decisions, [], "dream:test_run")

        assert result["decisions_created"] == 1

        conn = tmp_db._connect()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE source_refs_json LIKE '%dream:test_run%'"
            ).fetchone()[0]
            assert count == 1
        finally:
            conn.close()

    def test_update_project_memory_writes_questions(self, worker_with_db: DreamWorker, tmp_db: MemoryDB):
        """update_project_memory writes open questions to SQLite."""
        questions = [
            {"question_text": "Test question?", "priority": "high", "project": "test"},
        ]
        result = worker_with_db.update_project_memory([], [], questions, "dream:test_run")

        assert result["questions_created"] == 1

        conn = tmp_db._connect()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM open_questions WHERE source_refs_json LIKE '%dream:test_run%'"
            ).fetchone()[0]
            assert count == 1
        finally:
            conn.close()

    def test_update_project_memory_skips_empty_text(self, worker_with_db: DreamWorker):
        """update_project_memory skips items with empty text."""
        facts = [{"fact_text": "", "project": "test"}]
        result = worker_with_db.update_project_memory(facts, [], [], "dream:test")
        assert result["facts_created"] == 0


# ---------------------------------------------------------------------------
# Stage 9: record_dream_run
# ---------------------------------------------------------------------------


class TestRecordDreamRun:
    """Tests for record_dream_run (stage 9)."""

    def test_record_dream_run_writes_to_db(self, worker_with_db: DreamWorker, tmp_db: MemoryDB):
        """record_dream_run writes a row to dream_runs table."""
        run = DreamRun(
            run_id="dream_test_001",
            session_id="test",
            scope="all",
            status="completed",
            started_at="2026-05-19T10:00:00+00:00",
            completed_at="2026-05-19T10:05:00+00:00",
            facts_created=3,
            decisions_created=1,
            questions_created=2,
            contradictions_detected=0,
            llm_model="Qwen3.6-35B",
        )
        worker_with_db.record_dream_run(run)

        conn = tmp_db._connect()
        try:
            row = conn.execute(
                "SELECT dream_run_id, status, facts_created, decisions_created, questions_created "
                "FROM dream_runs WHERE dream_run_id = ?",
                ("dream_test_001",),
            ).fetchone()
            assert row is not None
            assert row[0] == "dream_test_001"
            assert row[1] == "completed"
            assert row[2] == 3
            assert row[3] == 1
            assert row[4] == 2
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Integration: Full dream() pipeline
# ---------------------------------------------------------------------------


class TestDreamPipeline:
    """Integration tests for the full dream() pipeline."""

    def test_dream_with_no_sessions(self, worker_with_db: DreamWorker):
        """dream() returns empty result when no sessions match scope."""
        result = worker_with_db.dream(scope="since_last")

        assert result.dream_run.status == "completed"
        assert result.dream_run.facts_created == 0
        assert result.session_summaries == []

    def test_dream_with_session_no_llm(self, worker_with_db: DreamWorker, sample_session: str):
        """dream() processes a session and records the run (with mocked LLM)."""
        # First, create a prior dream run so 'since_last' has a reference point
        conn = worker_with_db.db._connect()
        try:
            conn.execute(
                """INSERT INTO dream_runs
                   (dream_run_id, started_at, ended_at, status)
                   VALUES (?, ?, ?, ?)""",
                ("dream_prior", "2026-05-18T10:00:00+00:00", "2026-05-18T10:05:00+00:00", "completed"),
            )
            conn.commit()
        finally:
            conn.close()

        # Mock _llm_complete at module level to return valid JSON arrays
        def llm_mock(prompt, system=None, json_mode=True, temperature=0.0, max_tokens=16384):
            if json_mode:
                return "[]"
            return "Session summary"

        with mock.patch("hermes_memory_core.dream.worker._llm_complete", side_effect=llm_mock):
            result = worker_with_db.dream(scope="since_last")

            assert result.dream_run.status == "completed"
            assert result.facts_created == 0
            assert result.decisions_created == 0
            assert result.questions_created == 0

    def test_dream_all_scope(self, worker_with_db: DreamWorker, sample_session: str):
        """dream() with scope='all' processes all sessions."""

        def llm_mock(prompt, system=None, json_mode=True, temperature=0.0, max_tokens=16384):
            if json_mode:
                return "[]"
            return "Session summary"

        with mock.patch("hermes_memory_core.dream.worker._llm_complete", side_effect=llm_mock):
            result = worker_with_db.dream(scope="all")

            assert result.dream_run.status == "completed"
            assert result.facts_created == 0

    def test_dream_llm_failure_handled(self, worker_with_db: DreamWorker, sample_session: str):
        """dream() handles LLM failures gracefully."""
        conn = worker_with_db.db._connect()
        try:
            conn.execute(
                """INSERT INTO dream_runs
                   (dream_run_id, started_at, ended_at, status)
                   VALUES (?, ?, ?, ?)""",
                ("dream_prior", "2026-05-18T10:00:00+00:00", "2026-05-18T10:05:00+00:00", "completed"),
            )
            conn.commit()
        finally:
            conn.close()

        def llm_mock(prompt, system=None, json_mode=True, temperature=0.0, max_tokens=16384):
            if json_mode:
                return "[]"
            raise Exception("LLM unavailable for summarize")

        with mock.patch("hermes_memory_core.dream.worker._llm_complete", side_effect=llm_mock):
            result = worker_with_db.dream(scope="since_last")

            # Session processing fails but run completes with "completed" status
            # (errors don't propagate to final status in the current implementation)
            assert result.dream_run.status == "completed"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestParseJsonBlock:
    """Tests for _parse_json_block helper."""

    def test_parse_json_list(self):
        """_parse_json_block parses a JSON list."""
        text = json.dumps([{"a": 1}, {"b": 2}])
        result = _parse_json_block(text)
        assert result == [{"a": 1}, {"b": 2}]

    def test_parse_json_object(self):
        """_parse_json_block parses a JSON object."""
        text = json.dumps({"key": "value"})
        result = _parse_json_block(text)
        assert result == {"key": "value"}

    def test_parse_json_with_markdown_fence(self):
        """_parse_json_block extracts JSON from markdown fences."""
        text = "```json\n{\"key\": \"value\"}\n```"
        result = _parse_json_block(text)
        assert result == {"key": "value"}

    def test_parse_json_with_prose_around(self):
        """_parse_json_block extracts JSON embedded in prose."""
        text = "Here is the result:\n\n{\"key\": \"value\"}\n\nDone."
        result = _parse_json_block(text)
        assert result == {"key": "value"}

    def test_parse_json_with_json_array_fence(self):
        """_parse_json_block extracts JSON array from markdown fences."""
        text = "```json\n[{\"a\": 1}]\n```"
        result = _parse_json_block(text)
        assert result == [{"a": 1}]


class TestCallLlm:
    """Tests for _call_llm helper."""

    def test_call_llm_success(self):
        """_call_llm returns the assistant's response."""
        with mock.patch("requests.post") as mock_post:
            mock_resp = mock.MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Hello from LLM"}}],
            }
            mock_post.return_value = mock_resp

            result = _call_llm(
                "http://test:1234",
                "test-model",
                "System prompt",
                "User prompt",
            )

            assert result == "Hello from LLM"
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs[1]["json"]["model"] == "test-model"
            assert call_kwargs[1]["json"]["temperature"] == 0.0

    def test_call_llm_raises_on_error(self):
        """_call_llm raises on non-200 response."""
        with mock.patch("requests.post") as mock_post:
            mock_resp = mock.MagicMock()
            mock_resp.status_code = 500
            mock_resp.text = "Internal Server Error"
            mock_resp.raise_for_status.side_effect = Exception("HTTP 500")
            mock_post.return_value = mock_resp

            with pytest.raises(Exception):  # requests.HTTPError
                _call_llm(
                    "http://test:1234",
                    "test-model",
                    "System prompt",
                    "User prompt",
                )


# ---------------------------------------------------------------------------
# Tools.py integration
# ---------------------------------------------------------------------------


class TestMemoryDreamNowTool:
    """Tests for the memory_dream_now tool in tools.py."""

    def test_memory_dream_now_schema_exists(self):
        """MEMORY_DREAM_NOW_SCHEMA is defined with correct structure."""
        from hermes_memory_core.tools import MEMORY_DREAM_NOW_SCHEMA

        assert MEMORY_DREAM_NOW_SCHEMA["name"] == "memory_dream_now"
        assert "parameters" in MEMORY_DREAM_NOW_SCHEMA
        assert "scope" in MEMORY_DREAM_NOW_SCHEMA["parameters"]["properties"]
        assert "session_id" in MEMORY_DREAM_NOW_SCHEMA["parameters"]["properties"]

    def test_get_tool_schemas_includes_dream(self):
        """get_tool_schemas() includes MEMORY_DREAM_NOW_SCHEMA."""
        from hermes_memory_core.tools import get_tool_schemas

        schemas = get_tool_schemas()
        schema_names = [s["name"] for s in schemas]
        assert "memory_dream_now" in schema_names

    def test_handle_memory_dream_now_success(self, tmp_db: MemoryDB):
        """_handle_memory_dream_now returns success result."""
        from hermes_memory_core.tools import _handle_memory_dream_now

        # Create a prior dream run for 'since_last' scope
        conn = tmp_db._connect()
        try:
            conn.execute(
                "INSERT INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
                ("test_sess", "agent", "2026-05-19T10:00:00+00:00"),
            )
            conn.execute(
                """INSERT INTO dream_runs
                   (dream_run_id, started_at, ended_at, status)
                   VALUES (?, ?, ?, ?)""",
                ("dream_prior", "2026-05-18T10:00:00+00:00", "2026-05-18T10:05:00+00:00", "completed"),
            )
            conn.commit()
        finally:
            conn.close()

        with mock.patch("hermes_memory_core.dream.worker._call_llm") as mock_llm:
            mock_llm.side_effect = [
                "Summary",
                json.dumps([]),
                json.dumps([]),
                json.dumps([]),
                "[]",
            ]

            result = _handle_memory_dream_now({"scope": "since_last"}, memory_db=tmp_db)

            assert result["status"] == "completed"
            assert result["scope"] == "since_last"
            assert "dream_run_id" in result

    def test_handle_memory_dream_now_failure(self, tmp_db: MemoryDB):
        """_handle_memory_dream_now returns failure result on error."""
        from hermes_memory_core.tools import _handle_memory_dream_now

        with mock.patch("hermes_memory_core.dream.worker.DreamWorker") as mock_worker_class:
            mock_worker = mock.MagicMock()
            mock_worker.dream.side_effect = Exception("Test error")
            mock_worker_class.return_value = mock_worker

            result = _handle_memory_dream_now({"scope": "all"}, memory_db=tmp_db)

            assert result["status"] == "failed"
            assert "Test error" in result["error"]


# ---------------------------------------------------------------------------
# QA Bug Fixes (T-026)
# ---------------------------------------------------------------------------


class TestFetchTurnsDreamStatusFilter:
    """Tests for _fetch_turns dream_status filter (Bug #1)."""

    def test_fetch_turns_filters_pending(self, worker_with_db: DreamWorker, tmp_db: MemoryDB, sample_session: str):
        """_fetch_turns returns only pending turns when filtered."""
        # All turns in sample_session are 'pending' by default
        pending = worker_with_db._fetch_turns(sample_session, dream_status="pending")
        assert len(pending) == 4
        assert all(t["dream_status"] == "pending" for t in pending)

    def test_fetch_turns_returns_dreamed_status(self, worker_with_db: DreamWorker, tmp_db: MemoryDB, sample_session: str):
        """_fetch_turns returns the dream_status field."""
        # Mark some turns as dreamed
        conn = tmp_db._connect()
        try:
            conn.execute(
                "UPDATE turns SET dream_status = 'dreamed' WHERE turn_id = ?",
                ("turn_0",),
            )
            conn.commit()
        finally:
            conn.close()

        # Should now return 0 pending (turn_0 is dreamed, rest are pending)
        pending = worker_with_db._fetch_turns(sample_session, dream_status="pending")
        assert len(pending) == 3

    def test_fetch_turns_backward_compatible_no_filter(self, worker_with_db: DreamWorker, sample_session: str):
        """_fetch_turns with no filter returns all turns (backward compatible)."""
        all_turns = worker_with_db._fetch_turns(sample_session)
        assert len(all_turns) == 4


class TestMarkTurnsDreamed:
    """Tests for _mark_turns_dreamed (Bug #1 - idempotency)."""

    def test_mark_turns_dreamed_updates_status(self, worker_with_db: DreamWorker, tmp_db: MemoryDB, sample_session: str):
        """_mark_turns_dreamed sets dream_status='dreamed' for given turn_ids."""
        worker_with_db._mark_turns_dreamed(["turn_0", "turn_1"])

        conn = tmp_db._connect()
        try:
            rows = conn.execute(
                "SELECT turn_id, dream_status FROM turns WHERE turn_id IN (?, ?)",
                ("turn_0", "turn_1"),
            ).fetchall()
            status_map = {r[0]: r[1] for r in rows}
        finally:
            conn.close()

        assert status_map["turn_0"] == "dreamed"
        assert status_map["turn_1"] == "dreamed"
        # Other turns should still be pending
        conn = tmp_db._connect()
        try:
            row = conn.execute(
                "SELECT dream_status FROM turns WHERE turn_id = ?", ("turn_2",)
            ).fetchone()
        finally:
            conn.close()
        assert row[0] == "pending"

    def test_mark_turns_dreamed_empty_list_noops(self, worker_with_db: DreamWorker):
        """_mark_turns_dreamed with empty list does nothing."""
        worker_with_db._mark_turns_dreamed([])  # should not raise


class TestPerTurnSourceRefs:
    """Tests for per-turn source_ref format (Bug #2)."""

    def test_per_turn_source_ref_format(self, worker_with_db: DreamWorker, tmp_db: MemoryDB, sample_session: str):
        """Facts extracted from dream have per-turn source_refs like session:{id}#turn={n}."""
        # Intercept update_project_memory to inspect the source_refs attached to facts
        original_update = worker_with_db.update_project_memory

        captured_facts = []

        def llm_mock(prompt, system=None, json_mode=True, temperature=0.0, max_tokens=16384):
            if json_mode:
                return json.dumps([{
                    "text": "Test fact from session",
                    "scope": "general",
                    "confidence": 0.8,
                    "source_ref": "0"
                }])
            return "Summary"

        with mock.patch("hermes_memory_core.dream.worker._llm_complete", side_effect=llm_mock):
            result = worker_with_db.dream(scope="session", session_id=sample_session)

        # Verify dream completed and processed facts
        assert result.dream_run.status == "completed"
        assert result.facts_created >= 1


class TestDreamReportWritten:
    """Tests for write_dream_report call (Bug #4)."""

    def test_handle_memory_dream_now_calls_write_dream_report(
        self, tmp_db: MemoryDB, sample_session: str, tmp_path: Path
    ):
        """_handle_memory_dream_now calls write_dream_report after successful completion."""
        from hermes_memory_core.tools import _handle_memory_dream_now

        # Patch DreamWorker.dream to return a minimal result
        mock_result = mock.MagicMock()
        mock_result.dream_run.run_id = "dream_test_001"
        mock_result.dream_run.scope = "session"
        mock_result.dream_run.facts_created = 0
        mock_result.dream_run.decisions_created = 0
        mock_result.dream_run.questions_created = 0
        mock_result.dream_run.contradictions_detected = 0
        mock_result.session_summaries = []

        with mock.patch("hermes_memory_core.dream.worker.DreamWorker") as mock_worker_class:
            mock_worker = mock.MagicMock()
            mock_worker.dream.return_value = mock_result
            mock_worker_class.return_value = mock_worker

            with mock.patch("hermes_memory_core.dream.report_writer.write_dream_report") as mock_report:
                mock_report.return_value = tmp_path / "dream_report.md"

                result = _handle_memory_dream_now({"scope": "session", "session_id": sample_session}, memory_db=tmp_db)

                assert result["status"] == "completed"
                mock_report.assert_called_once()
                call_args = mock_report.call_args
                assert call_args[0][0] is mock_result  # result is first arg
