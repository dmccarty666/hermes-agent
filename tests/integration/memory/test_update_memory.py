"""
Tests for update_memory and fact_feedback (Story 4.4.2).

Runs against real SQLite at tmp_path (pytest tmp_path fixture).
Covers:
  - update_memory: fact text update, status update, trust_delta, not_found
  - update_memory: decision text/status update, open_question status update
  - update_memory: invalid memory_type returns error
  - update_memory: nothing_to_update when no fields provided
  - fact_feedback: helpful (+0.05) increases confidence
  - fact_feedback: unhelpful (-0.10) decreases confidence
  - fact_feedback: confidence clamped to [0.0, 1.0]
  - fact_feedback: invalid action returns error
  - fact_feedback: not_found returns error
  - fact_feedback: audit_log row written
"""

import json
import sqlite3

import pytest

from hermes_memory_core.write.pipeline import update_memory, fact_feedback, write_memory


@pytest.fixture
def memory_db_path(tmp_path):
    """Provide a temp SQLite path, init schema, and clean up after."""
    db_path = tmp_path / "memory.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions "
        "(session_id TEXT PRIMARY KEY, agent TEXT NOT NULL, title TEXT, "
        "project TEXT, started_at TEXT NOT NULL, ended_at TEXT, source TEXT, platform TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS turns "
        "(turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, sequence INTEGER NOT NULL, "
        "timestamp TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, "
        "dream_status TEXT NOT NULL DEFAULT 'pending', index_status TEXT NOT NULL DEFAULT 'pending', "
        "source_refs_json TEXT NOT NULL DEFAULT '[]', parent_turn_id TEXT, "
        "redaction_count INTEGER NOT NULL DEFAULT 0, redaction_summary TEXT, "
        "redaction_applied TEXT, redaction_types_json TEXT NOT NULL DEFAULT '[]')"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS facts "
        "(fact_id TEXT PRIMARY KEY, fact_text TEXT NOT NULL, content_hash TEXT NOT NULL UNIQUE, "
        "scope TEXT NOT NULL, project TEXT, status TEXT NOT NULL DEFAULT 'active', "
        "confidence REAL, hrr_vector BLOB, source_refs_json TEXT NOT NULL DEFAULT '[]', "
        "entity_ids_json TEXT NOT NULL DEFAULT '[]', tags_json TEXT NOT NULL DEFAULT '[]', "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS decisions "
        "(decision_id TEXT PRIMARY KEY, decision_text TEXT NOT NULL, rationale TEXT, "
        "project TEXT, owner TEXT, status TEXT NOT NULL DEFAULT 'open', "
        "source_refs_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS open_questions "
        "(question_id TEXT PRIMARY KEY, question_text TEXT NOT NULL, project TEXT, "
        "priority TEXT, status TEXT DEFAULT 'open', source_refs_json TEXT NOT NULL DEFAULT '[]', "
        "next_action TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS audit_log "
        "(audit_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
        "actor TEXT NOT NULL, action TEXT NOT NULL, target_kind TEXT, target_id TEXT, detail_json TEXT)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
    conn.commit()
    conn.close()

    import hermes_memory_core as hmc
    original_get = hmc.get_memory_db

    def _patched(db_path=str(db_path)):
        db = hmc.MemoryDB(db_path)
        db.initialize()
        return db

    hmc.get_memory_db = _patched
    yield db_path
    hmc.get_memory_db = original_get


class TestUpdateMemory_Fact:
    def test_update_fact_text(self, memory_db_path):
        """Updating fact text changes fact_text and updated_at."""
        # Create a fact first
        result = write_memory(
            "fact",
            "Hermes uses SQLite for memory storage",
            source_ref="test://t_50a2eb10/update/test1",
            project="hermes-memory",
        )
        assert result["written"] is True
        fact_id = result["id"]

        # Update the text
        upd = update_memory(fact_id, "fact", text="Hermes uses SQLite with WAL mode")
        assert upd["updated"] is True
        assert upd["changes"]["text"] is True

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT fact_text FROM facts WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        conn.close()
        assert row[0] == "Hermes uses SQLite with WAL mode"

    def test_update_fact_status(self, memory_db_path):
        """Updating fact status changes status column."""
        result = write_memory(
            "fact",
            "Some fact to update",
            source_ref="test://t_50a2eb10/update/test2",
        )
        fact_id = result["id"]

        upd = update_memory(fact_id, "fact", status="superseded")
        assert upd["updated"] is True
        assert upd["changes"]["status"] == "superseded"

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT status FROM facts WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        conn.close()
        assert row[0] == "superseded"

    def test_update_fact_trust_delta(self, memory_db_path):
        """Updating fact trust_delta sets confidence column to the delta value."""
        result = write_memory(
            "fact",
            "Some fact with initial confidence",
            source_ref="test://t_50a2eb10/update/test3",
            confidence=0.7,
        )
        fact_id = result["id"]

        # update_memory stores trust_delta as absolute value in confidence
        upd = update_memory(fact_id, "fact", trust_delta=0.85)
        assert upd["updated"] is True
        assert upd["changes"]["trust_delta"] == 0.85

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT confidence FROM facts WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        conn.close()
        assert row[0] == 0.85

    def test_update_fact_not_found(self, memory_db_path):
        """Update with non-existent fact_id returns not_found."""
        upd = update_memory(
            "fact_deadbeef00000000",
            "fact",
            text="does not exist",
        )
        assert upd["updated"] is False
        assert upd["reason"] == "not_found"

    def test_update_fact_nothing_to_update(self, memory_db_path):
        """Update with no fields provided returns nothing_to_update."""
        result = write_memory(
            "fact",
            "Some fact",
            source_ref="test://t_50a2eb10/update/test4",
        )
        fact_id = result["id"]

        upd = update_memory(fact_id, "fact")
        assert upd["updated"] is False
        assert upd["reason"] == "nothing_to_update"


class TestUpdateMemory_Decision:
    def test_update_decision_text(self, memory_db_path):
        """Updating decision text changes decision_text."""
        result = write_memory(
            "decision",
            "Use Fly.io for deployments",
            source_ref="test://t_50a2eb10/update/dec1",
            owner="hm-developer",
        )
        decision_id = result["id"]

        upd = update_memory(decision_id, "decision", text="Use Railway for deployments")
        assert upd["updated"] is True
        assert upd["changes"]["text"] is True

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT decision_text FROM decisions WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        conn.close()
        assert row[0] == "Use Railway for deployments"

    def test_update_decision_status(self, memory_db_path):
        """Updating decision status changes status."""
        result = write_memory(
            "decision",
            "A decision to update",
            source_ref="test://t_50a2eb10/update/dec2",
        )
        decision_id = result["id"]

        upd = update_memory(decision_id, "decision", status="active")
        assert upd["updated"] is True
        assert upd["changes"]["status"] == "active"

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT status FROM decisions WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        conn.close()
        assert row[0] == "active"


class TestUpdateMemory_OpenQuestion:
    def test_update_open_question_status(self, memory_db_path):
        """Updating open_question status changes status."""
        result = write_memory(
            "open_question",
            "Should we migrate to PostgreSQL?",
            source_ref="test://t_50a2eb10/update/q1",
            priority="high",
        )
        question_id = result["id"]

        upd = update_memory(question_id, "open_question", status="archived")
        assert upd["updated"] is True
        assert upd["changes"]["status"] == "archived"

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT status FROM open_questions WHERE question_id = ?",
            (question_id,),
        ).fetchone()
        conn.close()
        assert row[0] == "archived"


class TestUpdateMemory_Invalid:
    def test_invalid_memory_type(self, memory_db_path):
        """Invalid memory_type returns error."""
        upd = update_memory("some_id", "not_a_type", text="something")
        assert upd["updated"] is False
        assert "invalid_memory_type" in upd["reason"]


class TestFactFeedback_Helpful:
    def test_helpful_increases_confidence(self, memory_db_path):
        """helpful action increases confidence by +0.05."""
        result = write_memory(
            "fact",
            "A test fact for feedback",
            source_ref="test://t_50a2eb10/feedback/helpful1",
            confidence=0.6,
        )
        fact_id = result["id"]

        fb = fact_feedback(fact_id, "helpful")
        assert fb["ok"] is True
        assert fb["action"] == "helpful"
        assert fb["old_confidence"] == 0.6
        assert abs(fb["new_confidence"] - 0.65) < 0.001

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT confidence FROM facts WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        conn.close()
        assert abs(row[0] - 0.65) < 0.001

    def test_helpful_clamps_at_1(self, memory_db_path):
        """helpful action clamps confidence at 1.0."""
        result = write_memory(
            "fact",
            "High confidence fact",
            source_ref="test://t_50a2eb10/feedback/helpful2",
            confidence=0.98,
        )
        fact_id = result["id"]

        fb = fact_feedback(fact_id, "helpful")
        assert fb["ok"] is True
        assert fb["new_confidence"] == 1.0

    def test_helpful_default_confidence(self, memory_db_path):
        """helpful on fact with NULL confidence uses 0.5 as base."""
        result = write_memory(
            "fact",
            "No explicit confidence",
            source_ref="test://t_50a2eb10/feedback/helpful3",
            # no confidence specified — defaults to 0.5 in write_memory
        )
        fact_id = result["id"]

        fb = fact_feedback(fact_id, "helpful")
        assert fb["ok"] is True
        assert fb["old_confidence"] == 0.5
        assert abs(fb["new_confidence"] - 0.55) < 0.001


class TestFactFeedback_Unhelpful:
    def test_unhelpful_decreases_confidence(self, memory_db_path):
        """unhelpful action decreases confidence by -0.10."""
        result = write_memory(
            "fact",
            "A test fact for feedback",
            source_ref="test://t_50a2eb10/feedback/unhelpful1",
            confidence=0.7,
        )
        fact_id = result["id"]

        fb = fact_feedback(fact_id, "unhelpful")
        assert fb["ok"] is True
        assert fb["action"] == "unhelpful"
        assert fb["old_confidence"] == 0.7
        assert abs(fb["new_confidence"] - 0.6) < 0.001

    def test_unhelpful_clamps_at_0(self, memory_db_path):
        """unhelpful action clamps confidence at 0.0."""
        result = write_memory(
            "fact",
            "Low confidence fact",
            source_ref="test://t_50a2eb10/feedback/unhelpful2",
            confidence=0.05,
        )
        fact_id = result["id"]

        fb = fact_feedback(fact_id, "unhelpful")
        assert fb["ok"] is True
        assert fb["new_confidence"] == 0.0

    def test_unhelpful_roundtrip(self, memory_db_path):
        """Multiple feedback calls: helpful then unhelpful."""
        result = write_memory(
            "fact",
            "Fact for roundtrip feedback",
            source_ref="test://t_50a2eb10/feedback/roundtrip",
            confidence=0.5,
        )
        fact_id = result["id"]

        fb1 = fact_feedback(fact_id, "helpful")
        assert abs(fb1["new_confidence"] - 0.55) < 0.001

        fb2 = fact_feedback(fact_id, "unhelpful")
        assert abs(fb2["old_confidence"] - 0.55) < 0.001
        assert abs(fb2["new_confidence"] - 0.45) < 0.001


class TestFactFeedback_Invalid:
    def test_invalid_action_returns_error(self, memory_db_path):
        """Invalid action string returns error."""
        fb = fact_feedback("fact_deadbeef00000000", "bad_action")
        assert fb["ok"] is False
        assert "invalid_action" in fb["reason"]

    def test_not_found_returns_error(self, memory_db_path):
        """Non-existent fact_id returns not_found."""
        fb = fact_feedback("fact_deadbeef00000000", "helpful")
        assert fb["ok"] is False
        assert fb["reason"] == "not_found"


class TestFactFeedback_Audit:
    def test_audit_log_written_on_feedback(self, memory_db_path):
        """Audit_log row written when fact_feedback succeeds."""
        result = write_memory(
            "fact",
            "Fact with audit",
            source_ref="test://t_50a2eb10/feedback/audit",
            confidence=0.5,
        )
        fact_id = result["id"]

        fb = fact_feedback(fact_id, "helpful")
        assert fb["ok"] is True

        conn = sqlite3.connect(str(memory_db_path))
        rows = conn.execute(
            "SELECT actor, action, target_kind, detail_json FROM audit_log "
            "WHERE action = 'fact_feedback' AND target_id = ?",
            (fact_id,),
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        actor, action, target_kind, detail_json = rows[0]
        assert actor == "plugin"
        assert action == "fact_feedback"
        assert target_kind == "fact"
        detail = json.loads(detail_json)
        assert detail["action"] == "helpful"
        assert detail["delta"] == 0.05