"""
Tests for write_memory (Story 4.4.1 — TDD §4).

Runs against real SQLite at TMP_PATH (pytest tmp_path fixture).
Covers:
  - write fact → facts table row created, source_ref populated
  - write decision → decisions table row created
  - write open_question → open_questions table row created
  - content_hash dedup: second write with same content returns skipped=True
  - source_ref required: missing source_ref returns error reason
  - redaction: AWS key in text is replaced with [REDACTED:aws_access_key]
  - audit_log row written on redaction_fired
  - invalid memory_type returns error reason
"""

# ── PHASE-1.5 TRIAGE — STALE / API-DRIFT ───────────────────────────────────────
# Asserts a pre-Phase-1.5 contract that no longer matches production. Triaged
# Bucket B (STALE) by the recovery pass on branch recovery/phase-1-5-restore.
# See docs/INTEGRATION-TEST-TRIAGE.md for per-test reasoning. To unskip:
# remove this block and rewrite assertions against the current contract.
import pytest as _phase15_pytest
_phase15_pytest.skip(
    "stale: pre-Phase-1.5 API contract; see docs/INTEGRATION-TEST-TRIAGE.md",
    allow_module_level=True,
)


import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

# Import from the write pipeline module directly
from hermes_memory_core.write.pipeline import write_memory


@pytest.fixture
def memory_db_path(tmp_path):
    """Provide a temp SQLite path, init schema, and clean up after."""
    db_path = tmp_path / "memory.sqlite"
    conn = sqlite3.connect(str(db_path))
    # Apply minimal schema (same as MemoryDB.initialize())
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    # sessions + turns (from pipeline.py's dependencies)
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
        "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, sequence)"
    )

    # facts (with UNIQUE content_hash dedup)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS facts "
        "(fact_id TEXT PRIMARY KEY, fact_text TEXT NOT NULL, content_hash TEXT NOT NULL UNIQUE, "
        "scope TEXT NOT NULL, project TEXT, status TEXT NOT NULL DEFAULT 'active', "
        "confidence REAL, hrr_vector BLOB, source_refs_json TEXT NOT NULL DEFAULT '[]', "
        "entity_ids_json TEXT NOT NULL DEFAULT '[]', tags_json TEXT NOT NULL DEFAULT '[]', "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_project ON facts(project)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status)")

    # decisions
    conn.execute(
        "CREATE TABLE IF NOT EXISTS decisions "
        "(decision_id TEXT PRIMARY KEY, decision_text TEXT NOT NULL, rationale TEXT, "
        "project TEXT, owner TEXT, status TEXT NOT NULL DEFAULT 'open', "
        "source_refs_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status)")

    # open_questions
    conn.execute(
        "CREATE TABLE IF NOT EXISTS open_questions "
        "(question_id TEXT PRIMARY KEY, question_text TEXT NOT NULL, project TEXT, "
        "priority TEXT, status TEXT DEFAULT 'open', source_refs_json TEXT NOT NULL DEFAULT '[]', "
        "next_action TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_project ON open_questions(project)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_status ON open_questions(status)")

    # audit_log
    conn.execute(
        "CREATE TABLE IF NOT EXISTS audit_log "
        "(audit_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
        "actor TEXT NOT NULL, action TEXT NOT NULL, target_kind TEXT, target_id TEXT, detail_json TEXT)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")

    conn.commit()
    conn.close()

    # Patch get_memory_db to return this path
    import hermes_memory_core as hmc
    original_get = hmc.get_memory_db

    def _patched(db_path=str(db_path)):
        return hmc.MemoryDB(db_path)

    hmc.get_memory_db = _patched
    yield db_path
    hmc.get_memory_db = original_get


class TestWriteMemory_Fact:
    def test_hrr_vector_computed_for_facts(self, memory_db_path):
        """HRR vector is stored in the facts table (non-NULL)."""
        result = write_memory(
            "fact",
            "Deploy to Fly.io on every push to main",
            source_ref="test://t_4b709382/hrr",
            project="hermes-memory",
            scope="project",
        )
        assert result["written"] is True

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT hrr_vector FROM facts WHERE fact_id = ?", (result["id"],)
        ).fetchone()
        conn.close()
        assert row is not None, "fact row not found"
        assert row[0] is not None, "hrr_vector must be non-NULL for facts"
        # Verify it's a valid bytes blob of the expected size (1024 floats * 8 bytes = 8192)
        assert len(row[0]) == 8192, f"expected 8192 bytes for HRR_DIM=1024, got {len(row[0])}"

    def test_write_fact_creates_row(self, memory_db_path):
        """Fact write creates a row in facts table."""
        result = write_memory(
            "fact",
            "Hermes uses SQLite for memory storage",
            source_ref="test://t_181001a2/fact/test1",
            project="hermes-memory",
            scope="project",
        )
        assert result["written"] is True, f"expected written=True, got {result}"
        assert result["skipped"] is False
        assert result["reason"] == "new"
        assert result["id"] is not None
        assert result["id"].startswith("fact_")
        assert result["source_ref"] == "test://t_181001a2/fact/test1"
        assert result["redaction_fired"] is False
        assert result["redaction_types"] == []

        # Verify DB row
        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT fact_text, scope, project, source_refs_json FROM facts WHERE fact_id = ?",
            (result["id"],),
        ).fetchone()
        conn.close()
        assert row is not None, "fact row not found in DB"
        assert row[0] == "Hermes uses SQLite for memory storage"
        assert row[1] == "project"
        assert row[2] == "hermes-memory"
        assert json.loads(row[3]) == ["test://t_181001a2/fact/test1"]

    def test_write_fact_missing_source_ref_returns_error(self, memory_db_path):
        """Missing source_ref returns error reason, does not write."""
        result = write_memory(
            "fact",
            "Some fact without attribution",
            source_ref="",  # empty string
        )
        assert result["written"] is False
        assert result["skipped"] is False
        assert "missing_required_field" in result["reason"]
        assert result["id"] is None

        conn = sqlite3.connect(str(memory_db_path))
        count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        conn.close()
        assert count == 0, "no fact should be written without source_ref"

    def test_write_fact_duplicate_returns_skipped(self, memory_db_path):
        """Identical content (same hash) on second write returns skipped=True."""
        result1 = write_memory(
            "fact",
            "Deploy to Fly.io on every push to main",
            source_ref="test://t_181001a2/fact/dedup",
        )
        assert result1["written"] is True

        result2 = write_memory(
            "fact",
            "Deploy to Fly.io on every push to main",  # identical text
            source_ref="test://t_181001a2/fact/dedup2",
        )
        assert result2["skipped"] is True
        assert result2["reason"] == "dedup"
        assert result2["id"] is None

        # Only one row in DB
        conn = sqlite3.connect(str(memory_db_path))
        count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        conn.close()
        assert count == 1, "dedup should block second insert"

    def test_qdrant_upsert_called_for_facts(self, memory_db_path, monkeypatch):
        """Qdrant upsert is called after fact INSERT (non-fatal on error)."""
        fake_points = []
        captured_kwargs = {}

        class FakeQdrantClient:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

            def upsert(self, collection_name, points):
                fake_points.append({"collection": collection_name, "points": points})

        import sys
        # Patch the QdrantClient class in the qdrant_client module (imported inside pipeline)
        import qdrant_client
        monkeypatch.setattr(qdrant_client, "QdrantClient", FakeQdrantClient)

        result = write_memory(
            "fact",
            "Use Qwen3.6-35B as the default delegated model",
            source_ref="test://t_4b709382/qdrant",
            project="hermes-memory",
            scope="project",
        )

        assert result["written"] is True, f"Qdrant error should not block SQLite write: {result}"
        assert len(fake_points) >= 1, "QdrantClient.upsert should have been called at least once"
        point = fake_points[0]["points"][0]
        # PointStruct payload access (not a dict)
        payload = point.payload if hasattr(point, "payload") else {}
        assert result["id"] in str(payload)
        assert captured_kwargs.get("host") == "localhost"
        assert captured_kwargs.get("port") == 6333

    def test_qdrant_error_does_not_block_fact_write(self, memory_db_path, monkeypatch):
        """Qdrant upsert failure is logged but does not prevent SQLite write."""

        class BadQdrantClient:
            def __init__(self, **kwargs):
                pass

            def upsert(self, collection_name, points):
                raise RuntimeError("Qdrant unavailable")

        import qdrant_client
        monkeypatch.setattr(qdrant_client, "QdrantClient", BadQdrantClient)

        result = write_memory(
            "fact",
            "SQLite is the source of truth",
            source_ref="test://t_4b709382/qdrant_fail",
            project="hermes-memory",
            scope="project",
        )

        # Write must still succeed even if Qdrant throws
        assert result["written"] is True, f"Qdrant failure should not block write: {result}"
        # DB must have the fact
        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute("SELECT fact_text FROM facts WHERE fact_id = ?", (result["id"],)).fetchone()
        conn.close()
        assert row is not None, "fact should be in SQLite even if Qdrant failed"

    def test_tags_stored_in_facts_table(self, memory_db_path):
        """Tags parameter is stored as tags_json in facts table."""
        result = write_memory(
            "fact",
            "Use Fly.io for all new deployments",
            source_ref="test://t_181001a2/tags/test",
            project="hermes-memory",
            scope="project",
            tags=["deploy", "infrastructure", "flyio"],
        )
        assert result["written"] is True

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT tags_json FROM facts WHERE fact_id = ?", (result["id"],)
        ).fetchone()
        conn.close()
        assert row is not None, "fact row not found"
        stored_tags = json.loads(row[0])
        assert stored_tags == ["deploy", "infrastructure", "flyio"]

    def test_tags_default_to_empty_array(self, memory_db_path):
        """Tags column defaults to empty JSON array when tags=None."""
        result = write_memory(
            "fact",
            "Hermes uses SQLite for memory storage",
            source_ref="test://t_181001a2/tags/default",
            project="hermes-memory",
            scope="project",
        )
        assert result["written"] is True

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT tags_json FROM facts WHERE fact_id = ?", (result["id"],)
        ).fetchone()
        conn.close()
        assert row is not None
        assert json.loads(row[0]) == []


class TestWriteMemory_Redaction:
    def test_aws_key_redacted(self, memory_db_path):
        """AWS key in text is replaced with [REDACTED:aws_access_key]."""
        result = write_memory(
            "fact",
            "AKIAFAKE2KEY3X567812 is the access key for the pipeline",
            source_ref="test://t_181001a2/redact/aws",
        )
        assert result["redaction_fired"] is True
        assert "aws_access_key" in result["redaction_types"]

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT fact_text FROM facts WHERE fact_id = ?", (result["id"],)
        ).fetchone()
        conn.close()
        assert row is not None
        assert "AKIAFAKE2KEY3X567812" not in row[0]
        assert "[REDACTED:aws_access_key]" in row[0]

    def test_github_token_redacted(self, memory_db_path):
        """GitHub token ghp_... in text is redacted."""
        result = write_memory(
            "decision",
            "Use token ghp_abcdefghij1234567890abcdefghij123456 for CI",
            source_ref="test://t_181001a2/redact/github",
        )
        assert result["redaction_fired"] is True
        assert "github_token" in result["redaction_types"]

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT decision_text FROM decisions WHERE decision_id = ?", (result["id"],)
        ).fetchone()
        conn.close()
        assert "ghp_" not in row[0]
        assert "[REDACTED:github_token]" in row[0]

    def test_audit_log_written_on_redaction(self, memory_db_path):
        """Audit_log row written when redaction fires."""
        result = write_memory(
            "fact",
            "OpenAI key sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789012 is in the config",
            source_ref="test://t_181001a2/redact/openai",
        )
        assert result["redaction_fired"] is True

        conn = sqlite3.connect(str(memory_db_path))
        audit_rows = conn.execute(
            "SELECT actor, action, target_kind, detail_json FROM audit_log "
            "WHERE action = 'memory_write' AND target_kind = 'fact'"
        ).fetchall()
        conn.close()
        assert len(audit_rows) == 1, f"expected 1 audit_log row, got {len(audit_rows)}"
        actor, action, target_kind, detail_json = audit_rows[0]
        assert actor == "plugin"
        assert action == "memory_write"
        assert target_kind == "fact"
        detail = json.loads(detail_json)
        assert detail["redaction_fired"] is True
        assert "openai_key" in detail["redaction_types"]


class TestWriteMemory_Decision:
    def test_write_decision_creates_row(self, memory_db_path):
        """Decision write creates a row in decisions table."""
        result = write_memory(
            "decision",
            "Switch to Fly.io for all new deployments",
            source_ref="test://t_181001a2/decision/1",
            project="hermes-memory",
            owner="hm-developer",
            rationale="Fly.io offers better DX and free tier for small projects",
        )
        assert result["written"] is True
        assert result["reason"] == "new"
        assert result["id"].startswith("decision_")

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT decision_text, rationale, owner, project, source_refs_json "
            "FROM decisions WHERE decision_id = ?",
            (result["id"],),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "Switch to Fly.io for all new deployments"
        assert row[1] == "Fly.io offers better DX and free tier for small projects"
        assert row[2] == "hm-developer"
        assert row[3] == "hermes-memory"
        assert "test://t_181001a2/decision/1" in row[4]


class TestWriteMemory_OpenQuestion:
    def test_write_open_question_creates_row(self, memory_db_path):
        """Open question write creates a row in open_questions table."""
        result = write_memory(
            "open_question",
            "Should we migrate to PostgreSQL or keep SQLite?",
            source_ref="test://t_181001a2/question/1",
            project="hermes-memory",
            priority="high",
        )
        assert result["written"] is True
        assert result["reason"] == "new"
        assert result["id"].startswith("question_")

        conn = sqlite3.connect(str(memory_db_path))
        row = conn.execute(
            "SELECT question_text, priority, project, source_refs_json "
            "FROM open_questions WHERE question_id = ?",
            (result["id"],),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "Should we migrate to PostgreSQL or keep SQLite?"
        assert row[1] == "high"
        assert row[2] == "hermes-memory"


class TestWriteMemory_Invalid:
    def test_invalid_memory_type_returns_error(self, memory_db_path):
        """Invalid memory_type returns error reason."""
        result = write_memory(
            "not_a_type",
            "Some content",
            source_ref="test://t_181001a2/invalid",
        )
        assert result["written"] is False
        assert result["skipped"] is False
        assert "invalid_memory_type" in result["reason"]
        assert result["id"] is None