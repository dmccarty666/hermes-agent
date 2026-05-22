"""Tests for hermes-memory health endpoints and metrics.

TDD RED phase: tests written BEFORE health/metrics implementation.
Covers Story T-045 (Epic 6.3, Story 6.3.1) acceptance criteria.
Uses real Qdrant, real SQLite, real LMS — no mocking of infrastructure.

Run with: scripts/run_tests.sh tests/integration/memory/test_health.py
"""

from __future__ import annotations

# ── PHASE-1.5 TRIAGE — NEEDS ISOLATED QDRANT ───────────────────────────────────
# Triaged Bucket A by the recovery pass on branch recovery/phase-1-5-restore.
# Tests want their own isolated Qdrant collection and ECONNREFUSE when nothing
# is listening on :6333. The live system's :6333 is NOT to be repurposed —
# see docs/INTEGRATION-TEST-TRIAGE.md for how to spin up a transient test
# instance on a non-conflicting port and unskip.
import os as _phase15_os
import socket as _phase15_socket
import pytest as _phase15_pytest

def _phase15_qdrant_reachable() -> bool:
    host = _phase15_os.environ.get("QDRANT_HOST", "localhost")
    port = int(_phase15_os.environ.get("QDRANT_PORT", "6333"))
    try:
        with _phase15_socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False

if not _phase15_qdrant_reachable():
    _phase15_pytest.skip(
        "needs isolated Qdrant; see docs/INTEGRATION-TEST-TRIAGE.md",
        allow_module_level=True,
    )


import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERMES_HOME = Path.home() / ".hermes"
PROFILE_HOME = Path(os.environ.get("HERMES_HOME", str(HERMES_HOME)))
MEMORY_DIR = PROFILE_HOME / "memory"
METRICS_PATH = MEMORY_DIR / "metrics.json"
HL_DB = MEMORY_DIR / "index" / "memory.sqlite"

# External endpoints (from config)
EMBEDDING_ENDPOINT = "http://192.168.2.105:1235"
LLM_ENDPOINT = "http://192.168.2.105:1234"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path):
    """Create a minimal hermes-local SQLite schema for testing."""
    db = tmp_path / "test_memory.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE IF NOT EXISTS sessions(session_id TEXT PRIMARY KEY, started_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS turns(turn_id TEXT PRIMARY KEY, session_id TEXT, timestamp TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS chunks(chunk_id TEXT PRIMARY KEY, session_id TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS facts(fact_id TEXT PRIMARY KEY, status TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT, created_at TEXT)")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    yield conn, db
    conn.close()


@pytest.fixture
def qdrant_client():
    """Real Qdrant client for health checks."""
    from qdrant_client import QdrantClient
    client = QdrantClient("http://localhost:6333", timeout=3)
    yield client
    # No teardown — shared instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# AC-1: GET /health/sqlite — SQLite connectivity and WAL status
# ---------------------------------------------------------------------------

def test_health_sqlite_returns_ok(temp_db):
    """Given /health/sqlite, then it returns status=ok and WAL mode."""
    from hermes_memory_core.health import check_sqlite

    conn, db_path = temp_db
    result = check_sqlite(db_path)

    assert result["status"] == "ok"
    assert "wal" in result.get("journal_mode", "").lower() or "wal" in result.get("mode", "").lower()
    assert "path" in result


def test_health_sqlite_nonexistent_db():
    """Given /health/sqlite with a non-existent DB, then it returns status=error."""
    from hermes_memory_core.health import check_sqlite

    result = check_sqlite(Path("/nonexistent/path/to/db.sqlite"))
    assert result["status"] == "error"
    assert "message" in result or "error" in result


def test_health_sqlite_real_db():
    """Given the real hermes-local DB, then /health/sqlite returns status=ok."""
    from hermes_memory_core.health import check_sqlite

    if not HL_DB.exists():
        pytest.skip("Real DB not present")
    result = check_sqlite(HL_DB)
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# AC-2: GET /health/qdrant — Qdrant cluster health and collection count
# ---------------------------------------------------------------------------

def test_health_qdrant_returns_ok(qdrant_client):
    """Given /health/qdrant, then it returns status=ok and collection count."""
    from hermes_memory_core.health import check_qdrant

    result = check_qdrant(qdrant_client)

    assert result["status"] == "ok"
    assert "collections_count" in result or "collections" in result
    assert result.get("status") == "ok"


def test_health_qdrant_unreachable():
    """Given Qdrant is unreachable, then /health/qdrant returns status=error."""
    from hermes_memory_core.health import check_qdrant
    from qdrant_client import QdrantClient

    bad_client = QdrantClient("http://localhost:19999", timeout=2)
    result = check_qdrant(bad_client)

    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# AC-3: GET /health/embedding — LMS embedding endpoint reachability and model info
# ---------------------------------------------------------------------------

def test_health_embedding_returns_ok():
    """Given /health/embedding, then it returns status=ok and model info."""
    from hermes_memory_core.health import check_embedding

    result = check_embedding(EMBEDDING_ENDPOINT, timeout=5)
    if result["status"] == "error":
        pytest.skip(f"Embedding endpoint unreachable: {result.get('message', 'unknown error')}")
    assert result["status"] == "ok"
    assert "model" in result or "model_name" in result


def test_health_embedding_unreachable():
    """Given embedding endpoint is unreachable, then /health/embedding returns status=error."""
    from hermes_memory_core.health import check_embedding

    result = check_embedding("http://localhost:19999", timeout=3)
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# AC-4: GET /health/llm — dreamer LLM (Qwen3.6-35B) reachability
# ---------------------------------------------------------------------------

def test_health_llm_returns_ok():
    """Given /health/llm, then it returns status=ok and model name."""
    from hermes_memory_core.health import check_llm

    result = check_llm(LLM_ENDPOINT, timeout=10)

    assert result["status"] == "ok"
    assert "model" in result or "model_name" in result or "model_name" in str(result)


def test_health_llm_unreachable():
    """Given LLM endpoint is unreachable, then /health/llm returns status=error."""
    from hermes_memory_core.health import check_llm

    result = check_llm("http://localhost:19999", timeout=3)
    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# AC-5: GET /health — rolled-up status of all components
# ---------------------------------------------------------------------------

def test_health_rolled_up_returns_all_components(temp_db, qdrant_client):
    """Given the gateway is running, when GET /health is called,
    then a rolled-up status is returned with sub-statuses for each component."""
    from hermes_memory_core.health import health_check

    conn, db_path = temp_db
    result = health_check(
        db_path=db_path,
        qdrant_client=qdrant_client,
        embedding_endpoint=EMBEDDING_ENDPOINT,
        llm_endpoint=LLM_ENDPOINT,
    )

    assert "sqlite" in result
    assert "qdrant" in result
    assert "embedding" in result
    assert "llm" in result
    assert "overall" in result
    # Overall should be "ok" only if all components are ok
    assert result["overall"] in ("ok", "degraded", "error")


def test_health_rolled_up_includes_disk_space(temp_db, qdrant_client):
    """Given GET /health, then disk space check is included."""
    from hermes_memory_core.health import health_check

    conn, db_path = temp_db
    result = health_check(
        db_path=db_path,
        qdrant_client=qdrant_client,
        embedding_endpoint=EMBEDDING_ENDPOINT,
        llm_endpoint=LLM_ENDPOINT,
    )

    assert "disk" in result
    assert "free_bytes" in result["disk"] or "free_gb" in result["disk"]


# ---------------------------------------------------------------------------
# AC-6: metrics.json updated after capture/index/dream batch
# ---------------------------------------------------------------------------

def test_metrics_writer_updates_all_gauges(temp_db):
    """Given metrics.json, then it includes all required gauges:
    captured_turns_24h, chunks_indexed_24h, chunks_pending, facts_total,
    facts_active, qdrant_points, last_dream_run_at, last_dream_status, redactions_24h."""
    from hermes_memory_core.metrics import MetricsWriter

    conn, db_path = temp_db

    with tempfile.TemporaryDirectory() as td:
        mem_dir = Path(td) / "memory"
        mem_dir.mkdir()
        (mem_dir / "index").mkdir()
        metrics_path = mem_dir / "metrics.json"

        writer = MetricsWriter(memory_dir=mem_dir, db_path=db_path)
        writer.update()

        assert metrics_path.exists()
        data = json.loads(metrics_path.read_text())

        required_gauges = [
            "captured_turns_24h",
            "chunks_indexed_24h",
            "chunks_pending",
            "facts_total",
            "facts_active",
            "qdrant_points",
            "last_dream_run_at",
            "last_dream_status",
            "redactions_24h",
        ]
        for gauge in required_gauges:
            assert gauge in data, f"Missing gauge: {gauge}"


def test_metrics_writer_idempotent(temp_db):
    """Given metrics.json is rewritten multiple times, then it remains valid JSON."""
    from hermes_memory_core.metrics import MetricsWriter

    conn, db_path = temp_db

    with tempfile.TemporaryDirectory() as td:
        mem_dir = Path(td) / "memory"
        mem_dir.mkdir()
        (mem_dir / "index").mkdir()
        metrics_path = mem_dir / "metrics.json"

        writer = MetricsWriter(memory_dir=mem_dir, db_path=db_path)
        writer.update()
        writer.update()
        writer.update()

        data = json.loads(metrics_path.read_text())
        assert isinstance(data, dict)
        assert len(data) > 0


def test_metrics_writer_creates_directory_if_missing(temp_db):
    """Given metrics.json dir does not exist, then it is created."""
    from hermes_memory_core.metrics import MetricsWriter

    conn, db_path = temp_db

    with tempfile.TemporaryDirectory() as td:
        mem_dir = Path(td) / "memory"
        # intentionally don't create mem_dir
        metrics_path = mem_dir / "metrics.json"

        writer = MetricsWriter(memory_dir=mem_dir, db_path=db_path)
        writer.update()

        assert metrics_path.exists()


def test_metrics_writer_last_dream_status_defaults_to_unknown(temp_db):
    """Given no dream runs exist, then last_dream_status defaults to 'unknown'."""
    from hermes_memory_core.metrics import MetricsWriter

    conn, db_path = temp_db

    with tempfile.TemporaryDirectory() as td:
        mem_dir = Path(td) / "memory"
        mem_dir.mkdir()
        (mem_dir / "index").mkdir()

        writer = MetricsWriter(memory_dir=mem_dir, db_path=db_path)
        writer.update()

        data = json.loads((mem_dir / "metrics.json").read_text())
        assert data["last_dream_status"] == "unknown"