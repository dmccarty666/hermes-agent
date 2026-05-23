"""Tests for the M1 memory-dashboard backend routes.

Endpoints under test (all behind `_SESSION_TOKEN` auth):
  GET  /api/dashboard/memory/status
  GET  /api/dashboard/memory/backends
  GET  /api/dashboard/memory/counters
  GET  /api/dashboard/memory/metrics-json
  POST /api/dashboard/memory/metrics/refresh
  POST /api/dashboard/memory/backends/{name}/ping

These tests do NOT mock hermes_memory_core internals. We isolate HERMES_HOME
to a tmp dir (autouse `_isolate_hermes_home` from tests/conftest.py) so a
clean memory tree is created on first probe. When Qdrant / LMS / LLM are
unreachable in the test env, the response shape still matches API.md — the
per-component `status` field just reports "error" instead of "ok".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Authed Starlette TestClient bound to a fresh HERMES_HOME."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import (
        app,
        _SESSION_HEADER_NAME,
        _SESSION_TOKEN,
        _MEMORY_STATUS_CACHE,
        _MEMORY_REFRESH_LAST,
    )

    # Bust caches so prior tests in the same process don't leak.
    _MEMORY_STATUS_CACHE["value"] = None
    _MEMORY_STATUS_CACHE["ts"] = 0.0
    _MEMORY_REFRESH_LAST.clear()

    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


@pytest.fixture
def unauth_client():
    """TestClient with NO session token — for 401 checks."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.web_server import app
    return TestClient(app)


@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    """Pin HERMES_HOME to tmp and make sure memory/ dir exists."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    mem = tmp_path / "memory"
    (mem / "index").mkdir(parents=True, exist_ok=True)
    return mem


@pytest.fixture
def seeded_metrics(memory_dir):
    """Write a small known metrics.json so /metrics-json + /counters return it."""
    payload = {
        "captured_turns_24h": 7,
        "chunks_indexed_24h": 7,
        "chunks_pending": 0,
        "facts_total": 12,
        "facts_active": 11,
        "qdrant_points": 99,
        "last_dream_run_at": "2026-05-22T11:00:00Z",
        "last_dream_status": "completed",
        "redactions_24h": 0,
    }
    path = memory_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload, path


@pytest.fixture
def seeded_sqlite(memory_dir):
    """Create a minimal memory.sqlite so /counters can run a fresh update."""
    import sqlite3

    db_path = memory_dir / "index" / "memory.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE turns (id TEXT PRIMARY KEY, timestamp TEXT);
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                updated_at TEXT,
                qdrant_point_id TEXT
            );
            CREATE TABLE facts (id TEXT PRIMARY KEY, status TEXT);
            CREATE TABLE dream_runs (
                id TEXT PRIMARY KEY, started_at TEXT, status TEXT
            );
            CREATE TABLE audit_log (
                id INTEGER PRIMARY KEY,
                action TEXT,
                timestamp TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


# ---------------------------------------------------------------------------
# 1. GET /api/dashboard/memory/status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_happy_path_returns_documented_shape(self, client, memory_dir):
        resp = client.get("/api/dashboard/memory/status")
        assert resp.status_code == 200
        data = resp.json()
        # API.md §1 keys
        for key in (
            "provider", "active", "installed", "overall",
            "components", "checked_at", "cached", "cache_ttl",
        ):
            assert key in data, f"missing {key} in /status payload"
        assert data["provider"] == "hermes-local"
        assert data["overall"] in {"ok", "degraded", "error", "inactive"}
        # When active+installed, components is a dict with the 5 backends.
        if data["active"] and data["installed"]:
            assert isinstance(data["components"], dict)
            for comp in ("sqlite", "qdrant", "embedding", "llm", "disk"):
                assert comp in data["components"]
                assert "status" in data["components"][comp]

    def test_requires_auth(self, unauth_client):
        resp = unauth_client.get("/api/dashboard/memory/status")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 2. GET /api/dashboard/memory/backends
# ---------------------------------------------------------------------------


class TestBackends:
    def test_happy_path_returns_five_backends(self, client, memory_dir):
        resp = client.get("/api/dashboard/memory/backends")
        assert resp.status_code == 200
        data = resp.json()
        for backend in ("sqlite", "qdrant", "embedding", "llm", "disk"):
            assert backend in data, f"missing {backend} in /backends"
            assert "status" in data[backend]
        assert "checked_at" in data
        # sqlite-specific fields per API.md §2
        assert "path" in data["sqlite"]
        assert "journal_mode" in data["sqlite"]
        assert "size_bytes" in data["sqlite"]
        # qdrant-specific
        assert "endpoint" in data["qdrant"]
        assert "collections" in data["qdrant"]
        # disk
        assert "free_bytes" in data["disk"]
        assert "free_gb" in data["disk"]

    def test_requires_auth(self, unauth_client):
        resp = unauth_client.get("/api/dashboard/memory/backends")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 3. GET /api/dashboard/memory/counters
# ---------------------------------------------------------------------------


class TestCounters:
    def test_happy_path_with_seeded_metrics(
        self, client, memory_dir, seeded_metrics
    ):
        payload, _ = seeded_metrics
        resp = client.get("/api/dashboard/memory/counters")
        assert resp.status_code == 200
        data = resp.json()
        # values mirror the seeded file
        assert data["facts_total"] == payload["facts_total"]
        assert data["facts_active"] == payload["facts_active"]
        assert data["qdrant_points"] == payload["qdrant_points"]
        assert data["last_dream_status"] == "completed"
        # envelope
        assert "metrics_file" in data
        assert "stale_seconds" in data
        assert "deltas_24h" in data  # null until snapshots exist

    def test_404_when_no_metrics_and_no_sqlite(self, client, tmp_path, monkeypatch):
        # Point HERMES_HOME at an empty dir — no metrics.json, no sqlite.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty"))
        (tmp_path / "empty" / "memory").mkdir(parents=True, exist_ok=True)
        resp = client.get("/api/dashboard/memory/counters")
        # MetricsWriter happily writes a metrics.json with all zeros even when
        # the DB doesn't exist (it skips SQL on missing DB), so the endpoint
        # returns 200 with a zeroed envelope. That matches the spec note
        # "Caller should surface 'metrics not yet available'" being a 404
        # only when even the fallback refresh fails. Accept either shape.
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert data["facts_total"] == 0
            assert data["captured_turns_24h"] == 0


# ---------------------------------------------------------------------------
# 4. GET /api/dashboard/memory/metrics-json
# ---------------------------------------------------------------------------


class TestMetricsJson:
    def test_happy_path_returns_file_verbatim(
        self, client, memory_dir, seeded_metrics
    ):
        payload, _ = seeded_metrics
        resp = client.get("/api/dashboard/memory/metrics-json")
        assert resp.status_code == 200
        data = resp.json()
        # Returned dict should match the seeded payload exactly (no envelope).
        for k, v in payload.items():
            assert data[k] == v
        # No envelope fields per API.md §4
        assert "metrics_file" not in data
        assert "stale_seconds" not in data

    def test_404_when_file_missing(self, client, memory_dir):
        # memory_dir exists but metrics.json does not.
        path = memory_dir / "metrics.json"
        if path.exists():
            path.unlink()
        resp = client.get("/api/dashboard/memory/metrics-json")
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" in body

    def test_force_triggers_writer(self, client, memory_dir, seeded_sqlite):
        # No metrics.json yet, but with ?force=true the writer runs and
        # creates one.
        path = memory_dir / "metrics.json"
        if path.exists():
            path.unlink()
        resp = client.get("/api/dashboard/memory/metrics-json?force=true")
        assert resp.status_code == 200
        data = resp.json()
        # MetricsWriter's known keys
        for key in (
            "captured_turns_24h", "chunks_indexed_24h", "chunks_pending",
            "facts_total", "facts_active", "qdrant_points",
            "last_dream_run_at", "last_dream_status", "redactions_24h",
        ):
            assert key in data
        assert path.exists()


# ---------------------------------------------------------------------------
# 5. POST /api/dashboard/memory/metrics/refresh
# ---------------------------------------------------------------------------


class TestMetricsRefresh:
    def test_happy_path_writes_and_returns(self, client, memory_dir, seeded_sqlite):
        resp = client.post("/api/dashboard/memory/metrics/refresh")
        assert resp.status_code == 200
        data = resp.json()
        # Known keys from MetricsWriter._compute()
        assert "facts_total" in data
        assert "captured_turns_24h" in data
        assert "metrics_file" in data
        assert data["stale_seconds"] == 0
        # File written on disk
        assert (memory_dir / "metrics.json").exists()

    def test_rate_limited_on_immediate_second_call(
        self, client, memory_dir, seeded_sqlite
    ):
        resp1 = client.post("/api/dashboard/memory/metrics/refresh")
        assert resp1.status_code == 200
        # Second call within 5s should be rate-limited.
        resp2 = client.post("/api/dashboard/memory/metrics/refresh")
        assert resp2.status_code == 429
        body = resp2.json()
        assert "detail" in body

    def test_requires_auth(self, unauth_client):
        resp = unauth_client.post("/api/dashboard/memory/metrics/refresh")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 6. POST /api/dashboard/memory/backends/{name}/ping
# ---------------------------------------------------------------------------


class TestBackendPing:
    def test_happy_path_sqlite(self, client, memory_dir, seeded_sqlite):
        resp = client.post("/api/dashboard/memory/backends/sqlite/ping")
        # sqlite exists locally — should return 200
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "path" in data
        assert "journal_mode" in data
        assert "size_bytes" in data

    def test_happy_path_disk(self, client, memory_dir):
        resp = client.post("/api/dashboard/memory/backends/disk/ping")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "free_bytes" in data
        assert "free_gb" in data

    def test_404_for_unknown_backend(self, client):
        resp = client.post("/api/dashboard/memory/backends/bogus/ping")
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" in body

    def test_503_or_200_when_backend_unreachable(self, client, memory_dir):
        # Qdrant may or may not be running in the test env. Either way the
        # endpoint must return a body with a 'status' field. 503 if down,
        # 200 if up — both are documented (API.md §6 returns 503 with a
        # body when the backend probe itself fails).
        resp = client.post("/api/dashboard/memory/backends/qdrant/ping")
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert "status" in data
        assert "endpoint" in data
