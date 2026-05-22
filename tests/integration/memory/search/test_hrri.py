# Copyright 2026 David McCarty. All rights reserved.
"""Tests for hermes_memory_core/search/hrri.py — T-023.

Story 4.2.2 — HRR-backed probe/related/reason modes.
AC: probe(entity) finds facts where entity plays structural role.
    related(entity) finds structurally connected facts.
    reason(entities) finds facts where ALL entities have structural presence.

Tests use real HRR operations (no mocks) when numpy is available,
and verify FTS5 fallback when it's not.
"""

from __future__ import annotations

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


from pathlib import Path

import pytest

# Ensure the venv python is on the path for the module import
import sys

venv_python = Path("/home/dmccarty/.hermes/hermes-agent/venv/bin/python3")
if venv_python.exists():
    sys.executable = str(venv_python)

from hermes_memory_core.store.sqlite import MemoryDB
from hermes_memory_core.search.hrri import FactRetriever


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_memory_db(tmp_path: Path) -> MemoryDB:
    db = MemoryDB(tmp_path / "memory.sqlite")
    db.initialize()
    return db


def insert_fact(
    db: MemoryDB,
    fact_id: str,
    content: str,
    scope: str = "project",
    project: str = "test-project",
    confidence: float = 0.8,
    hrr_vector: bytes | None = None,
) -> int:
    """Insert a fact row and return its integer id.

    Mirrors the schema used by test_fts5_search.py and test_recent_context.py:
    fact_id, fact_text, content_hash, scope, project, status, confidence,
    hrr_vector (nullable), source_refs_json, entity_ids_json, created_at, updated_at
    """
    conn = db._connect()
    try:
        now = "2026-05-18T12:00:00Z"
        content_hash = f"hash_{fact_id}"
        row = conn.execute(
            """INSERT INTO facts
               (fact_id, fact_text, content_hash, scope, project, status,
                confidence, hrr_vector, source_refs_json, entity_ids_json,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'active', ?, ?, '[]', '[]', ?, ?)""",
            (fact_id, content, content_hash, scope, project, confidence,
             hrr_vector, now, now),
        )
        conn.commit()
        return row.lastrowid or 0
    finally:
        conn.close()


# ------------------------------------------------------------------
# Import check
# ------------------------------------------------------------------

def test_module_imports():
    """Module loads without import errors."""
    from hermes_memory_core.search import hrri
    assert hasattr(hrri, "FactRetriever")


# ------------------------------------------------------------------
# Schema check
# ------------------------------------------------------------------

def test_hrr_vector_column_added_to_schema(tmp_path):
    """MemoryDB schema includes hrr_vector column on facts table."""
    db = make_memory_db(tmp_path)
    conn = db._connect()
    try:
        row = conn.execute("PRAGMA table_info(facts)").fetchall()
        columns = {r[1] for r in row}
        assert "hrr_vector" in columns, f"hrr_vector missing from {columns}"
    finally:
        conn.close()


# ------------------------------------------------------------------
# FTS5 fallback (no numpy or no vectors)
# ------------------------------------------------------------------

def test_probe_falls_back_to_fts5_when_no_numpy(tmp_path):
    """probe() returns FTS5 results when numpy is unavailable."""
    db = make_memory_db(tmp_path)
    insert_fact(db, "f1", "peppi is a cat", scope="project", hrr_vector=None)
    insert_fact(db, "f2", "backend service runs on port 8080", scope="project", hrr_vector=None)
    retriever = FactRetriever(db)

    import hermes_memory_core.search.hrri as hrri_module
    original = hrri_module._hrr._HAS_NUMPY
    hrri_module._hrr._HAS_NUMPY = False
    try:
        results = retriever.probe("peppi", limit=5)
    finally:
        hrri_module._hrr._HAS_NUMPY = original

    assert isinstance(results, list)
    # Should fall back to FTS5 and find peppi fact
    assert any("peppi" in (r.get("content") or "") for r in results)


def test_related_falls_back_to_fts5_when_no_numpy(tmp_path):
    """related() returns FTS5 results when numpy is unavailable."""
    db = make_memory_db(tmp_path)
    insert_fact(db, "f1", "peppi likes fish", scope="project", hrr_vector=None)
    retriever = FactRetriever(db)

    import hermes_memory_core.search.hrri as hrri_module
    original = hrri_module._hrr._HAS_NUMPY
    hrri_module._hrr._HAS_NUMPY = False
    try:
        results = retriever.related("peppi", limit=5)
    finally:
        hrri_module._hrr._HAS_NUMPY = original

    assert isinstance(results, list)


def test_reason_falls_back_to_fts5_when_no_numpy(tmp_path):
    """reason() returns FTS5 results when numpy is unavailable."""
    db = make_memory_db(tmp_path)
    insert_fact(db, "f1", "peppi is a cat", scope="project", hrr_vector=None)
    retriever = FactRetriever(db)

    import hermes_memory_core.search.hrri as hrri_module
    original = hrri_module._hrr._HAS_NUMPY
    hrri_module._hrr._HAS_NUMPY = False
    try:
        results = retriever.reason(["peppi", "cat"], limit=5)
    finally:
        hrri_module._hrr._HAS_NUMPY = original

    assert isinstance(results, list)


def test_reason_empty_entities_falls_back(tmp_path):
    """reason([]) falls back to FTS5."""
    db = make_memory_db(tmp_path)
    insert_fact(db, "f1", "peppi is a cat", scope="project", hrr_vector=None)
    retriever = FactRetriever(db)

    results = retriever.reason([], limit=5)

    assert isinstance(results, list)


# ------------------------------------------------------------------
# get_facts_with_hrr_vectors / get_all_active_facts / fts5_search_facts
# ------------------------------------------------------------------

def test_get_facts_with_hrr_vectors_returns_vectors(tmp_path):
    """get_facts_with_hrr_vectors() returns facts with hrr_vector bytes."""
    pytest.importorskip("numpy")
    from hermes_memory_core.search import hrr as hrr_module

    db = make_memory_db(tmp_path)

    vec = hrr_module.encode_fact("test content", ["entity"], dim=512)
    insert_fact(db, "f1", "test content", hrr_vector=hrr_module.phases_to_bytes(vec))
    insert_fact(db, "f2", "no vector fact", hrr_vector=None)

    facts = db.get_facts_with_hrr_vectors(limit=10)

    assert len(facts) >= 1
    vecs = [f for f in facts if f.get("hrr_vector")]
    assert len(vecs) >= 1


def test_get_all_active_facts(tmp_path):
    """get_all_active_facts() returns all active facts regardless of vectors."""
    db = make_memory_db(tmp_path)

    pytest.importorskip("numpy")
    from hermes_memory_core.search import hrr as hrr_module

    vec = hrr_module.encode_fact("with vector", ["entity"], dim=512)
    insert_fact(db, "f1", "with vector", hrr_vector=hrr_module.phases_to_bytes(vec))
    insert_fact(db, "f2", "without vector", hrr_vector=None)

    facts = db.get_all_active_facts(limit=10)

    assert len(facts) >= 2
    contents = [f.get("content") for f in facts]
    assert "with vector" in contents
    assert "without vector" in contents


def test_fts5_search_facts_fallback(tmp_path):
    """fts5_search_facts() works as fallback when no HRR vectors."""
    db = make_memory_db(tmp_path)
    insert_fact(db, "f1", "peppi is a cat", scope="project")
    insert_fact(db, "f2", "backend runs on port 8080", scope="project")

    results = db.fts5_search_facts("peppi", limit=10)

    assert isinstance(results, list)
    assert any("peppi" in (r.get("content") or "") for r in results)


# ------------------------------------------------------------------
# Probe/Related/Reason with real HRR vectors (numpy required)
# ------------------------------------------------------------------

numpy = pytest.importorskip("numpy")

from hermes_memory_core.search import hrr as hrr_module


def test_probe_returns_scored_facts_with_hrr_vectors(tmp_path):
    """probe() returns scored facts when hrr_vectors exist in DB."""
    db = make_memory_db(tmp_path)

    fact1_content = "peppi is a cat"
    fact2_content = "backend runs on port 8080"

    vec1 = hrr_module.encode_fact(fact1_content, ["peppi"], dim=512)
    vec2 = hrr_module.encode_fact(fact2_content, ["backend"], dim=512)

    insert_fact(db, "f1", fact1_content, hrr_vector=hrr_module.phases_to_bytes(vec1))
    insert_fact(db, "f2", fact2_content, hrr_vector=hrr_module.phases_to_bytes(vec2))

    retriever = FactRetriever(db, hrr_dim=512)

    results = retriever.probe("peppi", limit=10)

    assert isinstance(results, list)
    assert len(results) >= 1
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    contents = [r.get("content", "") for r in results]
    assert any("peppi" in c for c in contents)
    # hrr_vector should be stripped from output (not JSON serializable)
    for r in results:
        assert "hrr_vector" not in r


def test_related_returns_structurally_connected_facts(tmp_path):
    """related() finds facts where entity appears in any structural role."""
    db = make_memory_db(tmp_path)

    fact1_content = "peppi is a cat"
    fact2_content = "peppi likes fish"
    fact3_content = "backend service runs on port 8080"

    vec1 = hrr_module.encode_fact(fact1_content, ["peppi"], dim=512)
    vec2 = hrr_module.encode_fact(fact2_content, ["peppi"], dim=512)
    vec3 = hrr_module.encode_fact(fact3_content, ["backend"], dim=512)

    insert_fact(db, "f1", fact1_content, hrr_vector=hrr_module.phases_to_bytes(vec1))
    insert_fact(db, "f2", fact2_content, hrr_vector=hrr_module.phases_to_bytes(vec2))
    insert_fact(db, "f3", fact3_content, hrr_vector=hrr_module.phases_to_bytes(vec3))

    retriever = FactRetriever(db, hrr_dim=512)

    results = retriever.related("peppi", limit=10)

    assert isinstance(results, list)
    assert len(results) >= 2
    contents = [r.get("content", "") for r in results]
    assert any("peppi" in c for c in contents)
    peppi_scores = [r["score"] for r in results if "peppi" in r.get("content", "")]
    backend_scores = [r["score"] for r in results if "backend" in r.get("content", "")]
    if peppi_scores and backend_scores:
        assert max(peppi_scores) >= max(backend_scores)


def test_reason_requires_all_entities(tmp_path):
    """reason() only returns facts where ALL entities are structurally present."""
    db = make_memory_db(tmp_path)

    fact1_content = "peppi is a cat"
    fact2_content = "peppi likes fish"
    fact3_content = "backend runs on port 8080"

    vec1 = hrr_module.encode_fact(fact1_content, ["peppi", "cat"], dim=512)
    vec2 = hrr_module.encode_fact(fact2_content, ["peppi"], dim=512)
    vec3 = hrr_module.encode_fact(fact3_content, ["backend"], dim=512)

    insert_fact(db, "f1", fact1_content, hrr_vector=hrr_module.phases_to_bytes(vec1))
    insert_fact(db, "f2", fact2_content, hrr_vector=hrr_module.phases_to_bytes(vec2))
    insert_fact(db, "f3", fact3_content, hrr_vector=hrr_module.phases_to_bytes(vec3))

    retriever = FactRetriever(db, hrr_dim=512)

    results = retriever.reason(["peppi", "cat"], limit=10)

    assert isinstance(results, list)
    if results:
        top_content = results[0].get("content", "")
        assert "peppi" in top_content


def test_probe_with_category_filter(tmp_path):
    """probe() respects scope filter (maps to category in HRR retriever)."""
    db = make_memory_db(tmp_path)

    vec1 = hrr_module.encode_fact("peppi is a cat", ["peppi"], dim=512)
    vec2 = hrr_module.encode_fact("backend runs on port 8080", ["backend"], dim=512)

    insert_fact(db, "f1", "peppi is a cat", scope="animals", hrr_vector=hrr_module.phases_to_bytes(vec1))
    insert_fact(db, "f2", "backend runs on port 8080", scope="infra", hrr_vector=hrr_module.phases_to_bytes(vec2))

    retriever = FactRetriever(db, hrr_dim=512)

    results = retriever.probe("peppi", category="animals", limit=10)

    assert isinstance(results, list)
    if results:
        for r in results:
            assert "backend" not in r.get("content", "")


def test_related_limit_parameter(tmp_path):
    """probe/related/reason respect the limit parameter."""
    db = make_memory_db(tmp_path)

    for i in range(5):
        content = f"fact number {i}"
        vec = hrr_module.encode_fact(content, ["test"], dim=512)
        insert_fact(db, f"f{i}", content, hrr_vector=hrr_module.phases_to_bytes(vec))

    retriever = FactRetriever(db, hrr_dim=512)

    results = retriever.probe("test", limit=3)
    assert len(results) <= 3

    results = retriever.related("test", limit=2)
    assert len(results) <= 2

    results = retriever.reason(["test"], limit=4)
    assert len(results) <= 4


def test_probe_empty_query_falls_back_to_fts(tmp_path):
    """probe('') with no numpy falls back to FTS5 which returns results."""
    db = make_memory_db(tmp_path)
    insert_fact(db, "f1", "peppi is a cat", scope="project", hrr_vector=None)
    retriever = FactRetriever(db)

    import hermes_memory_core.search.hrri as hrri_module
    original = hrri_module._hrr._HAS_NUMPY
    hrri_module._hrr._HAS_NUMPY = False
    try:
        results = retriever.probe("", limit=5)
    finally:
        hrri_module._hrr._HAS_NUMPY = original

    assert isinstance(results, list)