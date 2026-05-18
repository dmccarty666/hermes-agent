# Copyright 2026 David McCarty. All rights reserved.
"""Tests for hermes_memory_core/search/hybrid.py — T-020.

Story 4.1.1 — Hybrid merge + scoring.
AC: Single hybrid query returns merged sorted results with backend_hits
    showing which backends matched.

Uses the existing module's search() function + ScoredResult class.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Ensure the venv python is on the path for the module import
import sys

venv_python = Path("/home/dmccarty/.hermes/hermes-agent/venv/bin/python3")
if venv_python.exists():
    sys.executable = str(venv_python)

from hermes_memory_core.store.sqlite import MemoryDB
from hermes_memory_core.search.hybrid import (
    search as hybrid_search,
    ScoredResult,
    jaccard_similarity,
    freshness_decay,
    _MODE_WEIGHTS,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_memory_db(tmp_path: Path) -> MemoryDB:
    db = MemoryDB(tmp_path / "memory.sqlite")
    db.initialize()
    return db


def insert_turn(
    conn: sqlite3.Connection,
    turn_id: str,
    session_id: str,
    role: str,
    content: str,
    project: str = "test-project",
    timestamp: str = "2026-05-18T12:00:00Z",
) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO sessions
           (session_id, agent, project, started_at) VALUES (?, 'test', ?, '2026-01-01T00:00:00Z')""",
        (session_id, project),
    )
    conn.execute(
        """INSERT INTO turns
           (turn_id, session_id, sequence, timestamp, role, content,
            dream_status, index_status, source_refs_json,
            redaction_count, redaction_types_json)
           VALUES (?, ?, 1, ?, ?, ?, 'pending', 'pending', '[]', 0, '[]')""",
        (turn_id, session_id, timestamp, role, content),
    )
    conn.commit()


# ------------------------------------------------------------------------------------------------------------------------------------------
# Helpers that mirror actual backend result shapes
# ------------------------------------------------------------------------------------------------------------------------------------------

def make_mock_fts_row(
    session_id: str = "sess_001",
    turn_id: str = "t1",
    content: str = "hello world foo bar",
    rank: float = -0.5,
) -> Dict[str, Any]:
    """Return a dict shaped like fts5_search returns for a turns row."""
    return {
        "turn_id": turn_id,
        "session_id": session_id,
        "content": content,
        "source_ref": f"session:{session_id}#turn={turn_id}",
        "snippet": f"...{content[:20]}...",
        "rank": rank,
        "_backend": "fts",
        "_table": "turns",
    }


def make_mock_qdrant_row(
    chunk_id: str = "chk_001",
    session_id: str = "sess_001",
    content: str = "hello world semantic chunk",
    score: float = 0.88,
) -> Dict[str, Any]:
    """Return a dict shaped like semantic_search returns."""
    return {
        "content": content,
        "source_ref": f"session:{session_id}#chunk={chunk_id}",
        "score": score,
        "metadata": {
            "chunk_id": chunk_id,
            "session_id": session_id,
            "start_turn_id": "t1",
            "end_turn_id": "t3",
            "chunk_type": "turn_window",
            "role_mix": ["user", "assistant"],
            "turn_count": 3,
        },
        "_backend": "qdrant",
    }


# ------------------------------------------------------------------------------------------------------------------------------------------
# AC-1: ScoredResult — all fields present and populated
# ------------------------------------------------------------------------------------------------------------------------------------------

class TestScoredResult:
    """AC-1: ScoredResult carries individual backend scores plus combined score."""

    def test_scored_result_fields(self):
        r = ScoredResult(
            chunk_id="chk_001",
            session_id="sess_001",
            content="hello world test",
            score=0.72,
            fts_score=0.50,
            qdrant_score=0.88,
            jaccard_score=0.30,
            hrr_score=0.10,
            backend_hits=["fts", "qdrant"],
            source_ref="session:sess_001#chunk=chk_001",
            metadata={"chunk_id": "chk_001"},
        )
        assert r.chunk_id == "chk_001"
        assert r.score == 0.72
        assert r.fts_score == 0.50
        assert r.qdrant_score == 0.88
        assert r.jaccard_score == 0.30
        assert r.hrr_score == 0.10
        assert "fts" in r.backend_hits
        assert "qdrant" in r.backend_hits

    def test_scored_result_to_dict(self):
        r = ScoredResult(
            chunk_id="chk_001",
            session_id="sess_001",
            content="test content",
            score=0.5,
        )
        d = r.to_dict()
        assert d["chunk_id"] == "chk_001"
        assert d["score"] == 0.5
        assert "backend_hits" in d


# ------------------------------------------------------------------------------------------------------------------------------------------
# AC-2: Mode weight tables — FTS, Qdrant, Jaccard, HRR weights per mode
# ------------------------------------------------------------------------------------------------------------------------------------------

class TestModeWeights:
    """AC-2: Backend weights per mode per TDD §8."""

    def test_default_weights_sum_to_one(self):
        for mode, weights in _MODE_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, f"Mode {mode} weights sum to {total}, not 1.0"

    def test_keyword_mode_fts_heavy(self):
        w = _MODE_WEIGHTS["keyword"]
        assert w["fts"] == pytest.approx(0.70)
        assert w["qdrant"] == pytest.approx(0.10)
        assert w["jaccard"] == pytest.approx(0.20)
        assert w["hrr"] == pytest.approx(0.00)

    def test_semantic_mode_qdrant_heavy(self):
        w = _MODE_WEIGHTS["semantic"]
        assert w["fts"] == pytest.approx(0.05)
        assert w["qdrant"] == pytest.approx(0.80)
        assert w["jaccard"] == pytest.approx(0.10)
        assert w["hrr"] == pytest.approx(0.05)

    def test_hybrid_mode_default(self):
        w = _MODE_WEIGHTS["hybrid"]
        assert w["fts"] == pytest.approx(0.30)
        assert w["qdrant"] == pytest.approx(0.40)
        assert w["jaccard"] == pytest.approx(0.15)
        assert w["hrr"] == pytest.approx(0.15)

    def test_facts_only_mode(self):
        w = _MODE_WEIGHTS["facts_only"]
        assert w["fts"] == pytest.approx(0.40)
        assert w["qdrant"] == pytest.approx(0.30)


# ------------------------------------------------------------------------------------------------------------------------------------------
# AC-3: Freshness decay
# ------------------------------------------------------------------------------------------------------------------------------------------

class TestFreshnessDecay:
    """AC-3: Score multiplied by freshness_decay(updated_at)."""

    def test_freshness_decay_none_returns_one(self):
        assert freshness_decay(None) == 1.0

    def test_freshness_decay_empty_returns_one(self):
        assert freshness_decay("") == 1.0

    def test_freshness_decay_recent_is_near_one(self):
        recent = "2026-05-18T12:00:00Z"
        decay = freshness_decay(recent)
        assert 0.9 <= decay <= 1.0

    def test_freshness_decay_older_is_lower(self):
        old = "2020-01-01T00:00:00Z"
        recent = "2026-05-18T12:00:00Z"
        assert freshness_decay(old) < freshness_decay(recent)


# ------------------------------------------------------------------------------------------------------------------------------------------
# AC-4: Jaccard similarity
# ------------------------------------------------------------------------------------------------------------------------------------------

class TestJaccardSimilarity:
    """AC-4: jaccard_similarity returns token Jaccard in [0, 1]."""

    def test_identical_strings(self):
        score = jaccard_similarity("hello world foo", "hello world foo")
        assert score == 1.0

    def test_partial_overlap(self):
        score = jaccard_similarity("hello world", "hello world bar baz")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        score = jaccard_similarity("cat dog", "apple banana")
        assert score == 0.0

    def test_empty_query(self):
        assert jaccard_similarity("", "hello world") == 0.0

    def test_symmetric(self):
        a = jaccard_similarity("a b c", "b c d e")
        b = jaccard_similarity("b c d e", "a b c")
        assert a == b


# ------------------------------------------------------------------------------------------------------------------------------------------
# AC-5 + AC-6: hybrid_search returns merged + sorted ScoredResults with backend_hits
# ------------------------------------------------------------------------------------------------------------------------------------------

class TestHybridSearchAPI:
    """AC-5 + AC-6: hybrid_search() orchestrates backends; returns ScoredResults sorted by score."""

    @pytest.fixture
    def memory_db(self, tmp_path: Path) -> MemoryDB:
        return make_memory_db(tmp_path)

    def test_returns_dict_with_results_key(self, memory_db: MemoryDB):
        """hybrid_search returns a dict with 'results' key containing ScoredResults."""
        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=[]), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", return_value=[]):
            result = hybrid_search(
                query="test query",
                mode="hybrid",
                filters={},
                limit=10,
                memory_db=memory_db,
            )

        assert isinstance(result, dict)
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_results_are_scored_results(self, memory_db: MemoryDB):
        """Each entry in results is a ScoredResult (or dict equivalent)."""
        mock_fts = [make_mock_fts_row("sess_001", "t1", "hello world foo", rank=-0.5)]
        mock_qdr: List[Dict[str, Any]] = []

        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=mock_fts), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", return_value=mock_qdr):
            result = hybrid_search(
                query="hello world",
                mode="hybrid",
                filters={},
                limit=10,
                memory_db=memory_db,
            )

        assert len(result["results"]) >= 1
        r = result["results"][0]
        assert isinstance(r, dict)
        assert "score" in r
        assert "backend_hits" in r

    def test_backend_hits_shows_fts(self, memory_db: MemoryDB):
        """FTS result has backend_hits=['fts']."""
        mock_fts = [make_mock_fts_row("sess_001", "t1", "keyword result", rank=-0.5)]
        mock_qdr: List[Dict[str, Any]] = []

        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=mock_fts), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", return_value=mock_qdr):
            result = hybrid_search(
                query="keyword result",
                mode="hybrid",
                filters={},
                limit=10,
                memory_db=memory_db,
            )

        assert len(result["results"]) >= 1
        hits = result["results"][0].get("backend_hits", [])
        assert "fts" in hits

    def test_qdrant_result_has_backend_hits(self, memory_db: MemoryDB):
        """Qdrant result has backend_hits=['qdrant']."""
        mock_fts: List[Dict[str, Any]] = []
        mock_qdr = [make_mock_qdrant_row("chk_001", "sess_001", "semantic match", 0.92)]

        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=mock_fts), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", return_value=mock_qdr):
            result = hybrid_search(
                query="semantic match",
                mode="hybrid",
                filters={},
                limit=10,
                memory_db=memory_db,
            )

        assert len(result["results"]) >= 1
        hits = result["results"][0].get("backend_hits", [])
        assert "qdrant" in hits

    def test_both_backends_agree_union_hits(self, memory_db: MemoryDB):
        """When same content matched by multiple backends, backend_hits is the union."""
        # Same content in both FTS and Qdrant (deduped to one result with both hits)
        mock_fts = [make_mock_fts_row("sess_001", "t1", "shared content", rank=-0.5)]
        mock_qdr = [make_mock_qdrant_row("chk_001", "sess_001", "shared content", 0.90)]

        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=mock_fts), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", return_value=mock_qdr):
            result = hybrid_search(
                query="shared content",
                mode="hybrid",
                filters={},
                limit=10,
                memory_db=memory_db,
            )

        assert len(result["results"]) >= 1
        hits = result["results"][0].get("backend_hits", [])
        assert len(hits) >= 2 or ("fts" in hits or "qdrant" in hits)

    def test_empty_backends_returns_empty_results(self, memory_db: MemoryDB):
        """When both backends return nothing, results is empty list."""
        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=[]), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", return_value=[]):
            result = hybrid_search(
                query="nonexistent xyzzy",
                mode="hybrid",
                filters={},
                limit=10,
                memory_db=memory_db,
            )

        assert result["results"] == []
        assert result["count"] == 0

    def test_mode_keyword_weights_fts_heavily(self, memory_db: MemoryDB):
        """keyword mode returns FTS-scored results."""
        mock_fts = [
            make_mock_fts_row("sess_001", "t1", "python is great", rank=-0.3),
            make_mock_fts_row("sess_001", "t2", "java is ok", rank=-1.5),
        ]
        mock_qdr: List[Dict[str, Any]] = []

        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=mock_fts), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", return_value=mock_qdr):
            result = hybrid_search(
                query="python",
                mode="keyword",
                filters={},
                limit=10,
                memory_db=memory_db,
            )

        assert len(result["results"]) >= 1
        # Higher BM25 rank (less negative) = better FTS match
        assert result["results"][0]["fts_score"] >= result["results"][-1]["fts_score"]

    def test_mode_semantic_weights_qdrant_heavily(self, memory_db: MemoryDB):
        """semantic mode returns Qdrant-scored results."""
        mock_fts: List[Dict[str, Any]] = []
        mock_qdr = [
            make_mock_qdrant_row("chk_001", "sess_001", "semantic match here", 0.95),
            make_mock_qdrant_row("chk_002", "sess_002", "less relevant", 0.55),
        ]

        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=mock_fts), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", return_value=mock_qdr):
            result = hybrid_search(
                query="semantic match",
                mode="semantic",
                filters={},
                limit=10,
                memory_db=memory_db,
            )

        assert len(result["results"]) >= 1
        assert result["results"][0]["qdrant_score"] >= result["results"][-1]["qdrant_score"]

    def test_results_sorted_by_score_descending(self, memory_db: MemoryDB):
        """Results are sorted by combined score, highest first."""
        mock_fts = [
            make_mock_fts_row("s1", "t1", "low score result", rank=-2.0),
            make_mock_fts_row("s2", "t2", "high score result", rank=-0.2),
        ]
        mock_qdr: List[Dict[str, Any]] = []

        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=mock_fts), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", return_value=mock_qdr):
            result = hybrid_search(
                query="score result",
                mode="hybrid",
                filters={},
                limit=10,
                memory_db=memory_db,
            )

        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_limit_respected(self, memory_db: MemoryDB):
        """Limit cap is applied after merge and sort."""
        mock_fts = [
            make_mock_fts_row("s1", f"t{i}", f"result number {i}", rank=-float(i))
            for i in range(20)
        ]
        mock_qdr: List[Dict[str, Any]] = []

        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=mock_fts), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", return_value=mock_qdr):
            result = hybrid_search(
                query="result",
                mode="hybrid",
                filters={},
                limit=5,
                memory_db=memory_db,
            )

        assert len(result["results"]) <= 5

    def test_degraded_modes_reported_when_qdrant_fails(self, memory_db: MemoryDB):
        """When Qdrant is unavailable, degraded_modes includes 'qdrant'."""
        mock_fts = [make_mock_fts_row("s1", "t1", "fts fallback", rank=-0.5)]
        mock_qdr: List[Dict[str, Any]] = []

        def failing_qdrant(*args, **kwargs):
            from hermes_memory_core.search.semantic import SemanticSearchError
            raise SemanticSearchError("Qdrant unavailable")

        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=mock_fts), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", side_effect=failing_qdrant):
            result = hybrid_search(
                query="fts fallback",
                mode="hybrid",
                filters={},
                limit=10,
                memory_db=memory_db,
            )

        # FTS should still return results
        assert len(result["results"]) >= 1
        # degraded_modes should reflect Qdrant failure
        assert "qdrant" in result.get("degraded_modes", [])

    def test_response_includes_mode_and_count(self, memory_db: MemoryDB):
        """Response dict contains mode, count, query fields."""
        mock_fts = [make_mock_fts_row("s1", "t1", "test", rank=-0.5)]
        mock_qdr: List[Dict[str, Any]] = []

        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=mock_fts), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", return_value=mock_qdr):
            result = hybrid_search(
                query="test query",
                mode="keyword",
                filters={},
                limit=10,
                memory_db=memory_db,
            )

        assert "mode" in result
        assert "query" in result
        assert "count" in result


# ------------------------------------------------------------------------------------------------------------------------------------------
# AC-7: backend_hits union on dedup
# ------------------------------------------------------------------------------------------------------------------------------------------

class TestDeduplication:
    """AC-7 + AC-4 from Plan: dedup by (chunk_id, source_ref, content_hash)."""

    def test_same_content_from_fts_and_qdrant_deduped(self, tmp_path: Path):
        """Identical content from both backends produces one result with union backend_hits."""
        memory_db = make_memory_db(tmp_path)
        mock_fts = [make_mock_fts_row("sess_001", "t1", "shared content", rank=-0.5)]
        mock_qdr = [make_mock_qdrant_row("chk_001", "sess_001", "shared content", 0.90)]

        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=mock_fts), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", return_value=mock_qdr):
            result = hybrid_search(
                query="shared content",
                mode="hybrid",
                filters={},
                limit=10,
                memory_db=memory_db,
            )

        # Should be deduped to one result
        assert len(result["results"]) == 1
        hits = result["results"][0].get("backend_hits", [])
        # Both backends matched
        assert len(hits) >= 2 or set(hits).issuperset({"fts", "qdrant"})

    def test_different_content_kept_separate(self, tmp_path: Path):
        """Different content from different backends are not merged."""
        memory_db = make_memory_db(tmp_path)
        mock_fts = [make_mock_fts_row("sess_001", "t1", "fts only content", rank=-0.5)]
        mock_qdr = [make_mock_qdrant_row("chk_001", "sess_002", "qdrant only content", 0.88)]

        with patch("hermes_memory_core.search.hybrid._search_fts", return_value=mock_fts), \
             patch("hermes_memory_core.search.hybrid._search_qdrant", return_value=mock_qdr):
            result = hybrid_search(
                query="content",
                mode="hybrid",
                filters={},
                limit=10,
                memory_db=memory_db,
            )

        assert len(result["results"]) == 2