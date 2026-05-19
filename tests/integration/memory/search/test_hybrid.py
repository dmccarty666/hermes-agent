# Copyright 2026 David McCarty. All rights reserved.
"""Tests for hermes_memory_core/search/hybrid.py — T-020 (Epic 4.1.1).

Covers:
- AC-1: Single hybrid query returns merged sorted results with backend_hits
- Mode-driven weights (keyword, semantic, hybrid, facts_only, default)
- Score normalization (FTS BM25 → [0,1], Qdrant cosine already in [0,1])
- Dedup by (chunk_id, session_id, content_hash)
- Trust + freshness decay
- Graceful degradation: when backends fail, weights redistribute
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hermes_memory_core.search.hybrid import (
    HybridScorer,
    ScoredResult,
    search,
    _normalize_bm25,
    _content_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fts_hit(chunk_id, session_id, content, rank):
    """Minimal FTS result dict as returned by fts5_search."""
    return {
        "chunk_id": chunk_id,
        "session_id": session_id,
        "chunk_text": content,
        "content": content,
        "rank": rank,
        "source_ref": f"session:{session_id}#chunk={chunk_id}",
        "timestamp": "2026-05-18T12:00:00Z",
    }


def make_qdrant_hit(chunk_id, session_id, content, score):
    """Minimal Qdrant result dict as returned by semantic_search."""
    return {
        "content": content,
        "source_ref": f"session:{session_id}#chunk={chunk_id}",
        "score": score,
        "metadata": {
            "chunk_id": chunk_id,
            "session_id": session_id,
            "date": "2026-05-18",
        },
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestNormalizeBm25:
    def test_most_negative_maps_to_0(self):
        # BM25 lower is better (more negative = higher rank), so max abs → 0
        assert _normalize_bm25(-20.0) == pytest.approx(0.0)

    def test_perfect_match_maps_to_1(self):
        assert _normalize_bm25(0.0) == pytest.approx(1.0)

    def test_midpoint(self):
        assert _normalize_bm25(-10.0) == pytest.approx(0.5)

    def test_clamped_to_0_1(self):
        assert _normalize_bm25(-100.0) == 0.0
        assert _normalize_bm25(10.0) == 1.0


class TestContentHash:
    def test_same_content_same_hash(self):
        h1 = _content_hash("hello world")
        h2 = _content_hash("hello world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = _content_hash("hello world")
        h2 = _content_hash("goodbye world")
        assert h1 != h2

    def test_hash_is_16_hex_chars(self):
        h = _content_hash("test")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestModeWeights:
    def test_all_modes_sum_to_1(self):
        for mode, weights in HybridScorer._MODE_WEIGHTS.items():
            total = sum(weights.values())
            assert total == pytest.approx(1.0), f"{mode} weights sum to {total}"

    def test_all_modes_have_required_backends(self):
        required = {"fts", "qdrant", "jaccard", "hrr"}
        for mode, weights in HybridScorer._MODE_WEIGHTS.items():
            assert required.issubset(weights.keys()), f"{mode} missing backends"


class TestHybridScorerDedup:
    """Dedup by (chunk_id, session_id, content_hash)."""

    def test_duplicate_merged_highest_score_wins(self):
        scorer = HybridScorer()
        results = [
            ScoredResult(
                chunk_id="c1", session_id="s1", text="same content",
                score=0.8, fts_score=0.5, backend_hits=["fts"],
            ),
            ScoredResult(
                chunk_id="c1", session_id="s1", text="same content",
                score=0.9, qdrant_score=0.9, backend_hits=["qdrant"],
            ),
        ]
        deduped = scorer._deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0].score == 0.9  # highest score wins
        assert "fts" in deduped[0].backend_hits
        assert "qdrant" in deduped[0].backend_hits

    def test_different_content_not_deduped(self):
        scorer = HybridScorer()
        results = [
            ScoredResult(
                chunk_id="c1", session_id="s1", text="content A",
                score=0.8, fts_score=0.8,
            ),
            ScoredResult(
                chunk_id="c1", session_id="s1", text="content B",
                score=0.9, qdrant_score=0.9,
            ),
        ]
        deduped = scorer._deduplicate(results)
        # Different content → different hash → not deduped
        assert len(deduped) == 2


class TestFreshnessDecay:
    """Exponential decay: 0.5 ** (age_days / half_life)."""

    def test_no_timestamp_returns_1(self):
        scorer = HybridScorer()
        assert scorer._freshness_decay(None) == 1.0

    def test_zero_half_life_returns_1(self):
        scorer = HybridScorer()
        assert scorer._freshness_decay("2026-01-01T00:00:00Z", 0.0) == 1.0

    def test_recent_entry_near_1(self):
        from datetime import datetime, timezone
        scorer = HybridScorer()
        recent = datetime.now(timezone.utc).isoformat()
        assert scorer._freshness_decay(recent) == pytest.approx(1.0, rel=0.05)

    def test_old_entry_decay(self):
        scorer = HybridScorer()
        result = scorer._freshness_decay("2020-01-01T00:00:00Z", 90.0)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# search() function tests
# ---------------------------------------------------------------------------

class TestSearchBasic:
    """AC-1: Single hybrid query returns merged sorted results with backend_hits."""

    @pytest.fixture
    def mock_fts(self):
        with patch("hermes_memory_core.search.hybrid.fts5_search") as m:
            m.return_value = [
                make_fts_hit("c1", "sess1", "avoiding paid provider Honcho cost", -3.0),
                make_fts_hit("c2", "sess2", "local memory is free on-premise", -5.0),
            ]
            yield m

    @pytest.fixture
    def mock_semantic(self):
        with patch("hermes_memory_core.search.hybrid.semantic_search") as m:
            m.return_value = [
                make_qdrant_hit("c1", "sess1", "avoiding paid provider Honcho cost", 0.92),
                make_qdrant_hit("c3", "sess3", "local memory avoids per-token charges", 0.88),
            ]
            yield m

    def test_returns_dict_with_results(self, mock_fts, mock_semantic):
        result = search("avoiding paid provider cost")
        assert isinstance(result, dict)
        assert "results" in result

    def test_results_have_backend_hits(self, mock_fts, mock_semantic):
        result = search("avoiding paid provider cost")
        assert len(result["results"]) >= 1
        for r in result["results"]:
            assert "backend_hits" in r
            assert isinstance(r["backend_hits"], list)
            assert len(r["backend_hits"]) >= 1

    def test_results_sorted_by_score_desc(self, mock_fts, mock_semantic):
        result = search("avoiding paid provider cost")
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_mode_field_set(self, mock_fts, mock_semantic):
        result = search("avoiding paid provider cost", mode="hybrid")
        assert result["mode"] == "hybrid"

    def test_count_matches_results(self, mock_fts, mock_semantic):
        result = search("avoiding paid provider cost", limit=5)
        assert result["count"] == len(result["results"])
        assert result["count"] <= 5

    def test_query_field_echoed(self, mock_fts, mock_semantic):
        result = search("my test query")
        assert result["query"] == "my test query"


class TestDedupByChunkId:
    """Same (chunk_id, session_id, content_hash) → merged into one result."""

    @pytest.fixture
    def mock_fts(self):
        with patch("hermes_memory_core.search.hybrid.fts5_search") as m:
            m.return_value = [
                make_fts_hit("c1", "sess1", "same content here", -3.0),
            ]
            yield m

    @pytest.fixture
    def mock_semantic(self):
        with patch("hermes_memory_core.search.hybrid.semantic_search") as m:
            m.return_value = [
                make_qdrant_hit("c1", "sess1", "same content here", 0.92),
            ]
            yield m

    def test_duplicate_across_backends_merges(self, mock_fts, mock_semantic):
        result = search("test query")
        chunk_ids = [r["chunk_id"] for r in result["results"]]
        assert chunk_ids.count("c1") == 1

    def test_merged_result_has_both_backends(self, mock_fts, mock_semantic):
        result = search("test query")
        c1 = next((r for r in result["results"] if r["chunk_id"] == "c1"), None)
        assert c1 is not None
        assert "fts" in c1["backend_hits"]
        assert "qdrant" in c1["backend_hits"]


class TestModeDrivenWeights:
    """Different modes produce different effective weights."""

    @pytest.fixture
    def mock_fts_empty(self):
        with patch("hermes_memory_core.search.hybrid.fts5_search") as m:
            m.return_value = []
            yield m

    @pytest.fixture
    def mock_semantic_empty(self):
        with patch("hermes_memory_core.search.hybrid.semantic_search") as m:
            m.return_value = []
            yield m

    def test_keyword_mode_fts_heavy(self, mock_fts_empty, mock_semantic_empty):
        result = search("test query", mode="keyword")
        assert result["backend_weights"]["fts"] > 0.6

    def test_semantic_mode_qdrant_heavy(self, mock_fts_empty, mock_semantic_empty):
        result = search("test query", mode="semantic")
        assert result["backend_weights"]["qdrant"] > 0.6

    def test_hybrid_mode_balanced(self, mock_fts_empty, mock_semantic_empty):
        result = search("test query", mode="hybrid")
        w = result["backend_weights"]
        assert w["fts"] > 0.2
        assert w["qdrant"] > 0.3

    def test_default_mode(self, mock_fts_empty, mock_semantic_empty):
        result = search("test query", mode="default")
        assert "default" in result["mode"]

    def test_unknown_mode_falls_back_to_hybrid(self, mock_fts_empty, mock_semantic_empty):
        result = search("test query", mode="unknown_mode")
        # Falls back to _DEFAULT_WEIGHTS (hybrid values)
        assert result["backend_weights"]["fts"] == 0.30


class TestLimit:
    """The limit parameter is respected."""

    @pytest.fixture
    def mock_fts(self):
        with patch("hermes_memory_core.search.hybrid.fts5_search") as m:
            m.return_value = [
                make_fts_hit(f"c{i}", "s", f"content {i}", -float(i))
                for i in range(1, 21)
            ]
            yield m

    @pytest.fixture
    def mock_semantic(self):
        with patch("hermes_memory_core.search.hybrid.semantic_search") as m:
            m.return_value = [
                make_qdrant_hit(f"c{i}", "s", f"content {i}", 0.9 - i * 0.01)
                for i in range(1, 21)
            ]
            yield m

    def test_limit_5(self, mock_fts, mock_semantic):
        result = search("test query", limit=5)
        assert result["count"] <= 5

    def test_limit_10(self, mock_fts, mock_semantic):
        result = search("test query", limit=10)
        assert result["count"] <= 10

    def test_default_limit_10(self, mock_fts, mock_semantic):
        result = search("test query")
        assert result["count"] <= 10


class TestBackendDegradation:
    """When FTS fails, scorer returns empty list (graceful degradation)."""

    @pytest.fixture
    def mock_fts_fails(self):
        with patch("hermes_memory_core.search.hybrid.fts5_search") as m:
            m.side_effect = Exception("FTS unavailable")
            yield m

    @pytest.fixture
    def mock_semantic(self):
        with patch("hermes_memory_core.search.hybrid.semantic_search") as m:
            m.return_value = [
                make_qdrant_hit("c1", "sess1", "local memory content", 0.95),
            ]
            yield m

    def test_qdrant_only_returns_results(self, mock_fts_fails, mock_semantic):
        result = search("test query", mode="hybrid")
        assert result["count"] >= 1

    def test_all_backends_fail_returns_empty_results(self):
        with patch("hermes_memory_core.search.hybrid.fts5_search") as fts:
            fts.side_effect = Exception("FTS down")
            with patch("hermes_memory_core.search.hybrid.semantic_search") as sem:
                sem.side_effect = Exception("Qdrant down")
                result = search("test query")
                # Returns dict with empty results list
                assert result["results"] == []
                assert result["count"] == 0


class TestWeightRedistribution:
    """Phase 4 Story 4.1.2 — graceful degradation weight redistribution."""

    def test_no_degradation_returns_base_weights(self):
        """When nothing is degraded, weights unchanged."""
        from hermes_memory_core.search.hybrid import _redistribute_weights
        base = {"fts": 0.30, "qdrant": 0.40, "jaccard": 0.15, "hrr": 0.15}
        result = _redistribute_weights(base, [])
        assert result == base

    def test_qdrant_down_proportional_redistribution(self):
        """Qdrant weight redistributed to fts, jaccard, hrr proportionally."""
        from hermes_memory_core.search.hybrid import _redistribute_weights
        base = {"fts": 0.30, "qdrant": 0.40, "jaccard": 0.15, "hrr": 0.15}
        result = _redistribute_weights(base, ["qdrant"])
        # qdrant → 0, rest redistributed so total = 1.0
        assert result["qdrant"] == 0.0
        # All available sum to 1.0
        available_sum = sum(v for k, v in result.items() if k != "qdrant")
        assert available_sum == pytest.approx(1.0)
        # Relative proportions maintained: fts:jaccard:hrr = 0.30:0.15:0.15
        # fts: 0.30/(0.30+0.15+0.15) = 0.30/0.60 = 0.50
        assert result["fts"] == pytest.approx(0.50)
        assert result["jaccard"] == pytest.approx(0.25)
        assert result["hrr"] == pytest.approx(0.25)

    def test_fts_down_proportional_redistribution(self):
        """FTS weight redistributed to qdrant, jaccard, hrr proportionally."""
        from hermes_memory_core.search.hybrid import _redistribute_weights
        base = {"fts": 0.30, "qdrant": 0.40, "jaccard": 0.15, "hrr": 0.15}
        result = _redistribute_weights(base, ["fts"])
        assert result["fts"] == 0.0
        available_sum = sum(v for k, v in result.items() if k != "fts")
        assert available_sum == pytest.approx(1.0)
        # qdrant gets 0.40/0.70, jaccard 0.15/0.70, hrr 0.15/0.70
        assert result["qdrant"] == pytest.approx(0.40 / 0.70)
        assert result["jaccard"] == pytest.approx(0.15 / 0.70)
        assert result["hrr"] == pytest.approx(0.15 / 0.70)

    def test_multiple_backends_down(self):
        """When fts+qdrant down, jaccard+hrr split the full weight."""
        from hermes_memory_core.search.hybrid import _redistribute_weights
        base = {"fts": 0.30, "qdrant": 0.40, "jaccard": 0.15, "hrr": 0.15}
        result = _redistribute_weights(base, ["fts", "qdrant"])
        assert result["fts"] == 0.0
        assert result["qdrant"] == 0.0
        available_sum = result["jaccard"] + result["hrr"]
        assert available_sum == pytest.approx(1.0)
        # Equal split: both 0.50
        assert result["jaccard"] == pytest.approx(0.50)
        assert result["hrr"] == pytest.approx(0.50)

    def test_all_backends_down_returns_zeros(self):
        """All four backends down → all weights 0.0."""
        from hermes_memory_core.search.hybrid import _redistribute_weights
        base = {"fts": 0.30, "qdrant": 0.40, "jaccard": 0.15, "hrr": 0.15}
        result = _redistribute_weights(base, ["fts", "qdrant", "jaccard", "hrr"])
        for k in result:
            assert result[k] == 0.0

    def test_search_reports_degraded_modes_on_qdrant_failure(self):
        """When Qdrant fails, degraded_modes includes 'qdrant' in response."""
        with patch("hermes_memory_core.search.hybrid._search_fts") as mock_fts, \
             patch("hermes_memory_core.search.hybrid._search_qdrant") as mock_qdr:
            mock_fts.return_value = [make_fts_hit("c1", "s1", "fts content", -0.5)]
            mock_qdr.side_effect = Exception("Qdrant unavailable")
            result = search("fts content", mode="hybrid", filters={}, limit=10, memory_db=None)
            assert "qdrant" in result["degraded_modes"]
            # But results should still come from FTS
            assert result["count"] >= 1

    def test_search_reports_degraded_modes_on_fts_failure(self):
        """When FTS fails, degraded_modes includes 'fts' in response."""
        with patch("hermes_memory_core.search.hybrid._search_fts") as mock_fts, \
             patch("hermes_memory_core.search.hybrid._search_qdrant") as mock_qdr:
            mock_fts.side_effect = Exception("FTS unavailable")
            mock_qdr.return_value = [make_qdrant_hit("c1", "s1", "qdrant content", 0.9)]
            result = search("qdrant content", mode="hybrid", filters={}, limit=10, memory_db=None)
            assert "fts" in result["degraded_modes"]
            assert result["count"] >= 1

    def test_backend_weights_reflects_redistribution(self):
        """backend_weights in response is the redistributed set, not original."""
        with patch("hermes_memory_core.search.hybrid._search_fts") as mock_fts, \
             patch("hermes_memory_core.search.hybrid._search_qdrant") as mock_qdr:
            mock_fts.return_value = [make_fts_hit("c1", "s1", "test", -0.5)]
            mock_qdr.side_effect = Exception("Qdrant unavailable")
            result = search("test", mode="hybrid", filters={}, limit=10, memory_db=None)
            # qdrant's 0.40 weight should be redistributed
            assert result["backend_weights"]["qdrant"] == 0.0
            assert result["backend_weights"]["fts"] > 0.30  # gets some of qdrant's share


class TestScoredResultFields:
    """Each result dict has all required fields."""

    @pytest.fixture
    def mock_fts(self):
        with patch("hermes_memory_core.search.hybrid.fts5_search") as m:
            m.return_value = [
                make_fts_hit("c1", "sess1", "test content", -2.0),
            ]
            yield m

    @pytest.fixture
    def mock_semantic(self):
        with patch("hermes_memory_core.search.hybrid.semantic_search") as m:
            m.return_value = []
            yield m

    def test_result_has_all_score_fields(self, mock_fts, mock_semantic):
        result = search("test query")
        r = result["results"][0]
        for field in ("chunk_id", "session_id", "content", "score",
                      "fts_score", "qdrant_score", "jaccard_score", "hrr_score",
                      "backend_hits", "source_ref", "metadata"):
            assert field in r, f"missing field: {field}"

    def test_source_ref_format(self, mock_fts, mock_semantic):
        result = search("test query")
        r = result["results"][0]
        assert "session:" in r["source_ref"] or "chunk:" in r["source_ref"]

    def test_metadata_has_trust_and_freshness(self, mock_fts, mock_semantic):
        result = search("test query")
        r = result["results"][0]
        assert "trust" in r["metadata"]
        assert "freshness_decay" in r["metadata"]


class TestCombinedScoreFormula:
    """Score = sum(weights * normalized) * trust * freshness."""

    def test_combined_score_uses_weights(self):
        scorer = HybridScorer()
        score = scorer._combined_score(
            fts_score=0.5, qdrant_score=1.0, jaccard_score=0.0, hrr_score=0.0,
            trust_score=1.0, freshness_decay=1.0, mode="hybrid",
        )
        # hybrid: fts=0.30, qdrant=0.40 → 0.30*0.5 + 0.40*1.0 = 0.55
        assert score == pytest.approx(0.55)

    def test_trust_multiplier_applies(self):
        scorer = HybridScorer()
        base = scorer._combined_score(
            fts_score=0.5, qdrant_score=1.0, jaccard_score=0.0, hrr_score=0.0,
            trust_score=0.5, freshness_decay=1.0, mode="hybrid",
        )
        # 0.55 * 0.5 = 0.275
        assert base == pytest.approx(0.275)

    def test_freshness_decay_applies(self):
        scorer = HybridScorer()
        base = scorer._combined_score(
            fts_score=0.5, qdrant_score=1.0, jaccard_score=0.0, hrr_score=0.0,
            trust_score=1.0, freshness_decay=0.5, mode="hybrid",
        )
        # 0.55 * 0.5 = 0.275
        assert base == pytest.approx(0.275)

    def test_keyword_mode_weights(self):
        scorer = HybridScorer()
        score = scorer._combined_score(
            fts_score=1.0, qdrant_score=1.0, jaccard_score=0.0, hrr_score=0.0,
            trust_score=1.0, freshness_decay=1.0, mode="keyword",
        )
        # keyword: fts=0.70, qdrant=0.10 → 0.70*1.0 + 0.10*1.0 = 0.80
        assert score == pytest.approx(0.80)