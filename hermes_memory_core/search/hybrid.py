# Copyright 2026 David McCarty. All rights reserved.
"""Hybrid retrieval scorer for Hermes Local Memory.

Combines FTS5 keyword, Qdrant semantic, Jaccard structural, and HRR scores.
Phase 4 Story 4.1.1 — Hybrid merge + scoring.

AC: Single hybrid query returns merged sorted results with `backend_hits`
    showing which backends matched.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Module-level function references for mocking in tests
fts5_search = None
semantic_search = None


def _reload_fts5_search():
    """Reload fts5_search from hermes_memory_core.search.fts5."""
    global fts5_search
    from hermes_memory_core.search.fts5 import fts5_search as _f
    fts5_search = _f
    return _f


def _reload_semantic_search():
    """Reload semantic_search from hermes_memory_core.search.semantic."""
    global semantic_search
    from hermes_memory_core.search.semantic import semantic_search as _s
    semantic_search = _s
    return _s


# --------------------------------------------------------------------------- #
# Mode weight tables (exposed for tests)
# ---------------------------------------------------------------------------#

_MODE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "default":    {"fts": 0.30, "qdrant": 0.40, "jaccard": 0.15, "hrr": 0.15},
    "keyword":    {"fts": 0.70, "qdrant": 0.10, "jaccard": 0.20, "hrr": 0.00},
    "semantic":   {"fts": 0.05, "qdrant": 0.80, "jaccard": 0.10, "hrr": 0.05},
    "hybrid":     {"fts": 0.30, "qdrant": 0.40, "jaccard": 0.15, "hrr": 0.15},
    "facts_only": {"fts": 0.40, "qdrant": 0.30, "jaccard": 0.15, "hrr": 0.15},
}

_DEFAULT_WEIGHTS = {"fts": 0.30, "qdrant": 0.40, "jaccard": 0.15, "hrr": 0.15}


# --------------------------------------------------------------------------- #
# ScoredResult
# ---------------------------------------------------------------------------#

@dataclass
class ScoredResult:
    """A retrieval result with its hybrid score breakdown."""
    chunk_id: str
    session_id: str
    text: str = ""          # searchable content
    score: float = 0.0
    fts_score: float = 0.0
    qdrant_score: float = 0.0
    jaccard_score: float = 0.0
    hrr_score: float = 0.0
    rank: int = 0
    backend_hits: List[str] = field(default_factory=list)
    trust_score: float = 1.0
    freshness_decay: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for API responses."""
        return {
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "content": self.text,
            "score": self.score,
            "fts_score": self.fts_score,
            "qdrant_score": self.qdrant_score,
            "jaccard_score": self.jaccard_score,
            "hrr_score": self.hrr_score,
            "rank": self.rank,
            "backend_hits": list(self.backend_hits),
            "source_ref": f"session:{self.session_id}#chunk={self.chunk_id}",
            "metadata": {"trust": self.trust_score, "freshness_decay": self.freshness_decay},
        }


# --------------------------------------------------------------------------- #
# Freshness decay (exposed for tests)
# ---------------------------------------------------------------------------#

def freshness_decay(updated_at: Optional[str], half_life_days: float = 90.0) -> float:
    """Apply exponential decay based on content age.

    Returns 1.0 when updated_at is missing.
    """
    if not updated_at:
        return 1.0
    try:
        if updated_at.endswith("Z"):
            updated_at = updated_at[:-1] + "+00:00"
        dt = datetime.fromisoformat(updated_at).replace(tzinfo=timezone.utc)
    except ValueError:
        return 1.0
    now = datetime.now(timezone.utc)
    age_days = (now - dt).total_seconds() / 86400.0
    return 0.5 ** (age_days / half_life_days) if half_life_days else 1.0


# --------------------------------------------------------------------------- #
# Jaccard similarity (exposed for tests)
# ---------------------------------------------------------------------------#

def jaccard_similarity(query: str, content: str) -> float:
    """Token-level Jaccard similarity between query and content strings.

    Returns a float in [0, 1]: 0 = no overlap, 1 = identical token sets.
    Phase 4 Story 4.1.1: full implementation.
    """
    if not query or not content:
        return 0.0

    def _tokens(text: str) -> set:
        return set(re.findall(r"\b\w+\b", text.lower()))

    q_tokens = _tokens(query)
    c_tokens = _tokens(content)

    if not q_tokens and not c_tokens:
        return 0.0

    intersection = len(q_tokens & c_tokens)
    union = len(q_tokens | c_tokens)
    return intersection / union if union else 0.0


# --------------------------------------------------------------------------- #
# Content hash (dedup key)
# ---------------------------------------------------------------------------#

def _content_hash(text: str) -> str:
    """Compute 16-char SHA-256 hex digest of text for dedup key."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Backend score normalization
# ---------------------------------------------------------------------------#

def _normalize_bm25(bm25_score: float) -> float:
    """Convert FTS5 BM25 (lower is better) to [0, 1] similarity.

    BM25 is negative for ranked results; zero or positive means perfect match.
    The scale maps abs(-20) to 0 (worst), abs(0) to 1 (best).
    Clamped to [0, 1] always.
    """
    if bm25_score >= 0:
        return 1.0
    abs_score = abs(bm25_score)
    scale = 20.0
    return max(0.0, min(1.0, 1.0 - abs_score / scale))


def _normalize_qdrant(score: float) -> float:
    """Qdrant cosine similarity is already in [0, 1]."""
    return max(0.0, min(1.0, score))


# --------------------------------------------------------------------------- #
# Per-backend search functions (exposed for mocking)
# ---------------------------------------------------------------------------#

def _search_fts(
    query: str,
    filters: Dict[str, Any],
    limit: int,
    memory_db: Any,
) -> List[Dict[str, Any]]:
    """Call fts5_search against the chunks table."""
    global fts5_search
    try:
        if fts5_search is None:
            fts5_search = _reload_fts5_search()
        return fts5_search(
            query=query,
            filters=filters,
            table="chunks",
            limit=limit,
            memory_db=memory_db,
        )
    except Exception as exc:
        logger.warning("FTS search failed: %s", exc)
        return []


def _search_qdrant(
    query: str,
    filters: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    """Call semantic_search against Qdrant."""
    global semantic_search
    try:
        if semantic_search is None:
            semantic_search = _reload_semantic_search()
        return semantic_search(query=query, filters=filters, limit=limit)
    except Exception as exc:
        logger.warning("Qdrant search failed: %s", exc)
        return []


# --------------------------------------------------------------------------- #
# Hybrid scorer logic
# ---------------------------------------------------------------------------#

def _score(
    fts_norm: float,
    qdrant_norm: float,
    jaccard_sim: float,
    hrr_sim: float,
    trust_score: float,
    freshness: float,
    mode: str,
) -> float:
    """Compute weighted combined score = sum(w * norm) * trust * freshness."""
    weights = _MODE_WEIGHTS.get(mode, _DEFAULT_WEIGHTS)
    relevance = (
        weights["fts"]      * fts_norm
        + weights["qdrant"]  * qdrant_norm
        + weights["jaccard"] * jaccard_sim
        + weights["hrr"]     * hrr_sim
    )
    return relevance * trust_score * freshness


def _deduplicate(results: List[ScoredResult]) -> List[ScoredResult]:
    """Merge results sharing the same (chunk_id, session_id, content_hash).

    When duplicates collapse, the highest-scoring entry wins and its
    backend_hits list is unioned from all agreeing backends.
    """
    seen: Dict[str, ScoredResult] = {}
    for r in results:
        key = f"{r.chunk_id}:{r.session_id}:{_content_hash(r.text)}"
        if key not in seen:
            seen[key] = r
        else:
            existing = seen[key]
            if r.score > existing.score:
                # Update all score fields to new winner's values
                existing.text = r.text
                existing.score = r.score
                existing.fts_score = r.fts_score
                existing.qdrant_score = r.qdrant_score
                existing.jaccard_score = r.jaccard_score
                existing.hrr_score = r.hrr_score
                existing.trust_score = r.trust_score
                existing.freshness_decay = r.freshness_decay
            # Always merge backend_hits from incoming result
            for hit in r.backend_hits:
                if hit not in existing.backend_hits:
                    existing.backend_hits.append(hit)
    return list(seen.values())


def _dedup_results(
    fts_results: List[Dict[str, Any]],
    qdrant_results: List[Dict[str, Any]],
) -> Dict[str, ScoredResult]:
    """Merge FTS and Qdrant results by content hash.

    Uses content hash as the primary dedup key so identical text from
    different backends (e.g., same turn indexed as FTS chunk and Qdrant
    chunk with different ids) is merged into a single ScoredResult with
    per-backend scores unioned.
    """
    candidates: Dict[str, ScoredResult] = {}

    # Process FTS results
    for hit in fts_results:
        chunk_id = hit.get("chunk_id", hit.get("turn_id", ""))
        session_id = hit.get("session_id", "")
        content = hit.get("chunk_text") or hit.get("content", "")
        raw_score = hit.get("rank", 0.0)
        fts_norm = _normalize_bm25(raw_score)
        updated_at = hit.get("timestamp")

        key = _content_hash(content)
        if key not in candidates:
            candidates[key] = ScoredResult(
                chunk_id=chunk_id,
                session_id=session_id,
                text=content,
                score=0.0,
                fts_score=fts_norm,
                qdrant_score=0.0,
                jaccard_score=0.0,
                hrr_score=0.0,
                backend_hits=["fts"],
                trust_score=1.0,
                freshness_decay=freshness_decay(updated_at) if updated_at else 1.0,
            )
        else:
            existing = candidates[key]
            if fts_norm > existing.fts_score:
                existing.fts_score = fts_norm
            if "fts" not in existing.backend_hits:
                existing.backend_hits.append("fts")

    # Process Qdrant results
    for hit in qdrant_results:
        chunk_id = hit.get("metadata", {}).get("chunk_id", "")
        session_id = hit.get("metadata", {}).get("session_id", "")
        content = hit.get("content", "")
        qdrant_score = _normalize_qdrant(hit.get("score", 0.0))
        updated_at = hit.get("metadata", {}).get("date")

        key = _content_hash(content)
        if key not in candidates:
            candidates[key] = ScoredResult(
                chunk_id=chunk_id,
                session_id=session_id,
                text=content,
                score=0.0,
                fts_score=0.0,
                qdrant_score=qdrant_score,
                jaccard_score=0.0,
                hrr_score=0.0,
                backend_hits=["qdrant"],
                trust_score=1.0,
                freshness_decay=freshness_decay(updated_at) if updated_at else 1.0,
            )
        else:
            existing = candidates[key]
            if qdrant_score > existing.qdrant_score:
                existing.qdrant_score = qdrant_score
            if "qdrant" not in existing.backend_hits:
                existing.backend_hits.append("qdrant")

    return candidates


# --------------------------------------------------------------------------- #
# Trust score helper
# ---------------------------------------------------------------------------#

def _fetch_trust(memory_db: Any, chunk_id: str) -> float:
    """Look up trust_score from chunks table. Falls back to 1.0."""
    try:
        conn = memory_db._connect()
        row = conn.execute(
            "SELECT trust_score FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        conn.close()
        if row and row[0] is not None:
            return float(row[0])
    except Exception:
        pass
    return 1.0


# --------------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------------#

def search(
    query: str,
    mode: str = "hybrid",
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    memory_db: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run hybrid search across FTS + Qdrant + Jaccard + HRR backends.

    Returns a dict with:
        - results: list of result dicts, sorted by combined score
        - count: number of results
        - mode: the effective mode used
        - query: the original query
        - degraded_modes: list of backends that failed (if any)
        - backend_weights: the weight dict for the effective mode

    Phase 4 Story 4.1.1: integrates fts5_search and semantic_search.
    Jaccard similarity is fully implemented.
    HRR returns 0.0 (stub — full HRR in story 4.2).
    """
    filters = filters or {}
    limit = max(1, limit)
    degraded_modes: List[str] = []

    effective_mode = mode if mode in _MODE_WEIGHTS else "default"
    backend_weights = dict(_MODE_WEIGHTS[effective_mode])

    # Fetch from each backend
    fts_raw: List[Dict[str, Any]] = []
    qdrant_raw: List[Dict[str, Any]] = []

    try:
        fts_raw = _search_fts(query, filters, limit, memory_db)
    except Exception as exc:
        logger.warning("FTS backend failed: %s", exc)
        degraded_modes.append("fts")

    try:
        qdrant_raw = _search_qdrant(query, filters, limit)
    except Exception as exc:
        logger.warning("Qdrant backend failed: %s", exc)
        degraded_modes.append("qdrant")

    # Merge candidates
    candidates = _dedup_results(fts_raw, qdrant_raw)

    # Compute final combined score for each candidate
    scored: List[ScoredResult] = []
    for key, result in candidates.items():
        # Jaccard (fully implemented)
        result.jaccard_score = jaccard_similarity(query, result.text)
        # HRR stub
        result.hrr_score = 0.0
        # Trust from chunks table if available
        if memory_db is not None:
            result.trust_score = _fetch_trust(memory_db, result.chunk_id)
        # Combined score
        result.score = _score(
            fts_norm=result.fts_score,
            qdrant_norm=result.qdrant_score,
            jaccard_sim=result.jaccard_score,
            hrr_sim=result.hrr_score,
            trust_score=result.trust_score,
            freshness=result.freshness_decay,
            mode=effective_mode,
        )
        scored.append(result)

    # Dedup
    deduped = _deduplicate(scored)

    # Sort descending by combined score
    deduped.sort(key=lambda r: r.score, reverse=True)

    # Assign ranks
    for i, r in enumerate(deduped):
        r.rank = i + 1

    # Trim to limit
    results_out = [r.to_dict() for r in deduped[:limit]]

    return {
        "results": results_out,
        "count": len(results_out),
        "mode": effective_mode,
        "query": query,
        "degraded_modes": degraded_modes,
        "backend_weights": backend_weights,
    }


# Alias for backwards compatibility
hybrid_search = search


# --------------------------------------------------------------------------- #
# HybridScorer (backwards-compatible class wrapper)
# ---------------------------------------------------------------------------#

class HybridScorer:
    """Hybrid scorer class with instance-level dedup and mode methods.

    For the canonical module-level search() function, use search().
    """

    _MODE_WEIGHTS = _MODE_WEIGHTS  # class-level reference to module dict

    def __init__(
        self,
        fts_weight: float = 0.30,
        qdrant_weight: float = 0.40,
        jaccard_weight: float = 0.15,
        hrr_weight: float = 0.15,
    ):
        self.fts_weight = fts_weight
        self.qdrant_weight = qdrant_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight

    def _mode_weights(self, mode: str) -> Dict[str, float]:
        """Return weight dict for given mode, falling back to default."""
        return _MODE_WEIGHTS.get(mode, _DEFAULT_WEIGHTS)

    def _redistribute_weights(
        self, weights: Dict[str, float], dead_backends: List[str]
    ) -> Dict[str, float]:
        """Redistribute weight from dead backends proportionally to alive ones."""
        alive = {k: v for k, v in weights.items() if k not in dead_backends}
        if not alive:
            return {"fts": 0.25, "qdrant": 0.25, "jaccard": 0.25, "hrr": 0.25}
        total = sum(alive.values())
        if total == 0:
            return {k: 1.0 / len(alive) for k in alive}
        return {k: v / total for k, v in alive.items()}

    def _freshness_decay(
        self, updated_at: Optional[str], half_life_days: float = 90.0
    ) -> float:
        """Apply freshness decay; delegates to module-level function."""
        return freshness_decay(updated_at, half_life_days)

    def _deduplicate(self, results: List[ScoredResult]) -> List[ScoredResult]:
        """Instance method wrapping module-level _deduplicate."""
        return _deduplicate(results)

    def _combined_score(
        self,
        fts_score: float,
        qdrant_score: float,
        jaccard_score: float,
        hrr_score: float,
        trust_score: float,
        freshness_decay: float,
        mode: str,
    ) -> float:
        """Compute weighted combined score."""
        return _score(fts_score, qdrant_score, jaccard_score, hrr_score,
                      trust_score, freshness_decay, mode)

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        memory_db: Optional[Any] = None,
    ) -> List[ScoredResult]:
        """Run hybrid search. Returns list of ScoredResult (not wrapped dict)."""
        result = search(query=query, mode=mode, filters=filters,
                       limit=limit, memory_db=memory_db)
        return [ScoredResult(**r) for r in result["results"]]