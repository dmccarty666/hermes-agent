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


# -------------------------------------------------------------------------- #
# PageRank entity centrality cache
# ---------------------------------------------------------------------------#

# Module-level cache for entity pagerank scores (lazy, TTL-based expiry)
_pagerank_cache: dict[str, float] | None = None
_pagerank_cache_time: float | None = None  # monotonic timestamp when cache was built

_PAGERANK_CACHE_TTL: float = 300.0  # seconds (5 minutes)


def _get_entity_pagerank(entity_name: str) -> float:
    """Return cached PageRank score for an entity name, or 0.0 if unknown.

    The cache is populated on first call and expires after _PAGERANK_CACHE_TTL
    seconds (default 5 min), after which it is rebuilt on the next access.
    """
    global _pagerank_cache, _pagerank_cache_time
    import time
    now = time.monotonic()
    if _pagerank_cache is not None and _pagerank_cache_time is not None:
        if now - _pagerank_cache_time < _PAGERANK_CACHE_TTL:
            return _pagerank_cache.get(entity_name, 0.0)
    # Cache miss or expired — recompute
    from hermes_memory_core.dream.graph import EntityGraph
    from hermes_memory_core.store.sqlite import get_memory_store
    from pathlib import Path
    try:
        store = get_memory_store(Path.home() / ".hermes/memory/index/memory.sqlite")
        eg = EntityGraph(store)
        ranked = eg.page_rank(top_k=500)
        _pagerank_cache = {name: score for name, score in ranked}
        _pagerank_cache_time = now
    except Exception:
        _pagerank_cache = {}
        _pagerank_cache_time = now
    return _pagerank_cache.get(entity_name, 0.0)


def _compute_centrality_boost(
    store, chunk_id: str, centrality_weight: float = 0.05
) -> float:
    """Get the max PageRank score of any entity linked to this fact.

    Queries fact_entities for all entities in this fact, looks up each entity's
    PageRank, returns the max multiplied by centrality_weight.
    Returns 0.0 if no linked entities or on error.
    """
    if centrality_weight <= 0:
        return 0.0
    try:
        conn = store._connect()
        entity_ids = conn.execute(
            "SELECT e.name FROM fact_entities fe JOIN entities e ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
            (chunk_id,),
        ).fetchall()
        conn.close()
        if not entity_ids:
            return 0.0
        scores = [_get_entity_pagerank(name) for (name,) in entity_ids]
        return centrality_weight * max(scores)
    except Exception:
        return 0.0


# -------------------------------------------------------------------------- #
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


# -------------------------------------------------------------------------- #
# ScoredResult
# ---------------------------------------------------------------------------#

@dataclass
class ScoredResult:
    """A retrieval result with its hybrid score breakdown."""
    chunk_id: str
    session_id: str
    content: str = ""           # searchable text
    score: float = 0.0
    fts_score: float = 0.0
    qdrant_score: float = 0.0
    jaccard_score: float = 0.0
    hrr_score: float = 0.0
    rank: int = 0
    backend_hits: List[str] = field(default_factory=list)
    trust_score: float = 1.0
    freshness_decay: float = 1.0
    source_ref: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for API responses."""
        return {
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "content": self.content,
            "score": self.score,
            "fts_score": self.fts_score,
            "qdrant_score": self.qdrant_score,
            "jaccard_score": self.jaccard_score,
            "hrr_score": self.hrr_score,
            "backend_hits": self.backend_hits,
            "source_ref": self.source_ref,
            "metadata": {
                "trust": self.trust_score,
                "freshness_decay": self.freshness_decay,
                **self.metadata,
            },
        }


# -------------------------------------------------------------------------- #
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


# -------------------------------------------------------------------------- #
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


# -------------------------------------------------------------------------- #
# Content hash (dedup key)
# ---------------------------------------------------------------------------#

def _content_hash(text: str) -> str:
    """Compute SHA-256 hex digest of text for dedup key; truncated to 16 hex chars."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


# -------------------------------------------------------------------------- #
# Backend score normalization
# ---------------------------------------------------------------------------#

def _normalize_bm25(bm25_score: float) -> float:
    """Convert FTS5 BM25 (lower is better) to [0, 1] similarity.

    BM25 is negative (more negative = worse); 0 is a perfect match.
    Positive scores are clamped to 1.0.
    """
    if bm25_score >= 0.0:
        return 1.0
    abs_score = abs(bm25_score)
    scale = 20.0
    return max(0.0, min(1.0, 1.0 - abs_score / scale))


def _normalize_qdrant(score: float) -> float:
    """Qdrant cosine similarity is already in [0, 1]."""
    return max(0.0, min(1.0, score))


# -------------------------------------------------------------------------- #
# Per-backend search functions (exposed for mocking)
# ---------------------------------------------------------------------------#

def _search_fts(
    query: str,
    filters: Dict[str, Any],
    limit: int,
    memory_db: Any,
) -> List[Dict[str, Any]]:
    """Call fts5_search against the chunks + facts + decisions tables.

    Why all three? Hybrid retrieval should surface any memory regardless of
    where it lives. The chunks table is the canonical source for turn-derived
    content, while facts/decisions hold curated knowledge that dream produced.
    We merge raw hits here; ``_dedup_results`` collapses true duplicates by
    content hash downstream.

    Per-table failures are isolated (logged + skipped) so e.g. an empty
    chunks FTS doesn't kill a fact-only query.
    """
    global fts5_search
    if fts5_search is None:
        fts5_search = _reload_fts5_search()

    merged: List[Dict[str, Any]] = []
    for table in ("chunks", "facts", "decisions"):
        try:
            rows = fts5_search(
                query=query,
                filters=filters,
                table=table,
                limit=limit,
                memory_db=memory_db,
            )
        except Exception as exc:
            logger.warning("FTS search failed (table=%s): %s", table, exc)
            continue
        # Normalize the content + id keys so _dedup_results can handle facts
        # and decisions the same way it handles chunks.
        for r in rows:
            if table == "facts":
                r.setdefault("content", r.get("fact_text", ""))
                r.setdefault("chunk_id", r.get("fact_id", ""))
                r.setdefault("session_id", "")
            elif table == "decisions":
                r.setdefault("content", r.get("decision_text", ""))
                r.setdefault("chunk_id", r.get("decision_id", ""))
                r.setdefault("session_id", "")
            else:  # chunks
                r.setdefault("content", r.get("chunk_text", ""))
            r["_fts_table"] = table
        merged.extend(rows)

    return merged


def _search_qdrant(
    query: str,
    filters: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    """Call semantic_search against Qdrant across chunks + facts + decisions.

    semantic_search() takes a single ``collection`` arg, so we call it once
    per memory-bearing collection and merge. Per-collection failures are
    isolated.
    """
    global semantic_search
    if semantic_search is None:
        semantic_search = _reload_semantic_search()

    collections = (
        ("hermes_memory_chunks_nomic_v15", "chunk"),
        ("hermes_memory_facts_nomic_v15", "fact"),
        ("hermes_memory_decisions_nomic_v15", "decision"),
    )

    merged: List[Dict[str, Any]] = []
    for collection, kind in collections:
        try:
            hits = semantic_search(
                query=query,
                filters=filters,
                limit=limit,
                collection=collection,
            )
        except TypeError:
            # Older signature (no `collection` kwarg) — only the chunks
            # collection is queried.
            if kind != "chunk":
                continue
            try:
                hits = semantic_search(query=query, filters=filters, limit=limit)
            except Exception as exc:
                logger.warning("Qdrant search failed (collection=%s): %s", collection, exc)
                continue
        except Exception as exc:
            logger.warning("Qdrant search failed (collection=%s): %s", collection, exc)
            continue

        # Tag origin so callers / debug logs know which collection produced
        # the hit, and patch the content field for fact/decision payloads
        # (which use ``text`` instead of ``content`` from semantic.py defaults).
        for h in hits:
            h["_qdrant_collection"] = collection
            if not h.get("content"):
                payload_text = (h.get("metadata") or {}).get("text") or h.get("text")
                if payload_text:
                    h["content"] = payload_text
        merged.extend(hits)

    return merged


# -------------------------------------------------------------------------- #
# Weight redistribution (graceful degradation)
# ---------------------------------------------------------------------------#

def _redistribute_weights(
    base_weights: Dict[str, float],
    degraded: List[str],
) -> Dict[str, float]:
    """Redistribute weights when backends fail.

    When one or more backends are unavailable, their weights are redistributed
    proportionally to the remaining backends so the total = 1.0.

    Phase 4 Story 4.1.2 — graceful degradation.
    """
    if not degraded:
        return dict(base_weights)

    available = {k: v for k, v in base_weights.items() if k not in degraded}
    if not available:
        # All backends down — return zeros with degraded标记
        return {k: 0.0 for k in base_weights}

    total_available = sum(available.values())
    if total_available == 0.0:
        return {k: 0.0 for k in base_weights}

    # Redistribute degraded weights proportionally to available backends
    redistributed = {}
    for k, v in base_weights.items():
        if k in degraded:
            redistributed[k] = 0.0
        else:
            # Proportional share of the freed weight
            redistributed[k] = v / total_available

    return redistributed


# -------------------------------------------------------------------------- #
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
    weights: Optional[Dict[str, float]] = None,
    centrality_boost: float = 0.0,
) -> float:
    """Compute weighted combined score = sum(w * norm) * trust * freshness.

    centrality_boost is added to the relevance before trust/freshness scaling.
    """
    if weights is None:
        weights = _MODE_WEIGHTS.get(mode, _DEFAULT_WEIGHTS)
    relevance = (
        weights["fts"]      * fts_norm
        + weights["qdrant"]  * qdrant_norm
        + weights["jaccard"] * jaccard_sim
        + weights["hrr"]     * hrr_sim
    )
    relevance += centrality_boost
    return relevance * trust_score * freshness


def _deduplicate(results: List[ScoredResult]) -> List[ScoredResult]:
    """Merge results sharing the same (chunk_id, session_id, content_hash).

    When duplicates collapse, the highest-scoring entry wins and its
    backend_hits list is unioned from all agreeing backends.
    """
    seen: Dict[str, ScoredResult] = {}
    for r in results:
        key = f"{r.chunk_id}:{r.session_id}:{_content_hash(r.content)}"
        if key not in seen:
            seen[key] = r
        else:
            existing = seen[key]
            if r.score > existing.score:
                seen[key] = r
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
        source_ref = hit.get("source_ref", f"chunk:{chunk_id}")
        raw_score = hit.get("rank", 0.0)
        fts_norm = _normalize_bm25(raw_score)
        updated_at = hit.get("timestamp")
        fts_table = hit.get("_fts_table", "chunks")

        key = _content_hash(content)
        if key not in candidates:
            candidates[key] = ScoredResult(
                chunk_id=chunk_id,
                session_id=session_id,
                content=content,
                score=0.0,
                fts_score=fts_norm,
                qdrant_score=0.0,
                jaccard_score=0.0,
                hrr_score=0.0,
                backend_hits=["fts"],
                source_ref=source_ref,
                metadata={"timestamp": updated_at, "backend": "fts", "fts_table": fts_table},
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
        source_ref = hit.get("source_ref", f"chunk:{chunk_id}")
        qdrant_score = _normalize_qdrant(hit.get("score", 0.0))
        updated_at = hit.get("metadata", {}).get("date")

        key = _content_hash(content)
        if key not in candidates:
            candidates[key] = ScoredResult(
                chunk_id=chunk_id,
                session_id=session_id,
                content=content,
                score=0.0,
                fts_score=0.0,
                qdrant_score=qdrant_score,
                jaccard_score=0.0,
                hrr_score=0.0,
                backend_hits=["qdrant"],
                source_ref=source_ref,
                metadata={"date": updated_at, "backend": "qdrant"},
            )
        else:
            existing = candidates[key]
            if qdrant_score > existing.qdrant_score:
                existing.qdrant_score = qdrant_score
            if "qdrant" not in existing.backend_hits:
                existing.backend_hits.append("qdrant")

    return candidates


# -------------------------------------------------------------------------- #
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


# -------------------------------------------------------------------------- #
# Public API — search() function
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
        - results: list of ScoredResult dicts, sorted by combined score
        - count: number of results
        - mode: the mode used
        - query: the original query
        - backend_weights: the weights used for this mode
        - degraded_modes: list of backends that failed (if any)

    Phase 4 Story 4.1.2 — graceful degradation: auto-redistribute weights when
    a backend reports unavailable. Returns `degraded_modes: [...]` in response.
    """
    filters = filters or {}
    limit = max(1, limit)
    degraded_modes: List[str] = []
    base_weights = _MODE_WEIGHTS.get(mode, _DEFAULT_WEIGHTS)

    fts_raw: List[Dict[str, Any]] = []
    qdrant_raw: List[Dict[str, Any]] = []
    hrr_score = 0.0

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

    # Check HRR availability (numpy required)
    try:
        from hermes_memory_core.search import hrr as _hrr_mod
        if not getattr(_hrr_mod, "_HAS_NUMPY", True):
            degraded_modes.append("hrr")
    except Exception:
        degraded_modes.append("hrr")

    # Redistribute weights if any backends are down
    effective_weights = _redistribute_weights(base_weights, degraded_modes)

    candidates = _dedup_results(fts_raw, qdrant_raw)

    scored: List[ScoredResult] = []
    for key, result in candidates.items():
        result.jaccard_score = jaccard_similarity(query, result.content)
        result.hrr_score = hrr_score
        if memory_db is not None:
            result.trust_score = _fetch_trust(memory_db, result.chunk_id)
        result.freshness_decay = freshness_decay(
            result.metadata.get("timestamp") or result.metadata.get("date")
        )
        # Compute centrality boost for facts with linked entities
        centrality_boost = 0.0
        if memory_db is not None and result.metadata.get("fts_table") == "facts":
            centrality_boost = _compute_centrality_boost(memory_db, result.chunk_id)
            result.metadata["centrality_boost"] = centrality_boost
        result.score = _score(
            fts_norm=result.fts_score,
            qdrant_norm=result.qdrant_score,
            jaccard_sim=result.jaccard_score,
            hrr_sim=result.hrr_score,
            trust_score=result.trust_score,
            freshness=result.freshness_decay,
            mode=mode,
            weights=effective_weights,
            centrality_boost=centrality_boost,
        )
        scored.append(result)

    deduped = _deduplicate(scored)
    deduped.sort(key=lambda r: r.score, reverse=True)
    results_out = [r.to_dict() for r in deduped[:limit]]

    return {
        "results": results_out,
        "count": len(results_out),
        "mode": mode,
        "query": query,
        "backend_weights": effective_weights,
        "degraded_modes": degraded_modes,
    }


# Alias for backwards compatibility
hybrid_search = search


# -------------------------------------------------------------------------- #
# HybridScorer (backwards-compatible class wrapper)
# ---------------------------------------------------------------------------#

class HybridScorer:
    """Backwards-compatible hybrid scorer class.

    Wraps the module-level :func:`search` function. New code should use
    ``hybrid_search(query, mode, filters, limit, memory_db)`` directly.
    """

    # Class-level weight table (for test compatibility)
    _MODE_WEIGHTS: Dict[str, Dict[str, float]] = {
        "default":    {"fts": 0.30, "qdrant": 0.40, "jaccard": 0.15, "hrr": 0.15},
        "keyword":    {"fts": 0.70, "qdrant": 0.10, "jaccard": 0.20, "hrr": 0.00},
        "semantic":   {"fts": 0.05, "qdrant": 0.80, "jaccard": 0.10, "hrr": 0.05},
        "hybrid":     {"fts": 0.30, "qdrant": 0.40, "jaccard": 0.15, "hrr": 0.15},
        "facts_only": {"fts": 0.40, "qdrant": 0.30, "jaccard": 0.15, "hrr": 0.15},
    }

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

    def _freshness_decay(self, updated_at: Optional[str], half_life_days: float = 90.0) -> float:
        """Instance method wrapper around module-level freshness_decay."""
        return freshness_decay(updated_at, half_life_days)

    def _deduplicate(self, results: List[ScoredResult]) -> List[ScoredResult]:
        """Instance method wrapper around module-level _deduplicate."""
        return _deduplicate(results)

    def _combined_score(
        self,
        fts_score: float,
        qdrant_score: float,
        jaccard_score: float,
        hrr_score: float,
        trust_score: float,
        freshness_decay: float,
        mode: str = "hybrid",
    ) -> float:
        """Compute weighted combined score = sum(w * norm) * trust * freshness."""
        return _score(
            fts_norm=fts_score,
            qdrant_norm=qdrant_score,
            jaccard_sim=jaccard_score,
            hrr_sim=hrr_score,
            trust_score=trust_score,
            freshness=freshness_decay,
            mode=mode,
        )