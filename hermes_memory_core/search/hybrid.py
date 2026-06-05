"""Hybrid retrieval scorer for Hermes Local Memory.

Combines FTS5 keyword, Qdrant semantic, Jaccard structural, and HRR scores.
Phase 1 (T-001): stub scorer. Full implementation in Phase 4 (story 4.1).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_memory_core.embed import get_embedding_client, EmbeddingError
from hermes_memory_core.store.sqlite import get_memory_store

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB = Path.home() / ".hermes" / "memory" / "index" / "memory.sqlite"


@dataclass
class ScoredResult:
    """A retrieval result with its hybrid score breakdown."""
    chunk_id: str
    session_id: str
    text: str
    score: float
    fts_score: float = 0.0
    qdrant_score: float = 0.0
    jaccard_score: float = 0.0
    hrr_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "text": self.text,
            "score": round(self.score, 6),
            "fts_score": round(self.fts_score, 6),
            "qdrant_score": round(self.qdrant_score, 6),
            "jaccard_score": round(self.jaccard_score, 6),
            "hrr_score": round(self.hrr_score, 6),
            "rank": self.rank,
        }


class HybridScorer:
    """Hybrid retrieval scorer combining multiple rankers.

    Phase 1 (T-001): stub. Full scorer in Phase 4 (story 4.1).
    """

    def __init__(
        self,
        fts_weight: float = 0.30,
        qdrant_weight: float = 0.40,
        jaccard_weight: float = 0.15,
        hrr_weight: float = 0.15,
        db_path: Path | str | None = None,
    ):
        self.fts_weight = fts_weight
        self.qdrant_weight = qdrant_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight
        self._db_path = db_path or DEFAULT_DB
        self._embed_client = None

    @property
    def embed_client(self):
        """Lazy-load the embedding client."""
        if self._embed_client is None:
            self._embed_client = get_embedding_client()
        return self._embed_client

    def _fts_search(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """FTS5 keyword search over chunks and facts."""
        try:
            store = get_memory_store(self._db_path)
            conn = store._connect()
            try:
                # Search chunks FTS
                chunk_rows = conn.execute("""
                    SELECT c.chunk_id, c.session_id, c.text,
                           c.qdrant_point_id, c.embed_model,
                           bm25(chunks_fts) as fts_rank
                    FROM chunks_fts
                    JOIN chunks c ON c.rowid = chunks_fts.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY fts_rank
                    LIMIT ?
                """, (query, limit)).fetchall()

                results = []
                for row in chunk_rows:
                    results.append({
                        "chunk_id": row[0],
                        "session_id": row[1],
                        "text": row[2],
                        "qdrant_point_id": row[3],
                        "embed_model": row[4],
                        "fts_rank": row[5],
                        "source": "chunk",
                    })

                # Also search facts FTS
                fact_rows = conn.execute("""
                    SELECT f.fact_id, f.fact_text, f.project,
                           bm25(facts_fts) as fts_rank
                    FROM facts_fts
                    JOIN facts f ON f.rowid = facts_fts.rowid
                    WHERE facts_fts MATCH ?
                    ORDER BY fts_rank
                    LIMIT ?
                """, (query, limit)).fetchall()

                for row in fact_rows:
                    results.append({
                        "chunk_id": row[0],
                        "session_id": row[1],
                        "text": row[2],
                        "fts_rank": row[3],
                        "source": "fact",
                    })

                return results
            finally:
                conn.close()
        except Exception as e:
            logger.debug("FTS search failed: %s", e)
            return []

    def _semantic_search(
        self,
        query_vector: List[float],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Qdrant semantic search over both chunks and facts collections."""
        try:
            from hermes_memory_core.store.qdrant import QdrantStore
            qdrant = QdrantStore()
            if not qdrant.is_available():
                return []
            results = qdrant.search(query_vector, top_k=limit, collection="hermes_memory_facts_nomic_v15")
            # Also search chunks for hybrid coverage
            chunk_results = qdrant.search(query_vector, top_k=limit, collection="hermes_memory_chunks_nomic_v15")
            # Tag results by source so hybrid ranker can weight them
            for r in results:
                r["_search_collection"] = "facts"
            for r in chunk_results:
                r["_search_collection"] = "chunks"
            return results + chunk_results
        except Exception as e:
            logger.debug("Qdrant search unavailable: %s", e)
            return []

    def _jaccard_score(self, query: str, text: str) -> float:
        """Compute Jaccard similarity between query and text word sets."""
        if not text:
            return 0.0
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        if not query_words or not text_words:
            return 0.0
        intersection = query_words & text_words
        union = query_words | text_words
        return len(intersection) / len(union) if union else 0.0

    def _compute_fts_score(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize FTS BM25 ranks to 0-1 scores (higher = more relevant)."""
        if not results:
            return results
        # BM25 returns negative values (lower = more relevant).
        # We want high score = more relevant, so invert.
        min_rank = min(r.get("fts_rank", 0) for r in results)
        max_rank = max(r.get("fts_rank", 0) for r in results)
        rank_range = max_rank - min_rank
        if rank_range == 0:
            rank_range = 1
        for r in results:
            rank = r.get("fts_rank", 0)
            # Normalize: 0 = worst (max_rank), 1 = best (min_rank)
            r["fts_score"] = (max_rank - rank) / rank_range
        return results

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[ScoredResult]:
        """Run hybrid search across keyword + semantic + structural rankers.

        Phase 1 (T-001): stub — returns empty list.
        Full implementation combining FTS5 + semantic + Jaccard.
        """
        if not query or not query.strip():
            return []

        filters = filters or {}
        limit = min(limit, 100)

        all_results: Dict[str, Dict[str, Any]] = {}

        # 1. FTS5 keyword search
        fts_results = self._fts_search(query, limit=limit * 2)
        fts_results = self._compute_fts_score(query, fts_results)
        for r in fts_results:
            key = r["chunk_id"]
            all_results[key] = {
                "chunk_id": r["chunk_id"],
                "session_id": r.get("session_id", ""),
                "text": r["text"],
                "fts_score": r.get("fts_score", 0.0),
                "source": r.get("source", "chunk"),
            }

        # 2. Semantic search (Qdrant) — only if query embedding succeeds
        try:
            query_vector = self.embed_client.embed(query)
            semantic_results = self._semantic_search(query_vector, limit=limit)
            for r in semantic_results:
                key = r.get("chunk_id", r.get("id", ""))
                # Fact results have fact_text in payload, chunks have text at top level
                text = r.get("text") or r.get("payload", {}).get("fact_text", "") or r.get("payload", {}).get("text", "")
                if key in all_results:
                    all_results[key]["qdrant_score"] = r.get("score", 0.0)
                else:
                    all_results[key] = {
                        "chunk_id": key,
                        "session_id": r.get("session_id", ""),
                        "text": text,
                        "fts_score": 0.0,
                        "qdrant_score": r.get("score", 0.0),
                        "source": "qdrant",
                    }
        except EmbeddingError:
            logger.debug("Embedding failed, continuing without semantic scores")
        except Exception as e:
            logger.debug("Semantic search failed: %s", e)

        # 3. Jaccard structural score for all results
        for key, r in all_results.items():
            r["jaccard_score"] = self._jaccard_score(query, r["text"])

        # 4. Compute hybrid score
        scored_results: List[ScoredResult] = []
        for key, r in all_results.items():
            hybrid_score = (
                self.fts_weight * r.get("fts_score", 0.0) +
                self.qdrant_weight * r.get("qdrant_score", 0.0) +
                self.jaccard_weight * r.get("jaccard_score", 0.0)
            )
            scored_results.append(ScoredResult(
                chunk_id=r["chunk_id"],
                session_id=r.get("session_id", ""),
                text=r["text"],
                score=hybrid_score,
                fts_score=r.get("fts_score", 0.0),
                qdrant_score=r.get("qdrant_score", 0.0),
                jaccard_score=r.get("jaccard_score", 0.0),
                hrr_score=0.0,  # HRR not yet wired
            ))

        # Sort by score descending, assign ranks
        scored_results.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(scored_results[:limit]):
            r.rank = i + 1

        return scored_results[:limit]


# ── Module-level singleton ─────────────────────────────────────────────────────

_scorer: HybridScorer | None = None


def get_hybrid_scorer(
    fts_weight: float = 0.30,
    qdrant_weight: float = 0.40,
    jaccard_weight: float = 0.15,
    hrr_weight: float = 0.15,
    db_path: Path | str | None = None,
) -> HybridScorer:
    """Return a process-wide HybridScorer singleton."""
    global _scorer
    if _scorer is None:
        _scorer = HybridScorer(
            fts_weight=fts_weight,
            qdrant_weight=qdrant_weight,
            jaccard_weight=jaccard_weight,
            hrr_weight=hrr_weight,
            db_path=db_path,
        )
    return _scorer