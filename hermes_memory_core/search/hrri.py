"""HRR-backed probe/related/reason for hermes-memory.

Forked from plugins/memory/holographic/retrieval.py (T-022 context).

Unlike the holographic plugin which stores per-bank vectors in memory_banks,
this implementation stores hrr_vector directly on each fact row and queries
them via MemoryDB.get_facts_with_hrr_vectors().

Algebraic operations use hrr.bind / hrr.unbind / hrr.similarity from hrr.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from hermes_memory_core.search import hrr as _hrr

if TYPE_CHECKING:
    from hermes_memory_core.store.sqlite import MemoryDB

logger = logging.getLogger(__name__)

# Role atoms — must match encode_fact() in hrr.py
_ROLE_ENTITY = "__hrr_role_entity__"
_ROLE_CONTENT = "__hrr_role_content__"
_EMPTY = "__hrr_empty__"


class FactRetriever:
    """HRR-backed multi-strategy fact retrieval.

    Provides three compositional query modes that use phase-vector algebra
    to traverse structured memory representations:

      probe   — "what facts does this entity play a structural role in?"
      related — "what facts share structural connections with this entity?"
      reason  — "what facts are related to ALL of these entities simultaneously?"

    Falls back to FTS5 when numpy is unavailable or no HRR vectors exist.
    """

    def __init__(
        self,
        memory_db: "MemoryDB",
        hrr_dim: int = 1024,
    ) -> None:
        self.db = memory_db
        self.hrr_dim = hrr_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe(
        self,
        entity: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Compositional entity query via HRR algebra.

        Unbinds the entity+role probe key from each fact vector to extract
        residual signal, then scores by similarity to content role vector.

        Fallback: FTS5 keyword search when numpy unavailable or no vectors.
        """
        if not _hrr._HAS_NUMPY:
            return self._fts_fallback(entity, category, limit)

        role_entity_vec = _hrr.encode_atom(_ROLE_ENTITY, self.hrr_dim)
        entity_vec = _hrr.encode_atom(entity.lower(), self.hrr_dim)
        probe_key = _hrr.bind(entity_vec, role_entity_vec)

        facts = self.db.get_facts_with_hrr_vectors(category=category, limit=500)
        if not facts:
            return self._fts_fallback(entity, category, limit)

        role_content_vec = _hrr.encode_atom(_ROLE_CONTENT, self.hrr_dim)

        scored: List[Dict[str, Any]] = []
        for fact in facts:
            hrr_bytes = fact.get("hrr_vector")
            if not hrr_bytes:
                continue
            fact_vec = _hrr.bytes_to_phases(hrr_bytes)
            residual = _hrr.unbind(fact_vec, probe_key)
            content_vec = _hrr.bind(
                _hrr.encode_text(fact.get("content", ""), self.hrr_dim),
                role_content_vec,
            )
            sim = _hrr.similarity(residual, content_vec)
            score = (sim + 1.0) / 2.0 * fact.get("trust_score", 0.5)
            fact_dict = {k: v for k, v in fact.items() if k != "hrr_vector"}
            fact_dict["score"] = round(score, 4)
            scored.append(fact_dict)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def related(
        self,
        entity: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Discover facts that share structural connections with an entity.

        Unlike probe() (finds facts *about* an entity), related() finds
        facts where the entity appears in either entity-role or content-role
        position — structural neighbors rather than topical matches.

        Fallback: FTS5 keyword search when numpy unavailable or no vectors.
        """
        if not _hrr._HAS_NUMPY:
            return self._fts_fallback(entity, category, limit)

        entity_vec = _hrr.encode_atom(entity.lower(), self.hrr_dim)
        role_entity_vec = _hrr.encode_atom(_ROLE_ENTITY, self.hrr_dim)
        role_content_vec = _hrr.encode_atom(_ROLE_CONTENT, self.hrr_dim)

        facts = self.db.get_facts_with_hrr_vectors(category=category, limit=500)
        if not facts:
            return self._fts_fallback(entity, category, limit)

        scored: List[Dict[str, Any]] = []
        for fact in facts:
            hrr_bytes = fact.get("hrr_vector")
            if not hrr_bytes:
                continue
            fact_vec = _hrr.bytes_to_phases(hrr_bytes)
            # Unbind bare entity (not role-bound — catch any structural position)
            residual = _hrr.unbind(fact_vec, entity_vec)
            # Score residual against both role vectors, take the better match
            entity_sim = _hrr.similarity(residual, role_entity_vec)
            content_sim = _hrr.similarity(residual, role_content_vec)
            best_sim = max(entity_sim, content_sim)
            score = (best_sim + 1.0) / 2.0 * fact.get("trust_score", 0.5)
            fact_dict = {k: v for k, v in fact.items() if k != "hrr_vector"}
            fact_dict["score"] = round(score, 4)
            scored.append(fact_dict)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def reason(
        self,
        entities: List[str],
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Multi-entity compositional query — algebraic vector-space JOIN.

        Computes entity residual for each entity, then scores each fact by
        the MINIMUM similarity across all entities (AND semantics: a fact
        must have structural presence of every entity to score highly).

        Example: reason(["peppi", "backend"]) → facts where both peppi
        AND backend play structural roles simultaneously.

        Fallback: FTS5 keyword search when numpy unavailable or no vectors.
        """
        if not _hrr._HAS_NUMPY or not entities:
            query = " ".join(entities) if entities else ""
            return self._fts_fallback(query, category, limit)

        role_entity_vec = _hrr.encode_atom(_ROLE_ENTITY, self.hrr_dim)
        role_content_vec = _hrr.encode_atom(_ROLE_CONTENT, self.hrr_dim)

        entity_residuals: List[Any] = []
        for entity in entities:
            entity_vec = _hrr.encode_atom(entity.lower(), self.hrr_dim)
            probe_key = _hrr.bind(entity_vec, role_entity_vec)
            entity_residuals.append(probe_key)

        facts = self.db.get_facts_with_hrr_vectors(category=category, limit=500)
        if not facts:
            query = " ".join(entities)
            return self._fts_fallback(query, category, limit)

        scored: List[Dict[str, Any]] = []
        for fact in facts:
            hrr_bytes = fact.get("hrr_vector")
            if not hrr_bytes:
                continue
            fact_vec = _hrr.bytes_to_phases(hrr_bytes)

            entity_scores: List[float] = []
            for probe_key in entity_residuals:
                residual = _hrr.unbind(fact_vec, probe_key)
                sim = _hrr.similarity(residual, role_content_vec)
                entity_scores.append(sim)

            # AND semantics: score is limited by the weakest link
            min_sim = min(entity_scores)
            score = (min_sim + 1.0) / 2.0 * fact.get("trust_score", 0.5)
            fact_dict = {k: v for k, v in fact.items() if k != "hrr_vector"}
            fact_dict["score"] = round(score, 4)
            scored.append(fact_dict)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fts_fallback(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Keyword search fallback when HRR vectors unavailable."""
        results = self.db.fts5_search_facts(
            query=query,
            category=category,
            min_trust=0.3,
            limit=limit,
        )
        # Normalise to same shape as HRR results
        for r in results:
            r["score"] = r.pop("fts_rank", 0.0)
        return results