"""Hybrid retrieval scorer for Hermes Local Memory.

Combines FTS5 keyword, Qdrant semantic, Jaccard structural, and HRR scores.
Phase 1 (T-001): stub scorer. Full implementation in Phase 4 (story 4.1).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
    ) -> List[ScoredResult]:
        """Run hybrid search across keyword + semantic + structural rankers.

        Phase 1 (T-001): stub — returns empty list.
        """
        return []  # stub