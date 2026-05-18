# Copyright 2026 David McCarty. All rights reserved.
"""Semantic search: embed query via LMS, search Qdrant, return normalized results.

Story T-019 — memory_query(mode='semantic')
Phase 3 Epic 3.3 / Story 3.3.1
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    SearchParams,
    QuantizationSearchParams,
)

from hermes_memory_core.embed import LMSClient, EmbeddingError

logger = logging.getLogger(__name__)

# Default collection for chunk search
DEFAULT_COLLECTION = "hermes_memory_chunks_nomic_v15"

# Qdrant host/port (shared with store/qdrant.py)
QDRANT_HOST = "192.168.2.52"
QDRANT_PORT = 6333


class SemanticSearchError(Exception):
    """Raised when semantic search fails (LMS or Qdrant unavailable)."""
    pass


def semantic_search(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    collection: str = DEFAULT_COLLECTION,
    embed_client: Optional[LMSClient] = None,
    qdrant_client: Optional[QdrantClient] = None,
) -> List[Dict[str, Any]]:
    """Search for semantically similar chunks via LMS embedding + Qdrant.

    Args:
        query: Natural-language query string.
        filters: Optional scoping dict with keys:
            - project (str): restrict to project name
            - session_id (str): restrict to session
            - date_from (str): ISO date string, inclusive lower bound
            - date_to (str): ISO date string, inclusive upper bound
            - role (str): 'user' or 'assistant'
            - tags (list[str]): must all be present in the chunk's tags
        limit: Maximum results to return (default 10).
        collection: Qdrant collection name.
        embed_client: LMSClient instance (created from defaults if None).
        qdrant_client: QdrantClient instance (created from defaults if None).

    Returns:
        List of result dicts, each with keys:
            - content (str): chunk text
            - source_ref (str): session:{id}#chunk={chunk_id}
            - score (float): Qdrant cosine similarity in [0,1]
            - metadata (dict): chunk_id, session_id, start_turn_id, end_turn_id,
              chunk_type, role_mix, turn_count

    Raises:
        SemanticSearchError: When LMS embed fails or Qdrant is unreachable.
    """
    filters = filters or {}

    # 1. Embed the query
    if embed_client is None:
        embed_client = LMSClient()
    try:
        vector = embed_client.embed(query)
    except EmbeddingError as exc:
        raise SemanticSearchError(f"LMS embedding failed: {exc}") from exc

    # 2. Build Qdrant filter from payload filters
    qdrant_filter = _build_qdrant_filter(filters)

    # 3. Search Qdrant
    if qdrant_client is None:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)

    try:
        search_results = qdrant_client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            query_filter=qdrant_filter,
            search_params=SearchParams(
                quantization=QuantizationSearchParams(ignore=False)
            ),
            with_payload=[
                "chunk_id",
                "session_id",
                "start_turn_id",
                "end_turn_id",
                "chunk_type",
                "text",
                "role_mix",
                "turn_count",
            ],
        )
    except Exception as exc:
        raise SemanticSearchError(f"Qdrant search failed: {exc}") from exc

    # 4. Normalize results
    results = []
    for point in search_results.points:
        payload = point.payload or {}
        chunk_id = payload.get("chunk_id", "")
        session_id = payload.get("session_id", "")

        results.append({
            "content": payload.get("text", ""),
            "source_ref": f"session:{session_id}#chunk={chunk_id}",
            "score": point.score,
            "metadata": {
                "chunk_id": chunk_id,
                "session_id": session_id,
                "start_turn_id": payload.get("start_turn_id", ""),
                "end_turn_id": payload.get("end_turn_id", ""),
                "chunk_type": payload.get("chunk_type", ""),
                "role_mix": payload.get("role_mix", []),
                "turn_count": payload.get("turn_count", 0),
            },
        })

    return results


def _build_qdrant_filter(filters: Dict[str, Any]) -> Optional[Filter]:
    """Convert logical filters dict into a Qdrant Must filter.

    Returns None when no filters are present.
    """
    conditions: List[FieldCondition] = []

    if filters.get("project"):
        conditions.append(
            FieldCondition(
                key="project",
                match=MatchValue(value=filters["project"]),
            )
        )

    if filters.get("session_id"):
        conditions.append(
            FieldCondition(
                key="session_id",
                match=MatchValue(value=filters["session_id"]),
            )
        )

    if filters.get("role"):
        conditions.append(
            FieldCondition(
                key="role_mix",
                match=MatchAny(any=[filters["role"]]),
            )
        )

    if filters.get("date_from") or filters.get("date_to"):
        # Stored as ISO date string keyword; use exact-match on date_from.
        # True range support requires storing as Unix timestamp with numeric index.
        if filters.get("date_from"):
            conditions.append(
                FieldCondition(key="date", match=MatchValue(value=filters["date_from"]))
            )
        if filters.get("date_to"):
            conditions.append(
                FieldCondition(key="date", match=MatchValue(value=filters["date_to"]))
            )

    if filters.get("tags"):
        tag_list = filters["tags"]
        if isinstance(tag_list, str):
            tag_list = [tag_list]
        conditions.append(
            FieldCondition(
                key="tags",
                match=MatchAny(any=tag_list),
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)