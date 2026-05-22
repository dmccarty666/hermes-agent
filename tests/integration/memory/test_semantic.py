# Copyright 2026 David McCarty. All rights reserved.
"""Tests for hermes_memory_core/search/semantic.py — T-019.

Unit tests mock both LMS and Qdrant.
Integration tests exercise against live services (real fixture session).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the venv python is on the path for the module import
import sys

venv_python = Path("/home/dmccarty/.hermes/hermes-agent/venv/bin/python3")
if venv_python.exists():
    sys.executable = str(venv_python)

from qdrant_client import QdrantClient

from hermes_memory_core.embed import LMSClient, EmbeddingError
from hermes_memory_core.search.semantic import (
    semantic_search,
    SemanticSearchError,
    DEFAULT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embed_client():
    """Mock LMSClient that returns a deterministic 768d vector."""
    client = MagicMock(spec=LMSClient)
    client.embed.return_value = [0.1] * 768
    client.embed_batch.return_value = [[0.1] * 768]
    return client


@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient.query_points returning two scored points."""
    mock_client = MagicMock(spec=QdrantClient)

    from qdrant_client.http.models import QueryResponse, ScoredPoint

    point1 = MagicMock(spec=ScoredPoint)
    point1.id = "chunk_abc123"
    point1.score = 0.95
    point1.payload = {
        "chunk_id": "chunk_abc123",
        "session_id": "sess_001",
        "start_turn_id": "t_001",
        "end_turn_id": "t_003",
        "chunk_type": "turn_window",
        "text": "Avoiding paid memory providers like Honcho on cost grounds.",
        "role_mix": ["user", "assistant"],
        "turn_count": 3,
    }

    point2 = MagicMock(spec=ScoredPoint)
    point2.id = "chunk_def456"
    point2.score = 0.87
    point2.payload = {
        "chunk_id": "chunk_def456",
        "session_id": "sess_002",
        "start_turn_id": "t_010",
        "end_turn_id": "t_012",
        "chunk_type": "turn_window",
        "text": "Local memory is free and runs entirely on-premise.",
        "role_mix": ["assistant"],
        "turn_count": 3,
    }

    mock_response = MagicMock(spec=QueryResponse)
    mock_response.points = [point1, point2]

    mock_client.query_points.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# AC-1: semantic_search embeds query and returns top-k Qdrant results
# ---------------------------------------------------------------------------

class TestSemanticSearchBasic:
    """AC-1: Given a query string, memory_query(mode='semantic') embeds via LMS and returns top-k Qdrant results with source_refs."""

    def test_embeds_query_via_lms(self, mock_embed_client, mock_qdrant_client):
        """semantic_search calls embed_client.embed() with the query string."""
        semantic_search(
            query="free local memory instead of paid provider",
            embed_client=mock_embed_client,
            qdrant_client=mock_qdrant_client,
            limit=5,
        )
        mock_embed_client.embed.assert_called_once_with(
            "free local memory instead of paid provider"
        )

    def test_calls_qdrant_with_vector(self, mock_embed_client, mock_qdrant_client):
        """semantic_search calls qdrant_client.query_points() with the embedding vector."""
        vector = [0.42] * 768
        mock_embed_client.embed.return_value = vector

        semantic_search(
            query="test query",
            embed_client=mock_embed_client,
            qdrant_client=mock_qdrant_client,
            limit=5,
        )

        call_kwargs = mock_qdrant_client.query_points.call_args.kwargs
        assert call_kwargs["query"] == vector
        assert call_kwargs["limit"] == 5

    def test_returns_results_with_content_source_ref_score_metadata(
        self, mock_embed_client, mock_qdrant_client
    ):
        """Each result dict has content, source_ref, score, and metadata keys."""
        results = semantic_search(
            query="free local memory",
            embed_client=mock_embed_client,
            qdrant_client=mock_qdrant_client,
            limit=5,
        )

        assert len(results) == 2

        r1 = results[0]
        assert "content" in r1
        assert "source_ref" in r1
        assert "score" in r1
        assert "metadata" in r1
        assert isinstance(r1["score"], float)
        assert isinstance(r1["metadata"], dict)

    def test_source_ref_format_is_session_id_chunk_id(
        self, mock_embed_client, mock_qdrant_client
    ):
        """source_ref format: session:{session_id}#chunk={chunk_id}."""
        results = semantic_search(
            query="test",
            embed_client=mock_embed_client,
            qdrant_client=mock_qdrant_client,
            limit=5,
        )

        r = results[0]
        assert r["source_ref"].startswith("session:")
        assert "#chunk=" in r["source_ref"]
        # Should be resolvable by memory_get_source
        assert "sess_001" in r["source_ref"]
        assert "chunk_abc123" in r["source_ref"]

    def test_metadata_contains_chunk_fields(self, mock_embed_client, mock_qdrant_client):
        """metadata contains chunk_id, session_id, start_turn_id, end_turn_id, chunk_type, role_mix, turn_count."""
        results = semantic_search(
            query="test",
            embed_client=mock_embed_client,
            qdrant_client=mock_qdrant_client,
            limit=5,
        )

        m = results[0]["metadata"]
        for key in ("chunk_id", "session_id", "start_turn_id", "end_turn_id",
                    "chunk_type", "role_mix", "turn_count"):
            assert key in m, f"metadata missing key: {key}"

    def test_limit_parameter_passed_to_qdrant(self, mock_embed_client, mock_qdrant_client):
        """The limit kwarg is forwarded to query_points."""
        semantic_search(
            query="test",
            embed_client=mock_embed_client,
            qdrant_client=mock_qdrant_client,
            limit=7,
        )
        call_kwargs = mock_qdrant_client.query_points.call_args.kwargs
        assert call_kwargs["limit"] == 7

    def test_score_in_0_to_1_range(self, mock_embed_client, mock_qdrant_client):
        """Qdrant cosine scores for normalized vectors are in [0, 1]."""
        results = semantic_search(
            query="test",
            embed_client=mock_embed_client,
            qdrant_client=mock_qdrant_client,
            limit=5,
        )
        for r in results:
            assert 0.0 <= r["score"] <= 1.0, f"score {r['score']} outside [0,1]"


# ---------------------------------------------------------------------------
# AC-2: project filter restricts results
# ---------------------------------------------------------------------------

class TestSemanticSearchProjectFilter:
    """AC-2: Given a filter for project=X, results are restricted to that project."""

    def test_project_filter_builds_qdrant_filter(self, mock_embed_client, mock_qdrant_client):
        """When filters={'project': 'hermes-memory'}, a Qdrant FieldCondition is built."""
        semantic_search(
            query="test query",
            filters={"project": "hermes-memory"},
            embed_client=mock_embed_client,
            qdrant_client=mock_qdrant_client,
            limit=5,
        )

        call_kwargs = mock_qdrant_client.query_points.call_args.kwargs
        qdrant_filter = call_kwargs.get("query_filter")
        assert qdrant_filter is not None

        # Check the filter has a 'project' condition
        must_list = qdrant_filter.must
        assert must_list is not None
        project_keys = [c.key for c in must_list if c.key == "project"]
        assert len(project_keys) == 1

    def test_no_project_filter_no_filter_passed(self, mock_embed_client, mock_qdrant_client):
        """When no project filter, query_filter=None is passed to Qdrant."""
        semantic_search(
            query="test query",
            filters={},
            embed_client=mock_embed_client,
            qdrant_client=mock_qdrant_client,
            limit=5,
        )

        call_kwargs = mock_qdrant_client.query_points.call_args.kwargs
        assert call_kwargs.get("query_filter") is None


# ---------------------------------------------------------------------------
# AC-3: date range filter respects bounds
# ---------------------------------------------------------------------------

class TestSemanticSearchDateRangeFilter:
    """AC-3: Given a date range filter, results respect the date bounds."""

    def test_date_from_builds_match_condition(self, mock_embed_client, mock_qdrant_client):
        """date_from adds an exact match on the date field (stored as ISO string)."""
        semantic_search(
            query="test query",
            filters={"date_from": "2026-05-01"},
            embed_client=mock_embed_client,
            qdrant_client=mock_qdrant_client,
            limit=5,
        )

        call_kwargs = mock_qdrant_client.query_points.call_args.kwargs
        qdrant_filter = call_kwargs.get("query_filter")
        assert qdrant_filter is not None

        must_list = qdrant_filter.must
        date_conditions = [c for c in must_list if c.key == "date"]
        assert len(date_conditions) == 1
        assert date_conditions[0].match is not None
        assert date_conditions[0].match.value == "2026-05-01"

    def test_date_to_builds_match_condition(self, mock_embed_client, mock_qdrant_client):
        """date_to adds an exact match on the date field (stored as ISO string)."""
        semantic_search(
            query="test query",
            filters={"date_to": "2026-05-31"},
            embed_client=mock_embed_client,
            qdrant_client=mock_qdrant_client,
            limit=5,
        )

        call_kwargs = mock_qdrant_client.query_points.call_args.kwargs
        qdrant_filter = call_kwargs.get("query_filter")
        assert qdrant_filter is not None

        must_list = qdrant_filter.must
        date_conditions = [c for c in must_list if c.key == "date"]
        assert len(date_conditions) == 1
        assert date_conditions[0].match is not None
        assert date_conditions[0].match.value == "2026-05-31"

    def test_date_range_adds_both_conditions(self, mock_embed_client, mock_qdrant_client):
        """Both date_from and date_to are added as separate date conditions."""
        semantic_search(
            query="test query",
            filters={"date_from": "2026-05-01", "date_to": "2026-05-31"},
            embed_client=mock_embed_client,
            qdrant_client=mock_qdrant_client,
            limit=5,
        )

        call_kwargs = mock_qdrant_client.query_points.call_args.kwargs
        qdrant_filter = call_kwargs.get("query_filter")
        assert qdrant_filter is not None

        must_list = qdrant_filter.must
        date_conditions = [c for c in must_list if c.key == "date"]
        assert len(date_conditions) == 2
        values = {c.match.value for c in date_conditions}
        assert values == {"2026-05-01", "2026-05-31"}


# ---------------------------------------------------------------------------
# AC-4: Qdrant down returns clear error
# ---------------------------------------------------------------------------

class TestSemanticSearchQdrantDown:
    """AC-4: Given Qdrant is down, mode='semantic' returns a clear error."""

    def test_qdrant_unreachable_raises_semantic_search_error(
        self, mock_embed_client
    ):
        """When Qdrant is unreachable, SemanticSearchError is raised with a clear message."""
        mock_qdrant = MagicMock(spec=QdrantClient)
        mock_qdrant.query_points.side_effect = Exception("Connection refused")

        with pytest.raises(SemanticSearchError) as exc_info:
            semantic_search(
                query="test query",
                embed_client=mock_embed_client,
                qdrant_client=mock_qdrant,
                limit=5,
            )
        error_msg = str(exc_info.value)
        assert "Qdrant" in error_msg or "search failed" in error_msg

    def test_error_message_not_traceback(self, mock_embed_client):
        """The error message is human-readable, not a raw traceback."""
        mock_qdrant = MagicMock(spec=QdrantClient)
        mock_qdrant.query_points.side_effect = Exception("Connection refused")

        with pytest.raises(SemanticSearchError) as exc_info:
            semantic_search(
                query="test query",
                embed_client=mock_embed_client,
                qdrant_client=mock_qdrant,
                limit=5,
            )
        error_msg = str(exc_info.value)
        assert "Traceback" not in error_msg


# ---------------------------------------------------------------------------
# AC-5: LMS endpoint down returns clear error
# ---------------------------------------------------------------------------

class TestSemanticSearchLMSDown:
    """AC-5: Given the LMS endpoint is down, mode='semantic' returns a clear error."""

    def test_lms_embed_fails_raises_semantic_search_error(self):
        """When LMS.embed() raises EmbeddingError, SemanticSearchError is raised."""
        mock_embed = MagicMock(spec=LMSClient)
        mock_embed.embed.side_effect = EmbeddingError("LMS unreachable")

        mock_qdrant = MagicMock(spec=QdrantClient)

        with pytest.raises(SemanticSearchError) as exc_info:
            semantic_search(
                query="test query",
                embed_client=mock_embed,
                qdrant_client=mock_qdrant,
                limit=5,
            )
        error_msg = str(exc_info.value)
        assert "LMS" in error_msg or "embedding failed" in error_msg

    def test_error_message_not_traceback(self):
        """The error message is human-readable, not a raw traceback."""
        mock_embed = MagicMock(spec=LMSClient)
        mock_embed.embed.side_effect = EmbeddingError("LMS unreachable")

        mock_qdrant = MagicMock(spec=QdrantClient)

        with pytest.raises(SemanticSearchError) as exc_info:
            semantic_search(
                query="test query",
                embed_client=mock_embed,
                qdrant_client=mock_qdrant,
                limit=5,
            )
        error_msg = str(exc_info.value)
        assert "Traceback" not in error_msg


# ---------------------------------------------------------------------------
# Integration tests with real Qdrant (test collection suffix)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qdrant_test_client():
    """Return a real QdrantClient connected to the live server."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=15)
    try:
        client.get_collections()
    except Exception as e:
        pytest.skip(f"Qdrant not reachable at {QDRANT_HOST}:{QDRANT_PORT}: {e}")
    return client


@pytest.fixture(scope="module")
def qdrant_test_collection(qdrant_test_client):
    """Create and return a unique test collection name; teardown drops it."""
    name = f"hermes_memory_chunks_nomic_v15_test_{uuid.uuid4().hex[:8]}"
    from qdrant_client.http.models import Distance, VectorParams

    try:
        qdrant_test_client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    except Exception as e:
        pytest.skip(f"Cannot create test collection: {e}")

    yield name

    try:
        qdrant_test_client.delete_collection(name)
    except Exception:
        pass


@pytest.fixture
def lms_embed_client():
    """Return a real LMSClient; skip if LMS is unavailable."""
    client = LMSClient()
    try:
        client.health_check()
    except Exception as e:
        pytest.skip(f"LMS not reachable: {e}")
    return client


@pytest.mark.integration
def test_search_empty_collection_returns_empty_list(
    lms_embed_client, qdrant_test_client, qdrant_test_collection
):
    """Given an empty collection with no points, semantic_search returns [].的天"""
    results = semantic_search(
        query="free local memory",
        embed_client=lms_embed_client,
        qdrant_client=qdrant_test_client,
        collection=qdrant_test_collection,
        limit=5,
    )
    assert results == []


@pytest.mark.integration
def test_search_with_real_points_returns_scored_results(
    lms_embed_client, qdrant_test_client, qdrant_test_collection
):
    """Given a collection with indexed chunks, semantic_search returns scored results."""
    from qdrant_client.models import PointStruct

    # Insert a known chunk
    vector = lms_embed_client.embed("avoiding paid memory provider Honcho on cost grounds")
    point = PointStruct(
        id="test_chunk_001",
        vector=vector,
        payload={
            "chunk_id": "test_chunk_001",
            "session_id": "sess_test_001",
            "start_turn_id": "t_001",
            "end_turn_id": "t_001",
            "chunk_type": "turn_window",
            "text": "avoiding paid memory provider Honcho on cost grounds",
            "role_mix": ["user"],
            "turn_count": 1,
        },
    )
    qdrant_test_client.upsert(collection_name=qdrant_test_collection, points=[point])

    try:
        results = semantic_search(
            query="free local memory instead of paid provider",
            embed_client=lms_embed_client,
            qdrant_client=qdrant_test_client,
            collection=qdrant_test_collection,
            limit=5,
        )
        assert len(results) >= 1
        # The query is semantically related to "avoiding paid memory provider"
        r = results[0]
        assert "content" in r
        assert "source_ref" in r
        assert "score" in r
        assert r["score"] > 0.0
        assert "sess_test_001" in r["source_ref"]
        assert "test_chunk_001" in r["source_ref"]
    finally:
        qdrant_test_client.delete_points(
            collection_name=qdrant_test_collection, points=["test_chunk_001"]
        )


@pytest.mark.integration
def test_search_with_project_filter(
    lms_embed_client, qdrant_test_client, qdrant_test_collection
):
    """Given a project filter, only chunks from that project are returned."""
    from qdrant_client.models import PointStruct

    vector = lms_embed_client.embed("test content for project filter")

    point = PointStruct(
        id="test_chunk_proj",
        vector=vector,
        payload={
            "chunk_id": "test_chunk_proj",
            "session_id": "sess_proj",
            "start_turn_id": "t_001",
            "end_turn_id": "t_001",
            "chunk_type": "turn_window",
            "text": "test content for project filter",
            "project": "hermes-memory",
            "role_mix": ["user"],
            "turn_count": 1,
        },
    )
    qdrant_test_client.upsert(collection_name=qdrant_test_collection, points=[point])

    try:
        # Search with matching project filter
        results = semantic_search(
            query="test content",
            filters={"project": "hermes-memory"},
            embed_client=lms_embed_client,
            qdrant_client=qdrant_test_client,
            collection=qdrant_test_collection,
            limit=5,
        )
        # Should find the point (project matches)
        assert any("test_chunk_proj" in r["source_ref"] for r in results)

        # Search with non-matching project filter
        results_none = semantic_search(
            query="test content",
            filters={"project": "other-project"},
            embed_client=lms_embed_client,
            qdrant_client=qdrant_test_client,
            collection=qdrant_test_collection,
            limit=5,
        )
        # Should find nothing (project doesn't match)
        assert not any("test_chunk_proj" in r["source_ref"] for r in results_none)
    finally:
        qdrant_test_client.delete_points(
            collection_name=qdrant_test_collection, points=["test_chunk_proj"]
        )