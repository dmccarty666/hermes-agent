"""Tests for hermes_memory_core/store/qdrant.py — T-015: Qdrant collections (versioned).

Tests:
  - init_collections() creates all four collections with correct vector dim and payload indexes
  - Second call is a no-op (idempotent via on-disk marker)
  - Invalid embed_dim raises QdrantInitError with a clear message
  - All four collection names follow the versioned naming convention
  - Payload indexes: project, date, memory_type, session_id, tags, status

These tests use a real Qdrant instance at 192.168.2.105:6333 against a
dedicated test collection suffix so they are safe to run against the live server.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

# Ensure the venv python is on the path for the module import
import sys

venv_python = Path("/home/dmccarty/.hermes/hermes-agent/venv/bin/python3")
if venv_python.exists():
    sys.executable = str(venv_python)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams, PayloadIndexInfo

from hermes_memory_core.store.qdrant import (
    QDRANT_HOST,
    QDRANT_PORT,
    EMBED_DIM,
    VERSION_SUFFIX,
    COLLECTION_CHUNKS,
    COLLECTION_SUMMARIES,
    COLLECTION_FACTS,
    COLLECTION_DECISIONS,
    ALL_COLLECTIONS,
    PAYLOAD_INDEX_FIELDS,
    QdrantStore,
    init_collections,
    QdrantInitError,
)


# ---------------------------------------------------------------------------
# Test collection suffix — all test collections get this suffix so they can be
# dropped after the test without affecting production collections.
# ---------------------------------------------------------------------------

_TEST_SUFFIX = f"_test_{uuid.uuid4().hex[:8]}"


def _test_collection_name(name: str) -> str:
    """Append _test_<uuid> to a collection name."""
    return f"{name}{_TEST_SUFFIX}"


@pytest.fixture(scope="module")
def qdrant_client():
    """Return a real QdrantClient connected to the live server."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=15)
    # Verify connectivity
    try:
        client.get_collections()
    except Exception as e:
        pytest.skip(f"Qdrant not reachable at {QDRANT_HOST}:{QDRANT_PORT}: {e}")
    return client


@pytest.fixture(scope="module")
def qdrant_test_collections(qdrant_client):
    """Yield test collection names, then drop them all after the test."""
    names = [_test_collection_name(n) for n in ALL_COLLECTIONS]
    # We don't create them here — tests do that via init_collections override
    yield names
    # Teardown: drop all test collections
    for name in names:
        try:
            qdrant_client.delete_collection(name)
        except Exception:
            pass


@pytest.fixture
def marker_path(tmp_path, monkeypatch):
    """Override the marker path to use a temp directory instead of ~/.hermes."""
    from hermes_memory_core.store import qdrant

    test_marker = tmp_path / "memory" / "config" / "qdrant_initialized"
    monkeypatch.setattr(qdrant, "_marker_path", lambda: test_marker)
    return test_marker


# ---------------------------------------------------------------------------
# Tests: collection naming
# ---------------------------------------------------------------------------

class TestCollectionNaming:
    """AC: Four collections with versioned `_nomic_v15` suffix."""

    def test_all_four_collection_names_have_version_suffix(self):
        assert COLLECTION_CHUNKS.endswith(f"_{VERSION_SUFFIX}")
        assert COLLECTION_SUMMARIES.endswith(f"_{VERSION_SUFFIX}")
        assert COLLECTION_FACTS.endswith(f"_{VERSION_SUFFIX}")
        assert COLLECTION_DECISIONS.endswith(f"_{VERSION_SUFFIX}")

    def test_all_collections_are_distinct(self):
        assert len(set(ALL_COLLECTIONS)) == 4

    def test_embed_dim_is_768(self):
        assert EMBED_DIM == 768


# ---------------------------------------------------------------------------
# Tests: init_collections() — first run (creates)
# ---------------------------------------------------------------------------

class TestInitCollectionsCreate:
    """AC: Given Qdrant is running, `init_collections()` creates all four collections."""

    def test_creates_all_four_collections(self, qdrant_client, qdrant_test_collections, marker_path):
        # Override ALL_COLLECTIONS to use test names
        import hermes_memory_core.store.qdrant as qdrant_mod
        original_all = qdrant_mod.ALL_COLLECTIONS
        qdrant_mod.ALL_COLLECTIONS = qdrant_test_collections

        try:
            result = qdrant_mod.init_collections(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                embed_dim=EMBED_DIM,
            )
            assert result["status"] == "created"
            assert len(result["errors"]) == 0

            # Verify each collection exists
            for name in qdrant_test_collections:
                assert qdrant_client.collection_exists(name), f"Collection {name} was not created"
        finally:
            qdrant_mod.ALL_COLLECTIONS = original_all

    def test_collections_have_correct_vector_dim(self, qdrant_client, qdrant_test_collections, marker_path):
        import hermes_memory_core.store.qdrant as qdrant_mod
        original_all = qdrant_mod.ALL_COLLECTIONS
        qdrant_mod.ALL_COLLECTIONS = qdrant_test_collections

        try:
            qdrant_mod.init_collections(host=QDRANT_HOST, port=QDRANT_PORT, embed_dim=EMBED_DIM)

            for name in qdrant_test_collections:
                info = qdrant_client.get_collection(name)
                assert info.config.params.vectors.size == EMBED_DIM, f"Wrong dim for {name}"
                assert (
                    info.config.params.vectors.distance == Distance.COSINE
                ), f"Wrong distance metric for {name}"
        finally:
            qdrant_mod.ALL_COLLECTIONS = original_all

    def test_collections_have_payload_indexes(
        self, qdrant_client, qdrant_test_collections, marker_path
    ):
        """AC: project, date, memory_type, session_id, tags, status are indexable for filtering."""
        import hermes_memory_core.store.qdrant as qdrant_mod
        original_all = qdrant_mod.ALL_COLLECTIONS
        qdrant_mod.ALL_COLLECTIONS = qdrant_test_collections

        try:
            qdrant_mod.init_collections(host=QDRANT_HOST, port=QDRANT_PORT, embed_dim=EMBED_DIM)

            for name in qdrant_test_collections:
                info = qdrant_client.get_collection(name)
                payload_schema = info.payload_schema
                assert payload_schema is not None, f"{name}: no payload schema"
                for field in PAYLOAD_INDEX_FIELDS:
                    assert field in payload_schema, f"{name}: field '{field}' not in payload schema keys: {list(payload_schema.keys())}"
        finally:
            qdrant_mod.ALL_COLLECTIONS = original_all

    def test_marker_file_is_written(self, qdrant_client, qdrant_test_collections, marker_path):
        import hermes_memory_core.store.qdrant as qdrant_mod
        original_all = qdrant_mod.ALL_COLLECTIONS
        qdrant_mod.ALL_COLLECTIONS = qdrant_test_collections

        try:
            assert not marker_path.exists(), "Marker should not exist before init"
            result = qdrant_mod.init_collections(host=QDRANT_HOST, port=QDRANT_PORT, embed_dim=EMBED_DIM)
            assert result["status"] == "created"
            assert marker_path.exists(), "Marker should exist after successful init"
        finally:
            qdrant_mod.ALL_COLLECTIONS = original_all


# ---------------------------------------------------------------------------
# Tests: init_collections() — second run (idempotent / no-op)
# ---------------------------------------------------------------------------

class TestInitCollectionsIdempotent:
    """AC: Calling `init_collections()` a second time is a no-op (marker prevents recreation)."""

    def test_second_call_is_noop(self, qdrant_client, qdrant_test_collections, marker_path):
        import hermes_memory_core.store.qdrant as qdrant_mod
        original_all = qdrant_mod.ALL_COLLECTIONS
        qdrant_mod.ALL_COLLECTIONS = qdrant_test_collections

        try:
            # First call
            result1 = qdrant_mod.init_collections(host=QDRANT_HOST, port=QDRANT_PORT, embed_dim=EMBED_DIM)
            assert result1["status"] == "created"

            # Capture vector config before second call
            info_before = qdrant_client.get_collection(qdrant_test_collections[0])

            # Second call — should be no-op
            result2 = qdrant_mod.init_collections(host=QDRANT_HOST, port=QDRANT_PORT, embed_dim=EMBED_DIM)
            assert result2["status"] == "already_initialized"
            assert result2["errors"] == []

            # Verify collection was NOT modified (still has correct config)
            info_after = qdrant_client.get_collection(qdrant_test_collections[0])
            assert info_before.config.params.vectors.size == info_after.config.params.vectors.size
        finally:
            qdrant_mod.ALL_COLLECTIONS = original_all


# ---------------------------------------------------------------------------
# Tests: invalid embed_dim
# ---------------------------------------------------------------------------

class TestInitCollectionsInvalidDim:
    """AC: Given an invalid vector dim, setup fails with a clear error."""

    def test_zero_dim_raises_qdrant_init_error(self, marker_path):
        import hermes_memory_core.store.qdrant as qdrant_mod

        with pytest.raises(qdrant_mod.QdrantInitError) as exc_info:
            qdrant_mod.init_collections(host=QDRANT_HOST, port=QDRANT_PORT, embed_dim=0)
        assert "embed_dim" in str(exc_info.value).lower()
        assert "0" in str(exc_info.value)

    def test_negative_dim_raises_qdrant_init_error(self, marker_path):
        import hermes_memory_core.store.qdrant as qdrant_mod

        with pytest.raises(qdrant_mod.QdrantInitError) as exc_info:
            qdrant_mod.init_collections(host=QDRANT_HOST, port=QDRANT_PORT, embed_dim=-1)
        assert "embed_dim" in str(exc_info.value).lower()

    def test_non_integer_dim_causes_collection_errors(
        self, marker_path, qdrant_client, qdrant_test_collections
    ):
        """Float dim (768.5) fails pydantic validation at VectorParams creation.

        The error is caught inside _create_collection_with_indexes and returned
        in the 'errors' list rather than raised as QdrantInitError.
        """
        import hermes_memory_core.store.qdrant as qdrant_mod
        original_all = qdrant_mod.ALL_COLLECTIONS
        qdrant_mod.ALL_COLLECTIONS = qdrant_test_collections

        try:
            result = qdrant_mod.init_collections(
                host=QDRANT_HOST, port=QDRANT_PORT, embed_dim=768.5
            )
            # Should return errors, not raise
            assert len(result["errors"]) == 4, f"Expected 4 collection errors, got: {result['errors']}"
            for err in result["errors"]:
                assert "embed_dim" in err.lower() or "integer" in err.lower(), f"Unexpected error: {err}"
        finally:
            qdrant_mod.ALL_COLLECTIONS = original_all


# ---------------------------------------------------------------------------
# Tests: QdrantStore.is_available()
# ---------------------------------------------------------------------------

class TestQdrantStoreAvailable:
    """QdrantStore.is_available() returns True when Qdrant is reachable."""

    def test_is_available_true_when_qdrant_running(self):
        store = QdrantStore(host=QDRANT_HOST, port=QDRANT_PORT)
        # Note: this test makes a real network call
        available = store.is_available()
        # Only assert True if Qdrant is actually reachable; otherwise skip
        if not available:
            pytest.skip(f"Qdrant not reachable at {QDRANT_HOST}:{QDRANT_PORT}")
        assert available is True

    def test_is_available_false_for_invalid_host(self):
        store = QdrantStore(host="invalid.host.example.com", port=QDRANT_PORT)
        assert store.is_available() is False