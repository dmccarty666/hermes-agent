"""Qdrant vector store for Hermes Local Memory.

Owns Qdrant collections ``hermes_memory_chunks_nomic_v15`` (versioned suffix).
Phase 1 (T-001): stub client. Full implementation in Phase 3 (story T-015).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

logger = logging.getLogger(__name__)

QDRANT_HOST = "192.168.2.52"
QDRANT_PORT = 6333
EMBED_DIM = 768  # nomic-embed-text-v1.5
VERSION_SUFFIX = "nomic_v15"

# Four collections for Phase 3
COLLECTION_CHUNKS = f"hermes_memory_chunks_{VERSION_SUFFIX}"
COLLECTION_SUMMARIES = f"hermes_memory_summaries_{VERSION_SUFFIX}"
COLLECTION_FACTS = f"hermes_memory_facts_{VERSION_SUFFIX}"
COLLECTION_DECISIONS = f"hermes_memory_decisions_{VERSION_SUFFIX}"

ALL_COLLECTIONS = [COLLECTION_CHUNKS, COLLECTION_SUMMARIES, COLLECTION_FACTS, COLLECTION_DECISIONS]

# Payload fields that need indexes for filtering
PAYLOAD_INDEX_FIELDS = ["project", "date", "memory_type", "session_id", "tags", "status"]

# On-disk marker directory
_CONFIG_DIR = "memory/config"
_INITIALIZED_MARKER = "qdrant_initialized"


def _marker_path() -> Path:
    """Return path to the on-disk initialization marker."""
    try:
        from hermes_constants import get_hermes_home
    except ImportError:
        hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
        hermes_home = Path(hermes_home)
    else:
        hermes_home = Path(get_hermes_home())
    return hermes_home / _CONFIG_DIR / _INITIALIZED_MARKER


class QdrantStore:
    """Client for Qdrant vector storage.

    Phase 1 (T-001): stub — all methods raise NotImplementedError.
    Full implementation in Phase 3 (T-015).
    """

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection: str = COLLECTION_CHUNKS,
    ):
        self.host = host
        self.port = port
        self.collection = collection
        self._client: Optional[QdrantClient] = None

    def _get_client(self) -> QdrantClient:
        """Lazily create the Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port, timeout=10)
        return self._client

    def is_available(self) -> bool:
        """Return True if Qdrant is reachable."""
        try:
            client = self._get_client()
            client.get_collections()
            return True
        except Exception:
            return False

    def upsert(self, points: List[Dict[str, Any]]) -> None:
        """Insert / update embedding points."""
        raise NotImplementedError("QdrantStore.upsert() — Phase 3 story T-015")

    def search(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for nearest-neighbour vectors."""
        raise NotImplementedError("QdrantStore.search() — Phase 3 story T-015")

    def delete_collection(self) -> None:
        """Drop the collection (used during rebuild)."""
        raise NotImplementedError("QdrantStore.delete_collection() — Phase 3 story T-015")

    def create_collection(self, dimension: int) -> None:
        """Create the collection if it does not exist."""
        raise NotImplementedError("QdrantStore.create_collection() — Phase 3 story T-015")


# ---------------------------------------------------------------------------
# Top-level init_collections — idempotent, creates all four collections
# ---------------------------------------------------------------------------


def init_collections(
    host: str = QDRANT_HOST,
    port: int = QDRANT_PORT,
    embed_dim: int = EMBED_DIM,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Create all four Phase 3 Qdrant collections with payload indexes.

    Idempotent: checks for the on-disk marker before attempting creation.
    A second call is a no-op (returns 'already_initialized').

    Args:
        host: Qdrant HTTP host.
        port: Qdrant HTTP port.
        embed_dim: Vector dimension (nomic-embed-text-v1.5 = 768).
        dry_run: If True, validate but do not write anything.

    Returns:
        Dict with keys: 'status' ('already_initialized' | 'created'),
        'collections' (list of names), 'errors' (list of error strings).

    Raises:
        QdrantInitError: if embed_dim is invalid (<=0) or collection creation fails.
    """
    marker = _marker_path()

    if marker.exists():
        return {"status": "already_initialized", "collections": ALL_COLLECTIONS, "errors": []}

    if embed_dim <= 0:
        raise QdrantInitError(f"Invalid embed_dim: {embed_dim!r}. Must be a positive integer.")

    client = QdrantClient(host=host, port=port, timeout=30)
    errors: List[str] = []

    # Verify connectivity first
    try:
        client.get_collections()
    except Exception as e:
        raise QdrantInitError(f"Cannot connect to Qdrant at {host}:{port}: {e}")

    if dry_run:
        return {"status": "dry_run", "collections": ALL_COLLECTIONS, "errors": []}

    for name in ALL_COLLECTIONS:
        _create_collection_with_indexes(client, name, embed_dim, errors)

    if not errors:
        # Write marker only if all collections succeeded
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"version": VERSION_SUFFIX, "embed_dim": embed_dim}))
        status = "created"
    else:
        status = "partial" if errors else "created"

    return {"status": status, "collections": ALL_COLLECTIONS, "errors": errors}


def _create_collection_with_indexes(
    client: QdrantClient,
    collection_name: str,
    embed_dim: int,
    errors: List[str],
) -> None:
    """Create one collection with vector params + all payload indexes. Appends errors to list."""
    try:
        # Check if already exists
        existing = client.collection_exists(collection_name)
        if existing:
            logger.debug("Collection %s already exists — skipping", collection_name)
            return

        # Create collection with vector config
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )

        # Add payload indexes for filtering fields
        # tags is a list of strings → use text index
        # status/project/date/memory_type/session_id are keywords
        index_field_names = [f for f in PAYLOAD_INDEX_FIELDS if f != "tags"]
        for field_name in index_field_names:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )

        # tags: array of strings — use text index for full-text search
        client.create_payload_index(
            collection_name=collection_name,
            field_name="tags",
            field_schema=PayloadSchemaType.TEXT,
        )

        logger.info("Created collection %s (dim=%d, indexes on %s)", collection_name, embed_dim, PAYLOAD_INDEX_FIELDS)

    except Exception as e:
        errors.append(f"{collection_name}: {e}")
        logger.error("Failed to create collection %s: %s", collection_name, e)


class QdrantInitError(Exception):
    """Raised when Qdrant initialization fails with a clear error message."""
    pass