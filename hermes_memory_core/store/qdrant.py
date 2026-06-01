"""Qdrant vector store for Hermes Local Memory.

Owns Qdrant collections ``hermes_memory_chunks_nomic_v15`` (versioned suffix).
Phase 1 (T-001): stub client. Full integration in Phase 3 (story 3.2).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
DEFAULT_COLLECTION = "hermes_memory_chunks_nomic_v15"


class QdrantStore:
    """Client for Qdrant vector storage.

    Phase 1 (T-001): stub — all methods raise NotImplementedError.
    Full implementation in Phase 3 (story 3.2).
    """

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection: str = DEFAULT_COLLECTION,
    ):
        self.host = host
        self.port = port
        self.collection = collection
        self._client: Optional[object] = None

    def is_available(self) -> bool:
        """Return True if Qdrant is reachable."""
        return False  # stub

    def upsert(self, points: List[Dict[str, Any]]) -> None:
        """Insert / update embedding points."""
        raise NotImplementedError("QdrantStore.upsert() — Phase 3 story 3.2")

    def search(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for nearest-neighbour vectors."""
        raise NotImplementedError("QdrantStore.search() — Phase 3 story 3.2")

    def delete_collection(self) -> None:
        """Drop the collection (used during rebuild)."""
        raise NotImplementedError("QdrantStore.delete_collection() — Phase 3 story 3.2")

    def create_collection(self, dimension: int) -> None:
        """Create the collection if it does not exist."""
        raise NotImplementedError("QdrantStore.create_collection() — Phase 3 story 3.2")