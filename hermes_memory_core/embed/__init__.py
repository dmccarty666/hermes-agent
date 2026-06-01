"""LMS embedding client for Hermes Local Memory.

Uses the local LMS server at ``192.168.2.105:1235`` with
``text-embedding-nomic-embed-text-v1.5`` (dimension 768).

Phase 1 (T-001): stub. Full client in Phase 3 (story 3.1).
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

LMS_ENDPOINT = "http://192.168.2.105:1235/v1"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
EMBED_DIM = 768


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class LMSClient:
    """Client for the local LMS embedding endpoint."""

    def __init__(
        self,
        endpoint: str = LMS_ENDPOINT,
        model: str = EMBED_MODEL,
        dimension: int = EMBED_DIM,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.dimension = dimension
        self._session: Optional[object] = None

    def embed(self, text: str) -> List[float]:
        """Return the embedding vector for a single text.

        Raises:
            EmbeddingError: if the endpoint is unavailable or returns an error.
        """
        raise NotImplementedError("LMSClient.embed() — Phase 3 story 3.1")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a batch of texts.

        Raises:
            EmbeddingError: if the endpoint is unavailable or returns an error.
        """
        raise NotImplementedError("LMSClient.embed_batch() — Phase 3 story 3.1")

    def health_check(self) -> bool:
        """Return True if the LMS endpoint is reachable."""
        return False  # stub