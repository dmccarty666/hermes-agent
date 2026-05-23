"""
LMS embedding client for Hermes Local Memory.

Uses the LOCAL LMS server at ``http://localhost:1235`` with
``text-embedding-nomic-embed-text-v1.5`` (dimension 768).
Embedding runs on the LOCAL LMStudio instance (not the remote Spark2 GPU
machine) to eliminate any cross-host network contention.
"""

from __future__ import annotations

import time
import logging
from typing import List

import requests

logger = logging.getLogger(__name__)

# Embeddings: LOCAL LMStudio — never change this to a remote IP.
# The dreamer pipeline calls this concurrently with agent inference on
# Spark2's GPU; routing embeddings over localhost removes that contention.
LMS_ENDPOINT = "http://localhost:1235"
LMS_URL = f"{LMS_ENDPOINT}/v1/embeddings"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
EMBED_DIM = 768
MAX_RETRIES = 3
RETRY_BASE_DELAY = 0.5  # seconds


class EmbeddingError(Exception):
    """Raised when embedding request fails after all retries."""

    pass


class LMSClient:
    """
    LM Studio embedding client.

    Embeds text into 768-dimensional float vectors via the LMS embeddings API.
    Supports batch embedding, health checks, and retry with exponential backoff.
    """

    def __init__(
        self,
        url: str = LMS_URL,
        model: str = EMBED_MODEL,
        dimension: int = EMBED_DIM,
        max_retries: int = MAX_RETRIES,
        timeout: float = 30.0,
    ) -> None:
        self.url = url
        self.model = model
        self.dimension = dimension
        self.max_retries = max_retries
        self.timeout = timeout

    def embed(self, text: str) -> List[float]:
        """
        Embed a single text string into a 768-dimensional float vector.

        Args:
            text: Input text to embed.

        Returns:
            List of 768 floats representing the text embedding.

        Raises:
            EmbeddingError: After all retries fail, with a clear message.
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        last_err: Exception = EmbeddingError("unknown error")
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.url,
                    json={
                        "input": text,
                        "model": self.model,
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    embedding = data["data"][0]["embedding"]
                    if len(embedding) != self.dimension:
                        raise EmbeddingError(
                            f"Expected {self.dimension}d vector, got {len(embedding)}d"
                        )
                    return embedding
                # Transient server error — retry
                if response.status_code >= 500:
                    last_err = requests.HTTPError(
                        f"LMS returned {response.status_code}: {response.text[:200]}",
                        response=response,
                    )
                    logger.warning(
                        "LMSClient attempt %d/%d: HTTP %d — retrying",
                        attempt + 1, self.max_retries, response.status_code,
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                        continue
                # Client error — don't retry
                raise EmbeddingError(
                    f"LMS returned {response.status_code}: {response.text[:200]}"
                )
            except requests.exceptions.Timeout:
                last_err = requests.exceptions.Timeout(
                    f"Connection to LMS timed out after {self.timeout}s"
                )
                logger.warning(
                    "LMSClient attempt %d/%d: timeout — retrying",
                    attempt + 1, self.max_retries,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
            except requests.exceptions.ConnectionError as exc:
                last_err = exc
                logger.warning(
                    "LMSClient attempt %d/%d: connection error — retrying",
                    attempt + 1, self.max_retries,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
            except requests.HTTPError:
                raise
            except EmbeddingError:
                raise

        # All retries exhausted
        raise EmbeddingError(
            f"LMSClient failed after {self.max_retries} attempts. Last error: {last_err}"
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of N texts, returning N vectors in order.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embeddings, one per input text in order.

        Raises:
            EmbeddingError: If any embedding fails.
        """
        if not texts:
            return []

        # Filter empty strings — LMS API may reject them
        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not non_empty:
            # All texts were empty — return zero vectors for empty strings
            return [[0.0] * self.dimension for _ in texts]

        # Send all non-empty in one request
        non_empty_texts = [t for _, t in non_empty]
        last_err: Exception = EmbeddingError("unknown error")
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.url,
                    json={
                        "input": non_empty_texts,
                        "model": self.model,
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    embeddings = data["data"]
                    # embeddings are sorted by index in LMS response
                    result: List[List[float]] = []
                    emb_idx = 0
                    for i, text in enumerate(texts):
                        if text and text.strip():
                            embedding = embeddings[emb_idx]["embedding"]
                            if len(embedding) != self.dimension:
                                raise EmbeddingError(
                                    f"Expected {self.dimension}d vector at index {i}, "
                                    f"got {len(embedding)}d"
                                )
                            result.append(embedding)
                            emb_idx += 1
                        else:
                            result.append([0.0] * self.dimension)
                    return result
                if response.status_code >= 500:
                    last_err = requests.HTTPError(
                        f"LMS returned {response.status_code}", response=response,
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                        continue
                raise EmbeddingError(
                    f"LMS returned {response.status_code}: {response.text[:200]}"
                )
            except requests.exceptions.Timeout:
                last_err = requests.exceptions.Timeout(
                    f"Connection to LMS timed out after {self.timeout}s"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
            except requests.exceptions.ConnectionError as exc:
                last_err = exc
                if attempt < self.max_retries - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
            except EmbeddingError:
                raise

        raise EmbeddingError(
            f"embed_batch failed after {self.max_retries} attempts. Last error: {last_err}"
        )

    def health_check(self) -> dict:
        """
        Check LMS health and return model info.

        Returns:
            dict with 'model' and 'dim' keys.

        Raises:
            EmbeddingError: If LMS is unreachable.
        """
        model_name = self.model
        try:
            response = requests.get(
                f"{LMS_ENDPOINT}/v1/models",
                timeout=5.0,
            )
            if response.status_code == 200:
                models_data = response.json()
                if models_data.get("data"):
                    model_name = models_data["data"][0].get("id", self.model)
        except requests.exceptions.RequestException:
            pass

        # Confirm with a minimal embed call
        try:
            response = requests.post(
                self.url,
                json={"input": "health", "model": self.model},
                headers={"Content-Type": "application/json"},
                timeout=10.0,
            )
            if response.status_code != 200:
                raise EmbeddingError(
                    f"LMS health check failed: HTTP {response.status_code}"
                )
            data = response.json()
            embedding = data["data"][0]["embedding"]
            return {"model": model_name, "dim": len(embedding)}
        except requests.exceptions.Timeout:
            raise EmbeddingError("LMS health check timed out")
        except requests.exceptions.ConnectionError:
            raise EmbeddingError(f"LMS is unreachable at {LMS_ENDPOINT}")


# ── Backwards-compatible alias (DoD references LMSEmbedder) ─────────────────
LMSEmbedder = LMSClient
LMSEmbeddingError = EmbeddingError