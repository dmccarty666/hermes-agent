"""
Tests for LMSEmbedder (hermes_memory_core.embed).

These tests hit the real LMS endpoint at 192.168.2.105:1235.
The DoD requires "unit tests mock the LMS endpoint" — these integration
tests run against the live local service to verify correctness. A separate
mock-based test class below provides the mocked-unit coverage.
"""

import pytest
import requests

from hermes_memory_core.embed import (
    LMSClient,
    LMSEmbedder,
    EmbeddingError,
    LMSEmbeddingError,
    EMBED_DIM,
    EMBED_MODEL,
    LMS_ENDPOINT,
    LMS_URL,
)


class TestLMSEmbedder:
    """LMSEmbedder (alias for LMSClient) integration tests against real LMS endpoint."""

    @pytest.mark.integration
    def test_embed_returns_768d_vector(self):
        """Given a sample text string, embed() returns a 768-dimensional float vector."""
        embedder = LMSEmbedder()
        vector = embedder.embed("Hello, world!")
        assert isinstance(vector, list), "embed() should return a list"
        assert len(vector) == 768, f"Expected 768d, got {len(vector)}d"
        assert all(isinstance(x, float) for x in vector), "All elements should be floats"

    @pytest.mark.integration
    def test_embed_unreachable_returns_clear_error(self):
        """Given the LMS endpoint is unreachable, embed() returns a clear error (not a traceback)."""
        # Point at a guaranteed-unreachable address
        bad_embedder = LMSEmbedder(
            url="http://192.168.2.254:9999/v1/embeddings",
            timeout=2.0,
        )
        with pytest.raises(EmbeddingError) as exc_info:
            bad_embedder.embed("test")
        error_msg = str(exc_info.value)
        # Should be a clear human-readable message, not a raw traceback
        assert ("failed after 3 attempts" in error_msg or
                "unreachable" in error_msg.lower() or
                "connection" in error_msg.lower() or
                "ConnectionError" in error_msg)
        assert "Traceback" not in error_msg

    @pytest.mark.integration
    def test_health_check_returns_model_and_dim(self):
        """Given a health-check call, health_check() returns the model name and vector dim."""
        embedder = LMSEmbedder()
        result = embedder.health_check()
        assert isinstance(result, dict), "health_check should return a dict"
        assert "model" in result, "health_check should include 'model' key"
        assert "dim" in result, "health_check should include 'dim' key"
        assert result["dim"] == 768, f"Expected dim=768, got {result['dim']}"
        assert isinstance(result["model"], str), "model should be a string"

    @pytest.mark.integration
    def test_embed_batch_returns_n_vectors_in_order(self):
        """Given a batch of N texts, embed_batch() returns N vectors in order."""
        embedder = LMSEmbedder()
        texts = ["apple", "banana", "cherry", "date", "elderberry"]
        vectors = embedder.embed_batch(texts)
        assert isinstance(vectors, list), "embed_batch should return a list"
        assert len(vectors) == len(texts), (
            f"Expected {len(texts)} vectors, got {len(vectors)}"
        )
        for i, v in enumerate(vectors):
            assert len(v) == 768, f"Vector {i} ({texts[i]}) is {len(v)}d, expected 768d"
        # Each text should embed to a different vector
        assert len(set(tuple(v) for v in vectors)) == len(vectors), (
            "Vectors should all be unique for different texts"
        )

    @pytest.mark.integration
    def test_embed_batch_empty_list(self):
        """Given an empty list, embed_batch() returns an empty list."""
        embedder = LMSEmbedder()
        vectors = embedder.embed_batch([])
        assert vectors == []

    @pytest.mark.integration
    def test_embed_batch_with_empty_strings(self):
        """Given a batch containing empty strings, embed_batch() returns 768d vectors in order."""
        embedder = LMSEmbedder()
        texts = ["hello", "", "world", ""]
        vectors = embedder.embed_batch(texts)
        assert len(vectors) == 4, f"Expected 4 vectors, got {len(vectors)}"
        for v in vectors:
            assert len(v) == 768, f"Expected 768d, got {len(v)}d"
        # Empty strings get zero vectors
        assert vectors[1] == [0.0] * 768, "Empty string should return zero vector"
        assert vectors[3] == [0.0] * 768, "Empty string should return zero vector"

    def test_embed_empty_text_raises(self):
        """Given an empty/whitespace-only string, embed() raises EmbeddingError."""
        embedder = LMSEmbedder()
        with pytest.raises(EmbeddingError) as exc_info:
            embedder.embed("")
        assert "empty" in str(exc_info.value).lower()

        with pytest.raises(EmbeddingError):
            embedder.embed("   ")

    def test_alias_lmsembedder_equals_lmsclient(self):
        """LMSEmbedder is an alias for LMSClient for DoD compatibility."""
        assert LMSEmbedder is LMSClient


class TestLMSClientMocked:
    """Unit tests that mock the LMS HTTP endpoint."""

    def test_embed_successful(self, requests_mock):
        """A successful embed call returns the vector and correct length."""
        requests_mock.post(LMS_URL, json={
            "data": [{"embedding": [0.1] * 768}]
        })
        client = LMSClient()
        vector = client.embed("test text")
        assert len(vector) == 768
        assert vector[0] == 0.1

    def test_embed_retries_on_5xx(self, requests_mock):
        """Transient 5xx errors trigger retry up to 3 times."""
        # Fail twice, then succeed
        requests_mock.post(LMS_URL, [
            {"status_code": 503, "text": "Service Unavailable"},
            {"status_code": 503, "text": "Service Unavailable"},
            {"status_code": 200, "json": {"data": [{"embedding": [0.42] * 768}]}},
        ])
        client = LMSClient()
        vector = client.embed("test")
        assert vector[0] == 0.42
        assert len(requests_mock.request_history) == 3

    def test_embed_no_retry_on_4xx(self, requests_mock):
        """Client errors (4xx) fail immediately without retry."""
        requests_mock.post(LMS_URL, status_code=400, text="Bad Request")
        client = LMSClient()
        with pytest.raises(EmbeddingError) as exc_info:
            client.embed("test")
        assert "400" in str(exc_info.value)
        assert len(requests_mock.request_history) == 1

    def test_health_check_success(self, requests_mock):
        """health_check() returns model and dim from live probe."""
        requests_mock.get(f"{LMS_ENDPOINT}/v1/models", json={
            "data": [{"id": "text-embedding-nomic-embed-text-v1.5"}]
        })
        requests_mock.post(LMS_URL, json={
            "data": [{"embedding": [0.0] * 768}]
        })
        client = LMSClient()
        result = client.health_check()
        assert result["dim"] == 768
        assert isinstance(result["model"], str)

    def test_health_check_raises_on_unreachable(self, requests_mock):
        """health_check() raises EmbeddingError with clear message when LMS is down."""
        requests_mock.get(f"{LMS_ENDPOINT}/v1/models", exc=requests.exceptions.ConnectionError("Connection refused"))
        requests_mock.post(LMS_URL, exc=requests.exceptions.ConnectionError("Connection refused"))
        client = LMSClient()
        with pytest.raises(EmbeddingError) as exc_info:
            client.health_check()
        assert "unreachable" in str(exc_info.value).lower() or "ConnectionError" in str(exc_info.value)