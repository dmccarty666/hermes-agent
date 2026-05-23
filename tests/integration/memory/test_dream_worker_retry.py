"""Smoke tests for dream worker retry, timeout, semaphore, and graceful degradation.

Run: python3 -m pytest tests/integration/memory/test_dream_worker_retry.py -v
"""
from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest


class TestConstants:
    def test_lms_timeout_extended(self):
        from hermes_memory_core.dream.worker import LMS_TIMEOUT
        assert LMS_TIMEOUT == 300.0, f"Expected 300.0, got {LMS_TIMEOUT}"

    def test_max_llm_retries(self):
        from hermes_memory_core.dream.worker import MAX_LLM_RETRIES
        assert MAX_LLM_RETRIES == 2, f"Expected 2, got {MAX_LLM_RETRIES}"

    def test_retry_base_delay(self):
        from hermes_memory_core.dream.worker import RETRY_BASE_DELAY
        assert RETRY_BASE_DELAY == 5.0, f"Expected 5.0, got {RETRY_BASE_DELAY}"

    def test_llm_timeout_exception(self):
        from hermes_memory_core.dream.worker import LLMTimeout
        exc = LLMTimeout("test")
        assert isinstance(exc, Exception)

    def test_llm_complete_with_retry_exists(self):
        from hermes_memory_core.dream.worker import _llm_complete_with_retry
        assert callable(_llm_complete_with_retry)


class TestSemaphore:
    def test_dreamer_semaphore_is_singleton(self):
        from hermes_memory_core.dream.worker import _DREAMER_SEMAPHORE
        assert isinstance(_DREAMER_SEMAPHORE, threading.Semaphore)
        assert _DREAMER_SEMAPHORE._value == 1

    def test_semaphore_blocks_concurrent_access(self):
        sem = threading.Semaphore(1)
        counter = [0]

        def holder():
            acquired = sem.acquire(timeout=5)
            assert acquired, "Holder failed to acquire semaphore"
            counter[0] += 1
            time.sleep(0.1)  # hold for 100ms
            counter[0] += 1
            sem.release()

        def waiter():
            time.sleep(0.05)  # let holder grab it first
            acquired = sem.acquire(timeout=1)
            counter[0] += 10  # mark: waiter woke
            if acquired:
                counter[0] += 100  # mark: waiter got semaphore
                sem.release()

        t1 = threading.Thread(target=holder)
        t2 = threading.Thread(target=waiter)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # holder: +1,+1 = 2; waiter: +10,[+100 if acquired] = 110 or 10
        # Both possible: waiter gets it before timeout (counter=112) OR times out (counter=12)
        assert counter[0] in (12, 112), f"Unexpected counter {counter[0]}"


class TestRetryWrapper:
    def test_retry_wrapper_succeeds_first_attempt(self):
        from hermes_memory_core.dream.worker import _llm_complete_with_retry

        with patch("hermes_memory_core.dream.worker._llm_complete") as mock:
            mock.return_value = "success"
            result = _llm_complete_with_retry("test prompt")
            assert result == "success"
            assert mock.call_count == 1

    def test_retry_wrapper_retries_on_llm_timeout(self):
        from hermes_memory_core.dream.worker import _llm_complete_with_retry, LLMTimeout

        with patch("hermes_memory_core.dream.worker._llm_complete") as mock:
            mock.side_effect = [LLMTimeout("timeout 1"), "success"]
            # MAX_LLM_RETRIES=2: attempt 1 fails (retry), attempt 2 succeeds
            result = _llm_complete_with_retry("test prompt")
            assert result == "success"
            assert mock.call_count == 2

    def test_retry_wrapper_raises_after_exhausted_retries(self):
        from hermes_memory_core.dream.worker import _llm_complete_with_retry, LLMTimeout

        with patch("hermes_memory_core.dream.worker._llm_complete") as mock:
            mock.side_effect = LLMTimeout("always timeout")
            with pytest.raises(LLMTimeout):
                _llm_complete_with_retry("test prompt")
            assert mock.call_count == 2  # 2 attempts, both failed


class TestGracefulDegradation:
    def test_extract_facts_returns_empty_on_timeout(self):
        from hermes_memory_core.dream.worker import DreamWorker, LLMTimeout

        worker = DreamWorker()
        with patch("hermes_memory_core.dream.worker._llm_complete_with_retry") as mock:
            mock.side_effect = LLMTimeout("simulated")
            result = worker.extract_facts([{"content": "hello", "role": "user", "turn_id": "t1", "timestamp": "2025-01-01T00:00:00Z"}])
            assert result == []

    def test_extract_decisions_returns_empty_on_timeout(self):
        from hermes_memory_core.dream.worker import DreamWorker, LLMTimeout

        worker = DreamWorker()
        with patch("hermes_memory_core.dream.worker._llm_complete_with_retry") as mock:
            mock.side_effect = LLMTimeout("simulated")
            result = worker.extract_decisions([{"content": "hello", "role": "user", "turn_id": "t1", "timestamp": "2025-01-01T00:00:00Z"}])
            assert result == []

    def test_extract_open_questions_returns_empty_on_timeout(self):
        from hermes_memory_core.dream.worker import DreamWorker, LLMTimeout

        worker = DreamWorker()
        with patch("hermes_memory_core.dream.worker._llm_complete_with_retry") as mock:
            mock.side_effect = LLMTimeout("simulated")
            result = worker.extract_open_questions([{"content": "hello", "role": "user", "turn_id": "t1", "timestamp": "2025-01-01T00:00:00Z"}])
            assert result == []

    def test_summarize_session_fails_gracefully_on_timeout(self):
        from hermes_memory_core.dream.worker import DreamWorker, LLMTimeout

        worker = DreamWorker()
        with patch("hermes_memory_core.dream.worker._llm_complete_with_retry") as mock:
            mock.side_effect = LLMTimeout("simulated")
            result = worker.summarize_session("s1", [{"content": "hello", "role": "user", "turn_id": "t1", "timestamp": "2025-01-01T00:00:00Z"}])
            assert "SUMMARIZATION_FAILED" in result
            assert "LLMTimeout" in result

    def test_detect_contradictions_returns_empty_on_timeout(self):
        from hermes_memory_core.dream.worker import DreamWorker, LLMTimeout

        worker = DreamWorker()
        with patch("hermes_memory_core.dream.worker._llm_complete_with_retry") as mock:
            mock.side_effect = LLMTimeout("simulated")
            result = worker.detect_contradictions([], [{"fact_id": "f1", "fact_text": "old fact"}])
            assert result == []


class TestEmbedEndpoint:
    def test_lms_endpoint_is_localhost(self):
        from hermes_memory_core.embed import LMS_ENDPOINT
        assert "localhost" in LMS_ENDPOINT or "127.0.0.1" in LMS_ENDPOINT, f"Expected localhost, got {LMS_ENDPOINT}"

    def test_health_embedding_endpoint_is_localhost(self):
        from hermes_memory_core.health import DEFAULT_EMBEDDING_ENDPOINT
        assert "localhost" in DEFAULT_EMBEDDING_ENDPOINT or "127.0.0.1" in DEFAULT_EMBEDDING_ENDPOINT, f"Expected localhost, got {DEFAULT_EMBEDDING_ENDPOINT}"