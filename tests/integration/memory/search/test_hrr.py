# Copyright 2026 David McCarty. All rights reserved.
"""Tests for hermes_memory_core/search/hrr.py — T-022 (Epic 4.2.1).

Story 4.2.1 — Fork HRR library from holographic.

AC: Tests from holographic copied + pass against our fork.
Surface: encode_atom, encode_text, encode_fact, bind, unbind, bundle,
         similarity, phases_to_bytes, bytes_to_phases, snr_estimate.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# Ensure the venv python is on the path for the module import
import sys

venv_python = Path("/home/dmccarty/.hermes/hermes-agent/venv/bin/python3")
if venv_python.exists():
    sys.executable = str(venv_python)


# --------------------------------------------------------------------------
# HRR module under test
# --------------------------------------------------------------------------

from hermes_memory_core.search import hrr as hrr_module


# ------------------------------------------------------------------------------------------------------------------------------------------
# Helper — bypass numpy requirement for tests that don't need it
# --------------------------------------------------------------------------


def make_fake_module():
    """Return a clean hrr module with numpy forcibly disabled."""
    import importlib
    import sys
    # remove cached import
    key = "hermes_memory_core.search.hrr"
    if key in sys.modules:
        del sys.modules[key]
    # stub numpy out before import
    fake_hrr = type(sys)("hermes_memory_core.search.hrr")
    return fake_hrr


# ------------------------------------------------------------------------------------------------------------------------------------------
# encode_atom
# ------------------------------------------------------------------------------------------------------------------------------------------


class TestEncodeAtom:
    def test_deterministic_same_input_same_output(self):
        """Identical inputs produce identical vectors."""
        vec1 = hrr_module.encode_atom("test", dim=512)
        vec2 = hrr_module.encode_atom("test", dim=512)
        assert vec1.shape == (512,)
        assert vec1.shape == vec2.shape
        assert vec1 == pytest.approx(vec2)

    def test_different_inputs_different_vectors(self):
        """Different words produce different vectors."""
        vec1 = hrr_module.encode_atom("alice", dim=512)
        vec2 = hrr_module.encode_atom("bob", dim=512)
        assert vec1.shape == (512,)
        assert not (vec1 == vec2).all()

    def test_dim_truncation(self):
        """Output length matches dim."""
        for dim in [128, 256, 512, 1024, 2048]:
            vec = hrr_module.encode_atom("token", dim=dim)
            assert vec.shape == (dim,)

    def test_values_in_range(self):
        """Phase values are in [0, 2π)."""
        vec = hrr_module.encode_atom("test", dim=512)
        assert vec.min() >= 0.0
        assert vec.max() < 2 * 3.141592653589793

    def test_requires_numpy(self):
        """Raises RuntimeError gracefully when numpy is absent."""
        with patch.object(hrr_module, "_HAS_NUMPY", False):
            with pytest.raises(RuntimeError, match="numpy"):
                hrr_module.encode_atom("word")


# ------------------------------------------------------------------------------------------------------------------------------------------
# bind / unbind
# ------------------------------------------------------------------------------------------------------------------------------------------


class TestBindUnbind:
    def test_bind_unbind_roundtrip(self):
        """unbind(bind(a, b), a) ≈ b (phase subtraction is exact for single binding)."""
        a = hrr_module.encode_atom("key", dim=512)
        b = hrr_module.encode_atom("value", dim=512)
        bound = hrr_module.bind(a, b)
        recovered = hrr_module.unbind(bound, a)
        # Phase subtraction is exact — should be bit-for-bit identical
        assert recovered == pytest.approx(b)

    def test_bind_is_commutative(self):
        """Phase addition is commutative: bind(a, b) == bind(b, a)."""
        a = hrr_module.encode_atom("alice", dim=512)
        b = hrr_module.encode_atom("bob", dim=512)
        ab = hrr_module.bind(a, b)
        ba = hrr_module.bind(b, a)
        # Phase addition is element-wise: (a+b) % 2π == (b+a) % 2π
        assert (ab == ba).all()

    def test_unbind_wrong_key(self):
        """Unbinding with a wrong key produces garbage (not the original)."""
        a = hrr_module.encode_atom("key", dim=512)
        b = hrr_module.encode_atom("value", dim=512)
        wrong = hrr_module.encode_atom("wrong", dim=512)
        bound = hrr_module.bind(a, b)
        recovered = hrr_module.unbind(bound, wrong)
        # Should NOT equal b
        assert not (recovered == b).all()

    def test_requires_numpy(self):
        """Raises RuntimeError when numpy is absent."""
        with patch.object(hrr_module, "_HAS_NUMPY", False):
            with pytest.raises(RuntimeError):
                hrr_module.bind(
                    hrr_module.encode_atom("a", dim=128),
                    hrr_module.encode_atom("b", dim=128),
                )
            with pytest.raises(RuntimeError):
                hrr_module.unbind(
                    hrr_module.encode_atom("a", dim=128),
                    hrr_module.encode_atom("b", dim=128),
                )


# ------------------------------------------------------------------------------------------------------------------------------------------
# bundle
# ------------------------------------------------------------------------------------------------------------------------------------------


class TestBundle:
    def test_bundle_self_similar(self):
        """Bundle of the same vector is similar to the original."""
        v = hrr_module.encode_atom("same", dim=512)
        bundled = hrr_module.bundle(v, v, v)
        sim = hrr_module.similarity(v, bundled)
        assert sim > 0.9  # very high similarity

    def test_bundle_order_independent(self):
        """Bundle order doesn't change the result."""
        a = hrr_module.encode_atom("a", dim=512)
        b = hrr_module.encode_atom("b", dim=512)
        c = hrr_module.encode_atom("c", dim=512)
        bc = hrr_module.bundle(b, c)
        cb = hrr_module.bundle(c, b)
        assert bc == pytest.approx(cb)

    def test_bundle_empty(self):
        """Bundle with no arguments produces a scalar zero (no superposition)."""
        result = hrr_module.bundle()
        # np.sum([]) = 0j → np.angle(0j) = 0.0 (scalar, not a vector)
        assert result.shape == ()  # scalar
        assert result == 0.0

    def test_bundle_distinct_vectors(self):
        """Bundling distinct vectors produces a vector similar to all of them."""
        a = hrr_module.encode_atom("alpha", dim=512)
        b = hrr_module.encode_atom("beta", dim=512)
        c = hrr_module.encode_atom("gamma", dim=512)
        bundled = hrr_module.bundle(a, b, c)
        # Should be more similar to each than random would be
        for v in [a, b, c]:
            sim = hrr_module.similarity(v, bundled)
            assert sim > 0.1  # at least weakly similar

    def test_requires_numpy(self):
        """Raises RuntimeError when numpy is absent."""
        with patch.object(hrr_module, "_HAS_NUMPY", False):
            with pytest.raises(RuntimeError):
                hrr_module.bundle(hrr_module.encode_atom("x", dim=128))


# ------------------------------------------------------------------------------------------------------------------------------------------
# similarity
# ------------------------------------------------------------------------------------------------------------------------------------------


class TestSimilarity:
    def test_identical_vectors(self):
        """Same vector returns similarity ≈ 1.0."""
        v = hrr_module.encode_atom("identical", dim=512)
        sim = hrr_module.similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_negated_vectors(self):
        """Opposite-phase vector returns similarity ≈ -1.0."""
        v = hrr_module.encode_atom("pos", dim=512)
        neg = (v + math.pi) % (2 * math.pi)  # flip all phases
        sim = hrr_module.similarity(v, neg)
        assert sim < -0.99

    def test_random_vectors_near_zero(self):
        """Two independent random words are approximately uncorrelated."""
        sim = hrr_module.similarity(
            hrr_module.encode_atom("random_a_xyz123", dim=512),
            hrr_module.encode_atom("random_b_abc789", dim=512),
        )
        assert abs(sim) < 0.2  # near zero

    def test_range_bounds(self):
        """Similarity is always in [-1, 1]."""
        words = ["word" + str(i) for i in range(20)]
        for w1 in words:
            for w2 in words:
                sim = hrr_module.similarity(
                    hrr_module.encode_atom(w1, dim=256),
                    hrr_module.encode_atom(w2, dim=256),
                )
                assert -1.0 <= sim <= 1.0

    def test_requires_numpy(self):
        """Raises RuntimeError when numpy is absent."""
        with patch.object(hrr_module, "_HAS_NUMPY", False):
            with pytest.raises(RuntimeError):
                hrr_module.similarity(
                    hrr_module.encode_atom("a", dim=128),
                    hrr_module.encode_atom("b", dim=128),
                )


# ------------------------------------------------------------------------------------------------------------------------------------------
# encode_text
# ------------------------------------------------------------------------------------------------------------------------------------------


class TestEncodeText:
    def test_deterministic(self):
        """Same text produces identical vectors."""
        v1 = hrr_module.encode_text("hello world", dim=512)
        v2 = hrr_module.encode_text("hello world", dim=512)
        assert v1 == pytest.approx(v2)

    def test_empty_string_fallback(self):
        """Empty string falls back to __hrr_empty__."""
        vec = hrr_module.encode_text("", dim=512)
        expected = hrr_module.encode_atom("__hrr_empty__", dim=512)
        assert vec == pytest.approx(expected)

    def test_whitespace_only_fallback(self):
        """Whitespace-only string falls back to __hrr_empty__."""
        vec = hrr_module.encode_text("   \t\n  ", dim=512)
        expected = hrr_module.encode_atom("__hrr_empty__", dim=512)
        assert vec == pytest.approx(expected)

    def test_token_extraction(self):
        """Punctuation is stripped and case is normalized."""
        v1 = hrr_module.encode_text("Hello, world!", dim=512)
        v2 = hrr_module.encode_text("hello world", dim=512)
        # Both should produce similar vectors (same core tokens)
        sim = hrr_module.similarity(v1, v2)
        assert sim > 0.8

    def test_dim_truncation(self):
        """Output length matches dim."""
        for dim in [256, 512, 1024]:
            vec = hrr_module.encode_text("some tokens here", dim=dim)
            assert vec.shape == (dim,)

    def test_requires_numpy(self):
        """Raises RuntimeError when numpy is absent."""
        with patch.object(hrr_module, "_HAS_NUMPY", False):
            with pytest.raises(RuntimeError):
                hrr_module.encode_text("hello")


# ------------------------------------------------------------------------------------------------------------------------------------------
# encode_fact
# ------------------------------------------------------------------------------------------------------------------------------------------


class TestEncodeFact:
    def test_deterministic(self):
        """Same fact + entities produces identical vectors."""
        v1 = hrr_module.encode_fact("Alice likes coffee", ["Alice"], dim=512)
        v2 = hrr_module.encode_fact("Alice likes coffee", ["Alice"], dim=512)
        assert v1 == pytest.approx(v2)

    def test_different_content_different_vector(self):
        """Different fact content produces different vectors."""
        v1 = hrr_module.encode_fact("Alice likes coffee", ["Alice"], dim=512)
        v2 = hrr_module.encode_fact("Bob likes tea", ["Bob"], dim=512)
        assert not (v1 == v2).all()

    def test_entity_role_bound(self):
        """Entities are bound to a distinct role vector."""
        fact = hrr_module.encode_fact("Alice is tall", ["Alice"], dim=512)
        alice_atom = hrr_module.encode_atom("alice", dim=512)
        role_entity = hrr_module.encode_atom("__hrr_role_entity__", dim=512)
        # Unbinding the entity role should reveal the entity atom
        unbound = hrr_module.unbind(fact, role_entity)
        # At least weakly similar to alice atom
        sim = hrr_module.similarity(unbound, alice_atom)
        assert sim > 0.0  # above random baseline

    def test_multiple_entities(self):
        """Multiple entities are all bound into the fact vector."""
        fact = hrr_module.encode_fact(
            "Alice and Bob are friends",
            ["Alice", "Bob"],
            dim=512,
        )
        assert fact.shape == (512,)

    def test_empty_entities_list(self):
        """Facts without entities still encode content (bound to role_content)."""
        vec = hrr_module.encode_fact("Something happened", [], dim=512)
        assert vec.shape == (512,)
        # The fact vector is bind(encode_text(content), role_content) — structured
        # differently from plain text but is a valid phase vector
        assert vec.min() >= 0.0
        assert vec.max() < 2 * 3.141592653589793

    def test_requires_numpy(self):
        """Raises RuntimeError when numpy is absent."""
        with patch.object(hrr_module, "_HAS_NUMPY", False):
            with pytest.raises(RuntimeError):
                hrr_module.encode_fact("content", ["entity"], dim=256)


# ------------------------------------------------------------------------------------------------------------------------------------------
# phases_to_bytes / bytes_to_phases
# ------------------------------------------------------------------------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip(self):
        """phases_to_bytes → bytes_to_phases recovers the original vector."""
        original = hrr_module.encode_atom("roundtrip_test", dim=512)
        data = hrr_module.phases_to_bytes(original)
        recovered = hrr_module.bytes_to_phases(data)
        assert recovered == pytest.approx(original)

    def test_bytes_length(self):
        """Serialized bytes are 8 * dim (float64)."""
        for dim in [256, 512, 1024]:
            vec = hrr_module.encode_atom("size_test", dim=dim)
            data = hrr_module.phases_to_bytes(vec)
            assert len(data) == 8 * dim

    def test_requires_numpy(self):
        """Raises RuntimeError when numpy is absent."""
        dummy_bytes = bytes(8 * 128)
        with patch.object(hrr_module, "_HAS_NUMPY", False):
            with pytest.raises(RuntimeError):
                hrr_module.phases_to_bytes(
                    hrr_module.encode_atom("x", dim=128),
                )
            with pytest.raises(RuntimeError):
                hrr_module.bytes_to_phases(dummy_bytes)


# ------------------------------------------------------------------------------------------------------------------------------------------
# snr_estimate
# ------------------------------------------------------------------------------------------------------------------------------------------


class TestSnrEstimate:
    def test_empty_storage(self):
        """Zero items returns inf."""
        snr = hrr_module.snr_estimate(dim=1024, n_items=0)
        assert snr == float("inf")

    def test_positive_dim_positive_items(self):
        """Returns sqrt(dim / n_items)."""
        snr = hrr_module.snr_estimate(dim=1024, n_items=256)
        expected = math.sqrt(1024 / 256)
        assert abs(snr - expected) < 1e-9

    def test_high_items_warns(self, caplog):
        """SNR < 2.0 logs a warning."""
        with caplog.at_level("WARNING"):
            snr = hrr_module.snr_estimate(dim=1024, n_items=512)
        assert snr < 2.0
        assert "SNR" in caplog.text

    def test_low_items_no_warn(self, caplog):
        """SNR >= 2.0 does not log a warning."""
        with caplog.at_level("WARNING"):
            snr = hrr_module.snr_estimate(dim=1024, n_items=100)
        assert snr >= 2.0
        # Should be no HRR-specific warnings
        hrr_warnings = [r for r in caplog.records if "HRR" in r.message or "SNR" in r.message]
        assert len(hrr_warnings) == 0

    def test_requires_numpy(self):
        """Raises RuntimeError when numpy is absent."""
        with patch.object(hrr_module, "_HAS_NUMPY", False):
            with pytest.raises(RuntimeError):
                hrr_module.snr_estimate(dim=1024, n_items=100)