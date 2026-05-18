"""Test redaction scanner (Story 1.4.1 — T-006).

RED PHASE: Tests are written first against the TDD §7.1 spec.
GREEN PHASE: Implementation follows to make tests pass.
REFACTOR: Clean up after green.

The redaction scanner is security-critical — tests use fixture secrets
in AKIAFAKE... / sk-test... style (clearly fake, no risk).
"""
from __future__ import annotations

import inspect
from typing import List

import pytest

from hermes_memory_core.write import redaction as redaction_module
from hermes_memory_core.write.redaction import (
    RedactionResult,
    Redactor,
    _luhn_is_valid,
    scan,
)


# ---------------------------------------------------------------------------
# Fixtures — fake secrets (clearly fake, no risk)
# ---------------------------------------------------------------------------

# Luhn-valid test cards (check digit verified)
_CARDS = {
    "visa": "4532015112830366",      # valid Luhn, 16 digits
    "mastercard": "5425233430109903",  # valid Luhn, 16 digits
    "amex": "340000000000009",        # valid Luhn, 15 digits (Amex 34 prefix)
}

# Invalid cards for negative tests
_CARD_INCOMPLETE = "453201511283036"   # 15 digits (below 13-19 range)
_CARD_INVALID_LUHN = "4532015112830361"  # wrong check digit


FAKE = {
    "openai_key": "sk-testAbCdEfGhIjKlMnOpQrStUvWxYz1234",
    "anthropic_key": "sk-ant-testAbCdEfGhIjKlMnOpQrStUvWxYz1234",
    "aws_access_key": "AKIAFAKEFAKEFAKEFAKE",
    "aws_secret": "fakeawssecretkey1234567890abcdefghijklmnop",  # 44 chars for high_entropy
    "github_token_ghp": "ghp_fakeFakeFakeFakeFakeFakeFakeFakeFake",
    "github_token_gho": "gho_fakeFakeFakeFakeFakeFakeFakeFakeFAKE",
    "github_pat": "github_pat_fakeFakeFakeFakeFakeFake_FakeFake",
    "private_key_rsa": (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIBogIBAAJBALRiMLAHudeSA2FAoZV3mgLVbUcvnogR9YqCZ0aYfL4L\n"
        "-----END RSA PRIVATE KEY-----"
    ),
    "private_key_ec": (
        "-----BEGIN EC PRIVATE KEY-----\n"
        "MHQCAQEEIIrYSSNQFaA2Hwf1duRsxqLYj5R0mLaTi8JlrCQo2eOloAcGBSuBBAAKoUQD\n"
        "-----END EC PRIVATE KEY-----"
    ),
    "card_visa": _CARDS["visa"],
    "card_mastercard": _CARDS["mastercard"],
    "card_amex": _CARDS["amex"],
    "card_incomplete": _CARD_INCOMPLETE,
    "card_invalid_luhn": _CARD_INVALID_LUHN,
    "ssn": "123-45-6789",
    "ssn_no_dashes": "1234567890",     # 10 digits, no dashes — not SSN pattern
    "high_entropy_base64": "aGVsbG93b3JsZGhlbGxvYWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXo=",  # 64 chars
    "high_entropy_hex": "deadbeefcafebabe0123456789abcdef0123456789abcdef",  # 56 chars hex
    "low_entropy": "hello world this is just plain text with no secrets in it",
}

ORDINARY_TEXT = (
    "Hello, this is a conversation about building a chess engine in Python. "
    "We discussed alpha-beta pruning, move ordering, and the importance of "
    "quiescence search to handle tactical sequences. The user prefers Python 3.11 "
    "and uses black for formatting. Nothing sensitive here."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_redacted(content: str, redacted: str, secret: str, label: str) -> None:
    """Assert secret is absent from redacted output and placeholder is present."""
    assert secret not in redacted, f"{label}: raw value leaked into redacted output"
    assert f"[REDACTED:{label}]" in redacted, f"{label}: placeholder missing in output"


# ---------------------------------------------------------------------------
# Tests — one class per AC group
# ---------------------------------------------------------------------------

class TestOpenAIKey:
    def test_openai_key_redacted(self):
        secret = FAKE["openai_key"]
        result = scan(f"Please use key: {secret}")
        assert "[REDACTED:openai_key]" in result.redacted_content
        assert result.fired
        assert any(h.pattern_name == "openai_key" for h in result.hits)
        assert secret not in result.redacted_content

    def test_openai_key_twice_both_redacted(self):
        secret = FAKE["openai_key"]
        result = scan(f"key1={secret} and key2={secret}")
        assert result.redacted_content.count("[REDACTED:openai_key]") == 2
        assert secret not in result.redacted_content


class TestAnthropicKey:
    def test_anthropic_key(self):
        secret = FAKE["anthropic_key"]
        result = scan(f"key={secret}")
        assert "[REDACTED:anthropic_key]" in result.redacted_content
        assert any(h.pattern_name == "anthropic_key" for h in result.hits)
        assert secret not in result.redacted_content


class TestAWSKey:
    def test_aws_access_key(self):
        secret = FAKE["aws_access_key"]
        result = scan(f"ID={secret}")
        assert "[REDACTED:aws_access_key]" in result.redacted_content
        assert any(h.pattern_name == "aws_access_key" for h in result.hits)
        assert secret not in result.redacted_content

    def test_aws_secret_key(self):
        secret = FAKE["aws_secret"]
        result = scan(f"secret={secret}")
        # AWS secret is caught by high-entropy pass (40+ chars, no spaces)
        assert result.fired
        assert secret not in result.redacted_content


class TestGitHubToken:
    def test_github_token_ghp(self):
        secret = FAKE["github_token_ghp"]
        result = scan(f"token={secret}")
        assert "[REDACTED:github_token]" in result.redacted_content
        assert any(h.pattern_name == "github_token" for h in result.hits)
        assert secret not in result.redacted_content

    def test_github_token_gho(self):
        secret = FAKE["github_token_gho"]
        result = scan(f"token={secret}")
        assert "[REDACTED:github_token]" in result.redacted_content
        assert any(h.pattern_name == "github_token" for h in result.hits)
        assert secret not in result.redacted_content

    def test_github_pat(self):
        secret = FAKE["github_pat"]
        result = scan(f"pat={secret}")
        assert "[REDACTED:github_token]" in result.redacted_content
        assert any(h.pattern_name == "github_token" for h in result.hits)


class TestPrivateKey:
    def test_rsa_private_key(self):
        content = FAKE["private_key_rsa"]
        result = scan(content)
        assert "[REDACTED:private_key]" in result.redacted_content
        assert any(h.pattern_name == "private_key" for h in result.hits)
        assert "-----BEGIN RSA PRIVATE KEY-----" not in result.redacted_content

    def test_ec_private_key(self):
        content = FAKE["private_key_ec"]
        result = scan(content)
        assert "[REDACTED:private_key]" in result.redacted_content
        assert any(h.pattern_name == "private_key" for h in result.hits)
        assert "-----BEGIN EC PRIVATE KEY-----" not in result.redacted_content


class TestCreditCard:
    def test_visa_card(self):
        secret = FAKE["card_visa"]
        result = scan(f"cc={secret}")
        assert "[REDACTED:card]" in result.redacted_content
        assert any(h.pattern_name == "card" for h in result.hits)
        assert secret not in result.redacted_content

    def test_mastercard(self):
        secret = FAKE["card_mastercard"]
        result = scan(f"cc={secret}")
        assert "[REDACTED:card]" in result.redacted_content
        assert any(h.pattern_name == "card" for h in result.hits)

    def test_amex(self):
        secret = FAKE["card_amex"]
        result = scan(f"cc={secret}")
        assert "[REDACTED:card]" in result.redacted_content
        assert any(h.pattern_name == "card" for h in result.hits)

    def test_incomplete_not_flagged(self):
        """Card number too short should not trigger."""
        result = scan(f"cc={FAKE['card_incomplete']}")
        assert result.redacted_content == f"cc={FAKE['card_incomplete']}"
        assert not result.fired

    def test_invalid_luhn_not_flagged(self):
        """Wrong check digit should not trigger."""
        result = scan(f"cc={FAKE['card_invalid_luhn']}")
        assert result.redacted_content == f"cc={FAKE['card_invalid_luhn']}"
        assert not result.fired


class TestSSN:
    def test_ssn_dashes(self):
        secret = FAKE["ssn"]
        result = scan(f"ssn={secret}")
        assert "[REDACTED:ssn]" in result.redacted_content
        assert any(h.pattern_name == "ssn" for h in result.hits)
        assert secret not in result.redacted_content

    def test_ssn_no_dashes_not_flagged(self):
        """10 consecutive digits without dashes is not SSN pattern."""
        result = scan(f"raw={FAKE['ssn_no_dashes']}")
        assert result.redacted_content == f"raw={FAKE['ssn_no_dashes']}"
        assert not result.fired


class TestHighEntropy:
    def test_base64_high_entropy(self):
        secret = FAKE["high_entropy_base64"]
        result = scan(f"secret={secret}")
        assert "[REDACTED:high_entropy]" in result.redacted_content
        assert any(h.pattern_name == "high_entropy" for h in result.hits)
        assert secret not in result.redacted_content

    def test_hex_high_entropy(self):
        secret = FAKE["high_entropy_hex"]
        result = scan(f"secret={secret}")
        assert "[REDACTED:high_entropy]" in result.redacted_content
        assert any(h.pattern_name == "high_entropy" for h in result.hits)

    def test_low_entropy_not_flagged(self):
        """Short strings with spaces are not high-entropy."""
        result = scan(f"text={FAKE['low_entropy']}")
        assert result.redacted_content == f"text={FAKE['low_entropy']}"
        assert not result.fired


class TestOrdinaryText:
    def test_ordinary_text_unchanged(self):
        result = scan(ORDINARY_TEXT)
        assert result.redacted_content == ORDINARY_TEXT
        assert result.hits == []
        assert not result.fired

    def test_empty_string(self):
        result = scan("")
        assert result.redacted_content == ""
        assert result.hits == []
        assert not result.fired


class TestMultipleSecretTypes:
    """AC: when multiple secret types appear, all are caught and listed."""

    def test_openai_github_card_all_caught(self):
        content = (
            f"openai={FAKE['openai_key']}, "
            f"github={FAKE['github_token_ghp']}, "
            f"card={FAKE['card_visa']}"
        )
        result = scan(content)
        hit_names = {h.pattern_name for h in result.hits}
        assert "openai_key" in hit_names
        assert "github_token" in hit_names
        assert "card" in hit_names
        # Raw values absent
        assert FAKE["openai_key"] not in result.redacted_content
        assert FAKE["github_token_ghp"] not in result.redacted_content
        assert FAKE["card_visa"] not in result.redacted_content


class TestNoDoubleWrapping:
    """AC: one replacement per match, already-redacted content not re-scanned."""

    def test_already_redacted_not_rewrapped(self):
        """Content already containing [REDACTED:...] is not re-scanned/re-wrapped."""
        secret = FAKE["openai_key"]
        content = f"secret={secret} and already masked: [REDACTED:openai_key]"
        result = scan(content)
        # Only the real secret should be caught as a hit (not the literal placeholder)
        assert len(result.hits) == 1, f"Expected 1 hit (real secret), got {len(result.hits)}: {result.hits}"
        assert result.redacted_content.count("[REDACTED:openai_key]") >= 1
        assert secret not in result.redacted_content


class TestAuditLogRedactionType:
    """AC: audit_log row contains redaction type but NEVER raw original value.

    The RedactionHit dataclass carries pattern_name + offsets only —
    matched_text is intentionally absent (never logged).
    """

    def test_hits_contain_type_not_value(self):
        secret = FAKE["openai_key"]
        result = scan(f"key={secret}")
        assert result.fired
        hit = next(h for h in result.hits if h.pattern_name == "openai_key")
        # hit has start/end offsets but NOT the raw secret value
        assert hasattr(hit, "pattern_name")
        assert hasattr(hit, "start")
        assert hasattr(hit, "end")
        # RedactionHit is frozen — check it has no 'matched_text' field
        assert not hasattr(hit, "matched_text"), "matched_text must not exist on RedactionHit"

    def test_result_shape(self):
        """scan() returns RedactionResult with non-None redacted_content and hits list."""
        result = scan("hello world")
        assert isinstance(result, RedactionResult)
        assert isinstance(result.redacted_content, str)
        assert isinstance(result.hits, list)
        assert isinstance(result.fired, bool)
        assert result.redacted_content is not None
        assert result.hits is not None


class TestNoForceNoRedact:
    """Per Issue 7 (v0.2-critique.md): force_no_redact is REMOVED from MVP.

    The scanner always scans. There is no override argument.
    """

    def test_no_force_no_redact_parameter(self):
        """redaction.scan() does not accept force_no_redact."""
        sig = inspect.signature(scan)
        assert "force_no_redact" not in sig.parameters

    def test_scanner_always_runs_on_every_content(self):
        """No content is exempt from scanning."""
        for label, text in FAKE.items():
            if label in ("low_entropy", "ssn_no_dashes", "card_incomplete", "card_invalid_luhn"):
                continue  # expected to not fire
            result = scan(text)
            assert result.fired, f"Expected redaction for {label} but fired=False"
            assert any(h.pattern_name for h in result.hits), f"Expected at least one hit for {label}"


class TestLuhnValidation:
    """Unit-test the Luhn helper directly."""

    @pytest.mark.parametrize("card,expected", [
        (_CARDS["visa"], True),
        (_CARDS["mastercard"], True),
        (_CARDS["amex"], True),
        (_CARD_INCOMPLETE, False),   # too short
        (_CARD_INVALID_LUHN, False),  # wrong check digit
        ("1234567890123456", False),   # invalid check digit
        ("4242424242424242", True),    # known valid
    ])
    def test_luhn_is_valid(self, card: str, expected: bool):
        assert _luhn_is_valid(card) == expected