"""Secret redaction scanner for Hermes Local Memory.

Phase 1 redaction: scans content BEFORE any write (JSONL, SQLite, QMD).
Catches: AWS keys, GitHub tokens, OpenAI / Anthropic API keys,
private keys, credit cards (Luhn-validated), SSN, high-entropy strings.

Audit log: every redaction event produces a hit record with type but NOT the raw value.

Per v0.2-critique.md Issue 7: ``force_no_redact`` is REMOVED from MVP.
The scanner always runs. Override lives at the pipeline level, not here.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_PATTERNS: List[tuple[str, re.Pattern]] = [
    # AWS Access Key: AKIA + 16 uppercase alphanumeric chars
    (
        "aws_access_key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    # OpenAI key: sk- followed by 20+ alphanumeric/hyphen/underscore
    (
        "openai_key",
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    ),
    # Anthropic key: sk-ant- followed by 20+ chars
    (
        "anthropic_key",
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    ),
    # GitHub token: ghp_ / gho_ / ghs_ / ghu_ / ghr_ or github_pat_ followed by
    # sufficient chars. Bounded by non-alphanumeric on both sides.
    (
        "github_token",
        re.compile(
            r"(?<![A-Za-z0-9])"
            r"(?:ghp_[A-Za-z0-9_]{36}|gho_[A-Za-z0-9_]{36}|"
            r"ghs_[A-Za-z0-9_]{36}|ghu_[A-Za-z0-9_]{36}|"
            r"ghr_[A-Za-z0-9_]{36}|github_pat_[A-Za-z0-9_]{22,})"
            r"(?![A-Za-z0-9])"
        ),
    ),
    # PEM private key headers (RSA, EC, DSA, OPENSSH, etc.)
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z ][A-Za-z0-9 ]*PRIVATE KEY-----"),
    ),
    # SSN: 3 digits - 2 digits - 4 digits
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
]


# High-entropy: secondary pass only. Matches strings >40 chars that are
# base64-ish (alphanumeric + / + =) or pure hex, with no internal spaces.
_HE_PATTERN = re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9+/=]{40,}(?![A-Za-z0-9/+=])")


# Luhn-validated card numbers: 13–19 consecutive digits bounded by non-digits.
_LUHN_DIGITS = re.compile(r"(?<!\d)\d{13,19}(?!\d)")


# ---------------------------------------------------------------------------
# Luhn validation
# ---------------------------------------------------------------------------

def _luhn_is_valid(number: str) -> bool:
    """Return True if number passes Luhn algorithm (13–19 digit card numbers)."""
    stripped = number.replace(" ", "").replace("-", "")
    if not stripped.isdigit() or not (13 <= len(stripped) <= 19):
        return False
    digits = [int(d) for d in stripped]
    # Luhn: double every second digit from the right, sum digits of products
    # plus the undoubled digits. Total must be divisible by 10.
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:  # every second digit from right (index 1,3,5,...)
            d = d * 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RedactionHit:
    """A single secret detected in content — type only, never raw value."""
    pattern_name: str
    start: int
    end: int


@dataclass(frozen=True)
class RedactionResult:
    """Result of a redaction scan."""
    redacted_content: str
    hits: List[RedactionHit]
    fired: bool


# ---------------------------------------------------------------------------
# Redactor
# ---------------------------------------------------------------------------

class Redactor:
    """Secret scanner. Scan always runs — no force_no_redact (removed per Issue 7)."""

    def scan(self, content: str) -> RedactionResult:
        """
        Scan ``content`` for secrets, replace each with ``[REDACTED:<type>]``.

        Returns ``RedactionResult`` with redacted content, list of hits (type only),
        and ``fired`` bool.

        No double-wrapping: already-replaced spans are tracked and skipped.
        """
        if not content:
            return RedactionResult(redacted_content="", hits=[], fired=False)

        hits: List[RedactionHit] = []
        replaced: List[tuple[int, int]] = []   # [(start, end), ...] merged

        def mark(start: int, end: int) -> None:
            replaced.append((start, end))

        def is_overlapping(start: int, end: int) -> bool:
            return any(s < end and e > start for s, e in replaced)

        result = content

        # --- Pass 1: specific patterns ---
        for name, pattern in _PATTERNS:
            parts: List[str] = []
            last_pos = 0

            for m in pattern.finditer(result):
                s, e = m.start(), m.end()
                if is_overlapping(s, e):
                    continue
                parts.append(result[last_pos:s])
                parts.append(f"[REDACTED:{name}]")
                mark(s, e)
                last_pos = e
                hits.append(RedactionHit(pattern_name=name, start=s, end=e))

            if parts:
                parts.append(result[last_pos:])
                result = "".join(parts)

        # --- Pass 2: Luhn-validated card numbers ---
        parts = []
        last_pos = 0

        for m in _LUHN_DIGITS.finditer(result):
            s, e = m.start(), m.end()
            if is_overlapping(s, e):
                continue
            number = m.group(0)
            if _luhn_is_valid(number):
                parts.append(result[last_pos:s])
                parts.append("[REDACTED:card]")
                mark(s, e)
                last_pos = e
                hits.append(RedactionHit(pattern_name="card", start=s, end=e))

        if parts:
            parts.append(result[last_pos:])
            result = "".join(parts)

        # --- Pass 3: high-entropy strings (secondary — catches long base64/hex) ---
        parts = []
        last_pos = 0

        for m in _HE_PATTERN.finditer(result):
            s, e = m.start(), m.end()
            if is_overlapping(s, e):
                continue
            parts.append(result[last_pos:s])
            parts.append("[REDACTED:high_entropy]")
            mark(s, e)
            last_pos = e
            hits.append(RedactionHit(pattern_name="high_entropy", start=s, end=e))

        if parts:
            parts.append(result[last_pos:])
            result = "".join(parts)

        if hits:
            logger.info("Redaction fired: %d secret(s) detected", len(hits))

        return RedactionResult(redacted_content=result, hits=hits, fired=bool(hits))


# Module-level default instance (pipeline imports this)
default_redactor = Redactor()


def scan(content: str) -> RedactionResult:
    """Convenience wrapper: ``from hermes_memory_core.write.redaction import scan``."""
    return default_redactor.scan(content)