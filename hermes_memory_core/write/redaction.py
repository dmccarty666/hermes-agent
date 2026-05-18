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
    # Anthropic key: sk-ant- followed by 20+ chars (before openai_key)
    (
        "anthropic_key",
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    ),
    # OpenAI key: sk- followed by 20+ alphanumeric/hyphen/underscore
    (
        "openai_key",
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    ),
    # GitHub token: ghp_/gho_/ghs_/ghu_/ghr_/github_pat_ + sufficient chars
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
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit = digit * 2
            if digit > 9:
                digit -= 9
        total += digit
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

        Algorithm: collect all matches across all patterns (in original positions),
        sort by position, filter overlapping spans, then apply replacements
        in one pass. This avoids offset-drift bugs from sequential string rebuilding.
        """
        if not content:
            return RedactionResult(redacted_content="", hits=[], fired=False)

        # --- Pass 1: collect all specific-pattern hits (in original positions) ---
        raw_hits: List[tuple[int, int, str]] = []  # [(start, end, name), ...]

        for name, pattern in _PATTERNS:
            for m in pattern.finditer(content):
                raw_hits.append((m.start(), m.end(), name))

        # --- Pass 2: Luhn-validated card numbers ---
        for m in _LUHN_DIGITS.finditer(content):
            if _luhn_is_valid(m.group(0)):
                raw_hits.append((m.start(), m.end(), "card"))

        # Sort by start position (ascending) — same start, longer first
        raw_hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        # Filter overlapping spans: keep the first match that starts,
        # skip any that overlap with an already-kept span
        kept: List[tuple[int, int, str]] = []
        for hit in raw_hits:
            s, e, name = hit
            if not any(s < oe and e > os for os, oe, _ in kept):
                kept.append(hit)

        # Sort kept hits by start for reconstruction
        kept.sort(key=lambda x: x[0])

        # --- Pass 3: high-entropy strings (secondary) ---
        # Run after keeping the above hits so we skip anything already matched
        he_hits: List[tuple[int, int, str]] = []
        for m in _HE_PATTERN.finditer(content):
            s, e = m.start(), m.end()
            if not any(s < oe and e > os for (os, oe, _) in kept + he_hits):
                he_hits.append((s, e, "high_entropy"))

        all_hits = kept + he_hits
        all_hits.sort(key=lambda x: x[0])

        # --- Build redacted content by splicing ---
        hits_out: List[RedactionHit] = []
        result_parts: List[str] = []
        cursor = 0

        for s, e, name in all_hits:
            result_parts.append(content[cursor:s])
            result_parts.append(f"[REDACTED:{name}]")
            cursor = e
            hits_out.append(RedactionHit(pattern_name=name, start=s, end=e))

        result_parts.append(content[cursor:])
        redacted_content = "".join(result_parts)

        if hits_out:
            logger.info("Redaction fired: %d secret(s) detected", len(hits_out))

        return RedactionResult(redacted_content=redacted_content, hits=hits_out, fired=bool(hits_out))


# Module-level default instance (pipeline imports this)
default_redactor = Redactor()


def scan(content: str) -> RedactionResult:
    """Convenience wrapper: ``from hermes_memory_core.write.redaction import scan``."""
    return default_redactor.scan(content)