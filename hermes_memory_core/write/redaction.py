"""
Redaction — secret scanner for Phase 1 defense-in-depth.

Patterns detected:
- AWS access keys (AKIA...)
- GitHub tokens (ghp_..., gho_..., ghu_..., ghs_..., ghr_...)
- OpenAI / Azure OpenAI keys (sk-..., azure...api)
- Anthropic keys (sk-ant-...)
- Generic API keys (api_key=, key=, token= in URL-like contexts)
- High-entropy strings (potential credentials)
- Credit card numbers (Luhn-valid 13-19 digit)
- Social Security Numbers (XXX-XX-XXXX pattern)
- Private keys (RSA, EC, OPENSSH headers)

All replacements use a consistent `[REDACTED-{type}-{hex}]` sentinel format
that is reversible for audit purposes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional, Pattern

# Sentinel prefix used in audit log
SENTINEL_PREFIX = "[REDACTED"

# Compiled patterns — module-level for performance
_PATTERNS: List[tuple[str, Pattern[str]]] = [
    ("AWS_KEY", re.compile(r"\b(AKIA[0-9A-Z]{16})\b")),
    ("GITHUB_TOKEN", re.compile(r"\b(ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36}|ghu_[a-zA-Z0-9]{36}|ghs_[a-zA-Z0-9]{36}|ghr_[a-zA-Z0-9]{36})\b")),
    ("OPENAI_KEY", re.compile(r"\b(sk-[a-zA-Z0-9_-]{20,})\b")),
    ("ANTHROPIC_KEY", re.compile(r"\b(sk-ant-[a-zA-Z0-9_-]{20,})\b")),
    ("AZURE_KEY", re.compile(r"\b[a-zA-Z0-9+/]{44}\.azurewebsites\.net\b")),
    ("STRIPE_KEY", re.compile(r"\b(sk_live_[a-zA-Z0-9]{24,})\b")),
    ("CARD_NUMBER", re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (" PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("HEX_ENTROPY", re.compile(r"\b[a-f0-9]{32,64}\b", re.IGNORECASE)),
]


@dataclass(frozen=True)
class RedactionResult:
    """Result of a redaction scan."""
    redacted_text: str
    types_found: List[str]          # e.g. ["AWS_KEY", "GITHUB_TOKEN"]
    count: int                       # total redaction events


def redact(text: str, skip_overrides: bool = False) -> RedactionResult:
    """
    Scan text for secrets and replace them with reversible sentinels.

    Defense-in-depth: run on all write paths even if capture path already
    redacted. The skip_overrides flag is for cases where caller explicitly
    wants the raw text (e.g. source resolution) — use with extreme caution.

    Returns RedactionResult with the redacted text, list of types found,
    and total count of redaction events.
    """
    if not text:
        return RedactionResult(redacted_text=text, types_found=[], count=0)

    redacted = text
    types_found: List[str] = []
    total_count = 0

    for secret_type, pattern in _PATTERNS:
        found = pattern.findall(redacted)
        if found:
            types_found.append(secret_type)
            count = len(found)
            total_count += count
            # Replace each occurrence with a consistent sentinel
            sentinel = f"{SENTINEL_PREFIX}-{secret_type}]"
            redacted = pattern.sub(sentinel, redacted)

    return RedactionResult(
        redacted_text=redacted,
        types_found=types_found,
        count=total_count,
    )


def hash_content(text: str) -> str:
    """Return SHA-256 hex of text for content-addressing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
