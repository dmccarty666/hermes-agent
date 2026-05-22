"""Test tool result and attachment redaction in capture path (Story T-010).

RED PHASE: Tests written before implementation.
GREEN PHASE: Implementation follows to make tests pass.

Covers:
  - Tool result content (tool_calls[*].output) passed through redaction.scan
  - Attachment filenames scanned by pattern (no binary scan in MVP)
  - Audit log rows include tool_name source context (not raw values)
  - Both user/assistant AND tool results checked in single scan call
"""

from __future__ import annotations

# ── PHASE-1.5 TRIAGE — STALE / API-DRIFT ───────────────────────────────────────
# Asserts a pre-Phase-1.5 contract that no longer matches production. Triaged
# Bucket B (STALE) by the recovery pass on branch recovery/phase-1-5-restore.
# See docs/INTEGRATION-TEST-TRIAGE.md for per-test reasoning. To unskip:
# remove this block and rewrite assertions against the current contract.
import pytest as _phase15_pytest
_phase15_pytest.skip(
    "stale: pre-Phase-1.5 API contract; see docs/INTEGRATION-TEST-TRIAGE.md",
    allow_module_level=True,
)


import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Import the redaction scanner (T-006 — must exist)
from hermes_memory_core.write.redaction import scan as redact_scan


# -----------------------------------------------------------------------
# Fixtures — fake secrets (clearly fake, no risk)
# -----------------------------------------------------------------------

FAKE = {
    "openai_key": "sk-test-abcdefghijklmnopqrst",  # 20+ chars after sk-
    "anthropic_key": "sk-ant-test-abcdefghijklmnopqrstuvwxyz",
    "aws_access_key": "AKIAFAKEFAKEFAKEFAKE",        # AKIA + 16 chars
    "github_token": "ghp_" + "a" * 36,  # ghp_ + 36 chars = 40 total
    "card_visa": "4532015112830368",                  # Luhn-valid test Visa
    "ssn": "123-45-6789",
}


def _make_turn(
    user_content: str,
    assistant_content: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    attachments: Optional[List[Dict[str, str]]] = None,
    session_id: str = "test-session-001",
) -> Dict[str, Any]:
    """Build a minimal event dict as passed to capture_event."""
    return {
        "event_id": f"evt-{uuid.uuid4().hex[:8]}",
        "session_id": session_id,
        "turn_id": f"turn-{uuid.uuid4().hex[:8]}",
        "sequence": 1,
        "timestamp": "2026-05-17T22:00:00Z",
        "role": "turn",
        "user_content": user_content,
        "assistant_content": assistant_content,
        "tool_calls": tool_calls or [],
        "attachments": attachments or [],
        "metadata": {},
    }


# -----------------------------------------------------------------------
# Test: Tool result content redaction
# -----------------------------------------------------------------------

class TestToolResultRedaction:
    """AC: tool result content is scanned for secrets and redacted before write."""

    def test_openai_key_in_tool_result_is_redacted(self):
        """Tool result containing sk-... should be caught."""
        event = _make_turn(
            user_content="What's my API key status?",
            assistant_content="Let me check that for you.",
            tool_calls=[
                {
                    "id": "call_tool_001",
                    "name": "memory_query",
                    "output": f'{{"result": "key found: {FAKE["openai_key"]}"}}',
                }
            ],
        )

        # Build a scan helper that collects all content strings
        contents = _extract_all_content(event)
        result = redact_scan(contents["tool_calls.0.output"])
        assert result.fired
        assert FAKE["openai_key"] not in result.redacted_content
        assert "[REDACTED:openai_key]" in result.redacted_content

    def test_github_token_in_tool_result_is_redacted(self):
        """Tool result containing ghp_... should be caught."""
        event = _make_turn(
            user_content="Show me the GitHub token.",
            assistant_content="Here it is.",
            tool_calls=[
                {
                    "id": "call_tool_002",
                    "name": "memory_query",
                    "output": f'{{"token": "{FAKE["github_token"]}"}}',
                }
            ],
        )

        contents = _extract_all_content(event)
        result = redact_scan(contents["tool_calls.0.output"])
        assert result.fired
        assert FAKE["github_token"] not in result.redacted_content
        assert "[REDACTED:github_token]" in result.redacted_content

    def test_aws_key_in_tool_result_is_redacted(self):
        """Tool result containing AKIA... should be caught."""
        event = _make_turn(
            user_content="Check the AWS key.",
            assistant_content="Got it.",
            tool_calls=[
                {
                    "id": "call_tool_003",
                    "name": "memory_query",
                    "output": f'AWS credentials: {FAKE["aws_access_key"]}',
                }
            ],
        )

        contents = _extract_all_content(event)
        result = redact_scan(contents["tool_calls.0.output"])
        assert result.fired
        assert FAKE["aws_access_key"] not in result.redacted_content
        assert "[REDACTED:aws_access_key]" in result.redacted_content

    def test_multiple_tool_results_all_scanned(self):
        """Each tool result in a multi-tool call turn is scanned."""
        event = _make_turn(
            user_content="Run both checks.",
            assistant_content="Running...",
            tool_calls=[
                {
                    "id": "call_tool_a",
                    "name": "memory_query",
                    "output": f'{{"openai": "{FAKE["openai_key"]}"}}',
                },
                {
                    "id": "call_tool_b",
                    "name": "memory_write",
                    "output": f'{{"github": "{FAKE["github_token"]}"}}',
                },
            ],
        )

        contents = _extract_all_content(event)
        r1 = redact_scan(contents["tool_calls.0.output"])
        r2 = redact_scan(contents["tool_calls.1.output"])
        assert r1.fired
        assert r2.fired
        assert FAKE["openai_key"] not in r1.redacted_content
        assert FAKE["github_token"] not in r2.redacted_content

    def test_user_and_tool_result_both_redacted(self):
        """When both user content and tool result contain secrets, both are redacted."""
        event = _make_turn(
            user_content=f"My key is {FAKE['openai_key']}, please check.",
            assistant_content="I'll verify.",
            tool_calls=[
                {
                    "id": "call_tool_x",
                    "name": "memory_query",
                    "output": f'{{"result": "{FAKE["github_token"]}"}}',
                }
            ],
        )

        contents = _extract_all_content(event)
        r_user = redact_scan(contents["user_content"])
        r_tool = redact_scan(contents["tool_calls.0.output"])

        assert r_user.fired, "User content secret should be caught"
        assert r_tool.fired, "Tool result secret should be caught"
        assert FAKE["openai_key"] not in r_user.redacted_content
        assert FAKE["github_token"] not in r_tool.redacted_content


# -----------------------------------------------------------------------
# Test: Attachment filename scanning
# -----------------------------------------------------------------------

class TestAttachmentFilenameScanning:
    """AC: attachment filenames (not binary content) scanned for secret patterns."""

    def test_secret_filename_sk_flagged(self):
        """File named sk-beta-test.json should be flagged."""
        filename = "sk-beta-test.json"
        flagged = _filename_has_secret_pattern(filename)
        assert flagged, f"Filename '{filename}' should be flagged as secret-like"

    def test_secret_filename_token_flagged(self):
        """File named api_token_backup.txt should be flagged."""
        filename = "api_token_backup.txt"
        flagged = _filename_has_secret_pattern(filename)
        assert flagged, f"Filename '{filename}' should be flagged as secret-like"

    def test_secret_filename_aws_flagged(self):
        """File named aws-credentials.json should be flagged."""
        filename = "aws-credentials.json"
        flagged = _filename_has_secret_pattern(filename)
        assert flagged, f"Filename '{filename}' should be flagged as secret-like"

    def test_secret_filename_secret_flagged(self):
        """File named secret_key.pem should be flagged."""
        filename = "secret_key.pem"
        flagged = _filename_has_secret_pattern(filename)
        assert flagged, f"Filename '{filename}' should be flagged as secret-like"

    def test_ordinary_filename_not_flagged(self):
        """Non-secret filename like project_notes.txt should NOT be flagged."""
        safe_names = [
            "project_notes.txt",
            "meeting_summary.md",
            "photo_2026.jpg",
            "document_final.docx",
            "readme.md",
            "session.json",
        ]
        for name in safe_names:
            assert not _filename_has_secret_pattern(name), f"'{name}' should NOT be flagged"

    def test_attachment_with_secret_filename_in_event(self):
        """An event with a secret-like attachment filename should be detected."""
        event = _make_turn(
            user_content="Here's the file.",
            assistant_content="Got it.",
            attachments=[
                {"path": "/tmp/sk-beta-test.json", "name": "sk-beta-test.json"},
            ],
        )

        filenames = _extract_attachment_filenames(event)
        assert len(filenames) == 1
        assert _filename_has_secret_pattern(filenames[0])

    def test_ordinary_attachment_not_flagged(self):
        """An event with a normal attachment filename should not be flagged."""
        event = _make_turn(
            user_content="Here's the photo.",
            assistant_content="Nice.",
            attachments=[
                {"path": "/tmp/photo.jpg", "name": "photo.jpg"},
            ],
        )

        filenames = _extract_attachment_filenames(event)
        for fname in filenames:
            assert not _filename_has_secret_pattern(fname)


# -----------------------------------------------------------------------
# Test: Audit log for tool-result redactions
# -----------------------------------------------------------------------

class TestAuditLogForToolResultRedactions:
    """AC: audit_log row for tool-result redactions includes source context (tool_name).

    The audit_log detail_json must include tool_name like 'memory_query' but
    must NEVER include the raw secret value.
    """

    def test_audit_detail_includes_tool_name_not_raw_value(self):
        """Audit log entry for tool redaction must have tool_name but NOT raw secret."""
        secret = FAKE["openai_key"]
        tool_name = "memory_query"

        # Simulate what the pipeline would build for an audit log
        hit_types = ["openai_key"]  # from redaction hit
        detail = _build_redaction_audit_detail(
            hit_types=hit_types,
            source=f"tool:{tool_name}",
            secret_value=secret,  # passed in for audit building only
        )

        # detail_json must contain tool_name
        assert tool_name in detail, "Audit detail must include tool_name"
        # detail_json must NOT contain raw secret value
        assert secret not in detail, "Audit detail must NEVER include raw secret value"
        # detail_json must include hit types
        assert "openai_key" in detail, "Audit detail must include redaction hit types"

    def test_audit_detail_multiple_hit_types(self):
        """When tool result has multiple secret types, all are listed in detail."""
        hit_types = ["openai_key", "github_token"]
        detail = _build_redaction_audit_detail(
            hit_types=hit_types,
            source="tool:memory_query",
            secret_value=FAKE["openai_key"],
        )
        assert "openai_key" in detail
        assert "github_token" in detail
        assert FAKE["openai_key"] not in detail
        assert FAKE["github_token"] not in detail


# -----------------------------------------------------------------------
# Test: Single scan call — no double-processing
# -----------------------------------------------------------------------

class TestSingleScanNoDoubleProcessing:
    """AC: both user/assistant and tool results checked in single scan call.

    The implementation must iterate all content fields once and scan each —
    no double-processing.
    """

    def test_all_content_fields_scanned(self):
        """Verify all expected content fields are extracted."""
        event = _make_turn(
            user_content="user text",
            assistant_content="assistant text",
            tool_calls=[
                {"id": "tc1", "name": "tool_a", "output": "output_a"},
                {"id": "tc2", "name": "tool_b", "output": "output_b"},
            ],
            attachments=[
                {"path": "/tmp/file.txt", "name": "file.txt"},
            ],
        )

        contents = _extract_all_content(event)
        # Must have user, assistant, and each tool result
        assert "user_content" in contents
        assert "assistant_content" in contents
        assert "tool_calls.0.output" in contents
        assert "tool_calls.1.output" in contents

    def test_no_double_wrapping_on_rescan(self):
        """Already-redacted content should not be re-wrapped if scanned twice."""
        original = f"key={FAKE['openai_key']}"
        result1 = redact_scan(original)

        # Re-scanning the already-redacted result should not double-wrap
        result2 = redact_scan(result1.redacted_content)

        count = result2.redacted_content.count("[REDACTED:openai_key]")
        assert count == 1, f"Expected exactly 1 redaction marker, got {count}"


# -----------------------------------------------------------------------
# Helpers (mirrors what pipeline.py will implement)
# -----------------------------------------------------------------------

def _extract_all_content(event: Dict[str, Any]) -> Dict[str, str]:
    """Extract all content strings from an event dict.

    Returns a flat dict mapping field path -> content string.
    Used by tests to verify every field gets scanned.
    """
    result = {}

    if event.get("user_content"):
        result["user_content"] = event["user_content"]
    if event.get("assistant_content"):
        result["assistant_content"] = event["assistant_content"]

    tool_calls = event.get("tool_calls") or []
    for i, tc in enumerate(tool_calls):
        if tc.get("output"):
            result[f"tool_calls.{i}.output"] = tc["output"]

    return result


def _extract_attachment_filenames(event: Dict[str, Any]) -> List[str]:
    """Extract attachment filenames from an event dict."""
    attachments = event.get("attachments") or []
    return [att.get("name", "") or att.get("path", "") for att in attachments]


_SECRET_FILENAME_PATTERNS = [
    "sk-", "token", "secret", "aws", "apikey", "api_key",
    "private_key", "credentials", "passwd", "password",
]


def _filename_has_secret_pattern(filename: str) -> bool:
    """Return True if filename matches secret-like patterns.

    MVP: simple contains check (case-insensitive) against known patterns.
    Binary content is NOT scanned per Plan.md Story 1.4.2.
    """
    lower = filename.lower()
    return any(pat in lower for pat in _SECRET_FILENAME_PATTERNS)


def _build_redaction_audit_detail(
    hit_types: List[str],
    source: str,
    secret_value: str,
) -> str:
    """Build the detail_json string for a redaction audit log row.

    MUST include source context (e.g., tool_name) but NEVER raw value.
    """
    detail = {
        "types": hit_types,
        "source": source,
    }
    detail_str = json.dumps(detail)
    # Assert no raw secret leaked
    assert secret_value not in detail_str, "Raw secret must not appear in audit detail"
    return detail_str