"""Test canonical event schema and validator (Story T-004).

RED PHASE: Tests are written first against the Plan.md + TDD §4.1/§7.3 spec.
GREEN PHASE: Implementation follows to make tests pass.
REFACTOR: Clean up after green.

The event is the atomic unit captured by hermes-memory. Every Hermes turn
generates one event. Schema version is exposed as EventSchema.VERSION.
"""
from __future__ import annotations

import pytest

# The module under test
from hermes_memory_core.schema import (
    EVENT_STATUS_VALUES,
    EventSchema,
    EventValidationError,
    EventValidator,
    StatusValue,
    validate_event,
)


# -----------------------------------------------------------------------
# Helper: minimal valid event
# -----------------------------------------------------------------------

def make_event(overrides=None):
    """Return a minimal valid event dict, with optional field overrides."""
    base = {
        "event_id": "evt_test_001",
        "session_id": "sess_abc123",
        "turn_id": "turn_001",
        "sequence": 1,
        "timestamp": "2026-05-17T22:00:00Z",
        "role": "user",
        "content": "Hello, world!",
        "agent": "mini-max/minimax-m2.7",
        "project": "default",
        "source": "message",
        "tags": [],
        "tool_calls": None,
        "attachments": [],
        "metadata": {},
        "hash": "a" * 64,  # SHA-256 hex = 64 chars
        "parent_turn_id": None,
        "embedding_status": "pending",
        "index_status": "pending",
        "dream_status": "pending",
    }
    if overrides:
        base.update(overrides)
    return base


# -----------------------------------------------------------------------
# Tests: EventSchema.VERSION
# -----------------------------------------------------------------------

class TestSchemaVersion:
    """AC: schema version is accessible via EventSchema.VERSION."""

    def test_version_is_exposed(self):
        assert hasattr(EventSchema, "VERSION")
        assert isinstance(EventSchema.VERSION, str)
        assert len(EventSchema.VERSION) > 0

    def test_version_is_semver_like(self):
        parts = EventSchema.VERSION.split(".")
        assert len(parts) == 3, f"VERSION '{EventSchema.VERSION}' should be semver X.Y.Z"
        for p in parts:
            assert p.isdigit(), f"Each part of VERSION should be numeric: {EventSchema.VERSION}"


# -----------------------------------------------------------------------
# Tests: valid event passes validation
# -----------------------------------------------------------------------

class TestValidEvent:
    """AC: given a full valid event, validate_event returns without error."""

    def test_minimal_valid_event_passes(self):
        event = make_event()
        # Should not raise
        result = validate_event(event)
        assert result is None

    def test_valid_event_with_all_fields(self):
        event = make_event({
            "tool_calls": [
                {"id": "call_1", "name": "search", "args": {"query": "pizza"}}
            ],
            "attachments": [{"type": "image", "url": "http://example.com/img.png"}],
            "metadata": {"ip": "1.2.3.4"},
            "tags": ["urgent", "flagged"],
            "parent_turn_id": "turn_000",
        })
        result = validate_event(event)
        assert result is None


# -----------------------------------------------------------------------
# Tests: missing required field raises clear error
# -----------------------------------------------------------------------

class TestMissingRequiredFields:
    """AC: missing required field raises EventValidationError naming the field."""

    @pytest.mark.parametrize("field", [
        "event_id", "session_id", "turn_id", "sequence",
        "timestamp", "role", "content", "agent",
        "project", "source", "hash",
    ])
    def test_missing_required_field_raises(self, field):
        event = make_event()
        del event[field]
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert field in str(exc_info.value)

    def test_missing_event_id_message_contains_event_id(self):
        event = make_event()
        del event["event_id"]
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert "event_id" in str(exc_info.value).lower()

    def test_missing_role_message_contains_role(self):
        event = make_event()
        del event["role"]
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert "role" in str(exc_info.value).lower()


# -----------------------------------------------------------------------
# Tests: forward-reference parent_turn_id is weakly validated
# -----------------------------------------------------------------------

class TestParentTurnIdForwardRef:
    """AC: parent_turn_id referencing non-existent turn is accepted (weak validation)."""

    def test_none_parent_turn_id_accepted(self):
        event = make_event({"parent_turn_id": None})
        result = validate_event(event)
        assert result is None

    def test_string_parent_turn_id_accepted(self):
        event = make_event({"parent_turn_id": "turn_nonexistent_999"})
        result = validate_event(event)
        assert result is None


# -----------------------------------------------------------------------
# Tests: status enum fields
# -----------------------------------------------------------------------

class TestStatusEnums:
    """AC: embedding_status / index_status / dream_status accept 'pending'|'indexed'|'failed'."""

    @pytest.mark.parametrize("field", ["embedding_status", "index_status", "dream_status"])
    @pytest.mark.parametrize("value", ["pending", "indexed", "failed"])
    def test_valid_status_value_accepted(self, field, value):
        event = make_event({field: value})
        result = validate_event(event)
        assert result is None

    @pytest.mark.parametrize("field", ["embedding_status", "index_status", "dream_status"])
    def test_invalid_status_value_rejected(self, field):
        event = make_event({field: "invalid_status"})
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert field in str(exc_info.value)


# -----------------------------------------------------------------------
# Tests: tool_calls is list or null
# -----------------------------------------------------------------------

class TestToolCalls:
    """AC: tool_calls is accepted as list or null."""

    def test_tool_calls_null_accepted(self):
        event = make_event({"tool_calls": None})
        result = validate_event(event)
        assert result is None

    def test_tool_calls_list_accepted(self):
        event = make_event({
            "tool_calls": [
                {"id": "call_1", "name": "search", "args": {}},
            ]
        })
        result = validate_event(event)
        assert result is None

    def test_tool_calls_empty_list_accepted(self):
        event = make_event({"tool_calls": []})
        result = validate_event(event)
        assert result is None

    def test_tool_calls_dict_rejected(self):
        event = make_event({"tool_calls": {"id": "call_1"}})
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert "tool_calls" in str(exc_info.value)

    def test_tool_calls_string_rejected(self):
        event = make_event({"tool_calls": "not a list"})
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert "tool_calls" in str(exc_info.value)


# -----------------------------------------------------------------------
# Tests: attachments is list
# -----------------------------------------------------------------------

class TestAttachments:
    """AC: attachments must be a list."""

    def test_attachments_empty_list_accepted(self):
        event = make_event({"attachments": []})
        result = validate_event(event)
        assert result is None

    def test_attachments_list_accepted(self):
        event = make_event({"attachments": [{"type": "file", "name": "report.pdf"}]})
        result = validate_event(event)
        assert result is None

    def test_attachments_dict_rejected(self):
        event = make_event({"attachments": {}})
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert "attachments" in str(exc_info.value)


# -----------------------------------------------------------------------
# Tests: tags is list
# -----------------------------------------------------------------------

class TestTags:
    """AC: tags must be a list."""

    def test_tags_empty_list_accepted(self):
        event = make_event({"tags": []})
        result = validate_event(event)
        assert result is None

    def test_tags_list_accepted(self):
        event = make_event({"tags": ["flagged", "important"]})
        result = validate_event(event)
        assert result is None

    def test_tags_string_rejected(self):
        event = make_event({"tags": "flagged"})
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert "tags" in str(exc_info.value)


# -----------------------------------------------------------------------
# Tests: metadata is dict
# -----------------------------------------------------------------------

class TestMetadata:
    """AC: metadata must be a dict."""

    def test_metadata_empty_dict_accepted(self):
        event = make_event({"metadata": {}})
        result = validate_event(event)
        assert result is None

    def test_metadata_dict_accepted(self):
        event = make_event({"metadata": {"key": "value", "count": 42}})
        result = validate_event(event)
        assert result is None

    def test_metadata_list_rejected(self):
        event = make_event({"metadata": []})
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert "metadata" in str(exc_info.value)

    def test_metadata_string_rejected(self):
        event = make_event({"metadata": "not a dict"})
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert "metadata" in str(exc_info.value)


# -----------------------------------------------------------------------
# Tests: sequence is non-negative integer
# -----------------------------------------------------------------------

class TestSequence:
    """AC: sequence must be a non-negative integer."""

    @pytest.mark.parametrize("value", [0, 1, 42, 999999])
    def test_valid_sequence_accepted(self, value):
        event = make_event({"sequence": value})
        result = validate_event(event)
        assert result is None

    def test_sequence_negative_rejected(self):
        event = make_event({"sequence": -1})
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert "sequence" in str(exc_info.value)

    def test_sequence_string_rejected(self):
        event = make_event({"sequence": "1"})
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert "sequence" in str(exc_info.value)


# -----------------------------------------------------------------------
# Tests: hash is string of expected length
# -----------------------------------------------------------------------

class TestHash:
    """AC: hash must be a string (SHA-256 hex = 64 chars)."""

    def test_hash_64_char_hex_accepted(self):
        event = make_event({"hash": "a" * 64})
        result = validate_event(event)
        assert result is None

    def test_hash_empty_string_rejected(self):
        event = make_event({"hash": ""})
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert "hash" in str(exc_info.value)

    def test_hash_too_short_rejected(self):
        event = make_event({"hash": "abc"})
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert "hash" in str(exc_info.value)


# -----------------------------------------------------------------------
# Tests: EventValidationError shape
# -----------------------------------------------------------------------

class TestEventValidationError:
    """EventValidationError has meaningful attributes."""

    def test_error_has_field_name(self):
        event = make_event()
        del event["role"]
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert exc_info.value.field_name == "role"

    def test_error_has_message(self):
        event = make_event()
        del event["content"]
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert len(exc_info.value.message) > 0

    def test_error_is_runtime_exception(self):
        with pytest.raises(EventValidationError):
            validate_event({})
        # Should be catchable as Exception


# -----------------------------------------------------------------------
# Tests: EventValidator class API
# -----------------------------------------------------------------------

class TestEventValidatorClass:
    """EventValidator.validate() is the primary API."""

    def test_validate_returns_none_on_valid(self):
        validator = EventValidator()
        result = validator.validate(make_event())
        assert result is None

    def test_validate_returns_error_on_invalid(self):
        validator = EventValidator()
        event = make_event()
        del event["event_id"]
        result = validator.validate(event)
        assert isinstance(result, EventValidationError)
        assert "event_id" in str(result)