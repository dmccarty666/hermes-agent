"""Canonical event schema for Hermes Local Memory.

Schema version: ``EventSchema.VERSION`` (semver).

The event is the atomic unit captured per Hermes turn. All fields are
defined here with their types and constraints. The validator
(``EventValidator`` / ``validate_event()``) enforces required fields,
type correctness, and enum constraints.

Reference: Plan.md §3 Epic 1.3, Story 1.3.1 — TDD §4.1 (capture flow),
TDD §7.3 (event fields).
"""

from __future__ import annotations

__all__ = [
    "EventSchema",
    "EventValidationError",
    "EventValidator",
    "validate_event",
    "EVENT_STATUS_VALUES",
    "StatusValue",
]

# -----------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------

StatusValue = {"pending", "indexed", "failed"}
"""Allowed values for embedding_status / index_status / dream_status."""

EVENT_STATUS_VALUES: set = StatusValue  # backward-compatible alias


# -----------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------

class EventValidationError(Exception):
    """Raised when an event fails schema validation.

    Attributes:
        field_name:  The field that failed validation.
        message:     Human-readable description of what went wrong.
    """

    __slots__ = ("field_name", "message")

    def __init__(self, field_name: str, message: str):
        self.field_name = field_name
        self.message = message
        super().__init__(f"[{field_name}] {message}")

    def __repr__(self) -> str:
        return f"EventValidationError(field_name={self.field_name!r}, message={self.message!r})"

    def __str__(self) -> str:
        return super().__str__()


# -----------------------------------------------------------------------
# Schema definition
# -----------------------------------------------------------------------

class EventSchema:
    """Canonical event schema definition.

    Attributes:
        VERSION: Schema version string (semver). Consumers can read this
                 to understand which fields are present.
    """

    VERSION = "1.0.0"

    # Required string fields (non-empty)
    REQUIRED_STRINGS: tuple = (
        "event_id",
        "session_id",
        "turn_id",
        "timestamp",
        "role",
        "content",
        "agent",
        "project",
        "source",
    )

    # Optional string fields (can be None or non-empty string)
    OPTIONAL_STRINGS: tuple = (
        "parent_turn_id",
    )

    # Required list fields
    REQUIRED_LISTS: tuple = (
        "tags",
        "attachments",
    )

    # Required dict field
    REQUIRED_DICTS: tuple = (
        "metadata",
    )

    # Fields that must be non-negative integers
    NON_NEGATIVE_INTEGERS: tuple = (
        "sequence",
    )

    # Fields that must be SHA-256 hex strings (64 chars)
    HASH_FIELDS: tuple = (
        "hash",
    )

    # Status enum fields
    STATUS_FIELDS: tuple = (
        "embedding_status",
        "index_status",
        "dream_status",
    )

    # Fields that must be list OR null
    LIST_OR_NULL_FIELDS: tuple = (
        "tool_calls",
    )

    @classmethod
    def required_fields(cls) -> frozenset:
        """All required field names (excludes optional/nullable fields)."""
        return frozenset(
            cls.REQUIRED_STRINGS
            + cls.REQUIRED_LISTS
            + cls.REQUIRED_DICTS
            + cls.NON_NEGATIVE_INTEGERS
            + cls.HASH_FIELDS
            + cls.STATUS_FIELDS
        )

    @classmethod
    def all_fields(cls) -> frozenset:
        """Every field name defined by the schema."""
        return frozenset(
            cls.REQUIRED_STRINGS
            + cls.OPTIONAL_STRINGS
            + cls.REQUIRED_LISTS
            + cls.REQUIRED_DICTS
            + cls.NON_NEGATIVE_INTEGERS
            + cls.HASH_FIELDS
            + cls.STATUS_FIELDS
            + cls.LIST_OR_NULL_FIELDS
        )


# -----------------------------------------------------------------------
# Validator
# -----------------------------------------------------------------------

class EventValidator:
    """Validates event dicts against ``EventSchema``.

    The ``validate(event)`` method returns ``None`` on success and
    an ``EventValidationError`` on failure. It does NOT mutate the input.

    Example::

        validator = EventValidator()
        error = validator.validate(event_dict)
        if error:
            print(f"Invalid: {error.field_name} — {error.message}")
    """

    def validate(self, event: dict) -> EventValidationError | None:
        """Validate an event dict.

        Returns:
            ``None`` if the event is valid.
            ``EventValidationError`` describing the first failure found.
        """
        if not isinstance(event, dict):
            return EventValidationError(
                "<root>",
                f"event must be a dict, got {type(event).__name__}",
            )

        # Check required string fields (must be present and non-empty string)
        for field in EventSchema.REQUIRED_STRINGS:
            if field not in event:
                return EventValidationError(field, "required field is missing")
            value = event[field]
            if not isinstance(value, str):
                return EventValidationError(field, f"must be a string, got {type(value).__name__}")
            if value == "":
                return EventValidationError(field, "must be a non-empty string")

        # Check optional string fields (None or non-empty string)
        for field in EventSchema.OPTIONAL_STRINGS:
            if field in event:
                value = event[field]
                if value is not None and not isinstance(value, str):
                    return EventValidationError(field, f"must be a string or null, got {type(value).__name__}")

        # Check required list fields (must be present and a list)
        for field in EventSchema.REQUIRED_LISTS:
            if field not in event:
                return EventValidationError(field, "required field is missing")
            value = event[field]
            if not isinstance(value, list):
                return EventValidationError(field, f"must be a list, got {type(value).__name__}")

        # Check required dict field (must be present and a dict)
        for field in EventSchema.REQUIRED_DICTS:
            if field not in event:
                return EventValidationError(field, "required field is missing")
            value = event[field]
            if not isinstance(value, dict):
                return EventValidationError(field, f"must be a dict, got {type(value).__name__}")

        # Check non-negative integer fields
        for field in EventSchema.NON_NEGATIVE_INTEGERS:
            if field not in event:
                return EventValidationError(field, "required field is missing")
            value = event[field]
            if not isinstance(value, int) or isinstance(value, bool):
                return EventValidationError(field, f"must be an integer, got {type(value).__name__}")
            if value < 0:
                return EventValidationError(field, f"must be non-negative, got {value}")

        # Check hash fields (must be 64-char hex string)
        for field in EventSchema.HASH_FIELDS:
            if field not in event:
                return EventValidationError(field, "required field is missing")
            value = event[field]
            if not isinstance(value, str):
                return EventValidationError(field, f"must be a string, got {type(value).__name__}")
            if len(value) < 64:
                return EventValidationError(field, f"must be a 64-char SHA-256 hex string, got length {len(value)}")

        # Check status enum fields
        for field in EventSchema.STATUS_FIELDS:
            if field not in event:
                return EventValidationError(field, "required field is missing")
            value = event[field]
            if not isinstance(value, str):
                return EventValidationError(field, f"must be a string, got {type(value).__name__}")
            if value not in EVENT_STATUS_VALUES:
                return EventValidationError(
                    field,
                    f"must be one of {sorted(EVENT_STATUS_VALUES)}, got {value!r}",
                )

        # Check list-or-null fields
        for field in EventSchema.LIST_OR_NULL_FIELDS:
            if field in event:
                value = event[field]
                if value is not None and not isinstance(value, list):
                    return EventValidationError(field, f"must be a list or null, got {type(value).__name__}")

        return None


# -----------------------------------------------------------------------
# Module-level convenience function
# -----------------------------------------------------------------------

_validator = EventValidator()


def validate_event(event: dict) -> None:
    """Validate an event dict against the canonical schema.

    This is a convenience wrapper around ``EventValidator().validate()``.

    Returns:
        ``None`` if the event is valid.

    Raises:
        EventValidationError: if the event is invalid. The exception's
            ``field_name`` attribute names the offending field and
            ``message`` gives a human-readable description.

    Example::

        >>> validate_event({"event_id": "evt_1", ...})  # raises if invalid
    """
    error = _validator.validate(event)
    if error is not None:
        raise error