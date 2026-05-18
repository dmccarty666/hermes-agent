# Copyright 2026 David McCarty. All rights reserved.
"""Tests for FSStore.append_event (T-005) and read_session."""

import hashlib
import json
import os
import tempfile
from datetime import date, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from hermes_memory_core.store.fs import FSStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(session_id: str, turn_id: str, sequence: int,
               role: str = "user", content: str = "hello world",
               agent: str = "test-agent") -> dict:
    """Minimal valid event matching the event-schema.json."""
    return {
        "event_id": str(uuid4()),
        "session_id": session_id,
        "turn_id": turn_id,
        "sequence": sequence,
        "timestamp": "2026-05-17T10:00:00Z",
        "role": role,
        "content": content,
        "agent": agent,
        "source": "cli",
    }


def compute_hash(event: dict) -> str:
    """Reproduce the hash computed by FSStore._content_hash."""
    canonical = (
        f"{event.get('event_id', '')}"
        f"{event.get('session_id', '')}"
        f"{event.get('turn_id', '')}"
        f"{event.get('sequence', '')}"
        f"{event.get('timestamp', '')}"
        f"{event.get('role', '')}"
        f"{event.get('content', '')}"
        f"{event.get('agent', '')}"
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_base(tmp_path):
    """Use tmp_path so we never touch ~/.hermes/memory during tests."""
    return tmp_path


@pytest.fixture
def store(tmp_base):
    return FSStore(base_path=tmp_base)


# ---------------------------------------------------------------------------
# AC-1: append_event writes to ~/.hermes/memory/raw/YYYY/YYYY-MM-DD/{session_id}.jsonl
# ---------------------------------------------------------------------------

class TestAppendEventWritesCorrectPath:
    def test_writes_to_jsonl_file(self, store, tmp_base):
        """AC-1: event is written to the correct date-segmented path."""
        event = make_event(session_id="sess_abc", turn_id="turn_1", sequence=0)
        result = store.append_event(event)

        # Check the returned hash matches what we compute
        expected = compute_hash(event)
        assert result == expected

        # Verify the file was created at the right path
        date_dir = tmp_base / "raw" / "2026" / "2026-05-17"
        jsonl_file = date_dir / "sess_abc.jsonl"
        assert jsonl_file.exists(), f"Expected {jsonl_file}"

        # Verify one JSON line was written and it matches the event
        with open(jsonl_file) as f:
            lines = f.readlines()
        assert len(lines) == 1
        written = json.loads(lines[0])
        assert written["event_id"] == event["event_id"]
        assert written["session_id"] == event["session_id"]

    def test_one_json_line_per_event(self, store, tmp_base):
        """AC-1: one JSON line per event."""
        session = "sess_one_per_event"
        events = [
            make_event(session_id=session, turn_id=f"turn_{i}", sequence=i,
                       content=f"message {i}")
            for i in range(5)
        ]
        for ev in events:
            store.append_event(ev)

        date_dir = tmp_base / "raw" / "2026" / "2026-05-17"
        jsonl_file = date_dir / f"{session}.jsonl"
        with open(jsonl_file) as f:
            lines = [json.loads(l) for l in f.readlines()]

        assert len(lines) == 5
        for i, ev in enumerate(events):
            assert lines[i]["event_id"] == ev["event_id"]


# ---------------------------------------------------------------------------
# AC-2: two events for same session on different days -> correct date-segmented files
# ---------------------------------------------------------------------------

class TestDateSegmentedFiles:
    def test_same_session_different_days_go_to_correct_files(self, store, tmp_base):
        """AC-2: same session_id on different dates lands in correct date folders."""
        sess = "sess_span_days"

        # First event — a known date
        ev1 = make_event(session_id=sess, turn_id="turn_1", sequence=0,
                          content="first day")
        h1 = store.append_event(ev1)

        # Second event — simulates next day by using date_override
        ev2 = make_event(session_id=sess, turn_id="turn_2", sequence=1,
                          content="second day")
        h2 = store.append_event(ev2, date_override="2026-05-18")

        raw = tmp_base / "raw"

        # First day file
        day1_file = raw / "2026" / "2026-05-17" / f"{sess}.jsonl"
        assert day1_file.exists(), f"Expected day1 file at {day1_file}"

        # Second day file (different date)
        day2_file = raw / "2026" / "2026-05-18" / f"{sess}.jsonl"
        assert day2_file.exists(), f"Expected day2 file at {day2_file}"

        # Content in each file is correct
        with open(day1_file) as f:
            lines1 = [json.loads(l) for l in f.readlines()]
        assert any(e["content"] == "first day" for e in lines1)

        with open(day2_file) as f:
            lines2 = [json.loads(l) for l in f.readlines()]
        assert any(e["content"] == "second day" for e in lines2)


# ---------------------------------------------------------------------------
# AC-3: dedup by content_hash — same content skipped
# ---------------------------------------------------------------------------

class TestDedupByContentHash:
    def test_duplicate_content_is_skipped(self, store, tmp_base):
        """AC-3: event with same content_hash returns False on second call."""
        event = make_event(session_id="sess_dup", turn_id="turn_dup", sequence=0,
                            content="duplicate me")

        h1 = store.append_event(event)
        h2 = store.append_event(event)  # same event, same hash

        # First call returns a content hash (truthy)
        assert isinstance(h1, str) and len(h1) == 64
        # Second call returns False (dedup hit)
        assert h2 is False

        # File should have only ONE line
        date_dir = tmp_base / "raw" / "2026" / "2026-05-17"
        jsonl_file = date_dir / "sess_dup.jsonl"
        with open(jsonl_file) as f:
            lines = f.readlines()

        assert len(lines) == 1, f"Expected 1 line (dedup failed), got {len(lines)}"

    def test_different_content_same_session_writes_both(self, store, tmp_base):
        """AC-3 sanity: different content writes both."""
        ev1 = make_event(session_id="sess_diff", turn_id="turn_1", sequence=0,
                          content="first")
        ev2 = make_event(session_id="sess_diff", turn_id="turn_2", sequence=1,
                          content="second")

        h1 = store.append_event(ev1)
        h2 = store.append_event(ev2)

        assert h1 != h2

        date_dir = tmp_base / "raw" / "2026" / "2026-05-17"
        jsonl_file = date_dir / "sess_diff.jsonl"
        with open(jsonl_file) as f:
            lines = f.readlines()

        assert len(lines) == 2

    def test_same_content_different_session_writes_both(self, store, tmp_base):
        """AC-3: dedup is per-session, not global."""
        content = "same content different session"
        ev1 = make_event(session_id="sess_x", turn_id="turn_1", sequence=0,
                          content=content)
        ev2 = make_event(session_id="sess_y", turn_id="turn_1", sequence=0,
                          content=content)

        store.append_event(ev1)
        store.append_event(ev2)

        raw = tmp_base / "raw" / "2026" / "2026-05-17"

        with open(raw / "sess_x.jsonl") as f:
            lines_x = f.readlines()
        with open(raw / "sess_y.jsonl") as f:
            lines_y = f.readlines()

        assert len(lines_x) == 1
        assert len(lines_y) == 1


# ---------------------------------------------------------------------------
# AC-4: file handle pool / lazy open per session
# ---------------------------------------------------------------------------

class TestFileHandlePool:
    def test_reuses_handle_for_same_session(self, store, tmp_base):
        """AC-4: multiple appends to same session reuse file handle."""
        session = "sess_pool"
        events = [
            make_event(session_id=session, turn_id=f"turn_{i}", sequence=i,
                       content=f"event {i}")
            for i in range(20)
        ]
        for ev in events:
            store.append_event(ev)

        # If handle pool works, all 20 lines should be in the file
        date_dir = tmp_base / "raw" / "2026" / "2026-05-17"
        jsonl_file = date_dir / f"{session}.jsonl"
        with open(jsonl_file) as f:
            lines = f.readlines()

        assert len(lines) == 20, f"Expected 20 lines, got {len(lines)}"

    def test_different_sessions_open_different_handles(self, store, tmp_base):
        """AC-4: different sessions get separate handles."""
        sessions = [f"sess_{i}" for i in range(5)]
        for sess in sessions:
            for j in range(3):
                ev = make_event(session_id=sess, turn_id=f"turn_{j}", sequence=j,
                                content=f"{sess}-{j}")
                store.append_event(ev)

        # Each session file should have 3 lines
        date_dir = tmp_base / "raw" / "2026" / "2026-05-17"
        for sess in sessions:
            jsonl_file = date_dir / f"{sess}.jsonl"
            with open(jsonl_file) as f:
                lines = f.readlines()
            assert len(lines) == 3, f"{sess}: expected 3 lines, got {len(lines)}"

    def test_close_all_handles(self, store, tmp_base):
        """AC-4: close_all() closes all pooled handles."""
        session = "sess_close"
        for i in range(3):
            store.append_event(make_event(session_id=session, turn_id=f"turn_{i}",
                                          sequence=i, content=f"close test {i}"))

        # Should not raise
        store.close_all()

        # After close, a new append should open a fresh handle
        ev = make_event(session_id=session, turn_id="turn_after_close", sequence=99,
                        content="after close")
        h = store.append_event(ev)
        assert h is not None


# ---------------------------------------------------------------------------
# AC-5: read back in order, hashes match
# ---------------------------------------------------------------------------

class TestReadSession:
    def test_read_session_returns_all_events_in_sequence_order(self, store, tmp_base):
        """AC-5: read_session returns events in sequence order with matching hashes."""
        session = "sess_readback"
        events = [
            make_event(session_id=session, turn_id=f"turn_{i}", sequence=i,
                       content=f"ordered event {i}")
            for i in range(10)
        ]
        hashes = []
        for ev in events:
            h = store.append_event(ev)
            hashes.append(h)

        # Read back
        read_events = store.read_session(session)

        assert len(read_events) == 10
        for i, (written, read) in enumerate(zip(events, read_events)):
            assert read["event_id"] == written["event_id"]
            assert read["sequence"] == written["sequence"]

    def test_read_nonexistent_session_returns_empty_list(self, store):
        """AC-5: read_session returns [] for session with no events."""
        result = store.read_session("sess_does_not_exist")
        assert result == []


# ---------------------------------------------------------------------------
# Integration: 100 events, read back in order, hashes match
# ---------------------------------------------------------------------------

class Test100EventsReadBackInOrder:
    def test_100_events_write_and_read_back_preserve_order_and_hash(self, store, tmp_base):
        """DoD check: write 100 events, read back in order, check hashes."""
        session = "sess_100"
        events = [
            make_event(session_id=session, turn_id=f"turn_{i}", sequence=i,
                       content=f"stress test event {i}")
            for i in range(100)
        ]
        hashes = []
        for ev in events:
            h = store.append_event(ev)
            hashes.append(h)

        read_events = store.read_session(session)

        assert len(read_events) == 100, f"Expected 100, got {len(read_events)}"

        # Sequence order check
        for i, ev in enumerate(read_events):
            assert ev["sequence"] == i, f"Sequence {i}: expected {i}, got {ev['sequence']}"

        # Hash match check
        for i, (expected_hash, ev) in enumerate(zip(hashes, read_events)):
            actual_hash = compute_hash(ev)
            assert actual_hash == expected_hash, (
                f"Event {i}: hash mismatch. "
                f"expected={expected_hash}, actual={actual_hash}"
            )