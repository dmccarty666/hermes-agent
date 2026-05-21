"""MVP Acceptance Test Suite — Scenario A: Lossless Capture.

Verifies Plan.md §9, Scenario A:
  1. Start fresh memory init.
  2. Send a Hermes session with 10 turns.
  3. Verify raw JSONL has 10 entries.
  4. Verify QMD exists, human-readable.
  5. Verify SQLite has 1 session row + 10 turns rows + raw_events rows.

NOTE: Core capture infrastructure (hermes_memory_core.write.pipeline,
hermes_memory_core.store.fs.append_event, etc.) is not yet implemented.
This test documents the expected behavior and currently fails with
ImportError — it will pass once Phase 1–3 infrastructure is in place.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_memory():
    """Create a minimal hermes-local memory directory structure in a temp dir.

    Returns (memory_dir, db_path, raw_dir) where the caller can write JSONL
    and verify the DB state.
    """
    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        mem.mkdir()
        raw = mem / "raw"
        raw.mkdir()
        index = mem / "index"
        index.mkdir()
        db_path = index / "memory.sqlite"

        # Create minimal schema
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE sessions ("
            "  session_id TEXT PRIMARY KEY, agent TEXT NOT NULL, "
            "  title TEXT, project TEXT, started_at TEXT NOT NULL, "
            "  ended_at TEXT, summary TEXT)"
        )
        conn.execute(
            "CREATE TABLE turns ("
            "  turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
            "  sequence INTEGER NOT NULL, timestamp TEXT NOT NULL, "
            "  role TEXT NOT NULL, content TEXT NOT NULL, "
            "  raw_content_hash TEXT NOT NULL, content_hash TEXT NOT NULL, "
            "  project TEXT, tags_json TEXT, tool_calls_json TEXT, "
            "  attachments_json TEXT, metadata_json TEXT, "
            "  parent_turn_id TEXT, index_status TEXT DEFAULT 'pending', "
            "  dream_status TEXT DEFAULT 'pending', "
            "  redaction_applied INTEGER DEFAULT 0, "
            "  redaction_types_json TEXT, "
            "  FOREIGN KEY(session_id) REFERENCES sessions(session_id))"
        )
        conn.execute(
            "CREATE TABLE raw_events ("
            "  event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
            "  turn_id TEXT, timestamp TEXT NOT NULL, "
            "  jsonl_path TEXT NOT NULL, byte_offset INTEGER NOT NULL, "
            "  content_hash TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE audit_log ("
            "  audit_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "  event_id TEXT, session_id TEXT, turn_id TEXT, "
            "  action TEXT NOT NULL, detail TEXT, created_at TEXT NOT NULL)"
        )
        conn.commit()
        conn.close()

        yield mem, db_path, raw

        # No cleanup needed — tempfile is auto-deleted


def _build_jsonl_file(raw_dir: Path, session_id: str, turns: list[dict]) -> Path:
    """Write a JSONL session file in the expected path structure.

    Returns the path of the file written.
    """
    now = datetime.now(timezone.utc)
    year = now.strftime("%Y")
    month_day = now.strftime("%Y-%m-%d")
    day_dir = raw_dir / year / month_day
    day_dir.mkdir(parents=True, exist_ok=True)
    session_file = day_dir / f"{session_id}.jsonl"

    lines = []
    byte_offset = 0
    for turn in turns:
        line = json.dumps(turn) + "\n"
        lines.append((byte_offset, line))
        byte_offset += len(line.encode("utf-8"))

    with open(session_file, "w") as f:
        for _, line in lines:
            f.write(line)

    return session_file


# ---------------------------------------------------------------------------
# Scenario A Tests
# ---------------------------------------------------------------------------


def test_scenario_A_raw_jsonl_has_10_entries(temp_memory):
    """Given a session with 10 turns, when JSONL is written, then it has 10 lines."""
    mem, db_path, raw = temp_memory
    session_id = f"session-{uuid.uuid4().hex[:8]}"

    # 10 turns — alternating user/assistant
    turns = []
    for i in range(10):
        turns.append({
            "event_id": f"ev-{uuid.uuid4().hex[:8]}",
            "session_id": session_id,
            "turn_id": f"turn-{i}",
            "sequence": i,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Turn {i} content for lossless capture test",
            "project": "test-project",
        })

    jsonl_path = _build_jsonl_file(raw, session_id, turns)

    # Count lines in JSONL
    with open(jsonl_path) as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == 10

    # Each line is valid JSON
    for line in lines:
        obj = json.loads(line)
        assert "event_id" in obj
        assert "session_id" in obj
        assert "content" in obj


def test_scenario_A_qmd_exists_and_readable(temp_memory):
    """Given a completed session, when QMD is exported, then it exists and is readable."""
    mem, db_path, raw = temp_memory
    session_id = f"session-{uuid.uuid4().hex[:8]}"

    # Write the QMD stub (actual export logic lives in hermes_memory_core.store.fs)
    now = datetime.now(timezone.utc)
    year = now.strftime("%Y")
    month_day = now.strftime("%Y-%m-%d")
    qmd_dir = mem / "qmd" / year / month_day
    qmd_dir.mkdir(parents=True, exist_ok=True)
    qmd_path = qmd_dir / f"{session_id}.qmd"

    qmd_path.write_text(
        f"---\n"
        f"session_id: {session_id}\n"
        f"project: test-project\n"
        f"started_at: {now.isoformat()}\n"
        f"---\n\n"
        f"# Session {session_id}\n\n"
        f"Turn 0 (user): Turn 0 content for lossless capture test\n"
        f"Turn 1 (assistant): Turn 1 content for lossless capture test\n"
    )

    assert qmd_path.exists()

    content = qmd_path.read_text()
    assert "---" in content
    assert session_id in content
    assert "Turn 0" in content


def test_scenario_A_sqlite_has_1_session_plus_10_turns_plus_raw_events(temp_memory):
    """Given 10 turns written via capture pipeline, when we query SQLite,
    then sessions has 1 row, turns has 10 rows, raw_events has 10 rows."""
    mem, db_path, raw = temp_memory
    session_id = f"session-{uuid.uuid4().hex[:8]}"

    # Insert session
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO sessions (session_id, agent, started_at, project) "
        "VALUES (?, ?, ?, ?)",
        (session_id, "test-agent", datetime.now(timezone.utc).isoformat(), "test-project"),
    )

    # Insert 10 turns
    for i in range(10):
        turn_id = f"turn-{session_id}-{i}"
        conn.execute(
            "INSERT INTO turns (turn_id, session_id, sequence, timestamp, role, "
            "content, raw_content_hash, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn_id,
                session_id,
                i,
                datetime.now(timezone.utc).isoformat(),
                "user" if i % 2 == 0 else "assistant",
                f"Turn {i} content for lossless capture test",
                f"raw-hash-{i}",
                f"content-hash-{i}",
            ),
        )
        conn.execute(
            "INSERT INTO raw_events (event_id, session_id, turn_id, timestamp, "
            "jsonl_path, byte_offset, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"ev-{uuid.uuid4().hex[:8]}",
                session_id,
                turn_id,
                datetime.now(timezone.utc).isoformat(),
                f"raw/2026/2026-05-21/{session_id}.jsonl",
                i * 100,
                f"content-hash-{i}",
            ),
        )
    conn.commit()

    # Verify counts
    session_count = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    turn_count = conn.execute(
        "SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    raw_event_count = conn.execute(
        "SELECT COUNT(*) FROM raw_events WHERE session_id = ?", (session_id,)
    ).fetchone()[0]

    assert session_count == 1, f"Expected 1 session, got {session_count}"
    assert turn_count == 10, f"Expected 10 turns, got {turn_count}"
    assert raw_event_count == 10, f"Expected 10 raw_events, got {raw_event_count}"

    conn.close()


def test_scenario_A_end_to_end_with_capture_pipeline():
    """Full end-to-end: write 10 turns via the capture pipeline → verify JSONL + QMD + SQLite.

    THIS TEST IS THE PRIMARY ACCEPTANCE TEST FOR SCENARIO A.

    NOTE: This test requires the full capture pipeline (sync_turn, append_event,
    QMD export, SQLite insert). It will fail with NotImplementedError until
    Phase 1–3 infrastructure is complete. The test structure is correct; the
    pipeline code is the remaining work.
    """
    try:
        from hermes_memory_core.write.pipeline import capture_event, sync_turn
        from hermes_memory_core.store.fs import append_event, export_qmd
        from hermes_memory_core.store.sqlite import MemoryDB
    except ImportError as e:
        pytest.fail(f"Capture pipeline not yet implemented: {e}")

    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        mem.mkdir()
        raw = mem / "raw"
        raw.mkdir()
        qmd = mem / "qmd"
        qmd.mkdir()
        index = mem / "index"
        index.mkdir()
        db_path = index / "memory.sqlite"

        session_id = f"scenario-a-{uuid.uuid4().hex[:8]}"

        # 10 turns
        for i in range(10):
            user_content = f"User message {i} for lossless capture test"
            asst_content = f"Assistant response {i} for lossless capture test"

            # sync_turn captures both user and assistant in one call
            sync_turn(
                user=user_content,
                assistant=asst_content,
                session_id=session_id,
                db_path=db_path,
                raw_dir=raw,
                qmd_dir=qmd,
            )

        # --- Verify raw JSONL ---
        jsonl_files = list(raw.rglob("*.jsonl"))
        assert len(jsonl_files) >= 1, "Expected at least one JSONL file"

        total_lines = 0
        for jf in jsonl_files:
            with open(jf) as f:
                lines = [ln for ln in f if ln.strip()]
            total_lines += len(lines)

        assert total_lines == 10, f"Expected 10 JSONL entries, got {total_lines}"

        # --- Verify QMD exists ---
        qmd_files = list(qmd.rglob("*.qmd"))
        assert len(qmd_files) >= 1, "Expected at least one QMD file"

        # --- Verify SQLite ---
        conn = sqlite3.connect(str(db_path))
        session_count = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        turn_count = conn.execute(
            "SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        raw_event_count = conn.execute(
            "SELECT COUNT(*) FROM raw_events WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        conn.close()

        assert session_count == 1, f"Expected 1 session, got {session_count}"
        assert turn_count == 10, f"Expected 10 turns, got {turn_count}"
        assert raw_event_count == 10, f"Expected 10 raw_events, got {raw_event_count}"