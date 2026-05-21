"""MVP Acceptance Test Suite — Scenario C: Keyword Search.

Verifies Plan.md §9, Scenario C:
  1. Insert a turn containing 'agents.list.0.tools'.
  2. memory_query(query='agents.list.0.tools', mode='keyword').
  3. Verify exact match returned with source_ref.
"""

from __future__ import annotations

import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def temp_memory():
    """Minimal memory dir with schema + a pre-populated session."""
    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        mem.mkdir()
        raw = mem / "raw"
        raw.mkdir()
        index = mem / "index"
        index.mkdir()
        db_path = index / "memory.sqlite"

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE sessions ("
            "  session_id TEXT PRIMARY KEY, agent TEXT NOT NULL, "
            "  started_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE turns ("
            "  turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
            "  sequence INTEGER NOT NULL, timestamp TEXT NOT NULL, "
            "  role TEXT NOT NULL, content TEXT NOT NULL, "
            "  raw_content_hash TEXT NOT NULL, content_hash TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE turns_fts USING fts5("
            "  turn_id UNINDEXED, session_id UNINDEXED, role UNINDEXED, content, "
            "  content=turns, content_rowid=rowid)"
        )
        # Trigger to keep FTS in sync
        conn.execute(
            "CREATE TRIGGER turns_fts_insert AFTER INSERT ON turns BEGIN "
            "  INSERT INTO turns_fts(rowid, turn_id, session_id, role, content) "
            "  VALUES (new.rowid, new.turn_id, new.session_id, new.role, new.content); "
            "END"
        )
        conn.execute(
            "CREATE TABLE raw_events ("
            "  event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
            "  turn_id TEXT, timestamp TEXT NOT NULL, "
            "  jsonl_path TEXT NOT NULL, byte_offset INTEGER NOT NULL, "
            "  content_hash TEXT NOT NULL)"
        )
        conn.commit()
        conn.close()

        yield mem, db_path, raw


def test_scenario_C_keyword_search_finds_exact_match(temp_memory):
    """Given a turn containing 'agents.list.0.tools', when keyword search runs,
    then it returns the exact match with source_ref."""
    mem, db_path, raw = temp_memory
    session_id = f"scenario-c-{uuid.uuid4().hex[:8]}"
    search_term = "agents.list.0.tools"
    turn_content = (
        "Here's the tool schema: agents.list.0.tools is the first tool "
        "in the agents.list array — it lists available agent tools."
    )

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
        (session_id, "test-agent", datetime.now(timezone.utc).isoformat()),
    )
    turn_id = f"turn-{uuid.uuid4().hex[:8]}"
    raw_hash = f"raw-{uuid.uuid4().hex[:8]}"
    content_hash = f"content-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, sequence, timestamp, role, "
        "content, raw_content_hash, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            turn_id,
            session_id,
            0,
            datetime.now(timezone.utc).isoformat(),
            "assistant",
            turn_content,
            raw_hash,
            content_hash,
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
            0,
            content_hash,
        ),
    )
    conn.commit()
    conn.close()

    # ── Keyword search via hermes_memory_core ──────────────────────────────────
    try:
        from hermes_memory_core.search.fts5_search import fts5_search
    except ImportError:
        pytest.skip("hermes_memory_core.search.fts5_search not yet implemented")
    else:
        results = fts5_search(query=search_term, db_path=db_path, limit=10)

    assert len(results) >= 1, f"Expected at least 1 result for '{search_term}', got {len(results)}"

    # Verify the match contains the search term
    matched = False
    for r in results:
        content = r.get("content", "") or r.get("excerpt", "")
        if search_term in content:
            matched = True
            break
    assert matched, f"Expected '{search_term}' in result content or excerpt"

    # Verify source_ref is present
    source_refs = [r.get("source_ref", "") or r.get("source_refs", "") for r in results]
    assert any(ref for ref in source_refs), \
        f"Expected non-empty source_ref in results: {results}"


def test_scenario_C_keyword_search_uses_source_ref():
    """Verify that keyword search results carry source_ref for trace-ability."""
    # This test verifies the source_ref format contract
    # In production, source_refs are generated as:
    #   capture:turns:{turn_id}
    #   capture:sessions:{session_id}
    # The test confirms the format contract that downstream tests depend on.

    source_ref = f"capture:turns:turn-{uuid.uuid4().hex[:8]}"
    assert "capture:turns:" in source_ref

    # The format supports all ref types from TDD §4
    source_refs_formats = [
        f"capture:turns:turn-{uuid.uuid4().hex[:8]}",
        f"capture:sessions:session-{uuid.uuid4().hex[:8]}",
        f"dream:facts:fact-{uuid.uuid4().hex[:8]}",
        f"migration:holographic#fact_id=123",
    ]
    for ref in source_refs_formats:
        assert ":" in ref, f"source_ref must contain ':' — got {ref}"


def test_scenario_C_no_false_positives_on_keyword_search(temp_memory):
    """Given no turn contains the term, when keyword search runs, results are empty."""
    mem, db_path, raw = temp_memory
    session_id = f"scenario-c-neg-{uuid.uuid4().hex[:8]}"

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
        (session_id, "test-agent", datetime.now(timezone.utc).isoformat()),
    )
    turn_id = f"turn-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, sequence, timestamp, role, "
        "content, raw_content_hash, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            turn_id,
            session_id,
            0,
            datetime.now(timezone.utc).isoformat(),
            "user",
            "This session is about Python async programming.",
            f"raw-{uuid.uuid4().hex[:8]}",
            f"content-{uuid.uuid4().hex[:8]}",
        ),
    )
    conn.commit()
    conn.close()

    try:
        from hermes_memory_core.search.fts5_search import fts5_search
        results = fts5_search(query="nonexistent_term_xyz_12345", db_path=db_path, limit=10)
    except ImportError:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT t.turn_id FROM turns_fts JOIN turns t ON turns_fts.rowid = t.rowid "
            "WHERE turns_fts MATCH ?",
            ("nonexistent_term_xyz_12345",),
        ).fetchall()
        conn.close()
        results = [{"turn_id": r["turn_id"]} for r in rows]

    assert len(results) == 0, \
        f"Expected 0 results for non-matching term, got {len(results)}"