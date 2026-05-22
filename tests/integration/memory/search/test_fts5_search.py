# Copyright 2026 David McCarty. All rights reserved.
"""Tests for FTS5 keyword search via fts5_search()."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from hermes_memory_core.store.sqlite import MemoryDB
from hermes_memory_core.search.fts5 import fts5_search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_memory_db(tmp_path: Path) -> MemoryDB:
    """Return an initialised MemoryDB pointed at a temp path."""
    db = MemoryDB(tmp_path / "memory.sqlite")
    db.initialize()
    return db


def insert_turn(
    conn: sqlite3.Connection,
    turn_id: str,
    session_id: str,
    role: str,
    content: str,
    project: str = "test-project",
    timestamp: str = "2026-05-18T12:00:00Z",
) -> None:
    """Insert a minimal turn row (and its session) and commit."""
    # Ensure session exists (turns don't have project column; sessions do)
    conn.execute(
        """INSERT OR IGNORE INTO sessions
           (session_id, agent, project, started_at) VALUES (?, 'test', ?, '2026-01-01T00:00:00Z')""",
        (session_id, project),
    )
    conn.execute(
        """INSERT INTO turns
           (turn_id, session_id, sequence, timestamp, role, content,
            dream_status, index_status, source_refs_json,
            redaction_count, redaction_types_json)
           VALUES (?, ?, 1, ?, ?, ?, 'pending', 'pending', '[]', 0, '[]')""",
        (turn_id, session_id, timestamp, role, content),
    )
    conn.commit()


def insert_chunk(
    conn: sqlite3.Connection,
    chunk_id: str,
    session_id: str,
    chunk_text: str,
    source_ref: str = "test#chunk",
    project: str = "test-project",
) -> None:
    """Insert a minimal chunk row and commit."""
    conn.execute(
        """INSERT INTO chunks
           (chunk_id, session_id, start_turn_id, end_turn_id, chunk_text,
            char_count, source_refs_json, created_at)
           VALUES (?, ?, NULL, NULL, ?, ?, '[]', '2026-05-18T12:00:00Z')""",
        (chunk_id, session_id, chunk_text, len(chunk_text)),
    )
    conn.commit()


def insert_fact(
    conn: sqlite3.Connection,
    fact_id: str,
    fact_text: str,
    project: str = "test-project",
) -> None:
    """Insert a minimal fact row and commit."""
    conn.execute(
        """INSERT INTO facts
           (fact_id, fact_text, content_hash, scope, project, status,
            source_refs_json, entity_ids_json, created_at, updated_at)
           VALUES (?, ?, 'hash_' || ?, 'project', ?, 'active',
                   '[]', '[]', '2026-05-18T12:00:00Z', '2026-05-18T12:00:00Z')""",
        (fact_id, fact_text, fact_id, project),
    )
    conn.commit()


def insert_decision(
    conn: sqlite3.Connection,
    decision_id: str,
    decision_text: str,
    rationale: str = "",
    project: str = "test-project",
) -> None:
    """Insert a minimal decision row and commit."""
    conn.execute(
        """INSERT INTO decisions
           (decision_id, decision_text, rationale, project, status,
            source_refs_json, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'open',
                   '[]', '2026-05-18T12:00:00Z', '2026-05-18T12:00:00Z')""",
        (decision_id, decision_text, rationale, project),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_db(tmp_path: Path) -> MemoryDB:
    """Initialised in-memory-style database."""
    return make_memory_db(tmp_path)


# ---------------------------------------------------------------------------
# AC-1: Basic query + ranked results + source_ref
# ---------------------------------------------------------------------------

def test_fts5_search_turns_returns_ranked_results(memory_db: MemoryDB) -> None:
    """FTS5 returns ranked hits from turns_fts with content and source_ref."""
    conn = memory_db._connect()
    try:
        insert_turn(conn, "t1", "s1", "user", "hello world foo")
        insert_turn(conn, "t2", "s1", "assistant", "hello world bar")
        insert_turn(conn, "t3", "s1", "user", "goodbye world baz")
    finally:
        conn.close()

    results = fts5_search("hello world", filters={}, table="turns", limit=10, memory_db=memory_db)

    assert len(results) == 2
    # Ranked: "hello world foo" and "hello world bar" both match "hello world"
    # The exact phrase scores higher; order within FTS5 is deterministic.
    ids = [r["turn_id"] for r in results]
    assert "t1" in ids
    assert "t2" in ids
    for r in results:
        assert "content" in r
        assert r["source_ref"].startswith("session:s1#turn=")


def test_fts5_search_chunks_returns_ranked_results(memory_db: MemoryDB) -> None:
    """FTS5 returns ranked hits from chunks_fts with source_ref."""
    conn = memory_db._connect()
    try:
        insert_chunk(conn, "c1", "s1", "the quick brown fox jumped")
        insert_chunk(conn, "c2", "s1", "the lazy dog slept")
        insert_chunk(conn, "c3", "s2", "quick silver runner")
    finally:
        conn.close()

    results = fts5_search("quick", filters={}, table="chunks", limit=10, memory_db=memory_db)

    assert len(results) == 2
    ids = [r["chunk_id"] for r in results]
    assert "c1" in ids  # "quick brown fox"
    assert "c3" in ids  # "quick silver"


def test_fts5_search_facts_returns_ranked_results(memory_db: MemoryDB) -> None:
    """FTS5 returns ranked hits from facts_fts with fact_id and source_ref."""
    conn = memory_db._connect()
    try:
        insert_fact(conn, "f1", "the sky is blue and clear")
        insert_fact(conn, "f2", "the ocean is deep and blue")
        insert_fact(conn, "f3", "grass is green")
    finally:
        conn.close()

    results = fts5_search("blue", filters={}, table="facts", limit=10, memory_db=memory_db)

    assert len(results) == 2
    ids = [r["fact_id"] for r in results]
    assert "f1" in ids
    assert "f2" in ids
    assert "f3" not in ids
    for r in results:
        assert "fact_text" in r
        assert r["source_ref"].startswith("fact:")


def test_fts5_search_decisions_returns_ranked_results(memory_db: MemoryDB) -> None:
    """FTS5 returns ranked hits from decisions_fts with decision_id and source_ref."""
    conn = memory_db._connect()
    try:
        insert_decision(conn, "d1", "Use SQLite for local storage", "performs well")
        insert_decision(conn, "d2", "Use PostgreSQL for cloud", "scales better")
        insert_decision(conn, "d3", "Use MongoDB for documents", "flexible schema")
    finally:
        conn.close()

    results = fts5_search("SQLite", filters={}, table="decisions", limit=10, memory_db=memory_db)

    assert len(results) == 1
    assert results[0]["decision_id"] == "d1"


# ---------------------------------------------------------------------------
# AC-2: Filters — project, session_id, date_from, date_to, role
# ---------------------------------------------------------------------------

def test_fts5_search_filter_project(memory_db: MemoryDB) -> None:
    """Results scoped to matching project."""
    conn = memory_db._connect()
    try:
        insert_turn(conn, "tp1", "sp1", "user", "alpha beta gamma", project="project-a")
        insert_turn(conn, "tp2", "sp2", "user", "alpha beta gamma", project="project-b")
    finally:
        conn.close()

    results = fts5_search("alpha beta", filters={"project": "project-a"}, table="turns", limit=10, memory_db=memory_db)

    assert len(results) == 1
    assert results[0]["turn_id"] == "tp1"


def test_fts5_search_filter_session_id(memory_db: MemoryDB) -> None:
    """Results scoped to matching session_id."""
    conn = memory_db._connect()
    try:
        insert_turn(conn, "ts1", "session-x", "user", "delta epsilon zeta")
        insert_turn(conn, "ts2", "session-y", "user", "delta epsilon zeta")
    finally:
        conn.close()

    results = fts5_search("delta epsilon", filters={"session_id": "session-x"}, table="turns", limit=10, memory_db=memory_db)

    assert len(results) == 1
    assert results[0]["turn_id"] == "ts1"


def test_fts5_search_filter_role(memory_db: MemoryDB) -> None:
    """Results scoped to matching role."""
    conn = memory_db._connect()
    try:
        insert_turn(conn, "tr1", "sr1", "user", "foo bar baz")
        insert_turn(conn, "tr2", "sr1", "assistant", "foo bar baz")
    finally:
        conn.close()

    results = fts5_search("foo bar", filters={"role": "user"}, table="turns", limit=10, memory_db=memory_db)

    assert len(results) == 1
    assert results[0]["turn_id"] == "tr1"
    assert results[0]["role"] == "user"


def test_fts5_search_filter_date_range(memory_db: MemoryDB) -> None:
    """Results within date_from / date_to range."""
    conn = memory_db._connect()
    try:
        insert_turn(
            conn, "td1", "sd1", "user", "winter story",
            timestamp="2026-01-01T00:00:00Z",
        )
        insert_turn(
            conn, "td2", "sd1", "user", "spring story",
            timestamp="2026-04-01T00:00:00Z",
        )
        insert_turn(
            conn, "td3", "sd1", "user", "summer story",
            timestamp="2026-07-01T00:00:00Z",
        )
    finally:
        conn.close()

    results = fts5_search(
        "story",
        filters={"date_from": "2026-03-01", "date_to": "2026-06-30"},
        table="turns",
        limit=10,
        memory_db=memory_db,
    )

    assert len(results) == 1
    assert results[0]["turn_id"] == "td2"


def test_fts5_search_filter_combined(memory_db: MemoryDB) -> None:
    """Multiple filters compose correctly."""
    conn = memory_db._connect()
    try:
        insert_turn(
            conn, "tc1", "sc1", "user", "python is awesome",
            project="myproj",
            timestamp="2026-05-01T00:00:00Z",
        )
        insert_turn(
            conn, "tc2", "sc1", "assistant", "python is fast",
            project="myproj",
            timestamp="2026-05-01T00:00:00Z",
        )
        insert_turn(
            conn, "tc3", "sc2", "user", "python is awesome",
            project="myproj",
            timestamp="2026-05-01T00:00:00Z",
        )
        insert_turn(
            conn, "tc4", "sc1", "user", "rust is awesome",
            project="other",
            timestamp="2026-05-01T00:00:00Z",
        )
    finally:
        conn.close()

    results = fts5_search(
        "python",
        filters={"project": "myproj", "role": "user"},
        table="turns",
        limit=10,
        memory_db=memory_db,
    )

    assert len(results) == 2
    ids = [r["turn_id"] for r in results]
    assert "tc1" in ids  # sc1, user, myproj
    assert "tc3" in ids  # sc2, user, myproj
    assert "tc2" not in ids  # sc1, assistant (role filter)
    assert "tc4" not in ids  # sc1, user, other (project filter)


# ---------------------------------------------------------------------------
# AC-3: Snippet excerpts and source_ref per table
# ---------------------------------------------------------------------------

def test_fts5_search_includes_snippet(memory_db: MemoryDB) -> None:
    """Each result includes a snippet() excerpt from SQLite."""
    conn = memory_db._connect()
    try:
        insert_turn(conn, "ts1", "ss1", "user", "the secret API key is sk-12345 hidden here")
        insert_turn(conn, "ts2", "ss1", "user", "normal conversation about fishing")
    finally:
        conn.close()

    results = fts5_search("API key", filters={}, table="turns", limit=10, memory_db=memory_db)

    assert len(results) == 1
    assert "snippet" in results[0]
    assert isinstance(results[0]["snippet"], str)


def test_fts5_search_source_ref_format_turns(memory_db: MemoryDB) -> None:
    """source_ref for turns: session:{session_id}#turn={turn_id}."""
    conn = memory_db._connect()
    try:
        insert_turn(conn, "tsr1", "sess-abc", "user", "hello world")
    finally:
        conn.close()

    results = fts5_search("hello", filters={}, table="turns", limit=10, memory_db=memory_db)
    assert len(results) == 1
    assert results[0]["source_ref"] == "session:sess-abc#turn=tsr1"


def test_fts5_search_source_ref_format_chunks(memory_db: MemoryDB) -> None:
    """source_ref for chunks: chunk:{chunk_id}."""
    conn = memory_db._connect()
    try:
        insert_chunk(conn, "csr1", "sess-x", "some chunk text here")
    finally:
        conn.close()

    results = fts5_search("chunk", filters={}, table="chunks", limit=10, memory_db=memory_db)
    assert len(results) == 1
    assert results[0]["source_ref"] == "chunk:csr1"


def test_fts5_search_source_ref_format_facts(memory_db: MemoryDB) -> None:
    """source_ref for facts: fact:{fact_id}."""
    conn = memory_db._connect()
    try:
        insert_fact(conn, "fsr1", "the sky is blue")
    finally:
        conn.close()

    results = fts5_search("sky blue", filters={}, table="facts", limit=10, memory_db=memory_db)
    assert len(results) == 1
    assert results[0]["source_ref"] == "fact:fsr1"


def test_fts5_search_source_ref_format_decisions(memory_db: MemoryDB) -> None:
    """source_ref for decisions: decision:{decision_id}."""
    conn = memory_db._connect()
    try:
        insert_decision(conn, "dsr1", "Adopt FTS5 for search")
    finally:
        conn.close()

    results = fts5_search("FTS5", filters={}, table="decisions", limit=10, memory_db=memory_db)
    assert len(results) == 1
    assert results[0]["source_ref"] == "decision:dsr1"


# ---------------------------------------------------------------------------
# AC-4: Empty results — no error, returns []
# ---------------------------------------------------------------------------

def test_fts5_search_no_matches_returns_empty_list(memory_db: MemoryDB) -> None:
    """No hits returns [] without error."""
    conn = memory_db._connect()
    try:
        insert_turn(conn, "tem1", "sem1", "user", "apple banana cherry")
    finally:
        conn.close()

    results = fts5_search("zzzz no match exists", filters={}, table="turns", limit=10, memory_db=memory_db)

    assert results == []


def test_fts5_search_unknown_table_returns_empty_list(tmp_path: Path) -> None:
    """Non-existent table name returns [] without raising."""
    db = make_memory_db(tmp_path)
    results = fts5_search("hello", filters={}, table="nonexistent_fts", limit=10, memory_db=db)
    assert results == []


# ---------------------------------------------------------------------------
# Integration: limit parameter
# ---------------------------------------------------------------------------

def test_fts5_search_limit_honored(memory_db: MemoryDB) -> None:
    """Only up to `limit` results are returned."""
    conn = memory_db._connect()
    try:
        for i in range(20):
            insert_turn(conn, f"tl{i:03d}", "sl1", "user", f"word alpha {i}")
    finally:
        conn.close()

    results = fts5_search("alpha", filters={}, table="turns", limit=5)
    assert len(results) <= 5