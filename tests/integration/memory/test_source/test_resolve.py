# Copyright 2026 David McCarty. All rights reserved.
"""Tests for source resolver — T-013.

Covers:
  AC-1: session:{id}#turn={n} -> raw turn content (role, content, timestamp, project)
  AC-2: missing session/turn -> {kind:'missing', source_ref, reason}
  AC-3: fact:{id}, decision:{id}, chunk:{id} -> corresponding row
  AC-4: expand=true -> also resolves nested source_refs in tool_call JSON blobs
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


# ------------------------------------------------------------------
# Test helpers
# ------------------------------------------------------------------

def memory_db_from_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal memory SQLite using MemoryDB's own schema + return (db_path, memory_home)."""
    memory_home = tmp_path / "hermes_memory"
    memory_home.mkdir(parents=True, exist_ok=True)
    index_dir = memory_home / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "memory.sqlite"

    # Use MemoryDB to build the actual schema
    from hermes_memory_core.store.sqlite import MemoryDB
    db = MemoryDB(db_path)
    db.initialize()
    return db_path, memory_home


def insert_turn(db_path: Path, session_id: str, turn_id: str, sequence: int,
                role: str, content: str, project: str = "test-project") -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, sequence, timestamp, role, content) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (turn_id, session_id, sequence, "2026-05-18T12:00:00Z", role, content),
    )
    conn.commit()
    conn.close()


def insert_session(db_path: Path, session_id: str, project: str = "test-project") -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO sessions (session_id, agent, project, started_at) VALUES (?, ?, ?, ?)",
        (session_id, "test-agent", project, "2026-05-18T11:00:00Z"),
    )
    conn.commit()
    conn.close()


def insert_chunk(db_path: Path, chunk_id: str, session_id: str, chunk_text: str) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO chunks (chunk_id, session_id, chunk_text, char_count, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (chunk_id, session_id, chunk_text, len(chunk_text), "2026-05-18T12:00:00Z"),
    )
    conn.commit()
    conn.close()


def insert_fact(db_path: Path, fact_id: str, fact_text: str, project: str = "test-project") -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO facts (fact_id, fact_text, scope, project, content_hash, source_refs_json, "
        "entity_ids_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (fact_id, fact_text, "project", project, "hash123", "[]", "[]",
         "2026-05-18T12:00:00Z", "2026-05-18T12:00:00Z"),
    )
    conn.commit()
    conn.close()


def insert_decision(db_path: Path, decision_id: str, decision_text: str, project: str = "test-project") -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO decisions (decision_id, decision_text, project, source_refs_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (decision_id, decision_text, project, "[]", "2026-05-18T12:00:00Z", "2026-05-18T12:00:00Z"),
    )
    conn.commit()
    conn.close()


# ------------------------------------------------------------------
# AC-1: session:{id}#turn={n} resolves to raw turn content
# ------------------------------------------------------------------

def test_resolve_session_turn_returns_role_content_timestamp_project(tmp_path, monkeypatch):
    """AC-1: resolve(session:id#turn=n) returns role, content, timestamp, project."""
    from hermes_memory_core.store.sqlite import MemoryDB

    db_path, memory_home = memory_db_from_paths(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(memory_home))
    monkeypatch.chdir(tmp_path)

    insert_session(db_path, "abc")
    insert_turn(db_path, "abc", "t1", sequence=1, role="user", content="hello world")

    db = MemoryDB(db_path)
    from hermes_memory_core.source import resolve
    result = resolve("session:abc#turn=t1", memory_db=db)

    assert result["kind"] == "session:turn"
    assert result["role"] == "user"
    assert result["content"] == "hello world"
    assert result["timestamp"] == "2026-05-18T12:00:00Z"
    assert result["project"] == "test-project"
    assert result["source_ref"] == "session:abc#turn=t1"


# ------------------------------------------------------------------
# AC-2: missing session/turn returns {kind:'missing', ...}
# ------------------------------------------------------------------

def test_resolve_missing_session_returns_kind_missing(tmp_path, monkeypatch):
    """AC-2: missing session returns kind=missing, reason='session archived'."""
    from hermes_memory_core.store.sqlite import MemoryDB

    db_path, memory_home = memory_db_from_paths(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(memory_home))
    monkeypatch.chdir(tmp_path)

    db = MemoryDB(db_path)
    from hermes_memory_core.source import resolve
    result = resolve("session:nonexistent#turn=t1", memory_db=db)

    assert result["kind"] == "missing"
    assert result["source_ref"] == "session:nonexistent#turn=t1"
    assert result["reason"] == "session archived"


def test_resolve_missing_turn_returns_kind_missing(tmp_path, monkeypatch):
    """AC-2: session exists but turn doesn't -> kind=missing."""
    from hermes_memory_core.store.sqlite import MemoryDB

    db_path, memory_home = memory_db_from_paths(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(memory_home))
    monkeypatch.chdir(tmp_path)

    insert_session(db_path, "abc")

    db = MemoryDB(db_path)
    from hermes_memory_core.source import resolve
    result = resolve("session:abc#turn=nonexistent", memory_db=db)

    assert result["kind"] == "missing"
    assert "turn" in result["reason"].lower()


# ------------------------------------------------------------------
# AC-3: fact:, decision:, chunk: refs resolve to corresponding rows
# ------------------------------------------------------------------

def test_resolve_fact_returns_fact_text_and_metadata(tmp_path, monkeypatch):
    """AC-3: resolve(fact:id) returns fact_text and metadata."""
    from hermes_memory_core.store.sqlite import MemoryDB

    db_path, memory_home = memory_db_from_paths(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(memory_home))
    monkeypatch.chdir(tmp_path)

    insert_fact(db_path, "f_abc123", "David prefers concise responses")

    db = MemoryDB(db_path)
    from hermes_memory_core.source import resolve
    result = resolve("fact:f_abc123", memory_db=db)

    assert result["kind"] == "fact"
    assert result["fact_text"] == "David prefers concise responses"
    assert result["fact_id"] == "f_abc123"


def test_resolve_decision_returns_decision_text_and_project(tmp_path, monkeypatch):
    """AC-3: resolve(decision:id) returns decision_text and project."""
    from hermes_memory_core.store.sqlite import MemoryDB

    db_path, memory_home = memory_db_from_paths(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(memory_home))
    monkeypatch.chdir(tmp_path)

    insert_decision(db_path, "d_xyz789", "Use Qwen3.6-35B as default for delegated tasks")

    db = MemoryDB(db_path)
    from hermes_memory_core.source import resolve
    result = resolve("decision:d_xyz789", memory_db=db)

    assert result["kind"] == "decision"
    assert result["decision_text"] == "Use Qwen3.6-35B as default for delegated tasks"


def test_resolve_chunk_returns_chunk_text(tmp_path, monkeypatch):
    """AC-3: resolve(chunk:id) returns chunk_text."""
    from hermes_memory_core.store.sqlite import MemoryDB

    db_path, memory_home = memory_db_from_paths(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(memory_home))
    monkeypatch.chdir(tmp_path)

    insert_chunk(db_path, "ck_001", "sess_abc", "This is a chunk about Python decorators")

    db = MemoryDB(db_path)
    from hermes_memory_core.source import resolve
    result = resolve("chunk:ck_001", memory_db=db)

    assert result["kind"] == "chunk"
    assert result["chunk_text"] == "This is a chunk about Python decorators"


# ------------------------------------------------------------------
# AC-4: expand=true resolves nested source_refs in tool_call JSON
# ------------------------------------------------------------------

def test_resolve_expand_true_inlines_tool_call_source_refs(tmp_path, monkeypatch):
    """AC-4: expand=true resolves inline source_refs found in source_refs_json."""
    from hermes_memory_core.store.sqlite import MemoryDB

    db_path, memory_home = memory_db_from_paths(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(memory_home))
    monkeypatch.chdir(tmp_path)

    insert_session(db_path, "abc")
    # Turn with source_refs_json pointing to a prior turn
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, sequence, timestamp, role, content, source_refs_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("t2", "abc", 2, "2026-05-18T12:01:00Z", "assistant",
         "Tool result from memory query",
         json.dumps(["session:abc#turn=t1"])),
    )
    conn.commit()
    conn.close()
    # Prior turn
    insert_turn(db_path, "abc", "t1", sequence=1, role="user", content="hello world")

    db = MemoryDB(db_path)
    from hermes_memory_core.source import resolve
    result = resolve("session:abc#turn=t2", memory_db=db, expand=True)

    assert result["kind"] == "session:turn"
    assert "expanded_refs" in result
    assert len(result["expanded_refs"]) == 1
    assert result["expanded_refs"][0]["source_ref"] == "session:abc#turn=t1"
    assert result["expanded_refs"][0]["content"] == "hello world"
    assert result["expanded_refs"][0]["role"] == "user"


def test_resolve_expand_false_no_expansion(tmp_path, monkeypatch):
    """AC-4: expand=False (default) does not expand nested refs."""
    from hermes_memory_core.store.sqlite import MemoryDB

    db_path, memory_home = memory_db_from_paths(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(memory_home))
    monkeypatch.chdir(tmp_path)

    insert_session(db_path, "abc")
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO turns (turn_id, session_id, sequence, timestamp, role, content, source_refs_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("t2", "abc", 2, "2026-05-18T12:01:00Z", "assistant",
         "Result", json.dumps(["session:abc#turn=t1"])),
    )
    conn.close()

    db = MemoryDB(db_path)
    from hermes_memory_core.source import resolve
    result = resolve("session:abc#turn=t2", memory_db=db, expand=False)

    assert "expanded_refs" not in result