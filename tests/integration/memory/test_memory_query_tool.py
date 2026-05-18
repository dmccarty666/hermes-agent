# Copyright 2026 David McCarty. All rights reserved.
"""Tests for memory_query tool — T-012."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def plugin_dir() -> Path:
    return Path(__file__).parents[3] / "plugins" / "memory" / "hermes-local"


def load_provider():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "hermes_local_plugin",
        str(plugin_dir() / "__init__.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HermesLocalProvider


# ------------------------------------------------------------------
# Test: get_tool_schemas() returns exactly ONE schema named memory_query
# ------------------------------------------------------------------

def test_get_tool_schemas_returns_memory_query_only(tmp_path: Path, monkeypatch) -> None:
    """AC-1: With hermes-local config, get_tool_schemas() returns exactly ONE schema."""
    fake_home = tmp_path / "hermes_home"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("memory:\n  provider: hermes-local\n")
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    HermesLocalProvider = load_provider()
    provider = HermesLocalProvider()
    schemas = provider.get_tool_schemas()

    assert len(schemas) == 3, f"Expected 3 schemas (memory_query + memory_get_source + memory_recent_context), got {len(schemas)}: {schemas}"
    names = {s["name"] for s in schemas}
    assert "memory_query" in names, f"memory_query schema missing, got: {names}"
    assert "memory_get_source" in names, f"memory_get_source schema missing, got: {names}"
    assert "memory_recent_context" in names, f"memory_recent_context schema missing, got: {names}"


def test_memory_query_schema_has_modes_keyword_sessions_recent(tmp_path: Path, monkeypatch) -> None:
    """AC-1: memory_query schema exposes modes keyword, sessions, recent."""
    fake_home = tmp_path / "hermes_home"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("memory:\n  provider: hermes-local\n")
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    HermesLocalProvider = load_provider()
    provider = HermesLocalProvider()
    schemas = provider.get_tool_schemas()
    schema = schemas[0]

    # mode parameter must have enum with keyword, sessions, recent
    mode_param = schema["parameters"]["properties"]["mode"]
    assert "enum" in mode_param, f"mode param has no enum: {mode_param}"
    assert set(mode_param["enum"]) == {"keyword", "sessions", "recent"}, \
        f"mode enum must be {{keyword, sessions, recent}}, got {mode_param['enum']}"


def test_memory_query_schema_has_required_fields(tmp_path: Path, monkeypatch) -> None:
    """Schema includes query, mode, project, filters, limit properties."""
    fake_home = tmp_path / "hermes_home"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("memory:\n  provider: hermes-local\n")
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    HermesLocalProvider = load_provider()
    provider = HermesLocalProvider()
    schema = provider.get_tool_schemas()[0]
    props = schema["parameters"]["properties"]

    assert "query" in props
    assert "mode" in props
    assert "filters" in props
    assert "limit" in props


# ------------------------------------------------------------------
# Test: memory_query(mode='keyword') calls fts5_search and returns normalized results
# ------------------------------------------------------------------

def test_memory_query_keyword_mode_calls_fts5_search(tmp_path: Path, monkeypatch) -> None:
    """AC-2: memory_query(query='agents.list', mode='keyword') returns normalized results."""
    fake_home = tmp_path / "hermes_home"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("memory:\n  provider: hermes-local\n")
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    # Set up a temp SQLite with known data
    from hermes_memory_core.store.sqlite import MemoryDB
    memory_db = MemoryDB(tmp_path / "memory.sqlite")
    memory_db.initialize()

    # Insert a turn with searchable content
    conn = memory_db._connect()
    conn.execute(
        """INSERT OR IGNORE INTO sessions
           (session_id, agent, project, started_at)
           VALUES ('sess_test', 'test', 'test-project', '2026-01-01T00:00:00Z')"""
    )
    conn.execute(
        """INSERT INTO turns
           (turn_id, session_id, sequence, timestamp, role, content,
            dream_status, index_status, source_refs_json,
            redaction_count, redaction_types_json)
           VALUES ('turn_001', 'sess_test', 1, '2026-05-18T12:00:00Z',
                   'assistant', 'Use agents.list() to enumerate all running agents.',
                   'pending', 'pending', '[]', 0, '[]')"""
    )
    conn.commit()
    conn.close()

    HermesLocalProvider = load_provider()
    provider = HermesLocalProvider()

    result_str = provider.handle_tool_call(
        "memory_query",
        {"query": "agents.list", "mode": "keyword"},
        memory_db=memory_db,
    )
    result = json.loads(result_str)

    # Result shape per TDD §5.2 / PRD §10
    assert "results" in result, f"Missing 'results' key: {result}"
    assert "query" in result, f"Missing 'query' key: {result}"
    assert "mode" in result, f"Missing 'mode' key: {result}"
    assert "backend_hints" in result, f"Missing 'backend_hints' key: {result}"
    assert result["mode"] == "keyword"
    assert result["backend_hints"] == ["fts5"]

    # Each result entry has required fields
    for r in result["results"]:
        assert "content" in r, f"Result missing 'content': {r}"
        assert "source_ref" in r, f"Result missing 'source_ref': {r}"
        assert "excerpt" in r, f"Result missing 'excerpt': {r}"
        assert "score" in r, f"Result missing 'score': {r}"
        assert "mode" in r, f"Result missing 'mode': {r}"


# ------------------------------------------------------------------
# Test: memory_query(mode='keyword', query='X', project='hermes') filters by project
# ------------------------------------------------------------------

def test_memory_query_keyword_with_project_filter(tmp_path: Path, monkeypatch) -> None:
    """AC-3: memory_query with project filter scopes FTS5 results to that project."""
    fake_home = tmp_path / "hermes_home"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("memory:\n  provider: hermes-local\n")
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    from hermes_memory_core.store.sqlite import MemoryDB
    memory_db = MemoryDB(tmp_path / "memory.sqlite")
    memory_db.initialize()

    conn = memory_db._connect()
    # Two sessions with different projects
    conn.execute(
        """INSERT OR IGNORE INTO sessions
           (session_id, agent, project, started_at)
           VALUES ('sess_hermes', 'test', 'hermes', '2026-01-01T00:00:00Z')"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO sessions
           (session_id, agent, project, started_at)
           VALUES ('sess_other', 'test', 'other-project', '2026-01-01T00:00:00Z')"""
    )
    conn.execute(
        """INSERT INTO turns
           (turn_id, session_id, sequence, timestamp, role, content,
            dream_status, index_status, source_refs_json,
            redaction_count, redaction_types_json)
           VALUES ('turn_hermes', 'sess_hermes', 1, '2026-05-18T12:00:00Z',
                   'assistant', 'Python agents work well in hermes.',
                   'pending', 'pending', '[]', 0, '[]')"""
    )
    conn.execute(
        """INSERT INTO turns
           (turn_id, session_id, sequence, timestamp, role, content,
            dream_status, index_status, source_refs_json,
            redaction_count, redaction_types_json)
           VALUES ('turn_other', 'sess_other', 1, '2026-05-18T12:00:00Z',
                   'assistant', 'Python agents work well in hermes.',
                   'pending', 'pending', '[]', 0, '[]')"""
    )
    conn.commit()
    conn.close()

    HermesLocalProvider = load_provider()
    provider = HermesLocalProvider()

    # Scope to 'hermes' project only
    result_str = provider.handle_tool_call(
        "memory_query",
        {"query": "agents", "mode": "keyword", "project": "hermes"},
        memory_db=memory_db,
    )
    result = json.loads(result_str)

    assert len(result["results"]) == 1, f"Expected 1 result for project=hermes, got {len(result['results'])}"
    assert "sess_hermes" in result["results"][0]["source_ref"], \
        f"Expected source from sess_hermes, got: {result['results'][0]['source_ref']}"


# ------------------------------------------------------------------
# Test: unimplemented modes raise NotImplementedError
# ------------------------------------------------------------------

def test_memory_query_semantic_calls_semantic_search_and_returns_results(tmp_path: Path, monkeypatch) -> None:
    """AC-4: memory_query(mode='semantic') calls semantic_search and returns normalized results."""
    fake_home = tmp_path / "hermes_home"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("memory:\n  provider: hermes-local\n")
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    HermesLocalProvider = load_provider()
    provider = HermesLocalProvider()

    # Mock semantic_search to avoid needing real LMS/Qdrant
    fake_results = [
        {
            "content": "test chunk text",
            "source_ref": "session:sess_001#chunk=chunk_abc",
            "score": 0.95,
            "metadata": {"chunk_id": "chunk_abc", "session_id": "sess_001"},
        }
    ]
    from unittest.mock import patch
    with patch("hermes_memory_core.tools.semantic_search", return_value=fake_results) as mock_ss:
        result_str = provider.handle_tool_call(
            "memory_query",
            {"query": "test semantic query", "mode": "semantic", "limit": 5},
        )
        result = json.loads(result_str)

        mock_ss.assert_called_once()
        call_kwargs = mock_ss.call_args.kwargs
        assert call_kwargs["query"] == "test semantic query"
        assert call_kwargs["limit"] == 5

        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == "test chunk text"
        assert result["mode"] == "semantic"
        assert result["backend_hints"] == ["qdrant"]

    # semantic mode no longer raises NotImplementedError — T-019 implemented


def test_memory_query_hybrid_raises_not_implemented(tmp_path: Path, monkeypatch) -> None:
    """AC-4: memory_query(mode='hybrid') raises NotImplementedError."""
    fake_home = tmp_path / "hermes_home"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("memory:\n  provider: hermes-local\n")
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    HermesLocalProvider = load_provider()
    provider = HermesLocalProvider()

    with pytest.raises(NotImplementedError) as exc_info:
        provider.handle_tool_call(
            "memory_query",
            {"query": "test", "mode": "hybrid"},
        )
    assert "not yet implemented" in str(exc_info.value)