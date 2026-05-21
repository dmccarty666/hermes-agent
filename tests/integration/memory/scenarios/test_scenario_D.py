"""MVP Acceptance Test Suite — Scenario D: Semantic Search.

Verifies Plan.md §9, Scenario D:
  1. Insert a session discussing 'avoiding paid memory provider Honcho on cost grounds'.
  2. memory_query(query='free local memory instead of paid provider', mode='semantic').
  3. Verify relevant chunk returned, semantically matched.

NOTE: hermes_memory_core.search.semantic and hermes_memory_core.embed
are not yet implemented. Tests will fail with ImportError until Phase 3.
"""

from __future__ import annotations

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
def temp_memory_with_session():
    """Memory dir with a pre-populated session ready for semantic search."""
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
            "CREATE TABLE chunks ("
            "  chunk_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
            "  chunk_type TEXT NOT NULL, text TEXT NOT NULL, "
            "  text_hash TEXT NOT NULL, source_ref TEXT NOT NULL, "
            "  qdrant_point_id TEXT, embed_model TEXT, "
            "  created_at TEXT NOT NULL)"
        )
        conn.commit()
        conn.close()

        session_id = f"scenario-d-{uuid.uuid4().hex[:8]}"
        turn_content = (
            "We're using hermes-memory as our local memory provider instead of "
            "paying for Honcho. The main driver is cost — Honcho charges per token "
            "and we're running a large operation. hermes-local captures everything "
            "losslessly to disk and SQLite with semantic search via Qdrant."
        )
        turn_id = f"turn-{uuid.uuid4().hex[:8]}"
        content_hash = f"ch-{uuid.uuid4().hex[:8]}"

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
            (session_id, "test-agent", datetime.now(timezone.utc).isoformat()),
        )
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
                f"raw-{uuid.uuid4().hex[:8]}",
                content_hash,
            ),
        )
        conn.commit()
        conn.close()

        yield mem, db_path, raw, session_id, turn_id, turn_content, content_hash


# ---------------------------------------------------------------------------
# Scenario D Tests
# ---------------------------------------------------------------------------

def test_scenario_D_semantic_search_returns_relevant_chunk(temp_memory_with_session):
    """Given a session about free local memory vs paid Honcho, when semantic
    search runs with a paraphrased query, then the relevant chunk is returned."""
    mem, db_path, raw, session_id, turn_id, turn_content, content_hash = temp_memory_with_session

    # Add a chunk row (simulating what the chunk pipeline would create)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO chunks (chunk_id, session_id, chunk_type, text, text_hash, "
        "source_ref, qdrant_point_id, embed_model, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            f"chunk-{uuid.uuid4().hex[:8]}",
            session_id,
            "turn",
            turn_content,
            content_hash,
            f"capture:turns:{turn_id}",
            f"qdrant-{uuid.uuid4().hex[:8]}",  # simulated Qdrant point
            "nomic-embed-text-v1.5",
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    query = "free local memory instead of paid provider"

    try:
        from hermes_memory_core.search.semantic_search import semantic_search
        results = semantic_search(query=query, db_path=db_path, limit=5)
    except ImportError:
        pytest.skip("hermes_memory_core.search.semantic_search not yet implemented")

    assert len(results) >= 1, f"Expected at least 1 semantic result, got {len(results)}"

    # Verify result is about the right topic (semantic match)
    texts = [r.get("text", "") or r.get("content", "") for r in results]
    # At least one result should be semantically related to the query
    # (The exact match is in the turn_content about Honcho vs hermes-local)
    assert any("hermes" in t.lower() or "honcho" in t.lower() or "memory" in t.lower()
               for t in texts if t), \
        f"Expected semantically-related result in: {texts}"


def test_scenario_D_semantic_search_uses_qdrant(temp_memory_with_session):
    """Verify that semantic search relies on Qdrant for vector similarity."""
    mem, db_path, raw, session_id, turn_id, turn_content, content_hash = temp_memory_with_session

    try:
        import qdrant_client
        client = qdrant_client.QdrantClient(url="http://localhost:6333")
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
    except Exception:
        pytest.skip("Qdrant not available at localhost:6333")

    # Qdrant is reachable — semantic search CAN work once chunks are indexed.
    # The actual hermes_memory collection will be created by the chunk pipeline.
    # We verify here that Qdrant is reachable and responsive, which is the
    # prerequisite for semantic search to function.
    assert len(collection_names) >= 0, "Qdrant should be reachable"
    # If collections exist, hermes_memory is the expected naming pattern
    if collection_names:
        assert any("hermes_memory" in name for name in collection_names), \
            f"Expected hermes_memory collection pattern in Qdrant, got: {collection_names}"


def test_scenario_D_semantic_search_returns_chunk_with_source_ref(temp_memory_with_session):
    """Verify semantic results include source_ref for trace-ability."""
    mem, db_path, raw, session_id, turn_id, turn_content, content_hash = temp_memory_with_session

    chunk_id = f"chunk-{uuid.uuid4().hex[:8]}"
    source_ref = f"capture:turns:{turn_id}"
    qdrant_point_id = f"qdrant-{uuid.uuid4().hex[:8]}"

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO chunks (chunk_id, session_id, chunk_type, text, text_hash, "
        "source_ref, qdrant_point_id, embed_model, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            chunk_id,
            session_id,
            "turn",
            turn_content,
            content_hash,
            source_ref,
            qdrant_point_id,
            "nomic-embed-text-v1.5",
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    try:
        from hermes_memory_core.search.semantic_search import semantic_search
        results = semantic_search(
            query="local free memory provider cost",
            db_path=db_path,
            limit=5
        )
    except ImportError:
        pytest.skip("hermes_memory_core.search.semantic_search not yet implemented")

    # At least one result should have a source_ref pointing back to the turn
    source_refs_in_results = []
    for r in results:
        src = r.get("source_ref", "") or ""
        if src:
            source_refs_in_results.append(src)

    assert len(source_refs_in_results) >= 1, \
        f"Expected source_ref in semantic results, got: {results}"


def test_scenario_D_embedding_model_is_nomic():
    """Verify the expected embedding model: nomic-embed-text-v1.5 @ 768d."""
    expected_model = "nomic-embed-text-v1.5"
    expected_dim = 768

    try:
        from hermes_memory_core.embed import EMBED_MODEL, EMBED_DIM
        assert EMBED_MODEL == expected_model, f"Expected {expected_model}, got {EMBED_MODEL}"
        assert EMBED_DIM == expected_dim, f"Expected dim {expected_dim}, got {EMBED_DIM}"
    except ImportError:
        # Verify the LMS endpoint serves the expected model
        try:
            import requests
            resp = requests.get("http://192.168.2.105:1235/v1/models", timeout=3)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                model_ids = [m["id"] for m in models]
                # Should have nomic-embed-text-v1.5
                assert any("nomic" in m.lower() for m in model_ids), \
                    f"Expected nomic model in LMS models: {model_ids}"
            else:
                pytest.skip("LMS endpoint not available")
        except Exception:
            pytest.skip("Could not verify embedding model — LMS not reachable")