"""Tests for hermes_memory_core/indexer.py — T-018 (Async indexer)

These tests cover the IndexerWorker class and batch_index function:
- AC-1: 5 pending turns → chunked → embedded → Qdrant upserted → index_status='indexed'
- AC-2: Restart → no duplicate Qdrant points (idempotent by chunk_id)
- AC-3: Catch-up on init — processes all pending turns before polling
- AC-4: Embedding failure → retry → index_status='failed' after 3 retries
- AC-5: Polling interval ≤30s for new pending turns

Uses real Qdrant (:6333) and real LMS (:1235) when available.
Test collection suffix _test ensures clean teardown.
"""

# ── PHASE-1.5 TRIAGE — NEEDS ISOLATED QDRANT ───────────────────────────────────
# Triaged Bucket A by the recovery pass on branch recovery/phase-1-5-restore.
# Tests want their own isolated Qdrant collection and ECONNREFUSE when nothing
# is listening on :6333. The live system's :6333 is NOT to be repurposed —
# see docs/INTEGRATION-TEST-TRIAGE.md for how to spin up a transient test
# instance on a non-conflicting port and unskip.
import os as _phase15_os
import socket as _phase15_socket
import pytest as _phase15_pytest

def _phase15_qdrant_reachable() -> bool:
    host = _phase15_os.environ.get("QDRANT_HOST", "localhost")
    port = int(_phase15_os.environ.get("QDRANT_PORT", "6333"))
    try:
        with _phase15_socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False

if not _phase15_qdrant_reachable():
    _phase15_pytest.skip(
        "needs isolated Qdrant; see docs/INTEGRATION-TEST-TRIAGE.md",
        allow_module_level=True,
    )


import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pytest

# ---------------------------------------------------------------------------
# Qdrant test collection name
# ---------------------------------------------------------------------------
_CHUNKS_COLLECTION = "hermes_memory_chunks_nomic_v15_test_indexer"

# ---------------------------------------------------------------------------
# LMS availability check (session-scoped)
# ---------------------------------------------------------------------------
def _lms_available() -> bool:
    """Return True if LMS at 192.168.2.105:1235 is reachable."""
    try:
        import requests
        resp = requests.get(
            "http://192.168.2.105:1235/v1/models",
            timeout=3,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Qdrant fixture — real client, test collection, teardown on exit
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qdrant_client():
    """Real QdrantClient against localhost:6333, test collection, auto-teardown."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient("http://localhost:6333", timeout=10)
    collection_name = _CHUNKS_COLLECTION

    # Recreate clean collection
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    client.create_collection(
        collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    yield client, collection_name

    # Teardown
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# LMS fixture — skip if unavailable
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lms_available():
    return _lms_available()


# ---------------------------------------------------------------------------
# MemoryDB fixture — fresh in-process SQLite
# ---------------------------------------------------------------------------
@pytest.fixture
def memory_db(tmp_path) -> Generator:
    """Fresh MemoryDB backed by a temporary SQLite file."""
    from hermes_memory_core.store.sqlite import MemoryDB

    db_file = tmp_path / "test_memory.sqlite"
    db = MemoryDB(str(db_file))
    db.initialize()
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Helper: insert 5 pending turns for a session
# ---------------------------------------------------------------------------
def insert_5_pending_turns(db, session_id: str) -> list:
    """Insert 5 turns with index_status='pending' and return their turn_ids."""
    conn = db._connect()
    try:
        ts = datetime.now(timezone.utc).isoformat()
        turn_ids = []
        for seq in range(1, 6):
            turn_id = f"turn_{uuid.uuid4().hex[:12]}"
            turn_ids.append(turn_id)
            conn.execute(
                """INSERT INTO turns
                   (turn_id, session_id, sequence, timestamp, role, content,
                    dream_status, index_status, source_refs_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    seq,
                    ts,
                    "user" if seq % 2 == 1 else "assistant",
                    f"Test content for turn {seq} — testing async indexer",
                    "pending",
                    "pending",
                    "[]",
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return turn_ids


# ---------------------------------------------------------------------------
# AC-1: 5 pending turns → chunked → embedded → Qdrant upserted → indexed
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _lms_available(), reason="LMS at 192.168.2.105:1235 not reachable")
def test_indexer_5_turns_indexed(memory_db, qdrant_client, lms_available):
    """AC-1: Given 5 pending turns, when the indexer runs, all 5 are indexed."""
    client, collection_name = qdrant_client
    session_id = f"sess_{uuid.uuid4().hex[:12]}"

    # Insert 5 pending turns
    turn_ids = insert_5_pending_turns(memory_db, session_id)
    assert len(turn_ids) == 5

    # Run the indexer
    from hermes_memory_core.indexer import batch_index

    result = batch_index(
        memory_db,
        batch_size=20,
        collection_name=collection_name,
    )
    assert result.processed >= 5, f"Expected ≥5, got {result.processed}"

    # Verify: all 5 turns now have index_status='indexed'
    conn = memory_db._connect()
    try:
        rows = conn.execute(
            "SELECT turn_id, index_status FROM turns WHERE session_id=? ORDER BY sequence",
            (session_id,),
        ).fetchall()
        assert len(rows) == 5
        for turn_id, status in rows:
            assert status == "indexed", f"{turn_id}: expected 'indexed', got '{status}'"
    finally:
        conn.close()

    # Verify: Qdrant has points for the chunks
    time.sleep(0.5)
    count_result = client.count(collection_name, exact=True)
    assert count_result.count >= 5, f"Expected ≥5 Qdrant points, got {count_result.count}"


# ---------------------------------------------------------------------------
# AC-2: Restart → no duplicate Qdrant points (idempotent by chunk_id)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _lms_available(), reason="LMS at 192.168.2.105:1235 not reachable")
def test_indexer_idempotent_no_duplicates(memory_db, qdrant_client, lms_available):
    """AC-2: Given indexer processes a session and agent restarts, no duplicate points."""
    client, collection_name = qdrant_client
    session_id = f"sess_{uuid.uuid4().hex[:12]}"

    from hermes_memory_core.indexer import batch_index

    # First run
    insert_5_pending_turns(memory_db, session_id)
    result1 = batch_index(memory_db, batch_size=20, collection_name=collection_name)
    assert result1.processed >= 5

    # Wait for Qdrant to settle
    time.sleep(0.5)
    count_result1 = client.count(collection_name, exact=True)
    points_after_first = count_result1.count

    # Second run (simulating restart) — same chunk IDs should produce no new points
    result2 = batch_index(memory_db, batch_size=20, collection_name=collection_name)
    # Second run should process 0 (all already indexed)
    assert result2.processed == 0, f"Second run should process 0, got {result2.processed}"

    time.sleep(0.5)
    count_result2 = client.count(collection_name, exact=True)
    points_after_second = count_result2.count

    # No new points created (upsert is idempotent by chunk_id)
    assert points_after_second == points_after_first, (
        f"Duplicate Qdrant points: first={points_after_first}, second={points_after_second}"
    )


# ---------------------------------------------------------------------------
# AC-3: Catch-up on init — processes all pending turns before normal polling
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _lms_available(), reason="LMS at 192.168.2.105:1235 not reachable")
def test_indexer_catch_up_on_init(memory_db, qdrant_client, lms_available):
    """AC-3: Given pending turns at plugin init, catch-up runs and processes them."""
    from hermes_memory_core.indexer import IndexerWorker

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    client, collection_name = qdrant_client

    # Insert some pending turns before worker starts
    insert_5_pending_turns(memory_db, session_id)

    # Create worker — its __init__ calls catch_up()
    worker = IndexerWorker(
        memory_db=memory_db,
        poll_interval=30,
        batch_size=20,
        collection_name=collection_name,
        gateway_url=None,
    )

    try:
        # Worker should have processed the pending turns during catch_up
        conn = memory_db._connect()
        try:
            rows = conn.execute(
                "SELECT turn_id, index_status FROM turns WHERE session_id=?",
                (session_id,),
            ).fetchall()
            for turn_id, status in rows:
                assert status == "indexed", (
                    f"Catch-up failed: {turn_id} is '{status}', expected 'indexed'"
                )
        finally:
            conn.close()
    finally:
        worker.stop()


# ---------------------------------------------------------------------------
# AC-4: Embedding failure → retry → index_status='failed' after max retries
# ---------------------------------------------------------------------------
def test_indexer_embedding_failure_sets_failed(memory_db, qdrant_client, monkeypatch):
    """AC-4: Given embedding failure, index_status='failed' after max retries."""
    session_id = f"sess_{uuid.uuid4().hex[:12]}"

    # Insert one pending turn
    insert_5_pending_turns(memory_db, session_id)

    # Patch embed_batch at the module level to always raise
    call_count = 0

    def bad_embed_batch(self, texts):
        nonlocal call_count
        call_count += 1
        raise Exception("LMS endpoint unreachable")

    # Patch it on the actual LMSClient class
    import hermes_memory_core.embed as embed_module
    monkeypatch.setattr(embed_module.LMSClient, "embed_batch", bad_embed_batch)

    from hermes_memory_core.indexer import batch_index

    _, collection_name = qdrant_client
    result = batch_index(memory_db, batch_size=20, collection_name=collection_name)
    assert result.processed >= 1, f"Expected ≥1 processed, got {result.processed}"

    # Verify: at least one turn now has index_status='failed'
    conn = memory_db._connect()
    try:
        rows = conn.execute(
            "SELECT index_status FROM turns WHERE session_id=?",
            (session_id,),
        ).fetchall()
        assert len(rows) > 0, "No turns found after batch_index"
        statuses = {r[0] for r in rows}
        assert "failed" in statuses, f"Expected 'failed' in {statuses}"
    finally:
        conn.close()

    assert call_count >= 1, f"Expected ≥1 embed_batch call, got {call_count}"


# ---------------------------------------------------------------------------
# AC-5: Polling interval ≤30s for new pending turns
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _lms_available(), reason="LMS at 192.168.2.105:1235 not reachable")
def test_indexer_polling_interval(memory_db, qdrant_client, lms_available):
    """AC-5: New pending turn is picked up within 30s polling interval."""
    from hermes_memory_core.indexer import IndexerWorker

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    _, collection_name = qdrant_client

    worker = IndexerWorker(
        memory_db=memory_db,
        poll_interval=5,  # Short interval for test
        batch_size=20,
        collection_name=collection_name,
        gateway_url=None,
    )

    try:
        # Give worker time to do its first poll + catch-up
        time.sleep(2)

        # Insert a new pending turn after worker started
        conn = memory_db._connect()
        try:
            turn_id = f"turn_{uuid.uuid4().hex[:12]}"
            conn.execute(
                """INSERT INTO turns
                   (turn_id, session_id, sequence, timestamp, role, content,
                    dream_status, index_status, source_refs_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    999,
                    datetime.now(timezone.utc).isoformat(),
                    "user",
                    "Late-arriving pending turn",
                    "pending",
                    "pending",
                    "[]",
                ),
            )
            conn.commit()
        finally:
            conn.close()

        # Wait for next poll cycle (poll_interval=5s, give 8s margin)
        time.sleep(8)

        # Verify: the late turn was picked up and indexed
        conn = memory_db._connect()
        try:
            row = conn.execute(
                "SELECT index_status FROM turns WHERE turn_id=?",
                (turn_id,),
            ).fetchone()
            assert row is not None, f"Turn {turn_id} not found in DB"
            assert row[0] == "indexed", f"Expected 'indexed' within 30s, got '{row[0]}'"
        finally:
            conn.close()
    finally:
        worker.stop()