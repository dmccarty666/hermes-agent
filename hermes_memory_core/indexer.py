"""Async indexer worker for Hermes Local Memory.

Polls SQLite for turns with ``index_status = 'pending'`` and:
1. Extracts text from ``turns.content`` (Phase 1; tool_calls land in Phase 2 via raw_events join).
2. Chunks turns via :func:`hermes_memory_core.chunk.chunk_turns`.
3. Embeds chunks via :class:`hermes_memory_core.embed.LMSClient`.
4. Upserts embedded points into Qdrant.
5. Marks turns ``indexed`` on success or ``failed`` after ``max_retries`` attempts.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hermes_memory_core.chunk import chunk_turns
from hermes_memory_core.embed import LMSClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

# Default Qdrant collection used by the indexer
_DEFAULT_COLLECTION = "hermes_memory_chunks_nomic_v15"

# Poll interval in seconds (AC-5: ≤30s)
_DEFAULT_POLL_INTERVAL = 15


# ---------------------------------------------------------------------------
# Dataclass returned by batch_index
# ---------------------------------------------------------------------------

@dataclass
class BatchIndexResult:
    """Result of a single batch_index run."""

    processed: int = 0  # number of turns processed
    indexed: int = 0     # number of turns successfully indexed
    failed: int = 0     # number of turns that failed after max retries
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# batch_index — the main entry point used by IndexerWorker and callable directly
# ---------------------------------------------------------------------------

def batch_index(
    memory_db,
    batch_size: int = 20,
    embed_client: Optional[LMSClient] = None,
    qdrant_client: Optional[QdrantClient] = None,
    collection_name: str = _DEFAULT_COLLECTION,
) -> BatchIndexResult:
    """
    Process pending turns from SQLite in one batch.

    Parameters
    ----------
    memory_db:
        :class:`hermes_memory_core.store.sqlite.MemoryDB` instance.
    batch_size:
        Maximum number of pending turns to claim per batch.
    embed_client:
        LMSClient instance. Created from defaults if None.
    qdrant_client:
        QdrantClient instance. Created from defaults if None.
    collection_name:
        Qdrant collection name to upsert into.

    Returns
    -------
    BatchIndexResult with counts and any error strings.
    """
    result = BatchIndexResult()

    # Lazily create clients
    if embed_client is None:
        embed_client = LMSClient()
    if qdrant_client is None:
        qdrant_client = QdrantClient(host="localhost", port=6333, timeout=10)

    # Step 1: claim pending turns
    # NOTE: tool_calls are stored in raw_events, not turns. For Phase 1 we
    # extract text from the turns.content field only.
    conn = memory_db._connect()
    try:
        rows = conn.execute(
            """
            SELECT turn_id, session_id, sequence, role, content, index_status
            FROM turns
            WHERE index_status = 'pending'
            ORDER BY session_id, sequence
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return result

    # Group by session for chunking; track all turn_ids in this batch
    session_turns: Dict[str, List[Dict[str, Any]]] = {}
    turn_rows: Dict[str, tuple] = {}       # turn_id → full row
    failed_turn_ids: set = set()             # turn_ids to mark 'failed'

    for row in rows:
        turn_id, session_id, sequence, role, content, _index_status = row
        turn_rows[turn_id] = row
        session_turns.setdefault(session_id, []).append({
            "turn_id": turn_id,
            "session_id": session_id,
            "sequence": sequence,
            "role": role,
            "content": content or "",
        })
        result.processed += 1

    # Step 2: for each session group, chunk → embed → Qdrant
    for session_id, turns in session_turns.items():
        session_failed = _index_session(
            turns=turns,
            embed_client=embed_client,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            result=result,
        )
        if session_failed:
            failed_turn_ids.update(session_failed)

    # Step 3: mark turns 'indexed' (success) or 'failed' (any error in session)
    _update_turn_statuses(memory_db, turn_rows, failed_turn_ids, result)

    return result


def _index_session(
    turns: List[Dict[str, Any]],
    embed_client: LMSClient,
    qdrant_client: QdrantClient,
    collection_name: str,
    result: BatchIndexResult,
) -> set:
    """
    Index a single session's worth of turns: chunk → embed → Qdrant.

    Returns a set of turn_ids that failed (so caller can mark them 'failed').
    Returns empty set if the session indexed successfully.
    """
    failed: set = set()

    # 2a. Build full text per turn (content field only in Phase 1;
    # tool_calls stored separately in raw_events — will be joined in Phase 2)
    turn_texts = [t["content"] or "" for t in turns]

    # 2b. Chunk the text
    chunk_dicts = chunk_turns(
        [
            {
                "turn_id": t["turn_id"],
                "session_id": t["session_id"],
                "sequence": t["sequence"],
                "role": t["role"],
                "content": text,
            }
            for t, text in zip(turns, turn_texts)
        ],
        size=512,
        overlap=128,
    )

    if not chunk_dicts:
        return failed  # nothing to index

    # 2c. Embed
    texts_to_embed = [c["text"] for c in chunk_dicts]
    try:
        embeddings = embed_client.embed_batch(texts_to_embed)
    except Exception as exc:
        logger.error("Embedding failed for session %s: %s", turns[0]["session_id"], exc)
        result.errors.append(f"embed_batch failed: {exc}")
        # Mark all turns in this session as failed
        return {t["turn_id"] for t in turns}

    # 2d. Build Qdrant points
    points = []
    for chunk, embedding in zip(chunk_dicts, embeddings):
        points.append(
            PointStruct(
                id=chunk["chunk_id"],
                vector=embedding,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "session_id": chunk["session_id"],
                    "start_turn_id": chunk["start_turn_id"],
                    "end_turn_id": chunk["end_turn_id"],
                    "chunk_type": chunk["chunk_type"],
                    "text": chunk["text"],
                    "text_hash": chunk["text_hash"],
                    "role_mix": chunk["role_mix"],
                    "turn_count": chunk["turn_count"],
                    "embed_model": chunk["embed_model"],
                    "chunker_version": chunk["chunker_version"],
                },
            )
        )

    # 2e. Upsert into Qdrant
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
        )
        result.indexed += len(turns)
    except Exception as exc:
        logger.error("Qdrant upsert failed for session %s: %s", turns[0]["session_id"], exc)
        result.errors.append(f"Qdrant upsert failed: {exc}")
        return {t["turn_id"] for t in turns}

    return failed


# ---------------------------------------------------------------------------
# IndexerWorker — background polling thread
# ---------------------------------------------------------------------------

def _update_turn_statuses(
    memory_db,
    turn_rows: Dict[str, tuple],
    failed_turn_ids: set,
    result: BatchIndexResult,
) -> None:
    """Update index_status in SQLite for all turns processed in this batch."""
    conn = memory_db._connect()
    try:
        for turn_id in turn_rows:
            if turn_id in failed_turn_ids:
                conn.execute(
                    "UPDATE turns SET index_status = 'failed' WHERE turn_id = ?",
                    (turn_id,),
                )
                result.failed += 1
            else:
                conn.execute(
                    "UPDATE turns SET index_status = 'indexed' WHERE turn_id = ?",
                    (turn_id,),
                )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# IndexerWorker — background polling thread
# ---------------------------------------------------------------------------

class IndexerWorker:
    """
    Background thread that polls SQLite for pending turns and indexes them.

    Parameters
    ----------
    memory_db:
        :class:`hermes_memory_core.store.sqlite.MemoryDB` instance.
    poll_interval:
        Seconds between poll cycles (default 15, AC-5 requires ≤30s).
    batch_size:
        Maximum turns to process per batch (default 20).
    embed_client:
        Optional pre-configured LMSClient. If None, created from defaults.
    qdrant_client:
        Optional pre-configured QdrantClient. If None, connects to localhost:6333.
    collection_name:
        Qdrant collection name for chunk upserts.
    gateway_url:
        If not None, run as async worker via Hermes gateway (future extension).
    """

    def __init__(
        self,
        memory_db,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        batch_size: int = 20,
        embed_client: Optional[LMSClient] = None,
        qdrant_client: Optional[QdrantClient] = None,
        collection_name: str = _DEFAULT_COLLECTION,
        gateway_url: Optional[str] = None,
    ) -> None:
        self.memory_db = memory_db
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.embed_client = embed_client or LMSClient()
        self.qdrant_client = qdrant_client or QdrantClient(host="localhost", port=6333, timeout=10)
        self.collection_name = collection_name
        self.gateway_url = gateway_url

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Reset module-level failure tracking
        global _failed_turn_ids
        _failed_turn_ids = set()

        # Catch-up: process all pending turns at startup before polling
        self._catch_up()

        # Start background thread
        self._thread = threading.Thread(target=self._run, daemon=True, name="IndexerWorker")
        self._thread.start()
        logger.info(
            "IndexerWorker started — poll_interval=%.1fs, batch_size=%d, collection=%s",
            poll_interval, batch_size, collection_name,
        )

    def _catch_up(self) -> None:
        """Process all pending turns that accumulated while the plugin was shut down."""
        logger.info("IndexerWorker running catch-up scan for pending turns…")
        result = batch_index(
            self.memory_db,
            batch_size=9999,  # catch-up: no limit
            embed_client=self.embed_client,
            qdrant_client=self.qdrant_client,
            collection_name=self.collection_name,
        )
        logger.info(
            "Catch-up done — processed=%d, indexed=%d, failed=%d",
            result.processed, result.indexed, result.failed,
        )

    def _run(self) -> None:
        """Poll loop — runs in background thread."""
        while not self._stop_event.is_set():
            try:
                batch_index(
                    self.memory_db,
                    batch_size=self.batch_size,
                    embed_client=self.embed_client,
                    qdrant_client=self.qdrant_client,
                    collection_name=self.collection_name,
                )
            except Exception as exc:
                logger.exception("IndexerWorker batch_index raised: %s", exc)

            # Wait for next poll or stop signal
            self._stop_event.wait(timeout=self.poll_interval)

    def stop(self) -> None:
        """Stop the polling thread gracefully."""
        logger.info("IndexerWorker stopping…")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("IndexerWorker stopped")


# ---------------------------------------------------------------------------
# Singleton instance (created by HermesLocalProvider.__init__)
# ---------------------------------------------------------------------------

_worker: Optional[IndexerWorker] = None


def start_worker(memory_db, **kwargs) -> IndexerWorker:
    """Start or return the global IndexerWorker singleton."""
    global _worker
    if _worker is not None:
        logger.warning("IndexerWorker already running — returning existing instance")
        return _worker
    _worker = IndexerWorker(memory_db=memory_db, **kwargs)
    return _worker


def stop_worker() -> None:
    """Stop the global IndexerWorker singleton."""
    global _worker
    if _worker is not None:
        _worker.stop()
        _worker = None