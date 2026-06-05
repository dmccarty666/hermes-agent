"""
Resumable batch chunk → embed → Qdrant indexer.

Run standalone:
    python -m hermes_memory_core.chunk.indexer --daemon --batch-limit 3 --max-tokens 512

Or imported as a library:
    from hermes_memory_core.chunk.indexer import ChunkIndexer
    indexer = ChunkIndexer()
    stats = indexer.run_once()
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from hermes_memory_core.chunk.indexer import (
    BatchJob,
    ChunkBatchStore,
    BatchState,
    MAX_RETRIES,
    INITIAL_BACKOFF,
    MAX_BACKOFF,
    BACKOFF_MULTIPLIER,
    BATCH_WATCHDOG,
    MAX_EMBED_BATCH,
)

logger = logging.getLogger(__name__)


# ── Paths ────────────────────────────────────────────────────────────────────

HERMES_HOME = Path.home() / ".hermes"
DB_PATH = HERMES_HOME / "memory" / "index" / "memory.sqlite"


# ── ID helpers ───────────────────────────────────────────────────────────────

def _make_qdrant_id(chunk_id: str) -> str:
    """Derive a valid Qdrant UUID from a chunk_id (alphanumeric string)."""
    import uuid
    b = chunk_id.encode()[:16].ljust(16, b'\0')
    return str(uuid.UUID(bytes=b))


# ── Indexer ──────────────────────────────────────────────────────────────────

class ChunkIndexer:
    """
    Processes chunk batches one at a time through the pipeline:

        pending → chunking → embedding → indexing → done

    Each stage is guarded by a watchdog timer. On failure the batch goes
    back to pending with exponential backoff. On exhaustion it goes to failed.

    Thread-safe for concurrent instances (SQLite WAL + claim pattern).
    """

    def __init__(
        self,
        db_path: str | Path = DB_PATH,
        embed_batch_size: int = MAX_EMBED_BATCH,
        max_tokens_per_chunk: int = 512,
        overlap_tokens: int = 64,
        watchdog: int = BATCH_WATCHDOG,
    ):
        self.db_path        = Path(db_path)
        self.batch_size    = embed_batch_size
        self.max_tokens    = max_tokens_per_chunk
        self.overlap       = overlap_tokens
        self.watchdog_sec  = watchdog
        self._stop         = False

        # Lazy imports — keep startup fast
        self._embed_client = None
        self._qdrant_store = None

        # Wire up signal handlers so daemon mode can stop cleanly
        signal.signal(signal.SIGTERM, self._sig)
        signal.signal(signal.SIGALRM, self._watchdog_handler)

    # ── lazy clients ─────────────────────────────────────────────────────────

    def _embed(self):
        if self._embed_client is None:
            from hermes_memory_core.embed import get_embedding_client
            self._embed_client = get_embedding_client()
        return self._embed_client

    def _qdrant(self):
        if self._qdrant_store is None:
            from hermes_memory_core.store.qdrant import QdrantStore
            self._qdrant_store = QdrantStore()
        return self._qdrant_store

    # ── signal handlers ───────────────────────────────────────────────────────

    def _sig(self, signum, frame):
        logger.info("Received signal %d — stopping after current batch", signum)
        self._stop = True

    def _watchdog_handler(self, signum, frame):
        raise TimeoutError(f"Batch stage exceeded {self.watchdog_sec}s")

    # ── stage watchdog ───────────────────────────────────────────────────────

    def _watchdog_start(self) -> None:
        signal.alarm(self.watchdog_sec)

    def _watchdog_cancel(self) -> None:
        signal.alarm(0)

    # ── one complete batch ───────────────────────────────────────────────────

    def process_one(self, batch: BatchJob) -> BatchJob:
        """
        Run a single batch through all three stages.
        Returns the updated BatchJob (state may be done/failed/pending).
        """
        from hermes_memory_core.chunk import chunk_turns
        import json

        store = ChunkBatchStore(self.db_path)
        batch_id = batch.batch_id

        try:
            # ── Stage 1: chunk ────────────────────────────────────────────────
            logger.info("[%s] chunking %d turns", batch_id, len(batch.turns_data))
            self._watchdog_start()
            try:
                # Build turn dicts compatible with chunk_turns()
                turns = []
                for td in batch.turns_data:
                    turns.append({
                        "turn_id":   td["turn_id"],
                        "session_id": batch.session_id,
                        "content":    td.get("content", ""),
                    })

                chunks = chunk_turns(
                    turns,
                    max_tokens=self.max_tokens,
                    overlap_tokens=self.overlap,
                    embed_model="nomic-embed-text-v1.5",
                )
                chunk_ids = [c.chunk_id for c in chunks]
                chunk_texts = [c.text for c in chunks]
                logger.info("[%s] created %d chunks", batch_id, len(chunks))
            finally:
                self._watchdog_cancel()

            # Persist chunk_ids before moving to embedding (idempotent)
            store.mark_chunking_done(batch_id, chunk_ids)

            # ── Stage 2: embed ────────────────────────────────────────────────
            logger.info("[%s] embedding %d chunks", batch_id, len(chunk_texts))
            self._watchdog_start()
            try:
                vectors: list[list[float]] = []
                texts_for_embed: list[str] = chunk_texts

                # Slice into embed_batch_size chunks to keep HTTP request bounded
                for i in range(0, len(texts_for_embed), self.batch_size):
                    slice_texts = texts_for_embed[i:i + self.batch_size]
                    vecs = self._embed().embed_batch(slice_texts)
                    vectors.extend(vecs)
                    logger.debug("[%s] embed slice %d-%d done", batch_id, i, i + len(slice_texts))

                logger.info("[%s] embedded %d vectors (dim=%d)",
                            batch_id, len(vectors), len(vectors[0]) if vectors else 0)
            finally:
                self._watchdog_cancel()

            store.mark_embedding_done(batch_id)

            # ── Stage 3: index ────────────────────────────────────────────────
            logger.info("[%s] upserting %d points to Qdrant", batch_id, len(chunk_texts))
            self._watchdog_start()
            try:
                qdrant = self._qdrant()
                if not qdrant.is_available():
                    raise RuntimeError("Qdrant not available")

                # Build Points list with pre-computed vectors
                from qdrant_client.http.models import PointStruct

                collection = "hermes_memory_chunks_nomic_v15"
                points = []
                for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                    qdrant_id = _make_qdrant_id(chunk.chunk_id)
                    payload = {
                        "chunk_id":   chunk.chunk_id,
                        "session_id": batch.session_id,
                        "text":       chunk.text,
                        "token_count": chunk.token_count,
                        "embed_model": chunk.embed_model,
                        "batch_id":   batch_id,
                    }
                    points.append(PointStruct(
                        id=qdrant_id,
                        vector=vec,
                        payload=payload,
                    ))

                n = qdrant.upsert(collection=collection, points=[
                    {"id": p.id, "vector": p.vector, "payload": p.payload}
                    for p in points
                ])
                point_ids = [_make_qdrant_id(c.chunk_id) for c in chunks]
                logger.info("[%s] upserted %d points to Qdrant", batch_id, n)
            finally:
                self._watchdog_cancel()

            store.mark_done(batch_id, point_ids)

            # Update turns table so dream pipeline skips these
            self._mark_turns_indexed(batch.turn_ids)

            logger.info("[%s] DONE — %d chunks indexed", batch_id, len(point_ids))
            return store.get_batch(batch_id)

        except Exception as exc:
            self._watchdog_cancel()
            return self._handle_failure(batch, exc)

    def _handle_failure(self, batch: BatchJob, exc: Exception) -> BatchJob:
        """Apply backoff and either retry or give up."""
        store = ChunkBatchStore(self.db_path)
        new_retry  = batch.retry_count + 1
        new_backoff = min(INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** batch.retry_count), MAX_BACKOFF)

        logger.warning("[%s] stage=%s error=%s — retry %d/%d (backoff=%.0fs)",
                       batch.batch_id, batch.stage, exc, new_retry, MAX_RETRIES, new_backoff)

        store.mark_failed(batch.batch_id, str(exc), new_retry, new_backoff)
        return store.get_batch(batch.batch_id)

    def _mark_turns_indexed(self, turn_ids: list[str]) -> None:
        """Mark turns as indexed so dream pipeline skips them."""
        from hermes_memory_core.store.sqlite import get_memory_store
        store = get_memory_store()
        conn = store._conn_or_init()
        conn.execute(
            "UPDATE turns SET index_status = 'indexed' WHERE turn_id IN ("
            + ",".join("?" * len(turn_ids)) + ")",
            turn_ids,
        )
        conn.commit()

    # ── run once / daemon ───────────────────────────────────────────────────

    def run_once(self, batch_limit: int = 1) -> dict:
        """
        Claim and process up to `batch_limit` batches.
        Returns a stats dict.
        """
        store = ChunkBatchStore(self.db_path)
        processed = 0
        failed = 0

        for _ in range(batch_limit):
            if self._stop:
                break

            batch = store.claim_next_batch()
            if batch is None:
                logger.debug("No pending batches — sleeping")
                break

            result = self.process_one(batch)
            if result.state in (BatchState.DONE,):
                processed += 1
            else:
                failed += 1

        return {
            "processed": processed,
            "failed":    failed,
            "queue":     store.get_queue_stats(),
        }

    def run_daemon(
        self,
        poll_interval: float = 15.0,
        batch_limit: int = 3,
    ) -> None:
        """
        Run forever: poll, process, sleep.
        Exit cleanly on SIGTERM.
        """
        logger.info("ChunkIndexer daemon starting — poll_interval=%.0fs batch_limit=%d",
                    poll_interval, batch_limit)

        while not self._stop:
            stats = self.run_once(batch_limit=batch_limit)
            if stats["processed"] == 0 and stats["failed"] == 0:
                # Nothing to do — sleep longer
                time.sleep(poll_interval)
            elif stats["failed"] > 0:
                # Some failed — back off slightly to avoid tight retry loop
                time.sleep(5.0)

        logger.info("ChunkIndexer daemon stopped")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(description="Hermes Memory Chunk Indexer")
    parser.add_argument("--daemon", action="store_true", help="Run forever as a daemon")
    parser.add_argument("--batch-limit", type=int, default=3, help="Max batches per poll cycle")
    parser.add_argument("--poll-interval", type=float, default=15.0, help="Seconds between polls (daemon)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per chunk")
    parser.add_argument("--overlap-tokens", type=int, default=64, help="Overlap between chunks")
    parser.add_argument("--watchdog", type=int, default=BATCH_WATCHDOG, help="Seconds per batch stage")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
    )

    indexer = ChunkIndexer(
        max_tokens_per_chunk=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        watchdog=args.watchdog,
    )

    if args.daemon:
        indexer.run_daemon(poll_interval=args.poll_interval, batch_limit=args.batch_limit)
    else:
        stats = indexer.run_once(batch_limit=args.batch_limit)
        print(stats)


if __name__ == "__main__":
    _main()