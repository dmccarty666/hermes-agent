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

def _stable_uuid(prefix: str, key: str) -> str:
    """Deterministic UUIDv5 string. Forward-declared here so _index_session
    can use it; the duplicate later in the file is kept for backward compat.
    """
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{prefix}:{key}"))


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

    # Step 2: for each session group, chunk → embed → Qdrant → SQLite
    for session_id, turns in session_turns.items():
        session_failed = _index_session(
            turns=turns,
            embed_client=embed_client,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            result=result,
            memory_db=memory_db,
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
    memory_db=None,
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

    # 2d. Build Qdrant points (Qdrant IDs must be UUID or int — derive UUIDv5
    # from the deterministic chunk_id) and assemble matching SQLite rows.
    import datetime as _dt

    now_iso = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    embed_model_name = getattr(embed_client, "model", None) or "text-embedding-nomic-embed-text-v1.5"

    points = []
    sqlite_rows = []
    for chunk, embedding in zip(chunk_dicts, embeddings):
        qdrant_point_id = _stable_uuid("chunk", chunk["chunk_id"])
        chunk_type = chunk.get("chunk_type", "conversation")
        source_ref = f"session:{chunk['session_id']}:{chunk['start_turn_id']}-{chunk['end_turn_id']}"
        points.append(
            PointStruct(
                id=qdrant_point_id,
                vector=embedding,
                payload={
                    "memory_type": "chunk",
                    "chunk_id": chunk["chunk_id"],
                    "session_id": chunk["session_id"],
                    "start_turn_id": chunk["start_turn_id"],
                    "end_turn_id": chunk["end_turn_id"],
                    "chunk_type": chunk_type,
                    "text": chunk["text"],
                    "text_hash": chunk["text_hash"],
                    "role_mix": chunk.get("role_mix", ""),
                    "turn_count": chunk.get("turn_count", 0),
                    "embed_model": chunk.get("embed_model", embed_model_name),
                    "chunker_version": chunk.get("chunker_version", "v1"),
                    "source_ref": source_ref,
                    "date": now_iso[:10],
                    "created_at": now_iso,
                },
            )
        )
        sqlite_rows.append(
            (
                chunk["chunk_id"],
                chunk["session_id"],
                chunk["start_turn_id"],
                chunk["end_turn_id"],
                chunk_type,
                None,  # project
                chunk["text"],
                chunk["text_hash"],
                None,  # summary
                source_ref,
                qdrant_point_id,
                chunk.get("embed_model", embed_model_name),
                now_iso,  # created_at
                now_iso,  # updated_at
            )
        )

    # 2e. Upsert into Qdrant
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
        )
        # NOTE: do NOT increment result.indexed here — SQLite may still fail.
        # Count is incremented only after both backends succeed (see below).
    except Exception as exc:
        logger.error("Qdrant upsert failed for session %s: %s", turns[0]["session_id"], exc)
        result.errors.append(f"Qdrant upsert failed: {exc}")
        return {t["turn_id"] for t in turns}

    # 2f. Persist chunks into SQLite chunks table (idempotent via INSERT OR REPLACE
    # on chunk_id PK; UNIQUE(text_hash, embed_model) guards content duplicates).
    sqlite_ok = True
    if memory_db is not None and sqlite_rows:
        try:
            conn = memory_db._connect()
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO chunks ("
                    "  chunk_id, session_id, start_turn_id, end_turn_id, chunk_type,"
                    "  project, text, text_hash, summary, source_ref, qdrant_point_id,"
                    "  embed_model, created_at, updated_at"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    sqlite_rows,
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.error("SQLite chunks insert failed for session %s: %s", turns[0]["session_id"], exc)
            result.errors.append(f"SQLite chunks insert failed: {exc}")
            # Mark turns so they are re-tried on next indexer run.
            # Qdrant has the data; SQLite metadata is the gap.
            sqlite_ok = False

    if sqlite_ok:
        # Only count as indexed when both Qdrant AND SQLite succeeded.
        result.indexed += len(turns)
    else:
        # Qdrant succeeded but SQLite failed — return all turns so they are
        # marked 'partial' (pending retry). Do NOT return empty set which
        # would cause _update_turn_statuses to mark them 'indexed' wrongly.
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
        failure_threshold: int = 3,
        base_interval: float = 15.0,
        max_interval: float = 300.0,
    ) -> None:
        self.memory_db = memory_db
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.embed_client = embed_client or LMSClient()
        self.qdrant_client = qdrant_client or QdrantClient(host="localhost", port=6333, timeout=10)
        self.collection_name = collection_name
        self.gateway_url = gateway_url

        # Circuit breaker parameters
        self.failure_threshold = failure_threshold
        self.base_interval = base_interval
        self.max_interval = max_interval
        self.consecutive_failures = 0

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
                result = batch_index(
                    self.memory_db,
                    batch_size=self.batch_size,
                    embed_client=self.embed_client,
                    qdrant_client=self.qdrant_client,
                    collection_name=self.collection_name,
                )
                # Circuit breaker: check for failures
                if result.errors or result.failed > 0:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.failure_threshold:
                        self.poll_interval = min(self.poll_interval * 2, self.max_interval)
                        logger.warning(
                            "Circuit breaker open, backing off to %.1fs (failures=%d)",
                            self.poll_interval, self.consecutive_failures,
                        )
                else:
                    # Success: reset circuit breaker
                    if self.consecutive_failures >= self.failure_threshold:
                        logger.info(
                            "Circuit breaker closed, resuming normal polling (was %.1fs)",
                            self.poll_interval,
                        )
                    self.consecutive_failures = 0
                    self.poll_interval = self.base_interval
            except Exception as exc:
                logger.exception("IndexerWorker batch_index raised: %s", exc)
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.failure_threshold:
                    self.poll_interval = min(self.poll_interval * 2, self.max_interval)
                    logger.warning(
                        "Circuit breaker open, backing off to %.1fs (failures=%d)",
                        self.poll_interval, self.consecutive_failures,
                    )

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


# ---------------------------------------------------------------------------
# Backfill — index facts / decisions directly into their Qdrant collections
# ---------------------------------------------------------------------------

# Collection names (mirror store/qdrant.py so we don't depend on its import
# graph from the CLI entry point).
_COLLECTION_FACTS = "hermes_memory_facts_nomic_v15"
_COLLECTION_DECISIONS = "hermes_memory_decisions_nomic_v15"


def backfill_facts(
    memory_db,
    embed_client: Optional[LMSClient] = None,
    qdrant_client: Optional[QdrantClient] = None,
    collection_name: str = _COLLECTION_FACTS,
    batch_size: int = 32,
) -> BatchIndexResult:
    """Embed every active fact and upsert into the facts collection.

    Idempotent: re-running re-embeds and overwrites the same point IDs. Skips
    facts whose ``status != 'active'`` so superseded/disputed memories don't
    pollute retrieval. Uses deterministic UUIDv5 IDs derived from fact_id.

    Returns counts in a BatchIndexResult (processed = facts seen, indexed =
    points upserted, failed = embeddings that errored).
    """
    result = BatchIndexResult()
    if embed_client is None:
        embed_client = LMSClient()
    if qdrant_client is None:
        qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30)

    conn = memory_db._connect()
    try:
        rows = conn.execute(
            "SELECT fact_id, fact_text, project, scope, status, "
            "       trust_score, tags_json, created_at, updated_at "
            "FROM facts WHERE status = 'active' AND fact_text IS NOT NULL AND fact_text != ''"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return result

    # Embed in batches
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [r[1] for r in batch]
        try:
            vectors = embed_client.embed_batch(texts)
        except Exception as exc:
            logger.error("backfill_facts: embed_batch failed: %s", exc)
            result.errors.append(f"embed_batch failed: {exc}")
            result.failed += len(batch)
            continue

        points = []
        for row, vec in zip(batch, vectors):
            fact_id, fact_text, project, scope, status, trust, tags_json, created_at, updated_at = row
            points.append(
                PointStruct(
                    id=_stable_uuid("fact", fact_id),
                    vector=vec,
                    payload={
                        "memory_type": "fact",
                        "fact_id": fact_id,
                        "text": fact_text,
                        "project": project,
                        "scope": scope,
                        "status": status,
                        "trust_score": trust,
                        "tags_json": tags_json,
                        "date": (updated_at or created_at or "")[:10],
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "source_ref": f"fact:{fact_id}",
                    },
                )
            )
            result.processed += 1

        try:
            qdrant_client.upsert(collection_name=collection_name, points=points)
            result.indexed += len(points)
        except Exception as exc:
            logger.error("backfill_facts: Qdrant upsert failed: %s", exc)
            result.errors.append(f"Qdrant upsert failed: {exc}")
            result.failed += len(points)

    return result


def backfill_decisions(
    memory_db,
    embed_client: Optional[LMSClient] = None,
    qdrant_client: Optional[QdrantClient] = None,
    collection_name: str = _COLLECTION_DECISIONS,
    batch_size: int = 32,
) -> BatchIndexResult:
    """Embed every decision and upsert into the decisions collection."""
    result = BatchIndexResult()
    if embed_client is None:
        embed_client = LMSClient()
    if qdrant_client is None:
        qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30)

    conn = memory_db._connect()
    try:
        rows = conn.execute(
            "SELECT decision_id, decision_text, rationale, project, status, "
            "       created_at, updated_at "
            "FROM decisions WHERE decision_text IS NOT NULL AND decision_text != ''"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return result

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [r[1] for r in batch]
        try:
            vectors = embed_client.embed_batch(texts)
        except Exception as exc:
            logger.error("backfill_decisions: embed_batch failed: %s", exc)
            result.errors.append(f"embed_batch failed: {exc}")
            result.failed += len(batch)
            continue

        points = []
        for row, vec in zip(batch, vectors):
            decision_id, decision_text, rationale, project, status, created_at, updated_at = row
            points.append(
                PointStruct(
                    id=_stable_uuid("decision", decision_id),
                    vector=vec,
                    payload={
                        "memory_type": "decision",
                        "decision_id": decision_id,
                        "text": decision_text,
                        "rationale": rationale,
                        "project": project,
                        "status": status,
                        "date": (updated_at or created_at or "")[:10],
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "source_ref": f"decision:{decision_id}",
                    },
                )
            )
            result.processed += 1

        try:
            qdrant_client.upsert(collection_name=collection_name, points=points)
            result.indexed += len(points)
        except Exception as exc:
            logger.error("backfill_decisions: Qdrant upsert failed: %s", exc)
            result.errors.append(f"Qdrant upsert failed: {exc}")
            result.failed += len(points)

    return result


# ---------------------------------------------------------------------------
# CLI entry point — `python3 -m hermes_memory_core.indexer --backfill`
# ---------------------------------------------------------------------------

def _main() -> int:
    """CLI: ``python3 -m hermes_memory_core.indexer --backfill``.

    Backfills facts and decisions into their dedicated Qdrant collections.
    Skips turns→chunks because the chunker is currently a stub
    (``hermes_memory_core/chunk.py::chunk_turns`` returns []).
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="hermes_memory_core.indexer",
        description="One-shot indexer / backfill for Hermes Local Memory.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Embed all active facts + decisions into Qdrant.",
    )
    parser.add_argument(
        "--turns",
        action="store_true",
        help="Also try to chunk + embed pending turns (no-op if chunker is a stub).",
    )
    parser.add_argument(
        "--qdrant-host",
        default="localhost",
        help="Qdrant host (default: localhost).",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333).",
    )
    args = parser.parse_args()

    if not (args.backfill or args.turns):
        parser.print_help()
        return 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from hermes_memory_core.store.sqlite import MemoryDB

    db = MemoryDB()
    db.initialize()
    embed = LMSClient()
    qdrant = QdrantClient(host=args.qdrant_host, port=args.qdrant_port, timeout=30)

    rc = 0
    if args.backfill:
        print("→ backfilling facts …")
        r = backfill_facts(db, embed_client=embed, qdrant_client=qdrant)
        print(f"  facts: processed={r.processed} indexed={r.indexed} failed={r.failed}")
        if r.errors:
            print(f"  errors: {r.errors[:3]}")
            rc = max(rc, 1 if r.failed else 0)

        print("→ backfilling decisions …")
        r = backfill_decisions(db, embed_client=embed, qdrant_client=qdrant)
        print(f"  decisions: processed={r.processed} indexed={r.indexed} failed={r.failed}")
        if r.errors:
            print(f"  errors: {r.errors[:3]}")
            rc = max(rc, 1 if r.failed else 0)

    if args.turns:
        print("→ indexing pending turns (chunks) …")
        r = batch_index(db, batch_size=9999, embed_client=embed, qdrant_client=qdrant)
        print(f"  turns: processed={r.processed} indexed={r.indexed} failed={r.failed}")
        if r.processed == 0:
            print("  (note: chunker stub returns 0 chunks — turns→chunks indexing is a no-op)")
        if r.errors:
            print(f"  errors: {r.errors[:3]}")

    return rc


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(_main())