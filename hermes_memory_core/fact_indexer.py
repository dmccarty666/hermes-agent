"""
Resumable batch fact → embed → Qdrant indexer.

Run standalone:
    python -m hermes_memory_core.fact_indexer --backfill          # one-shot catch-up
    python -m hermes_memory_core.fact_indexer --backfill --limit 500  # partial
    python -m hermes_memory_core.fact_indexer --daemon             # continuous

Or imported as a library:
    from hermes_memory_core.fact_indexer import FactIndexer
    indexer = FactIndexer()
    indexer.backfill()          # full catch-up
    stats = indexer.run_once()  # process one cycle in daemon mode
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import signal
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Paths / Constants ────────────────────────────────────────────────────────
HERMES_HOME = Path.home() / ".hermes"
DB_PATH = HERMES_HOME / "memory" / "index" / "memory.sqlite"
COLLECTION = "hermes_memory_facts_nomic_v15"
MAX_EMBED_BATCH = 50
BATCH_WATCHDOG = 300
DEFAULT_BATCH_SIZE = 50


# ── Batch state enum ───────────────────────────────────────────────────────
class FactBatchState(str, Enum):
    PENDING = "pending"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    DONE = "done"
    FAILED = "failed"


# ── Batch job dataclass ─────────────────────────────────────────────────────
@dataclass
class FactBatch:
    batch_id: str
    state: FactBatchState
    fact_ids_json: str
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    backoff_sec: float = 0.0


# ── Store ─────────────────────────────────────────────────────────────────────
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json_list(val: Any) -> list:
    """Parse a JSON list, tolerating bare comma-separated strings."""
    if not val:
        return []
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return [v.strip() for v in str(val).split(",") if v.strip()]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create fact_index_batches table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fact_index_batches (
            batch_id       TEXT PRIMARY KEY,
            state          TEXT NOT NULL DEFAULT 'pending',
            fact_ids_json  TEXT NOT NULL,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL,
            started_at     TEXT,
            completed_at   TEXT,
            error          TEXT,
            retry_count    INTEGER NOT NULL DEFAULT 0,
            backoff_sec    REAL NOT NULL DEFAULT 0.0
        )
    """)


def _ensure_indexed_at_column(db_path: Path) -> None:
    """Add indexed_at TEXT column to facts table if missing."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(facts)")
    cols = [row[1] for row in cur.fetchall()]
    if "indexed_at" not in cols:
        cur.execute("ALTER TABLE facts ADD COLUMN indexed_at TEXT")
        conn.commit()
        logger.info("Added indexed_at column to facts table")
    conn.close()


class FactBatchStore:
    """CRUD operations on fact_index_batches."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        _ensure_schema(conn)
        return conn

    # ── write ────────────────────────────────────────────────────────────────
    def create_batch(self, fact_ids: list[str]) -> str:
        """Insert a batch of fact IDs as PENDING. Returns batch_id."""
        batch_id = hashlib.sha1(
            "".join(sorted(fact_ids)).encode()
        ).hexdigest()[:16]
        now = _utc_now()
        conn = self._conn()
        try:
            conn.execute("""
                INSERT OR IGNORE INTO fact_index_batches
                (batch_id, state, fact_ids_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (batch_id, FactBatchState.PENDING.value,
                  json.dumps(fact_ids), now, now))
            conn.commit()
        finally:
            conn.close()
        return batch_id

    def advance(self, batch_id: str, state: FactBatchState,
                error: Optional[str] = None) -> None:
        """Move a batch to a new state."""
        conn = self._conn()
        now = _utc_now()
        conn.execute("""
            UPDATE fact_index_batches
            SET state = ?, updated_at = ?, started_at = COALESCE(started_at, ?),
                completed_at = ?, error = ?, retry_count = retry_count + 1,
                backoff_sec = CASE WHEN ? = 'failed'
                                   THEN MIN(backoff_sec * 2 + 5, 300)
                                   ELSE backoff_sec END
            WHERE batch_id = ?
        """, (state.value, now, now,
              now if state in (FactBatchState.DONE, FactBatchState.FAILED) else None,
              error, state.value, batch_id))
        conn.commit()
        conn.close()

    def get_batch(self, batch_id: str) -> Optional[FactBatch]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM fact_index_batches WHERE batch_id = ?",
            (batch_id,)
        ).fetchone()
        all_cols = [c[1] for c in conn.execute(
            "PRAGMA table_info(fact_index_batches)"
        ).fetchall()]
        conn.close()
        if not row:
            return None
        kwargs = dict(zip(all_cols, row))
        kwargs["state"] = FactBatchState(kwargs["state"])
        return FactBatch(**kwargs)

    def claim_next_batch(self) -> Optional[FactBatch]:
        """
        Atomically claim the oldest ready-to-process pending batch.
        Returns None if no batch is ready.
        """
        conn = self._conn()
        now = _utc_now()
        try:
            row = conn.execute("""
                SELECT batch_id, state, fact_ids_json, created_at, updated_at
                FROM fact_index_batches
                WHERE state = 'pending'
                  AND (backoff_sec <= 0.0
                       OR updated_at <= datetime(?, '-' || CAST(backoff_sec AS INTEGER) || ' seconds'))
                ORDER BY updated_at ASC
                LIMIT 1
            """, (now,)).fetchone()

            if not row:
                conn.close()
                return None

            batch_id = row[0]
            conn.execute("""
                UPDATE fact_index_batches
                SET state = 'embedding', started_at = ?, updated_at = ?
                WHERE batch_id = ? AND state = 'pending'
            """, (now, now, batch_id))
            conn.commit()

            row2 = conn.execute(
                "SELECT * FROM fact_index_batches WHERE batch_id = ?",
                (batch_id,)
            ).fetchone()
            all_cols = [c[1] for c in conn.execute(
                "PRAGMA table_info(fact_index_batches)"
            ).fetchall()]
            conn.close()
            if not row2:
                return None
            kwargs = dict(zip(all_cols, row2))
            kwargs["state"] = FactBatchState(kwargs["state"])
            return FactBatch(**kwargs)
        except Exception:
            conn.close()
            raise

    def get_queue_stats(self) -> dict:
        conn = self._conn()
        stats = {}
        for row in conn.execute(
            "SELECT state, COUNT(*) FROM fact_index_batches GROUP BY state"
        ).fetchall():
            stats[row[0]] = row[1]
        conn.close()
        return stats

    # ── unindexed facts helpers ─────────────────────────────────────────────
    def count_unindexed(self) -> int:
        conn = self._conn()
        n = conn.execute("""
            SELECT COUNT(*) FROM facts
            WHERE status = 'active' AND indexed_at IS NULL
        """).fetchone()[0]
        conn.close()
        return n

    def enqueue_all_unindexed(self, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
        """Enqueue all unindexed active facts as batches. Returns batch count."""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT fact_id FROM facts
            WHERE status = 'active' AND indexed_at IS NULL
            ORDER BY created_at ASC
        """)
        fact_ids = [row[0] for row in cur.fetchall()]
        conn.close()

        if not fact_ids:
            return 0

        enqueued = 0
        for i in range(0, len(fact_ids), batch_size):
            batch = fact_ids[i:i + batch_size]
            self.create_batch(batch)
            enqueued += 1
        logger.info("Enqueued %d facts in %d batches", len(fact_ids), enqueued)
        return enqueued


# ── Indexer ─────────────────────────────────────────────────────────────────
class FactIndexer:
    """
    Three-stage pipeline for facts: load → embed → Qdrant upsert.

    Two run modes:
      backfill()  — one-shot catch-up of all unindexed facts (for cron)
      run_once()  — process one poll cycle (for daemon mode)
      run_daemon() — run forever calling run_once()

    Thread-safe (SQLite WAL + per-batch claim pattern).
    """

    def __init__(
        self,
        db_path: str | Path = DB_PATH,
        embed_batch_size: int = MAX_EMBED_BATCH,
        watchdog: int = BATCH_WATCHDOG,
    ):
        self.db_path = Path(db_path)
        self.batch_size = embed_batch_size
        self.watchdog_sec = watchdog
        self._stop = False
        self._embed = None
        self._qdrant = None

        _ensure_indexed_at_column(self.db_path)
        signal.signal(signal.SIGTERM, self._sig)
        signal.signal(signal.SIGALRM, self._watchdog_handler)

    def _sig(self, signum, frame):
        logger.info("Received signal %d — stopping", signum)
        self._stop = True

    def _watchdog_handler(self, signum, frame):
        raise TimeoutError(f"Stage exceeded {self.watchdog_sec}s")

    def _wd_start(self):
        signal.alarm(self.watchdog_sec)

    def _wd_cancel(self):
        signal.alarm(0)

    @property
    def _embed_client(self):
        if self._embed is None:
            from hermes_memory_core.embed import get_embedding_client
            self._embed = get_embedding_client()
        return self._embed

    @property
    def _qdrant_store(self):
        if self._qdrant is None:
            from hermes_memory_core.store.qdrant import QdrantStore
            self._qdrant = QdrantStore()
        return self._qdrant

    def _load_facts(self, fact_ids: list[str]) -> list[dict]:
        """Load full fact rows from SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        placeholders = ",".join("?" * len(fact_ids))
        cur.execute(f"""
            SELECT fact_id, fact_text, category, project, entity,
                   status, tags_json, source_refs_json, created_at
            FROM facts
            WHERE fact_id IN ({placeholders}) AND status = 'active'
        """, fact_ids)
        rows = []
        for (fid, text, cat, proj, entity,
             status, tags, refs, created) in cur.fetchall():
            rows.append({
                "fact_id": fid,
                "fact_text": text,
                "category": cat or "general",
                "project": proj,
                "entity": entity,
                "status": status,
                "tags": _parse_json_list(tags),
                "source_refs": _parse_json_list(refs),
                "created_at": created,
            })
        conn.close()
        return rows

    def _mark_indexed(self, fact_ids: list[str]) -> None:
        """Mark facts as indexed in SQLite."""
        if not fact_ids:
            return
        conn = sqlite3.connect(str(self.db_path))
        now = _utc_now()
        conn.execute(
            "UPDATE facts SET indexed_at = ? WHERE fact_id IN ("
            + ",".join("?" * len(fact_ids)) + ")",
            [now] + fact_ids
        )
        conn.commit()
        conn.close()

    def _make_qdrant_id(self, fact_id: str) -> str:
        import uuid
        b = fact_id.encode()[:16].ljust(16, b'\0')
        return str(uuid.UUID(bytes=b))

    # ── core processing ─────────────────────────────────────────────────────
    def process_one(self, batch: FactBatch) -> FactBatch:
        """Run one batch: embed → Qdrant upsert. Returns updated FactBatch."""
        store = FactBatchStore(self.db_path)
        fact_ids = json.loads(batch.fact_ids_json)

        if not fact_ids:
            store.advance(batch.batch_id, FactBatchState.DONE)
            result = store.get_batch(batch.batch_id)
            assert result is not None
            return result

        try:
            # ── load ───────────────────────────────────────────────────────
            logger.info("[%s] loading %d facts", batch.batch_id, len(fact_ids))
            self._wd_start()
            try:
                facts = self._load_facts(fact_ids)
                texts = [f["fact_text"] for f in facts]
                logger.info("[%s] loaded %d texts", batch.batch_id, len(texts))
            finally:
                self._wd_cancel()

            if not texts:
                store.advance(batch.batch_id, FactBatchState.DONE)
                return store.get_batch(batch.batch_id)

            # ── embed ───────────────────────────────────────────────────────
            logger.info("[%s] embedding %d texts", batch.batch_id, len(texts))
            self._wd_start()
            try:
                vectors: list[list[float]] = []
                for i in range(0, len(texts), self.batch_size):
                    vecs = self._embed_client.embed_batch(texts[i:i + self.batch_size])
                    vectors.extend(vecs)
                logger.info("[%s] embedded %d vectors (dim=%d)",
                            batch.batch_id, len(vectors),
                            len(vectors[0]) if vectors else 0)
            finally:
                self._wd_cancel()
            store.advance(batch.batch_id, FactBatchState.INDEXING)

            # ── upsert ─────────────────────────────────────────────────────
            logger.info("[%s] upserting %d points to Qdrant", batch.batch_id, len(facts))
            self._wd_start()
            try:
                qdrant = self._qdrant_store
                if not qdrant.is_available():
                    raise RuntimeError("Qdrant not available")

                points = []
                for fact, vec in zip(facts, vectors):
                    points.append({
                        "id": self._make_qdrant_id(fact["fact_id"]),
                        "vector": vec,
                        "payload": {
                            "fact_id":    fact["fact_id"],
                            "fact_text":  fact["fact_text"],
                            "category":   fact["category"],
                            "project":    fact["project"] or "",
                            "entity":     fact["entity"] or "",
                            "status":     fact["status"],
                            "tags":       fact["tags"],
                            "source_refs": fact["source_refs"],
                            "created_at": fact["created_at"],
                            "memory_type": "fact",
                            "date":        fact["created_at"][:10] if fact["created_at"] else "",
                        },
                    })

                n = qdrant.upsert(collection=COLLECTION, points=points)
                logger.info("[%s] upserted %d points", batch.batch_id, n)
            finally:
                self._wd_cancel()

            self._mark_indexed(fact_ids)
            store.advance(batch.batch_id, FactBatchState.DONE)
            result = store.get_batch(batch.batch_id)
            logger.info("[%s] DONE — %d facts indexed", batch.batch_id, len(facts))
            return result

        except Exception as exc:
            self._wd_cancel()
            logger.warning("[%s] error: %s", batch.batch_id, exc)
            store.advance(batch.batch_id, FactBatchState.FAILED, error=str(exc))
            result = store.get_batch(batch.batch_id)
            assert result is not None
            return result

    # ── run modes ───────────────────────────────────────────────────────────
    def run_once(self, batch_limit: int = 3) -> dict:
        """Process up to batch_limit batches. Returns stats."""
        store = FactBatchStore(self.db_path)
        processed, failed = 0, 0
        for _ in range(batch_limit):
            if self._stop:
                break
            batch = store.claim_next_batch()
            if batch is None:
                break
            result = self.process_one(batch)
            if result and result.state == FactBatchState.DONE:
                processed += 1
            else:
                failed += 1
        return {
            "processed": processed,
            "failed": failed,
            "queue": store.get_queue_stats(),
            "unindexed_remaining": store.count_unindexed(),
        }

    def run_daemon(self, poll_interval: float = 60.0,
                   batch_limit: int = 3) -> None:
        """Run forever: enqueue new unindexed facts, process, sleep."""
        logger.info("FactIndexer daemon starting — poll=%.0fs batch_limit=%d",
                    poll_interval, batch_limit)
        while not self._stop:
            # Enqueue any new unindexed facts that showed up since last poll
            enqueued = FactBatchStore(self.db_path).enqueue_all_unindexed()
            if enqueued > 0:
                logger.info("Daemondaemon: enqueued %d new batches", enqueued)

            stats = self.run_once(batch_limit=batch_limit)
            if stats["processed"] == 0 and stats["failed"] == 0:
                time.sleep(poll_interval)
            elif stats["failed"] > 0:
                time.sleep(5.0)
            else:
                time.sleep(max(poll_interval * 0.5, 10))

        logger.info("FactIndexer daemon stopped")

    def backfill(self, dry_run: bool = False,
                 max_facts: int = 0) -> dict:
        """
        One-shot backfill: enqueue all unindexed facts, process all batches.

        Args:
            dry_run  : count only, don't enqueue or process
            max_facts: cap total facts to process (0 = unlimited)
        """
        store = FactBatchStore(self.db_path)
        total = store.count_unindexed()
        target = min(total, max_facts) if max_facts else total

        logger.info("Backfill: %d unindexed facts (targeting %d)", total, target)
        if dry_run:
            return {"total": total, "target": target, "dry_run": True}

        # Enqueue in batches
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute("""
            SELECT fact_id FROM facts
            WHERE status = 'active' AND indexed_at IS NULL
            ORDER BY created_at ASC
            LIMIT ?
        """, (target,))
        fact_ids = [row[0] for row in cur.fetchall()]
        conn.close()

        enqueued = 0
        for i in range(0, len(fact_ids), DEFAULT_BATCH_SIZE):
            batch_fids = fact_ids[i:i + DEFAULT_BATCH_SIZE]
            store.create_batch(batch_fids)
            enqueued += 1

        logger.info("Backfill: enqueued %d facts in %d batches", len(fact_ids), enqueued)

        # Process all batches
        processed = failed = 0
        while True:
            stats = self.run_once(batch_limit=5)
            processed += stats["processed"]
            failed += stats["failed"]
            if stats["processed"] == 0 and stats["failed"] == 0:
                break

        return {
            "total": total,
            "target": target,
            "facts_enqueued": len(fact_ids),
            "batches_enqueued": enqueued,
            "processed": processed,
            "failed": failed,
            "dry_run": False,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────
def _main():
    parser = argparse.ArgumentParser(description="Hermes Memory Fact Indexer")
    parser.add_argument("--daemon", action="store_true", help="Run forever")
    parser.add_argument("--backfill", action="store_true", help="One-shot backfill")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --backfill: count only")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max facts to backfill (0 = all)")
    parser.add_argument("--batch-limit", type=int, default=3,
                        help="Batches per poll cycle")
    parser.add_argument("--poll-interval", type=float, default=60.0,
                        help="Seconds between polls (daemon)")
    parser.add_argument("--watchdog", type=int, default=BATCH_WATCHDOG,
                        help="Seconds per stage watchdog")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
    )

    indexer = FactIndexer(watchdog=args.watchdog)

    if args.backfill or args.dry_run:
        result = indexer.backfill(dry_run=args.dry_run, max_facts=args.limit)
        print(result)
    elif args.daemon:
        indexer.run_daemon(poll_interval=args.poll_interval,
                           batch_limit=args.batch_limit)
    else:
        # Single-shot run (no daemon, no backfill) — useful for cron
        result = indexer.backfill()
        print(result)


if __name__ == "__main__":
    _main()
