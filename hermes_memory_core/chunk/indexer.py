"""
Job queue for resumable batch operations (chunking, embedding, Qdrant upsert).

Design:
  - A batch represents N turns from one session, ready to be:
      1. chunked  (chunk_turns)
      2. embedded (LMS embed API)
      3. indexed   (Qdrant upsert)

  - A batch goes through states: pending → chunking → embedding → indexing → done | failed

  - On failure the batch stays in its current state so it can be retried.
  - A backoff counter tracks exponential backoff per batch.
  - The indexer picks up the oldest pending batch, processes it, and marks done/failed.
  - A watchdog timer per-batch prevents any single batch from taking too long.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Paths ────────────────────────────────────────────────────────────────────

HERMES_HOME = Path.home() / ".hermes"
DB_PATH = HERMES_HOME / "memory" / "index" / "memory.sqlite"


# ── State machine ─────────────────────────────────────────────────────────────

class BatchState(str, Enum):
    PENDING    = "pending"
    CHUNKING   = "chunking"
    EMBEDDING  = "embedding"
    INDEXING   = "indexing"
    DONE       = "done"
    FAILED     = "failed"


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_BATCH_TURNS    = 50
MAX_EMBED_BATCH    = 100
INITIAL_BACKOFF    = 30
MAX_BACKOFF        = 3600
BACKOFF_MULTIPLIER = 3.0
MAX_RETRIES        = 5
BATCH_WATCHDOG     = 300


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class BatchJob:
    batch_id:       str
    session_id:     str
    state:          BatchState
    turn_ids:       list[str]
    turns_data:     list[dict]
    created_at:     str
    updated_at:     str
    started_at:     Optional[str] = None
    completed_at:   Optional[str] = None
    chunk_ids:      list[str] = field(default_factory=list)
    point_ids:      list[str] = field(default_factory=list)
    error:          Optional[str] = None
    retry_count:    int = 0
    backoff_sec:    float = 0.0
    stage:          str = "pending"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> BatchJob:
        d["state"] = BatchState(d["state"])
        return cls(**d)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gen_id(parts: str) -> str:
    return hashlib.sha1(parts.encode()).hexdigest()[:16]


def _safe_json_parse(val: Any, default: Any = None) -> Any:
    """Parse a JSON string field, return default on failure."""
    if not val:
        return default
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return default


# ── Schema migration ──────────────────────────────────────────────────────────

def ensure_schema(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_batches (
            batch_id      TEXT PRIMARY KEY,
            session_id    TEXT NOT NULL,
            state         TEXT NOT NULL DEFAULT 'pending',
            turn_ids      TEXT NOT NULL,
            turns_data    TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            started_at    TEXT,
            completed_at  TEXT,
            chunk_ids     TEXT NOT NULL DEFAULT '[]',
            point_ids     TEXT NOT NULL DEFAULT '[]',
            error         TEXT,
            retry_count   INTEGER NOT NULL DEFAULT 0,
            backoff_sec   REAL    NOT NULL DEFAULT 0.0,
            stage         TEXT    NOT NULL DEFAULT 'pending'
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunk_batches_state
        ON chunk_batches(state, updated_at ASC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunk_batches_session
        ON chunk_batches(session_id, created_at ASC)
    """)
    conn.commit()


# ── Row → BatchJob ─────────────────────────────────────────────────────────────

def _row_to_batch(row: tuple, cols: list[str]) -> BatchJob:
    d = dict(zip(cols, row))
    for field_ in ("turn_ids", "turns_data", "chunk_ids", "point_ids"):
        d[field_] = _safe_json_parse(d.get(field_), [])
    return BatchJob.from_dict(d)


# ── Store ─────────────────────────────────────────────────────────────────────

class ChunkBatchStore:
    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        import sqlite3
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        ensure_schema(conn)
        conn.close()

    def _conn(self):
        import sqlite3
        return sqlite3.connect(str(self.db_path), timeout=10)

    # ── write ────────────────────────────────────────────────────────────────

    def create_batch(self, session_id: str, turn_ids: list[str],
                     turns_data: list[dict]) -> BatchJob:
        batch_id = _gen_id(f"{session_id}:{','.join(turn_ids)}")
        now = _utc_now()
        job = BatchJob(
            batch_id=batch_id,
            session_id=session_id,
            state=BatchState.PENDING,
            turn_ids=turn_ids,
            turns_data=turns_data,
            created_at=now,
            updated_at=now,
        )
        conn = self._conn()
        conn.execute("""
            INSERT INTO chunk_batches
                (batch_id, session_id, state, turn_ids, turns_data,
                 created_at, updated_at, chunk_ids, point_ids, retry_count, backoff_sec, stage)
            VALUES (?, ?, ?, ?, ?, ?, ?, '[]', '[]', 0, 0.0, 'pending')
        """, (batch_id, session_id, BatchState.PENDING.value,
              json.dumps(turn_ids), json.dumps(turns_data), now, now))
        conn.commit()
        conn.close()
        return job

    def claim_next_batch(self, max_turns: int = MAX_BATCH_TURNS) -> Optional[BatchJob]:
        """
        Claim oldest pending batch, OR build one by coalescing dreamed-but-not-indexed turns.
        """
        import sqlite3
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        ensure_schema(conn)
        now = _utc_now()

        # Try to claim an existing pending batch
        row = conn.execute("""
            SELECT batch_id FROM chunk_batches
            WHERE state = 'pending'
              AND (backoff_sec <= 0.0
                   OR updated_at <= datetime(?, '-' || CAST(backoff_sec AS INTEGER) || ' seconds'))
            ORDER BY updated_at ASC
        """, (now,)).fetchone()

        if row:
            batch_id = row[0]
            conn.execute("""
                UPDATE chunk_batches
                SET state = 'chunking', started_at = ?, updated_at = ?, stage = 'chunking'
                WHERE batch_id = ? AND state = 'pending'
            """, (now, now, batch_id))
            conn.commit()
            cols = [c[1] for c in conn.execute("PRAGMA table_info(chunk_batches)").fetchall()]
            row2 = conn.execute("SELECT * FROM chunk_batches WHERE batch_id = ?",
                                (batch_id,)).fetchone()
            conn.close()
            if row2 is None:
                return None
            return _row_to_batch(row2, cols)

        conn.close()

        # No pending batch — try to coalesce from dreamed-but-not-indexed turns
        return self._coalesce_batch(max_turns=max_turns)

    def _coalesce_batch(self, max_turns: int) -> Optional[BatchJob]:
        """
        Select dreamed-but-not-indexed turns, group by session, insert as 'chunking'.
        """
        import sqlite3
        store = get_memory_store()
        conn = store._conn_or_init()

        rows = conn.execute("""
            SELECT turn_id, session_id, content
            FROM turns
            WHERE dream_status = 'dreamed'
              AND (index_status IS NULL OR index_status != 'indexed')
              AND turn_id NOT IN (
                  SELECT value FROM chunk_batches, json_each(turn_ids)
                  WHERE state NOT IN ('done', 'failed')
              )
            ORDER BY session_id, sequence
            LIMIT ?
        """, (max_turns,)).fetchall()

        if not rows:
            return None

        by_session: dict[str, list] = {}
        for turn_id, session_id, content in rows:
            sid = session_id or "default"
            by_session.setdefault(sid, []).append(
                {"turn_id": turn_id, "session_id": sid, "content": content or ""}
            )

        sid, turns = next(iter(sorted(by_session.items(),
                                      key=lambda x: x[1][0].get("timestamp", ""))))
        turn_ids   = [t["turn_id"] for t in turns]
        turns_data = turns

        now = _utc_now()
        batch_id = _gen_id(f"{sid}:{','.join(turn_ids)}")

        try:
            conn.execute("""
                INSERT INTO chunk_batches
                    (batch_id, session_id, state, turn_ids, turns_data,
                     created_at, updated_at, started_at, chunk_ids, point_ids,
                     retry_count, backoff_sec, stage)
                VALUES (?, ?, 'chunking', ?, ?, ?, ?, ?, '[]', '[]', 0, 0.0, 'chunking')
            """, (batch_id, sid, json.dumps(turn_ids), json.dumps(turns_data),
                  now, now, now))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return None

        return BatchJob(
            batch_id=batch_id,
            session_id=sid,
            state=BatchState.CHUNKING,
            turn_ids=turn_ids,
            turns_data=turns_data,
            created_at=now,
            updated_at=now,
            started_at=now,
            chunk_ids=[],
            point_ids=[],
            error=None,
            retry_count=0,
            backoff_sec=0.0,
            stage="chunking",
        )

    def mark_chunking_done(self, batch_id: str, chunk_ids: list[str]) -> None:
        self._update(batch_id, state=BatchState.EMBEDDING,
                     chunk_ids=chunk_ids, stage="embedding")

    def mark_embedding_done(self, batch_id: str) -> None:
        self._update(batch_id, state=BatchState.INDEXING, stage="indexing")

    def mark_done(self, batch_id: str, point_ids: list[str]) -> None:
        now = _utc_now()
        self._update(batch_id, state=BatchState.DONE,
                     point_ids=point_ids, stage="done", completed_at=now)

    def mark_failed(self, batch_id: str, error: str,
                    retry_count: int, backoff_sec: float) -> None:
        state = BatchState.FAILED if retry_count >= MAX_RETRIES else BatchState.PENDING
        self._update(batch_id, state=state, error=error,
                     retry_count=retry_count, backoff_sec=backoff_sec)

    def _update(
        self,
        batch_id: str,
        state: BatchState | None = None,
        chunk_ids: list | None = None,
        point_ids: list | None = None,
        stage: str | None = None,
        error: str | None = None,
        retry_count: int | None = None,
        backoff_sec: float | None = None,
        completed_at: str | None = None,
    ) -> None:
        conn = self._conn()
        now = _utc_now()
        sets = ["updated_at = ?"]
        args: list[object] = [now]
        if state        is not None:  sets.append("state = ?");         args.append(state.value)
        if chunk_ids    is not None:  sets.append("chunk_ids = ?");     args.append(json.dumps(chunk_ids))
        if point_ids    is not None:  sets.append("point_ids = ?");     args.append(json.dumps(point_ids))
        if stage        is not None:  sets.append("stage = ?");         args.append(stage)
        if error        is not None:  sets.append("error = ?");         args.append(error)
        if retry_count  is not None:  sets.append("retry_count = ?");   args.append(retry_count)
        if backoff_sec  is not None:  sets.append("backoff_sec = ?");   args.append(backoff_sec)
        if completed_at is not None:  sets.append("completed_at = ?");  args.append(completed_at)
        args.append(batch_id)
        conn.execute(f"UPDATE chunk_batches SET {', '.join(sets)} WHERE batch_id = ?", args)
        conn.commit()
        conn.close()

    # ── read ─────────────────────────────────────────────────────────────────

    def get_batch(self, batch_id: str) -> Optional[BatchJob]:
        conn = self._conn()
        row = conn.execute("SELECT * FROM chunk_batches WHERE batch_id = ?",
                           (batch_id,)).fetchone()
        cols = [c[1] for c in conn.execute("PRAGMA table_info(chunk_batches)").fetchall()]
        conn.close()
        if row is None:
            return None
        return _row_to_batch(row, cols)

    def get_queue_stats(self) -> dict[str, Any]:
        conn = self._conn()
        stats = {}
        for state in BatchState:
            cnt = conn.execute(
                "SELECT COUNT(*) FROM chunk_batches WHERE state = ?",
                (state.value,)).fetchone()[0]
            stats[state.value] = cnt
        total = conn.execute("SELECT COUNT(*) FROM chunk_batches").fetchone()[0]
        conn.close()
        stats["total"] = total
        return stats

    def get_stuck_batches(self, stuck_after_sec: int = 600) -> list[BatchJob]:
        conn = self._conn()
        cutoff = _utc_now()
        rows = conn.execute(f"""
            SELECT * FROM chunk_batches
            WHERE state NOT IN ('done', 'failed')
              AND updated_at <= datetime(?, '-' || ? || ' seconds')
        """, (cutoff, stuck_after_sec)).fetchall()
        cols = [c[1] for c in conn.execute("PRAGMA table_info(chunk_batches)").fetchall()]
        conn.close()
        return [_row_to_batch(row, cols) for row in rows]

    def reset_stuck_batches(self, stuck_after_sec: int = 600) -> int:
        conn = self._conn()
        now = _utc_now()
        cur = conn.execute(f"""
            UPDATE chunk_batches
            SET state = 'pending', updated_at = ?, stage = 'pending',
                backoff_sec = MIN(backoff_sec * 1.5 + 10, ?)
            WHERE state NOT IN ('done', 'failed')
              AND updated_at <= datetime(?, '-' || ? || ' seconds')
        """, (now, MAX_BACKOFF, now, stuck_after_sec))
        conn.commit()
        n = cur.rowcount
        conn.close()
        if n:
            logger.warning("Reset %d stuck batches", n)
        return n

    def gc_completed(self, keep: int = 1000) -> int:
        conn = self._conn()
        cur = conn.execute("""
            DELETE FROM chunk_batches
            WHERE state = 'done'
              AND batch_id NOT IN (
                  SELECT batch_id FROM chunk_batches
                  WHERE state = 'done'
                  ORDER BY completed_at DESC LIMIT ?
              )
        """, (keep,))
        conn.commit()
        n = cur.rowcount
        conn.close()
        return n


# ── Import at bottom to avoid circular import ──────────────────────────────────

from hermes_memory_core.store.sqlite import get_memory_store