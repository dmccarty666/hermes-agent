"""
SQLite store for Hermes Local Memory — schema initialization and CRUD.

Schema is applied lazily on first connection. All write operations
are idempotent where possible (content_hash dedup, dream_status flags).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

HERMES_HOME = os.environ.get(
    "HERMES_HOME",
    str(Path.home() / ".hermes"),
)
MEMORY_BASE = Path(HERMES_HOME) / "memory"
DB_DIR = MEMORY_BASE / "index"
DB_PATH = DB_DIR / "memory.sqlite"

# ── Schema ───────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
-- sessions
CREATE TABLE IF NOT EXISTS sessions (
  session_id   TEXT PRIMARY KEY,
  agent        TEXT NOT NULL,
  title        TEXT,
  project      TEXT,
  started_at   TEXT NOT NULL,
  ended_at     TEXT,
  summary      TEXT,
  qmd_path     TEXT,
  raw_path     TEXT,
  source       TEXT,
  platform     TEXT,
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL
);

-- turns
CREATE TABLE IF NOT EXISTS turns (
  turn_id          TEXT PRIMARY KEY,
  session_id       TEXT NOT NULL,
  sequence         INTEGER NOT NULL,
  timestamp        TEXT NOT NULL,
  role             TEXT NOT NULL,
  content          TEXT NOT NULL,
  raw_content_hash TEXT NOT NULL,
  content_hash     TEXT NOT NULL,
  project          TEXT,
  tags_json        TEXT,
  tool_calls_json  TEXT,
  attachments_json TEXT,
  metadata_json    TEXT,
  parent_turn_id   TEXT,
  index_status     TEXT DEFAULT 'pending',
  dream_status     TEXT DEFAULT 'pending',
  redaction_applied INTEGER DEFAULT 0,
  redaction_types_json TEXT,
  FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, sequence);
CREATE INDEX IF NOT EXISTS idx_turns_index_status ON turns(index_status);
CREATE INDEX IF NOT EXISTS idx_turns_dream_status ON turns(dream_status);

-- raw events
CREATE TABLE IF NOT EXISTS raw_events (
  event_id     TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL,
  turn_id      TEXT,
  timestamp    TEXT NOT NULL,
  jsonl_path   TEXT NOT NULL,
  byte_offset  INTEGER NOT NULL,
  content_hash TEXT NOT NULL
);

-- chunks
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id        TEXT PRIMARY KEY,
  session_id      TEXT NOT NULL,
  start_turn_id   TEXT,
  end_turn_id     TEXT,
  chunk_type      TEXT NOT NULL,
  project         TEXT,
  text            TEXT NOT NULL,
  text_hash       TEXT NOT NULL,
  summary         TEXT,
  source_ref      TEXT NOT NULL,
  qdrant_point_id TEXT,
  embed_model     TEXT,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL,
  UNIQUE(text_hash, embed_model)
);

-- facts
CREATE TABLE IF NOT EXISTS facts (
  fact_id              TEXT PRIMARY KEY,
  fact_text            TEXT NOT NULL,
  content_hash         TEXT NOT NULL UNIQUE,
  scope                TEXT NOT NULL,
  category             TEXT DEFAULT 'general',
  project              TEXT,
  entity               TEXT,
  confidence           REAL,
  trust_score          REAL DEFAULT 0.5,
  status               TEXT DEFAULT 'active',
  first_seen_at        TEXT,
  last_confirmed_at    TEXT,
  source_refs_json     TEXT NOT NULL,
  supersedes_fact_id   TEXT,
  superseded_by_fact_id TEXT,
  tags_json            TEXT,
  retrieval_count      INTEGER DEFAULT 0,
  helpful_count       INTEGER DEFAULT 0,
  hrr_vector          BLOB,
  created_at          TEXT NOT NULL,
  updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_project ON facts(project);
CREATE INDEX IF NOT EXISTS idx_facts_status  ON facts(status);
CREATE INDEX IF NOT EXISTS idx_facts_entity  ON facts(entity);

-- entities
CREATE TABLE IF NOT EXISTS entities (
  entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL,
  entity_type TEXT DEFAULT 'unknown',
  aliases     TEXT DEFAULT '',
  created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);

CREATE TABLE IF NOT EXISTS fact_entities (
  fact_id   TEXT REFERENCES facts(fact_id),
  entity_id INTEGER REFERENCES entities(entity_id),
  PRIMARY KEY (fact_id, entity_id)
);

-- decisions
CREATE TABLE IF NOT EXISTS decisions (
  decision_id         TEXT PRIMARY KEY,
  decision_text       TEXT NOT NULL,
  rationale           TEXT,
  project             TEXT,
  status              TEXT DEFAULT 'active',
  decision_date       TEXT,
  owner               TEXT,
  source_refs_json    TEXT NOT NULL,
  related_fact_ids_json TEXT,
  implications        TEXT,
  created_at          TEXT NOT NULL,
  updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project);

-- open questions
CREATE TABLE IF NOT EXISTS open_questions (
  question_id     TEXT PRIMARY KEY,
  question_text   TEXT NOT NULL,
  project         TEXT,
  priority        TEXT,
  status          TEXT DEFAULT 'open',
  source_refs_json TEXT NOT NULL,
  next_action     TEXT,
  created_at      TEXT NOT NULL,
  updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_questions_project ON open_questions(project);
CREATE INDEX IF NOT EXISTS idx_questions_status  ON open_questions(status);

-- dream runs
CREATE TABLE IF NOT EXISTS dream_runs (
  dream_run_id        TEXT PRIMARY KEY,
  started_at          TEXT NOT NULL,
  ended_at            TEXT,
  status              TEXT NOT NULL,
  input_scope_json    TEXT,
  output_path         TEXT,
  facts_created       INTEGER DEFAULT 0,
  facts_updated       INTEGER DEFAULT 0,
  decisions_created   INTEGER DEFAULT 0,
  questions_created   INTEGER DEFAULT 0,
  contradictions_detected INTEGER DEFAULT 0,
  errors_json         TEXT,
  llm_model           TEXT,
  llm_endpoint        TEXT
);

-- memory banks
CREATE TABLE IF NOT EXISTS memory_banks (
  bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_name  TEXT NOT NULL UNIQUE,
  vector     BLOB NOT NULL,
  dim        INTEGER NOT NULL,
  fact_count INTEGER DEFAULT 0,
  updated_at TEXT NOT NULL
);

-- schema version
CREATE TABLE IF NOT EXISTS schema_version (
  applied_at TEXT NOT NULL,
  version    INTEGER NOT NULL PRIMARY KEY,
  notes      TEXT
);

-- audit log
CREATE TABLE IF NOT EXISTS audit_log (
  audit_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp   TEXT NOT NULL,
  actor       TEXT NOT NULL,
  action      TEXT NOT NULL,
  target_kind TEXT,
  target_id   TEXT,
  detail_json TEXT,
  source_ref  TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);

-- FTS5 virtual tables (all auto-synced via triggers)
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
  content, session_id UNINDEXED, turn_id UNINDEXED, project UNINDEXED, timestamp UNINDEXED,
  content=turns, content_rowid=ROWID
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, chunk_id UNINDEXED, session_id UNINDEXED, project UNINDEXED, source_ref UNINDEXED,
  content=chunks, content_rowid=ROWID
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
  fact_text, fact_id UNINDEXED, project UNINDEXED, entity UNINDEXED,
  content=facts, content_rowid=ROWID
);

-- FTS sync triggers for turns
CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
  INSERT INTO turns_fts(rowid, content, session_id, turn_id, project, timestamp)
  VALUES (new.ROWID, new.content, new.session_id, new.turn_id, new.project, new.timestamp);
END;

CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
  INSERT INTO turns_fts(turns_fts, rowid, content, session_id, turn_id, project, timestamp)
  VALUES ('delete', old.ROWID, old.content, old.session_id, old.turn_id, old.project, old.timestamp);
END;

CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN
  INSERT INTO turns_fts(turns_fts, rowid, content, session_id, turn_id, project, timestamp)
  VALUES ('delete', old.ROWID, old.content, old.session_id, old.turn_id, old.project, old.timestamp);
  INSERT INTO turns_fts(rowid, content, session_id, turn_id, project, timestamp)
  VALUES (new.ROWID, new.content, new.session_id, new.turn_id, new.project, new.timestamp);
END;
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Store class ───────────────────────────────────────────────────────────────


class MemoryStore:
    """
    Thread-safe SQLite store for Hermes Local Memory.

    Lazily initializes the schema on first connection.
    All public methods are safe for concurrent use.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else DB_PATH
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._initialized = False

    # ── connection management ────────────────────────────────────────────────

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            DB_DIR.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), timeout=30.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA_SQL)
            # Migration: add trust_score column to pre-existing DBs that lack it
            try:
                conn.execute("ALTER TABLE facts ADD COLUMN trust_score REAL DEFAULT 0.5")
            except sqlite3.OperationalError:
                pass  # column already exists or table freshly created
            # Record schema version
            try:
                conn.execute(
                    "INSERT INTO schema_version VALUES (?, ?, ?)",
                    (_utc_now(), 1, "initial v0.2 schema"),
                )
            except sqlite3.IntegrityError:
                pass  # already initialized
            self._conn = conn
            self._initialized = True
            logger.info("MemoryStore initialized at %s", self._db_path)

    def _conn_or_init(self) -> sqlite3.Connection:
        self._ensure_init()
        assert self._conn is not None
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
                self._initialized = False

    # ── turns ────────────────────────────────────────────────────────────────

    def get_turns_by_dream_status(
        self,
        status: str = "pending",
        session_id: str | None = None,
        before_timestamp: str | None = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Fetch turns with a specific dream_status, optionally filtered by session
        and/or timestamp. Ordered by (session_id, sequence).

        Used by the dream worker to load unprocessed turns (stage 1).
        """
        conn = self._conn_or_init()
        sql = "SELECT * FROM turns WHERE dream_status = ?"
        args: list[Any] = [status]
        if session_id:
            sql += " AND session_id = ?"
            args.append(session_id)
        if before_timestamp:
            sql += " AND timestamp < ?"
            args.append(before_timestamp)
        sql += " ORDER BY session_id, sequence LIMIT ?"
        args.append(limit)

        rows = conn.execute(sql, args).fetchall()
        pragma_cursor = conn.execute("PRAGMA table_info(turns)")
        cols = [c[1] for c in pragma_cursor.fetchall()]
        return [dict(zip(cols, row)) for row in rows]

    def get_last_dream_run(self) -> Optional[Dict[str, Any]]:
        """Return the most recent dream_runs row (by started_at), or None."""
        conn = self._conn_or_init()
        cursor = conn.execute(
            "SELECT * FROM dream_runs ORDER BY started_at DESC LIMIT 1",
        )
        row = cursor.fetchone()
        if not row:
            return None
        pragma_cursor = conn.execute("PRAGMA table_info(dream_runs)")
        cols = [c[1] for c in pragma_cursor.fetchall()]
        return dict(zip(cols, row))

    def update_turn_dream_status(
        self,
        turn_id: str,
        status: str,
    ) -> None:
        """Mark a single turn's dream_status (idempotent)."""
        conn = self._conn_or_init()
        conn.execute(
            "UPDATE turns SET dream_status = ? WHERE turn_id = ?",
            (status, turn_id),
        )

    def update_turns_dream_status(
        self,
        turn_ids: List[str],
        status: str,
    ) -> None:
        """Batch-update dream_status for multiple turns."""
        if not turn_ids:
            return
        conn = self._conn_or_init()
        placeholders = ",".join("?" * len(turn_ids))
        conn.execute(
            f"UPDATE turns SET dream_status = ? WHERE turn_id IN ({placeholders})",
            [status] + turn_ids,
        )

    def insert_turn_if_not_exists(self, turn: Dict[str, Any]) -> bool:
        """
        Insert a turn row. Returns True if inserted, False if already existed
        (content_hash dedup).
        """
        conn = self._conn_or_init()
        try:
            cols = [
                "turn_id", "session_id", "sequence", "timestamp", "role",
                "content", "raw_content_hash", "content_hash", "project",
                "tags_json", "tool_calls_json", "attachments_json", "metadata_json",
                "parent_turn_id", "index_status", "dream_status",
                "redaction_applied", "redaction_types_json",
            ]
            sql = f"INSERT INTO turns ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
            conn.execute(sql, [turn.get(c) for c in cols])
            return True
        except sqlite3.IntegrityError:
            return False

    def upsert_session(self, session: Dict[str, Any]) -> None:
        """Insert or replace a session row."""
        conn = self._conn_or_init()
        cols = [
            "session_id", "agent", "title", "project", "started_at", "ended_at",
            "summary", "qmd_path", "raw_path", "source", "platform", "created_at", "updated_at",
        ]
        sql = f"""INSERT OR REPLACE INTO sessions
                  ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"""
        conn.execute(sql, [session.get(c) for c in cols])

    # ── dream_runs ────────────────────────────────────────────────────────────

    def create_dream_run(self, dream_run_id: str, scope_json: str, llm_model: str, llm_endpoint: str) -> Dict[str, Any]:
        """Insert a new dream_runs row with status='running'."""
        conn = self._conn_or_init()
        now = _utc_now()
        conn.execute(
            """INSERT INTO dream_runs
               (dream_run_id, started_at, status, input_scope_json, llm_model, llm_endpoint)
               VALUES (?, ?, 'running', ?, ?, ?)""",
            (dream_run_id, now, scope_json, llm_model, llm_endpoint),
        )
        return {
            "dream_run_id": dream_run_id,
            "started_at": now,
            "status": "running",
            "input_scope_json": scope_json,
            "llm_model": llm_model,
            "llm_endpoint": llm_endpoint,
        }

    def complete_dream_run(
        self,
        dream_run_id: str,
        output_path: str,
        facts_created: int = 0,
        facts_updated: int = 0,
        decisions_created: int = 0,
        questions_created: int = 0,
        contradictions_detected: int = 0,
        errors: List[str] | None = None,
    ) -> None:
        """Mark a dream run as completed (or failed if errors present)."""
        conn = self._conn_or_init()
        status = "completed" if not errors else "failed"
        conn.execute(
            """UPDATE dream_runs SET
               ended_at = ?, status = ?, output_path = ?,
               facts_created = ?, facts_updated = ?,
               decisions_created = ?, questions_created = ?,
               contradictions_detected = ?, errors_json = ?
               WHERE dream_run_id = ?""",
            (
                _utc_now(), status, output_path,
                facts_created, facts_updated,
                decisions_created, questions_created,
                contradictions_detected,
                json.dumps(errors or []),
                dream_run_id,
            ),
        )

    # ── facts ──────────────────────────────────────────────────────────────

    def upsert_fact(
        self,
        fact_id: str,
        fact_text: str,
        content_hash: str,
        scope: str,
        source_refs_json: str,
        project: str | None = None,
        entity: str | None = None,
        category: str = "general",
        confidence: float | None = None,
        tags_json: str = "[]",
        supersedes_fact_id: str | None = None,
    ) -> Tuple[str, bool]:
        """
        Insert or update a fact.

        If content_hash already exists, bump last_confirmed_at and return (existing_id, False).
        Otherwise insert and return (fact_id, True).

        When supersedes_fact_id is provided, the superseded fact is marked disputed.
        """
        conn = self._conn_or_init()
        now = _utc_now()

        # Check for existing by content_hash
        existing = conn.execute(
            "SELECT fact_id FROM facts WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE facts SET updated_at = ? WHERE fact_id = ?",
                (now, existing[0]),
            )
            return existing[0], False

        # Mark superseded fact as disputed if applicable
        if supersedes_fact_id:
            conn.execute(
                """UPDATE facts SET status = 'disputed',
                   superseded_by_fact_id = ? WHERE fact_id = ?""",
                (fact_id, supersedes_fact_id),
            )

        conn.execute(
            """INSERT INTO facts
               (fact_id, fact_text, content_hash, scope, project, entity, category,
                confidence, status, created_at, updated_at,
                source_refs_json, tags_json, supersedes_fact_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?,
                       ?, ?, ?)""",
            (
                fact_id, fact_text, content_hash, scope, project, entity, category,
                confidence, now, now,
                source_refs_json, tags_json, supersedes_fact_id,
            ),
        )
        return fact_id, True

    def get_facts_for_contradiction_check(
        self,
        project: str | None = None,
        entity: str | None = None,
        category: str | None = None,
        status: str = "active",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Fetch existing facts for contradiction detection.
        Filters by project/entity/category/status.
        """
        conn = self._conn_or_init()
        sql = "SELECT * FROM facts WHERE status = ?"
        args: list[Any] = [status]
        if project:
            sql += " AND project = ?"
            args.append(project)
        if entity:
            sql += " AND entity = ?"
            args.append(entity)
        if category:
            sql += " AND category = ?"
            args.append(category)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        rows = conn.execute(sql, args).fetchall()
        pragma_cursor = conn.execute("PRAGMA table_info(facts)")
        cols = [c[1] for c in pragma_cursor.fetchall()]
        return [dict(zip(cols, row)) for row in rows]

    def mark_fact_disputed(self, fact_id: str, superseded_by_fact_id: str) -> None:
        """Mark a fact as disputed with a supersession link."""
        conn = self._conn_or_init()
        conn.execute(
            """UPDATE facts SET status = 'disputed', superseded_by_fact_id = ?
               WHERE fact_id = ?""",
            (superseded_by_fact_id, fact_id),
        )

    def get_fact_by_content_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Look up a fact by content_hash."""
        conn = self._conn_or_init()
        row = conn.execute(
            "SELECT * FROM facts WHERE content_hash = ?", (content_hash,),
        ).fetchone()
        if not row:
            return None
        cols = [c[0] for c in conn.execute("PRAGMA table_info(facts)").fetchall()]
        return dict(zip(cols, row))

    # ── decisions ──────────────────────────────────────────────────────────

    def upsert_decision(
        self,
        decision_id: str,
        decision_text: str,
        source_refs_json: str,
        rationale: str | None = None,
        project: str | None = None,
        owner: str | None = None,
    ) -> Tuple[str, bool]:
        """Insert a decision. Returns (id, created)."""
        conn = self._conn_or_init()
        now = _utc_now()
        try:
            conn.execute(
                """INSERT INTO decisions
                   (decision_id, decision_text, rationale, project, source_refs_json,
                    owner, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (decision_id, decision_text, rationale, project, source_refs_json,
                 owner, now, now),
            )
            return decision_id, True
        except sqlite3.IntegrityError:
            return decision_id, False

    # ── open questions ─────────────────────────────────────────────────────

    def upsert_open_question(
        self,
        question_id: str,
        question_text: str,
        source_refs_json: str,
        project: str | None = None,
        priority: str | None = None,
    ) -> Tuple[str, bool]:
        """Insert an open question. Returns (id, created)."""
        conn = self._conn_or_init()
        now = _utc_now()
        try:
            conn.execute(
                """INSERT INTO open_questions
                   (question_id, question_text, project, source_refs_json,
                    priority, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (question_id, question_text, project, source_refs_json,
                 priority, now, now),
            )
            return question_id, True
        except sqlite3.IntegrityError:
            return question_id, False

    # ── sessions ────────────────────────────────────────────────────────────

    def get_sessions_with_pending_turns(self, limit: int = 50) -> List[str]:
        """Return session_ids that have turns with dream_status='pending'."""
        conn = self._conn_or_init()
        rows = conn.execute(
            """SELECT DISTINCT session_id FROM turns
               WHERE dream_status = 'pending'
               ORDER BY (SELECT MIN(timestamp) FROM turns t2 WHERE t2.session_id = turns.session_id)
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_session_turns(
        self,
        session_id: str,
        dream_status: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return all turns for a session, ordered by sequence."""
        conn = self._conn_or_init()
        sql = "SELECT * FROM turns WHERE session_id = ?"
        args: list[Any] = [session_id]
        if dream_status:
            sql += " AND dream_status = ?"
            args.append(dream_status)
        sql += " ORDER BY sequence"
        rows = conn.execute(sql, args).fetchall()
        pragma_cursor = conn.execute("PRAGMA table_info(turns)")
        cols = [c[1] for c in pragma_cursor.fetchall()]
        return [dict(zip(cols, row)) for row in rows]

    # ── query helpers (used by tool handlers) ────────────────────────────────

    def get_facts(
        self,
        project: str | None = None,
        entity: str | None = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch active facts, ordered by trust desc."""
        conn = self._conn_or_init()
        sql = "SELECT * FROM facts WHERE status = 'active'"
        args: list[Any] = []
        if project:
            sql += " AND project = ?"
            args.append(project)
        if entity:
            sql += " AND entity = ?"
            args.append(entity)
        sql += " ORDER BY trust_score DESC LIMIT ?"
        args.append(limit)
        rows = conn.execute(sql, args).fetchall()
        pragma_cursor = conn.execute("PRAGMA table_info(facts)")
        cols = [c[1] for c in pragma_cursor.fetchall()]
        return [dict(zip(cols, row)) for row in rows]

    def get_decisions(
        self,
        project: str | None = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch active decisions, ordered by created_at desc."""
        conn = self._conn_or_init()
        sql = "SELECT * FROM decisions WHERE status = 'active'"
        args: list[Any] = []
        if project:
            sql += " AND project = ?"
            args.append(project)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        rows = conn.execute(sql, args).fetchall()
        pragma_cursor = conn.execute("PRAGMA table_info(decisions)")
        cols = [c[1] for c in pragma_cursor.fetchall()]
        return [dict(zip(cols, row)) for row in rows]

    def get_open_questions(
        self,
        project: str | None = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch open questions, ordered by created_at desc."""
        conn = self._conn_or_init()
        sql = "SELECT * FROM open_questions WHERE status = 'open'"
        args: list[Any] = []
        if project:
            sql += " AND project = ?"
            args.append(project)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        rows = conn.execute(sql, args).fetchall()
        pragma_cursor = conn.execute("PRAGMA table_info(open_questions)")
        cols = [c[1] for c in pragma_cursor.fetchall()]
        return [dict(zip(cols, row)) for row in rows]

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch recent sessions, ordered by started_at desc."""
        conn = self._conn_or_init()
        cursor = conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        pragma_cursor = conn.execute("PRAGMA table_info(sessions)")
        cols = [c[1] for c in pragma_cursor.fetchall()]
        return [dict(zip(cols, row)) for row in rows]

    def get_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single turn by id."""
        conn = self._conn_or_init()
        cursor = conn.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,))
        row = cursor.fetchone()
        if not row:
            return None
        pragma_cursor = conn.execute("PRAGMA table_info(turns)")
        cols = [c[1] for c in pragma_cursor.fetchall()]
        return dict(zip(cols, row))

    def get_dream_run(self, dream_run_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single dream_run by id."""
        conn = self._conn_or_init()
        row = conn.execute(
            "SELECT * FROM dream_runs WHERE dream_run_id = ?", (dream_run_id,)
        ).fetchone()
        if not row:
            return None
        pragma_cursor = conn.execute("PRAGMA table_info(dream_runs)")
        cols = [c[1] for c in pragma_cursor.fetchall()]
        return dict(zip(cols, row))

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        trust_delta: float | None = None,
        tags: str | None = None,
        status: str | None = None,
        category: str | None = None,
    ) -> None:
        """Update a fact's content, trust, tags, status, or category by fact_id."""
        conn = self._conn_or_init()
        now = _utc_now()
        updates: list[str] = []
        args: list[Any] = []
        if content is not None:
            updates.append("fact_text = ?")
            args.append(content)
        if trust_delta is not None:
            updates.append("trust_score = MAX(0.0, MIN(1.0, trust_score + ?))")
            args.append(trust_delta)
        if tags is not None:
            updates.append("tags_json = ?")
            args.append(tags)
        if status is not None:
            updates.append("status = ?")
            args.append(status)
        if category is not None:
            updates.append("category = ?")
            args.append(category)
        if not updates:
            return
        updates.append("updated_at = ?")
        args.append(now)
        args.append(memory_id)
        conn.execute(
            f"UPDATE facts SET {', '.join(updates)} WHERE fact_id = ?",
            args,
        )

    # ── audit ────────────────────────────────────────────────────────────────

    def write_audit(
        self,
        actor: str,
        action: str,
        target_kind: str | None = None,
        target_id: str | None = None,
        detail: Dict[str, Any] | None = None,
        source_ref: str | None = None,
    ) -> None:
        """Append a row to the audit log."""
        conn = self._conn_or_init()
        conn.execute(
            """INSERT INTO audit_log
               (timestamp, actor, action, target_kind, target_id, detail_json, source_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
             (_utc_now(), actor, action, target_kind, target_id,
             json.dumps(detail) if detail else None, source_ref),
        )


# ── MemoryDB (full-schema variant for test compatibility) ───────────────────────
# This class inherits all store methods from MemoryStore but uses _FULL_SCHEMA
# which includes source_refs_json in the turns table (required by tests).
# _FULL_SCHEMA is inlined here to avoid circular imports with __init__.py.

_FULL_SCHEMA_MEMORY_DB = "\n".join([
    "CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, agent TEXT NOT NULL, title TEXT, project TEXT, started_at TEXT NOT NULL, ended_at TEXT, source TEXT, platform TEXT);",
    "CREATE TABLE IF NOT EXISTS turns (turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, sequence INTEGER NOT NULL, timestamp TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, dream_status TEXT NOT NULL DEFAULT 'pending', index_status TEXT NOT NULL DEFAULT 'pending', source_refs_json TEXT NOT NULL DEFAULT '[]', parent_turn_id TEXT, redaction_count INTEGER NOT NULL DEFAULT 0, redaction_summary TEXT, redaction_applied TEXT, redaction_types_json TEXT NOT NULL DEFAULT '[]');",
    "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, sequence);",
    "CREATE INDEX IF NOT EXISTS idx_turns_index_status ON turns(index_status);",
    "CREATE INDEX IF NOT EXISTS idx_turns_dream_status ON turns(dream_status);",
    "CREATE TABLE IF NOT EXISTS raw_events (event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, turn_id TEXT, timestamp TEXT NOT NULL, jsonl_path TEXT NOT NULL, byte_offset INTEGER NOT NULL, event_type TEXT NOT NULL, content_hash TEXT NOT NULL, raw_content TEXT NOT NULL);",
    "CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, start_turn_id TEXT, end_turn_id TEXT, chunk_text TEXT NOT NULL, char_count INTEGER NOT NULL, source_refs_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL);",
    "CREATE TABLE IF NOT EXISTS facts (fact_id TEXT PRIMARY KEY, fact_text TEXT NOT NULL, content_hash TEXT NOT NULL UNIQUE, scope TEXT NOT NULL, category TEXT DEFAULT 'general', project TEXT, entity TEXT, status TEXT NOT NULL DEFAULT 'active', confidence REAL, hrr_vector BLOB, source_refs_json TEXT NOT NULL DEFAULT '[]', entity_ids_json TEXT NOT NULL DEFAULT '[]', tags_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, updated_at TEXT NOT NULL, supersedes_fact_id TEXT);",
    "CREATE INDEX IF NOT EXISTS idx_facts_project ON facts(project);",
    "CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);",
    "CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity);",
    "CREATE TABLE IF NOT EXISTS entities (entity_id TEXT PRIMARY KEY, name TEXT NOT NULL, alias_json TEXT, entity_type TEXT, project TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);",
    "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);",
    "CREATE INDEX IF NOT EXISTS idx_entities_project ON entities(project);",
    "CREATE TABLE IF NOT EXISTS fact_entities (fact_id TEXT NOT NULL, entity_id TEXT NOT NULL, role TEXT, PRIMARY KEY (fact_id, entity_id));",
    "CREATE TABLE IF NOT EXISTS decisions (decision_id TEXT PRIMARY KEY, decision_text TEXT NOT NULL, rationale TEXT, project TEXT, owner TEXT, status TEXT NOT NULL DEFAULT 'open', source_refs_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);",
    "CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project);",
    "CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status);",
    "CREATE TABLE IF NOT EXISTS open_questions (question_id TEXT PRIMARY KEY, question_text TEXT NOT NULL, project TEXT, priority TEXT, status TEXT DEFAULT 'open', source_refs_json TEXT NOT NULL DEFAULT '[]', next_action TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);",
    "CREATE INDEX IF NOT EXISTS idx_questions_project ON open_questions(project);",
    "CREATE INDEX IF NOT EXISTS idx_questions_status ON open_questions(status);",
    "CREATE TABLE IF NOT EXISTS dream_runs (dream_run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT, status TEXT NOT NULL, input_scope_json TEXT, output_path TEXT, facts_created INTEGER DEFAULT 0, facts_updated INTEGER DEFAULT 0, decisions_created INTEGER DEFAULT 0, questions_created INTEGER DEFAULT 0, contradictions_detected INTEGER DEFAULT 0, errors_json TEXT, llm_model TEXT, llm_endpoint TEXT);",
    "CREATE TABLE IF NOT EXISTS memory_banks (bank_id INTEGER PRIMARY KEY AUTOINCREMENT, bank_name TEXT NOT NULL UNIQUE, vector BLOB NOT NULL, dim INTEGER NOT NULL, fact_count INTEGER DEFAULT 0, updated_at TEXT NOT NULL);",
    "CREATE TABLE IF NOT EXISTS schema_version (applied_at TEXT NOT NULL, version INTEGER NOT NULL PRIMARY KEY, notes TEXT);",
    "CREATE TABLE IF NOT EXISTS audit_log (audit_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, target_kind TEXT, target_id TEXT, detail_json TEXT, source_ref TEXT);",
    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);",
    # FTS5 virtual tables (content-less, synced from base tables via triggers).
    "CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(content, content=turns, content_rowid=rowid, tokenize='porter unicode61');",
    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_text, content=chunks, content_rowid=rowid, tokenize='porter unicode61');",
    "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(fact_text, content=facts, content_rowid=rowid, tokenize='porter unicode61');",
    "CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(decision_text, content=decisions, content_rowid=rowid, tokenize='porter unicode61');",
    # FTS sync triggers — keep FTS in lockstep with base tables.
    "CREATE TRIGGER IF NOT EXISTS turns_fts_ai AFTER INSERT ON turns BEGIN INSERT INTO turns_fts(rowid, content) VALUES (NEW.rowid, NEW.content); END;",
    "CREATE TRIGGER IF NOT EXISTS turns_fts_ad AFTER DELETE ON turns BEGIN INSERT INTO turns_fts(turns_fts, rowid, content) VALUES('delete', OLD.rowid, OLD.content); END;",
    "CREATE TRIGGER IF NOT EXISTS turns_fts_au AFTER UPDATE ON turns BEGIN INSERT INTO turns_fts(turns_fts, rowid, content) VALUES('delete', OLD.rowid, OLD.content); INSERT INTO turns_fts(rowid, content) VALUES (NEW.rowid, NEW.content); END;",
    "CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN INSERT INTO chunks_fts(rowid, chunk_text) VALUES (NEW.rowid, NEW.chunk_text); END;",
    "CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text) VALUES('delete', OLD.rowid, OLD.chunk_text); END;",
    "CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text) VALUES('delete', OLD.rowid, OLD.chunk_text); INSERT INTO chunks_fts(rowid, chunk_text) VALUES (NEW.rowid, NEW.chunk_text); END;",
    "CREATE TRIGGER IF NOT EXISTS facts_fts_ai AFTER INSERT ON facts BEGIN INSERT INTO facts_fts(rowid, fact_text) VALUES (NEW.rowid, NEW.fact_text); END;",
    "CREATE TRIGGER IF NOT EXISTS facts_fts_ad AFTER DELETE ON facts BEGIN INSERT INTO facts_fts(facts_fts, rowid, fact_text) VALUES('delete', OLD.rowid, OLD.fact_text); END;",
    "CREATE TRIGGER IF NOT EXISTS facts_fts_au AFTER UPDATE ON facts BEGIN INSERT INTO facts_fts(facts_fts, rowid, fact_text) VALUES('delete', OLD.rowid, OLD.fact_text); INSERT INTO facts_fts(rowid, fact_text) VALUES (NEW.rowid, NEW.fact_text); END;",
    "CREATE TRIGGER IF NOT EXISTS decisions_fts_ai AFTER INSERT ON decisions BEGIN INSERT INTO decisions_fts(rowid, decision_text) VALUES (NEW.rowid, NEW.decision_text); END;",
    "CREATE TRIGGER IF NOT EXISTS decisions_fts_ad AFTER DELETE ON decisions BEGIN INSERT INTO decisions_fts(decisions_fts, rowid, decision_text) VALUES('delete', OLD.rowid, OLD.decision_text); END;",
    "CREATE TRIGGER IF NOT EXISTS decisions_fts_au AFTER UPDATE ON decisions BEGIN INSERT INTO decisions_fts(decisions_fts, rowid, decision_text) VALUES('delete', OLD.rowid, OLD.decision_text); INSERT INTO decisions_fts(rowid, decision_text) VALUES (NEW.rowid, NEW.decision_text); END;",
])


class MemoryDB(MemoryStore):
    """SQLite handle with full schema (source_refs_json in turns table).

    Inherits all store methods from MemoryStore but uses _FULL_SCHEMA so that
    the turns.source_refs_json column exists (required by test fixtures and
    dream-worker source-ref tracking).
    """

    def initialize(self) -> None:
        """Apply the full schema (idempotent). Use this instead of lazy init."""
        self._ensure_init_full_schema()

    def is_initialized(self) -> bool:
        """Return True if the full schema has been applied to this DB.

        Checks for the presence of the ``facts`` table as a proxy for "schema
        present" — that table is created by ``_ensure_init_full_schema`` and
        is required by every search/write path.

        This is a read-only check and works even when the DB has not been
        explicitly connected: if ``_conn`` is None we open a short-lived
        connection just for the schema introspection and close it again.
        """
        import sqlite3
        # Use the live connection when present (cheap, no setup cost).
        if self._conn is not None:
            try:
                cur = self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='facts'"
                )
                return cur.fetchone() is not None
            except Exception:
                return False
        # Otherwise probe the file directly without mutating instance state.
        if not Path(self._db_path).exists():
            return False
        try:
            tmp = sqlite3.connect(str(self._db_path))
            try:
                cur = tmp.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='facts'"
                )
                return cur.fetchone() is not None
            finally:
                tmp.close()
        except Exception:
            return False

    def _ensure_init_full_schema(self) -> None:
        """Initialize with _FULL_SCHEMA (thread-safe, idempotent)."""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            DB_DIR.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), timeout=30.0)
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA mmap_size = 268435456")
            # Pre-schema migrations: add columns to pre-existing tables that
            # lack them, so the CREATE INDEX statements in _FULL_SCHEMA_MEMORY_DB
            # don't raise "no such column" on legacy DBs. These ALTERs are
            # idempotent — sqlite raises OperationalError when the column
            # already exists (or the table doesn't yet), which we swallow.
            _pre_schema_migrations = (
                "ALTER TABLE entities ADD COLUMN project TEXT",
                "ALTER TABLE entities ADD COLUMN alias_json TEXT",
                "ALTER TABLE entities ADD COLUMN updated_at TEXT",
                "ALTER TABLE facts ADD COLUMN trust_score REAL DEFAULT 0.5",
            )
            for stmt in _pre_schema_migrations:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # column already exists or table not yet created
            conn.executescript(_FULL_SCHEMA_MEMORY_DB)
            self._conn = conn
            self._initialized = True

    def _connect(self) -> sqlite3.Connection:
        """Return a raw connection (for test fixtures that execute raw SQL).

        If the owned connection has uncommitted changes (open transaction),
        commit them first so the new connection can see the data.
        """
        self._ensure_init_full_schema()
        # Flush any uncommitted writes from the owned connection
        owned = getattr(self, "_conn", None)
        if owned is not None and owned.in_transaction:
            owned.commit()
        return sqlite3.connect(str(self._db_path), timeout=30.0)

    def log_audit(
        self,
        actor: str,
        action: str,
        target_kind: Optional[str] = None,
        target_id: Optional[str] = None,
        detail_json: Optional[str] = None,
        source_ref: Optional[str] = None,
    ) -> None:
        """Log an audit event."""
        conn = self._connect()
        try:
            ts = datetime.now(timezone.utc).isoformat()
            conn.execute(
                ("INSERT INTO audit_log (timestamp, actor, action, target_kind, target_id, detail_json, source_ref) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?)"),
                (ts, actor, action, target_kind, target_id, detail_json, source_ref),
            )
            conn.commit()
        finally:
            conn.close()

    def health_check(self) -> dict:
        """Return health status."""
        try:
            conn = self._connect()
            try:
                result = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
                return {"status": "ok", "schema_version": result[0] if result else 0}
            finally:
                conn.close()
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ── Module-level singleton ─────────────────────────────────────────────────────

_store: MemoryStore | None = None
_store_lock = threading.Lock()


def get_memory_store(db_path: Path | str | None = None) -> MemoryStore:
    """Return a process-wide MemoryStore singleton."""
    global _store
    with _store_lock:
        if _store is None:
            _store = MemoryStore(db_path)
        return _store
