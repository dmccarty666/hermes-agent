"""
SQLite store for Hermes Local Memory — schema initialization and CRUD.

Schema is applied lazily on first connection. All write operations
are idempotent where possible (content_hash dedup, dream_status flags).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import sqlite3
import threading
import uuid
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

-- entities (unified schema — canonical: MemoryDB)
CREATE TABLE IF NOT EXISTS entities (
  entity_id     TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  alias_json    TEXT DEFAULT '[]',
  entity_type   TEXT,
  entity_subtype TEXT,
  project       TEXT,
  created_at    TEXT NOT NULL,
  updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_entities_name    ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_project ON entities(project);

-- fact_entities (unified schema — canonical: MemoryDB)
CREATE TABLE IF NOT EXISTS fact_entities (
  fact_id   TEXT NOT NULL,
  entity_id TEXT NOT NULL REFERENCES entities(entity_id),
  role      TEXT DEFAULT 'mentioned',
  PRIMARY KEY (fact_id, entity_id, role)
);

-- entity_relations (new graph edge table)
CREATE TABLE IF NOT EXISTS entity_relations (
  relation_id       TEXT PRIMARY KEY,
  source_entity_id  TEXT NOT NULL REFERENCES entities(entity_id),
  target_entity_id  TEXT NOT NULL REFERENCES entities(entity_id),
  relation_type     TEXT NOT NULL,
  source_ref        TEXT,
  confidence        REAL DEFAULT 0.5,
  created_at        TEXT NOT NULL,
  UNIQUE(source_entity_id, target_entity_id, relation_type)
);
CREATE INDEX IF NOT EXISTS idx_relations_source ON entity_relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON entity_relations(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_type   ON entity_relations(relation_type);

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

-- fact_links (cross-fact dependency graph — MEM-016)
CREATE TABLE IF NOT EXISTS fact_links (
  link_id    TEXT PRIMARY KEY,
  fact_id_a  TEXT NOT NULL REFERENCES facts(fact_id),
  fact_id_b  TEXT NOT NULL REFERENCES facts(fact_id),
  link_type  TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(fact_id_a, fact_id_b)
);
CREATE INDEX IF NOT EXISTS idx_fact_links_a ON fact_links(fact_id_a);
CREATE INDEX IF NOT EXISTS idx_fact_links_b ON fact_links(fact_id_b);

-- entity lifecycle (G3.4)
CREATE TABLE IF NOT EXISTS entity_lifecycle (
    entity_name   TEXT PRIMARY KEY,
    status        TEXT NOT NULL DEFAULT 'active',
    first_seen_at TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL,
    archived_at   TEXT,
    revived_at    TEXT,
    mention_count INTEGER DEFAULT 1,
    source_ref    TEXT DEFAULT ''
);

-- retrieval_audit (MEM-019): track every memory query hit for analytics
CREATE TABLE IF NOT EXISTS retrieval_audit (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  query      TEXT NOT NULL,
  mode       TEXT NOT NULL,
  fact_id    TEXT NOT NULL,
  score      REAL NOT NULL,
  hit_rank   INTEGER NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_audit_session ON retrieval_audit(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_fact    ON retrieval_audit(fact_id);
CREATE INDEX IF NOT EXISTS idx_audit_mode    ON retrieval_audit(mode);
CREATE INDEX IF NOT EXISTS idx_audit_created  ON retrieval_audit(created_at);

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
        # Prevent cross-process races on first init: acquire an exclusive
        # advisory lock on a sentinel file before doing any schema work.
        # LOCK_EX blocks until the lock is held; LOCK_UN is implicit on close.
        lock_path = str(self._db_path) + ".lock"
        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            if self._initialized:
                return
            with self._lock:
                if self._initialized:
                    return
                DB_DIR.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(self._db_path), timeout=30.0, isolation_level=None)
                conn.execute("PRAGMA journal_mode=WAL")
                # Verify WAL actually took — some filesystems (NFS, docker overlay)
                # silently reject WAL and fall back to rollback mode, which allows
                # concurrent writes to corrupt data. Detect this immediately.
                cur = conn.execute("PRAGMA journal_mode")
                mode = cur.fetchone()[0]
                if mode != "wal":
                    raise RuntimeError(
                        f"WAL mode requested but filesystem returned '{mode}'. "
                        "Concurrent writes may corrupt data. Check filesystem capabilities "
                        "or switch to a native filesystem for the memory DB path."
                    )
                conn.execute("PRAGMA foreign_keys=ON")
                conn.executescript(_SCHEMA_SQL)

                # ── Idempotent migrations for pre-existing DBs ─────────────────────
                # These ALTER TABLE statements are safe to re-run: SQLite raises
                # OperationalError when the column already exists (or the table
                # doesn't yet), which we swallow.
                _migrations = (
                    # entities: add missing columns from unified schema
                    "ALTER TABLE entities ADD COLUMN alias_json TEXT DEFAULT '[]'",
                    "ALTER TABLE entities ADD COLUMN entity_subtype TEXT",
                    "ALTER TABLE entities ADD COLUMN project TEXT",
                    "ALTER TABLE entities ADD COLUMN updated_at TEXT",
                    # facts: trust_score (was missing in some versions)
                    "ALTER TABLE facts ADD COLUMN trust_score REAL DEFAULT 0.5",
                    # fact_entities: add role column
                    "ALTER TABLE fact_entities ADD COLUMN role TEXT DEFAULT 'mentioned'",
                    # MEM-012: sessions.episode_id (episodes table already in _SCHEMA_SQL)
                    "ALTER TABLE sessions ADD COLUMN episode_id TEXT",
                    # MEM-018: per-fact decay rate override
                    "ALTER TABLE facts ADD COLUMN decay_rate_days REAL",
                    # MEM-019: retrieval_audit indexes already in _SCHEMA_SQL; just ensure table present via IF NOT EXISTS
                )
                for stmt in _migrations:
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError:
                        pass  # column already exists or table freshly created

                # Ensure entity_relations exists (may already exist from _SCHEMA_SQL)
                try:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS entity_relations ("
                        "  relation_id       TEXT PRIMARY KEY,"
                        "  source_entity_id TEXT NOT NULL REFERENCES entities(entity_id),"
                        "  target_entity_id TEXT NOT NULL REFERENCES entities(entity_id),"
                        "  relation_type    TEXT NOT NULL,"
                        "  source_ref       TEXT,"
                        "  confidence       REAL DEFAULT 0.5,"
                        "  created_at       TEXT NOT NULL,"
                        "  UNIQUE(source_entity_id, target_entity_id, relation_type)"
                        ")"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_relations_source "
                        "ON entity_relations(source_entity_id)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_relations_target "
                        "ON entity_relations(target_entity_id)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_relations_type "
                        "ON entity_relations(relation_type)"
                    )
                except sqlite3.OperationalError:
                    pass  # table already exists

                # Record schema version
                try:
                    conn.execute(
                        "INSERT INTO schema_version VALUES (?, ?, ?)",
                        (_utc_now(), 2, "unified entity/fact_entities schema + entity_relations"),
                    )
                except sqlite3.IntegrityError:
                    pass  # already initialized
                self._conn = conn
                self._initialized = True
                logger.info("MemoryStore initialized at %s", self._db_path)
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()

    def _conn_or_init(self) -> sqlite3.Connection:
        self._ensure_init()
        with self._lock:
            # Re-check after acquiring the lock: another thread may have
            # called close() between our first check and acquiring the lock.
            assert self._conn is not None, "connection was closed during _conn_or_init"
            return self._conn

    def _connect(self) -> sqlite3.Connection:
        """Return a short-lived raw connection.

        Some search paths (FTS5, hybrid trust-score lookup, indexer) expect to
        get their own sqlite3 connection so they can run raw SQL with custom
        row factories or commits without touching this store's owned
        connection. We flush any pending writes on the owned connection so
        the new handle sees consistent data, then return a fresh connection
        the caller is responsible for closing.

        Mirrors ``MemoryDB._connect()`` so any code that received a base
        ``MemoryStore`` (e.g. via ``get_memory_store()``) still works without
        having to know which concrete subclass it holds.
        """
        # Skip _ensure_init() — self._conn may be closed (caller closed it).
        # _connect() returns a brand-new independent connection; callers are
        # responsible for closing it. The store's owned conn does not need to be
        # flushed because we don't use it here.
        owned = getattr(self, "_conn", None)
        try:
            if owned is not None and not owned.in_transaction:
                owned.commit()
        except sqlite3.Error:
            pass  # owned conn was closed or otherwise unusable; ignore
        return sqlite3.connect(str(self._db_path), timeout=30.0)

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
        entity_id: str | None = None,
        entity_role: str = "mentioned",
    ) -> Tuple[str, bool]:
        """
        Insert or update a fact.

        If content_hash already exists, bump last_confirmed_at and return (existing_id, False).
        Otherwise insert and return (fact_id, True).

        When supersedes_fact_id is provided, the superseded fact is marked disputed.

        When entity_id is provided, a link is created in fact_entities with the
        specified role (default 'mentioned').
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

        # Link entity to fact if entity_id provided
        if entity_id:
            conn.execute(
                """INSERT INTO fact_entities (fact_id, entity_id, role)
                   VALUES (?, ?, ?)""",
                (fact_id, entity_id, entity_role),
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

    def get_fact_by_id(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Look up a single fact by its fact_id."""
        conn = self._conn_or_init()
        row = conn.execute(
            "SELECT * FROM facts WHERE fact_id = ?", (fact_id,),
        ).fetchone()
        if not row:
            return None
        cols = [c[0] for c in conn.execute("PRAGMA table_info(facts)").fetchall()]
        return dict(zip(cols, row))

    # ── entity-fact links ───────────────────────────────────────────────────────

    def upsert_entity_for_fact(
        self,
        fact_id: str,
        entity_id: str,
        role: str = "mentioned",
    ) -> None:
        """
        Link an entity to a fact with the specified role.

        The (fact_id, entity_id, role) combination must be unique; duplicates raise
        IntegrityError.
        """
        conn = self._conn_or_init()
        conn.execute(
            """INSERT INTO fact_entities (fact_id, entity_id, role)
               VALUES (?, ?, ?)""",
            (fact_id, entity_id, role),
        )

    def upsert_fact_link(
        self,
        fact_id_a: str,
        fact_id_b: str,
        link_type: str,
    ) -> None:
        """
        Insert a cross-fact link. Safe to call multiple times — UNIQUE constraint
        on (fact_id_a, fact_id_b) means duplicates are silently ignored.
        link_type examples: 'same_turn', 'causal', 'contradicts'.
        """
        import uuid
        conn = self._conn_or_init()
        link_id = f"fl:{uuid.uuid4().hex[:16]}"
        now = _utc_now()
        conn.execute(
            """INSERT OR IGNORE INTO fact_links
               (link_id, fact_id_a, fact_id_b, link_type, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (link_id, fact_id_a, fact_id_b, link_type, now),
        )

    def get_fact_links(self, fact_id: str) -> List[Dict[str, Any]]:
        """
        Return all fact_links involving fact_id (as fact_id_a or fact_id_b).
        Returns list of dicts with: link_id, fact_id_a, fact_id_b, link_type, created_at.
        """
        conn = self._conn_or_init()
        rows = conn.execute(
            """SELECT link_id, fact_id_a, fact_id_b, link_type, created_at
               FROM fact_links
               WHERE fact_id_a = ? OR fact_id_b = ?""",
            (fact_id, fact_id),
        ).fetchall()
        cols = ["link_id", "fact_id_a", "fact_id_b", "link_type", "created_at"]
        return [dict(zip(cols, row)) for row in rows]

    def upsert_entity(
        self,
        name: str,
        entity_type: str = "unknown",
        aliases: List[str] | None = None,
        project: str | None = None,
    ) -> str:
        """
        Insert or update an entity by name (case-insensitive dedup).
        Returns the entity_id (TEXT primary key).
        """
        conn = self._conn_or_init()
        now = _utc_now()
        aliases = aliases if aliases is not None else []
        alias_json = json.dumps(aliases)

        # Case-insensitive lookup
        row = conn.execute(
            "SELECT entity_id FROM entities WHERE LOWER(name)=LOWER(?)",
            (name,),
        ).fetchone()

        if row:
            entity_id = row[0]
            conn.execute(
                """UPDATE entities
                   SET alias_json = ?, entity_type = ?, updated_at = ?
                   WHERE entity_id = ?""",
                (alias_json, entity_type, now, entity_id),
            )
            return entity_id
        else:
            entity_id = f"ent:{uuid.uuid4().hex[:16]}"
            conn.execute(
                """INSERT INTO entities
                   (entity_id, name, alias_json, entity_type, project, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (entity_id, name, alias_json, entity_type, project, now, now),
            )
            return entity_id

    def _resolve_name_to_entity_id(self, name: str) -> str | None:
        """
        Resolve an entity name to its entity_id.

        Returns the entity_id if found, or None if no entity with that name exists.
        """
        conn = self._conn_or_init()
        row = conn.execute(
            "SELECT entity_id FROM entities WHERE LOWER(name)=LOWER(?)",
            (name,),
        ).fetchone()
        return row[0] if row else None

    def upsert_entity_relation(
        self,
        relation_id: str,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str,
        source_ref: str | None = None,
        confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Insert or update an entity relation.

        If (source_entity_id, target_entity_id, relation_type) already exists,
        returns the existing row without inserting. Otherwise inserts and returns
        the new row.
        """
        conn = self._conn_or_init()
        now = _utc_now()

        # Check for existing relation
        existing = conn.execute(
            """SELECT relation_id, source_entity_id, target_entity_id,
                      relation_type, source_ref, confidence, created_at
               FROM entity_relations
               WHERE source_entity_id = ? AND target_entity_id = ? AND relation_type = ?""",
            (source_entity_id, target_entity_id, relation_type),
        ).fetchone()

        if existing:
            cols = ["relation_id", "source_entity_id", "target_entity_id",
                    "relation_type", "source_ref", "confidence", "created_at"]
            return dict(zip(cols, existing))

        conn.execute(
            """INSERT INTO entity_relations (relation_id, source_entity_id, target_entity_id,
                                             relation_type, source_ref, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (relation_id, source_entity_id, target_entity_id, relation_type,
             source_ref, confidence, now),
        )
        return {
            "relation_id": relation_id,
            "source_entity_id": source_entity_id,
            "target_entity_id": target_entity_id,
            "relation_type": relation_type,
            "source_ref": source_ref,
            "confidence": confidence,
            "created_at": now,
        }

    def get_entity_relations(
        self,
        entity_id: str,
        relation_type: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all relations involving an entity (as source or target).

        If relation_type is provided, filter to that type only.
        """
        conn = self._conn_or_init()
        sql = """SELECT relation_id, source_entity_id, target_entity_id,
                        relation_type, source_ref, confidence, created_at
                 FROM entity_relations
                 WHERE source_entity_id = ? OR target_entity_id = ?"""
        args: list[Any] = [entity_id, entity_id]
        if relation_type:
            sql += " AND relation_type = ?"
            args.append(relation_type)
        sql += " ORDER BY created_at"
        rows = conn.execute(sql, args).fetchall()
        cols = ["relation_id", "source_entity_id", "target_entity_id",
                "relation_type", "source_ref", "confidence", "created_at"]
        return [dict(zip(cols, row)) for row in rows]

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

    # ── entity lifecycle (G3.4) ─────────────────────────────────────────────────

    def upsert_lifecycle(self, entity_name: str, source_ref: str = "") -> None:
        """
        Insert or update an entity lifecycle record.

        - If entity is new: insert with status='active'
        - If entity exists and status='archived': transition to 'revived'
        - Always update last_seen_at and increment mention_count
        """
        conn = self._conn_or_init()
        now = _utc_now()

        existing = conn.execute(
            "SELECT status, mention_count FROM entity_lifecycle WHERE entity_name = ?",
            (entity_name,),
        ).fetchone()

        if existing:
            old_status = existing[0]
            new_status = "revived" if old_status == "archived" else old_status
            revived_at = now if new_status == "revived" else None
            conn.execute(
                """UPDATE entity_lifecycle
                   SET last_seen_at = ?, mention_count = mention_count + 1,
                       status = ?, revived_at = COALESCE(?, revived_at),
                       source_ref = COALESCE(NULLIF(?, ''), source_ref)
                   WHERE entity_name = ?""",
                (now, new_status, revived_at, source_ref, entity_name),
            )
        else:
            conn.execute(
                """INSERT INTO entity_lifecycle
                   (entity_name, status, first_seen_at, last_seen_at, mention_count, source_ref)
                   VALUES (?, 'active', ?, ?, 1, ?)""",
                (entity_name, now, now, source_ref),
            )

    def archive_stale_entities(self, days: int = 30) -> int:
        """
        Mark entities as 'archived' if not seen in the last `days` days.
        Returns the number of entities archived.
        """
        conn = self._conn_or_init()
        now = _utc_now()

        # Archive entities with last_seen_at older than `days` days that are still 'active'
        cutoff = conn.execute(
            "SELECT datetime(?, '-' || ? || ' days')",
            (now, str(days)),
        ).fetchone()[0]

        cursor = conn.execute(
            """UPDATE entity_lifecycle
               SET status = 'archived', archived_at = ?
               WHERE status = 'active' AND last_seen_at < ?""",
            (now, cutoff),
        )
        return cursor.rowcount

    def get_active_entities(self, limit: int = 20) -> List[str]:
        """Return entity names with status='active' ordered by last_seen_at desc."""
        conn = self._conn_or_init()
        rows = conn.execute(
            """SELECT entity_name FROM entity_lifecycle
               WHERE status = 'active'
               ORDER BY last_seen_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_archived_entities(self, limit: int = 20) -> List[str]:
        """Return entity names with status='archived' ordered by archived_at desc."""
        conn = self._conn_or_init()
        rows = conn.execute(
            """SELECT entity_name FROM entity_lifecycle
               WHERE status = 'archived'
               ORDER BY archived_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_revived_entities(self, limit: int = 20) -> List[str]:
        """Return entity names with status='revived' (revived within last 7 days)."""
        conn = self._conn_or_init()
        from datetime import datetime, timezone, timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        rows = conn.execute(
            """SELECT entity_name FROM entity_lifecycle
               WHERE status = 'revived' AND revived_at > ?
               ORDER BY revived_at DESC LIMIT ?""",
            (cutoff, limit),
        ).fetchall()
        return [r[0] for r in rows]

    def get_entities_by_status(
        self, status: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return full lifecycle rows for a given status."""
        conn = self._conn_or_init()
        rows = conn.execute(
            """SELECT entity_name, status, first_seen_at, last_seen_at,
                      archived_at, revived_at, mention_count, source_ref
               FROM entity_lifecycle WHERE status = ?
               ORDER BY last_seen_at DESC LIMIT ?""",
            (status, limit),
        ).fetchall()
        cols = ["entity_name", "status", "first_seen_at", "last_seen_at",
                "archived_at", "revived_at", "mention_count", "source_ref"]
        return [dict(zip(cols, row)) for row in rows]

    def gc_entity_lifecycle(self) -> int:
        """
        Remove entity lifecycle entries older than 365 days.
        Returns the number of rows deleted.
        """
        conn = self._conn_or_init()
        now = conn.execute("SELECT datetime('now', '-365 days')").fetchone()[0]
        cursor = conn.execute(
            "DELETE FROM entity_lifecycle WHERE last_seen_at < ?",
            (now,),
        )
        return cursor.rowcount

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
                # entities: add missing columns from unified schema
                "ALTER TABLE entities ADD COLUMN project TEXT",
                "ALTER TABLE entities ADD COLUMN alias_json TEXT",
                "ALTER TABLE entities ADD COLUMN entity_subtype TEXT",
                "ALTER TABLE entities ADD COLUMN updated_at TEXT",
                # facts: trust_score (was missing in some versions)
                "ALTER TABLE facts ADD COLUMN trust_score REAL DEFAULT 0.5",
                # fact_entities: add role column
                "ALTER TABLE fact_entities ADD COLUMN role TEXT DEFAULT 'mentioned'",
            )
            for stmt in _pre_schema_migrations:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # column already exists or table not yet created

            # Ensure entity_relations exists (may already exist from _FULL_SCHEMA)
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS entity_relations ("
                    "  relation_id       TEXT PRIMARY KEY,"
                    "  source_entity_id TEXT NOT NULL REFERENCES entities(entity_id),"
                    "  target_entity_id TEXT NOT NULL REFERENCES entities(entity_id),"
                    "  relation_type    TEXT NOT NULL,"
                    "  source_ref       TEXT,"
                    "  confidence       REAL DEFAULT 0.5,"
                    "  created_at       TEXT NOT NULL,"
                    "  UNIQUE(source_entity_id, target_entity_id, relation_type)"
                    ")"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_source "
                    "ON entity_relations(source_entity_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_target "
                    "ON entity_relations(target_entity_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_relations_type "
                    "ON entity_relations(relation_type)"
                )
            except sqlite3.OperationalError:
                pass  # table already exists
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

    def audit_hit(
        self,
        session_id: str,
        query: str,
        mode: str,
        fact_id: str,
        score: float,
        hit_rank: int,
    ) -> None:
        """Record that a memory query returned a specific fact as a hit.

        Used for the retrieval analytics dashboard (MEM-019).
        """
        conn = self._conn_or_init()
        conn.execute(
            """INSERT INTO retrieval_audit (session_id, query, mode, fact_id, score, hit_rank)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, query, mode, fact_id, score, hit_rank),
        )
        conn.commit()

    def get_memory_stats(self, days: int = 30) -> dict:
        """Return memory usage analytics for the last N days.

        Returns:
            dict with keys:
              - total_queries: total query count
              - queries_by_mode: list of {mode, count} sorted by count desc
              - top_facts: list of {fact_id, hit_count} for most-retrieved facts
              - never_retrieved: list of {fact_id, fact_text, created_at} for
                facts written more than 7 days ago that were never retrieved
              - avg_latency_ms: average latency in ms across audit rows
              - hit_rate: percentage of queries that returned at least one result
        """
        conn = self._conn_or_init()
        cur = conn.cursor()

        # queries by mode
        cur.execute(
            """SELECT mode, COUNT(*) as cnt
               FROM retrieval_audit
               WHERE created_at >= datetime('now', '-' || ? || ' days')
               GROUP BY mode ORDER BY cnt DESC""",
            (days,),
        )
        queries_by_mode = [{"mode": r[0], "count": r[1]} for r in cur.fetchall()]

        # total queries
        cur.execute(
            """SELECT COUNT(*) FROM retrieval_audit
               WHERE created_at >= datetime('now', '-' || ? || ' days')""",
            (days,),
        )
        row = cur.fetchone()
        total_queries = row[0] if row else 0

        # top facts
        cur.execute(
            """SELECT fact_id, COUNT(*) as hits
               FROM retrieval_audit
               WHERE created_at >= datetime('now', '-' || ? || ' days')
               GROUP BY fact_id ORDER BY hits DESC LIMIT 20""",
            (days,),
        )
        top_facts = [{"fact_id": r[0], "hit_count": r[1]} for r in cur.fetchall()]

        # never retrieved: facts written > 7 days ago and never in retrieval_audit
        cur.execute(
            """SELECT f.fact_id, f.fact_text, f.created_at
               FROM facts f
               WHERE f.created_at < datetime('now', '-7 days')
                 AND f.status = 'active'
                 AND NOT EXISTS (
                     SELECT 1 FROM retrieval_audit ra WHERE ra.fact_id = f.fact_id
                 )
               ORDER BY f.created_at DESC LIMIT 50"""
        )
        never_retrieved = [
            {"fact_id": r[0], "fact_text": r[1], "created_at": r[2]}
            for r in cur.fetchall()
        ]

        return {
            "total_queries": total_queries,
            "queries_by_mode": queries_by_mode,
            "top_facts": top_facts,
            "never_retrieved": never_retrieved,
            "avg_latency_ms": _avg_latency_ms(conn, days),
            "hit_rate": _hit_rate(conn, days),
        }

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


# ── Helper functions for get_memory_stats ─────────────────────────────────────

def _avg_latency_ms(conn, days: int) -> float | None:
    """Compute average latency_ms across all audit rows in the period."""
    cur = conn.cursor()
    try:
        cur.execute(
            """SELECT AVG(latency_ms) FROM retrieval_audit
               WHERE created_at >= datetime('now', '-' || ? || ' days')
                 AND latency_ms IS NOT NULL""",
            (days,),
        )
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else None
    except sqlite3.OperationalError:
        # latency_ms column may not exist in older schemas
        return None


def _hit_rate(conn, days: int) -> float | None:
    """Compute percentage of queries that returned at least one result.

    hit_rate = (queries with at least one result) / (total queries) * 100
    """
    cur = conn.cursor()
    cur.execute(
        """SELECT COUNT(DISTINCT session_id || query || mode)
           FROM retrieval_audit
           WHERE created_at >= datetime('now', '-' || ? || ' days')
             AND fact_id IS NOT NULL""",
        (days,),
    )
    row_with_hits = cur.fetchone()
    queries_with_hits = row_with_hits[0] if row_with_hits else 0

    cur.execute(
        """SELECT COUNT(DISTINCT session_id || query || mode)
           FROM retrieval_audit
           WHERE created_at >= datetime('now', '-' || ? || ' days')""",
        (days,),
    )
    row_total = cur.fetchone()
    total_queries = row_total[0] if row_total else 0

    if total_queries == 0:
        return None
    return round((queries_with_hits / total_queries) * 100, 2)


# ── Module-level singleton ─────────────────────────────────────────────────────

_store: MemoryDB | None = None
_store_lock = threading.Lock()


def get_memory_store(db_path: Path | str | None = None) -> MemoryDB:
    """Return a process-wide MemoryDB singleton (MEM-019: audit_hit/get_memory_stats on MemoryDB)."""
    global _store
    with _store_lock:
        if _store is None:
            _store = MemoryDB(db_path)
        return _store
