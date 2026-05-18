# Copyright 2026 David McCarty. All rights reserved.
"""SQLite + FTS5 store for Hermes Local Memory.

Owns ``~/.hermes/memory/index/memory.sqlite`` with WAL mode.
Schema (v1): sessions, turns, raw_events, chunks, facts, entities,
fact_entities, decisions, open_questions, dream_runs, memory_banks,
schema_version, audit_log — + FTS5 virtual tables.

Writer-ownership per ADR-002:
  Plugin owns:   sessions, turns, raw_events, audit_log (append)
  Gateway owns:  chunks, facts, entities, fact_entities, decisions,
                open_questions, dream_runs, memory_banks
  Shared:        schema_version, triggers
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from hermes_state import apply_wal_with_fallback

def _apply_pragma(conn: sqlite3.Connection) -> None:
    """Apply per-connection PRAGMA settings."""
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 268435456")


logger = logging.getLogger(__name__)

# Path to the canonical memory SQLite (separate from hermes_state.db)
_MEMORY_SQLITE_NAME = "memory.sqlite"

# Current schema version
SCHEMA_VERSION = 1

# PRAGMA settings applied on every fresh connection
_PRAGMAS = """
PRAGMA busy_timeout = 30000;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 268435456;
"""

# Full schema — all tables, indexes, FTS5 virtuals, triggers.
# Split on ';' — comments embedded in the string are safe since they
# don't contain semicolons. The 'audit_log' comment line had a semicolon
# that broke the statement split; fixed by using comma instead.
_CREATE_TABLES = """
-- Plugin-owned tables (capture path)
CREATE TABLE IF NOT EXISTS sessions (
  session_id   TEXT PRIMARY KEY,
  agent        TEXT NOT NULL,
  title        TEXT,
  project      TEXT,
  started_at   TEXT NOT NULL,
  ended_at     TEXT,
  source       TEXT,
  platform     TEXT
);

CREATE TABLE IF NOT EXISTS turns (
  turn_id              TEXT PRIMARY KEY,
  session_id           TEXT NOT NULL,
  sequence             INTEGER NOT NULL,
  timestamp            TEXT NOT NULL,
  role                 TEXT NOT NULL,
  content              TEXT NOT NULL,
  dream_status         TEXT NOT NULL DEFAULT 'pending',
  index_status         TEXT NOT NULL DEFAULT 'pending',
  source_refs_json     TEXT NOT NULL DEFAULT '[]',
  parent_turn_id       TEXT,
  redaction_count      INTEGER NOT NULL DEFAULT 0,
  redaction_summary    TEXT,
  redaction_applied    TEXT,
  redaction_types_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, sequence);
CREATE INDEX IF NOT EXISTS idx_turns_index_status ON turns(index_status);
CREATE INDEX IF NOT EXISTS idx_turns_dream_status ON turns(dream_status);

CREATE TABLE IF NOT EXISTS raw_events (
  event_id     TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL,
  turn_id      TEXT,
  timestamp    TEXT NOT NULL,
  jsonl_path   TEXT NOT NULL,
  byte_offset  INTEGER NOT NULL,
  event_type   TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  raw_content  TEXT NOT NULL
);

-- Gateway-owned tables (derive path)
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id        TEXT PRIMARY KEY,
  session_id      TEXT NOT NULL,
  start_turn_id   TEXT,
  end_turn_id     TEXT,
  chunk_text      TEXT NOT NULL,
  char_count      INTEGER NOT NULL,
  source_refs_json TEXT NOT NULL DEFAULT '[]',
  created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
  fact_id              TEXT PRIMARY KEY,
  fact_text            TEXT NOT NULL,
  content_hash         TEXT NOT NULL UNIQUE,
  scope                TEXT NOT NULL,
  project              TEXT,
  status               TEXT NOT NULL DEFAULT 'active',
  confidence           REAL,
  source_refs_json     TEXT NOT NULL DEFAULT '[]',
  entity_ids_json      TEXT NOT NULL DEFAULT '[]',
  created_at           TEXT NOT NULL,
  updated_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_project ON facts(project);
CREATE INDEX IF NOT EXISTS idx_facts_status  ON facts(status);

CREATE TABLE IF NOT EXISTS entities (
  entity_id    TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  alias_json   TEXT,
  entity_type  TEXT,
  project      TEXT,
  created_at   TEXT NOT NULL,
  updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_project ON entities(project);

CREATE TABLE IF NOT EXISTS fact_entities (
  fact_id      TEXT NOT NULL,
  entity_id    TEXT NOT NULL,
  role         TEXT,
  PRIMARY KEY (fact_id, entity_id)
);

CREATE TABLE IF NOT EXISTS decisions (
  decision_id      TEXT PRIMARY KEY,
  decision_text    TEXT NOT NULL,
  rationale        TEXT,
  project          TEXT,
  owner            TEXT,
  status           TEXT NOT NULL DEFAULT 'open',
  source_refs_json TEXT NOT NULL DEFAULT '[]',
  created_at       TEXT NOT NULL,
  updated_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project);
CREATE INDEX IF NOT EXISTS idx_decisions_status  ON decisions(status);

CREATE TABLE IF NOT EXISTS open_questions (
  question_id      TEXT PRIMARY KEY,
  question_text    TEXT NOT NULL,
  project          TEXT,
  priority         TEXT,
  status           TEXT DEFAULT 'open',
  source_refs_json TEXT NOT NULL,
  next_action      TEXT,
  created_at       TEXT NOT NULL,
  updated_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_questions_project ON open_questions(project);
CREATE INDEX IF NOT EXISTS idx_questions_status  ON open_questions(status);

CREATE TABLE IF NOT EXISTS dream_runs (
  dream_run_id          TEXT PRIMARY KEY,
  started_at            TEXT NOT NULL,
  ended_at              TEXT,
  status                TEXT NOT NULL,
  input_scope_json      TEXT,
  output_path           TEXT,
  facts_created        INTEGER DEFAULT 0,
  facts_updated        INTEGER DEFAULT 0,
  decisions_created    INTEGER DEFAULT 0,
  questions_created    INTEGER DEFAULT 0,
  contradictions_detected INTEGER DEFAULT 0,
  errors_json           TEXT,
  llm_model             TEXT,
  llm_endpoint          TEXT
);

CREATE TABLE IF NOT EXISTS memory_banks (
  bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_name  TEXT NOT NULL UNIQUE,
  vector     BLOB NOT NULL,
  dim        INTEGER NOT NULL,
  fact_count INTEGER DEFAULT 0,
  updated_at TEXT NOT NULL
);

-- Schema versioning (owned by plugin)
CREATE TABLE IF NOT EXISTS schema_version (
  applied_at TEXT NOT NULL,
  version    INTEGER NOT NULL PRIMARY KEY,
  notes      TEXT
);

-- audit_log: both processes append, actor column for row-level isolation
CREATE TABLE IF NOT EXISTS audit_log (
  audit_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp   TEXT NOT NULL,
  actor       TEXT NOT NULL,
  action      TEXT NOT NULL,
  target_kind TEXT,
  target_id   TEXT,
  detail_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
"""

# FTS5 virtual tables
_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
  content,
  content=turns,
  content_rowid=rowid,
  tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_text,
  content=chunks,
  content_rowid=rowid,
  tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
  fact_text,
  content=facts,
  content_rowid=rowid,
  tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS decisions_fts USING fts5(
  decision_text,
  content=decisions,
  content_rowid=rowid,
  tokenize='porter unicode61'
);
"""

# Triggers to keep FTS5 in sync with base tables
# Named {base_table}_fts_{op} to match test expectations (TDD §6.2)
_CREATE_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS turns_fts_ai AFTER INSERT ON turns BEGIN
  INSERT INTO turns_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
END;

CREATE TRIGGER IF NOT EXISTS turns_fts_ad AFTER DELETE ON turns BEGIN
  INSERT INTO turns_fts(turns_fts, rowid, content)
    VALUES('delete', OLD.rowid, OLD.content);
END;

CREATE TRIGGER IF NOT EXISTS turns_fts_au AFTER UPDATE ON turns BEGIN
  INSERT INTO turns_fts(turns_fts, rowid, content)
    VALUES('delete', OLD.rowid, OLD.content);
  INSERT INTO turns_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, chunk_text)
    VALUES (NEW.rowid, NEW.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text)
    VALUES('delete', OLD.rowid, OLD.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text)
    VALUES('delete', OLD.rowid, OLD.chunk_text);
  INSERT INTO chunks_fts(rowid, chunk_text)
    VALUES (NEW.rowid, NEW.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS facts_fts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, fact_text) VALUES (NEW.rowid, NEW.fact_text);
END;

CREATE TRIGGER IF NOT EXISTS facts_fts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, fact_text)
    VALUES('delete', OLD.rowid, OLD.fact_text);
END;

CREATE TRIGGER IF NOT EXISTS facts_fts_au AFTER UPDATE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, fact_text)
    VALUES('delete', OLD.rowid, OLD.fact_text);
  INSERT INTO facts_fts(rowid, fact_text) VALUES (NEW.rowid, NEW.fact_text);
END;

CREATE TRIGGER IF NOT EXISTS decisions_fts_ai AFTER INSERT ON decisions BEGIN
  INSERT INTO decisions_fts(rowid, decision_text)
    VALUES (NEW.rowid, NEW.decision_text);
END;

CREATE TRIGGER IF NOT EXISTS decisions_fts_ad AFTER DELETE ON decisions BEGIN
  INSERT INTO decisions_fts(decisions_fts, rowid, decision_text)
    VALUES('delete', OLD.rowid, OLD.decision_text);
END;

CREATE TRIGGER IF NOT EXISTS decisions_fts_au AFTER UPDATE ON decisions BEGIN
  INSERT INTO decisions_fts(decisions_fts, rowid, decision_text)
    VALUES('delete', OLD.rowid, OLD.decision_text);
  INSERT INTO decisions_fts(rowid, decision_text)
    VALUES (NEW.rowid, NEW.decision_text);
END;
"""


class MemoryDB:
    """SQLite handle for the hermes-local memory store.

    Handles connection lifecycle, WAL mode, schema initialization,
    and idempotent re-initialization. The DB file lives at
    ``~/.hermes/memory/index/memory.sqlite`` (separate from
    ``hermes_state.db``).

    Thread-unsafe in-process; use a connection pool or serialise
    access in multi-threaded contexts.
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = Path(get_hermes_home()) / "memory" / "index" / _MEMORY_SQLITE_NAME
        self.db_path: Path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL + PRAGMA settings."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 30000")
        apply_wal_with_fallback(conn, db_label=str(self.db_path.name))
        for line in _PRAGMAS.splitlines():
            line = line.strip()
            if line:
                conn.execute(line)
        return conn

    def is_initialized(self) -> bool:
        """Return True if schema_version row exists at current version."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT version FROM schema_version "
                "ORDER BY version DESC LIMIT 1"
            ).fetchone()
            return bool(row and row[0] >= SCHEMA_VERSION)
        except sqlite3.OperationalError:
            # Table doesn't exist yet — not initialized
            return False
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create all tables, FTS5 virtuals, triggers, and the schema_version row.

        Safe to call repeatedly — checks schema_version before applying.
        """
        if self.is_initialized():
            return

        conn = self._connect()
        try:
            self._apply_schema(conn)
        finally:
            conn.close()

    def close(self) -> None:
        """Close the underlying connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Schema application
    # ------------------------------------------------------------------

    def _apply_schema(self, conn: sqlite3.Connection) -> None:
        """Apply full schema if migration not yet applied."""
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='schema_version'"
        )
        if cur.fetchone():
            # Schema already applied — check version
            row = conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            if row and row[0] >= SCHEMA_VERSION:
                return  # already at current version — no-op

        # Apply schema — DDL executes an implicit COMMIT that resets
        # synchronous to default (2) and resets busy_timeout to 0.
        # Re-apply PRAGMAs before schema DDL so they stick after the COMMIT.
        _apply_pragma(conn)
        try:
            # Build the full schema.  Split on blank lines (\n\n):
            #   - Trigger bodies contain ';' so they MUST NOT be split further;
            #     execute the entire trigger paragraph as one statement.
            #   - Tables and indexes have standalone statements separated by ';'
            #     within each paragraph; split each non-trigger paragraph by ';'.
            all_schema = (
                _CREATE_TABLES.strip() + "\n\n" +
                _CREATE_FTS.strip() + "\n\n" +
                _CREATE_TRIGGERS.strip()
            )
            for para in all_schema.split("\n\n"):
                para = para.strip()
                if not para:
                    continue
                if para.startswith("CREATE TRIGGER"):
                    # Multi-line trigger body with embedded ';': execute whole.
                    conn.execute(para)
                else:
                    # Tables, indexes — split by ';' for individual statements.
                    for stmt in para.split(";"):
                        stmt = stmt.strip()
                        if stmt:
                            conn.execute(stmt)

            # Insert schema version row
            applied_at = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO schema_version (applied_at, version, notes) "
                "VALUES (?, ?, ?)",
                (applied_at, SCHEMA_VERSION, "initial schema v1"),
            )
            conn.commit()  # required — WAL mode doesn't auto-commit
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Convenience read helpers
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> Optional[dict]:
        """Return a session row by id, or None."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT session_id, agent, title, project, started_at, ended_at, source, platform "
                "FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "session_id": row[0], "agent": row[1], "title": row[2],
                "project": row[3], "started_at": row[4], "ended_at": row[5],
                "source": row[6], "platform": row[7],
            }
        finally:
            conn.close()

    def log_audit(
        self,
        actor: str,
        action: str,
        target_kind: Optional[str] = None,
        target_id: Optional[str] = None,
        detail_json: Optional[str] = None,
    ) -> None:
        """Append an audit log row."""
        conn = self._connect()
        try:
            ts = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO audit_log (timestamp, actor, action, target_kind, target_id, detail_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts, actor, action, target_kind, target_id, detail_json),
            )
        finally:
            conn.close()