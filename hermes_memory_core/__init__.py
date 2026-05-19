"""Hermes Local Memory — shared core library.

Modules are importable independently:
    from hermes_memory_core.chunk import chunk_turns
    from hermes_memory_core import get_memory_db, write_memory, MemoryWriteInput
    etc.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# MemoryDB — SQLite handle for the hermes-local memory store
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

_FULL_SCHEMA = """
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
  hrr_vector           BLOB,
  source_refs_json     TEXT NOT NULL DEFAULT '[]',
  entity_ids_json      TEXT NOT NULL DEFAULT '[]',
  tags_json            TEXT NOT NULL DEFAULT '[]',
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
  source_refs_json TEXT NOT NULL DEFAULT '[]',
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

CREATE TABLE IF NOT EXISTS schema_version (
  applied_at TEXT NOT NULL,
  version    INTEGER NOT NULL PRIMARY KEY,
  notes      TEXT
);

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

CREATE TRIGGER IF NOT EXISTS turns_fts_ai AFTER INSERT ON turns BEGIN
  INSERT INTO turns_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
END;

CREATE TRIGGER IF NOT EXISTS turns_fts_ad AFTER DELETE ON turns BEGIN
  INSERT INTO turns_fts(turns_fts, rowid, content) VALUES('delete', OLD.rowid, OLD.content);
END;

CREATE TRIGGER IF NOT EXISTS turns_fts_au AFTER UPDATE ON turns BEGIN
  INSERT INTO turns_fts(turns_fts, rowid, content) VALUES('delete', OLD.rowid, OLD.content);
  INSERT INTO turns_fts(rowid, content) VALUES (NEW.rowid, NEW.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, chunk_text) VALUES (NEW.rowid, NEW.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text) VALUES('delete', OLD.rowid, OLD.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text) VALUES('delete', OLD.rowid, OLD.chunk_text);
  INSERT INTO chunks_fts(rowid, chunk_text) VALUES (NEW.rowid, NEW.chunk_text);
END;

CREATE TRIGGER IF NOT EXISTS facts_fts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, fact_text) VALUES (NEW.rowid, NEW.fact_text);
END;

CREATE TRIGGER IF NOT EXISTS facts_fts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, fact_text) VALUES('delete', OLD.rowid, OLD.fact_text);
END;

CREATE TRIGGER IF NOT EXISTS facts_fts_au AFTER UPDATE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, fact_text) VALUES('delete', OLD.rowid, OLD.fact_text);
  INSERT INTO facts_fts(rowid, fact_text) VALUES (NEW.rowid, NEW.fact_text);
END;

CREATE TRIGGER IF NOT EXISTS decisions_fts_ai AFTER INSERT ON decisions BEGIN
  INSERT INTO decisions_fts(rowid, decision_text) VALUES (NEW.rowid, NEW.decision_text);
END;

CREATE TRIGGER IF NOT EXISTS decisions_fts_ad AFTER DELETE ON decisions BEGIN
  INSERT INTO decisions_fts(decisions_fts, rowid, decision_text) VALUES('delete', OLD.rowid, OLD.decision_text);
END;

CREATE TRIGGER IF NOT EXISTS decisions_fts_au AFTER UPDATE ON decisions BEGIN
  INSERT INTO decisions_fts(decisions_fts, rowid, decision_text) VALUES('delete', OLD.rowid, OLD.decision_text);
  INSERT INTO decisions_fts(rowid, decision_text) VALUES (NEW.rowid, NEW.decision_text);
END;
"""


class MemoryDB:
    """SQLite handle for the hermes-local memory store.

    Handles connection lifecycle, WAL mode, schema initialization,
    and idempotent re-initialization. The DB file lives at
    ``~/.hermes/memory/index/memory.sqlite`` (separate from
    ``hermes_state.db``).
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        if db_path is None:
            try:
                from hermes_constants import get_hermes_home
                db_path = Path(get_hermes_home()) / "memory" / "index" / "memory.sqlite"
            except ImportError:
                db_path = Path.home() / ".hermes" / "memory" / "index" / "memory.sqlite"
        self.db_path: Path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 268435456")
        return conn

    def initialize(self) -> None:
        conn = self._connect()
        try:
            try:
                row = conn.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                ).fetchone()
                if row and row[0] >= SCHEMA_VERSION:
                    return
            except sqlite3.OperationalError:
                pass
            self._apply_schema(conn)
        finally:
            conn.close()

    def _apply_schema(self, conn: sqlite3.Connection) -> None:
        try:
            from hermes_state import apply_wal_with_fallback
            apply_wal_with_fallback(conn, db_label="memory")
        except ImportError:
            pass
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 268435456")

        for para in _FULL_SCHEMA.strip().split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if para.startswith("CREATE TRIGGER"):
                conn.execute(para)
            else:
                for stmt in para.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(stmt)

        conn.execute(
            "INSERT INTO schema_version (applied_at, version, notes) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), SCHEMA_VERSION, "initial"),
        )
        conn.commit()

    def log_audit(
        self,
        actor: str,
        action: str,
        target_kind: Optional[str] = None,
        target_id: Optional[str] = None,
        detail_json: Optional[str] = None,
    ) -> None:
        conn = self._connect()
        try:
            ts = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO audit_log (timestamp, actor, action, target_kind, target_id, detail_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts, actor, action, target_kind, target_id, detail_json),
            )
            conn.commit()
        finally:
            conn.close()

    def health_check(self) -> dict:
        try:
            conn = self._connect()
            try:
                result = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
                return {"status": "ok", "schema_version": result[0] if result else 0}
            finally:
                conn.close()
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_memory_db(db_path: Optional[str | Path] = None) -> MemoryDB:
    """Return a process-wide singleton MemoryDB instance.

    The DB file lives at ``~/.hermes/memory/index/memory.sqlite`` (separate
    from ``hermes_state.db``). Call ``.initialize()`` on the returned
    instance before using it if you need the schema applied.
    """
    global _singleton_db
    if db_path is not None:
        return MemoryDB(db_path)
    if _singleton_db is None:
        _singleton_db = MemoryDB()
    return _singleton_db


_singleton_db: Optional[MemoryDB] = None

# ---------------------------------------------------------------------------
# Re-export from write.pipeline so callers can use a single import
# ---------------------------------------------------------------------------

from hermes_memory_core.write.pipeline import write_memory, update_memory, fact_feedback, MemoryWriteInput

__all__ = [
    "MemoryDB",
    "get_memory_db",
    "write_memory",
    "update_memory",
    "fact_feedback",
    "MemoryWriteInput",
]