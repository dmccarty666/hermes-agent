-- Migration 0019: Add retrieval_audit table
-- Purpose: Track every memory query hit for analytics (MEM-019)
-- Enables answering after 30 days:
--   - Which facts get used most
--   - Which modes does the agent rely on
--   - Which facts were never retrieved after being written

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