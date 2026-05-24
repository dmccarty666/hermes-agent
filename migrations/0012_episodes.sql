-- Migration 0012: Add episodes table and sessions.episode_id column
-- Purpose: Group sessions into episodic narratives for memory dashboard
-- MEM-012: Episodic session grouping

-- Create episodes table
-- An episode is a named grouping of related sessions (a "conversation thread" or "project episode")
CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    description TEXT,
    source      TEXT NOT NULL,
    user_id     TEXT,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    session_count INTEGER DEFAULT 0,
    message_count INTEGER DEFAULT 0,
    token_count   INTEGER DEFAULT 0,
    created_at  REAL NOT NULL DEFAULT (unixepoch())
);

-- Index for efficient episode lookups by source and time
CREATE INDEX IF NOT EXISTS idx_episodes_source    ON episodes(source);
CREATE INDEX IF NOT EXISTS idx_episodes_started  ON episodes(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_user_id   ON episodes(user_id);

-- Add episode_id column to sessions table (nullable for backward compatibility)
ALTER TABLE sessions ADD COLUMN episode_id TEXT;

-- Index for fast session-to-episode lookups
CREATE INDEX IF NOT EXISTS idx_sessions_episode ON sessions(episode_id);