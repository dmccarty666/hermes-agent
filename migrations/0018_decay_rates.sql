-- Migration 0018: Add decay_rate_days column to facts table
-- Purpose: Enable per-fact temporal decay tuning by category
-- MEM-018: Temporal decay tuning by fact category

-- Add nullable decay_rate_days column to facts table
-- NULL means "use global half_life_days", numeric value overrides it for that fact
ALTER TABLE facts ADD COLUMN decay_rate_days REAL;

-- Index for efficient queries on decay_rate_days (useful for reporting/analytics)
CREATE INDEX IF NOT EXISTS idx_facts_decay_rate ON facts(decay_rate_days);

-- Backfill: Set decay_rate_days based on category conventions
-- Personal opinions/preferences: decay slowly (180 days)
UPDATE facts SET decay_rate_days = 180.0 WHERE category = 'preference' AND decay_rate_days IS NULL;
-- Factual knowledge: decay slowly (365 days)
UPDATE facts SET decay_rate_days = 365.0 WHERE category = 'fact' AND decay_rate_days IS NULL;
-- Transient context: decay quickly (14 days)
UPDATE facts SET decay_rate_days = 14.0 WHERE category = 'context' AND decay_rate_days IS NULL;
-- Code decisions: decay moderately (90 days - same as global default)
UPDATE facts SET decay_rate_days = 90.0 WHERE category = 'code_decision' AND decay_rate_days IS NULL;
-- All others keep NULL (global default)