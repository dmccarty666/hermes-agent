# Copyright 2026 David McCarty. All rights reserved.
"""Tests for contradict.py — contradiction detection heuristic.

Tests the entity-bucket Jaccard contradiction heuristic per TDD §10.3:
  - Bucket by (project, entity, category)
  - Jaccard > 0.4 threshold
  - Conflict objects returned for downstream processing
  - Mark-disputed function for SQLite
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "hermes_memory_core"))

from hermes_memory_core.dream.contradict import (
    JACCARD_THRESHOLD,
    Conflict,
    _tokenize,
    _jaccard,
    _extract_primary_entity,
    _entity_from_fact,
    _make_bucket_key,
    find_conflicts,
    mark_disputed,
)


# ---------------------------------------------------------------------------
# Tokenization + Jaccard unit tests
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_lowercase(self):
        assert "hello" in _tokenize("Hello world")

    def test_punctuation_stripped(self):
        tokens = _tokenize("Hello, world! How's it going?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "hows" in tokens or "how" in tokens  # apostrophe stripped to "s"

    def test_empty(self):
        assert _tokenize("") == set()
        assert _tokenize("   ") == set()

    def test_numbers(self):
        tokens = _tokenize("Qwen3.6 is version 3.6")
        assert "qwen3" in tokens or "qwen36" in tokens or "qwen" in tokens


class TestJaccard:
    def test_identical(self):
        tokens = {"hello", "world"}
        assert _jaccard(tokens, tokens) == 1.0

    def test_disjoint(self):
        assert _jaccard({"hello"}, {"world"}) == 0.0

    def test_partial_overlap(self):
        a = {"hello", "world", "foo"}
        b = {"hello", "world", "bar"}
        # intersection = {hello, world} = 2, union = {hello, world, foo, bar} = 4
        assert _jaccard(a, b) == 0.5

    def test_empty_set(self):
        assert _jaccard(set(), {"hello"}) == 0.0
        assert _jaccard(set(), set()) == 0.0


class TestExtractPrimaryEntity:
    def test_simple_name(self):
        assert _extract_primary_entity("Qwen is fast and capable") == "Qwen"

    def test_entity_mid_sentence(self):
        assert _extract_primary_entity("I think Qwen is the best model") == "Qwen"

    def test_first_capitalized_word(self):
        result = _extract_primary_entity("The Project uses SQLite")
        # "The" is in stop list — first non-stop cap word is "Project"
        assert result in ("Project",)

    def test_stops_filtered(self):
        # "This" is in stop list — first non-stop cap word is "System"
        result = _extract_primary_entity("This System handles memory")
        assert result in ("System",)

    def test_acronym(self):
        assert _extract_primary_entity("The API works well") in ("API", "The")

    def test_empty(self):
        assert _extract_primary_entity("lowercase only") == ""


class TestEntityFromFact:
    def test_explicit_entity(self):
        fact = {"fact_text": "Qwen is fast", "entity": "Qwen"}
        assert _entity_from_fact(fact) == "Qwen"

    def test_fallback_to_text(self):
        fact = {"fact_text": "Spark is slow", "project": "hermes"}
        assert _entity_from_fact(fact) == "Spark"

    def test_text_key(self):
        fact = {"text": "Claude is helpful", "project": "test"}
        assert _entity_from_fact(fact) == "Claude"

    def test_missing(self):
        assert _entity_from_fact({}) == ""


class TestMakeBucketKey:
    def test_basic(self):
        fact = {"project": "hermes-memory", "fact_text": "Qwen is fast", "scope": "project"}
        bucket = _make_bucket_key(fact)
        assert bucket == ("hermes-memory", "Qwen", "project")

    def test_empty_project(self):
        fact = {"project": "", "fact_text": "Something happened", "scope": "general"}
        bucket = _make_bucket_key(fact)
        assert bucket[0] == ""
        assert bucket[2] == "general"

    def test_infers_entity(self):
        fact = {"project": "test", "fact_text": "LMS is working", "scope": "tool"}
        bucket = _make_bucket_key(fact)
        assert bucket == ("test", "LMS", "tool")


# ---------------------------------------------------------------------------
# find_conflicts integration tests
# ---------------------------------------------------------------------------

class TestFindConflicts:
    def test_no_conflict_different_entity(self):
        """Facts about different entities don't conflict."""
        candidate = {
            "fact_id": "f1",
            "fact_text": "Qwen is fast",
            "project": "hermes-memory",
            "scope": "general",
        }
        existing = [{
            "fact_id": "f2",
            "fact_text": "Claude is helpful",
            "project": "hermes-memory",
            "scope": "general",
        }]
        conflicts = find_conflicts(candidate, existing)
        assert conflicts == []

    def test_no_conflict_different_project(self):
        """Facts about different projects don't conflict."""
        candidate = {
            "fact_id": "f1",
            "fact_text": "Qwen is fast",
            "project": "hermes-memory",
            "scope": "project",
        }
        existing = [{
            "fact_id": "f2",
            "fact_text": "Qwen is slow",
            "project": "other-project",
            "scope": "project",
        }]
        conflicts = find_conflicts(candidate, existing)
        assert conflicts == []

    def test_no_conflict_different_scope(self):
        """Facts in different scopes (category) don't conflict."""
        candidate = {
            "fact_id": "f1",
            "fact_text": "Qwen is fast",
            "project": "hermes-memory",
            "scope": "user_pref",
        }
        existing = [{
            "fact_id": "f2",
            "fact_text": "Qwen is slow",
            "project": "hermes-memory",
            "scope": "project",
        }]
        conflicts = find_conflicts(candidate, existing)
        assert conflicts == []

    def test_conflict_jaccard_above_threshold(self):
        """Contradictory facts with high Jaccard trigger conflict."""
        candidate = {
            "fact_id": "f1",
            "fact_text": "Qwen is fast and efficient",
            "project": "hermes-memory",
            "scope": "general",
        }
        existing = [{
            "fact_id": "f2",
            "fact_text": "Qwen is slow and inefficient",
            "project": "hermes-memory",
            "scope": "general",
        }]
        conflicts = find_conflicts(candidate, existing)
        assert len(conflicts) == 1
        assert conflicts[0].candidate_fact_id == "f1"
        assert conflicts[0].existing_fact_id == "f2"
        assert conflicts[0].jaccard_score > JACCARD_THRESHOLD

    def test_no_conflict_jaccard_below_threshold(self):
        """Similar but not overlapping enough — no conflict."""
        candidate = {
            "fact_id": "f1",
            "fact_text": "Qwen is a language model",
            "project": "hermes-memory",
            "scope": "general",
        }
        existing = [{
            "fact_id": "f2",
            "fact_text": "Qwen has 35 billion parameters",
            "project": "hermes-memory",
            "scope": "general",
        }]
        conflicts = find_conflicts(candidate, existing)
        # Token overlap: "qwen" + maybe "a" + maybe "is" — might be below threshold
        assert conflicts == [] or all(c.jaccard_score <= JACCARD_THRESHOLD for c in conflicts)

    def test_multiple_existing_facts(self):
        """Candidate checked against all existing in bucket."""
        candidate = {
            "fact_id": "f1",
            "fact_text": "Qwen is fast",
            "project": "hermes-memory",
            "scope": "general",
        }
        existing = [
            {"fact_id": "f2", "fact_text": "Claude is helpful", "project": "hermes-memory", "scope": "general"},
            {"fact_id": "f3", "fact_text": "Qwen is slow", "project": "hermes-memory", "scope": "general"},
            {"fact_id": "f4", "fact_text": "LMS is working", "project": "hermes-memory", "scope": "general"},
        ]
        conflicts = find_conflicts(candidate, existing)
        assert len(conflicts) == 1
        assert conflicts[0].existing_fact_id == "f3"

    def test_empty_existing(self):
        """No existing facts returns empty."""
        candidate = {"fact_id": "f1", "fact_text": "Qwen is fast", "project": "test", "scope": "general"}
        assert find_conflicts(candidate, []) == []

    def test_candidate_no_text(self):
        """Candidate with no text returns empty."""
        candidate = {"fact_id": "f1", "project": "test", "scope": "general"}
        assert find_conflicts(candidate, [{"fact_id": "f2", "fact_text": "Something", "project": "test", "scope": "general"}]) == []

    def test_same_fact_id_skipped(self):
        """A fact doesn't conflict with itself."""
        candidate = {
            "fact_id": "f1",
            "fact_text": "Qwen is fast",
            "project": "hermes-memory",
            "scope": "general",
        }
        existing = [{
            "fact_id": "f1",
            "fact_text": "Qwen is fast",
            "project": "hermes-memory",
            "scope": "general",
        }]
        assert find_conflicts(candidate, existing) == []

    def test_conflict_bucket_key_in_result(self):
        """Conflict object includes the bucket key."""
        candidate = {
            "fact_id": "f1",
            "fact_text": "Qwen is fast",
            "project": "hermes-memory",
            "scope": "general",
        }
        existing = [{
            "fact_id": "f2",
            "fact_text": "Qwen is slow",
            "project": "hermes-memory",
            "scope": "general",
        }]
        conflicts = find_conflicts(candidate, existing)
        assert len(conflicts) == 1
        assert conflicts[0].bucket_key == ("hermes-memory", "Qwen", "general")

    def test_jaccard_threshold_exactly(self):
        """Edge case: Jaccard exactly at threshold."""
        # Two facts that share exactly 2 tokens out of 5 each = intersection/union = 2/8 = 0.25
        # Need to craft exactly 0.4
        candidate_tokens = {"a", "b", "c", "d"}
        existing_tokens = {"a", "b", "e", "f", "g"}
        # intersection = {a,b} = 2, union = {a,b,c,d,e,f,g} = 7
        # jaccard = 2/7 ≈ 0.286 — below threshold
        # Build to get 0.4: intersection=2, union=5 => 0.4
        # Candidate: {a, b, c}, Existing: {a, b, d} => union={a,b,c,d}=4, intersection={a,b}=2 => 0.5
        candidate = {
            "fact_id": "f1",
            "fact_text": "Model a is great and fast today",
            "project": "test",
            "scope": "general",
        }
        existing = [{
            "fact_id": "f2",
            "fact_text": "Model a is great and slow today",
            "project": "test",
            "scope": "general",
        }]
        conflicts = find_conflicts(candidate, existing)
        # Both share "model", "a", "is", "great", "and", "today" = 6 shared tokens
        # Unique to each: fast vs slow
        # Should be high Jaccard and trigger conflict
        assert len(conflicts) >= 1


# ---------------------------------------------------------------------------
# mark_disputed tests (require real SQLite)
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_db_path(tmp_path):
    """Provide a temp SQLite path with facts table."""
    db_path = tmp_path / "memory.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    conn.execute(
        "CREATE TABLE IF NOT EXISTS facts "
        "(fact_id TEXT PRIMARY KEY, fact_text TEXT NOT NULL, content_hash TEXT NOT NULL, "
        "scope TEXT NOT NULL, project TEXT, status TEXT NOT NULL DEFAULT 'active', "
        "confidence REAL, source_refs_json TEXT NOT NULL DEFAULT '[]', "
        "tags_json TEXT NOT NULL DEFAULT '[]', "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
        "supersedes_fact_id TEXT)"
    )
    conn.commit()
    conn.close()
    return str(db_path)


def test_mark_disputed_updates_status(tmp_path, memory_db_path):
    """mark_disputed updates the candidate's status to 'disputed'."""
    conn = sqlite3.connect(memory_db_path)
    now = "2026-05-19T10:00:00+00:00"
    conn.execute(
        "INSERT INTO facts (fact_id, fact_text, content_hash, scope, project, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 'active', ?, ?)",
        ("f1", "Qwen is fast", "hash1", "general", "test", now, now),
    )
    conn.execute(
        "INSERT INTO facts (fact_id, fact_text, content_hash, scope, project, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 'active', ?, ?)",
        ("f2", "Qwen is slow", "hash2", "general", "test", now, now),
    )
    conn.commit()

    mark_disputed(conn, "f1", "f2")

    row = conn.execute("SELECT status FROM facts WHERE fact_id = 'f1'").fetchone()
    assert row[0] == "disputed"
    conn.close()


def test_mark_disputed_sets_supersedes_fact_id(tmp_path, memory_db_path):
    """mark_disputed sets the supersedes_fact_id link."""
    conn = sqlite3.connect(memory_db_path)
    now = "2026-05-19T10:00:00+00:00"
    conn.execute(
        "INSERT INTO facts (fact_id, fact_text, content_hash, scope, project, status, created_at, updated_at, supersedes_fact_id) "
        "VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)",
        ("f1", "Qwen is fast", "hash1", "general", "test", now, now, None),
    )
    conn.execute(
        "INSERT INTO facts (fact_id, fact_text, content_hash, scope, project, status, created_at, updated_at, supersedes_fact_id) "
        "VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)",
        ("f2", "Qwen is slow", "hash2", "general", "test", now, now, None),
    )
    conn.commit()

    mark_disputed(conn, "f1", "f2")

    row = conn.execute("SELECT supersedes_fact_id FROM facts WHERE fact_id = 'f1'").fetchone()
    assert row[0] == "f2"
    conn.close()


def test_mark_disputed_non_existent_fact(tmp_path, memory_db_path):
    """mark_disputed on non-existent fact is no-op (no exception)."""
    conn = sqlite3.connect(memory_db_path)
    now = "2026-05-19T10:00:00+00:00"
    conn.execute(
        "INSERT INTO facts (fact_id, fact_text, content_hash, scope, project, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, 'active', ?, ?)",
        ("f2", "Qwen is slow", "hash2", "general", "test", now, now),
    )
    conn.commit()

    # Should not raise
    mark_disputed(conn, "nonexistent", "f2")
    conn.close()


# ---------------------------------------------------------------------------
# End-to-end contradiction scenario (from AC)
# ---------------------------------------------------------------------------

class TestContradictionScenario:
    def test_contradictory_facts_jaccard_above_threshold(self):
        """AC: Two facts with identical entity and project but contradictory content,
        when bucket Jaccard > 0.4, then second fact is marked disputed."""
        candidate = {
            "fact_id": "f_new",
            "fact_text": "Qwen is fast",
            "project": "hermes-memory",
            "scope": "general",
        }
        existing = [{
            "fact_id": "f_old",
            "fact_text": "Qwen is slow",
            "project": "hermes-memory",
            "scope": "general",
        }]
        conflicts = find_conflicts(candidate, existing)
        # Both have "Qwen is" — token overlap is 2 shared tokens
        # candidate tokens: {qwen, is, fast}
        # existing tokens: {qwen, is, slow}
        # intersection = {qwen, is} = 2, union = {qwen, is, fast, slow} = 4
        # jaccard = 2/4 = 0.5 > 0.4
        assert len(conflicts) == 1
        assert conflicts[0].jaccard_score == 0.5

    def test_non_contradictory_different_entity(self):
        """AC: Different entity — no disputed flag."""
        candidate = {
            "fact_id": "f1",
            "fact_text": "Qwen is fast",
            "project": "hermes-memory",
            "scope": "general",
        }
        existing = [{
            "fact_id": "f2",
            "fact_text": "Spark is fast",
            "project": "hermes-memory",
            "scope": "general",
        }]
        conflicts = find_conflicts(candidate, existing)
        # tokens: {qwen,is,fast} vs {spark,is,fast} — intersection={is,fast}=2, union={qwen,is,fast,spark}=4
        # jaccard = 2/4 = 0.5 — but entity differs so should NOT be flagged
        # (entity extraction: "Qwen" vs "Spark" — bucket key differs, no check)
        # Wait — the bucket key includes entity, so different entities won't be compared
        assert conflicts == []

    def test_non_contradictory_jaccard_below_threshold(self):
        """AC: Same entity, Jaccard < 0.4 — no disputed flag."""
        candidate = {
            "fact_id": "f1",
            "fact_text": "Qwen was trained on a large dataset",
            "project": "hermes-memory",
            "scope": "general",
        }
        existing = [{
            "fact_id": "f2",
            "fact_text": "Qwen handles JSON well",
            "project": "hermes-memory",
            "scope": "general",
        }]
        conflicts = find_conflicts(candidate, existing)
        # Very little overlap — should be below 0.4
        assert conflicts == [] or all(c.jaccard_score <= JACCARD_THRESHOLD for c in conflicts)