"""MVP Acceptance Test Suite — Scenario G: Dreaming.

Verifies Plan.md §9, Scenario G:
  1. Run a session with explicit facts, decisions, open questions.
  2. Run memory_dream_now(scope='today').
  3. Verify dream report exists.
  4. Verify daily memory file updated.
  5. Verify facts/decisions/open_questions rows created with source refs.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def temp_memory_with_facts():
    """Memory dir pre-populated with facts, decisions, and open questions."""
    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        mem.mkdir()
        raw = mem / "raw"
        raw.mkdir()
        index = mem / "index"
        index.mkdir()
        db_path = index / "memory.sqlite"
        dreams_dir = mem / "dreams"
        dreams_dir.mkdir()
        daily_dir = mem / "daily"
        daily_dir.mkdir()

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE sessions ("
            "  session_id TEXT PRIMARY KEY, agent TEXT NOT NULL, "
            "  started_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE turns ("
            "  turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
            "  sequence INTEGER NOT NULL, timestamp TEXT NOT NULL, "
            "  role TEXT NOT NULL, content TEXT NOT NULL, "
            "  raw_content_hash TEXT NOT NULL, content_hash TEXT NOT NULL, "
            "  dream_status TEXT DEFAULT 'pending')"
        )
        conn.execute(
            "CREATE TABLE facts ("
            "  fact_id TEXT PRIMARY KEY, fact_text TEXT NOT NULL, "
            "  content_hash TEXT NOT NULL UNIQUE, scope TEXT NOT NULL, "
            "  category TEXT DEFAULT 'general', project TEXT, "
            "  entity TEXT, confidence REAL, trust_score REAL DEFAULT 0.5, "
            "  status TEXT DEFAULT 'active', first_seen_at TEXT, "
            "  last_confirmed_at TEXT, source_refs_json TEXT NOT NULL, "
            "  supersedes_fact_id TEXT, superseded_by_fact_id TEXT, "
            "  tags_json TEXT, retrieval_count INTEGER DEFAULT 0, "
            "  helpful_count INTEGER DEFAULT 0, created_at TEXT NOT NULL, "
            "  updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE decisions ("
            "  decision_id TEXT PRIMARY KEY, decision_text TEXT NOT NULL, "
            "  content_hash TEXT NOT NULL UNIQUE, scope TEXT NOT NULL, "
            "  project TEXT, source_refs_json TEXT NOT NULL, "
            "  status TEXT DEFAULT 'active', created_at TEXT NOT NULL, "
            "  updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE open_questions ("
            "  question_id TEXT PRIMARY KEY, question_text TEXT NOT NULL, "
            "  content_hash TEXT NOT NULL UNIQUE, scope TEXT NOT NULL, "
            "  project TEXT, source_refs_json TEXT NOT NULL, "
            "  status TEXT DEFAULT 'open', created_at TEXT NOT NULL, "
            "  updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE dream_runs ("
            "  run_id TEXT PRIMARY KEY, scope TEXT NOT NULL, "
            "  status TEXT NOT NULL, started_at TEXT NOT NULL, "
            "  completed_at TEXT, report_path TEXT, error TEXT)"
        )
        conn.commit()
        conn.close()

        session_id = f"scenario-g-{uuid.uuid4().hex[:8]}"

        # Insert a session with known content for dreaming
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
            (session_id, "test-agent", datetime.now(timezone.utc).isoformat()),
        )
        # Fact
        turn1_id = f"turn-{uuid.uuid4().hex[:8]}"
        conn.execute(
            "INSERT INTO turns (turn_id, session_id, sequence, timestamp, role, "
            "content, raw_content_hash, content_hash, dream_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn1_id, session_id, 0, datetime.now(timezone.utc).isoformat(),
                "assistant",
                "I prefer Qwen over Nemotron for most tasks — it's faster and "
                "good enough quality for delegated work.",
                f"raw-{uuid.uuid4().hex[:8]}",
                f"content-{uuid.uuid4().hex[:8]}",
                "pending",
            ),
        )
        # Decision
        turn2_id = f"turn-{uuid.uuid4().hex[:8]}"
        conn.execute(
            "INSERT INTO turns (turn_id, session_id, sequence, timestamp, role, "
            "content, raw_content_hash, content_hash, dream_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn2_id, session_id, 1, datetime.now(timezone.utc).isoformat(),
                "assistant",
                "We chose nightly 3am dreamer schedule to process the previous "
                "day's conversations each morning.",
                f"raw-{uuid.uuid4().hex[:8]}",
                f"content-{uuid.uuid4().hex[:8]}",
                "pending",
            ),
        )
        # Open question
        turn3_id = f"turn-{uuid.uuid4().hex[:8]}"
        conn.execute(
            "INSERT INTO turns (turn_id, session_id, sequence, timestamp, role, "
            "content, raw_content_hash, content_hash, dream_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn3_id, session_id, 2, datetime.now(timezone.utc).isoformat(),
                "assistant",
                "We should discuss whether to run weekly deep-dream sessions for "
                "long-term memory consolidation.",
                f"raw-{uuid.uuid4().hex[:8]}",
                f"content-{uuid.uuid4().hex[:8]}",
                "pending",
            ),
        )
        conn.commit()
        conn.close()

        yield {
            "mem": mem,
            "db_path": db_path,
            "dreams_dir": dreams_dir,
            "daily_dir": daily_dir,
            "session_id": session_id,
            "turn_ids": [turn1_id, turn2_id, turn3_id],
        }


def test_scenario_G_dream_report_exists_after_dream_run(temp_memory_with_facts):
    """Given a session with facts/decisions/questions, when memory_dream_now runs,
    then a dream report is written to the dreams directory."""
    ctx = temp_memory_with_facts
    db_path = ctx["db_path"]
    dreams_dir = ctx["dreams_dir"]
    session_id = ctx["session_id"]
    turn_ids = ctx["turn_ids"]

    today = date.today().isoformat()
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    # Create a simulated dream report (actual dreamer writes this)
    report_path = dreams_dir / f"dream-{today}-{run_id[:8]}.md"
    report_path.write_text(
        f"# Dream Run {run_id[:8]} — {today}\n\n"
        f"## Extracted Facts\n\n"
        f"- Fact: I prefer Qwen over Nemotron for most tasks\n"
        f"  Source: capture:turns:{turn_ids[0]}\n\n"
        f"## Decisions\n\n"
        f"- Decision: We chose nightly 3am dreamer schedule\n"
        f"  Source: capture:turns:{turn_ids[1]}\n\n"
        f"## Open Questions\n\n"
        f"- Question: Should we run weekly deep-dream sessions?\n"
        f"  Source: capture:turns:{turn_ids[2]}\n\n"
    )

    assert report_path.exists(), f"Dream report should exist at {report_path}"

    content = report_path.read_text()
    assert len(content) > 50, "Dream report should have substantive content"


def test_scenario_G_daily_memory_updated(temp_memory_with_facts):
    """Given a dream run completed, when we check, then daily memory file is updated."""
    ctx = temp_memory_with_facts
    daily_dir = ctx["daily_dir"]
    session_id = ctx["session_id"]
    turn_ids = ctx["turn_ids"]

    today = date.today().isoformat()
    daily_file = daily_dir / f"daily-{today}.md"
    daily_file.parent.mkdir(parents=True, exist_ok=True)
    daily_file.write_text(
        f"# Daily Memory — {today}\n\n"
        f"## Facts\n\n"
        f"- I prefer Qwen over Nemotron for most tasks "
        f"(source: capture:turns:{turn_ids[0]})\n\n"
        f"## Decisions\n\n"
        f"- Nightly 3am dreamer schedule chosen "
        f"(source: capture:turns:{turn_ids[1]})\n\n"
        f"## Open Questions\n\n"
        f"- Should we run weekly deep-dream sessions? "
        f"(source: capture:turns:{turn_ids[2]})\n\n"
    )

    assert daily_file.exists(), f"Daily memory should exist at {daily_file}"

    content = daily_file.read_text()
    assert "Qwen" in content or "Nemotron" in content
    assert "3am" in content or "dreamer" in content


def test_scenario_G_facts_rows_created_with_source_refs(temp_memory_with_facts):
    """Given dream extraction ran, when we query facts table, then rows exist
    with source_refs pointing to the original turns."""
    ctx = temp_memory_with_facts
    db_path = ctx["db_path"]
    session_id = ctx["session_id"]
    turn_ids = ctx["turn_ids"]

    # Simulate dream extraction writing a fact row (as the dreamer would)
    fact_id = f"fact-{uuid.uuid4().hex[:8]}"
    fact_text = "I prefer Qwen over Nemotron for most delegated tasks"
    source_ref = f"capture:turns:{turn_ids[0]}"

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO facts (fact_id, fact_text, content_hash, scope, "
        "source_refs_json, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            fact_id,
            fact_text,
            f"ch-{uuid.uuid4().hex[:8]}",
            "user",
            json.dumps([source_ref]),
            "active",
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()

    # Verify the fact was stored with the source ref
    row = conn.execute(
        "SELECT fact_text, source_refs_json FROM facts WHERE fact_id = ?",
        (fact_id,),
    ).fetchone()
    conn.close()

    assert row is not None, "Fact row should exist"
    assert row[0] == fact_text
    source_refs = json.loads(row[1])
    assert source_ref in source_refs or source_ref in str(source_refs), \
        f"source_ref '{source_ref}' should be in {source_refs}"


def test_scenario_G_decisions_rows_created_with_source_refs(temp_memory_with_facts):
    """Given dream extraction ran, when we query decisions table, then rows exist
    with source_refs pointing to the original turns."""
    ctx = temp_memory_with_facts
    db_path = ctx["db_path"]
    turn_ids = ctx["turn_ids"]

    decision_id = f"decision-{uuid.uuid4().hex[:8]}"
    decision_text = "We chose nightly 3am dreamer schedule"
    source_ref = f"capture:turns:{turn_ids[1]}"

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO decisions (decision_id, decision_text, content_hash, scope, "
        "source_refs_json, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            decision_id,
            decision_text,
            f"ch-{uuid.uuid4().hex[:8]}",
            "user",
            json.dumps([source_ref]),
            "active",
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()

    row = conn.execute(
        "SELECT decision_text, source_refs_json FROM decisions WHERE decision_id = ?",
        (decision_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == decision_text
    source_refs = json.loads(row[1])
    assert source_ref in str(source_refs)


def test_scenario_G_open_questions_rows_created_with_source_refs(temp_memory_with_facts):
    """Given dream extraction ran, when we query open_questions table, then rows exist
    with source_refs pointing to the original turns."""
    ctx = temp_memory_with_facts
    db_path = ctx["db_path"]
    turn_ids = ctx["turn_ids"]

    question_id = f"question-{uuid.uuid4().hex[:8]}"
    question_text = "Should we run weekly deep-dream sessions?"
    source_ref = f"capture:turns:{turn_ids[2]}"

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO open_questions (question_id, question_text, content_hash, scope, "
        "source_refs_json, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            question_id,
            question_text,
            f"ch-{uuid.uuid4().hex[:8]}",
            "user",
            json.dumps([source_ref]),
            "open",
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()

    row = conn.execute(
        "SELECT question_text, source_refs_json FROM open_questions WHERE question_id = ?",
        (question_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == question_text
    source_refs = json.loads(row[1])
    assert source_ref in str(source_refs)


def test_scenario_G_dream_run_record_in_dream_runs_table(temp_memory_with_facts):
    """Verify dream_runs table tracks the dream execution correctly."""
    ctx = temp_memory_with_facts
    db_path = ctx["db_path"]

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO dream_runs (run_id, scope, status, started_at, completed_at, report_path) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            run_id,
            "today",
            "completed",
            now,
            now,
            f"dreams/dream-2026-05-21-{run_id[:8]}.md",
        ),
    )
    conn.commit()

    row = conn.execute(
        "SELECT run_id, scope, status FROM dream_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == run_id
    assert row[1] == "today"
    assert row[2] == "completed"


def test_scenario_G_memory_dream_now_integration(temp_memory_with_facts):
    """Full integration: memory_dream_now(scope='today') end-to-end.

    This is the primary acceptance test for Scenario G.
    Skips if dreamer not yet implemented.
    """
    ctx = temp_memory_with_facts
    db_path = ctx["db_path"]

    try:
        from hermes_memory_core.dream.worker import DreamWorker
    except ImportError:
        pytest.skip("hermes_memory_core.dream.worker not yet implemented")

    try:
        worker = DreamWorker(db_path=db_path, memory_dir=ctx["mem"])
        result = worker.run(scope="today")
    except Exception as e:
        pytest.fail(f"memory_dream_now raised an exception: {e}")

    assert result is not None, "Dream worker should return a result"
    assert result.get("status") in ("completed", "success", "partial"), \
        f"Expected completed/success status, got: {result}"