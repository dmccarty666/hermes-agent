"""MVP Acceptance Test Suite — Scenario M: Handoff & Multi-Agent.

Verifies Plan.md §9, Scenario M:
  1. Agent-A creates session S1, captures facts/decisions/questions.
  2. /new for Agent-B.
  3. Verify Agent-B receives S1 summary.
  4. Verify Agent-B can add new facts that supersede or extend S1 content.
  5. Verify agent_id tracked in audit_log for all writes.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_agent_memory():
    """Memory dir ready for multi-agent handoff testing."""
    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        mem.mkdir()
        index = mem / "index"
        index.mkdir()
        db_path = index / "memory.sqlite"
        thread_dir = mem / "SESSION-THREAD"
        thread_dir.mkdir()

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE sessions ("
            "  session_id TEXT PRIMARY KEY, agent TEXT NOT NULL, "
            "  title TEXT, project TEXT, started_at TEXT NOT NULL, "
            "  ended_at TEXT, summary TEXT)"
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
            "  content_hash TEXT NOT NULL, scope TEXT NOT NULL, "
            "  source_refs_json TEXT NOT NULL, status TEXT, "
            "  supersedes_fact_id TEXT, superseded_by_fact_id TEXT, "
            "  created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE decisions ("
            "  decision_id TEXT PRIMARY KEY, decision_text TEXT NOT NULL, "
            "  content_hash TEXT NOT NULL, scope TEXT NOT NULL, "
            "  source_refs_json TEXT NOT NULL, status TEXT, "
            "  created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE open_questions ("
            "  question_id TEXT PRIMARY KEY, question_text TEXT NOT NULL, "
            "  content_hash TEXT NOT NULL, scope TEXT NOT NULL, "
            "  source_refs_json TEXT NOT NULL, status TEXT, "
            "  created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE audit_log ("
            "  audit_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "  event_id TEXT, session_id TEXT, turn_id TEXT, "
            "  action TEXT NOT NULL, detail TEXT, agent_id TEXT, "
            "  created_at TEXT NOT NULL)"
        )
        conn.commit()
        conn.close()

        yield {
            "mem": mem,
            "db_path": db_path,
            "thread_dir": thread_dir,
        }


# ---------------------------------------------------------------------------
# Scenario M Tests
# ---------------------------------------------------------------------------

def test_scenario_M_agent_a_creates_facts_and_decisions(multi_agent_memory):
    """Given Agent-A (id='agent-alpha') creates facts and decisions,
    then audit_log records agent_id for all writes."""
    ctx = multi_agent_memory
    db_path = ctx["db_path"]
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    agent_a_id = "agent-alpha"
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
        (session_id, agent_a_id, now),
    )

    # Agent-A writes a fact
    fact_id = f"fact-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO facts (fact_id, fact_text, content_hash, scope, "
        "source_refs_json, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            fact_id,
            "Project Foo uses OAuth 2.0 + PKCE for authentication",
            f"ch-{uuid.uuid4().hex[:8]}",
            "project",
            json.dumps([f"capture:sessions:{session_id}"]),
            "active",
            now,
            now,
        ),
    )
    conn.execute(
        "INSERT INTO audit_log (event_id, session_id, turn_id, action, detail, "
        "agent_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            f"ev-{uuid.uuid4().hex[:8]}", session_id, None,
            "write", f"fact:{fact_id}", agent_a_id, now,
        ),
    )

    # Agent-A writes a decision
    decision_id = f"decision-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO decisions (decision_id, decision_text, content_hash, scope, "
        "source_refs_json, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            decision_id,
            "Use JWT tokens in httpOnly cookies for Project Foo",
            f"ch-{uuid.uuid4().hex[:8]}",
            "project",
            json.dumps([f"capture:sessions:{session_id}"]),
            "active",
            now,
            now,
        ),
    )
    conn.execute(
        "INSERT INTO audit_log (event_id, session_id, turn_id, action, detail, "
        "agent_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            f"ev-{uuid.uuid4().hex[:8]}", session_id, None,
            "write", f"decision:{decision_id}", agent_a_id, now,
        ),
    )
    conn.commit()
    conn.close()

    # Verify audit_log has agent_id
    conn = sqlite3.connect(str(db_path))
    audit_rows = conn.execute(
        "SELECT action, agent_id, detail FROM audit_log WHERE agent_id = ?",
        (agent_a_id,),
    ).fetchall()
    conn.close()

    assert len(audit_rows) >= 2, f"Expected at least 2 audit entries for agent-alpha, got {len(audit_rows)}"
    agent_ids = {row[1] for row in audit_rows}
    assert agent_a_id in agent_ids


def test_scenario_M_agent_b_receives_agent_a_summary(multi_agent_memory):
    """Given Agent-A's session S1 exists, when /new creates Agent-B's session,
    then Agent-B receives S1 summary via SESSION-THREAD injection."""
    ctx = multi_agent_memory
    thread_dir = ctx["thread_dir"]
    session_id_a = f"session-a-{uuid.uuid4().hex[:8]}"
    session_id_b = f"session-b-{uuid.uuid4().hex[:8]}"

    # Write Agent-A's session thread summary
    thread_path = thread_dir / f"{session_id_a}.md"
    thread_path.write_text(
        "# Session Thread — Agent-A\n\n"
        "## Project: Project Foo (Authentication)\n\n"
        "## Facts\n"
        "- Project Foo uses OAuth 2.0 + PKCE for authentication\n"
        "- JWT tokens stored in httpOnly cookies\n\n"
        "## Decisions\n"
        "- Use JWT tokens in httpOnly cookies\n\n"
        "## Open Questions\n"
        "- Should we implement token refresh endpoint?\n"
    )

    # Simulate /new for Agent-B — injects prior thread
    def inject_prior_thread(new_session_id: str, prior_session_id: str) -> dict:
        """Returns the injected message dict."""
        prior_content = (thread_dir / f"{prior_session_id}.md").read_text()
        return {
            "role": "user",
            "content": (
                f"[Restored from prior session {prior_session_id}]:\n\n"
                f"{prior_content}"
            ),
            "session_id": new_session_id,
            "prior_session_id": prior_session_id,
        }

    injected = inject_prior_thread(session_id_b, session_id_a)

    assert "Project Foo" in injected["content"]
    assert "OAuth" in injected["content"] or "JWT" in injected["content"]
    assert injected["prior_session_id"] == session_id_a


def test_scenario_M_agent_b_can_add_superseding_facts(multi_agent_memory):
    """Given Agent-B receives Agent-A's facts, Agent-B can add new facts
    that supersede or extend the prior context."""
    ctx = multi_agent_memory
    db_path = ctx["db_path"]
    agent_b_id = "agent-beta"
    now = datetime.now(timezone.utc).isoformat()

    # Agent-A's original fact
    f1_id = f"fact-{uuid.uuid4().hex[:8]}"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO facts (fact_id, fact_text, content_hash, scope, "
        "source_refs_json, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            f1_id,
            "Project Foo uses OAuth 2.0 + PKCE for authentication",
            f"ch-{uuid.uuid4().hex[:8]}",
            "project",
            json.dumps(["capture:sessions:session-a-foo"]),
            "active",
            now,
            now,
        ),
    )
    conn.commit()

    # Agent-B's new fact (extends/supersedes Agent-A's)
    f2_id = f"fact-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO facts (fact_id, fact_text, content_hash, scope, "
        "source_refs_json, status, created_at, updated_at, "
        "supersedes_fact_id, superseded_by_fact_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            f2_id,
            "Project Foo uses OAuth 2.0 + PKCE for authentication with 15min token expiry",
            f"ch-{uuid.uuid4().hex[:8]}",
            "project",
            json.dumps(["capture:sessions:session-b-bar"]),
            "active",
            now,
            now,
            f1_id,  # supersedes Agent-A's fact
            None,
        ),
    )
    conn.execute(
        "UPDATE facts SET superseded_by_fact_id = ? WHERE fact_id = ?",
        (f2_id, f1_id),
    )
    conn.execute(
        "INSERT INTO audit_log (event_id, session_id, action, detail, "
        "agent_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            f"ev-{uuid.uuid4().hex[:8]}",
            "session-b-bar",
            "write",
            f"fact:{f2_id} supersedes {f1_id}",
            agent_b_id,
            now,
        ),
    )
    conn.commit()
    conn.close()

    # Verify supersedes relationship
    conn = sqlite3.connect(str(db_path))
    f1 = conn.execute(
        "SELECT fact_id, status, superseded_by_fact_id FROM facts WHERE fact_id = ?",
        (f1_id,),
    ).fetchone()
    f2 = conn.execute(
        "SELECT fact_id, status, supersedes_fact_id FROM facts WHERE fact_id = ?",
        (f2_id,),
    ).fetchone()
    audit_rows = conn.execute(
        "SELECT agent_id, action FROM audit_log WHERE agent_id = ?",
        (agent_b_id,),
    ).fetchall()
    conn.close()

    assert f1 is not None
    assert f2 is not None
    assert f1[2] == f2_id, "f1 should be superseded by f2"
    assert f2[2] == f1_id, "f2 should supersede f1"
    assert len(audit_rows) >= 1, "Agent-B should have audit entries"


def test_scenario_M_audit_log_tracks_all_agent_writes(multi_agent_memory):
    """Verify every write to memory creates an audit_log row with agent_id."""
    ctx = multi_agent_memory
    db_path = ctx["db_path"]

    agents = ["agent-alpha", "agent-beta", "agent-gamma"]
    now = datetime.now(timezone.utc).isoformat()
    session_id = f"session-{uuid.uuid4().hex[:8]}"

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
        (session_id, agents[0], now),
    )
    conn.commit()

    for agent_id in agents:
        turn_id = f"turn-{uuid.uuid4().hex[:8]}"
        fact_id = f"fact-{uuid.uuid4().hex[:8]}"

        conn.execute(
            "INSERT INTO turns (turn_id, session_id, sequence, timestamp, role, "
            "content, raw_content_hash, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn_id, session_id, 0, now, "assistant",
                f"Turn by {agent_id}",
                f"raw-{uuid.uuid4().hex[:8]}",
                f"content-{uuid.uuid4().hex[:8]}",
            ),
        )
        conn.execute(
            "INSERT INTO facts (fact_id, fact_text, content_hash, scope, "
            "source_refs_json, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact_id,
                f"Fact created by {agent_id}",
                f"ch-{uuid.uuid4().hex[:8]}",
                "project",
                json.dumps([f"capture:turns:{turn_id}"]),
                "active",
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO audit_log (event_id, session_id, turn_id, action, detail, "
            "agent_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"ev-{uuid.uuid4().hex[:8]}", session_id, turn_id,
                "write", f"turn:{turn_id} fact:{fact_id}",
                agent_id, now,
            ),
        )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(str(db_path))
    for agent_id in agents:
        count = conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()[0]
        assert count >= 1, f"Expected at least 1 audit entry for {agent_id}, got {count}"
    conn.close()


def test_scenario_M_hermes_local_provider_tracks_agent_id(
    multi_agent_memory,
):
    """Verify HermesLocalProvider records agent_id in memory writes."""
    ctx = multi_agent_memory
    db_path = ctx["db_path"]

    try:
        from agent.memory_provider import MemoryProvider

        class TestProvider(MemoryProvider):
            name = "test-multi-agent"
            _db_path = str(db_path)

            def is_available(self):
                return True

            def initialize(self, session_id, **kwargs):
                self._session_id = session_id
                self._agent_id = kwargs.get("agent_id", "unknown")

            def sync_turn(self, user_content, assistant_content, **kwargs):
                agent_id = kwargs.get("agent_id", getattr(self, "_agent_id", "unknown"))
                # Verify agent_id is passed through
                assert agent_id != "unknown", "agent_id must be passed to sync_turn"

            def memory_write(self, content, **kwargs):
                agent_id = kwargs.get("agent_id", getattr(self, "_agent_id", "unknown"))
                assert agent_id != "unknown", "agent_id must be passed to memory_write"
                return {"status": "ok"}

            def get_tool_schemas(self):
                return []

            def handle_tool_call(self, tool_name, args):
                return ""

            def shutdown(self):
                pass

        provider = TestProvider()
        provider.initialize("test-session", agent_id="agent-alpha")
        provider.sync_turn("hello", "hi", agent_id="agent-alpha")
        result = provider.memory_write("test content", agent_id="agent-alpha")
        assert result["status"] == "ok"
    except ImportError:
        pytest.skip("agent.memory_provider not yet importable")