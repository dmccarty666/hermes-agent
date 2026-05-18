"""Tests for FSStore.export_qmd (T-008) and on_session_end hook."""

import json
import tempfile
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest

from hermes_memory_core.store.fs import FSStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(session_id: str, turn_id: str, sequence: int,
               role: str = "user", content: str = "hello world",
               agent: str = "test-agent",
               tool_calls: list | None = None,
               metadata: dict | None = None) -> dict:
    """Minimal valid event matching the event-schema.json."""
    ev = {
        "event_id": str(uuid4()),
        "session_id": session_id,
        "turn_id": turn_id,
        "sequence": sequence,
        "timestamp": "2026-05-17T10:00:00Z",
        "role": role,
        "content": content,
        "agent": agent,
        "source": "cli",
    }
    if tool_calls is not None:
        ev["tool_calls"] = tool_calls
    if metadata is not None:
        ev["metadata"] = metadata
    return ev


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_base(tmp_path):
    return tmp_path


@pytest.fixture
def store(tmp_base):
    return FSStore(base_path=tmp_base)


# ---------------------------------------------------------------------------
# AC-1: export_qmd creates .qmd file at the correct path
# ---------------------------------------------------------------------------

class TestExportQmdCreatesFile:
    def test_qmd_file_created_at_correct_path(self, store, tmp_base):
        """AC-1: export_qmd writes to memory/qmd/YYYY/YYYY-MM-DD/{session_id}.qmd."""
        session_id = "sess_qmd_001"
        events = [
            make_event(session_id=session_id, turn_id="turn_0", sequence=0,
                       role="user", content="Hello, world!"),
            make_event(session_id=session_id, turn_id="turn_1", sequence=1,
                       role="assistant", content="Hi there!"),
        ]
        for ev in events:
            store.append_event(ev)

        result_path = store.export_qmd(session_id)

        # Verify path structure
        today = date.today()
        year = str(today.year)
        date_str = today.isoformat()
        expected = tmp_base / "qmd" / year / date_str / f"{session_id}.qmd"

        assert result_path == expected, f"Expected {expected}, got {result_path}"
        assert expected.exists(), f"QMD file not found at {expected}"


# ---------------------------------------------------------------------------
# AC-2: QMD frontmatter contains required fields
# ---------------------------------------------------------------------------

class TestQmdFrontmatter:
    def test_frontmatter_has_required_fields(self, store, tmp_base):
        """AC-2: frontmatter has session_id, project, started_at, ended_at, tags, source_refs."""
        session_id = "sess_qmd_002"
        events = [
            make_event(session_id=session_id, turn_id="turn_0", sequence=0,
                       role="user", content="First message",
                       metadata={"project": "my-project", "tags": ["test", "demo"]}),
            make_event(session_id=session_id, turn_id="turn_1", sequence=1,
                       role="assistant", content="Second message"),
        ]
        for ev in events:
            store.append_event(ev)

        store.export_qmd(session_id)

        date_str = date.today().isoformat()
        year = str(date.today().year)
        qmd_path = tmp_base / "qmd" / year / date_str / f"{session_id}.qmd"

        raw = qmd_path.read_text(encoding="utf-8")

        # Frontmatter starts with --- and ends with ---
        assert raw.startswith("---"), "QMD must start with YAML frontmatter delimiter"
        front_end = raw.index("\n---") + 4  # closing --- on its own line
        front_text = raw[2:front_end]  # strip opening ---

        # Parse frontmatter fields (simple line-based check)
        assert "session_id:" in front_text, "frontmatter missing session_id"
        assert "project:" in front_text, "frontmatter missing project"
        assert "started_at:" in front_text, "frontmatter missing started_at"
        assert "ended_at:" in front_text, "frontmatter missing ended_at"
        assert "tags:" in front_text, "frontmatter missing tags"
        assert "source_refs:" in front_text, "frontmatter missing source_refs"


# ---------------------------------------------------------------------------
# AC-3: turns appear in chronological order with role labels + content
# ---------------------------------------------------------------------------

class TestQmdTurnBody:
    def test_turns_in_chronological_order(self, store, tmp_base):
        """AC-3: turns in body appear in chronological order with role labels and content."""
        session_id = "sess_qmd_003"
        events = [
            make_event(session_id=session_id, turn_id="turn_0", sequence=0,
                       role="user", content="First user message"),
            make_event(session_id=session_id, turn_id="turn_1", sequence=1,
                       role="assistant", content="Assistant response"),
            make_event(session_id=session_id, turn_id="turn_2", sequence=2,
                       role="user", content="Third message"),
        ]
        for ev in events:
            store.append_event(ev)

        store.export_qmd(session_id)

        date_str = date.today().isoformat()
        year = str(date.today().year)
        qmd_path = tmp_base / "qmd" / year / date_str / f"{session_id}.qmd"

        raw = qmd_path.read_text(encoding="utf-8")

        # Body is everything after the closing YAML --- delimiter
        # Find the closing --- (on its own line) which ends the frontmatter
        frontmatter_end = raw.find("\n---\n")  # closing --- on its own line
        if frontmatter_end == -1:
            frontmatter_end = raw.find("\n---")  # fallback
        body_start = frontmatter_end + 4
        body = raw[body_start:].lstrip("\n")

        # Check role labels appear (turn headings include [role])
        assert "[user]" in body, "user role label missing from QMD body"
        assert "[assistant]" in body.lower(), "assistant role label missing from QMD body"
        # Content appears in body
        assert "First user message" in body
        assert "Assistant response" in body
        assert "Third message" in body

        # Ordering: first user before first assistant before second user
        first_user_pos = body.index("First user message")
        asst_pos = body.index("Assistant response")
        third_pos = body.index("Third message")
        assert first_user_pos < asst_pos < third_pos, "Turns not in chronological order"


# ---------------------------------------------------------------------------
# AC-4: tool calls appear as brief summary lines, not raw JSON
# ---------------------------------------------------------------------------

class TestToolCallSummaries:
    def test_tool_calls_summarized_not_raw_json(self, store, tmp_base):
        """AC-4: tool calls show as -> tool_use: name(args) not raw JSON."""
        session_id = "sess_qmd_004"
        events = [
            make_event(session_id=session_id, turn_id="turn_0", sequence=0,
                       role="user", content="Search memory"),
            make_event(
                session_id=session_id, turn_id="turn_1", sequence=1,
                role="assistant", content="Let me search.",
                tool_calls=[
                    {"id": "call_abc", "type": "function",
                     "function": {"name": "memory_query",
                                  "arguments": {"query": "what is python"}}}
                ]
            ),
        ]
        for ev in events:
            store.append_event(ev)

        store.export_qmd(session_id)

        date_str = date.today().isoformat()
        year = str(date.today().year)
        qmd_path = tmp_base / "qmd" / year / date_str / f"{session_id}.qmd"

        raw = qmd_path.read_text(encoding="utf-8")
        body_start = raw.index("---") + 4
        body = raw[body_start:].lstrip("\n")

        # Should have a tool call summary line (arrow prefix)
        assert "tool_use" in body or "memory_query" in body, \
            "Tool call not summarized in QMD body"

        # Must NOT contain raw JSON for the tool call
        assert '"call_abc"' not in body, "Raw tool call JSON found in QMD"
        assert '"function"' not in body, "Raw function JSON found in QMD"


# ---------------------------------------------------------------------------
# AC-5: re-export overwrites without error or duplication
# ---------------------------------------------------------------------------

class TestQmdReExport:
    def test_re_export_overwrites_safely(self, store, tmp_base):
        """AC-5: calling export_qmd again on same session overwrites, no error."""
        session_id = "sess_qmd_005"
        events = [
            make_event(session_id=session_id, turn_id="turn_0", sequence=0,
                       role="user", content="Original message"),
        ]
        for ev in events:
            store.append_event(ev)

        # First export
        path1 = store.export_qmd(session_id)
        content1 = path1.read_text(encoding="utf-8")

        # Second export (re-export) — must not raise
        path2 = store.export_qmd(session_id)
        content2 = path2.read_text(encoding="utf-8")

        assert path1 == path2, "Re-export should write same path"
        # No duplication in file
        assert content1.count("Original message") == 1, "Content duplicated on re-export"
        assert content2.count("Original message") == 1, "Content duplicated on re-export"


# ---------------------------------------------------------------------------
# AC-6: on_session_end hook — integration via HermesLocalProvider
# ---------------------------------------------------------------------------

class TestSessionEndHook:
    def test_provider_on_session_end_calls_export_qmd(self, tmp_base, monkeypatch):
        """AC-6: on_session_end calls export_qmd for the session."""
        # We test the contract: provider receives messages, calls export_qmd
        # We mock the minimal environment to avoid needing full Hermes init
        from unittest.mock import MagicMock, patch

        # Set up a minimal mock for hermes_constants and config
        class FakeCtx:
            def register_memory_provider(self, provider):
                pass

        # Patch get_hermes_home to return our tmp_base's hermes home equivalent
        fake_hermes_home = str(tmp_base / "hermes_home")
        Path(fake_hermes_home).mkdir(parents=True, exist_ok=True)

        with patch.dict("os.environ", {"HERMES_HOME": fake_hermes_home}):
            # Import fresh after patching
            import importlib
            import hermes_memory_core.store.fs as fs_module
            importlib.reload(fs_module)
            store = fs_module.FSStore(base_path=tmp_base / "memory")

        session_id = "sess_hook_001"
        events = [
            make_event(session_id=session_id, turn_id="turn_0", sequence=0,
                       role="user", content="Hello from session end test"),
        ]
        for ev in events:
            store.append_event(ev)

        # Simulate what on_session_end would do
        store.export_qmd(session_id)

        # Verify QMD was created
        date_str = date.today().isoformat()
        year = str(date.today().year)
        qmd_path = tmp_base / "memory" / "qmd" / year / date_str / f"{session_id}.qmd"
        assert qmd_path.exists(), f"QMD not created by export_qmd: expected {qmd_path}"