"""Tests for narrative.py (Phase 5, Story T-030).

Covers:
  - _write_thread + _read_thread_file round-trip
  - Rolling 5-exchange window (write 7, read 5)
  - Thread directory creation
  - Retention cleanup
  - Tool name tracking in metadata

Run with: scripts/run_tests.sh tests/integration/memory/test_narrative.py
"""

from __future__ import annotations

import importlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add hermes-agent root to sys.path so hermes_memory_core + hermes_constants resolve
_AGENT_ROOT = Path(__file__).parents[3]
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))


def load_narrative_module():
    """Load narrative.py via spec_from_file_location (no __init__.py needed)."""
    # Pre-load hermes_constants so narrative.py can import it
    hc_spec = importlib.util.spec_from_file_location(
        "hermes_constants",
        str(_AGENT_ROOT / "hermes_constants.py"),
    )
    hc_mod = importlib.util.module_from_spec(hc_spec)
    sys.modules["hermes_constants"] = hc_mod
    hc_spec.loader.exec_module(hc_mod)

    # Load narrative.py
    narrative_path = _AGENT_ROOT / "plugins" / "memory" / "hermes-local" / "narrative.py"
    spec = importlib.util.spec_from_file_location("narrative", str(narrative_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plugins.memory.hermes_local.narrative"] = mod
    spec.loader.exec_module(mod)
    return mod


narrative_mod = load_narrative_module()


# --------------------------------------------------------------------------:
# Fixtures
# --------------------------------------------------------------------------:

@pytest.fixture
def fake_hermes_home(tmp_path: Path, monkeypatch) -> Path:
    """Fake HERMES_HOME so thread files go to tmp_path."""
    fake_home = tmp_path / "hermes_home"
    fake_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    # Reset the module's cached path computation
    narrative_mod._reset_path_cache()
    return fake_home


# --------------------------------------------------------------------------:
# Test: _write_thread + _read_thread_file round-trip (basic)
# --------------------------------------------------------------------------:

def test_write_then_read_single_turn(fake_hermes_home: Path) -> None:
    """Write 1 turn, read it back — focus, turn_count, exchange all correct."""
    session_id = "test-session-1"
    turns = [{"user": "Hello, how are you?", "ai": "I'm doing great, thanks!"}]

    narrative_mod._write_thread(session_id, turns)
    focus, exchanges, turn_count = narrative_mod._read_thread_file(session_id)

    assert turn_count == 1
    assert len(exchanges) == 1
    assert "Hello, how are you?" in exchanges[0]["user"]
    assert "I'm doing great, thanks!" in exchanges[0]["ai"]


def test_write_then_read_preserves_user_and_ai(fake_hermes_home: Path) -> None:
    """Write 3 turns, read back — all 3 exchanges present in order."""
    session_id = "test-session-2"
    turns = [
        {"user": "First question", "ai": "First answer"},
        {"user": "Second question", "ai": "Second answer"},
        {"user": "Third question", "ai": "Third answer"},
    ]

    narrative_mod._write_thread(session_id, turns)
    focus, exchanges, turn_count = narrative_mod._read_thread_file(session_id)

    assert turn_count == 3
    assert len(exchanges) == 3
    assert exchanges[0]["user"] == "First question"
    assert exchanges[0]["ai"] == "First answer"
    assert exchanges[1]["user"] == "Second question"
    assert exchanges[1]["ai"] == "Second answer"
    assert exchanges[2]["user"] == "Third question"
    assert exchanges[2]["ai"] == "Third answer"


# --------------------------------------------------------------------------:
# Test: Rolling window — write 7 turns, read exactly last 5
# --------------------------------------------------------------------------:

def test_rolling_window_7_turns_returns_5(fake_hermes_home: Path) -> None:
    """Write 7 exchanges, read thread — assert exactly last 5 are returned.

    This is the primary DoD acceptance criterion:
    "Unit tests: write 7 turns → read thread → assert exactly last 5 exchanges returned"
    """
    session_id = "test-rolling-7"
    turns = [
        {"user": f"Turn {i} user", "ai": f"Turn {i} assistant"}
        for i in range(1, 8)
    ]

    narrative_mod._write_thread(session_id, turns)
    focus, exchanges, turn_count = narrative_mod._read_thread_file(session_id)

    # turn_count reflects total turns written (not just what's in the window)
    assert turn_count == 7
    # But only last 5 exchanges are in the file
    assert len(exchanges) == 5

    # First exchange should be Turn 3 (oldest of the 5 kept)
    assert exchanges[0]["user"] == "Turn 3 user"
    assert exchanges[0]["ai"] == "Turn 3 assistant"
    # Last exchange should be Turn 7
    assert exchanges[4]["user"] == "Turn 7 user"
    assert exchanges[4]["ai"] == "Turn 7 assistant"

    # Turn 1 and Turn 2 must NOT be in the window
    all_users = [ex["user"] for ex in exchanges]
    assert "Turn 1 user" not in all_users
    assert "Turn 2 user" not in all_users


def test_rolling_window_exactly_5_turns_no_trim(fake_hermes_home: Path) -> None:
    """Write exactly 5 turns — all 5 should be present (no trimming)."""
    session_id = "test-exact-5"
    turns = [
        {"user": f"T{i}", "ai": f"A{i}"} for i in range(1, 6)
    ]

    narrative_mod._write_thread(session_id, turns)
    focus, exchanges, turn_count = narrative_mod._read_thread_file(session_id)

    assert turn_count == 5
    assert len(exchanges) == 5
    all_users = [ex["user"] for ex in exchanges]
    for i in range(1, 6):
        assert f"T{i}" in all_users


def test_rolling_window_3_turns_all_kept(fake_hermes_home: Path) -> None:
    """Write 3 turns (below window size) — all 3 present."""
    session_id = "test-under-5"
    turns = [
        {"user": "Alpha", "ai": "Beta"},
        {"user": "Gamma", "ai": "Delta"},
        {"user": "Epsilon", "ai": "Zeta"},
    ]

    narrative_mod._write_thread(session_id, turns)
    focus, exchanges, turn_count = narrative_mod._read_thread_file(session_id)

    assert turn_count == 3
    assert len(exchanges) == 3


# --------------------------------------------------------------------------:
# Test: Thread file doesn't exist → sensible defaults
# --------------------------------------------------------------------------:

def test_read_missing_file_returns_defaults(fake_hermes_home: Path) -> None:
    """Reading a non-existent thread file returns empty defaults."""
    focus, exchanges, turn_count = narrative_mod._read_thread_file("nonexistent-session")
    assert focus == ""
    assert exchanges == []
    assert turn_count == 0


# --------------------------------------------------------------------------:
# Test: Thread directory created automatically
# --------------------------------------------------------------------------:

def test_thread_dir_created_on_write(fake_hermes_home: Path) -> None:
    """Writing a thread creates the SESSION-THREAD directory if it doesn't exist."""
    session_id = "test-dir-create"
    thread_dir = narrative_mod._thread_dir()

    assert not thread_dir.exists()

    narrative_mod._write_thread(session_id, [{"user": "x", "ai": "y"}])

    assert thread_dir.exists()
    assert thread_dir.is_dir()

    thread_file = thread_dir / f"{session_id}.md"
    assert thread_file.exists()


# --------------------------------------------------------------------------:
# Test: Tool names tracked in frontmatter
# --------------------------------------------------------------------------:

def test_tool_names_in_thread_file(fake_hermes_home: Path) -> None:
    """Write thread with tool list, read back and verify Tools Used Recently."""
    session_id = "test-tools"

    # _build_thread_content is internal — test the full round-trip via the
    # public content builder
    content = narrative_mod._build_thread_content(
        focus="Testing the agent",
        exchanges=[{"time": "10:30", "user": "Query", "ai": "Response"}],
        turn_count=1,
        session_start="2026-05-19 10:30 CDT",
        tools_used=["memory_query", "memory_write", "calculator"],
    )

    assert "memory_query, memory_write, calculator" in content
    assert "## Tools Used Recently" in content


def test_tools_list_empty_shows_none(fake_hermes_home: Path) -> None:
    """No tools used → 'none' appears in Tools Used Recently."""
    content = narrative_mod._build_thread_content(
        focus="",
        exchanges=[],
        turn_count=0,
        session_start="2026-05-19 10:30 CDT",
        tools_used=[],
    )
    assert "none" in content


# --------------------------------------------------------------------------:
# Test: Truncation of long content
# --------------------------------------------------------------------------:

def test_long_content_truncated_on_write(fake_hermes_home: Path) -> None:
    """Content exceeding SNIPPET_LEN is truncated before writing."""
    session_id = "test-truncate"
    long_user = "A" * 1000
    long_ai = "B" * 1000

    narrative_mod._write_thread(session_id, [{"user": long_user, "ai": long_ai}])
    focus, exchanges, turn_count = narrative_mod._read_thread_file(session_id)

    assert len(exchanges) == 1
    # User and AI should be truncated to SNIPPET_LEN (500)
    assert len(exchanges[0]["user"]) <= narrative_mod.SNIPPET_LEN
    assert len(exchanges[0]["ai"]) <= narrative_mod.SNIPPET_LEN


# --------------------------------------------------------------------------:
# Test: _cleanup_old_threads
# --------------------------------------------------------------------------:

def test_cleanup_removes_old_threads(fake_hermes_home: Path) -> None:
    """Thread files older than retention_days are deleted."""
    import time
    thread_dir = narrative_mod._thread_dir()
    thread_dir.mkdir(parents=True, exist_ok=True)

    # Create an old thread file
    old_file = thread_dir / "old-session.md"
    old_file.write_text("old content", "utf-8")

    # Set its mtime to 40 days ago
    old_mtime = datetime.now(timezone.utc).timestamp() - (40 * 86400)
    import os
    os.utime(old_file, (old_mtime, old_mtime))

    # Create a recent thread file
    recent_file = thread_dir / "recent-session.md"
    recent_file.write_text("recent content", "utf-8")

    narrative_mod._cleanup_old_threads(retention_days=30)

    assert not old_file.exists(), "Old thread file should be deleted"
    assert recent_file.exists(), "Recent thread file should remain"


def test_cleanup_preserves_recent_threads(fake_hermes_home: Path) -> None:
    """Thread files newer than retention_days are NOT deleted."""
    thread_dir = narrative_mod._thread_dir()
    thread_dir.mkdir(parents=True, exist_ok=True)

    recent_file = thread_dir / "recent-session.md"
    recent_file.write_text("recent content", "utf-8")

    narrative_mod._cleanup_old_threads(retention_days=30)

    assert recent_file.exists()


# --------------------------------------------------------------------------:
# Test: Focus line preserved
# --------------------------------------------------------------------------:

def test_focus_in_thread_content(fake_hermes_home: Path) -> None:
    """Focus string appears in the ## Current Focus section."""
    content = narrative_mod._build_thread_content(
        focus="Working on the memory plugin",
        exchanges=[],
        turn_count=0,
        session_start="2026-05-19 10:30 CDT",
        tools_used=[],
    )
    assert "## Current Focus" in content
    assert "Working on the memory plugin" in content


def test_focus_defaults_to_heartbeat_when_empty(fake_hermes_home: Path) -> None:
    """Empty focus shows '(system/heartbeat)' as default."""
    content = narrative_mod._build_thread_content(
        focus="",
        exchanges=[],
        turn_count=0,
        session_start="2026-05-19 10:30 CDT",
        tools_used=[],
    )
    assert "(system/heartbeat)" in content


# --------------------------------------------------------------------------:
# Test: session_id in file path
# --------------------------------------------------------------------------:

def test_different_session_ids_have_separate_files(fake_hermes_home: Path) -> None:
    """Two different session IDs produce two separate thread files."""
    narrative_mod._write_thread("session-A", [{"user": "A-user", "ai": "A-ai"}])
    narrative_mod._write_thread("session-B", [{"user": "B-user", "ai": "B-ai"}])

    _, exchanges_a, _ = narrative_mod._read_thread_file("session-A")
    _, exchanges_b, _ = narrative_mod._read_thread_file("session-B")

    assert exchanges_a[0]["user"] == "A-user"
    assert exchanges_b[0]["user"] == "B-user"


# --------------------------------------------------------------------------:
# Test: _thread_path and _thread_dir helpers
# --------------------------------------------------------------------------:

def test_thread_path_format(fake_hermes_home: Path) -> None:
    """_thread_path returns ~/.hermes/SESSION-THREAD/{session_id}.md."""
    path = narrative_mod._thread_path("abc123")
    assert path.name == "abc123.md"
    assert path.parent.name == "SESSION-THREAD"


def test_thread_dir_points_to_hermes_home(fake_hermes_home: Path) -> None:
    """_thread_dir is under HERMES_HOME, not hardcoded."""
    td = narrative_mod._thread_dir()
    assert td.parent == fake_hermes_home