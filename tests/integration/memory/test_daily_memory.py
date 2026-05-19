# Copyright 2026 David McCarty. All rights reserved.
"""Tests for hermes-memory daily memory file writer (T-027, TDD §10.1 stage 5).

Covers:
- write_daily_memory creates file at ~/.hermes/memories/YYYY-MM-DD.md
- File format: frontmatter + sections (sessions, topics, facts, decisions, open_questions)
- source_refs present on every fact/decision/question entry
- Re-run preserves manual content above <!-- AUTO-GENERATED BELOW -->
- Re-run merges auto-generated content below the marker
- New file created with proper structure (no sessions yet = proper empty state)
"""

import os
import sys
import tempfile
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "hermes_memory_core"))

from hermes_memory_core.dream.daily_memory import (
    write_daily_memory,
    _ensure_memories_dir,
    _date_file_path,
    _read_existing,
    _render_auto_content,
    MEMORIES_DIR,
    AUTO_MARKER,
)


# -------------------------------------------------------------------------- #
# Fixtures
# -------------------------------------------------------------------------- #


@pytest.fixture
def memories_tmp(tmp_path: Path, monkeypatch) -> Path:
    """Redirect ~/.hermes/memories to a temp directory for isolation."""
    tmp_memories = tmp_path / "memories"
    tmp_memories.mkdir()
    # Override at module level
    import hermes_memory_core.dream.daily_memory as dm
    monkeypatch.setattr(dm, "MEMORIES_DIR", str(tmp_memories))
    return tmp_memories


# -------------------------------------------------------------------------- #
# Tests — AC1: file structure + source_refs with 2 sessions
# -------------------------------------------------------------------------- #


def test_write_daily_memory_creates_file(memories_tmp: Path) -> None:
    """File is created at ~/.hermes/memories/YYYY-MM-DD.md."""
    d = date(2026, 5, 19)
    path = write_daily_memory(
        d,
        sessions_processed=[{"session_id": "s1", "title": "Session One", "project": "hermes-memory"}],
        facts=[{"content": "Test fact", "source_ref": "dream:test123"}],
        decisions=[],
        questions=[],
    )
    assert path.exists()
    assert path.parent.name == "memories"
    assert path.name == "2026-05-19.md"


def test_write_daily_memory_contains_sessions_section(memories_tmp: Path) -> None:
    """Sessions list appears in the output."""
    sessions = [
        {"session_id": "abc", "title": "Design review"},
        {"session_id": "def", "title": "Standup"},
    ]
    path = write_daily_memory(
        date(2026, 5, 19),
        sessions_processed=sessions,
        facts=[],
        decisions=[],
        questions=[],
    )
    content = path.read_text()
    assert "Sessions Processed" in content
    assert "[abc] Design review" in content
    assert "[def] Standup" in content


def test_write_daily_memory_contains_source_refs(memories_tmp: Path) -> None:
    """Facts, decisions, and questions each show their source_ref."""
    facts = [
        {"content": "Qwen3.6-35B is the fast model", "source_ref": "dream:dream_abc123"},
    ]
    decisions = [
        {"content": "Use LMS for local inference", "source_ref": "turn:t_abc_001"},
    ]
    questions = [
        {"content": "Should we migrate to Qdrant?", "source_ref": "session:s_xyz"},
    ]
    path = write_daily_memory(
        date(2026, 5, 19),
        sessions_processed=[{"session_id": "s1", "title": "Test"}],
        facts=facts,
        decisions=decisions,
        questions=questions,
    )
    content = path.read_text()
    assert "Qwen3.6-35B is the fast model" in content
    assert "`dream:dream_abc123`" in content
    assert "Use LMS for local inference" in content
    assert "`turn:t_abc_001`" in content
    assert "Should we migrate to Qdrant?" in content
    assert "`session:s_xyz`" in content


def test_write_daily_memory_topics_from_projects(memories_tmp: Path) -> None:
    """Topics section populated from session projects."""
    sessions = [
        {"session_id": "s1", "title": "One", "project": "hermes-memory"},
        {"session_id": "s2", "title": "Two", "project": "openclaw-workspace"},
    ]
    write_daily_memory(
        date(2026, 5, 19),
        sessions_processed=sessions,
        facts=[], decisions=[], questions=[],
    )
    content = (memories_tmp / "2026-05-19.md").read_text()
    assert "## Topics" in content
    assert "hermes-memory" in content
    assert "openclaw-workspace" in content


def test_write_daily_memory_empty_lists_render_none(memories_tmp: Path) -> None:
    """Sections with no items show _None recorded._"""
    path = write_daily_memory(
        date(2026, 5, 19),
        sessions_processed=[],
        facts=[], decisions=[], questions=[],
    )
    content = path.read_text()
    assert "_None recorded._" in content  # at least one section


# -------------------------------------------------------------------------- #
# Tests — AC2: re-run preserves manual content above marker
# -------------------------------------------------------------------------- #


def test_rerun_preserves_manual_content_above_marker(memories_tmp: Path) -> None:
    """Manual content above <!-- AUTO-GENERATED BELOW --> survives re-run."""
    d = date(2026, 5, 19)
    manual_pre = "# Notes\n\nPersonal notes for this day.\n\nSome manual thoughts.\n"
    auto_pre = f"{AUTO_MARKER}\n## Sessions Processed\n\n- [s1] First run\n"

    initial = manual_pre + auto_pre
    path = memories_tmp / "2026-05-19.md"
    path.write_text(initial, encoding="utf-8")

    # Second run — only auto portion below marker gets replaced
    write_daily_memory(
        d,
        sessions_processed=[{"session_id": "s2", "title": "Second run", "project": ""}],
        facts=[{"content": "New fact", "source_ref": "dream:dream_xyz"}],
        decisions=[], questions=[],
    )

    content = path.read_text()
    # Manual content above marker must be present
    assert "# Notes" in content
    assert "Personal notes for this day" in content
    assert "Some manual thoughts" in content
    # Marker must still be present
    assert AUTO_MARKER in content
    # New sessions must appear below marker
    assert "[s2] Second run" in content
    # Old session below marker gets replaced (re-run merges)
    assert "First run" not in content.split(AUTO_MARKER)[1]


def test_rerun_no_marker_preserves_full_content(memories_tmp: Path) -> None:
    """If no marker exists, full file is replaced (backward compat for existing files)."""
    d = date(2026, 5, 19)
    path = memories_tmp / "2026-05-19.md"
    path.write_text("Old content without marker\n", encoding="utf-8")

    write_daily_memory(
        d,
        sessions_processed=[{"session_id": "s1", "title": "New", "project": ""}],
        facts=[], decisions=[], questions=[],
    )

    content = path.read_text()
    # Full replacement since marker absent
    assert "Old content without marker" not in content
    assert "Sessions Processed" in content


# -------------------------------------------------------------------------- #
# Tests — AC3: new file created with proper structure
# -------------------------------------------------------------------------- #


def test_new_date_creates_file(memories_tmp: Path) -> None:
    """memory_dream_now for a never-dreamed date creates a proper file."""
    d = date(2025, 1, 1)  # a date with no prior history
    path = write_daily_memory(
        d,
        sessions_processed=[{"session_id": "new1", "title": "Fresh session", "project": "test"}],
        facts=[
            {"content": "Project started", "source_ref": "dream:dream_001"},
        ],
        decisions=[
            {"content": "Use local models only", "source_ref": "dream:dream_001"},
        ],
        questions=[
            {"content": "What is the architecture?", "source_ref": "dream:dream_001"},
        ],
    )
    assert path.exists()
    content = path.read_text()
    assert "# Memory — 2025-01-01" in content
    assert AUTO_MARKER in content
    assert "[new1] Fresh session" in content
    assert "Project started" in content
    assert "Use local models only" in content
    assert "What is the architecture?" in content
    # All source_refs present
    assert content.count("`dream:dream_001`") == 3


def test_date_iso_in_title(memories_tmp: Path) -> None:
    """The date ISO string appears in the file title."""
    d = date(2026, 5, 19)
    path = write_daily_memory(d, [], [], [], [])
    content = path.read_text()
    assert "# Memory — 2026-05-19" in content


# -------------------------------------------------------------------------- #
# Internal helper tests
# -------------------------------------------------------------------------- #


def test_read_existing_no_marker():
    """_read_existing returns (full_content, '') when no marker."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("All manual content\n")
        path = f.name
    try:
        before, after = _read_existing(Path(path))
        assert before == "All manual content\n"
        assert after == ""
    finally:
        os.unlink(path)


def test_read_existing_with_marker():
    """_read_existing splits at marker correctly."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        # "Above", then two newlines, then marker, then one newline, then "Below"
        f.write("Above\n\n<!-- AUTO-GENERATED BELOW -->\n\nBelow\n")
        path = f.name
    try:
        before, after = _read_existing(Path(path))
        # Content between start and marker: "Above" + trailing newline(s) up to marker
        assert before == "Above\n\n"
        assert after == "\n\nBelow\n"
    finally:
        os.unlink(path)


def test_read_existing_missing_file():
    """_read_existing on missing file returns ('', '')."""
    before, after = _read_existing(Path("/tmp/nonexistent_md_xyz123.md"))
    assert before == ""
    assert after == ""


def test_render_auto_content():
    """_render_auto_content produces expected sections."""
    out = _render_auto_content(
        date_iso="2026-05-19",
        sessions_processed=[
            {"session_id": "s1", "title": "Test", "project": "test-project"},
        ],
        facts=[{"content": "A fact", "source_ref": "dream:x"}],
        decisions=[{"content": "A decision", "source_ref": "turn:y"}],
        questions=[{"content": "A question", "source_ref": "session:z"}],
    )
    assert "# Memory — 2026-05-19" in out
    assert "## Sessions Processed" in out
    assert "[s1] Test" in out
    assert "## Topics" in out
    assert "test-project" in out
    assert "## Facts Extracted" in out
    assert "- A fact" in out
    assert "`dream:x`" in out
    assert "## Decisions Made" in out
    assert "- A decision" in out
    assert "`turn:y`" in out
    assert "## Open Questions" in out
    assert "- A question" in out
    assert "`session:z`" in out