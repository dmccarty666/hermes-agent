# Copyright 2026 David McCarty. All rights reserved.
"""Tests for project_memory.py — project memory file updater.

Tests T-028 AC:
  - Facts/decisions/questions appended to project memory files
  - Content above `<!-- AUTO-GENERATED BELOW -->` preserved
  - Auto-generated marker preserved on re-run
  - LLM fallback when no endpoint configured
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "hermes_memory_core"))

from hermes_memory_core.dream.project_memory import (
    _AUTO_GENERATED_MARKER,
    _format_facts,
    _format_decisions,
    _format_questions,
    _split_above_below_marker,
    _update_single_file,
    _ensure_project_dir,
    _read_existing,
    update_project_memory,
    read_project_memory,
)


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

class TestSplitAboveBelowMarker:
    def test_marker_exists(self):
        content = "Manual notes here\n\n<!-- AUTO-GENERATED BELOW -->\n\nGenerated facts"
        above, below = _split_above_below_marker(content)
        assert above == "Manual notes here"
        assert below == "\n\nGenerated facts"

    def test_marker_not_found(self):
        content = "All manual content"
        above, below = _split_above_below_marker(content)
        assert above == "All manual content"
        assert below == ""

    def test_empty_content(self):
        above, below = _split_above_below_marker("")
        assert above == ""
        assert below == ""

    def test_multiple_markers(self):
        content = "First marker\n<!-- AUTO-GENERATED BELOW -->\nstuff\n<!-- AUTO-GENERATED BELOW -->\nmore"
        above, below = _split_above_below_marker(content)
        # Split on first occurrence only — first marker is NOT in above (it's the delimiter)
        assert above == "First marker"
        assert "<!-- AUTO-GENERATED BELOW -->" in below


class TestFormatFacts:
    def test_empty(self):
        assert _format_facts([]) == ""

    def test_single_fact(self):
        fact = {
            "fact_text": "Qwen is fast",
            "scope": "project",
            "confidence": 0.8,
            "project": "hermes-memory",
            "entity": "Qwen",
            "tags": ["speed", "llm"],
            "source_ref": "session:s123:0",
        }
        result = _format_facts([fact])
        assert "Qwen is fast" in result
        assert "scope: project" in result
        assert "confidence: 0.8" in result
        assert "project: hermes-memory" in result
        assert "entity: Qwen" in result

    def test_text_key(self):
        """Handle 'text' key instead of 'fact_text'."""
        fact = {"text": "Spark is working", "scope": "general"}
        result = _format_facts([fact])
        assert "Spark is working" in result

    def test_source_refs_json(self):
        """Handle source_refs_json as JSON array string."""
        fact = {
            "fact_text": "Test fact",
            "scope": "general",
            "source_refs_json": '["session:s123:0"]',
        }
        result = _format_facts([fact])
        assert "session:s123:0" in result

    def test_source_ref_list(self):
        """Handle source_refs_json as actual list."""
        fact = {
            "fact_text": "Test fact",
            "scope": "general",
            "source_ref": ["session:s123:0"],
        }
        result = _format_facts([fact])
        assert "session:s123:0" in result


class TestFormatDecisions:
    def test_empty(self):
        assert _format_decisions([]) == ""

    def test_single_decision(self):
        dec = {
            "decision_text": "Use Qwen for inference",
            "rationale": "Fast and capable",
            "project": "hermes-memory",
            "owner": "david",
            "source_ref": "session:s123:1",
        }
        result = _format_decisions([dec])
        assert "Use Qwen for inference" in result
        assert "rationale: Fast and capable" in result
        assert "project: hermes-memory" in result
        assert "owner: david" in result


class TestFormatQuestions:
    def test_empty(self):
        assert _format_questions([]) == ""

    def test_single_question(self):
        q = {
            "question_text": "Should we switch to vLLM?",
            "priority": "high",
            "project": "hermes-memory",
            "source_ref": "session:s123:2",
        }
        result = _format_questions([q])
        assert "Should we switch to vLLM?" in result
        assert "priority: high" in result
        assert "project: hermes-memory" in result


# ---------------------------------------------------------------------------
# Integration tests for update_project_memory
# ---------------------------------------------------------------------------

class TestUpdateProjectMemory:
    def test_creates_project_dir(self, tmp_path, monkeypatch):
        """Project directory is created if it doesn't exist."""
        monkeypatch.setattr(
            "hermes_memory_core.dream.project_memory._PROJECTS_DIR",
            tmp_path / "projects",
        )
        project_dir = _ensure_project_dir("test-project")
        assert project_dir.exists()

    def test_update_facts_file(self, tmp_path, monkeypatch):
        """Facts are written to facts.md below the marker."""
        monkeypatch.setattr(
            "hermes_memory_core.dream.project_memory._PROJECTS_DIR",
            tmp_path / "projects",
        )
        project_dir = tmp_path / "projects" / "test-project"
        project_dir.mkdir(parents=True)

        facts_path = project_dir / "facts.md"
        new_items = "- Qwen is fast (scope: general, confidence: 0.8, source: session:s1:0)\n  - project: test-project"

        result = _update_single_file(facts_path, "facts", new_items)
        assert result is True
        content = facts_path.read_text()
        assert "Qwen is fast" in content
        assert _AUTO_GENERATED_MARKER in content

    def test_preserves_manual_content(self, tmp_path, monkeypatch):
        """Content above marker is preserved."""
        monkeypatch.setattr(
            "hermes_memory_core.dream.project_memory._PROJECTS_DIR",
            tmp_path / "projects",
        )
        project_dir = tmp_path / "projects" / "test-project"
        project_dir.mkdir(parents=True)

        facts_path = project_dir / "facts.md"
        facts_path.write_text(
            "# Manual notes\n\nThese are my manual notes.\n\n<!-- AUTO-GENERATED BELOW -->\n\n## Facts\n- Old fact\n"
        )

        new_items = "- New fact"
        _update_single_file(facts_path, "facts", new_items)

        content = facts_path.read_text()
        assert "Manual notes" in content
        assert "These are my manual notes." in content
        assert "New fact" in content
        assert "Old fact" in content  # preserved below marker

    def test_marker_preserved_on_rerun(self, tmp_path, monkeypatch):
        """Re-running preserves the marker."""
        monkeypatch.setattr(
            "hermes_memory_core.dream.project_memory._PROJECTS_DIR",
            tmp_path / "projects",
        )
        project_dir = tmp_path / "projects" / "test-project"
        project_dir.mkdir(parents=True)

        facts_path = project_dir / "facts.md"
        facts_path.write_text(
            "# Facts\n\n<!-- AUTO-GENERATED BELOW -->\n\n## Facts\n- First fact\n"
        )

        new_items = "- Second fact"
        _update_single_file(facts_path, "facts", new_items)

        content = facts_path.read_text()
        assert content.count(_AUTO_GENERATED_MARKER) == 1
        assert "- First fact" in content
        assert "- Second fact" in content

    def test_update_project_memory_no_llm(self, tmp_path, monkeypatch):
        """Without LLM endpoint, falls back to append below marker."""
        monkeypatch.setattr(
            "hermes_memory_core.dream.project_memory._PROJECTS_DIR",
            tmp_path / "projects",
        )

        facts = [
            {
                "fact_text": "Qwen is fast",
                "scope": "general",
                "confidence": 0.8,
                "project": "test-project",
                "entity": "Qwen",
                "source_ref": "session:s1:0",
            }
        ]

        result = update_project_memory(
            project="test-project",
            new_facts=facts,
            new_decisions=[],
            new_questions=[],
            llm_endpoint=None,
            dry_run=False,
        )

        assert result["facts_updated"] == 1
        assert "facts.md" in result["files_modified"]

        facts_path = tmp_path / "projects" / "test-project" / "facts.md"
        content = facts_path.read_text()
        assert "Qwen is fast" in content
        assert _AUTO_GENERATED_MARKER in content

    def test_update_project_memory_decisions(self, tmp_path, monkeypatch):
        """Decisions are written to decisions.md."""
        monkeypatch.setattr(
            "hermes_memory_core.dream.project_memory._PROJECTS_DIR",
            tmp_path / "projects",
        )

        decisions = [
            {
                "decision_text": "Use Qwen for inference",
                "rationale": "Fast and capable",
                "project": "test-project",
                "source_ref": "session:s1:1",
            }
        ]

        result = update_project_memory(
            project="test-project",
            new_facts=[],
            new_decisions=decisions,
            new_questions=[],
            llm_endpoint=None,
        )

        assert "decisions.md" in result["files_modified"]
        dec_path = tmp_path / "projects" / "test-project" / "decisions.md"
        content = dec_path.read_text()
        assert "Use Qwen for inference" in content

    def test_update_project_memory_questions(self, tmp_path, monkeypatch):
        """Open questions are written to open_questions.md."""
        monkeypatch.setattr(
            "hermes_memory_core.dream.project_memory._PROJECTS_DIR",
            tmp_path / "projects",
        )

        questions = [
            {
                "question_text": "Should we migrate to vLLM?",
                "priority": "high",
                "project": "test-project",
                "source_ref": "session:s1:2",
            }
        ]

        result = update_project_memory(
            project="test-project",
            new_facts=[],
            new_decisions=[],
            new_questions=questions,
            llm_endpoint=None,
        )

        assert "open_questions.md" in result["files_modified"]
        q_path = tmp_path / "projects" / "test-project" / "open_questions.md"
        content = q_path.read_text()
        assert "Should we migrate to vLLM?" in content

    def test_dry_run_no_write(self, tmp_path, monkeypatch):
        """dry_run=True doesn't write any files."""
        monkeypatch.setattr(
            "hermes_memory_core.dream.project_memory._PROJECTS_DIR",
            tmp_path / "projects",
        )

        # Pre-create the project dir and an EXISTING facts.md file with content
        # that would be overwritten by new facts.
        # In dry_run, files_modified should still be populated (report intent),
        # but the file should NOT be written.
        project_dir = tmp_path / "projects" / "test-project"
        project_dir.mkdir(parents=True)
        facts_path = project_dir / "facts.md"
        facts_path.write_text("# Facts\n\n<!-- AUTO-GENERATED BELOW -->\n\n## Facts\n- Old fact\n")

        result = update_project_memory(
            project="test-project",
            new_facts=[{"fact_text": "Test", "scope": "general", "confidence": 0.5}],
            new_decisions=[],
            new_questions=[],
            llm_endpoint=None,
            dry_run=True,
        )

        assert result["facts_updated"] == 1
        # dry_run should report what would be modified but not write
        assert "facts.md" in result["files_modified"]
        # File content should be unchanged (old fact preserved, not overwritten)
        content = facts_path.read_text()
        assert "- Old fact" in content
        assert "- Test" not in content

    def test_empty_lists_no_files_modified(self, tmp_path, monkeypatch):
        """Empty lists don't create files (memory.md is still written for consistency)."""
        monkeypatch.setattr(
            "hermes_memory_core.dream.project_memory._PROJECTS_DIR",
            tmp_path / "projects",
        )

        result = update_project_memory(
            project="test-project",
            new_facts=[],
            new_decisions=[],
            new_questions=[],
            llm_endpoint=None,
        )

        # memory.md is always written for consistency even when empty
        assert result["files_modified"] == ["memory.md"]
        assert result["facts_updated"] == 0

    def test_read_project_memory(self, tmp_path, monkeypatch):
        """read_project_memory returns all existing files."""
        monkeypatch.setattr(
            "hermes_memory_core.dream.project_memory._PROJECTS_DIR",
            tmp_path / "projects",
        )
        project_dir = tmp_path / "projects" / "test-project"
        project_dir.mkdir(parents=True)
        (project_dir / "facts.md").write_text("# Facts\n\ncontent")
        (project_dir / "decisions.md").write_text("# Decisions\n\ncontent")

        result = read_project_memory("test-project")
        assert "facts.md" in result
        assert "decisions.md" in result
        assert result["facts.md"] == "# Facts\n\ncontent"

    def test_read_project_memory_missing_project(self, tmp_path, monkeypatch):
        """read_project_memory handles missing project gracefully."""
        monkeypatch.setattr(
            "hermes_memory_core.dream.project_memory._PROJECTS_DIR",
            tmp_path / "projects",
        )
        result = read_project_memory("nonexistent")
        assert result == {}


# ---------------------------------------------------------------------------
# Auto-generated marker preservation test
# ---------------------------------------------------------------------------

class TestAutoGeneratedMarker:
    def test_marker_added_if_missing(self, tmp_path, monkeypatch):
        """If existing file has no marker, it is added."""
        monkeypatch.setattr(
            "hermes_memory_core.dream.project_memory._PROJECTS_DIR",
            tmp_path / "projects",
        )
        project_dir = tmp_path / "projects" / "test-project"
        project_dir.mkdir(parents=True)
        facts_path = project_dir / "facts.md"
        facts_path.write_text("# Facts\n\nSome manual content")

        _update_single_file(facts_path, "facts", "- New fact")

        content = facts_path.read_text()
        assert _AUTO_GENERATED_MARKER in content

    def test_no_duplicate_markers(self, tmp_path, monkeypatch):
        """Marker appears exactly once after multiple runs."""
        monkeypatch.setattr(
            "hermes_memory_core.dream.project_memory._PROJECTS_DIR",
            tmp_path / "projects",
        )
        project_dir = tmp_path / "projects" / "test-project"
        project_dir.mkdir(parents=True)
        facts_path = project_dir / "facts.md"

        for i in range(3):
            _update_single_file(facts_path, "facts", f"- Fact {i}")

        content = facts_path.read_text()
        assert content.count(_AUTO_GENERATED_MARKER) == 1