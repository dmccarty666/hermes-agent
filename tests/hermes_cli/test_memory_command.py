"""Tests for `hermes memory init` CLI command."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest


class TestMemoryInit:
    """Tests for `hermes memory init`."""

    @pytest.fixture
    def fake_home(self, tmp_path: Path) -> Path:
        """Fake Hermes home directory."""
        fake = tmp_path / "hermes"
        fake.mkdir()
        return fake

    def test_init_creates_root_and_subdirs(self, fake_home: Path, monkeypatch: pytest.MonkeyPatch):
        """Given no ~/.hermes/memory/, `memory init` creates the full tree."""
        # Patch hermes_constants to return our fake home
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(fake_home))

        from hermes_cli.memory import memory_init

        memory_init()

        mem = fake_home / "memory"
        assert mem.exists()
        for subdir in ("raw", "qmd", "daily", "projects", "entities",
                       "dreams", "prompts", "exports", "backups", "config"):
            assert (mem / subdir).is_dir(), f"{subdir} not created"
        # index/ is NOT created by init (created by db init in T-003)
        assert not (mem / "index").exists()

    def test_init_is_idempotent(self, fake_home: Path, monkeypatch: pytest.MonkeyPatch):
        """Given ~/.hermes/memory/ already exists, re-running is a no-op."""
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(fake_home))

        from hermes_cli.memory import memory_init

        # First run
        memory_init()
        mem = fake_home / "memory"

        # Touch a marker to detect any accidental overwrites
        marker = mem / "raw" / "marker.txt"
        marker.write_text("I was here")

        # Second run must not raise
        memory_init()

        assert marker.exists()
        assert marker.read_text() == "I was here"

    def test_init_creates_starter_projects(self, fake_home: Path, monkeypatch: pytest.MonkeyPatch):
        """Given no project folders, `memory init` creates starter projects."""
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(fake_home))

        from hermes_cli.memory import memory_init

        memory_init()

        projects = fake_home / "memory" / "projects"
        for name in ("hermes", "hermes-memory", "openclaw", "local-ai-lab", "personal-ai"):
            project_dir = projects / name
            assert project_dir.is_dir(), f"project {name} not created"

    def test_init_creates_project_stub_files(
        self, fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Each starter project has the 6 stub markdown files."""
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(fake_home))

        from hermes_cli.memory import memory_init

        memory_init()

        projects = fake_home / "memory" / "projects"
        stubs = ("memory.md", "facts.md", "decisions.md",
                 "open_questions.md", "timeline.md", "sources.md")
        for name in ("hermes", "hermes-memory", "openclaw", "local-ai-lab", "personal-ai"):
            for stub in stubs:
                path = projects / name / stub
                assert path.exists(), f"{name}/{stub} not created"
                # Each stub starts with a markdown header
                content = path.read_text()
                assert content.startswith("#"), f"{name}/{stub} has no markdown header"

    def test_init_creates_readme(self, fake_home: Path, monkeypatch: pytest.MonkeyPatch):
        """`memory init` writes a README.md documenting the folder structure."""
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(fake_home))

        from hermes_cli.memory import memory_init

        memory_init()

        readme = fake_home / "memory" / "README.md"
        assert readme.exists()
        content = readme.read_text()
        # README must document the main subdirectories
        for subdir in ("raw", "qmd", "daily", "projects"):
            assert subdir in content, f"README missing '{subdir}'"

    def test_run_slash_init(self, fake_home: Path, monkeypatch: pytest.MonkeyPatch):
        """`run_slash('init')` returns a success message and creates the tree."""
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(fake_home))

        from hermes_cli.memory import run_slash

        result = run_slash("init")
        assert "ready" in result.lower() or "initialized" in result.lower()

        assert (fake_home / "memory" / "projects" / "hermes").is_dir()

    def test_run_slash_init_already_initialized(
        self, fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """`run_slash('init')` on an already-initialized tree returns without error."""
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(fake_home))

        from hermes_cli.memory import run_slash

        run_slash("init")
        # Second call must not raise
        result = run_slash("init")
        assert "already" in result.lower() or "ready" in result.lower()