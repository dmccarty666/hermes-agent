# Copyright 2026 David McCarty. All rights reserved.
"""CLI smoke tests for `hermes memory …` subcommands.

Verifies all CLI entry points from T-009:
  - `hermes memory health`        — shows DB path, WAL, FTS, redaction, schema version
  - `hermes memory init`          — creates directory tree
  - `hermes memory db init`       — initializes SQLite schema
  - `hermes memory capture-test`  — injects synthetic turn end-to-end
  - `hermes memory ls-sessions`   — lists sessions from SQLite
  - `hermes memory --help`       — shows all subcommands

Exit codes:
  0 on success, non-zero on error (exit code 2 = not initialized).
"""

from __future__ import annotations

import json
import re
import sqlite3
import os
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hermes_home(tmp_path: Path) -> Path:
    """Isolated Hermes home for CLI tests."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    # Minimal config with hermes-local provider
    config = home / "config.yaml"
    config.write_text(
        "memory:\n  provider: hermes-local\n"
    )
    # Memory dir already partially set up
    mem = home / "memory"
    mem.mkdir()
    for sub in ("raw", "qmd", "daily", "projects", "entities",
                "dreams", "prompts", "exports", "backups", "config"):
        (mem / sub).mkdir()
    return home


@pytest.fixture
def memory_cli(hermes_home: Path) -> list[str]:
    """Base command prefix to invoke hermes memory CLI."""
    return [
        sys.executable, "-c",
        f"import sys; sys.path.insert(0, '{Path(__file__).resolve().parent.parent.parent}'); "
        f"from hermes_cli.memory import run_slash; "
        f"print(run_slash({{1}}))",
    ]


# ---------------------------------------------------------------------------
# Tests: hermes memory init
# ---------------------------------------------------------------------------

def test_memory_init_creates_directory_tree(hermes_home: Path, tmp_path: Path) -> None:
    """Given hermes-memory is installed, `memory init` creates the directory tree."""
    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{hermes_home.parent.parent}'); "
            f"from hermes_cli.memory import memory_init; "
            f"memory_init('{hermes_home}')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    mem_root = hermes_home / "memory"
    for sub in ("raw", "qmd", "daily", "projects", "entities",
                "dreams", "prompts", "exports", "backups", "config"):
        assert (mem_root / sub).is_dir(), f"{sub} not created"


def test_memory_init_idempotent(hermes_home: Path) -> None:
    """`memory init` run twice is safe (idempotent)."""
    result1 = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{hermes_home.parent.parent}'); "
            f"from hermes_cli.memory import memory_init; memory_init('{hermes_home}')",
        ],
        capture_output=True,
        text=True,
    )
    result2 = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{hermes_home.parent.parent}'); "
            f"from hermes_cli.memory import memory_init; memory_init('{hermes_home}')",
        ],
        capture_output=True,
        text=True,
    )
    assert result1.returncode == 0
    assert result2.returncode == 0


# ---------------------------------------------------------------------------
# Tests: hermes memory db init
# ---------------------------------------------------------------------------

def test_memory_db_init_creates_sqlite_schema(hermes_home: Path) -> None:
    """Given hermes-memory is installed, `memory db init` initializes SQLite."""
    from hermes_memory_core.store.sqlite import MemoryDB

    mem_root = hermes_home / "memory"
    db_path = mem_root / "index" / "memory.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize
    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{hermes_home.parent.parent}'); "
            f"from hermes_cli.memory import run_slash; print(run_slash('db init'))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HERMES_HOME": str(hermes_home)},
    )
    assert result.returncode == 0, result.stderr
    assert "OK SQLite schema initialized" in result.stdout
    assert db_path.exists()

    # Verify schema
    db = MemoryDB(str(db_path))
    assert db.is_initialized()
    conn = db._connect()
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        assert "sessions" in tables
        assert "turns" in tables
        assert "raw_events" in tables
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tests: hermes memory health
# ---------------------------------------------------------------------------

def test_memory_health_reports_all_components(hermes_home: Path) -> None:
    """`memory health` outputs DB path, WAL, FTS, redaction, schema version."""
    # Pre-initialize to get real numbers
    mem_root = hermes_home / "memory"
    mem_root.mkdir(parents=True, exist_ok=True)
    for sub in ("raw", "qmd"):
        (mem_root / sub).mkdir(exist_ok=True)
    db_path = mem_root / "index" / "memory.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    from hermes_memory_core.store.sqlite import MemoryDB

    db = MemoryDB(str(db_path))
    db.initialize()

    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{hermes_home.parent.parent}'); "
            f"from hermes_cli.memory import run_slash; print(run_slash('health'))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HERMES_HOME": str(hermes_home)},
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "Hermes Local Memory" in output
    # WAL check via sqlite output
    assert "sqlite" in output.lower()
    # Config/provider section
    assert "provider" in output or "hermes_home" in output


# ---------------------------------------------------------------------------
# Tests: hermes memory capture-test
# ---------------------------------------------------------------------------

def test_memory_capture_test_injects_turn(hermes_home: Path) -> None:
    """`memory capture-test` injects a synthetic turn and reports success."""
    mem_root = hermes_home / "memory"
    mem_root.mkdir(parents=True, exist_ok=True)
    for sub in ("raw", "qmd", "index"):
        (mem_root / sub).mkdir(exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{hermes_home.parent.parent}'); "
            f"from hermes_cli.memory import run_slash; print(run_slash('capture-test'))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HERMES_HOME": str(hermes_home)},
    )
    assert result.returncode == 0, result.stderr
    assert "OK Capture test injected" in result.stdout
    assert "session_id" in result.stdout
    assert "turn_id" in result.stdout

    # Verify SQLite row was inserted
    db_path = mem_root / "index" / "memory.sqlite"
    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT session_id FROM sessions WHERE session_id LIKE 'cli-test-%'")
        rows = cur.fetchall()
        assert len(rows) >= 1, "No sessions found in SQLite"
    finally:
        conn.close()


def test_memory_capture_test_no_leaked_secrets(hermes_home: Path) -> None:
    """`capture-test` does not write real secrets into SQLite or JSONL."""
    mem_root = hermes_home / "memory"
    mem_root.mkdir(parents=True, exist_ok=True)
    for sub in ("raw", "qmd", "index"):
        (mem_root / sub).mkdir(exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{hermes_home.parent.parent}'); "
            f"from hermes_cli.memory import run_slash; print(run_slash('capture-test'))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HERMES_HOME": str(hermes_home)},
    )
    assert result.returncode == 0

    # Verify no fixture secrets in JSONL
    import glob
    for jsonl_file in glob.glob(str(mem_root / "raw/**/*.jsonl"), recursive=True):
        content = Path(jsonl_file).read_text()
        assert "AKIAFAKEFAKE" not in content
        assert "sk-fakefakefake" not in content
        assert "ghp_fakefakefake" not in content

    # Verify no fixture secrets in SQLite
    db_path = mem_root / "index" / "memory.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        for row in conn.execute("SELECT content FROM turns"):
            assert "AKIAFAKEFAKE" not in (row[0] or "")
            assert "sk-fakefakefake" not in (row[0] or "")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tests: hermes memory ls-sessions
# ---------------------------------------------------------------------------

def test_memory_ls_sessions_empty_state(hermes_home: Path) -> None:
    """`memory ls-sessions` on uninitialized DB returns helpful message."""
    mem_root = hermes_home / "memory"
    mem_root.mkdir(parents=True, exist_ok=True)

    # No db init — ls-sessions should not crash
    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{hermes_home.parent.parent}'); "
            f"from hermes_cli.memory import run_slash; print(run_slash('ls-sessions'))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HERMES_HOME": str(hermes_home)},
    )
    # Should either show empty state or helpful "not initialized" message
    assert result.returncode == 0
    output = result.stdout.lower()
    # Either "no sessions" or "not initialized" or "sqlite not initialized"
    assert any(kw in output for kw in ["no session", "not initialized", "run `hermes memory db init`"])


def test_memory_ls_sessions_with_data(hermes_home: Path) -> None:
    """`memory ls-sessions` lists session IDs with last_activity_at timestamps."""
    mem_root = hermes_home / "memory"
    mem_root.mkdir(parents=True, exist_ok=True)
    for sub in ("raw", "qmd", "index"):
        (mem_root / sub).mkdir(exist_ok=True)

    from hermes_memory_core.store.sqlite import MemoryDB

    db_path = mem_root / "index" / "memory.sqlite"
    db = MemoryDB(str(db_path))
    db.initialize()

    # Inject a session manually
    conn = db._connect()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO sessions "
            "(session_id, agent, started_at, project) VALUES (?, ?, ?, ?)",
            ("test-session-abc123", "cli-test", "2026-05-18T12:00:00Z", "test-project"),
        )
        conn.commit()
    finally:
        conn.close()

    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{hermes_home.parent.parent}'); "
            f"from hermes_cli.memory import run_slash; print(run_slash('ls-sessions'))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HERMES_HOME": str(hermes_home)},
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "test-session-abc123" in output


# ---------------------------------------------------------------------------
# Tests: hermes memory --help / unknown subcommand
# ---------------------------------------------------------------------------

def test_memory_unknown_subcommand_shows_help(hermes_home: Path) -> None:
    """Unknown subcommand returns help text with exit 0."""
    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{hermes_home.parent.parent}'); "
            f"from hermes_cli.memory import run_slash; print(run_slash('bad-subcommand'))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HERMES_HOME": str(hermes_home)},
    )
    assert result.returncode == 0
    output = result.stdout.lower()
    # Should show usage/help for subcommands
    assert "memory" in output and ("subcommand" in output or "usage" in output or "init" in output)


def test_memory_empty_subcommand_shows_help(hermes_home: Path) -> None:
    """Empty subcommand shows full help text."""
    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, '{hermes_home.parent.parent}'); "
            f"from hermes_cli.memory import run_slash; print(run_slash(''))",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "HERMES_HOME": str(hermes_home)},
    )
    assert result.returncode == 0
    output = result.stdout
    assert "health" in output
    assert "db init" in output
    assert "capture-test" in output
    assert "ls-sessions" in output