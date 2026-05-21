"""Tests for hermes-memory backup command.

TDD RED phase: tests written BEFORE backup implementation.
Covers Story T-043 (Epic 6.2.1) acceptance criteria.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

# Use real paths resolved from the active profile
HERMES_HOME = Path.home() / ".hermes"
PROFILE_HOME = Path(os.environ.get("HERMES_HOME", str(HERMES_HOME)))
MEMORY_DIR = PROFILE_HOME / "memory"
BACKUP_DIR = MEMORY_DIR / "backups"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    """SHA256 of file contents, same as what backup.py records."""
    return sha256(path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backup_manager():
    """Import the backup manager from hermes_memory_core.store.backup."""
    from hermes_memory_core.store.backup import BackupManager
    return BackupManager


@pytest.fixture
def empty_backup_dir(tmp_path):
    """Return a clean temp backup directory."""
    bdir = tmp_path / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    return bdir


# ---------------------------------------------------------------------------
# AC-1: timestamped archive created at memory/backups/
# ---------------------------------------------------------------------------

def test_backup_creates_timestamped_archive(backup_manager, empty_backup_dir, monkeypatch):
    """Given memory is running, when hermes memory backup runs,
    then a timestamped archive is created at memory/backups/."""
    # Monkey-patch MEMORY_DIR to point at a temp dir with known structure
    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        mem.mkdir()
        raw = mem / "raw"
        raw.mkdir()

        # Create a sample JSONL file
        day_dir = raw / "2026" / "2026-05-21"
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "session-test123.jsonl").write_text(
            '{"event_id":"e1","session_id":"test123","role":"user","content":"hello"}\n'
        )

        # Create other dirs
        (mem / "qmd").mkdir()
        (mem / "dreams").mkdir()
        (mem / "prompts").mkdir()

        # Minimal SQLite for backup to snapshot
        db_path = mem / "index" / "memory.sqlite"
        db_path.parent.mkdir()
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)")
        conn.execute("INSERT OR IGNORE INTO sessions (id) VALUES ('test123')")
        conn.commit()
        conn.close()

        bdir = Path(td) / "backups"
        bdir.mkdir()

        bm = backup_manager(memory_dir=mem, backup_dir=bdir)
        archive_path = bm.run_backup()

        assert archive_path is not None
        assert archive_path.exists(), f"Archive not created: {archive_path}"

        # Archive is inside the backup dir
        assert archive_path.parent == bdir

        # Has timestamp pattern (YYYY-MM-DD-HHMMSS) and is .tar.gz format
        name = archive_path.name
        assert name.startswith("hermes-memory-backup-"), f"Archive name missing prefix: {name}"
        # .suffix on .tar.gz returns .gz — use name.endswith instead
        assert name.endswith(".tar.gz"), f"Archive must be .tar.gz, got: {name}"
        assert len(name) > 20, f"Archive name too short to contain timestamp: {name}"


# ---------------------------------------------------------------------------
# AC-2: manifest.json inside archive lists every file with size + content_hash
# ---------------------------------------------------------------------------

def test_archive_contains_manifest_with_sizes_and_hashes(backup_manager, empty_backup_dir, monkeypatch):
    """Given the backup archive, then manifest.json inside lists every file
    with size + content_hash."""
    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        mem.mkdir()
        raw = mem / "raw"
        raw.mkdir()

        day_dir = raw / "2026" / "2026-05-21"
        day_dir.mkdir(parents=True, exist_ok=True)
        session_file = day_dir / "session-abc.jsonl"
        session_file.write_text('{"event_id":"e1","content":"test content"}\n')

        (mem / "qmd").mkdir()
        (mem / "dreams").mkdir()

        db_path = mem / "index" / "memory.sqlite"
        db_path.parent.mkdir()
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)")
        conn.execute("INSERT OR IGNORE INTO sessions (id) VALUES ('abc')")
        conn.commit()
        conn.close()

        bdir = Path(td) / "backups"
        bdir.mkdir()

        bm = backup_manager(memory_dir=mem, backup_dir=bdir)
        archive_path = bm.run_backup()

        # Extract and check manifest
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                names = zf.namelist()
                assert "manifest.json" in names, f"manifest.json not in archive: {names}"
                manifest_bytes = zf.read("manifest.json")
        else:
            with tarfile.open(archive_path, "r:gz") as tf:
                members = [m.name for m in tf.getmembers()]
                assert "manifest.json" in members, f"manifest.json not in archive: {members}"
                manifest_bytes = tf.extractfile("manifest.json").read()

        manifest = json.loads(manifest_bytes)

        assert "files" in manifest, "manifest must have 'files' key"
        assert isinstance(manifest["files"], list), "manifest.files must be a list"
        assert len(manifest["files"]) > 0, "manifest.files must not be empty"

        # Every entry has filename, size, content_hash
        for entry in manifest["files"]:
            assert "filename" in entry, f"manifest entry missing 'filename': {entry}"
            assert "size" in entry, f"manifest entry missing 'size': {entry}"
            assert "content_hash" in entry, f"manifest entry missing 'content_hash': {entry}"
            assert isinstance(entry["size"], int), f"size must be int: {entry}"
            assert entry["size"] >= 0, f"size must be non-negative: {entry}"
            assert len(entry["content_hash"]) == 64, f"content_hash must be SHA256 (64 hex): {entry}"

        # The session JSONL we created is in the manifest
        filenames = [e["filename"] for e in manifest["files"]]
        assert any("session-abc.jsonl" in f for f in filenames), \
            f"session-abc.jsonl not in manifest: {filenames}"


# ---------------------------------------------------------------------------
# AC-3: secrets-relevant files + logs are excluded
# ---------------------------------------------------------------------------

def test_backup_excludes_logs_and_secret_files(backup_manager, empty_backup_dir):
    """Given backup runs, then secrets-relevant files (none if redaction works)
    + logs are excluded."""
    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        mem.mkdir()

        # Create logs dir with log files
        logs = mem / "logs"
        logs.mkdir()
        (logs / "agent.log").write_text("secret API key: sk-12345\n")
        (logs / "errors.log").write_text("ERROR details\n")

        # Create .env with secrets
        env_file = mem / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-FAKEFAKEFAKE\n")

        # Create backups dir (should be excluded — don't nest backups)
        backups = mem / "backups"
        backups.mkdir()
        (backups / "old.tar.gz").write_text("old backup content")

        # Create a real memory artifact that should be included
        raw = mem / "raw"
        raw.mkdir()
        day_dir = raw / "2026" / "2026-05-21"
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "session-secrets.jsonl").write_text(
            '{"event_id":"e1","content":"hello"}\n'
        )

        (mem / "qmd").mkdir()
        (mem / "dreams").mkdir()

        db_path = mem / "index" / "memory.sqlite"
        db_path.parent.mkdir()
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        bdir = Path(td) / "backups"
        bdir.mkdir()

        bm = backup_manager(memory_dir=mem, backup_dir=bdir)
        archive_path = bm.run_backup()

        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                names = zf.namelist()
        else:
            with tarfile.open(archive_path, "r:gz") as tf:
                names = [m.name for m in tf.getmembers()]

        # Logs must NOT be in archive
        assert not any("agent.log" in n for n in names), "agent.log must not be in archive"
        assert not any("errors.log" in n for n in names), "errors.log must not be in archive"

        # .env must NOT be in archive (secret file)
        assert not any(".env" in n for n in names), ".env must not be in archive"

        # Nested backups dir must NOT be in archive
        assert not any("backups" in n for n in names), "backups dir must not be in archive"

        # The real session JSONL MUST be in the archive
        assert any("session-secrets.jsonl" in n for n in names), \
            "real session data must be in archive"


# ---------------------------------------------------------------------------
# AC-4: archive can be extracted and contents verified against manifest
# ---------------------------------------------------------------------------

def test_archive_extraction_verifies_against_manifest(backup_manager, empty_backup_dir):
    """Given backup completes, then archive can be extracted and contents
    verified against manifest."""
    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        mem.mkdir()
        raw = mem / "raw"
        raw.mkdir()

        day_dir = raw / "2026" / "2026-05-21"
        day_dir.mkdir(parents=True, exist_ok=True)
        session_file = day_dir / "session-verify.jsonl"
        content = '{"event_id":"e1","content":"verify me"}\n'
        session_file.write_text(content)

        (mem / "qmd").mkdir()
        (mem / "dreams").mkdir()
        (mem / "prompts").mkdir()
        (mem / "projects").mkdir()

        db_path = mem / "index" / "memory.sqlite"
        db_path.parent.mkdir()
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)")
        conn.execute("INSERT OR IGNORE INTO sessions (id) VALUES ('verify')")
        conn.commit()
        conn.close()

        bdir = Path(td) / "backups"
        bdir.mkdir()

        bm = backup_manager(memory_dir=mem, backup_dir=bdir)
        archive_path = bm.run_backup()

        # Extract to a fresh directory
        extract_dir = Path(td) / "extracted"
        extract_dir.mkdir()

        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_dir)
                manifest_bytes = zf.read("manifest.json")
        else:
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(extract_dir)
                manifest_bytes = tf.extractfile("manifest.json").read()

        manifest = json.loads(manifest_bytes)

        # Verify every file in manifest exists on disk with matching hash
        verified = 0
        for entry in manifest["files"]:
            fpath = extract_dir / entry["filename"]
            assert fpath.exists(), f"Manifest says '{entry['filename']}' should exist but doesn't"
            actual_hash = sha256(fpath)
            assert actual_hash == entry["content_hash"], (
                f"Hash mismatch for {entry['filename']}: "
                f"expected {entry['content_hash']}, got {actual_hash}"
            )
            assert fpath.stat().st_size == entry["size"], (
                f"Size mismatch for {entry['filename']}"
            )
            verified += 1

        assert verified == len(manifest["files"]), \
            f"Only verified {verified}/{len(manifest['files'])} files"


# ---------------------------------------------------------------------------
# Qdrant snapshot is attempted (if Qdrant is running)
# ---------------------------------------------------------------------------

def test_backup_runs_without_qdrant_running(backup_manager, empty_backup_dir):
    """Given Qdrant is NOT running, when backup runs, it still succeeds
    without failing on the Qdrant snapshot attempt."""
    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        mem.mkdir()
        raw = mem / "raw"
        raw.mkdir()
        day_dir = raw / "2026" / "2026-05-21"
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "session-nqd.jsonl").write_text('{"event_id":"e1"}\n')

        (mem / "qmd").mkdir()

        db_path = mem / "index" / "memory.sqlite"
        db_path.parent.mkdir()
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        bdir = Path(td) / "backups"
        bdir.mkdir()

        bm = backup_manager(memory_dir=mem, backup_dir=bdir)

        # Patch Qdrant to be unreachable
        import unittest.mock as mock
        with mock.patch("hermes_memory_core.store.backup._qdrant_collections", return_value=[]):
            path = bm.run_backup()

        assert path is not None
        assert path.exists()
        # Should have created a valid tar.gz with the session JSONL
        with tarfile.open(path, "r:gz") as tf:
            names = [m.name for m in tf.getmembers()]
            assert any("session-nqd.jsonl" in n for n in names)


# ---------------------------------------------------------------------------
# BackupManager class interface
# ---------------------------------------------------------------------------

def test_backup_manager_init():
    """BackupManager can be instantiated with memory_dir and backup_dir."""
    from hermes_memory_core.store.backup import BackupManager

    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        bak = Path(td) / "backups"
        mem.mkdir()
        bak.mkdir()

        bm = BackupManager(memory_dir=mem, backup_dir=bak)
        assert bm.memory_dir == mem
        assert bm.backup_dir == bak


def test_backup_idempotent_no_crash(backup_manager, empty_backup_dir):
    """Running backup twice doesn't crash and produces two archives."""
    with tempfile.TemporaryDirectory() as td:
        mem = Path(td) / "memory"
        mem.mkdir()
        raw = mem / "raw"
        raw.mkdir()
        (mem / "index").mkdir()
        db_path = mem / "index" / "memory.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        bdir = Path(td) / "backups"
        bdir.mkdir()

        bm = backup_manager(memory_dir=mem, backup_dir=bdir)

        p1 = bm.run_backup()
        assert p1 is not None and p1.exists()

        p2 = bm.run_backup()
        assert p2 is not None and p2.exists()
        assert p1 != p2, "Second archive should have different timestamp"

        # Both files are valid archives
        for p in [p1, p2]:
            # .suffix on .tar.gz is .gz, so check name.endswith(".tar.gz") instead
            assert p.name.endswith(".tar.gz"), f"Archive must be .tar.gz, got: {p.name}"
            with tarfile.open(p, "r:gz") as tf:
                assert tf.getmembers()