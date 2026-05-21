"""Backup manager for hermes-local memory.

Creates timestamped archives of all memory artifacts with an accurate
manifest. Archives exclude logs and secret files.

Archive format: tar.gz
Manifest format: JSON with {files: [{filename, size, content_hash, archived_at}]}

Artifact inclusion:
  - raw JSONL sessions
  - QMD exports
  - SQLite .backup snapshot (via sqlite3.backup API)
  - project memory files
  - dream reports
  - prompts
  - config.yaml (not .env)

Exclusions:
  - logs/
  - backups/ (don't nest backups)
  - .env (secrets)
  - *.db-wal, *.db-shm, *.db-journal (SQLite transient)
  - hermes-agent repo
  - __pycache__, .git, node_modules

Qdrant: attempts snapshot if Qdrant is running at localhost:6333.
If Qdrant is unavailable, backup proceeds without it (no failure).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import requests

logger = logging.getLogger(__name__)

# Directories to skip entirely in backup walk
_EXCLUDED_DIRS = frozenset({
    "logs",
    "backups",
    "__pycache__",
    ".git",
    "node_modules",
    "checkpoints",
    ".pytest_cache",
})

# File suffixes to skip (transient SQLite sidecars)
_EXCLUDED_SUFFIXES = frozenset({
    ".db-wal",
    ".db-shm",
    ".db-journal",
    ".pyc",
    ".pyo",
})

# File names that are always excluded (runtime state / secrets)
_EXCLUDED_NAMES = frozenset({
    ".env",
    "auth.json",
    "gateway.pid",
    "cron.pid",
    "state.db",
    ".DS_Store",
})

# Subdirectory of memory/ that we back up
_INCLUDED_SUBDIRS = frozenset({
    "raw",
    "qmd",
    "dreams",
    "prompts",
    "projects",
    "entities",
    "daily",
    "index",
    "config",
    "exports",
    "memory",
})


def _sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(memory_dir: Path) -> Iterator[tuple[Path, str]]:
    """Walk memory_dir and yield (absolute_path, relative_path_str) for each
    file that should be included in the backup.

    Excludes: _EXCLUDED_DIRS, _EXCLUDED_SUFFIXES, _EXCLUDED_NAMES, and any
    file outside the standard memory subdirectories.
    """
    if not memory_dir.is_dir():
        return

    for dirpath, dirnames, filenames in os.walk(memory_dir):
        dp = Path(dirpath)

        # Prune excluded directories in-place so os.walk doesn't descend
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]

        for fname in filenames:
            fpath = dp / fname

            if fname in _EXCLUDED_NAMES:
                continue
            if any(fname.endswith(s) for s in _EXCLUDED_SUFFIXES):
                continue

            try:
                rel = fpath.relative_to(memory_dir)
            except ValueError:
                # Not under memory_dir for some reason — skip
                continue

            # Only include files under known memory subdirs
            if rel.parts[0] not in _INCLUDED_SUBDIRS:
                continue

            yield fpath, str(rel)


def _safe_copy_sqlite(src: Path, dst: Path) -> bool:
    """Copy a SQLite database safely using the backup() API.

    Produces a consistent snapshot even while the DB is being written to.
    Falls back to raw copy2 on failure.
    """
    try:
        conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        backup_conn = sqlite3.connect(str(dst))
        conn.backup(backup_conn)
        backup_conn.close()
        conn.close()
        return True
    except Exception as exc:
        logger.debug("SQLite backup API failed for %s: %s — falling back to raw copy", src, exc)
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as exc2:
            logger.warning("Raw SQLite copy also failed for %s: %s", src, exc2)
            return False


def _qdrant_collections(base_url: str = "http://localhost:6333") -> list[str]:
    """Return list of Qdrant collection names, or [] if unavailable."""
    try:
        resp = requests.get(f"{base_url}/collections", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return [c["name"] for c in data.get("result", {}).get("collections", [])]
    except Exception:
        pass
    return []


def _snapshot_qdrant_collection(
    collection_name: str,
    backup_dir: Path,
    base_url: str = "http://localhost:6333",
) -> Path | None:
    """Take a snapshot of one Qdrant collection. Returns path to snapshot dir,
    or None if snapshot fails."""
    try:
        resp = requests.post(
            f"{base_url}/collections/{collection_name}/snapshots",
            timeout=60,
        )
        if resp.status_code not in (200, 201):
            logger.debug("Qdrant snapshot failed for %s: HTTP %s", collection_name, resp.status_code)
            return None

        snapshot_path = resp.json().get("result", {}).get("location")
        if not snapshot_path:
            return None

        # snapshot_path is like /snapshots/...
        # Download it
        snap_resp = requests.get(f"{base_url}/collections/{collection_name}/snapshots/{snapshot_path.split('/')[-1]}", timeout=120)
        if snap_resp.status_code != 200:
            return None

        out = backup_dir / f"qdrant-{collection_name}.snapshot.tar.gz"
        out.write_bytes(snap_resp.content)
        return out
    except Exception as exc:
        logger.debug("Qdrant snapshot error for %s: %s", collection_name, exc)
        return None


# ---------------------------------------------------------------------------
# BackupManager
# ---------------------------------------------------------------------------

class BackupManager:
    """Create and verify hermes-local memory backups.

    Args:
        memory_dir: Root memory directory (default: ~/.hermes/memory).
        backup_dir: Directory to write archives (default: memory_dir/backups/).
    """

    def __init__(
        self,
        memory_dir: Path | None = None,
        backup_dir: Path | None = None,
    ):
        from hermes_constants import get_hermes_home

        self.memory_dir = memory_dir or (get_hermes_home() / "memory")
        self.backup_dir = backup_dir or (self.memory_dir / "backups")

        # Ensure backup dir exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_backup(self) -> Path | None:
        """Create a timestamped backup archive.

        Returns:
            Path to the created archive (tar.gz), or None if no files to back up.

        Raises:
            OSError: If the backup dir is not writable.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
        archive_name = f"hermes-memory-backup-{timestamp}.tar.gz"
        archive_path = self.backup_dir / archive_name

        manifest: list[dict] = []
        archived_at = datetime.now(timezone.utc).isoformat()

        # Temp dir for SQLite backups and Qdrant snapshots before archiving
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # SQLite backup (copy .backup API snapshot of memory.sqlite)
            sqlite_backup_path = tmp / "memory.sqlite"
            db_path = self.memory_dir / "index" / "memory.sqlite"
            if db_path.exists():
                _safe_copy_sqlite(db_path, sqlite_backup_path)
                # Add to manifest as if it were under memory_dir
                if sqlite_backup_path.exists():
                    manifest.append({
                        "filename": f"index/memory.sqlite",
                        "size": sqlite_backup_path.stat().st_size,
                        "content_hash": _sha256(sqlite_backup_path),
                        "archived_at": archived_at,
                    })

            # Qdrant snapshots (if available)
            qdrant_backups: list[Path] = []
            for coll_name in _qdrant_collections():
                snap_path = _snapshot_qdrant_collection(coll_name, tmp)
                if snap_path and snap_path.exists():
                    qdrant_backups.append(snap_path)

            # Walk the memory directory and build manifest
            files_to_archive: list[tuple[Path, str]] = []

            for fpath, rel in _iter_files(self.memory_dir):
                # SQLite .backup is handled above via tmp — skip raw index/memory.sqlite
                if rel.startswith("index/memory.sqlite"):
                    continue

                files_to_archive.append((fpath, rel))

                # Compute hash after confirming file is readable
                manifest.append({
                    "filename": rel,
                    "size": fpath.stat().st_size,
                    "content_hash": _sha256(fpath),
                    "archived_at": archived_at,
                })

            # Add Qdrant snapshot entries to manifest
            for snap_path in qdrant_backups:
                manifest.append({
                    "filename": f"qdrant/{snap_path.name}",
                    "size": snap_path.stat().st_size,
                    "content_hash": _sha256(snap_path),
                    "archived_at": archived_at,
                })

            if not manifest:
                logger.info("No memory files to back up.")
                return None

            # Write archive
            with tarfile.open(archive_path, "w:gz", compresslevel=6) as tf:
                # Add SQLite backup
                if sqlite_backup_path.exists():
                    tf.add(sqlite_backup_path, arcname="index/memory.sqlite")

                # Add Qdrant snapshots
                for snap_path in qdrant_backups:
                    tf.add(snap_path, arcname=f"qdrant/{snap_path.name}")

                # Add all memory files
                for fpath, rel in files_to_archive:
                    tf.add(fpath, arcname=rel)

                # Add manifest
                manifest_bytes = json.dumps(
                    {"version": 1, "archived_at": archived_at, "files": manifest},
                    indent=2,
                ).encode("utf-8")
                import io
                info = tarfile.TarInfo(name="manifest.json")
                info.size = len(manifest_bytes)
                info.mtime = time.time()
                tf.addfile(info, io.BytesIO(manifest_bytes))

        logger.info("Memory backup complete: %s (%d files, %s)",
                    archive_path.name, len(manifest),
                    _format_size(archive_path.stat().st_size))
        return archive_path

    def verify_archive(self, archive_path: Path) -> tuple[bool, list[str]]:
        """Verify an archive against its manifest.

        Args:
            archive_path: Path to a .tar.gz backup archive.

        Returns:
            (True, []) if all files match the manifest.
            (False, [error_messages]) if any file fails verification.
        """
        errors: list[str] = []

        if not archive_path.exists():
            return False, [f"Archive not found: {archive_path}"]

        try:
            with tarfile.open(archive_path, "r:gz") as tf:
                # Read manifest first
                try:
                    manifest_bytes = tf.extractfile("manifest.json").read()
                except KeyError:
                    return False, ["Archive is missing manifest.json"]

                manifest = json.loads(manifest_bytes)
                files_by_name = {e["filename"]: e for e in manifest.get("files", [])}

                # Verify each member
                for member in tf.getmembers():
                    if member.name == "manifest.json":
                        continue

                    if member.name not in files_by_name:
                        errors.append(f"Unexpected file in archive: {member.name}")
                        continue

                    entry = files_by_name[member.name]

                    # Size check
                    if member.size != entry["size"]:
                        errors.append(
                            f"Size mismatch for {member.name}: "
                            f"manifest says {entry['size']}, archive has {member.size}"
                        )

                    # Hash check — extract to temp file
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        errors.append(f"Cannot read {member.name} from archive")
                        continue

                    h = hashlib.sha256()
                    for chunk in iter(lambda: extracted.read(65536), b""):
                        h.update(chunk)
                    actual_hash = h.hexdigest()

                    if actual_hash != entry["content_hash"]:
                        errors.append(
                            f"Hash mismatch for {member.name}: "
                            f"expected {entry['content_hash']}, got {actual_hash}"
                        )

                # Check for missing files
                archived_names = {m.name for m in tf.getmembers() if m.name != "manifest.json"}
                manifest_names = set(files_by_name.keys())
                missing = manifest_names - archived_names
                if missing:
                    errors.append(f"Files listed in manifest but missing from archive: {missing}")

        except Exception as exc:
            errors.append(f"Verification error: {exc}")

        return len(errors) == 0, errors

    def list_backups(self) -> list[dict]:
        """List all backup archives in the backup dir with metadata.

        Returns:
            List of dicts with: {path, name, size, created_at}.
        """
        backups = []
        for p in sorted(self.backup_dir.glob("hermes-memory-backup-*.tar.gz")):
            backups.append({
                "path": str(p),
                "name": p.name,
                "size": p.stat().st_size,
                "created_at": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
            })
        return backups


# ---------------------------------------------------------------------------
# CLI command (wired in hermes_cli/main.py via memory_setup.py pattern)
# ---------------------------------------------------------------------------

def cmd_memory_backup(args) -> int:
    """CLI entry point: hermes memory backup."""
    from hermes_constants import get_hermes_home

    memory_dir = Path(args.memory_dir) if args.memory_dir else (get_hermes_home() / "memory")
    backup_dir = Path(args.backup_dir) if args.backup_dir else (memory_dir / "backups")

    if args.list:
        bm = BackupManager(memory_dir=memory_dir, backup_dir=backup_dir)
        backups = bm.list_backups()
        if not backups:
            print("  No backups found.")
            return 0
        print(f"  Memory backups in {backup_dir}:")
        for b in backups:
            print(f"    {b['name']}")
            print(f"      size:     {_format_size(b['size'])}")
            print(f"      created:  {b['created_at']}")
        return 0

    if args.verify:
        archive_path = Path(args.verify)
        bm = BackupManager(memory_dir=memory_dir, backup_dir=backup_dir)
        ok, errors = bm.verify_archive(archive_path)
        if ok:
            print(f"  VERIFIED: {archive_path.name}")
            return 0
        else:
            print(f"  VERIFICATION FAILED for {archive_path.name}:")
            for err in errors:
                print(f"    - {err}")
            return 1

    # Default: create backup
    bm = BackupManager(memory_dir=memory_dir, backup_dir=backup_dir)
    path = bm.run_backup()
    if path is None:
        print("  Nothing to back up.")
        return 1
    print(f"  Backup created: {path.name}")
    print(f"  Location: {path}")
    print(f"  Size: {_format_size(path.stat().st_size)}")
    return 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _format_size(nbytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"