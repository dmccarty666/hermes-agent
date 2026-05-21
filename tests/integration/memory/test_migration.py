"""
Integration tests for the holographic→hermes-local migration script.
Uses the REAL holographic DB at ~/.hermes/memory_store.db (read-only)
and the REAL hermes-local DB at ~/.hermes/memory/index/memory.sqlite.

AC coverage:
  AC1: all facts migrated with correct content_hash
  AC2: idempotent via content_hash dedup (0 new rows on re-run)
  AC3: schema introspection ignores extra/missing columns
  AC4: migration report written to memory/exports/
  AC5: holographic DB never modified (read-only verified)
  DoD: --dry-run flag, entity mapping, HRR note

Run with: scripts/run_tests.sh tests/integration/memory/test_migration.py
"""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Paths
HOLO_DB = Path("/home/dmccarty/.hermes/memory_store.db")
HL_DB   = Path("/home/dmccarty/.hermes/memory/index/memory.sqlite")
EXPORTS = Path("/home/dmccarty/.hermes/memory/exports")
SCRIPT  = Path("/home/dmccarty/.hermes/hermes-agent/scripts/migrate_from_holographic.py")

# ------------------------------------------------------------------ fixtures ---

@pytest.fixture
def holo_db():
    """Verify real holographic DB is present and readable."""
    assert HOLO_DB.exists(), f"Holographic DB not found at {HOLO_DB}"
    conn = sqlite3.connect(f"file:{HOLO_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def hl_db():
    """Fresh hermes-local DB for migration testing (tmp copy)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        copy = Path(tmpdir) / "memory.sqlite"
        shutil.copy(HL_DB, copy)
        conn = sqlite3.connect(str(copy))
        conn.row_factory = sqlite3.Row
        yield conn, copy
        conn.close()


@pytest.fixture
def holo_copy():
    """Temp copy of holographic DB to verify read-only access."""
    with tempfile.TemporaryDirectory() as tmpdir:
        copy = Path(tmpdir) / "memory_store_copy.db"
        shutil.copy(HOLO_DB, copy)
        yield copy


# ------------------------------------------------------------------ helpers ---

def run_migration(holo_path, hl_path, *, dry_run=False, extra_args=None):
    """Run the migration script and return (stdout+stderr, returncode)."""
    cmd = ["python", str(SCRIPT), "--holo-db", str(holo_path), "--hl-db", str(hl_path)]
    if dry_run:
        cmd.append("--dry-run")
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT.parent.parent)
    return result.stdout + result.stderr, result.returncode


def count_hl_facts(conn):
    return conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]


def get_all_hl_facts(conn):
    return conn.execute(
        "SELECT fact_id, fact_text, content_hash, category, source_refs_json, tags_json FROM facts"
    ).fetchall()


def get_hl_entities(conn):
    return conn.execute("SELECT entity_id, name, entity_type FROM entities").fetchall()


def get_hl_fact_entities(conn):
    return conn.execute("SELECT fact_id, entity_id FROM fact_entities").fetchall()


# ------------------------------------------------------------------ AC1: Fact Migration ---

class TestAC1_FactMigration:
    """AC1: Given a real holographic DB, all facts appear in hermes-local with correct content_hash."""

    def test_migrate_all_facts_from_real_db(self, holo_db, hl_db):
        """Migration must transfer all 69 facts from holographic."""
        conn_hl, hl_path = hl_db
        holo_count = holo_db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

        output, rc = run_migration(HOLO_DB, hl_path)
        assert rc == 0, f"Migration failed:\n{output}"

        hl_count = count_hl_facts(conn_hl)
        migrated = conn_hl.execute(
            "SELECT COUNT(*) FROM facts WHERE source_refs_json LIKE '%migration:holographic%'"
        ).fetchone()[0]

        assert migrated == holo_count, (
            f"Expected {holo_count} migrated facts, got {migrated}"
        )

    def test_content_hash_matches_sha256(self, holo_db, hl_db):
        """Each migrated fact's content_hash must equal SHA256(fact_text)."""
        conn_hl, hl_path = hl_db

        output, rc = run_migration(HOLO_DB, hl_path)
        assert rc == 0, f"Migration failed:\n{output}"

        facts = get_all_hl_facts(conn_hl)
        for row in facts:
            if "migration:holographic" not in (row["source_refs_json"] or ""):
                continue
            expected = hashlib.sha256(row["fact_text"].encode()).hexdigest()
            assert row["content_hash"] == expected, (
                f"content_hash mismatch for fact {row['fact_id']}: "
                f"expected {expected}, got {row['content_hash']}"
            )

    def test_source_ref_format(self, holo_db, hl_db):
        """Migrated facts must have source_ref = migration:holographic#fact_id=<old_id>."""
        conn_hl, hl_path = hl_db

        output, rc = run_migration(HOLO_DB, hl_path)
        assert rc == 0, f"Migration failed:\n{output}"

        facts = get_all_hl_facts(conn_hl)
        migrated = [f for f in facts if "migration:holographic" in (f["source_refs_json"] or "")]
        assert len(migrated) > 0, "No migrated facts found with migration source_ref"

        for row in migrated:
            refs = json.loads(row["source_refs_json"] or "[]")
            assert any("migration:holographic#fact_id=" in r for r in refs), (
                f"Missing migration source_ref on fact {row['fact_id']}: {refs}"
            )

    def test_fact_entity_links_preserved(self, holo_db, hl_db):
        """Fact-entity links must be preserved after migration."""
        conn_hl, hl_path = hl_db

        output, rc = run_migration(HOLO_DB, hl_path)
        assert rc == 0, f"Migration failed:\n{output}"

        links = get_hl_fact_entities(conn_hl)
        holo_links = holo_db.execute("SELECT COUNT(*) FROM fact_entities").fetchone()[0]
        assert len(links) == holo_links, (
            f"Expected {holo_links} fact_entity links, got {len(links)}"
        )


# ------------------------------------------------------------------ AC2: Idempotency ---

class TestAC2_Idempotency:
    """AC2: Re-running migration produces 0 new rows (idempotent via content_hash dedup)."""

    def test_re_run_produces_zero_new_rows(self, holo_db, hl_db):
        """Second run must not create new fact rows."""
        conn_hl, hl_path = hl_db

        # First run
        output1, rc1 = run_migration(HOLO_DB, hl_path)
        assert rc1 == 0, f"First migration failed:\n{output1}"
        count_after_first = count_hl_facts(conn_hl)

        # Second run
        output2, rc2 = run_migration(HOLO_DB, hl_path)
        assert rc2 == 0, f"Second migration failed:\n{output2}"
        count_after_second = count_hl_facts(conn_hl)

        assert count_after_second == count_after_first, (
            f"Second run added rows: {count_after_first} -> {count_after_second}"
        )

    def test_idempotency_via_content_hash(self, holo_db, hl_db):
        """Skipped (duplicate) count must equal total holographic facts on re-run."""
        conn_hl, hl_path = hl_db

        # Run twice
        run_migration(HOLO_DB, hl_path)
        output, _ = run_migration(HOLO_DB, hl_path)

        holo_count = holo_db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        assert str(holo_count) in output, (
            f"Expected '{holo_count} skipped (duplicate)' in output: {output}"
        )


# ------------------------------------------------------------------ AC3: Schema Introspection ---

class TestAC3_SchemaIntrospection:
    """AC3: Schema introspection handles extra/unknown columns gracefully."""

    def test_extra_columns_ignored_no_error(self, holo_db, hl_db):
        """Unknown columns in holographic must not raise errors."""
        conn_hl, hl_path = hl_db

        # Verify holographic has extra columns
        hcur = holo_db.execute("PRAGMA table_info(facts)")
        cols = {r["name"] for r in hcur.fetchall()}
        mapped_cols = {"content", "category", "tags", "trust_score",
                       "retrieval_count", "helpful_count", "created_at", "updated_at"}
        extra = cols - mapped_cols
        assert extra, "Test assumption: holographic should have extra columns"

        output, rc = run_migration(HOLO_DB, hl_path)
        assert rc == 0, f"Migration raised on extra columns: {output}\n{extra}"

    def test_pragma_table_info_in_script(self):
        """Script must use PRAGMA table_info for column discovery."""
        content = SCRIPT.read_text()
        assert "PRAGMA table_info" in content, (
            "Script must use PRAGMA table_info for schema introspection per AC3"
        )


# ------------------------------------------------------------------ AC4: Report Written ---

class TestAC4_ReportWritten:
    """AC4: Migration report written to memory/exports/migration-holographic-{timestamp}.md."""

    def test_report_written_to_exports(self, holo_db, hl_db):
        """Report file must be created in memory/exports/ with correct naming."""
        conn_hl, hl_path = hl_db

        output, rc = run_migration(HOLO_DB, hl_path)
        assert rc == 0, f"Migration failed:\n{output}"

        # Find most recent report (naming pattern: migration-holographic-YYYYMMDDTHHMMSSZ.md)
        reports = sorted(EXPORTS.glob("migration-holographic-*.md"))
        assert len(reports) > 0, f"No migration report found in {EXPORTS}"

        report = reports[-1]  # most recent
        assert report.name.startswith("migration-holographic-"), f"Wrong naming: {report.name}"
        content = report.read_text()

        assert "fact" in content.lower(), f"Report missing fact count: {content[:200]}"
        assert "entity" in content.lower(), f"Report missing entity count: {content[:200]}"

    def test_report_contains_numeric_counts(self, holo_db, hl_db):
        """Report must contain numeric migrated/skipped counts."""
        conn_hl, hl_path = hl_db

        output, rc = run_migration(HOLO_DB, hl_path)
        assert rc == 0, f"Migration failed:\n{output}"

        reports = sorted(EXPORTS.glob("migration-holographic-*.md"))
        assert reports
        content = reports[-1].read_text()

        assert re.search(r"\d+", content), "Report contains no numbers"


# ------------------------------------------------------------------ AC5: Read-Only ---

class TestAC5_ReadOnly:
    """AC5: Holographic DB must never be modified (read-only access verified)."""

    def test_holo_db_not_modified_after_migration(self, holo_copy, hl_db):
        """After migration, holographic copy must be byte-for-byte identical."""
        import filecmp

        conn_hl, hl_path = hl_db

        output, rc = run_migration(holo_copy, hl_path)
        assert rc == 0, f"Migration failed:\n{output}"

        # Compare file hashes
        def file_hash(p: Path) -> str:
            with open(p, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        assert file_hash(holo_copy) == file_hash(HOLO_DB), (
            "Holographic DB was modified by migration!"
        )

    def test_uri_read_only_mode_used(self):
        """Script must use uri=...?mode=ro to open holographic DB."""
        content = SCRIPT.read_text()
        assert "mode=ro" in content or "mode=RO" in content, (
            "Script must open holographic DB in read-only mode (uri with ?mode=ro)"
        )


# ------------------------------------------------------------------ DoD: --dry-run ---

class TestDoD_DryRun:
    """DoD: --dry-run flag shows what would migrate without writing."""

    def test_dry_run_does_not_write_facts(self, holo_db, hl_db):
        """--dry-run must not create any new fact rows."""
        conn_hl, hl_path = hl_db
        count_before = count_hl_facts(conn_hl)

        output, rc = run_migration(HOLO_DB, hl_path, dry_run=True)
        assert rc == 0, f"Dry-run failed:\n{output}"

        count_after = count_hl_facts(conn_hl)
        assert count_after == count_before, (
            f"--dry-run wrote rows: {count_before} -> {count_after}"
        )

    def test_dry_run_shows_counts(self, holo_db, hl_db):
        """Dry run output must show fact/entity counts."""
        conn_hl, hl_path = hl_db

        output, rc = run_migration(HOLO_DB, hl_path, dry_run=True)
        assert rc == 0, f"Dry-run failed:\n{output}"

        assert "fact" in output.lower(), f"Dry run output missing fact info: {output}"

    def test_dry_run_writes_report(self, holo_db, hl_db):
        """--dry-run must still write the report."""
        conn_hl, hl_path = hl_db

        output, rc = run_migration(HOLO_DB, hl_path, dry_run=True)
        assert rc == 0, f"Dry-run failed:\n{output}"

        reports = sorted(EXPORTS.glob("migration-holographic-*.md"))
        assert len(reports) > 0, "--dry-run must write a report"


# ------------------------------------------------------------------ DoD: Entity Mapping ---

class TestDoD_EntityMapping:
    """DoD: Entities migrated with name/type/aliases."""

    def test_entity_rows_created(self, holo_db, hl_db):
        """Entity rows must be created in hermes-local during migration."""
        conn_hl, hl_path = hl_db

        output, rc = run_migration(HOLO_DB, hl_path)
        assert rc == 0, f"Migration failed:\n{output}"

        entity_count = conn_hl.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        holo_entity_count = holo_db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        assert entity_count >= holo_entity_count, (
            f"Expected >= {holo_entity_count} entities, got {entity_count}"
        )

    def test_entity_names_preserved(self, holo_db, hl_db):
        """Migrated entities must have correct names from holographic."""
        conn_hl, hl_path = hl_db

        output, rc = run_migration(HOLO_DB, hl_path)
        assert rc == 0, f"Migration failed:\n{output}"

        hl_entities = {e["name"] for e in get_hl_entities(conn_hl)}
        holo_entities = {r["name"] for r in holo_db.execute("SELECT name FROM entities").fetchall()}

        assert holo_entities.issubset(hl_entities), (
            f"Some holographic entities missing in hermes-local: {holo_entities - hl_entities}"
        )


# ------------------------------------------------------------------ DoD: HRR Note ---

class TestDoD_HRRNote:
    """DoD: Report mentions HRR banks will be recomputed post-migration."""

    def test_hrr_note_in_report(self, holo_db, hl_db):
        """Report must mention HRR banks or vectors will be recomputed."""
        conn_hl, hl_path = hl_db

        output, rc = run_migration(HOLO_DB, hl_path)
        assert rc == 0, f"Migration failed:\n{output}"

        reports = sorted(EXPORTS.glob("migration-holographic-*.md"))
        assert reports
        content = reports[-1].read_text().lower()

        hrr_keywords = ["hrr", "vector", "bank", "rebuild"]
        assert any(k in content for k in hrr_keywords), (
            f"Report missing HRR/bank mention: {content[:300]}"
        )