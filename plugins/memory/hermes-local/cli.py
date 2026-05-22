# Copyright 2026 David McCarty. All rights reserved.
"""CLI commands for the hermes-local memory plugin.

Provides ``hermes memory`` subcommands:
  health        — Show plugin health and configuration
  db init       — Initialize the SQLite schema
  capture-test  — Inject a synthetic turn end-to-end
  ls-sessions   — List captured sessions

All commands are invoked by the CLI dispatcher in hermes_cli/commands.py
via the registered ``register_cli()`` entry point.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# -----------------------------------------------------------------------
# Public entry point — called by the CLI dispatcher
# -----------------------------------------------------------------------


def register_cli() -> None:
    """Register this plugin's CLI commands with the global command registry.

    Called once at startup by hermes_cli/commands.py after discovering
    the plugin via discover_plugin_cli_commands().
    """
    # Registration is passive — we export run_memory_subcommand which is
    # called by the memory subcommand dispatcher in commands.py.
    pass


# -----------------------------------------------------------------------
# Subcommand implementations
# -----------------------------------------------------------------------


def run_memory_subcommand(subcommand: str, rest: str = "") -> str:
    """Dispatch to the appropriate subcommand handler.

    Args:
        subcommand: One of: health, db, capture-test, ls-sessions
        rest: Everything after the subcommand token

    Returns:
        Formatted output string to display to the user.
    """
    if subcommand == "health":
        return _health()
    if subcommand == "db":
        return _db(rest)
    if subcommand == "qdrant-init":
        return _qdrant_init()
    if subcommand == "capture-test":
        return _capture_test()
    if subcommand == "ls-sessions":
        return _ls_sessions(rest)
    return _help()


# -----------------------------------------------------------------------
# Subcommand: health
# -----------------------------------------------------------------------


def _health() -> str:
    """Show plugin health, configuration, and connectivity."""
    lines = ["=== Hermes Local Memory — Health ===", ""]

    # Config
    try:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        config_path = Path(home) / "config.yaml"
        lines.append(f"hermes_home : {home}")
        lines.append(f"config       : {config_path}")
        if config_path.exists():
            import yaml

            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            provider = cfg.get("memory", {}).get("provider", "not set")
            lines.append(f"provider     : {provider}")
        else:
            lines.append("config       : (file not found)")
    except Exception as e:
        lines.append(f"config error : {e}")

    # Redaction
    try:
        from hermes_memory_core.write.redaction import redact

        # Realistic AWS key format: AKIA prefix + exactly 16 uppercase alphanumeric
        result = redact("test AKIAIOSFODNN7EXAMPLE test")
        if result.types_found:
            lines.append(
                f"redaction    : available ({len(result.types_found)} types: "
                f"{', '.join(sorted(result.types_found))})"
            )
        else:
            lines.append("redaction    : scan not working — no hits on test content")
    except Exception as e:
        lines.append(f"redaction    : unavailable — {e}")

    # JSONL root
    try:
        from hermes_constants import get_hermes_home

        raw_dir = Path(str(get_hermes_home())) / "memory" / "raw"
        if raw_dir.exists():
            year_dirs = [d.name for d in raw_dir.iterdir() if d.is_dir()]
            lines.append(f"jsonl root   : {raw_dir} ({len(year_dirs)} year dirs: {', '.join(sorted(year_dirs)) or 'none'})")
        else:
            lines.append("jsonl root   : not initialized (run `hermes memory init`)")
    except Exception as e:
        lines.append(f"jsonl root   : error — {e}")

    # SQLite
    try:
        from hermes_constants import get_hermes_home
        from hermes_memory_core.store.sqlite import MemoryDB

        home = get_hermes_home()
        db_path = Path(home) / "memory" / "index" / "memory.sqlite"
        if db_path.exists():
            size_kb = db_path.stat().st_size // 1024
            db = MemoryDB(str(db_path))
            if db.is_initialized():
                lines.append(f"sqlite       : {db_path} ({size_kb} KB, schema OK)")
            else:
                lines.append(f"sqlite       : {db_path} ({size_kb} KB, NOT initialized)")
        else:
            lines.append("sqlite       : not initialized (run `hermes memory db init`)")
    except Exception as e:
        lines.append(f"sqlite       : error — {e}")

    # Qdrant
    try:
        from hermes_memory_core.store.qdrant import QdrantStore

        qdrant = QdrantStore()
        if qdrant.is_available():
            lines.append("qdrant       : connected")
        else:
            lines.append("qdrant       : unreachable (Phase 3 feature)")
    except Exception as e:
        lines.append(f"qdrant       : error — {e}")

    # LMS embeddings
    try:
        from hermes_memory_core.embed import LMSClient, EMBED_MODEL, EMBED_DIM

        client = LMSClient()
        # Try a quick health-check by attempting embed (will raise if unavailable)
        try:
            client.embed("health check")
            lines.append(f"lms          : connected ({EMBED_MODEL}, dim={EMBED_DIM})")
        except NotImplementedError:
            lines.append(f"lms          : stub (Phase 3 — embed not yet implemented)")
        except Exception:
            lines.append(f"lms          : unreachable")
    except Exception as e:
        lines.append(f"lms          : error — {e}")

    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# Subcommand: db
# -----------------------------------------------------------------------


def _db(rest: str) -> str:
    """Handle `hermes memory db` subcommands."""
    tokens = rest.strip().split() if rest and rest.strip() else []
    sub = tokens[0] if tokens else ""

    if sub == "init":
        return _db_init()
    return "Usage: hermes memory db init\n\nInitialize the SQLite schema at ~/.hermes/memory/index/memory.sqlite."


def _db_init() -> str:
    """Initialize (or verify) the SQLite schema."""
    try:
        from hermes_constants import get_hermes_home
        from hermes_memory_core.store.sqlite import MemoryDB

        home = get_hermes_home()
        db_path = Path(home) / "memory" / "index" / "memory.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        db = MemoryDB(str(db_path))
        db.initialize()  # idempotent

        return f"OK SQLite schema initialized at {db_path}\n  Run `hermes memory health` to verify."
    except Exception as e:
        import traceback

        return f"DB init failed: {e}\n{traceback.format_exc()}"


def _qdrant_init() -> str:
    """Initialize (or verify) the Qdrant collections."""
    try:
        from hermes_memory_core.store.qdrant import init_collections, QdrantInitError

        result = init_collections()
        if result["status"] == "already_initialized":
            return "OK Qdrant collections already initialized (no-op).\n  Run `hermes memory health` to verify."
        if result["status"] == "created":
            collections = ", ".join(result["collections"])
            return f"OK Qdrant collections created: {collections}\n  Run `hermes memory health` to verify."
        if result["errors"]:
            return f"Qdrant init had errors: {'; '.join(result['errors'])}"
        return f"Qdrant init: {result['status']}"
    except QdrantInitError as e:
        return f"Qdrant init failed: {e}"
    except Exception as e:
        import traceback
        return f"Qdrant init error: {e}\n{traceback.format_exc()}"


# -----------------------------------------------------------------------
# Subcommand: capture-test
# -----------------------------------------------------------------------


def _capture_test() -> str:
    """Inject a synthetic turn end-to-end through the capture pipeline."""
    try:
        from hermes_constants import get_hermes_home
        from hermes_memory_core.store.fs import FSStore

        home = get_hermes_home()
        base = Path(home) / "memory"

        # Ensure dirs exist via hermes_cli.memory (already handles idempotency)
        from hermes_cli.memory import memory_init

        memory_init()

        # Ensure DB schema exists
        from hermes_memory_core.store.sqlite import MemoryDB

        db = MemoryDB(str(base / "index" / "memory.sqlite"))
        db.initialize()

        # Build a synthetic session + turn
        session_id = f"cli-test-{uuid.uuid4().hex[:8]}"
        turn_id = f"turn-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        synthetic_event = {
            "event_id": f"evt-{uuid.uuid4().hex[:8]}",
            "session_id": session_id,
            "turn_id": turn_id,
            "sequence": 1,
            "timestamp": now,
            "role": "user",
            "content": "This is a test turn with no real secrets. Just verifying the capture pipeline works.",
            "agent": "cli-test",
            "project": "hermes-memory",
            "source": "memory-cli",
            "tags": ["test"],
            "tool_calls": [],
            "attachments": [],
            "metadata": {},
            "hash": "",
            "parent_turn_id": None,
            "embedding_status": "pending",
            "index_status": "pending",
            "dream_status": "pending",
        }

        # Write JSONL
        fs = FSStore(base)
        result = fs.append_event(synthetic_event)
        if result is False:
            jsonl_path = "(dedup hit — no write)"
        else:
            date_str = now[:10]
            year = date_str[:4]
            jsonl_path = str(base / "raw" / year / date_str / f"{session_id}.jsonl")

        # Insert into SQLite using the same connection pattern as MemoryDB._connect
        db_path = base / "index" / "memory.sqlite"
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        try:
            conn.execute(
                """INSERT OR IGNORE INTO sessions (session_id, agent, started_at, project)
                   VALUES (?, ?, ?, ?)""",
                (session_id, "cli-test", now, "hermes-memory"),
            )
            conn.execute(
                """INSERT OR IGNORE INTO turns
                   (turn_id, session_id, sequence, timestamp, role, content,
                    dream_status, index_status, source_refs_json)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', 'pending', '[]')""",
                (turn_id, session_id, 1, now, "user", synthetic_event["content"]),
            )
            conn.commit()
        finally:
            conn.close()

        return (
            f"OK Capture test injected.\n"
            f"  session_id : {session_id}\n"
            f"  turn_id    : {turn_id}\n"
            f"  jsonl      : {jsonl_path}\n"
            f"  sqlite     : sessions + turns rows inserted\n"
            f"\nRun `hermes memory ls-sessions` to confirm."
        )
    except Exception as e:
        import traceback

        return f"Capture test failed: {e}\n{traceback.format_exc()}"


# -----------------------------------------------------------------------
# Subcommand: ls-sessions
# -----------------------------------------------------------------------


def _ls_sessions(rest: str) -> str:
    """List captured sessions from SQLite."""
    try:
        from hermes_constants import get_hermes_home
        from hermes_memory_core.store.sqlite import MemoryDB

        home = get_hermes_home()
        db_path = Path(home) / "memory" / "index" / "memory.sqlite"
        if not db_path.exists():
            return "SQLite not initialized. Run `hermes memory db init` first."

        db = MemoryDB(str(db_path))
        if not db.is_initialized():
            return "SQLite schema not initialized. Run `hermes memory db init` first."

        db_path_str = str(db_path)
        conn = sqlite3.connect(db_path_str, timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        try:
            cur = conn.execute(
                """SELECT session_id, agent, project, started_at,
                          COUNT(turn_id) as turn_count, ended_at
                   FROM sessions
                   LEFT JOIN turns USING(session_id)
                   GROUP BY session_id
                   ORDER BY started_at DESC
                   LIMIT 50"""
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return "No sessions found. Capture a session first."

        lines = []
        for (sid, agent, project, started, turns, ended) in rows:
            proj = (project or "")[:14]
            ended_s = ended or ""
            lines.append(f"{sid:<36} {agent:<8} {proj:<14} {started:<26} {str(turns):>4} {ended_s}")

        header = f"{'SESSION_ID':<36} {'AGENT':<8} {'PROJECT':<14} {'STARTED_AT':<26} {'N':>4} ENDED"
        return (
            "```\n"
            + header + "\n"
            + "-" * len(header) + "\n"
            + "\n".join(lines)
            + "\n```\n"
        )
    except Exception as e:
        return f"Failed to list sessions: {e}"


# -----------------------------------------------------------------------
# Help fallback
# -----------------------------------------------------------------------


def _help() -> str:
    return """\
Hermes Local Memory — available commands:

  hermes memory health        Show plugin health and configuration
  hermes memory db init       Initialize SQLite schema
  hermes memory qdrant-init   Initialize Qdrant collections (Phase 3)
  hermes memory capture-test  Inject a synthetic turn end-to-end
  hermes memory ls-sessions   List captured sessions (from SQLite)

Run `hermes memory init` first if you haven't initialized the directory tree.
"""


if __name__ == "__main__":
    # Allow running as: python -m plugins.memory.hermes_local.cli
    sub = sys.argv[1] if len(sys.argv) > 1 else "help"
    rest = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
    print(run_memory_subcommand(sub, rest))