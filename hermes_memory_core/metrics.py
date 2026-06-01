"""Metrics writer for hermes-local memory.

Updates ~/.hermes/memory/metrics.json after each capture, index, and dream batch.

Metrics written:
  - captured_turns_24h: turns captured in the last 24 hours
  - chunks_indexed_24h: chunks indexed in the last 24 hours
  - chunks_pending: chunks with index_status='pending'
  - facts_total: total facts in DB
  - facts_active: facts with status='active'
  - qdrant_points: total points across hermes_memory_* collections
  - last_dream_run_at: ISO timestamp of most recent dream run
  - last_dream_status: status of most recent dream run (completed/error/unknown)
  - redactions_24h: redaction audit events in the last 24 hours
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MEMORY_DIR = Path.home() / ".hermes" / "memory"
DEFAULT_DB_PATH = DEFAULT_MEMORY_DIR / "index" / "memory.sqlite"


class MetricsWriter:
    """Reads current memory system state and writes metrics.json."""

    def __init__(
        self,
        memory_dir: Path | str | None = None,
        db_path: Path | str | None = None,
    ) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.metrics_path = self.memory_dir / "metrics.json"

    def update(self) -> dict[str, Any]:
        """Compute and write current metrics to metrics.json.

        Returns the metrics dict that was written.
        """
        metrics = self._compute()
        self._write(metrics)
        return metrics

    def _compute(self) -> dict[str, Any]:
        """Compute all metric values from live data sources."""
        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(hours=24)

        captured_turns_24h = 0
        chunks_indexed_24h = 0
        chunks_pending = 0
        facts_total = 0
        facts_active = 0
        last_dream_run_at: str | None = None
        last_dream_status = "unknown"
        redactions_24h = 0

        if self.db_path.exists():
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=5)
                conn.row_factory = sqlite3.Row
                try:
                    # captured_turns_24h
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM turns WHERE timestamp >= ?",
                        (day_ago.isoformat(),),
                    )
                    captured_turns_24h = cur.fetchone()[0]

                    # chunks_indexed_24h (updated_at in last 24h)
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM chunks WHERE updated_at >= ?",
                        (day_ago.isoformat(),),
                    )
                    chunks_indexed_24h = cur.fetchone()[0]

                    # chunks_pending — chunks not yet indexed (no Qdrant point)
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM chunks WHERE qdrant_point_id IS NULL"
                    )
                    chunks_pending = cur.fetchone()[0]

                    # facts_total
                    cur = conn.execute("SELECT COUNT(*) FROM facts")
                    facts_total = cur.fetchone()[0]

                    # facts_active
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM facts WHERE status = 'active'"
                    )
                    facts_active = cur.fetchone()[0]

                    # last_dream_run_at and status
                    cur = conn.execute(
                        "SELECT started_at, status FROM dream_runs "
                        "ORDER BY started_at DESC LIMIT 1"
                    )
                    row = cur.fetchone()
                    if row:
                        last_dream_run_at = row["started_at"]
                        last_dream_status = row["status"] or "unknown"
                    else:
                        last_dream_status = "unknown"

                    # redactions_24h (audit_log action='redact' events)
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM audit_log "
                        "WHERE action = 'redact' AND timestamp >= ?",
                        (day_ago.isoformat(),),
                    )
                    redactions_24h = cur.fetchone()[0]

                finally:
                    conn.close()
            except Exception as e:
                logger.warning("Failed to compute SQLite metrics: %s", e)

        # qdrant_points — sum across all hermes_memory_* collections
        qdrant_points = self._count_qdrant_points()

        return {
            "captured_turns_24h": captured_turns_24h,
            "chunks_indexed_24h": chunks_indexed_24h,
            "chunks_pending": chunks_pending,
            "facts_total": facts_total,
            "facts_active": facts_active,
            "qdrant_points": qdrant_points,
            "last_dream_run_at": last_dream_run_at,
            "last_dream_status": last_dream_status,
            "redactions_24h": redactions_24h,
        }

    def _count_qdrant_points(self) -> int:
        """Count total points across all hermes_memory_* collections."""
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient("http://localhost:6333", timeout=5)
            collections = client.get_collections()
            total = 0
            for coll in collections.collections or []:
                if coll.name and coll.name.startswith("hermes_memory_"):
                    try:
                        info = client.get_collection(coll.name)
                        total += info.points_count if info.points_count else 0
                    except Exception:
                        pass
            return total
        except Exception as e:
            logger.warning("Failed to count Qdrant points: %s", e)
            return 0

    def _write(self, metrics: dict[str, Any]) -> None:
        """Write metrics dict to metrics.json, creating dir if needed."""
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.metrics_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(metrics, indent=2) + "\n")
        tmp.replace(self.metrics_path)