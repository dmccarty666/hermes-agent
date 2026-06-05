#!/usr/bin/env python3
"""
fact_qdrant_indexer.py — Embed all SQLite facts into Qdrant hermes_memory_facts_nomic_v15.

Design:
  - Reads all facts from SQLite (hrr_vector IS NULL → unindexed)
  - Batches embeddings via LMStudio embed_batch()
  - Writes vectors back to SQLite hrr_vector BLOB (avoids re-embedding on re-run)
  - Upserts to Qdrant with fact_id as point ID
  - Idempotent: can be re-run safely (only processes NULL hrr_vector rows)

Usage:
  python fact_qdrant_indexer.py [--batch-size 50] [--dry-run] [--all]
"""

import argparse
import struct
import logging
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# ── Add hermes-agent to path ────────────────────────────────────────────────
import sqlite3
import sys
import hashlib
from pathlib import Path

def _fact_id_to_num(fact_id: str) -> int:
    """Convert a string fact_id to a positive 63-bit integer for Qdrant."""
    h = hashlib.sha256(fact_id.encode()).hexdigest()
    return int(h[:12], 16) % (2**63)


AGENT_DIR = Path("/home/dmccarty/.hermes/hermes-agent")
sys.path.insert(0, str(AGENT_DIR))

from hermes_memory_core.embed import LMSClient, EmbeddingError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/home/dmccarty/.hermes/logs/fact-indexer.log")],
)
logger = logging.getLogger("fact_indexer")

DB_PATH = "/home/dmccarty/.hermes/memory/index/memory.sqlite"

QDRANT_COLLECTION = "hermes_memory_facts_nomic_v15"
BATCH_SIZE = 50  # LMStudio can handle this comfortably
DIM = 768


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_vector(data: bytes) -> list[float]:
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url="http://localhost:6333")


def index_facts(batch_size: int = BATCH_SIZE, dry_run: bool = False, all_facts: bool = False):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA busy_timeout = 30000")
    qdrant = get_qdrant_client()
    embed = LMSClient()

    # Count total
    if all_facts:
        total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        rows = conn.execute(
            "SELECT fact_id, fact_text, category, project, entity, trust_score, "
            "status, tags_json, source_refs_json, retrieval_count, helpful_count, "
            "decay_rate_days FROM facts ORDER BY created_at"
        ).fetchall()
    else:
        total = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE hrr_vector IS NULL"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT fact_id, fact_text, category, project, entity, trust_score, "
            "status, tags_json, source_refs_json, retrieval_count, helpful_count, "
            "decay_rate_days FROM facts WHERE hrr_vector IS NULL ORDER BY created_at"
        ).fetchall()

    logger.info(f"Starting fact indexer — {total} facts to process")
    if dry_run:
        logger.info("DRY RUN — not writing anything")
        return

    indexed = 0
    errors = 0
    t0 = time.time()

    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        fact_ids = [r[0] for r in batch_rows]
        texts = [r[1] for r in batch_rows]

        # ── Embed ────────────────────────────────────────────────────────────
        try:
            embeddings = embed.embed_batch(texts)
        except EmbeddingError as e:
            logger.error(f"Embedding batch failed at offset {offset}: {e}")
            errors += len(batch_rows)
            continue

        # ── Write to SQLite + Qdrant ──────────────────────────────────────────
        points = []
        for i, row in enumerate(batch_rows):
            fact_id, fact_text, category, project, entity, trust_score, status, tags_json, source_refs_json, retrieval_count, helpful_count, decay_rate_days = row
            vec = embeddings[i]

            # Write vector to SQLite
            packed = _pack_vector(vec)
            conn.execute(
                "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                (packed, fact_id),
            )

            # Build Qdrant point
            points.append(PointStruct(
                                    id=_fact_id_to_num(fact_id),
                                    vector=vec,
                                    payload={
                                        "fact_id":        fact_id,
                                        "fact_text":      fact_text[:2000],  # truncate long text
                                        "category":       category or "",
                                        "project":        project or "",
                                        "entity":         entity or "",
                                        "trust_score":    trust_score or 0.5,
                                        "status":         status or "active",
                                        "tags":           tags_json or "[]",
                                        "source_refs":    source_refs_json or "[]",
                                        "retrieval_count": retrieval_count or 0,
                                        "helpful_count":  helpful_count or 0,
                                        "decay_rate_days": decay_rate_days,
                                    },
                                ))

        # Upsert to Qdrant
        try:
            qdrant.upsert(
                collection_name=QDRANT_COLLECTION,
                points=[{"id": p.id, "vector": p.vector, "payload": p.payload} for p in points],
            )
        except Exception as e:
            logger.error(f"Qdrant upsert failed at offset {offset}: {e}")
            errors += len(points)
            continue

        conn.commit()
        indexed += len(points)
        elapsed = time.time() - t0
        rate = indexed / elapsed if elapsed > 0 else 0
        logger.info(f"  indexed {indexed}/{total} ({rate:.1f} fact/s)")

    logger.info(f"DONE — {indexed}/{total} facts indexed, {errors} errors in {time.time()-t0:.1f}s")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index SQLite facts into Qdrant")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="count only, no writes")
    parser.add_argument("--all", action="store_true", help="re-embed all facts, not just NULL hrr_vector")
    args = parser.parse_args()
    index_facts(batch_size=args.batch_size, dry_run=args.dry_run, all_facts=args.all)