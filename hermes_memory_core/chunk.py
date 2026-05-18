"""Chunker for Hermes Local Memory.

Splits conversation turns into smaller, embeddable chunks.
Phase 1 (T-001): stub. Full implementation in story 1.3.x indexing.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """A discrete embeddable chunk of conversation content."""
    chunk_id: str
    session_id: str
    start_turn_id: str
    end_turn_id: str
    text: str
    embed_model: str
    qdrant_point_id: str | None = None


def chunk_turns(turns: List[dict], max_tokens: int = 512) -> List[Chunk]:
    """Split turns into embeddable chunks.

    Phase 1 (T-001): stub — returns empty list.
    Full implementation in Phase 2 (story 2.2.1).
    """
    return []  # stub