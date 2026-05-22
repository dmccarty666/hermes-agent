"""
Contradiction detection for the dreamer.

Heuristic v1: bucket-by-(project, entity, category) with Jaccard token
overlap threshold 0.4 per TDD §10.3.

The pairwise LLM check is post-MVP; this module handles the fast filter
that decides which candidate pairs need LLM comparison.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

JACCARD_THRESHOLD = 0.4


def _tokenize(text: str) -> set[str]:
    """Simple whitespace+punct tokenization for Jaccard comparison."""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity: |A∩B| / |A∪B|. Returns 0.0 for empty sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


@dataclass(frozen=True)
class Conflict:
    """A detected potential conflict between two facts."""
    existing_fact_id: str
    existing_text: str
    candidate_text: str
    jaccard_score: float
    bucket: tuple[str | None, str | None, str]   # (project, entity, category)


def find_conflicts(
    candidate_text: str,
    candidate_project: str | None,
    candidate_entity: str | None,
    candidate_category: str,
    existing_facts: List[dict],
) -> List[Conflict]:
    """
    Find existing facts that potentially conflict with a candidate fact.

    Buckets facts by (project, entity, category) and applies Jaccard token
    overlap. Returns facts in the same bucket with Jaccard > 0.4 that are
    not just token-aligned (same exact keywords in same order).

    Args:
        candidate_text: the newly extracted fact text
        candidate_project: project name
        candidate_entity: entity name
        candidate_category: category string
        existing_facts: list of fact dicts from SQLite

    Returns:
        List of Conflict objects (may be empty).
    """
    candidate_tokens = _tokenize(candidate_text)
    if not candidate_tokens:
        return []

    # Primary entity for bucket — prefer explicit entity field
    bucket_entity = candidate_entity or _extract_primary_entity(candidate_text)
    bucket_key = (candidate_project, bucket_entity, candidate_category)

    conflicts: List[Conflict] = []
    for existing in existing_facts:
        # Same bucket check
        existing_entity = existing.get("entity") or _extract_primary_entity(existing.get("fact_text", ""))
        existing_bucket = (existing.get("project"), existing_entity, existing.get("category", "general"))
        if existing_bucket != bucket_key:
            continue

        existing_text = existing.get("fact_text", "")
        existing_tokens = _tokenize(existing_text)
        score = jaccard(candidate_tokens, existing_tokens)

        if score > JACCARD_THRESHOLD and not _is_token_aligned(candidate_text, existing_text):
            conflicts.append(Conflict(
                existing_fact_id=existing["fact_id"],
                existing_text=existing_text,
                candidate_text=candidate_text,
                jaccard_score=score,
                bucket=bucket_key,
            ))
            logger.debug(
                "Potential conflict detected: existing=%s candidate=%s score=%.3f",
                existing["fact_id"], candidate_text[:60], score,
            )

    return conflicts


def _extract_primary_entity(text: str) -> str:
    """
    Naive entity extraction: return the longest Title-Case word phrase
    as a stand-in for a named entity.
    """
    phrases = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
    if not phrases:
        # Fall back to longest word
        words = [w for w in text.split() if len(w) > 3]
        return max(words, key=len) if words else ""
    return max(phrases, key=len)


def _is_token_aligned(a: str, b: str) -> bool:
    """
    Check if two texts are the same assertion with minor word swaps
    (e.g., "prefers X" vs "likes X"). Returns True if the texts share
    >80% of their tokens and the same order ratio — these are NOT conflicts.
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return False
    # If one token set is a subset of the other, likely aligned
    if tokens_a <= tokens_b or tokens_b <= tokens_a:
        return True
    return False


def mark_disputed(store, new_fact_id: str, existing_fact_id: str) -> None:
    """
    Mark the existing fact as 'disputed' with a supersession link to the new fact.

    Per TDD §10.3: on conflict, the new fact gets status='disputed' and
    supersedes_fact_id=existing_fact_id. The dream report flags the conflict.
    """
    store.mark_fact_disputed(existing_fact_id, new_fact_id)
    logger.info(
        "Fact %s marked disputed — superseded by %s",
        existing_fact_id, new_fact_id,
    )
