# Copyright 2026 David McCarty. All rights reserved.
"""Contradiction detection heuristic for dreamer facts.

Implements TDD §10.3:
  - Bucket by (project, entity, category)
  - Jaccard > 0.4 threshold within bucket
  - Marks conflicting facts as 'disputed' with supersedes_fact_id link
  - Does NOT auto-resolve — surfaces conflicts for human review
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Threshold above which two facts in the same bucket are flagged as potentially conflicting
JACCARD_THRESHOLD = 0.4


@dataclass
class Conflict:
    """A detected conflict between two facts."""
    candidate_fact_id: str
    existing_fact_id: str
    candidate_text: str
    existing_text: str
    jaccard_score: float
    bucket_key: Tuple[str, str, str]


def _tokenize(text: str) -> set[str]:
    """Lowercase and split text into word tokens, stripping punctuation.

    Args:
        text: Raw text string.

    Returns:
        Set of lowercase word tokens.
    """
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets.

    Args:
        tokens_a: First token set.
        tokens_b: Second token set.

    Returns:
        Jaccard similarity in [0.0, 1.0]. Empty sets return 0.0.
    """
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def _extract_primary_entity(text: str) -> str:
    """Extract the primary entity from a fact text.

    Heuristic: first capitalized word that is not at the start of a sentence
    and is a noun-like word (length >= 2). Falls back to first capitalized word.

    Args:
        text: Fact text string.

    Returns:
        Extracted entity name or "" if none found.
    """
    # Common non-entity words to skip
    _STOP_WORDS = frozenset({
        "the", "a", "an", "this", "that", "these", "those",
        "he", "she", "it", "i", "you", "we", "they",
        "my", "your", "our", "their",
        "but", "and", "or", "not", "if", "when", "then", "so",
        "because", "although", "while",
    })

    # Find all capitalized words (allowing punctuation or start-of-string before them)
    capitalized = re.findall(r"(?:^|(?<=[^a-zA-Z]))[A-Z][a-zA-Z]{1,30}", text)

    if capitalized:
        # Filter out common non-entity words
        filtered = [w for w in capitalized if w.lower() not in _STOP_WORDS]
        if filtered:
            return filtered[0]

    # No result from capitalized words (either none found, or all were stop words).
    # Scan all words for the first capitalized word that is not a stop word.
    all_words = re.findall(r"[A-Za-z]+", text)
    for word in all_words:
        if word[0].isupper() and word.lower() not in _STOP_WORDS:
            return word
    return ""


def _entity_from_fact(fact: Dict[str, Any]) -> str:
    """Get entity from a fact dict, using extracted entity or fallback.

    Args:
        fact: Fact dict with at least 'fact_text' or 'text' key.

    Returns:
        Entity string, or "" if not available.
    """
    # Check for explicit entity field first
    if fact.get("entity"):
        return str(fact["entity"])
    # Extract from text
    text = fact.get("fact_text", fact.get("text", ""))
    return _extract_primary_entity(text)


def _make_bucket_key(fact: Dict[str, Any]) -> Tuple[str, str, str]:
    """Build a contradiction-bucket key from a fact dict.

    Bucket key = (project, entity, category)
    Category is derived from scope: user_pref | project | general | tool | session

    Args:
        fact: Fact dict.

    Returns:
        Tuple of (project, entity, category).
    """
    project = str(fact.get("project", "") or "")
    entity = _entity_from_fact(fact)
    scope = str(fact.get("scope", "general") or "general")
    # Normalize scope to category names
    category = scope  # scope values are already category names
    return (project, entity, category)


def find_conflicts(
    candidate: Dict[str, Any],
    existing_facts: List[Dict[str, Any]],
) -> List[Conflict]:
    """Find facts that conflict with the candidate within the same bucket.

    Compares the candidate against all existing facts in the same bucket
    (project + entity + category). Two facts conflict if their token Jaccard
    similarity exceeds 0.4 (meaning they share significant vocabulary).

    Does NOT mark facts as disputed — only returns Conflict objects for
    downstream processing (dreamer worker marks facts as disputed).

    Args:
        candidate: The new fact dict to check (requires: fact_id, fact_text or text,
                   project, scope).
        existing_facts: List of existing fact dicts to compare against.

    Returns:
        List of Conflict objects for each bucket match exceeding Jaccard threshold.
    """
    candidate_id = candidate.get("fact_id", candidate.get("id", ""))
    candidate_text = candidate.get("fact_text", candidate.get("text", ""))
    if not candidate_text:
        return []

    candidate_tokens = _tokenize(candidate_text)
    candidate_bucket = _make_bucket_key(candidate)

    conflicts: List[Conflict] = []

    for existing in existing_facts:
        existing_id = existing.get("fact_id", existing.get("id", ""))
        if not existing_id or existing_id == candidate_id:
            continue

        existing_text = existing.get("fact_text", existing.get("text", ""))
        if not existing_text:
            continue

        # Check if same bucket
        existing_bucket = _make_bucket_key(existing)
        if existing_bucket != candidate_bucket:
            continue

        # Compute Jaccard
        existing_tokens = _tokenize(existing_text)
        score = _jaccard(candidate_tokens, existing_tokens)

        if score > JACCARD_THRESHOLD:
            conflicts.append(Conflict(
                candidate_fact_id=candidate_id,
                existing_fact_id=existing_id,
                candidate_text=candidate_text,
                existing_text=existing_text,
                jaccard_score=score,
                bucket_key=candidate_bucket,
            ))

    return conflicts


def mark_disputed(conn, candidate_fact_id: str, existing_fact_id: str) -> None:
    """Mark a candidate fact as disputed in SQLite, linking to the superseded fact.

    Updates the candidate fact's status to 'disputed' and sets supersedes_fact_id
    to the existing fact's ID.

    Args:
        conn: SQLite connection (write transaction managed by caller).
        candidate_fact_id: ID of the new (conflicting) fact.
        existing_fact_id: ID of the existing fact being contradicted.
    """
    now = _utc_now()
    conn.execute(
        """UPDATE facts
           SET status = 'disputed',
               updated_at = ?
           WHERE fact_id = ? AND status = 'active'""",
        (now, candidate_fact_id),
    )
    # Also update the supersedes link in a dedicated column if it exists
    # (schema v2 may have supersedes_fact_id column)
    try:
        conn.execute(
            """UPDATE facts
               SET supersedes_fact_id = ?
               WHERE fact_id = ?""",
            (existing_fact_id, candidate_fact_id),
        )
    except Exception:
        # Column may not exist yet — ignore
        pass


def _utc_now() -> str:
    """Return current UTC ISO timestamp."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()