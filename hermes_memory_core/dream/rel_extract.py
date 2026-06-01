"""Relation extraction for the dreamer pipeline.

Extracts typed entity relations from text using heuristic patterns.
Used to build the knowledge graph edges between entities.

This is a lightweight heuristic extractor — the dreamer also uses the LLM
for relation extraction when available (see worker.py _extract_relations_llm).

Relation types:
  - USES / USED_FOR (tech → tech)
  - PART_OF (tech → project/org)
  - WORKS_FOR (person → org)
  - CREATED / DEVELOPED (person → tech/project)
  - RELATED_TO (generic)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from hermes_memory_core.dream.entity import Entity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Relation:
    """A single extracted relation between two entities."""
    source_entity: str
    target_entity: str
    relation_type: str
    source_ref: Optional[str] = None
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Heuristic relation patterns
# ---------------------------------------------------------------------------

# Patterns that indicate a relation between two entities
_RELATION_PATTERNS = [
    # A uses B
    (r'\b(uses?|using|utilize|utilizing)\b\s+(?:the\s+)?(.{3,60}?)\b(?:\s+for|\.|,)', 'USES'),
    # A is built with B
    (r'\b(built\s+with|built\s+on|powered\s+by|backed\s+by)\b\s+(.{3,60}?)\b', 'USES'),
    # A was created by B
    (r'\b(created\s+by|developed\s+by|written\s+by|designed\s+by)\b\s+(.{3,60}?)\b', 'CREATED'),
    # A works for B
    (r'\b(works?\s+for|employed\s+by)\b\s+(.{3,60}?)\b', 'WORKS_FOR'),
    # A is part of B
    (r'\b(part\s+of|component\s+of|module\s+of)\b\s+(.{3,60}?)\b', 'PART_OF'),
    # A related to B
    (r'\b(related\s+to|associated\s+with|connected\s+to)\b\s+(.{3,60}?)\b', 'RELATED_TO'),
    # A depends on B
    (r'\b(depends?\s+on|dependency.*\b)\b\s+(.{3,60}?)\b', 'DEPENDS_ON'),
    # A replaces B
    (r'\b(replaces?|replacing)\b\s+(.{3,60}?)\b', 'REPLACES'),
    # A vs B
    (r'\b(vs\.?|versus|compared\s+to|compared\s+against)\b\s+(.{3,60}?)\b', 'RELATED_TO'),
    # A and B (conjunction — weak signal)
    (r'\b(and|&)\b\s+(.{3,60}?)\b', 'RELATED_TO'),
]


class RelationExtractor:
    """Extract typed relations between entities from text.

    Uses heuristic patterns to identify relations between entities that
    were previously extracted. The extracted relations are fed into the
    knowledge graph.

    For production use, this should be replaced with an actual relation
    extraction model (e.g., spaCy + custom model, or an LLM prompt).

    Args:
        min_confidence: Minimum confidence threshold for relations.
        max_relations: Maximum number of relations to extract per text.
    """

    def __init__(self, min_confidence: float = 0.5, max_relations: int = 20):
        self.min_confidence = min_confidence
        self.max_relations = max_relations

    def __call__(self, text: str, entities: List[Entity], fact_id: str) -> List[Relation]:
        """Extract relations from text given extracted entities.

        Args:
            text: The full text to extract relations from.
            entities: Previously extracted entities.
            fact_id: The fact ID this text is associated with.

        Returns:
            List of Relation objects.
        """
        relations: List[Relation] = []
        seen: set = set()
        fact_ref = f"fact:{fact_id}" if fact_id else None

        for entity in entities:
            if entity.entity_type not in ("TECH", "ORG", "PERSON"):
                continue

            # Find patterns where this entity appears
            entity_lower = entity.text.lower()
            entity_pattern = re.escape(entity.text)

            for pattern, rel_type in _RELATION_PATTERNS:
                # Search for the pattern with this entity as one of the operands
                for m in re.finditer(entity_pattern, text, re.IGNORECASE):
                    # Get surrounding context
                    start = max(0, m.start() - 200)
                    end = min(len(text), m.end() + 200)
                    context = text[start:end]

                    # Find the other entity in the context
                    for other in entities:
                        if other.text == entity.text:
                            continue
                        if other.text.lower() in context.lower():
                            key = (entity.text, other.text, rel_type)
                            if key not in seen:
                                seen.add(key)
                                confidence = self._compute_confidence(
                                    entity, other, rel_type, context
                                )
                                if confidence >= self.min_confidence:
                                    relations.append(Relation(
                                        source_entity=entity.text,
                                        target_entity=other.text,
                                        relation_type=rel_type,
                                        source_ref=fact_ref,
                                        confidence=confidence,
                                    ))

        # Sort by confidence descending, cap at max_relations
        relations.sort(key=lambda r: r.confidence, reverse=True)
        return relations[:self.max_relations]

    def _compute_confidence(
        self,
        entity: Entity,
        other: Entity,
        rel_type: str,
        context: str,
    ) -> float:
        """Compute confidence score for a relation."""
        base = 0.6

        # Closer entities → higher confidence
        dist = abs(entity.start - other.start)
        if dist < 100:
            base += 0.15
        elif dist < 200:
            base += 0.1
        elif dist < 400:
            base += 0.05

        # Same entity type → lower confidence (less likely to be meaningful)
        if entity.entity_type == other.entity_type:
            base -= 0.1

        # Known relation type boost
        if rel_type in ("USES", "PART_OF", "CREATED", "WORKS_FOR"):
            base += 0.05

        return min(base, 0.95)
