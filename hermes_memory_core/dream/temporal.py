"""Temporal entity reasoning for the dreamer pipeline.

Detects versioned entities (e.g., "Project X → Project Y") and extracts
temporal relations between them. Uses heuristic-based detection with an
optional LLM fallback.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class TemporalEdge:
    """A temporal relation between two entities."""
    source_entity: str
    target_entity: str
    relation_type: str  # e.g., "evolved_from", "renamed_to", "merged_into"
    confidence: float = 1.0


# Patterns for detecting versioned entities
_VERSION_PATTERNS = [
    # "Project X → Project Y" or "X -> Y"
    r'([A-Za-z0-9_\-\.]+)\s*[→>]\s*([A-Za-z0-9_\-\.]+)',
    # "X becomes Y"
    r'([A-Za-z0-9_\-\.]+)\s+becomes?\s+([A-Za-z0-9_\-\.]+)',
    # "X was renamed to Y"
    r'([A-Za-z0-9_\-\.]+)\s+renamed?\s+to\s+([A-Za-z0-9_\-\.]+)',
    # "X replaced by Y"
    r'([A-Za-z0-9_\-\.]+)\s+replaced?\s+(?:by|with)\s+([A-Za-z0-9_\-\.]+)',
    # "X merged into Y"
    r'([A-Za-z0-9_\-\.]+)\s+merged?\s+into\s+([A-Za-z0-9_\-\.]+)',
    # "X split into Y"
    r'([A-Za-z0-9_\-\.]+)\s+split\s+into\s+([A-Za-z0-9_\-\.]+)',
    # "X deprecated in favor of Y"
    r'([A-Za-z0-9_\-\.]+)\s+deprecated\s+(?:in\s+)?(?:favor\s+of\s+)?([A-Za-z0-9_\-\.]+)',
    # "X superseded by Y"
    r'([A-Za-z0-9_\-\.]+)\s+superseded?\s+by\s+([A-Za-z0-9_\-\.]+)',
    # "X (now Y)"
    r'([A-Za-z0-9_\-\.]+)\s+\(now\s+([A-Za-z0-9_\-\.]+)\)',
    # "X (formerly Y)"
    r'([A-Za-z0-9_\-\.]+)\s+\(formerly\s+([A-Za-z0-9_\-\.]+)\)',
    # "X (previously Y)"
    r'([A-Za-z0-9_\-\.]+)\s+\(previously\s+([A-Za-z0-9_\-\.]+)\)',
    # "X (aka Y)"
    r'([A-Za-z0-9_\-\.]+)\s+\(aka\s+([A-Za-z0-9_\-\.]+)\)',
    # "X (alias Y)"
    r'([A-Za-z0-9_\-\.]+)\s+\(alias\s+([A-Za-z0-9_\-\.]+)\)',
]


def detect_versioned_entities(text: str) -> List[Tuple[str, str, str]]:
    """Detect versioned/renamed entities from text.
    
    Returns list of (source, target, relation_type) tuples.
    """
    edges: List[Tuple[str, str, str]] = []
    seen: set = set()
    
    for pattern in _VERSION_PATTERNS:
        for match in re.finditer(pattern, text):
            source = match.group(1).strip()
            target = match.group(2).strip()
            
            # Determine relation type based on the pattern
            if '→' in pattern or '>' in pattern:
                rel_type = "evolved_to"
            elif 'becomes' in pattern:
                rel_type = "became"
            elif 'renamed' in pattern:
                rel_type = "renamed_to"
            elif 'replaced' in pattern:
                rel_type = "replaced_by"
            elif 'merged' in pattern:
                rel_type = "merged_into"
            elif 'split' in pattern:
                rel_type = "split_into"
            elif 'deprecated' in pattern:
                rel_type = "deprecated_for"
            elif 'superseded' in pattern:
                rel_type = "superseded_by"
            elif 'now' in pattern:
                rel_type = "renamed_to"
            elif 'formerly' in pattern:
                rel_type = "formerly"
            elif 'previously' in pattern:
                rel_type = "formerly"
            elif 'aka' in pattern:
                rel_type = "aka"
            elif 'alias' in pattern:
                rel_type = "alias"
            else:
                rel_type = "related_to"
            
            # Skip trivial matches
            if len(source) < 2 or len(target) < 2:
                continue
            if source == target:
                continue
            if source.lower() in ('x', 'y', 'z', 'a', 'b', 'c'):
                continue
            if target.lower() in ('x', 'y', 'z', 'a', 'b', 'c'):
                continue
            
            key = (source.lower(), target.lower(), rel_type)
            if key not in seen:
                seen.add(key)
                edges.append((source, target, rel_type))
    
    return edges


def _extract_temporal_relations(
    text: str,
    entity_names: List[str],
    llm_complete: Callable,
) -> List[TemporalEdge]:
    """Extract temporal relations between entities using LLM.
    
    Falls back to heuristic detection if LLM call fails.
    """
    edges: List[TemporalEdge] = []
    
    # First, try heuristic detection
    heuristic_edges = detect_versioned_entities(text)
    for source, target, rel_type in heuristic_edges:
        edges.append(TemporalEdge(
            source_entity=source,
            target_entity=target,
            relation_type=rel_type,
        ))
    
    # Then, try LLM for additional relations
    try:
        prompt = f"""Analyze the following text and identify temporal/evolutionary relationships between these entities.

Entities: {', '.join(entity_names)}

Text:
{text}

Return a JSON array of objects with keys: source_entity, target_entity, relation_type.
relation_type should be one of: evolved_from, renamed_to, merged_into, split_into, deprecated_for, superseded_by, replaced_by, formerly, aka, alias, related_to.

If no temporal relations exist, return an empty array [].

JSON:"""
        response = llm_complete(prompt, json_mode=True)
        if response and response.strip():
            import json
            try:
                llm_edges = json.loads(response)
                if isinstance(llm_edges, list):
                    for edge in llm_edges:
                        if isinstance(edge, dict) and 'source_entity' in edge and 'target_entity' in edge:
                            # Check for duplicates
                            existing = {(e.source_entity.lower(), e.target_entity.lower()) for e in edges}
                            key = (edge['source_entity'].lower(), edge['target_entity'].lower())
                            if key not in existing:
                                edges.append(TemporalEdge(
                                    source_entity=edge['source_entity'],
                                    target_entity=edge['target_entity'],
                                    relation_type=edge.get('relation_type', 'related_to'),
                                ))
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug("LLM temporal extraction JSON parse error: %s", e)
    except Exception as e:
        logger.debug("LLM temporal extraction failed, using heuristics only: %s", e)
    
    return edges
