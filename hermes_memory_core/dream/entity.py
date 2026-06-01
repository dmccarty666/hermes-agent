"""Entity extraction for the dreamer pipeline.

Extracts typed entities (PERSON, ORG, PROJECT, TECH, DATE, LOCATION, etc.)
from text using regex heuristics. Used by the relation extractor and
entity-linking pipeline.

This is a lightweight heuristic extractor — the dreamer also uses the LLM
for entity extraction when available (see worker.py _extract_entities_llm).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A single extracted entity."""
    text: str
    entity_type: str  # PERSON, ORG, PROJECT, TECH, DATE, LOCATION, etc.
    confidence: float = 0.0
    start: int = -1
    end: int = -1
    # Aliases for entity resolution (e.g., "David McCarty" → "David")
    aliases: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Alias for 'text' to match the worker.py interface."""
        return self.text


# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

# Common tech/product/project name patterns
_TECH_PATTERNS = [
    r'\b(Qwen(?:3\.\d+)?(?:-?\d+[Bb]?))\b',
    r'\b(Nemotron(?:\s*\d+)?(?:\s*\d+B)?)\b',
    r'\b(DeepSeek(?:\s*\d+)?(?:\s*V\d+)?)\b',
    r'\b(GPT(?:-\d+)?(?:\s*\d+)?(?:\s*[TBK]b)?)\b',
    r'\b(Claude(?:\s*\d+)?(?:\s*[0-9a-zA-Z]+)?)\b',
    r'\b(Anthropic)\b',
    r'\b(OpenAI)\b',
    r'\b(Hermes(?:\s*(?:Agent|Memory|Dashboard|Gateway))?)\b',
    r'\b(Qdrant)\b',
    r'\b(LMStudio|LMS|lms)\b',
    r'\b(Spark(?:2)?(?:\s*\.\d+)?)\b',
    r'\b(Spark2)\b',
    r'\b(FastAPI)\b',
    r'\b(SQLite)\b',
    r'\b(Redis)\b',
    r'\b(Manim|Manim CE)\b',
    r'\b(Chart\.js)\b',
    r'\b(GSAP)\b',
    r'\b(Puppeteer|puppeteer)\b',
    r'\b(HyperFrames)\b',
    r'\b(Mem0)\b',
    r'\b(Plaid)\b',
    r'\b(Doppler)\b',
    r'\b(Openclaw|OpenClaw)\b',
    r'\b(Agent Zero)\b',
    r'\b(Firebase)\b',
    r'\b(Hume)\b',
    r'\b(PostgreSQL|Postgres)\b',
    r'\b(Chart\.js)\b',
    r'\b(Lucide)\b',
]

# Date patterns
_DATE_PATTERNS = [
    r'\b(202[0-6]-\d{2}-\d{2})\b',
    r'\b(\d{4}-\d{2}-\d{2})\b',
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
]

# Common person names (heuristic — would need a full NER model for production)
_PERSON_PATTERNS = [
    r'\b(David\s+McCarty)\b',
    r'\b(David\s+M\.)\b',
    r'\b(James)\b',
]

# Organization / company patterns
_ORG_PATTERNS = [
    r'\b(NousResearch)\b',
    r'\b(Google)\b',
    r'\b(Microsoft)\b',
    r'\b(Amazon|AWS)\b',
    r'\b(OpenRouter)\b',
]


def extract_entities(text: str) -> List[Entity]:
    """Extract typed entities from text using heuristic patterns.

    Returns a list of Entity objects. For production use, this should be
    replaced with an actual NER model (e.g., spaCy, transformers).

    Args:
        text: The text to extract entities from.

    Returns:
        List of Entity objects with text, entity_type, confidence, etc.
    """
    entities: List[Entity] = []
    seen: set = set()

    # Tech/product names
    for pattern in _TECH_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            key = (m.group(0).lower(), "TECH")
            if key not in seen:
                seen.add(key)
                entities.append(Entity(
                    text=m.group(0),
                    entity_type="TECH",
                    confidence=0.85,
                    start=m.start(),
                    end=m.end(),
                ))

    # Organizations
    for pattern in _ORG_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            key = (m.group(0).lower(), "ORG")
            if key not in seen:
                seen.add(key)
                entities.append(Entity(
                    text=m.group(0),
                    entity_type="ORG",
                    confidence=0.85,
                    start=m.start(),
                    end=m.end(),
                ))

    # Dates
    for pattern in _DATE_PATTERNS:
        for m in re.finditer(pattern, text):
            key = (m.group(0).lower(), "DATE")
            if key not in seen:
                seen.add(key)
                entities.append(Entity(
                    text=m.group(0),
                    entity_type="DATE",
                    confidence=0.95,
                    start=m.start(),
                    end=m.end(),
                ))

    # Person names
    for pattern in _PERSON_PATTERNS:
        for m in re.finditer(pattern, text):
            key = (m.group(0).lower(), "PERSON")
            if key not in seen:
                seen.add(key)
                entities.append(Entity(
                    text=m.group(0),
                    entity_type="PERSON",
                    confidence=0.75,
                    start=m.start(),
                    end=m.end(),
                ))

    return entities
