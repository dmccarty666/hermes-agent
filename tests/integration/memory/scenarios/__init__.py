"""MVP Acceptance Test Suite — Scenario Suite.

Each module (test_scenario_A through test_scenario_M) covers one scenario
from Plan.md §9. Run all with:
    python -m pytest tests/integration/memory/scenarios/ -v

Individual scenarios:
    python -m pytest tests/integration/memory/scenarios/test_scenario_A.py -v
    python -m pytest tests/integration/memory/scenarios/test_scenario_B.py -v
    ... etc.
"""

from pathlib import Path

SCENARIOS = [
    "A — Lossless Capture",
    "B — Redaction",
    "C — Keyword Search (FTS5)",
    "D — Semantic Search (Qdrant)",
    "E — Hybrid Search",
    "F — Graceful Degradation",
    "G — Dreaming",
    "H — Contradiction Detection",
    "I — Provider Swap",
    "J — Narrative Thread /new Injection",
    "K — Migration from Holographic",
    "L — Backup and Restore",
    "M — Handoff & Multi-Agent",
]

__all__ = ["SCENARIOS"]