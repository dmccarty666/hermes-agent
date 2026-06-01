"""Hermes Local Memory — shared library.

Covers:
  - SQLite + FTS5 persistence
  - Qdrant vector store
  - JSONL / QMD filesystem
  - Hybrid retrieval scorer
  - HRR compositional retrieval
  - Canonical write pipeline
  - Secret redaction scanner
  - Source reference resolver
  - LMS embedding client
  - Chunker
  - Dreamer worker

Plugin entry: ``plugins/memory/hermes-local/``
Gateway entry: ``hermes_memory_gateway/``
"""

__version__ = "0.2.0"


class HermesMemoryCore:
    """Placeholder. Full initialization lands in Phase 1.2.1 (SQLite) and beyond.

    The canonical write pipeline, search, and dreamer are accessed through
    submodules directly. This class will be populated in later stories.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "HermesMemoryCore initialization lands in Phase 1.2.1 (SQLite schema). "
            "Import submodules directly until then."
        )