"""Source reference resolver.

Converts string refs like ``turn:abc123``, ``fact:42``, ``session:xyz``
into structured provenance for every derived memory write.

Phase 1 (T-001): stub. Full implementation in later phases.
"""

from typing import Optional


class SourceRef:
    """Parsed source reference."""
    __slots__ = ("namespace", "id", "raw")

    def __init__(self, namespace: str, id: str, raw: str):
        self.namespace = namespace
        self.id = id
        self.raw = raw

    def __repr__(self) -> str:
        return f"SourceRef({self.namespace}:{self.id})"


def resolve_source_ref(ref: str) -> Optional[SourceRef]:
    """Parse a source reference string into a structured SourceRef.

    Supported formats:
      turn:<turn_id>         — a specific turn
      session:<session_id>   — an entire session
      fact:<fact_id>         — a stored fact
      decision:<id>          — a stored decision
      question:<id>          — a stored open question
      migration:<source>      — migrated from another system

    Returns None if the ref string is not parseable.
    """
    if not ref or not isinstance(ref, str):
        return None
    if ":" not in ref:
        return None
    namespace, _, id = ref.partition(":")
    if not namespace or not id:
        return None
    return SourceRef(namespace=namespace.strip(), id=id.strip(), raw=ref)


def format_source_ref(namespace: str, id: str) -> str:
    """Build a canonical source ref string."""
    return f"{namespace}:{id}"