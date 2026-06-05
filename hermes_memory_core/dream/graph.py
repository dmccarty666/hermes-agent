"""EntityGraph — NetworkX-backed co-occurrence graph for the memory gateway.

Builds an undirected weighted graph from entity co-occurrences in fact_entities.
Each node = an entity; each edge = two entities appearing in the same fact.
Edge weight = number of facts where the pair co-occurs.

Provides: page_rank, find_path (BFS), ego_neighbors, entity_facts, build.
Used by all /graph/* routes in the gateway.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import networkx as nx

if TYPE_CHECKING:
    from hermes_memory_core.store.sqlite import MemoryStore

logger = logging.getLogger(__name__)


class EntityGraph:
    """Entity co-occurrence graph backed by NetworkX.

    Nodes: entities (from the entities table)
    Edges: two entities that co-occur in one or more facts (via fact_entities join)

    The graph is built lazily on first access and cached on self._graph.
    """

    def __init__(self, store: "MemoryStore"):
        self._store = store
        self._graph: nx.Graph | None = None
        self._name_to_id: Dict[str, str] | None = None
        self._id_to_name: Dict[str, str] | None = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def build(self) -> nx.Graph:
        """Build (or return cached) NetworkX undirected weighted graph.

        Nodes are entity names. Edges connect entities that co-occur in facts.
        Edge weight = count of shared facts.
        """
        if self._graph is not None:
            return self._graph

        conn = self._store._conn
        if not conn:
            self._graph = nx.Graph()
            return self._graph

        # Load all entity name→id mappings
        id_name: List[Tuple[str, str]] = conn.execute(
            "SELECT entity_id, name FROM entities"
        ).fetchall()
        self._name_to_id = {name.lower(): eid for eid, name in id_name}
        self._id_to_name = {eid: name for eid, name in id_name}

        # Build co-occurrence edges: for each fact, connect all its entities
        co_occur: Dict[Tuple[str, str], int] = defaultdict(int)
        rows = conn.execute("""
            SELECT fact_id, entity_id
            FROM fact_entities
            ORDER BY fact_id
        """).fetchall()

        # Group entities by fact
        fact_entities: Dict[str, List[str]] = defaultdict(list)
        for fact_id, entity_id in rows:
            fact_entities[fact_id].append(entity_id)

        for entity_list in fact_entities.values():
            # Connect every pair in the same fact
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    a, b = entity_list[i], entity_list[j]
                    if a == b:
                        continue
                    edge = (a, b) if a < b else (b, a)
                    co_occur[edge] += 1

        # Also add entity_relations as typed edges (weighted by confidence)
        rel_rows = conn.execute("""
            SELECT source_entity_id, target_entity_id, relation_type, confidence
            FROM entity_relations
        """).fetchall()

        for src, tgt, rel_type, confidence in rel_rows:
            edge = (src, tgt) if src < tgt else (tgt, src)
            # Blend with any existing co-occurrence weight
            co_occur[edge] = co_occur.get(edge, 0) + int((confidence or 0.5) * 10)

        # Build NetworkX graph
        G = nx.Graph()
        for (a, b), weight in co_occur.items():
            name_a = self._id_to_name.get(a, a)
            name_b = self._id_to_name.get(b, b)
            G.add_edge(name_a, name_b, weight=weight, entity_id_a=a, entity_id_b=b)

        self._graph = G
        logger.info("EntityGraph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G

    def _ensure_built(self) -> None:
        """Eager build if not already built."""
        if self._graph is None:
            self.build()

    def _resolve_name(self, name: str) -> Optional[str]:
        """Resolve an entity name to its entity_id."""
        if self._name_to_id is None:
            self.build()
        return self._name_to_id.get(name.lower()) if self._name_to_id else None

    def page_rank(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """Return top-k entities by PageRank centrality (name, score pairs)."""
        self._ensure_built()
        if self._graph.number_of_nodes() == 0:
            return []
        pr = nx.pagerank(self._graph, weight="weight")
        return sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def find_path(self, source: str, target: str, max_depth: int = 4) -> List[str]:
        """BFS shortest path between two entity names. Returns [] if unreachable."""
        self._ensure_built()
        if source not in self._graph or target not in self._graph:
            return []
        try:
            path = nx.shortest_path(self._graph, source=source, target=target)
            return path[: max_depth + 1]
        except nx.NetworkXNoPath:
            return []

    def ego_neighbors(self, entity: str, depth: int = 1) -> List[str]:
        """All entity names within N hops of the given entity."""
        self._ensure_built()
        if entity not in self._graph:
            return []
        try:
            nodes = nx.single_source_shortest_path_length(self._graph, entity, cutoff=depth)
            return list(nodes.keys())
        except Exception:
            return []

    def entity_facts(self, entity: str, limit: int = 20) -> List[Dict[str, Any]]:
        """All facts involving this entity, with role annotation."""
        conn = self._store._conn
        if not conn:
            return []

        eid = self._resolve_name(entity) or entity
        rows = conn.execute("""
            SELECT f.fact_id, f.fact_text, f.project, f.created_at, fe.role as entity_role
            FROM facts f
            JOIN fact_entities fe ON f.fact_id = fe.fact_id
            JOIN entities e ON fe.entity_id = e.entity_id
            WHERE LOWER(e.name) = LOWER(?)
            ORDER BY f.created_at DESC
            LIMIT ?
        """, (entity, limit)).fetchall()

        return [
            {
                "fact_id": row[0],
                "fact_text": row[1],
                "project": row[2],
                "created_at": row[3],
                "role": row[4],
            }
            for row in rows
        ]

    # -------------------------------------------------------------------------
    # Lifecycle helpers (used by /graph/lifecycle)
    # -------------------------------------------------------------------------

    def get_entities_by_status(self, status: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Return entities filtered by lifecycle status."""
        conn = self._store._conn
        if not conn:
            return []
        try:
            rows = conn.execute("""
                SELECT entity_name, status, last_seen_at, mention_count
                FROM entity_lifecycle
                WHERE status = ?
                ORDER BY mention_count DESC
                LIMIT ?
            """, (status, limit)).fetchall()
            return [
                {"entity_name": r[0], "status": r[1], "last_seen_at": r[2], "mention_count": r[3]}
                for r in rows
            ]
        except Exception:
            return []
