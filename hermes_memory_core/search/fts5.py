# Copyright 2026 David McCarty. All rights reserved.
"""FTS5 keyword search: fts5_search(query, filters, table, limit=10).

Searches FTS5 virtual tables (turns_fts, chunks_fts, facts_fts, decisions_fts)
with filters applied to the companion base table and returns ranked rows with
SQLite snippet() excerpts and source_ref pointers.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from hermes_memory_core.store.sqlite import MemoryDB

logger = logging.getLogger(__name__)

# Map logical table name -> FTS5 virtual table name and content column.
_FTS_TABLE_MAP = {
    "turns":      "turns_fts",
    "chunks":     "chunks_fts",
    "facts":      "facts_fts",
    "decisions":  "decisions_fts",
}

# Content column per FTS5 table.
_CONTENT_COL = {
    "turns":     "content",
    "chunks":    "chunk_text",
    "facts":     "fact_text",
    "decisions": "decision_text",
}

# Join columns from base table (everything except the FTS5 content column).
_EXTRA_JOIN_COLS = {
    "turns":     ["turn_id",  "session_id", "role",       "timestamp"],
    "chunks":    ["chunk_id", "session_id",                              ],
    "facts":     ["fact_id",  "scope",                                        ],
    "decisions": ["decision_id", "rationale",                                ],
}

# Base tables (for join on rowid).
_BASE_TABLE = {
    "turns":     "turns",
    "chunks":    "chunks",
    "facts":     "facts",
    "decisions": "decisions",
}


def fts5_search(
    query: str,
    filters: Dict[str, Any],
    table: str = "turns",
    limit: int = 10,
    memory_db: Optional["MemoryDB"] = None,
) -> List[Dict[str, Any]]:
    """Search an FTS5 virtual table and return ranked results with snippets.

    Parameters
    ----------
    query : str
        FTS5 full-text query string.
    filters : dict
        Scoping predicates:
        - project     -- match rows via the session's project field
        - session_id  -- match rows in a specific session
        - date_from   -- ISO date string; rows must have timestamp >= this
        - date_to     -- ISO date string; rows must have timestamp <= this
        - role        -- for table="turns": role column must equal value
    table : str
        Logical table name ("turns", "chunks", "facts", "decisions").
    limit : int
        Maximum results to return (default 10).
    memory_db : MemoryDB, optional
        MemoryDB instance to use. If omitted, a default instance is created
        (connects to ~/.hermes/memory/index/memory.sqlite).

    Returns
    -------
    list[dict]
        Each dict contains at minimum:
        - source_ref -- pointer of the form {kind}:{id}
          (e.g. session:sess_001#turn=t_01)
        - snippet    -- excerpt from SQLite snippet() around the match
        - rank       -- bm25 score (lower is better for FTS5)
        - table-specific fields (content, turn_id, etc.)
        Returns an empty list when there are no hits or the table is unknown.
    """
    if table not in _FTS_TABLE_MAP:
        logger.warning("fts5_search: unknown table %r", table)
        return []

    fts_table   = _FTS_TABLE_MAP[table]
    base_table  = _BASE_TABLE[table]
    content_col = _CONTENT_COL[table]
    join_cols   = _EXTRA_JOIN_COLS[table]

    # FTS5 virtual tables only expose the content column plus rowid.
    # We match in FTS5 for ranking + snippet, then join on rowid to get
    # the business-key columns from the base table.

    # project is scoped via the sessions table because turns do not have
    # a project column (it lives on sessions).
    has_project_filter = bool(filters.get("project"))

    where_parts: List[str] = []
    params:      List[Any] = []

    if filters.get("session_id"):
        where_parts.append("t.session_id = ?")
        params.append(filters["session_id"])
    if has_project_filter:
        where_parts.append("s.project = ?")
        params.append(filters["project"])
    if filters.get("role"):
        where_parts.append("t.role = ?")
        params.append(filters["role"])
    if filters.get("date_from"):
        where_parts.append("t.timestamp >= ?")
        params.append(filters["date_from"])
    if filters.get("date_to"):
        where_parts.append("t.timestamp <= ?")
        params.append(filters["date_to"])

    filter_where = " AND ".join(where_parts) if where_parts else "1=1"

    # Join sessions when project filter is needed.
    sessions_join = "JOIN sessions s ON s.session_id = t.session_id" if has_project_filter else ""

    extra_cols = ", ".join([f"t.{c}" for c in join_cols])
    snippet_sql = f"snippet({fts_table}, 0, '[', ']', '...', 32)"

    sql = f"""
        SELECT
            t.rowid,
            {extra_cols},
            t.{content_col},
            {snippet_sql} AS snippet,
            bm25({fts_table}) AS rank
        FROM {fts_table}
        JOIN {base_table} t ON t.rowid = {fts_table}.rowid
        {sessions_join}
        WHERE {fts_table} MATCH ?
          AND ({filter_where})
        ORDER BY rank
        LIMIT ?
    """

    conn: Optional[sqlite3.Connection] = None
    try:
        from hermes_memory_core.store.sqlite import MemoryDB as MDB

        db: "MemoryDB"
        if memory_db is not None:
            db = memory_db
        else:
            db = MDB()
            db.initialize()

        conn = db._connect()
        cur = conn.execute(sql, [query] + params + [limit])
        rows = cur.fetchall()
        return [_row_to_dict(row, table, join_cols, content_col) for row in rows]

    except sqlite3.OperationalError as exc:
        logger.warning("fts5_search: query failed -- %s", exc)
        return []
    finally:
        if conn is not None:
            conn.close()


def _row_to_dict(
    row: tuple,
    table: str,
    join_cols: List[str],
    content_col: str,
) -> Dict[str, Any]:
    """Map a result row to a dict with a formatted source_ref.

    Row layout:
      [0]          = rowid (used for join, not surfaced)
      [1..N]       = join_cols values (variable per table)
      [-2]         = snippet
      [-1]         = rank
      one before those = content column value
    """
    n = len(join_cols)
    # Row layout: [0]=rowid, [1..n]=join_cols, [n+1]=content, [n+2]=snippet, [n+3]=rank
    content_value = row[n + 1]

    result: Dict[str, Any] = {}

    # Map join columns back to their names.
    for i, col in enumerate(join_cols):
        result[col] = row[1 + i]

    # Always surface the searchable content field.
    result[content_col] = content_value

    # Attach snippet and rank.
    result["snippet"] = row[-2]
    result["rank"]    = row[-1]

    # Build source_ref based on table kind.
    if table == "turns":
        result["source_ref"] = (
            f"session:{result.get('session_id', '')}#turn={result.get('turn_id', '')}"
        )
    elif table == "chunks":
        result["source_ref"] = f"chunk:{result.get('chunk_id', '')}"
    elif table == "facts":
        result["source_ref"] = f"fact:{result.get('fact_id', '')}"
    elif table == "decisions":
        result["source_ref"] = f"decision:{result.get('decision_id', '')}"

    return result