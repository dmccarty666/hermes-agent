# Copyright 2026 David McCarty. All rights reserved.
"""Tool schemas and dispatch for hermes-memory.

Exposes memory_query (keyword/sessions/recent modes) per TDD §5.2.
All other modes raise NotImplementedError.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from hermes_memory_core.search.fts5 import fts5_search

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Tool schemas
# ------------------------------------------------------------------

MEMORY_QUERY_SCHEMA: Dict[str, Any] = {
    "name": "memory_query",
    "description": (
        "Search the local memory. Modes: 'keyword' (FTS5 full-text), "
        "'sessions' (recent sessions), 'recent' (recent turns). "
        "Other modes ('semantic', 'hybrid', etc.) are not yet implemented."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query":   {"type": "string"},
            "mode":    {
                "type": "string",
                "enum": ["keyword", "sessions", "recent"],
                "default": "keyword",
            },
            "project": {"type": "string"},
            "entity":  {"type": "string"},
            "entities": {"type": "array", "items": {"type": "string"}},
            "filters": {"type": "object"},
            "limit":   {"type": "integer", "default": 10},
        },
        "required": ["query"],
    },
}


# ------------------------------------------------------------------
# get_tool_schemas
# ------------------------------------------------------------------

def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return the list of tool schemas for Phase 2."""
    return [MEMORY_QUERY_SCHEMA]


# ------------------------------------------------------------------
# handle_tool_call
# ------------------------------------------------------------------

def handle_tool_call(tool_name: str, args: Dict[str, Any], **kwargs) -> str:
    """Dispatch a tool call to its handler; return a JSON string result."""
    if tool_name == "memory_query":
        return json.dumps(_handle_memory_query(args, **kwargs))
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _handle_memory_query(args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Handle memory_query tool call.

    Args:
        args: The tool arguments (query, mode, project, filters, limit).
        **kwargs: May contain memory_db for test isolation.

    Returns:
        A result dict with shape:
          {results: [{content, source_ref, excerpt, score, mode}],
           query, mode, backend_hints: ['fts5']}

    Raises:
        NotImplementedError: for unimplemented modes.
    """
    query = args.get("query", "")
    mode = args.get("mode", "keyword")
    project = args.get("project")
    filters = args.get("filters", {})
    limit = args.get("limit", 10)

    # Wire project into filters
    if project:
        filters = dict(filters)  # copy to avoid mutation
        filters["project"] = project

    memory_db = kwargs.get("memory_db")

    # Route by mode
    if mode == "keyword":
        raw_results = fts5_search(query, filters, table="turns", limit=limit, memory_db=memory_db)
        results = [
            {
                "content": _safe_content(r),
                "source_ref": r.get("source_ref", ""),
                "excerpt": r.get("snippet", ""),
                "score": r.get("rank", 0.0),
                "mode": "keyword",
            }
            for r in raw_results
        ]
        return {
            "results": results,
            "query": query,
            "mode": mode,
            "backend_hints": ["fts5"],
        }

    if mode == "sessions":
        # Return recent sessions by scanning turns and grouping by session_id
        raw_results = fts5_search(query, filters, table="turns", limit=limit, memory_db=memory_db)
        # Deduplicate by session_id — keep first occurrence
        seen: set = set()
        deduped: list[Dict[str, Any]] = []
        for r in raw_results:
            sid = r.get("session_id", "")
            if sid and sid not in seen:
                seen.add(sid)
                deduped.append(r)
        results = [
            {
                "content": _safe_content(r),
                "source_ref": f"session:{r.get('session_id', '')}",
                "excerpt": r.get("snippet", ""),
                "score": r.get("rank", 0.0),
                "mode": "sessions",
            }
            for r in deduped
        ]
        return {
            "results": results,
            "query": query,
            "mode": mode,
            "backend_hints": ["fts5"],
        }

    if mode == "recent":
        # Recent turns: no query needed, just get latest turns
        raw_results = fts5_search(
            "", filters, table="turns", limit=limit, memory_db=memory_db
        )
        results = [
            {
                "content": _safe_content(r),
                "source_ref": r.get("source_ref", ""),
                "excerpt": r.get("snippet", ""),
                "score": r.get("rank", 0.0),
                "mode": "recent",
            }
            for r in raw_results
        ]
        return {
            "results": results,
            "query": query,
            "mode": mode,
            "backend_hints": ["fts5"],
        }

    # Unimplemented modes
    raise NotImplementedError(f"semantic mode not yet implemented")


def _safe_content(r: Dict[str, Any]) -> str:
    """Extract the content field, falling back to snippet or empty string."""
    for key in ("content", "content", "chunk_text", "fact_text", "decision_text"):
        if key in r and r[key]:
            return r[key]
    return r.get("snippet", "")