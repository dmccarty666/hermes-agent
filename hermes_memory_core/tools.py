# Copyright 2026 David McCarty. All rights reserved.
"""Tool schemas and dispatch for hermes-memory.

Exposes memory_query (keyword/sessions/recent modes) per TDD §5.2.
memory_get_source (resolve source refs) per TDD §5.2.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from hermes_memory_core.search.fts5 import fts5_search
from hermes_memory_core.search.semantic import semantic_search
from hermes_memory_core.source import resolve

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
                "enum": ["keyword", "sessions", "recent", "probe", "related", "reason"],
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

MEMORY_GET_SOURCE_SCHEMA: Dict[str, Any] = {
    "name": "memory_get_source",
    "description": (
        "Resolve a source_ref back to original content. "
        "Handles session turns, facts, decisions, and chunks. "
        "Returns {kind:'missing',...} for archived/missing refs — no error raised."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source_ref": {"type": "string", "description": "Ref to resolve: session:{id}#turn={n}, fact:{id}, decision:{id}, chunk:{id}"},
            "expand":     {"type": "boolean", "default": False, "description": "Also resolve nested source_refs found in tool_call provenance chains"},
        },
        "required": ["source_ref"],
    },
}

MEMORY_DREAM_NOW_SCHEMA: Dict[str, Any] = {
    "name": "memory_dream_now",
    "description": (
        "Trigger the nightly dreamer pipeline: extract facts, decisions, "
        "and open questions from recent conversation sessions. "
        "Runs the full 9-stage LLM-driven extraction pipeline. "
        "Scope options: 'session' (with session_id), 'since_last' (default), 'all'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["session", "since_last", "all"],
                "default": "since_last",
                "description": "Scope of sessions to process.",
            },
            "session_id": {
                "type": "string",
                "description": "Specific session_id when scope='session'.",
            },
        },
        "required": [],
    },
}


# ------------------------------------------------------------------
# get_tool_schemas
# ------------------------------------------------------------------

def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return the list of tool schemas for Phase 2."""
    return [
        MEMORY_QUERY_SCHEMA,
        MEMORY_GET_SOURCE_SCHEMA,
        MEMORY_RECENT_CONTEXT_SCHEMA,
        MEMORY_DREAM_NOW_SCHEMA,
    ]


# ------------------------------------------------------------------
# handle_tool_call
# ------------------------------------------------------------------

def handle_tool_call(tool_name: str, args: Dict[str, Any], **kwargs) -> str:
    """Dispatch a tool call to its handler; return a JSON string result."""
    if tool_name == "memory_query":
        return json.dumps(_handle_memory_query(args, **kwargs))
    if tool_name == "memory_get_source":
        return json.dumps(_handle_memory_get_source(args, **kwargs))
    if tool_name == "memory_recent_context":
        return json.dumps(_handle_memory_recent_context(args, **kwargs))
    if tool_name == "memory_dream_now":
        return json.dumps(_handle_memory_dream_now(args, **kwargs))
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

    if mode == "semantic":
        try:
            raw_results = semantic_search(
                query=query,
                filters=filters,
                limit=limit,
            )
        except Exception as exc:
            return {
                "results": [],
                "query": query,
                "mode": mode,
                "backend_hints": ["qdrant"],
                "error": str(exc),
            }
        results = [
            {
                "content": r.get("content", ""),
                "source_ref": r.get("source_ref", ""),
                "excerpt": r.get("content", "")[:200],
                "score": r.get("score", 0.0),
                "mode": "semantic",
            }
            for r in raw_results
        ]
        return {
            "results": results,
            "query": query,
            "mode": mode,
            "backend_hints": ["qdrant"],
        }

    # Unimplemented modes
    if mode in ("probe", "related", "reason"):
        from hermes_memory_core.search.hrri import FactRetriever

        entity_arg = args.get("entity", args.get("query", ""))
        entities_arg = args.get("entities", [])

        if mode in ("probe", "related") and not entity_arg:
            return {
                "results": [],
                "query": entity_arg,
                "mode": mode,
                "backend_hints": ["hrr"],
                "error": f"mode '{mode}' requires 'entity' parameter",
            }

        db = _get_memory_db(kwargs)
        retriever = FactRetriever(db)

        if mode == "probe":
            raw_results = retriever.probe(entity=entity_arg, category=filters.get("category"), limit=limit)
        elif mode == "related":
            raw_results = retriever.related(entity=entity_arg, category=filters.get("category"), limit=limit)
        else:  # reason
            entities_list = entities_arg if entities_arg else entity_arg.split() if entity_arg else []
            raw_results = retriever.reason(entities=entities_list, category=filters.get("category"), limit=limit)

        results = [
            {
                "content": r.get("content", ""),
                "source_ref": f"fact:{r.get('fact_id', '')}",
                "excerpt": (r.get("content", "") or "")[:200],
                "score": r.get("score", 0.0),
                "mode": mode,
            }
            for r in raw_results
        ]
        return {
            "results": results,
            "query": entity_arg or " ".join(entities_arg),
            "mode": mode,
            "backend_hints": ["hrr"],
        }

    raise NotImplementedError(f"mode '{mode}' not yet implemented")


def _safe_content(r: Dict[str, Any]) -> str:
    """Extract the content field, falling back to snippet or empty string."""
    for key in ("content", "content", "chunk_text", "fact_text", "decision_text"):
        if key in r and r[key]:
            return r[key]
    return r.get("snippet", "")


def _handle_memory_get_source(args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Handle memory_get_source tool call.

    Args:
        args: The tool arguments (source_ref, expand).
        **kwargs: May contain memory_db for test isolation.

    Returns:
        A result dict with kind field: session:turn, fact, decision, chunk,
        or {kind:'missing', source_ref, reason}.
    """
    source_ref = args.get("source_ref", "")
    expand = bool(args.get("expand", False))
    memory_db = kwargs.get("memory_db")
    return resolve(source_ref, memory_db=memory_db, expand=expand)


# ------------------------------------------------------------------
# memory_recent_context (Epic 4.3.1)
# ------------------------------------------------------------------

MEMORY_RECENT_CONTEXT_SCHEMA: Dict[str, Any] = {
    "name": "memory_recent_context",
    "description": (
        "Compact working set for session start: pinned facts + active project "
        "facts + recent decisions + open questions + recent dream summaries. "
        "Token-budget aware."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project":   {"type": "string"},
            "max_chars": {"type": "integer", "default": 4000},
        },
    },
}


def _get_memory_db(kwargs: Any) -> "MemoryDB":
    """Return a MemoryDB instance, using test override if provided."""
    from hermes_memory_core.store.sqlite import MemoryDB as MDB
    if kwargs.get("memory_db") is not None:
        return kwargs["memory_db"]
    db = MDB()
    db.initialize()
    return db


def _handle_memory_recent_context(
    args: Dict[str, Any], **kwargs
) -> Dict[str, Any]:
    """Build a compact recent-context response fitting max_chars budget.

    Sources (per Story 4.3.1 AC):
      - Pinned user facts (scope='user', status='active', top by trust)
      - Active project facts (top N by trust where project=current)
      - Recent decisions (last 14 days)
      - Open questions (status='open')
      - Recent dream summaries (last 7 days)
    """
    project = args.get("project", "")
    max_chars = int(args.get("max_chars", 4000))
    memory_db = kwargs.get("memory_db")

    from hermes_memory_core.store.sqlite import MemoryDB as MDB
    db: "MDB"
    if memory_db is not None:
        db = memory_db
    else:
        db = MDB()
        db.initialize()

    sections: List[Dict[str, Any]] = []

    # 1. Pinned user facts
    pinned = db.get_pinned_facts(limit=20)
    sections.append({"label": "pinned_facts", "items": pinned, "source_kind": "fact"})

    # 2. Project facts (if project specified)
    if project:
        project_facts = db.get_project_facts(project, limit=20)
        sections.append({"label": "project_facts", "items": project_facts, "source_kind": "fact"})

    # 3. Recent decisions (last 14 days)
    decisions = db.get_recent_decisions(days=14, limit=20)
    sections.append({"label": "recent_decisions", "items": decisions, "source_kind": "decision"})

    # 4. Open questions
    questions = db.get_open_questions(limit=20)
    sections.append({"label": "open_questions", "items": questions, "source_kind": "question"})

    # 5. Recent dream summaries (last 7 days)
    dreams = db.get_recent_dream_summaries(days=7, limit=10)
    sections.append({"label": "recent_dreams", "items": dreams, "source_kind": "dream"})

    # Build sections with source_refs
    output_sections = []
    total_chars = 0
    for section in sections:
        rendered = []
        for item in section["items"]:
            text = _render_context_item(section["label"], item, section["source_kind"])
            if total_chars + len(text) <= max_chars:
                rendered.append(item)
                total_chars += len(text)
            else:
                break
        output_sections.append({
            "label": section["label"],
            "count": len(rendered),
            "items": [_summarize_context_item(section["label"], i, section["source_kind"]) for i in rendered],
        })

    return {
        "sections": output_sections,
        "max_chars": max_chars,
        "total_chars": total_chars,
        "project": project or None,
    }


def _render_context_item(label: str, item: Dict[str, Any], kind: str) -> str:
    """Render a context item as a readable string."""
    if kind == "fact":
        return item.get("fact_text", "")
    elif kind == "decision":
        return item.get("decision_text", "")
    elif kind == "question":
        return item.get("question_text", "")
    elif kind == "dream":
        return f"Dream run {item.get('dream_run_id', '')}: {item.get('facts_created', 0)} facts, {item.get('decisions_created', 0)} decisions"
    return ""


def _summarize_context_item(label: str, item: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """Build a summary dict for a context item with source_ref."""
    if kind == "fact":
        return {
            "fact_id": item.get("fact_id"),
            "text": item.get("fact_text", ""),
            "source_ref": f"fact:{item.get('fact_id', '')}",
            "trust_score": 0.5,  # default; trust ranking is by created_at order
        }
    elif kind == "decision":
        return {
            "decision_id": item.get("decision_id"),
            "text": item.get("decision_text", ""),
            "rationale": item.get("rationale"),
            "status": item.get("status", "open"),
            "source_ref": f"decision:{item.get('decision_id', '')}",
        }
    elif kind == "question":
        return {
            "question_id": item.get("question_id"),
            "text": item.get("question_text", ""),
            "priority": item.get("priority"),
            "source_ref": f"question:{item.get('question_id', '')}",
        }
    elif kind == "dream":
        return {
            "dream_run_id": item.get("dream_run_id"),
            "started_at": item.get("started_at"),
            "facts_created": item.get("facts_created", 0),
            "decisions_created": item.get("decisions_created", 0),
            "questions_created": item.get("questions_created", 0),
            "source_ref": f"dream:{item.get('dream_run_id', '')}",
        }
    return {}


def _handle_memory_dream_now(args: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Handle memory_dream_now tool call.

    Triggers the full 9-stage dreamer pipeline.

    Args:
        args: Tool arguments (scope, session_id).
        **kwargs: May contain memory_db for test isolation.

    Returns:
        Result dict with dream run status and counts.
    """
    from hermes_memory_core.dream.worker import DreamWorker

    scope = args.get("scope", "since_last")
    session_id = args.get("session_id")

    worker = DreamWorker()
    try:
        result = worker.dream(scope=scope, session_id=session_id)
        return {
            "status": "completed",
            "dream_run_id": result.dream_run.run_id,
            "scope": result.dream_run.scope,
            "facts_created": result.dream_run.facts_created,
            "decisions_created": result.dream_run.decisions_created,
            "questions_created": result.dream_run.questions_created,
            "contradictions_detected": result.dream_run.contradictions_detected,
            "session_summaries": [
                {
                    "session_id": s.session_id,
                    "summary": s.summary[:500],  # truncate for brevity
                    "facts_count": len(s.facts),
                    "decisions_count": len(s.decisions),
                    "questions_count": len(s.questions),
                }
                for s in result.session_summaries
            ],
        }
    except Exception as exc:
        logger.error("memory_dream_now failed: %s", exc, exc_info=True)
        return {
            "status": "failed",
            "error": str(exc),
            "dream_run_id": "",
            "scope": scope,
            "facts_created": 0,
            "decisions_created": 0,
            "questions_created": 0,
            "contradictions_detected": 0,
            "session_summaries": [],
        }