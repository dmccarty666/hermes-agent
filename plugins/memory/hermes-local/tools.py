"""
Hermes Local Memory — tool schemas and handlers.

Implements the 7 tool interfaces defined in TDD §5.2:
  memory_query, memory_write, memory_update, memory_get_source,
  memory_recent_context, memory_dream_now, fact_feedback
"""

from typing import Any

# ── Tool Schemas ─────────────────────────────────────────────────────────────

MEMORY_QUERY_SCHEMA = {
    "name": "memory_query",
    "description": (
        "Search the local memory. Default mode 'hybrid' combines semantic + keyword + structural. "
        "Modes also include 'semantic', 'keyword', 'facts', 'decisions', 'open_questions', "
        "'sessions', 'daily', 'project', 'recent', 'probe', 'related', 'reason'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query":   {"type": "string"},
            "mode":    {"type": "string", "default": "hybrid"},
            "project": {"type": "string"},
            "entity":  {"type": "string"},
            "entities": {"type": "array", "items": {"type": "string"}},
            "filters": {"type": "object"},
            "limit":   {"type": "integer", "default": 10},
        },
        "required": ["query"],
    },
}

MEMORY_WRITE_SCHEMA = {
    "name": "memory_write",
    "description": (
        "Write a durable memory: fact / decision / open_question. "
        "Source reference required."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "type":       {"type": "string", "enum": ["fact", "decision", "open_question"]},
            "text":       {"type": "string"},
            "project":    {"type": "string"},
            "scope":      {"type": "string", "enum": ["user", "project", "general"]},
            "source_ref": {"type": "string"},
            "confidence": {"type": "number"},
            "tags":       {"type": "string"},
            "rationale":  {"type": "string"},
            "owner":      {"type": "string"},
            "priority":   {"type": "string"},
        },
        "required": ["type", "text"],
    },
}

MEMORY_UPDATE_SCHEMA = {
    "name": "memory_update",
    "description": "Update an existing memory's content / trust / tags / status / category.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id":   {"type": "string"},
            "text":        {"type": "string"},
            "trust_delta": {"type": "number"},
            "tags":        {"type": "string"},
            "status":      {"type": "string", "enum": ["active", "superseded", "disputed", "archived"]},
            "category":    {"type": "string"},
        },
        "required": ["memory_id"],
    },
}

MEMORY_GET_SOURCE_SCHEMA = {
    "name": "memory_get_source",
    "description": "Resolve a source_ref back to original content + excerpt.",
    "parameters": {
        "type": "object",
        "properties": {
            "source_ref": {"type": "string"},
            "expand":     {"type": "boolean", "default": False},
        },
        "required": ["source_ref"],
    },
}

MEMORY_RECENT_CONTEXT_SCHEMA = {
    "name": "memory_recent_context",
    "description": (
        "Compact working set for session start: pinned facts + active project facts + "
        "recent decisions + open questions, token-budget aware."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "project":   {"type": "string"},
            "max_chars": {"type": "integer", "default": 4000},
        },
    },
}

MEMORY_DREAM_NOW_SCHEMA = {
    "name": "memory_dream_now",
    "description": (
        "Trigger an immediate dream run. Default scope 'since_last' processes turns since last "
        "checkpoint. Use 'today' for all turns today, 'date' with a date param for a specific "
        "date, 'project' for a specific project, or 'weekly' for the last 7 days."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scope":   {
                "type": "string",
                "default": "since_last",
                "enum": ["since_last", "today", "date", "project", "weekly"],
            },
            "date":    {"type": "string"},
            "project": {"type": "string"},
            "deep":    {"type": "boolean", "default": False},
            "memory_db": {"type": "string", "description": "Path to alternate memory SQLite DB (for testing)."},
        },
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": "Rate a memory after using it. helpful=+0.05 trust, unhelpful=-0.10 trust.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string"},
            "action":    {"type": "string", "enum": ["helpful", "unhelpful"]},
        },
        "required": ["memory_id", "action"],
    },
}


# ── Registry ─────────────────────────────────────────────────────────────────

ALL_SCHEMAS = [
    MEMORY_QUERY_SCHEMA,
    MEMORY_WRITE_SCHEMA,
    MEMORY_UPDATE_SCHEMA,
    MEMORY_GET_SOURCE_SCHEMA,
    MEMORY_RECENT_CONTEXT_SCHEMA,
    MEMORY_DREAM_NOW_SCHEMA,
    FACT_FEEDBACK_SCHEMA,
]


def get_hermes_local_tool_schemas() -> list[dict[str, Any]]:
    """Return all 7 tool schemas for registration with the Hermes agent."""
    return ALL_SCHEMAS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_tags(tags_str: str | None) -> list[str] | None:
    """Parse comma-separated tags string into a list of tag strings."""
    if tags_str is None:
        return None
    result = [t.strip() for t in tags_str.split(",") if t.strip()]
    return result if result else None


def _tags_as_json(tags: list[str] | None) -> str | None:
    """Convert a list of tags to a JSON string for DB storage."""
    if tags is None:
        return None
    import json
    return json.dumps(tags)


def _format_fact(row: dict) -> dict:
    return {
        "id":         row.get("fact_id"),
        "content":   row.get("fact_text"),
        "project":    row.get("project"),
        "entity":     row.get("entity"),
        "category":   row.get("category"),
        "trust":      row.get("trust_score"),
        "status":     row.get("status"),
        "tags":       row.get("tags_json"),
        "source_refs": row.get("source_refs_json"),
    }


def _format_decision(row: dict) -> dict:
    return {
        "id":         row.get("decision_id"),
        "content":   row.get("decision_text"),
        "project":    row.get("project"),
        "owner":      row.get("owner"),
        "status":     row.get("status"),
        "source_refs": row.get("source_refs_json"),
    }


def _format_question(row: dict) -> dict:
    return {
        "id":          row.get("question_id"),
        "content":     row.get("question_text"),
        "project":     row.get("project"),
        "priority":    row.get("priority"),
        "status":      row.get("status"),
        "source_refs": row.get("source_refs_json"),
    }


def _format_session(row: dict) -> dict:
    return {
        "id":         row.get("session_id"),
        "project":    row.get("project"),
        "title":      row.get("title"),
        "started_at": row.get("started_at"),
        "ended_at":   row.get("ended_at"),
    }


# ── Tool Handlers ─────────────────────────────────────────────────────────────

def _handle_memory_dream_now(params: dict[str, Any]) -> dict[str, Any]:
    """Trigger a dream run and return the report path + summary."""
    from hermes_memory_core.dream import DreamWorker

    worker = DreamWorker()
    memory_db = params.get("memory_db")
    if memory_db:
        from hermes_memory_core.store.sqlite import MemoryStore
        worker = DreamWorker(store=MemoryStore(db_path=memory_db))
    scope = params.get("scope", "since_last")
    date = params.get("date")
    project = params.get("project")
    deep = params.get("deep", False)

    report_path = worker.run(scope=scope, date=date, project=project, deep=deep)

    return {
        "status":     "ok",
        "report_path": report_path,
        "scope":      scope,
        "message":    f"Dream run complete. Report: {report_path}",
    }


def _handle_memory_query(params: dict[str, Any]) -> dict[str, Any]:
    """Keyword / mode-based search over facts, decisions, and open questions."""
    from hermes_memory_core.store import get_memory_store

    store = get_memory_store()
    query  = params.get("query", "")
    mode   = params.get("mode", "hybrid")
    project = params.get("project")
    entity  = params.get("entity")
    limit  = params.get("limit", 10)

    if mode == "facts":
        rows = store.get_facts(project=project, entity=entity, limit=limit)
        return {"mode": "facts", "results": [_format_fact(r) for r in rows]}

    if mode == "decisions":
        rows = store.get_decisions(project=project, limit=limit)
        return {"mode": "decisions", "results": [_format_decision(r) for r in rows]}

    if mode == "open_questions":
        rows = store.get_open_questions(project=project, limit=limit)
        return {"mode": "open_questions", "results": [_format_question(r) for r in rows]}

    if mode == "sessions":
        rows = store.get_recent_sessions(limit=limit)
        return {"mode": "sessions", "results": [_format_session(r) for r in rows]}

    # Keyword fallback — simple substring scan
    if mode in ("keyword", "hybrid"):
        facts     = store.get_facts(project=project, entity=entity, limit=limit * 2)
        decisions = store.get_decisions(project=project, limit=limit)
        q_lower   = query.lower()
        matched_facts = [_format_fact(r) for r in facts
                         if q_lower in (r.get("fact_text") or "").lower()]
        matched_decisions = [_format_decision(r) for r in decisions
                             if q_lower in (r.get("decision_text") or "").lower()]
        combined = matched_facts + matched_decisions
        return {"mode": mode, "results": combined[:limit]}

    # Default fallback
    rows = store.get_facts(project=project, limit=limit)
    return {"mode": mode, "results": [_format_fact(r) for r in rows]}


def _handle_memory_write(params: dict[str, Any]) -> dict[str, Any]:
    """Write a fact, decision, or open_question through the canonical pipeline."""
    from hermes_memory_core.write.pipeline import write_memory
    from hermes_memory_core.write.redaction import redact

    text = params.get("text", "")
    # Redact before storing
    redacted = redact(text)
    tags = _parse_tags(params.get("tags"))  # list[str] | None — what write_memory wants

    result = write_memory(
        memory_type  = params.get("type", "fact"),
        text         = redacted.redacted_text,
        scope        = params.get("scope", "general"),
        project      = params.get("project"),
        source_ref   = params.get("source_ref", "manual"),
        confidence   = params.get("confidence"),
        tags         = tags,
        rationale    = params.get("rationale"),
        owner        = params.get("owner"),
        priority     = params.get("priority"),
        skip_redaction=True,  # we already redacted above
    )

    return {
        "status":     "ok",
        "id":         result.get("id"),
        "type":       params.get("type"),
        "text":       redacted.redacted_text,
        "redacted":   bool(redacted.types_found),
        "redaction_types": redacted.types_found,
        "message":    f"Written: {params.get('type')} — {redacted.redacted_text[:80]}",
    }


def _handle_memory_update(params: dict[str, Any]) -> dict[str, Any]:
    """Update an existing memory entry."""
    from hermes_memory_core.store import get_memory_store
    from hermes_memory_core.write.redaction import redact

    store = get_memory_store()
    memory_id = params.get("memory_id")

    content = params.get("text")
    if content:
        content = redact(content).redacted_text

    tags_str = _parse_tags(params.get("tags"))  # comma-joined string for DB

    store.update_memory(
        memory_id,
        content     = content,
        trust_delta = params.get("trust_delta"),
        tags        = tags_str,
        status      = params.get("status"),
        category    = params.get("category"),
    )

    return {"status": "ok", "memory_id": memory_id}


def _handle_memory_get_source(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve a source_ref to the original turn or dream run."""
    from hermes_memory_core.store import get_memory_store

    store = get_memory_store()
    source_ref = params.get("source_ref", "")

    if source_ref.startswith("turn:"):
        turn_id = source_ref.split(":", 1)[1]
        turn = store.get_turn(turn_id)
        if turn:
            return {
                "source_ref": source_ref,
                "type":       "turn",
                "content":    turn.get("content"),
                "role":       turn.get("role"),
                "project":    turn.get("project"),
            }

    if source_ref.startswith("dream:"):
        run_id = source_ref.split(":", 1)[1]
        run = store.get_dream_run(run_id)
        if run:
            return {
                "source_ref":   source_ref,
                "type":         "dream_run",
                "report_path":  run.get("output_path"),
                "scope":        run.get("input_scope_json"),
                "status":       run.get("status"),
            }

    return {"source_ref": source_ref, "error": "not found"}


def _handle_memory_recent_context(params: dict[str, Any]) -> dict[str, Any]:
    """Build a compact recent-context summary for session start."""
    from hermes_memory_core.store import get_memory_store

    store = get_memory_store()
    project  = params.get("project")
    max_chars = params.get("max_chars", 4000)

    facts     = store.get_facts(project=project, limit=20)
    decisions  = store.get_decisions(project=project, limit=10)
    questions  = store.get_open_questions(project=project, limit=10)

    lines = []
    if facts:
        lines.append("## Facts")
        for f in facts[:10]:
            lines.append(f"- {f.get('fact_text', '')}")
    if decisions:
        lines.append("## Decisions")
        for d in decisions[:5]:
            lines.append(f"- {d.get('decision_text', '')}")
    if questions:
        lines.append("## Open Questions")
        for q in questions[:5]:
            lines.append(f"- {q.get('question_text', '')}")

    context = "\n".join(lines)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n[truncated]"

    return {"context": context, "chars": len(context)}


def _handle_fact_feedback(params: dict[str, Any]) -> dict[str, Any]:
    """Rate a memory entry and adjust its trust score."""
    from hermes_memory_core.store import get_memory_store

    store = get_memory_store()
    memory_id = params.get("memory_id")
    action = params.get("action")

    delta = 0.05 if action == "helpful" else -0.10
    store.update_memory(memory_id, trust_delta=delta)

    return {
        "status":    "ok",
        "memory_id": memory_id,
        "action":    action,
        "trust_delta": delta,
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

_TOOL_HANDLERS = {
    "memory_query":           _handle_memory_query,
    "memory_write":          _handle_memory_write,
    "memory_update":         _handle_memory_update,
    "memory_get_source":     _handle_memory_get_source,
    "memory_recent_context": _handle_memory_recent_context,
    "memory_dream_now":      _handle_memory_dream_now,
    "fact_feedback":         _handle_fact_feedback,
}


def handle_hermes_local_tool_call(
    tool_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Dispatch a tool call to the appropriate handler.
    Returns a dict result; the gateway serialises it as the tool response.
    """
    handler = _TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return handler(params)
    except Exception as exc:
        return {"error": f"{tool_name} failed: {exc}", "detail": str(exc)}
