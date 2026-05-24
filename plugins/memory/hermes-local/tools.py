"""
Hermes Local Memory — tool schemas and handlers.

Implements the 7 tool interfaces defined in TDD §5.2:
  memory_query, memory_write, memory_update, memory_get_source,
  memory_recent_context, memory_dream_now, fact_feedback
"""

from typing import Any, Dict, List

from tools.approval import get_current_session_key

# ── Tool Schemas ─────────────────────────────────────────────────────────────

MEMORY_QUERY_SCHEMA = {
    "name": "memory_query",
    "description": (
        "Search the local memory. Default mode 'hybrid' fuses FTS keyword + Qdrant semantic + "
        "Jaccard + HRR for the most robust ranked retrieval — recommended for most queries. "
        "Other modes: 'semantic' (embedding-only), 'keyword', 'facts', 'decisions', "
        "'open_questions', 'sessions', 'daily', 'project', 'recent', 'probe', 'related', 'reason'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query":   {"type": "string"},
            "mode":    {
                "type": "string",
                "default": "hybrid",
                "enum": ["hybrid", "semantic", "keyword", "facts", "decisions",
                         "open_questions", "sessions", "daily", "project", "recent",
                         "probe", "related", "reason", "unknown"],
                "description": (
                    "Query mode. 'unknown' surfaces unresolved open questions "
                    "from recent sessions. 'episode' returns episodes for the project."
                ),
            },
            "project": {"type": "string"},
            "entity":  {"type": "string"},
            "entities": {"type": "array", "items": {"type": "string"}},
            "filters": {"type": "object"},
            "limit":   {"type": "integer", "default": 10},
            "min_score": {
                "type": "number",
                "default": 0.30,
                "description": "Minimum cosine similarity score for hybrid/semantic results to be included",
            },
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


def _format_hybrid_hit(hit: Any) -> dict:
    """Best-effort flatten of whatever hermes_memory_core.search.hybrid returns.

    The hybrid module may return dicts with varying shapes (chunk hits, fact
    hits, summary hits). We surface the common fields the model needs to
    ground a response: text content, where it came from, and a score.
    """
    if not isinstance(hit, dict):
        return {"content": str(hit)}
    result = {
        "content":     hit.get("text") or hit.get("content") or hit.get("fact_text") or "",
        "source":      hit.get("source") or hit.get("source_ref") or hit.get("source_refs_json"),
        "score":       hit.get("score") or hit.get("rank"),
        "kind":        hit.get("kind") or hit.get("type"),
        "id":          hit.get("id") or hit.get("fact_id") or hit.get("chunk_id"),
        "project":     hit.get("project"),
        "entity":      hit.get("entity"),
    }
    # MEM-016: surface fact_links adjacency when present
    if hit.get("linked_fact_ids"):
        result["linked_fact_ids"] = hit["linked_fact_ids"]
    return result


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
    query   = params.get("query", "")
    mode    = params.get("mode", "hybrid")
    project = params.get("project")
    entity  = params.get("entity")
    limit   = params.get("limit", 10)
    min_score = params.get("min_score", 0.30)

# Structural modes: facts / decisions / open_questions / sessions ─────────
    if mode in ("facts", "decisions", "open_questions", "sessions"):
        if mode == "facts":
            rows = store.get_facts(project=project, entity=entity, limit=limit)
            formatted = [_format_fact(r) for r in rows]
            # MEM-016: surface related facts via fact_links
            related_facts: List[Dict[str, Any]] = []
            for r in rows:
                fid = r.get("fact_id")
                if fid:
                    for link in store.get_fact_links(fid):
                        other_id = link["fact_id_a"] if link["fact_id_b"] == fid else link["fact_id_b"]
                        linked_row = store.get_fact_by_id(other_id)
                        if linked_row:
                            related_facts.append({
                                "fact_id": other_id,
                                "content": linked_row.get("fact_text", ""),
                                "link_type": link["link_type"],
                            })
            mode_info = {"project": project, "entity": entity, "related_facts": related_facts}
        elif mode == "decisions":
            rows = store.get_decisions(project=project, limit=limit)
            formatted = [_format_decision(r) for r in rows]
            mode_info = {"project": project}
        elif mode == "open_questions":
            rows = store.get_open_questions(project=project, limit=limit)
            formatted = [_format_question(r) for r in rows]
            mode_info = {"project": project}
        else:  # sessions
            rows = store.get_recent_sessions(limit=limit)
            formatted = [_format_session(r) for r in rows]
            mode_info = {}
        # MEM-019: audit fact hits for analytics
        if mode == "facts":
            session_id = get_current_session_key(default="unknown")
            for rank, item in enumerate(formatted, start=1):
                fid = item.get("id")
                if fid:
                    store.audit_hit(session_id=session_id, query=query, mode=mode, fact_id=fid, score=1.0, hit_rank=rank)
        return {
            "results": formatted,
            "mode": mode,
            "total": len(formatted),
            "query": query,
            "mode_info": mode_info,
        }

    # Semantic / hybrid — real embedding-based search via the hybrid module.
    # If hybrid returns empty (no Qdrant points yet, or merge-logic bug), fall
    # back to facts so the model still gets something relevant to ground its
    # response.
    if mode in ("semantic", "hybrid"):
        try:
            from hermes_memory_core.search import hybrid as _hybrid
            raw = _hybrid.search(query, mode=mode, limit=limit, memory_db=store)
            # hybrid.search returns dict {"results": [...], "count": N, ...}
            raw_hits = raw.get("results") if isinstance(raw, dict) else raw
            raw_hits = raw_hits or []

            # Filter out low-confidence vector noise (cosine similarity below
            # min_score). Real hits are typically 0.40+; junk hits 0.20-.
            def _score_of(h):
                try:
                    return float(h.get("score", 0.0)) if isinstance(h, dict) else 0.0
                except (TypeError, ValueError):
                    return 0.0

            filtered_hits = [h for h in raw_hits if _score_of(h) >= min_score]

            if filtered_hits:
                # Surface any open questions related to the project being queried.
                # open_questions table has: question_id, question_text, project,
                # priority, status, source_refs_json, next_action, created_at, updated_at
                open_qs = []
                if project:
                    rows = store.get_open_questions(project=project, limit=20)
                    open_qs = [{"question": r.get("question_text", ""),
                                "asked_at": r.get("created_at", ""),
                                "entity": project}
                               for r in rows]
                mode_info = {
                    "backend_weights": raw.get("backend_weights") if isinstance(raw, dict) else None,
                }
                if open_qs:
                    mode_info["open_questions"] = open_qs
                # MEM-019: audit retrieval hits for analytics
                session_id = get_current_session_key(default="unknown")
                formatted_results = [_format_hybrid_hit(h) for h in filtered_hits]
                for rank, item in enumerate(formatted_results, start=1):
                    fid = item.get("id")
                    if fid:
                        score = item.get("score", 0.0)
                        store.audit_hit(session_id=session_id, query=query, mode=mode, fact_id=fid, score=float(score), hit_rank=rank)
                return {
                    "results": formatted_results,
                    "mode": mode,
                    "total": len(filtered_hits),
                    "query": query,
                    "mode_info": mode_info,
                }

            # If we had raw hits but min_score filtered them all out, that's
            # an "irrelevant query" signal — return empty with a clear note.
            # Do NOT fall back to keyword scan in this case.
            if raw_hits:
                return {
                    "results": [],
                    "mode": mode,
                    "total": 0,
                    "query": query,
                    "mode_info": {
                        "note": f"no results above min_score={min_score:.2f} (lower threshold or rephrase query)",
                    },
                }

            # raw_hits was empty — Qdrant likely has no points yet. Fall back
            # to keyword substring scan on facts so the model still gets
            # something relevant to ground its response.
            facts = store.get_facts(project=project, entity=entity, limit=max(200, limit * 10))
            q_lower = query.lower()
            kw_matched = [_format_fact(r) for r in facts
                          if q_lower in (r.get("fact_text") or "").lower()]
            if kw_matched:
                return {
                    "results": kw_matched[:limit],
                    "mode": mode,
                    "total": len(kw_matched[:limit]),
                    "query": query,
                    "mode_info": {
                        "fallback": "keyword",
                        "note": f"{mode} returned 0 hits (Qdrant points=0?); keyword substring matched {len(kw_matched)}",
                    },
                }
            return {
                "results": [],
                "mode": mode,
                "total": 0,
                "query": query,
                "mode_info": {
                    "fallback": "none",
                    "note": f"{mode} returned 0 hits; keyword found nothing — query likely unrelated to stored memory",
                },
            }
        except Exception as e:
            # Last-resort: keyword substring scan
            facts = store.get_facts(project=project, entity=entity, limit=limit * 2)
            q_lower = query.lower()
            matched = [_format_fact(r) for r in facts
                       if q_lower in (r.get("fact_text") or "").lower()]
            return {
                "results": matched[:limit],
                "mode": mode,
                "total": len(matched[:limit]),
                "query": query,
                "mode_info": {
                    "fallback": "keyword",
                    "error": str(e),
                },
            }

    # Keyword — simple substring scan over facts + decisions
    if mode == "keyword":
        facts     = store.get_facts(project=project, entity=entity, limit=limit * 2)
        decisions = store.get_decisions(project=project, limit=limit)
        q_lower   = query.lower()
        matched_facts = [_format_fact(r) for r in facts
                         if q_lower in (r.get("fact_text") or "").lower()]
        matched_decisions = [_format_decision(r) for r in decisions
                             if q_lower in (r.get("decision_text") or "").lower()]
        combined = matched_facts + matched_decisions
        limited = combined[:limit]
        # MEM-019: audit fact hits for analytics
        session_id = get_current_session_key(default="unknown")
        for rank, item in enumerate(limited, start=1):
            fid = item.get("id")
            if fid:
                store.audit_hit(session_id=session_id, query=query, mode=mode, fact_id=fid, score=1.0, hit_rank=rank)
        return {
            "results": limited,
            "mode": mode,
            "total": len(limited),
            "query": query,
            "mode_info": {"sources": ["facts", "decisions"]},
        }

    # Unknown mode — return recent facts as a safe default
    rows = store.get_facts(project=project, limit=limit)
    formatted = [_format_fact(r) for r in rows]
    # MEM-019: audit fact hits for analytics
    session_id = get_current_session_key(default="unknown")
    for rank, item in enumerate(formatted, start=1):
        fid = item.get("id")
        if fid:
            store.audit_hit(session_id=session_id, query=query, mode=mode, fact_id=fid, score=1.0, hit_rank=rank)
    return {
        "results": formatted,
        "mode": mode,
        "total": len(formatted),
        "query": query,
        "mode_info": {"fallback": "facts"},
    }


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
