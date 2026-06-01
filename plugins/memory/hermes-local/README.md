# Hermes Local Memory Plugin

**Phase:** 1 (Foundation) — Story T-001

This plugin provides the `hermes-local` memory provider for Hermes Agent.
It captures every turn losslessly, indexes it for hybrid retrieval,
and runs a nightly dreamer to extract facts, decisions, and open questions.

## Status

- **Phase 1**: Scaffold only — plugin structure and config wiring established.
- **Phase 2**: Tool schemas (7 tools: query, write, browse, facts, decisions, questions, recent_context).
- **Phase 3**: Semantic search via Qdrant + LMS embeddings.
- **Phase 4**: Hybrid retrieval + prefetch.
- **Phase 5**: Narrative thread fix + dreamer v1.
- **Phase 6**: Migration from holographic + hardening.

## Activation

```yaml
# ~/.hermes/config.yaml
memory:
  provider: hermes-local

plugins:
  hermes-local-memory:
    base_path: "$HERMES_HOME/memory"
    sqlite_path: "$HERMES_HOME/memory/index/memory.sqlite"
    redaction:
      enabled: true
    embedding:
      provider: "openai-compatible"
      endpoint: "http://192.168.2.105:1235/v1"
      model: "text-embedding-nomic-embed-text-v1.5"
      dimension: 768
```

## Architecture

```
hermes-agent (in-process)
  └── plugins/memory/hermes-local/
        ├── __init__.py     — HermesLocalProvider (MemoryProvider ABC)
        ├── narrative.py    — Narrative thread (Phase 5)
        ├── tools.py        — 7 tool schemas + dispatch (Phase 2)
        └── prefetch.py     — Background hybrid prefetch (Phase 4)

hermes_memory_core/ (shared library — imported by both plugin and gateway)
  ├── store/           — SQLite + Qdrant + filesystem
  ├── search/          — Hybrid scorer + HRR
  ├── write/           — Pipeline + redaction
  ├── source.py        — Source ref resolver
  ├── embed.py         — LMS client
  ├── chunk.py         — Chunker
  └── dream/           — Dreamer worker + prompts
```

## Data Directory

```
~/.hermes/memory/
├── raw/              — JSONL (session_id/YYYY/YYYY-MM-DD/{session_id}.jsonl)
├── qmd/              — QMD exports per session
├── daily/            — Daily digest markdown
├── projects/         — Project-level memory
├── entities/         — Entity KB
├── dreams/           — Dreamer output
├── prompts/          — Dreamer prompt templates
├── exports/          — Exported snapshots
├── backups/          — Periodic backups
└── index/            — SQLite WAL + FTS5
    └── memory.sqlite
```

## Plugin Contract

- **Entry point**: `register(ctx)` in `__init__.py`
- **Activation gate**: `is_available()` returns True when `memory.provider == 'hermes-local'`
- **Config namespace**: `plugins.hermes-local-memory` in `config.yaml`
- **Schema independence**: plugin never touches `memory_store.db` or `hermes_state.db` directly

## Key Technical Decisions

- ADR-002: Plugin owns `capture.*` namespace in SQLite only
- ADR-003: JSONL as canonical raw event store; SQLite as secondary index
- Phase 1 redaction runs BEFORE any write (AWS keys, tokens, SSN, card numbers)
- Narrative thread injected via `on_session_switch` (Phase 5)

## Reviewer Notes

- This story does NOT need `kanban_block(review-required)` — scaffolding is low risk.
- Tool schemas remain empty until Phase 2.
- Capture pipeline (`sync_turn`) is stubbed until Phase 1.3.3.