# MemoryPage — API Reference (proposed)

**Sibling docs:** [SPEC.md](./SPEC.md) · [WIREFRAMES.md](./WIREFRAMES.md) · [MILESTONES.md](./MILESTONES.md)

This document specifies every new endpoint required by the MemoryPage. All
endpoints live in `hermes_cli/web_server.py`. None of these have been
implemented — this is the contract David is reviewing before code lands.

---

## Conventions

| | |
|---|---|
| **Base path** | `/api/memory/...` |
| **Auth** | Bearer token via the existing `_SESSION_TOKEN` mechanism (`web_server.py:86`, `auth_middleware` at line 237). No new auth surface. |
| **CSRF** | Mutating endpoints (`POST`, `PUT`, `DELETE`) additionally require the existing `X-Hermes-CSRF` header. |
| **Content-Type** | All requests/responses: `application/json`. Reports are returned as raw markdown *inside* JSON for consistency. |
| **Errors** | `400` validation / bad input · `401` no/bad bearer · `403` CSRF mismatch · `404` not found · `409` conflict (e.g. dreamer already running) · `500` unexpected · `503` backend unreachable |
| **Time** | All timestamps ISO-8601 with `Z` suffix. |
| **Pagination** | Uniform: `?limit=N&offset=M`. Response: `{ "items": [...], "total": N, "limit": N, "offset": M }`. |

### Standard error envelope

```json
{ "detail": "Qdrant unreachable: Connection refused",
  "code":   "BACKEND_DOWN",
  "backend": "qdrant" }
```

The `code` field is informal but stable per endpoint — frontend may switch on
it to render targeted messaging.

---

## Endpoint summary

| Method | Path | Milestone | Mutating? |
|---|---|---|---|
| GET    | `/api/memory/status`                    | M1 | n |
| GET    | `/api/memory/backends`                  | M1 | n |
| GET    | `/api/memory/counters`                  | M1 | n |
| GET    | `/api/memory/metrics-json`              | M1 | n |
| POST   | `/api/memory/metrics/refresh`           | M1 | y |
| POST   | `/api/memory/backends/{name}/ping`      | M1 | y* |
| GET    | `/api/memory/dreamer/last`              | M2 | n |
| GET    | `/api/memory/dreamer/runs`              | M2 | n |
| GET    | `/api/memory/dreamer/runs/{id}`         | M2 | n |
| GET    | `/api/memory/dreamer/runs/{id}/report`  | M2 | n |
| GET    | `/api/memory/dreamer/schedule`          | M2 | n |
| POST   | `/api/memory/dreamer/run-now`           | M2 | y |
| POST   | `/api/memory/search`                    | M3 | y* |
| GET    | `/api/memory/activity`                  | M3 | n |
| GET    | `/api/memory/activity/stream` (SSE)     | M3 | n |
| GET    | `/api/memory/entities`                  | M4 | n |
| GET    | `/api/memory/contradictions`            | M4 | n |
| POST   | `/api/memory/backup`                    | M4 | y |
| POST   | `/api/memory/qdrant/reinit`             | M4 | y |

`y*` = technically read-only on user data, but consumes server resources
(LMS/LLM calls), so treated as mutating for rate-limiting and CSRF purposes.

---

## 1. `GET /api/memory/status`  *(M1)*

Cheap rolled-up health for the Tier 1 banner.

**Query params:** none.

**Response 200:**
```json
{
  "provider":  "hermes-local",
  "active":    true,
  "installed": true,
  "overall":   "ok",
  "components": {
    "sqlite":    { "status": "ok",    "message": null },
    "qdrant":    { "status": "ok",    "message": null },
    "embedding": { "status": "ok",    "message": null },
    "llm":       { "status": "ok",    "message": null },
    "disk":      { "status": "ok",    "message": null }
  },
  "checked_at": "2026-05-22T14:03:11Z",
  "cached":     true,
  "cache_ttl":  5
}
```

`overall` ∈ `{ok, degraded, error, inactive}`. `inactive` is returned when
`provider != "hermes-local"`.

**Response 200 — inactive provider:**
```json
{ "provider": "hermes-local", "active": false, "installed": true,
  "overall": "inactive", "components": null, "checked_at": "..." }
```

**Implementation notes:**
- Calls `hermes_memory_core.health.health_check()`.
- 5s TTL on the result.
- Skips per-component probes when `active == false` (returns `components: null`).

---

## 2. `GET /api/memory/backends`  *(M1)*

Detailed per-backend report for Tier 3.

**Query params:**
- `refresh=true|false` (default false) — bypass the cache, re-probe everything.

**Response 200:**
```json
{
  "sqlite": {
    "status": "ok",
    "path":   "/home/d/.hermes/memory/index/memory.sqlite",
    "journal_mode": "wal",
    "size_bytes":   4326400,
    "last_success_at": "2026-05-22T14:03:09Z",
    "message": null
  },
  "qdrant": {
    "status": "ok",
    "endpoint": "http://localhost:6333",
    "collections_count": 3,
    "collections": ["hermes_memory_chunks", "hermes_memory_facts", "hermes_memory_summaries"],
    "points_count": 1432,
    "last_success_at": "2026-05-22T14:02:55Z",
    "message": null
  },
  "embedding": {
    "status": "ok",
    "endpoint": "http://192.168.2.105:1235",
    "model":    "Qwen3-Embed-8B",
    "dim":      1536,
    "last_success_at": "2026-05-22T14:03:08Z",
    "message": null
  },
  "llm": {
    "status": "ok",
    "endpoint": "http://192.168.2.105:1234",
    "model":    "qwen3.6-35b-instruct",
    "last_latency_ms": 8312,
    "last_success_at": "2026-05-22T11:00:47Z",
    "message": null
  },
  "disk": {
    "status": "ok",
    "path":   "/home/d/.hermes",
    "free_bytes": 142307328000,
    "free_gb":    132.5,
    "message": null
  },
  "checked_at": "2026-05-22T14:03:11Z"
}
```

**Error 503:** Returned only when the *server side* couldn't run any of the
checks (extremely unlikely). Per-backend failures show up as `status: "error"`
in the body, not as HTTP 503.

---

## 3. `GET /api/memory/counters`  *(M1)*

The Tier 2 counter strip.

**Query params:**
- `force=true` — synchronous `MetricsWriter().update()` before returning (equivalent to `POST /api/memory/metrics/refresh` + this GET).

**Response 200:**
```json
{
  "facts_total":         76,
  "facts_active":        76,
  "captured_turns_24h":  12,
  "chunks_indexed_24h":  12,
  "chunks_pending":      0,
  "qdrant_points":       1432,
  "last_dream_run_at":   "2026-05-22T11:00:00Z",
  "last_dream_status":   "completed",
  "redactions_24h":      0,
  "deltas_24h": {
    "facts_active":   3,
    "qdrant_points": 15
  },
  "metrics_file": "/home/d/.hermes/memory/metrics.json",
  "stale_seconds": 124
}
```

**Response 404:** Returned when `metrics.json` is absent *and* a synchronous
refresh failed (e.g. SQLite not initialized). Caller should surface "metrics
not yet available — initialize the memory tree."

**Notes:**
- `deltas_24h` is computed by diffing today's snapshot against a 24h-ago
  snapshot stored at `~/.hermes/memory/metrics-snapshots/YYYY-MM-DD.json`.
  If no snapshot exists yet, `deltas_24h` is `null`.
- `stale_seconds` = `now - mtime(metrics.json)`. The UI surfaces a
  "stale" badge when > 600.

---

## 4. `GET /api/memory/metrics-json`  *(M1)*

Raw passthrough of `~/.hermes/memory/metrics.json`. Stable, documented contract.

**Response 200:** the file contents verbatim, parsed and re-serialized.

**Response 404:** if the file doesn't exist.

This is useful for ops tooling and Prometheus-style scrapers that don't want
the dashboard's enriched envelope.

---

## 5. `POST /api/memory/metrics/refresh`  *(M1)*

Forces `MetricsWriter().update()` and returns the result.

**Body:** none (or `{}`).
**Response 200:** same shape as `GET /api/memory/counters` (without
`deltas_24h` — those are computed off snapshots).

**Rate-limit:** server-side; at most 1 call per 5s per session token.
**Idempotent.**

---

## 6. `POST /api/memory/backends/{name}/ping`  *(M1)*

Re-probes one backend in isolation. `{name}` ∈ `{sqlite, qdrant, embedding, llm, disk}`.

**Body:** none.
**Response 200:** same shape as the per-backend block in `/api/memory/backends`.
**Response 404:** unknown backend name.
**Response 503:** backend probe failed (still returns a body with `status:
"error"` — 503 is set so the frontend can color-code without parsing).

---

## 7. `GET /api/memory/dreamer/last`  *(M2)*

The most recent dream run.

**Response 200:**
```json
{
  "id":          "abc1234-5678",
  "started_at":  "2026-05-22T11:00:00Z",
  "finished_at": "2026-05-22T11:00:47Z",
  "duration_s":  47.3,
  "status":      "completed",
  "facts_extracted":      5,
  "decisions_extracted":  0,
  "open_questions_extracted": 1,
  "turns_processed":      12,
  "embed_calls":          4,
  "llm_tokens_in":        8432,
  "llm_tokens_out":       1121,
  "report_path":          "/home/d/.hermes/memory/dreams/2026-05-22T11-00-00.md",
  "report_available":     true,
  "error_message":        null
}
```

**Response 404:** the dreamer has never run.

**Backing query:**
```sql
SELECT * FROM dream_runs ORDER BY started_at DESC LIMIT 1;
```

---

## 8. `GET /api/memory/dreamer/runs`  *(M2)*

Paginated list of all dream runs.

**Query params:**
- `limit=20` (default), max 100
- `offset=0`
- `status=completed|error|running` (optional filter)

**Response 200:**
```json
{
  "items":  [ { "id": "...", "started_at": "...", "status": "...", "facts_extracted": 5, "duration_s": 47.3 }, ... ],
  "total":  47,
  "limit":  20,
  "offset": 0
}
```

Items are abbreviated rows (not the full schema). Full row available via
`GET /api/memory/dreamer/runs/{id}`.

---

## 9. `GET /api/memory/dreamer/runs/{id}`  *(M2)*

Full row for one dream run. Same shape as `/api/memory/dreamer/last`.

**Response 404:** unknown id.

---

## 10. `GET /api/memory/dreamer/runs/{id}/report`  *(M2)*

The markdown report for a run.

**Response 200:**
```json
{
  "run_id":    "abc1234-5678",
  "path":      "/home/d/.hermes/memory/dreams/2026-05-22T11-00-00.md",
  "report_md": "# Dream report — 2026-05-22 11:00\n\n## New facts (5)\n\n- ...",
  "size_bytes": 4129
}
```

**Response 404:** unknown run id, or report file missing on disk.

**Notes:**
- We bound the response at 1 MB. If a report is bigger, return a truncation
  flag and a `path` the operator can `xdg-open` locally.
- Markdown is returned **verbatim** — the frontend handles rendering.

---

## 11. `GET /api/memory/dreamer/schedule`  *(M2)*

Status of the `systemd` timer driving the scheduled dreamer.

**Response 200 (Linux/systemd):**
```json
{
  "supported":   true,
  "timer":       "hermes-memory-dream.timer",
  "service":     "hermes-memory-dream.service",
  "enabled":     true,
  "active":      true,
  "next_run_at": "2026-05-23T11:00:00Z",
  "last_run_at": "2026-05-22T11:00:00Z"
}
```

**Response 200 (unsupported host):**
```json
{ "supported": false, "reason": "systemctl not available on this host" }
```

**Implementation:** shells out to `systemctl list-timers --no-pager hermes-memory-dream.timer`,
`is-enabled`, `is-active`. Cached 30s.

---

## 12. `POST /api/memory/dreamer/run-now`  *(M2)*

Triggers the dreamer immediately. Streams progress through the existing job
system at `/api/actions/{name}/status` (mirroring `gateway/restart`).

**Body (all optional):**
```json
{
  "since_hours": 24,
  "force":       false
}
```
- `since_hours` overrides the default window (24h).
- `force` re-processes turns already marked `dream_status=done`.

**Response 202:**
```json
{
  "job_id":      "dream-run-1716387791",
  "started_at":  "2026-05-22T14:03:11Z",
  "status_url":  "/api/actions/dream-run-1716387791/status"
}
```

**Response 409:** a dream run is already in progress.

**Rate-limit:** at most 1 in-flight + 1 queued per session token.

---

## 13. `POST /api/memory/search`  *(M3)*

Interactive search across the memory backends.

**Body:**
```json
{
  "query":     "qdrant collection recovery",
  "mode":      "hybrid",
  "limit":     10,
  "min_score": 0.0
}
```
`mode` ∈ `{hybrid, semantic, keyword, facts}`.

**Response 200:**
```json
{
  "query":     "qdrant collection recovery",
  "mode":      "hybrid",
  "elapsed_ms": 142,
  "embed_calls": 1,
  "embed_ms":    38,
  "hits": [
    {
      "score":      0.86,
      "source":     "qdrant",
      "text":       "…we lost the hermes_memory_facts collection…",
      "session_id": "abc12345",
      "turn_id":    "abc12345-7",
      "timestamp":  "2026-05-22T11:14:22Z"
    }
  ]
}
```

**Response 400:** empty or > 1000-char query.
**Response 503:** the requested mode requires a backend that's offline (e.g.
`mode=semantic` while Qdrant down).

**Rate-limit:** 6 requests/minute per session token. Each `semantic`/`hybrid`
request consumes one embed call from the LMS budget.

---

## 14. `GET /api/memory/activity`  *(M3)*

Paginated recent activity feed.

**Query params:**
- `limit=50` (default), max 200
- `since=<iso>` (optional)
- `kinds=memory_write,memory_query,sync_turn,dream_run_completed,...` (optional comma-separated filter)

**Response 200:**
```json
{
  "items": [
    {
      "id":         "evt-abc",
      "timestamp":  "2026-05-22T14:03:09Z",
      "kind":       "memory_query",
      "source":     "agent",
      "summary":    "hybrid, q=\"qdrant\" → 3 hits",
      "session_id": "abc12345",
      "details":    { "mode": "hybrid", "hits": 3 }
    }
  ],
  "total": 47
}
```

Backing source: SQLite `audit_log` table joined with `dream_runs` (synthetic
events for `dream_run_started` / `dream_run_completed`).

---

## 15. `GET /api/memory/activity/stream`  *(M3)*

Server-Sent Events stream of new activity. Used by Tier 7 in lieu of polling.

**Headers:** `Accept: text/event-stream`.

**Events:**
```
event: memory_activity
data: { "id": "...", "timestamp": "...", "kind": "...", ... }
```

Heartbeat every 20s as `event: ping`.

**Alternative:** if SSE proves awkward, piggyback on the existing
`@app.websocket("/api/events")` bus with a `memory:event` channel filter.
The choice is M3-time; both shapes are sketched in the SPEC §9.5.

---

## 16. `GET /api/memory/entities`  *(M4)*

Top entities by fact count.

**Query params:**
- `limit=10` (default), max 100
- `sort=count|trust|recency` (default count)

**Response 200:**
```json
{
  "items": [
    { "entity": "David",  "fact_count": 47, "avg_trust": 0.84, "last_updated": "..." },
    { "entity": "Hermes", "fact_count": 32, "avg_trust": 0.78, "last_updated": "..." }
  ],
  "total": 18
}
```

**Backing query (subject to schema confirmation, see SPEC §13):**
```sql
SELECT entity, COUNT(*) AS fact_count, AVG(trust_score) AS avg_trust,
       MAX(updated_at) AS last_updated
  FROM facts
  WHERE status = 'active'
  GROUP BY entity
  ORDER BY fact_count DESC
  LIMIT ?;
```

---

## 17. `GET /api/memory/contradictions`  *(M4)*

Facts flagged as contradictory by the dreamer.

**Response 200:**
```json
{
  "items": [
    {
      "fact_a": { "id": "f-001", "text": "...", "trust": 0.9, "source_turn": "..." },
      "fact_b": { "id": "f-042", "text": "...", "trust": 0.6, "source_turn": "..." },
      "detected_in_run": "abc1234-5678",
      "detected_at":     "2026-05-22T11:00:30Z"
    }
  ],
  "total": 2
}
```

---

## 18. `POST /api/memory/backup`  *(M4)*

Snapshot SQLite + `metrics.json` to `~/.hermes/memory/backups/`.

**Body:** none, or `{ "label": "pre-upgrade" }`.

**Response 200:**
```json
{
  "ok":        true,
  "path":      "/home/d/.hermes/memory/backups/2026-05-22T14-03-11_pre-upgrade.tar.zst",
  "size_bytes": 4623881,
  "duration_s": 12.4
}
```

**Response 503:** backup script not found or failed; `detail` carries stderr.

**Notes:** the script itself does not exist yet — provisioning a
`scripts/hermes_memory_backup.sh` is part of M4. We use it via subprocess.

---

## 19. `POST /api/memory/qdrant/reinit`  *(M4)*

Re-create missing or all `hermes_memory_*` Qdrant collections.

**Body:**
```json
{ "force": false, "confirm": "RESET" }
```
- `force=false`: create only missing collections. `confirm` not required.
- `force=true`: drop and recreate ALL `hermes_memory_*` collections.
  `confirm` must equal the literal string `"RESET"` (typed by the operator
  in the UI). Server rejects 400 otherwise.

**Response 200:**
```json
{
  "status":      "created",
  "collections": ["hermes_memory_chunks", "hermes_memory_facts"],
  "errors":      []
}
```

**Response 400:** missing/incorrect `confirm` when `force=true`.

**Backing call:** `hermes_memory_core.store.qdrant.init_collections(force=...)`.

---

## Code & shell command map

This table maps each proposed endpoint to the existing primitive it should
wrap. Implementing subagents can pattern-match without inventing new logic.

| Endpoint | Wraps |
|---|---|
| `GET /status`              | `hermes_memory_core.health.health_check()` |
| `GET /backends`            | Same, with richer response shape |
| `GET /counters`            | Read `~/.hermes/memory/metrics.json` |
| `GET /metrics-json`        | Passthrough of `metrics.json` |
| `POST /metrics/refresh`    | `MetricsWriter().update()` |
| `POST /backends/{n}/ping`  | Per-backend functions in `health.py` |
| `GET /dreamer/last`        | `SELECT ... FROM dream_runs LIMIT 1` |
| `GET /dreamer/runs`        | Same, paginated |
| `GET /dreamer/runs/{id}`   | `SELECT ... FROM dream_runs WHERE id = ?` |
| `GET /dreamer/runs/{id}/report` | Read file from `report_path` column |
| `GET /dreamer/schedule`    | `systemctl list-timers ...` shell-out |
| `POST /dreamer/run-now`    | `systemctl start hermes-memory-dream.service` + job tracking |
| `POST /search`             | Direct call into the active memory provider's `query()` |
| `GET /activity`            | `SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?` |
| `GET /activity/stream`     | Bus filter on existing `/api/events` ws, or new SSE |
| `GET /entities`            | `SELECT entity, COUNT(*) ... GROUP BY entity` (M4) |
| `GET /contradictions`      | New view or column on `facts`; TBD M4 |
| `POST /backup`             | Subprocess of `scripts/hermes_memory_backup.sh` |
| `POST /qdrant/reinit`      | `hermes_memory_core.store.qdrant.init_collections()` |

---

## Example: end-to-end "everything green" rollup

`GET /api/memory/status` (Tier 1):
```json
{ "provider": "hermes-local", "active": true, "overall": "ok",
  "components": { "sqlite": {"status":"ok","message":null},
                  "qdrant": {"status":"ok","message":null},
                  "embedding": {"status":"ok","message":null},
                  "llm": {"status":"ok","message":null},
                  "disk": {"status":"ok","message":null} },
  "checked_at": "2026-05-22T14:03:11Z", "cached": false, "cache_ttl": 5 }
```

`GET /api/memory/counters` (Tier 2):
```json
{ "facts_total": 76, "facts_active": 76, "captured_turns_24h": 12,
  "chunks_indexed_24h": 12, "chunks_pending": 0, "qdrant_points": 1432,
  "last_dream_run_at": "2026-05-22T11:00:00Z", "last_dream_status": "completed",
  "redactions_24h": 0, "deltas_24h": { "facts_active": 3, "qdrant_points": 15 },
  "metrics_file": "/home/d/.hermes/memory/metrics.json", "stale_seconds": 124 }
```

That's everything M1 needs on the wire to render the landing view.

---

## Example: Qdrant offline scenario

`GET /api/memory/status`:
```json
{ "provider": "hermes-local", "active": true, "overall": "degraded",
  "components": {
    "sqlite":    { "status": "ok",    "message": null },
    "qdrant":    { "status": "error", "message": "Connection refused" },
    "embedding": { "status": "ok",    "message": null },
    "llm":       { "status": "ok",    "message": null },
    "disk":      { "status": "ok",    "message": null } },
  "checked_at": "2026-05-22T14:03:11Z", "cached": false, "cache_ttl": 5 }
```

`POST /api/memory/search` with `mode=hybrid`:
```
503 Service Unavailable
{ "detail": "Hybrid search requires Qdrant which is unreachable. Retry with mode=keyword.",
  "code":   "BACKEND_DOWN",
  "backend": "qdrant" }
```

The frontend translates that `code` into a tooltip on the disabled mode pill.

---

*End of API.md.*
