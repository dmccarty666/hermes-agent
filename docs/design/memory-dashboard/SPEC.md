# MemoryPage — Design Specification

**Status:** Draft for review
**Author:** Hermes Agent (subagent task)
**Audience:** David (review/approve), implementing subagents (after approval)
**Sibling docs:** [WIREFRAMES.md](./WIREFRAMES.md) · [API.md](./API.md) · [MILESTONES.md](./MILESTONES.md)

---

## 0. TL;DR

We add a new top-level dashboard page at `/memory` that surfaces every signal the
`hermes-local` memory provider emits today (SQLite, Qdrant, LMS embeddings,
Dreamer LLM, captured turns, dream runs, knowledge facts) and gives the operator
a small set of privileged controls (trigger a dream, force a backup, refresh
metrics, run an interactive search). It pattern-matches `PluginsPage.tsx`: a
single page composed of `<Card>` blocks, server-side state fetched through
`api.*` wrappers in `web/src/lib/api.ts`, and a sticky "Refresh" action injected
into the page header via `usePageHeader().setEnd(...)`.

The page is **operator-facing**, not end-user facing. The reader is the person
who runs the gateway, owns the LMS box, and gets paged when the dreamer
silently stops emitting facts. The page must answer five questions in under
five seconds without scrolling:

1.  Is the memory subsystem **alive right now**?
2.  Are all four backends (**SQLite, Qdrant, LMS, Dreamer**) reachable?
3.  When did the dreamer **last run**, and did it **succeed**?
4.  Are turns **flowing in** (capture rate ≠ 0)?
5.  Is anything **stuck** (chunks pending, redaction failures, error rate ↑)?

Everything else (entity graph, search playground, full activity feed) is below
the fold and is allowed to be slower.

---

## 1. Overview

### 1.1 The problem

`hermes-local` is a multi-process pipeline that touches a SQLite WAL, a Qdrant
vector store, an LMS embedding endpoint on a separate physical machine, a
Qwen3.6-35B "dreamer" LLM on that same machine, and a `systemd` timer that runs
the dreamer on a schedule. When *any one* of these silently breaks — Qdrant
restarts and forgets a collection, LMS reloads the wrong model, the dreamer
times out, the SQLite WAL grows unboundedly — the symptom the operator sees is
**not** an error. The symptom is "Hermes answers feel dumber today." By the
time the operator notices, dozens of turns have been captured but never
indexed, and they will not be recovered without a full re-dream.

The CLI tools we have today (`hermes memory health`, `hermes memory ls-sessions`,
`hermes memory db init`) are perfectly good *answers* — but they require the
operator to *ask the question*. Three months from now, when David hasn't looked
at memory internals in weeks, the question won't get asked, and the failure
will compound.

A dashboard reverses the polarity: the page answers all five questions above
*before* the operator asks. The CLI tools remain authoritative. The dashboard
is the always-on glance that tells you whether you need to drop into the CLI.

### 1.2 Who uses this

| Audience | Frequency | Primary use |
|---|---|---|
| **David (CTO / system owner)** | Daily, briefly | "Is memory healthy?" then close the tab |
| **David (incident mode)** | Hourly during a regression | Read the activity feed, trigger a dream, view last report |
| **Future operator** | First-time discovery | Understand what `hermes-local` is, what it touches, what's running |
| **Subagent (read-only)** | Programmatic | Hit `/api/memory/status` to gate test runs |

Note what's **not** on this list: end users (chat consumers), product
demos, and "Claude reading the page to learn about the system." If a feature
would only matter to those audiences, it does not belong here — it belongs in
`/docs` or in the `/sessions` page.

### 1.3 Why a dashboard is the right shape

Three reasons:

1.  **Multi-source rollup.** No single CLI command tells you "is everything
    OK?". The answer requires combining SQLite + Qdrant + LMS + LLM + filesystem
    state. A dashboard is precisely the affordance for "show me N independent
    sources, color-coded for status, on one screen."
2.  **Existing infrastructure.** `web/src/pages/PluginsPage.tsx` already
    demonstrates the exact pattern we need (card layout, server-fetched state,
    refresh button in header, action buttons with confirm dialogs, toast
    notifications, i18n keys). We can ship M1 in a single sitting by mirroring
    it.
3.  **Operator habit.** The `/plugins`, `/sessions`, `/logs`, `/analytics`,
    `/cron` pages are already where David goes when something feels off. Adding
    `/memory` in the same nav makes it discoverable without a learning curve.

### 1.4 What this is *not*

-   **Not** a replacement for `hermes memory health`. The CLI remains the
    authoritative diagnostic. The page calls the same underlying functions.
-   **Not** a knowledge-graph editor. Facts shown on this page are read-only in
    M1–M3. Editing/deleting facts is deferred to M4 (and arguably never — the
    dreamer should be the editor, not the human).
-   **Not** a chat surface. There is no "chat with your memory" widget. The
    `/chat` page already exists for that.
-   **Not** a multi-tenant control plane. Every counter, every backend, every
    button operates on the single `~/.hermes/memory` tree of the operator
    running the gateway. We do not show another user's facts.

---

## 2. Information Architecture

The page has seven information tiers, ordered top-to-bottom by decreasing
urgency-to-glance.

### Tier 1 — At-a-glance health banner (sticky, top)

One row. Renders as a thin colored bar across the full content width.

-   **Provider name** (e.g. `hermes-local`) + **active?** badge
-   **Overall status pill**: `OK` (green), `DEGRADED` (amber), `ERROR` (red),
    `INACTIVE` (gray — hermes-local is installed but not the active provider)
-   **Last refreshed timestamp** (e.g. "2s ago")
-   **Manual refresh button** (icon-only)

Refresh cadence: auto-poll every **15 seconds**, plus on-demand. Cheap call
(`/api/memory/status` — see §4).

### Tier 2 — Live counters strip

Four to six "stat tiles" in a horizontal row (wraps on narrow viewports).
Each tile shows: a big number, a small label, and a Δ (delta vs. 24h ago, where
applicable). We use `@nous-research/ui`'s `Stats` component the same way
`AnalyticsPage.tsx` does today.

| Tile | Value | Δ shown | Source |
|---|---|---|---|
| Facts (active) | int | yes — vs. 24h | `metrics.facts_active` |
| Turns captured | int (24h) | n/a (already a rate) | `metrics.captured_turns_24h` |
| Chunks indexed | int (24h) | n/a | `metrics.chunks_indexed_24h` |
| Chunks pending | int | yes — alert if >100 | `metrics.chunks_pending` |
| Qdrant points | int | yes — vs. 24h | `metrics.qdrant_points` |
| Last dream | relative time | n/a | `metrics.last_dream_run_at` |

Refresh cadence: **60 seconds** auto-poll. These numbers are not safety-critical
and the underlying `metrics.json` is only updated when the writer runs.

### Tier 3 — Backend status grid

A 2×2 grid (collapses to 1 column on narrow). Each cell is a small `<Card>`
showing one backend.

For each backend: name, status pill, endpoint/path, model or version string,
last-success timestamp, action buttons (where appropriate).

| Backend | Fields | Actions |
|---|---|---|
| **SQLite** | path, journal_mode, size_MB, status | "View tables" (opens a small modal — M2) |
| **Qdrant** | URL, collection count, points total | "Re-init collections" (M2, gated) |
| **LMS embedding** | endpoint, model, dim | "Ping now" (re-runs health probe) |
| **Dreamer LLM** | endpoint, model, last latency | "Ping now" |

Refresh cadence: **30 seconds** auto-poll, plus the per-card "Ping now" button.
Each per-card refresh is independent — pinging the Dreamer does not re-hit
Qdrant.

### Tier 4 — Dreamer activity panel

One full-width card.

Top half: "Last run" block.
-   `started_at` / `finished_at` / duration
-   Status pill (`completed`, `error`, `running`)
-   Counts: facts extracted, decisions, open_questions, turns_processed
-   "View report" link → opens the `*.md` report in a side drawer (M2)

Bottom half: "Schedule" block.
-   Next run time (from `systemctl list-timers hermes-memory-dream.timer`)
-   Timer enabled? (yes/no)
-   `[Run now]` button (confirm dialog, see §10)
-   Recent runs (last 5, dense table: started_at · duration · status · facts)
-   "See all" link → routes to a sub-view (`/memory/dreamer`) — M2

Refresh cadence: **on page load** + after `[Run now]` completes. Not polled —
runs are infrequent and the SSE/poll cost isn't worth it.

### Tier 5 — Knowledge graph snapshot (M4)

One full-width card. Shows:
-   Top 10 entities by fact count (e.g. `David`: 47, `Hermes`: 32, `Qdrant`: 9)
-   Number of contradictions detected (link to detail view)
-   Number of facts with `trust_score < 0.3` (link to detail view)

This panel is **expensive** (it runs aggregation queries on `facts` + a join to
an `entities` table that may not exist in M1). In M1–M3 we show a placeholder
card: "Knowledge graph view coming in M4 — see [docs](#) for current schema."
This is honest and prevents the page from feeling broken.

### Tier 6 — Search playground (M3)

One full-width card. Shows:
-   A text input ("query")
-   A `<Segmented>` mode picker: `hybrid` · `semantic` · `keyword` · `facts`
-   `limit` and `min_score` inputs (numeric, default 10 / 0.0)
-   "Search" button
-   Results table, including which backend each row came from, the score, the
    matched snippet, and a link to the originating session/turn

This is genuinely useful for the operator who wants to verify "if a user asked
*X*, would the memory return anything?" It is also a great regression-detection
tool: if the same query that returned 8 results yesterday returns 0 today, the
operator can isolate the failure to embedding vs. SQLite vs. ranking.

Refresh cadence: **on submit only** — never polled.

### Tier 7 — Recent activity feed (M3)

One full-width card. A reverse-chronological table of the last 50 memory
events (`memory_write`, `memory_query`, `sync_turn`, `dream_run_started`,
`dream_run_completed`, `redaction_applied`, `backup_taken`).

Each row: timestamp · event_type · source (agent/cli/dreamer) · short summary
(e.g. "wrote 3 chunks, session abc12345") · link if applicable.

Refresh cadence: **SSE if available** (we already have a `@app.websocket("/api/events")`
endpoint in `web_server.py:3586`) — fall back to **10s polling** if SSE is not
viable. If SSE fan-out is too noisy, downgrade to 30s polling and never poll
faster than the user's tab is visible (use `document.visibilityState === "visible"`
as a gate, the way `LogsPage.tsx` does for `autoRefresh`).

---

## 3. Page Layout (textual mockup)

The "default landing" view stacks the tiers vertically. Each tier is a row
inside a single `<div className="flex flex-col gap-8">` (matching the `gap-8`
the providers card uses in `PluginsPage.tsx:157`).

```
┌────────────────────────────────────────────────────────────────────────┐
│ Memory                                              [Refresh]          │  ← page header
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ╔══════════════════════════════════════════════════════════════════╗  │
│  ║  ● OK   hermes-local  ·  active  ·  refreshed 2s ago     [↻]    ║  │  ← Tier 1
│  ╚══════════════════════════════════════════════════════════════════╝  │
│                                                                        │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐             │
│  │ 76   │ │ 12   │ │ 12   │ │  0   │ │1,432 │ │ 3h ago   │             │  ← Tier 2
│  │facts │ │turns │ │chunks│ │pend. │ │qpts  │ │last dream│             │
│  │ ↑3   │ │      │ │      │ │      │ │ ↑15  │ │ ✓ ok     │             │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────────┘             │
│                                                                        │
│  ┌─ Backends ───────────────────────────────────────────────────────┐  │
│  │  ┌─ SQLite ──────────┐  ┌─ Qdrant ─────────────┐                 │  │  ← Tier 3
│  │  │ ● ok              │  │ ● ok                 │                 │  │
│  │  │ ~/.hermes/.../    │  │ http://localhost:6333│                 │  │
│  │  │ wal · 4.2 MB      │  │ 3 collections        │                 │
│  │  │ [view tables]     │  │ [re-init]            │                 │
│  │  └───────────────────┘  └──────────────────────┘                 │
│  │  ┌─ LMS embed ───────┐  ┌─ Dreamer LLM ────────┐                 │
│  │  │ ● ok              │  │ ● ok                 │
│  │  │ 192.168.2.105:1235│  │ 192.168.2.105:1234   │
│  │  │ Qwen3-Embed-8B    │  │ qwen3.6-35b-instruct │
│  │  │ dim=1536          │  │ p50 latency 8.3s     │
│  │  │ [ping]            │  │ [ping]               │
│  │  └───────────────────┘  └──────────────────────┘                 │
│  └─────────────────────────────────────────────────────────────────┘
│
│  ┌─ Dreamer ───────────────────────────────────────────────────────┐
│  │  Last run                                                       │  ← Tier 4
│  │  ● completed · started 3h ago · duration 47s                    │
│  │  facts +5  ·  decisions +0  ·  open_qs +1  ·  turns 12          │
│  │  [view report]                                                  │
│  │                                                                 │
│  │  Next run                                                       │
│  │  hermes-memory-dream.timer · in 21h · enabled                   │
│  │  [Run now ⚠]                                                    │
│  │                                                                 │
│  │  Recent runs                                                    │
│  │  3h ago     47s   ✓ completed   +5 facts                        │
│  │  1d ago     52s   ✓ completed   +12 facts                       │
│  │  2d ago     —     ✗ error       extraction parse                │
│  │  3d ago     38s   ✓ completed   +9 facts                        │
│  │  4d ago     41s   ✓ completed   +7 facts                        │
│  │  [see all →]                                                    │
│  └─────────────────────────────────────────────────────────────────┘
│
│  ┌─ Knowledge graph  (M4) ─────────────────────────────────────────┐
│  │  Coming in M4. See SPEC for current schema.                     │  ← Tier 5
│  └─────────────────────────────────────────────────────────────────┘
│
│  ┌─ Search playground  (M3) ───────────────────────────────────────┐
│  │  query: [_________________________]   [hybrid|semantic|kw|fact] │  ← Tier 6
│  │  limit [10]  min_score [0.0]   [Search]                         │
│  │                                                                 │
│  │  (results table appears here after submit)                      │
│  └─────────────────────────────────────────────────────────────────┘
│
│  ┌─ Recent activity  (M3) ─────────────────────────────────────────┐
│  │  2s ago    memory_query   agent  hybrid, q="qdrant"  → 3 hits   │  ← Tier 7
│  │  14s ago   sync_turn      agent  session abc12345    turn 4     │
│  │  47s ago   memory_write   agent  3 chunks, session abc12345     │
│  │  3h ago    dream_run_done dreamer +5 facts                      │
│  │  ...                                                            │
│  └─────────────────────────────────────────────────────────────────┘
│
└────────────────────────────────────────────────────────────────────────┘
```

A full visual rendering, including the search subview and the dream report
drawer, lives in [WIREFRAMES.md](./WIREFRAMES.md).

---

## 4. Backend API Endpoints

All endpoints live in `hermes_cli/web_server.py`. All are gated by the existing
`auth_middleware` (line 237). All return JSON. All read-only endpoints are
safe to call from anywhere; mutating endpoints additionally pass through the
existing CSRF check (header `X-Hermes-CSRF`).

Detailed schemas are in [API.md](./API.md). Here is the proposed minimal set.

### 4.1 Read-only endpoints

#### `GET /api/memory/status`
**Source:** `hermes_memory_core.health.health_check()`
**Caching:** 5s in-memory TTL (key: provider name).
**Response shape:**
```json
{
  "provider": "hermes-local",
  "active": true,
  "overall": "ok",                  // ok | degraded | error | inactive
  "components": {
    "sqlite":     { "status": "ok", "message": null },
    "qdrant":     { "status": "ok", "message": null },
    "embedding":  { "status": "ok", "message": null },
    "llm":        { "status": "ok", "message": null },
    "disk":       { "status": "ok", "message": null }
  },
  "checked_at": "2026-05-22T14:03:11Z"
}
```
This is the cheapest endpoint and feeds Tier 1. Polled at 15s.

#### `GET /api/memory/backends`
**Source:** Same `health_check()` plus a per-backend "last successful probe"
timestamp stored in process-local memory.
**Caching:** 10s TTL per backend.
**Response shape:** the *full* `RolledUpHealth` from `hermes_memory_core/health.py`
plus a `last_success_at` ISO timestamp per backend. Feeds Tier 3.

#### `GET /api/memory/counters`
**Source:** Read `~/.hermes/memory/metrics.json` directly. If absent, lazily
call `MetricsWriter().update()` once and return the result.
**Caching:** the file *is* the cache — we just stat+read.
**Response shape:**
```json
{
  "facts_total": 76,
  "facts_active": 76,
  "captured_turns_24h": 12,
  "chunks_indexed_24h": 12,
  "chunks_pending": 0,
  "qdrant_points": 1432,
  "last_dream_run_at": "2026-05-22T11:00:00Z",
  "last_dream_status": "completed",
  "redactions_24h": 0,
  "deltas_24h": {                 // computed server-side from a snapshot
    "facts_active": 3,
    "qdrant_points": 15
  },
  "stale_seconds": 124            // how old metrics.json is
}
```
Feeds Tier 2.

#### `GET /api/memory/dreamer/last`
**Source:** SQLite `dream_runs` table — `SELECT * ... ORDER BY started_at DESC LIMIT 1`.
**Response shape:** A single dream run row (see `API.md`). Feeds Tier 4 top half.

#### `GET /api/memory/dreamer/runs?limit=20&offset=0`
**Source:** Same table, paginated.
**Response shape:**
```json
{ "items": [ { "id": "...", "started_at": "...", ... } ], "total": 47 }
```
Feeds Tier 4 bottom half + `/memory/dreamer` subview.

#### `GET /api/memory/dreamer/runs/{run_id}/report`
**Source:** Look up the run's `report_path` (an absolute path under
`~/.hermes/memory/dreams/`). Read the markdown. Return its raw text.
**Response shape:**
```json
{ "run_id": "...", "report_md": "# Dream report ...", "path": "..." }
```
Feeds the "View report" drawer.

#### `GET /api/memory/dreamer/schedule`
**Source:** `systemctl list-timers --no-pager hermes-memory-dream.timer` (parsed)
plus `systemctl is-enabled` and `is-active`.
**Caching:** 30s TTL.
**Response shape:**
```json
{ "timer": "hermes-memory-dream.timer",
  "enabled": true, "active": true,
  "next_run_at": "2026-05-23T11:00:00Z",
  "last_run_at": "2026-05-22T11:00:00Z" }
```
Feeds Tier 4 schedule block.

#### `GET /api/memory/entities?limit=10`  (M4)
**Source:** `SELECT entity, COUNT(*) AS n FROM facts GROUP BY entity ORDER BY n DESC LIMIT ?`
(if `entity` column exists — TBD; see §13 open questions).
**Feeds:** Tier 5.

#### `GET /api/memory/activity?limit=50&since=<iso>`  (M3)
**Source:** SQLite `audit_log` table (or the analytics events stream if we
expose it — see §13). Plus dream_runs joined.
**Caching:** none.
**Feeds:** Tier 7 (with SSE alternative below).

#### `GET /api/memory/metrics-json`
**Source:** Raw passthrough of `~/.hermes/memory/metrics.json`. Useful for
power users and Prometheus-style scrapers. Documented and stable.

### 4.2 Mutating / privileged endpoints

All require the CSRF header and surface a confirm dialog client-side (§10).

#### `POST /api/memory/dreamer/run-now`
**Body:** `{ "since_hours": 24 }` (optional)
**Implementation:** Spawns `systemctl start hermes-memory-dream.service` (the
oneshot service backing the timer). Streams stdout via the existing job
mechanism (`@app.get("/api/actions/{name}/status"` at line 746 — pattern lifted
from `gateway/restart`).
**Response:** `{ "job_id": "...", "started_at": "..." }`

#### `POST /api/memory/backup`
**Implementation:** Runs the bundled backup script (`scripts/hermes_memory_backup.sh`
or equivalent — confirm name with David). Stores artifact under
`~/.hermes/memory/backups/`. Returns the artifact path.
**Response:** `{ "ok": true, "path": "...", "size_bytes": 12345 }`

#### `POST /api/memory/search`
**Body:** `{ "query": "...", "mode": "hybrid|semantic|keyword|facts", "limit": 10, "min_score": 0.0 }`
**Implementation:** Calls the existing `memory.query` provider API directly
(we already do this from the agent). Read-only on the data side, but counted
as "mutating" only because it consumes embedding budget; we do not include
this query in the agent's own history.
**Response:** `{ "hits": [ { "score": 0.81, "source": "qdrant|sqlite_fts|facts", "text": "...", "session_id": "...", "turn_id": "..." } ] }`

#### `POST /api/memory/metrics/refresh`
**Implementation:** `MetricsWriter().update()`. Used by Tier 2's "force
refresh" affordance. Idempotent.
**Response:** the refreshed metrics dict.

#### `POST /api/memory/qdrant/reinit`  (gated; see §10)
**Implementation:** Calls `hermes_memory_core.store.qdrant.init_collections()`
with `force=False` by default; `force=true` requires a typed confirmation.
**Response:** `{ "status": "...", "collections": [...] }`

---

## 5. Frontend Component Tree

```
MemoryPage (web/src/pages/MemoryPage.tsx)
│
├── usePageHeader().setEnd(<RefreshButton />)
│
├── <PluginSlot name="memory:top" />
│
├── <HealthBanner />                  // Tier 1, polled 15s
│   └── uses api.getMemoryStatus()
│
├── <CountersStrip />                 // Tier 2, polled 60s
│   └── uses api.getMemoryCounters()
│
├── <BackendsGrid />                  // Tier 3, polled 30s
│   ├── <BackendCard backend="sqlite" />
│   ├── <BackendCard backend="qdrant" />
│   ├── <BackendCard backend="embedding" />
│   └── <BackendCard backend="llm" />
│       └── each calls api.getMemoryBackends() (shared parent fetch)
│
├── <DreamerPanel />                  // Tier 4, fetch-on-mount + after action
│   ├── <DreamerLastRun />
│   ├── <DreamerSchedule />
│   │   └── <Button onClick=runNow> → <ConfirmDialog />
│   ├── <DreamerRecentRuns />
│   └── <DreamReportDrawer />         // opened by "View report"
│
├── <KnowledgeGraphCard />            // Tier 5 (M4 placeholder until then)
│
├── <SearchPlayground />              // Tier 6 (M3)
│   └── <SearchResultsTable />
│
├── <ActivityFeed />                  // Tier 7 (M3) — SSE if available
│
├── <Toast />
└── <PluginSlot name="memory:bottom" />
```

### State / data fetching

We deliberately do **not** introduce SWR or React Query. `PluginsPage.tsx` uses
plain `useEffect` + `useState` + `useCallback`, and so does `AnalyticsPage`,
`SessionsPage`, `LogsPage`. We mirror that — consistency is more valuable than
a 5% boilerplate win. Each top-level panel owns its own state, its own loading
flag, and its own poll interval. A small `useInterval(fn, ms, enabled)` hook
(borrow from `LogsPage`'s autoRefresh pattern) handles cadence.

Why no SWR? Three reasons: (1) every other page is consistent without it, (2)
the polling cadence is heterogeneous (15s / 30s / 60s / never), and a per-key
TTL is the wrong abstraction; (3) we already have a websocket events bus
(`/api/ws`) we'll lean on for Tier 7, which sidesteps the cache-invalidation
problem entirely.

Why no Zustand? Same reason. No cross-page state to share.

### API client extensions

We add typed wrappers in `web/src/lib/api.ts`:

```
api.getMemoryStatus()       → MemoryStatus
api.getMemoryBackends()     → MemoryBackends
api.getMemoryCounters()     → MemoryCounters
api.getDreamerLast()        → DreamerRun | null
api.getDreamerRuns(opts)    → DreamerRunsPage
api.getDreamerReport(id)    → DreamerReport
api.getDreamerSchedule()    → DreamerSchedule
api.runDreamerNow(opts)     → { job_id, started_at }
api.refreshMemoryMetrics()  → MemoryCounters
api.runMemorySearch(opts)   → SearchResults
api.runMemoryBackup()       → BackupResult
api.getMemoryActivity(opts) → ActivityPage
```

Types live next to `HubAgentPluginRow` and similar.

---

## 6. Navigation Integration

Exactly three changes in `web/src/App.tsx`:

1.  **Import** the page and an icon at the top of the file:
    ```
    import MemoryPage from "./pages/MemoryPage";
    import { Brain } from "lucide-react";    // see icon choice below
    ```

2.  **Register the route** in `BUILTIN_ROUTES_CORE` (around line 108):
    ```
      "/plugins": PluginsPage,
      "/memory":  MemoryPage,        // NEW
      "/profiles": ProfilesPage,
    ```
    Position: right after `/plugins` and before `/profiles`. Rationale: memory
    is an "internals/diagnostics" surface, same neighborhood as plugins.

3.  **Add a nav item** in `BUILTIN_NAV_REST` (around line 131):
    ```
      { path: "/plugins", labelKey: "plugins", label: "Plugins", icon: Puzzle },
      { path: "/memory",  labelKey: "memory",  label: "Memory",  icon: Brain  },
      { path: "/profiles", labelKey: "profiles", label: "Profiles", icon: Users },
    ```

We also add the icon to `ICON_MAP` (around line 165) for plugin-defined tabs.

### Icon choice — `Brain`, not `BrainCircuit`, not `Database`

`lucide-react` offers three plausible icons:

| Icon | Why considered | Why not chosen |
|---|---|---|
| `Database` | Memory *is* stored data | Already used in `ICON_MAP` for "data" generically; would conflict and confuse |
| `BrainCircuit` | Implies "AI-shaped memory" | Stylistically busy; the navbar uses flat, simple icons |
| **`Brain`** | Single concept ("memory"), reads at small sizes, no overlap with existing nav items | ✓ |

Pick: `Brain`. If David disagrees, the change is one line.

### i18n keys

Add to `web/src/i18n/{en,...}/index.ts`:
```
nav.memory:                "Memory"
memoryPage.title:          "Memory"
memoryPage.refresh:        "Refresh"
memoryPage.runDream:       "Run dream now"
memoryPage.confirmRunDream:"Trigger a dreamer run? This will consume LMS/LLM budget."
memoryPage.viewReport:     "View report"
memoryPage.searchPlaceholder:"Search captured memory…"
memoryPage.modeHybrid:     "Hybrid"
memoryPage.modeSemantic:   "Semantic"
memoryPage.modeKeyword:    "Keyword"
memoryPage.modeFacts:      "Facts"
memoryPage.empty.notActive:"hermes-local is installed but not the active provider. Activate it on the Plugins page."
memoryPage.empty.noSessions:"No sessions captured yet. Run a chat or invoke the capture pipeline."
memoryPage.empty.dreamerNeverRan:"The dreamer has never run on this system. Trigger it manually or wait for the timer."
memoryPage.empty.qdrantOffline:"Qdrant is unreachable. Vector search is unavailable; SQLite FTS still works."
memoryPage.empty.embeddingOffline:"LMS embeddings are unreachable. New chunks will queue as 'pending' until LMS returns."
```
Same shape and depth as `t.pluginsPage.*`.

---

## 7. Empty States

Each tier has a defined empty state. None of them should ever render as "0" or
a blank card — the user gets either a meaningful number or a meaningful
*sentence*.

| Condition | Affected tier(s) | What renders |
|---|---|---|
| `hermes-local` installed but not active provider | Whole page | A single full-width informational card: "hermes-local is installed but inactive. Active provider: `<name>`. [Switch to hermes-local on /plugins]". All other tiers are hidden. |
| Memory plugin not installed at all | Whole page | "The hermes-local memory plugin is not installed. [Install via the Plugins page]." with a link. No other tiers render. |
| No sessions captured yet | Tier 2 counters | Counters show 0 with a footnote: "No turns captured in the last 24h. Start chatting or invoke the capture pipeline." Search playground (M3) shows "No data to search yet." |
| Dreamer has never run | Tier 4 last-run block | "The dreamer has never run. Trigger it manually or wait for the timer." No metrics shown above this block. Recent-runs table is hidden. |
| Qdrant offline | Tier 3 Qdrant card | Red status, message field shown verbatim. Tier 2 `qdrant_points` shows last-known with a "stale" badge. Search playground (M3) disables `semantic` and `hybrid` modes with tooltip. |
| LMS offline | Tier 3 LMS card | Red status. Tier 2 `chunks_pending` is highlighted (we expect it to climb). Search playground disables `semantic`/`hybrid`. |
| Dreamer LLM offline | Tier 3 LLM card | Red status. `[Run now]` button disabled with tooltip "Dreamer LLM unreachable." |
| `metrics.json` absent | Tier 2 | Counters show "—" with a tiny "Refresh now" affordance that calls `POST /api/memory/metrics/refresh`. |
| Disk near full (< 1GB) | Tier 1 banner | Banner switches to `DEGRADED` even if all other backends are green. Specific message: "Disk free: 0.7 GB. SQLite and Qdrant will fail soon." |

The principle: never silently show "0". Either we have data and we show it,
or we explain *why* we don't.

---

## 8. Privileged Actions & Confirmations

We mirror `PluginsPage.tsx`'s use of `<ConfirmDialog>` (line 14, used at line
399 for plugin removal).

| Action | Confirm modal? | Copy |
|---|---|---|
| Refresh banner | No | — |
| Refresh counters | No | — |
| Per-backend "Ping" | No | — |
| **Run dream now** | Yes | "Trigger a dreamer run? This will consume LMS embedding and Qwen3.6-35B budget on `192.168.2.105`. Estimated cost: ~30–90s, ~2k tokens." [Cancel] [Run] |
| **Re-init Qdrant (non-force)** | Yes | "Re-create missing Qdrant collections? Existing collections are left intact." [Cancel] [Re-init] |
| **Re-init Qdrant (force)** | Yes, with typed name | "This will DROP and recreate all `hermes_memory_*` collections. All vector data will be lost (SQLite + facts are unaffected). Type `RESET` to confirm." [Cancel] [Force re-init] |
| **Trigger backup** | Yes | "Snapshot SQLite + `metrics.json` to `~/.hermes/memory/backups/`. This may take 10–30s." [Cancel] [Back up] |
| **Wipe-and-reset memory** | NOT in this page | Out of scope. We force the operator to use the CLI for this — too dangerous to expose in a browser. |

Why is `wipe-and-reset` not here? Because (a) any operator capable of wanting
that knows how to `rm -rf` the directory and re-init, (b) putting it in the
dashboard is a footgun for the operator who clicks the wrong button at 2 AM,
and (c) browser confirmations have a worse track record than CLI typing.

---

## 9. Performance Considerations

### 9.1 Polling cadence

| Endpoint | Cadence | Cost | Why |
|---|---|---|---|
| `/api/memory/status` | 15s | ~5–50ms (SQLite ping, cached LMS health) | Cheap; banner must feel live |
| `/api/memory/backends` | 30s | 200–800ms (real LMS+LLM probes) | Costly enough we never poll < 30s |
| `/api/memory/counters` | 60s | ~5ms (read JSON file) | File read is cheap; data only updates after writer runs |
| `/api/memory/dreamer/last` | on-mount | ~5ms | One row, indexed |
| `/api/memory/dreamer/schedule` | on-mount + after run | 50–150ms (`systemctl`) | Fork cost — never poll |
| `/api/memory/activity` | SSE preferred; 10s fallback | small | Cheap if SSE; if polling, only when tab visible |

### 9.2 Things we explicitly will NOT poll

-   **Qdrant points-count.** Each call hits Qdrant's `/collections/{name}` for
    every `hermes_memory_*` collection. On a system with 3 collections, that's
    3 HTTP calls per refresh. We rely on the `metrics.json` snapshot (updated
    by the writer at write/dream time) and only re-poll Qdrant when the user
    explicitly clicks the Qdrant card's "Ping" button.
-   **Embedding endpoint by issuing real embed calls.** We use `/v1/models`,
    which is metadata-only and free. Calling `/v1/embeddings` per probe would
    waste GPU cycles.
-   **Dreamer LLM by issuing chat completions.** Same reasoning. Use `/models`
    and fall back to a `max_tokens=1` ping only on the first probe; cache for
    30s.
-   **SQLite via `SELECT COUNT(*) FROM facts`** every 5s. `chunks` and `turns`
    can be huge over time. The metrics writer already aggregates these on
    write-edges; we trust its output.

### 9.3 Caching

A tiny TTL cache on the server (functools.lru_cache-style, but timed):
-   `/api/memory/status`: 5s TTL
-   `/api/memory/backends`: 10s TTL per backend (independent keys)
-   `/api/memory/dreamer/schedule`: 30s TTL

Per-component caching matters because the LLM/embedding probes have visible
latency. With caching, ten dashboard tabs open simultaneously cost the same
as one.

### 9.4 Visibility-gated polling

Borrow from `LogsPage.tsx`'s `autoRefresh` pattern: when `document.hidden ===
true`, pause all polling. Resume on visibility change. This is critical for
laptops left open overnight.

### 9.5 SSE / websocket fanout

We already have `@app.websocket("/api/events")` (web_server.py:3586) and
`@app.websocket("/api/pub")` (line 3557). For Tier 7, the path of least
resistance is to emit a `memory:event` message on the existing bus when
the memory provider emits an audit event, and have `<ActivityFeed>` filter
client-side. **If David prefers**, we can instead expose `/api/memory/activity/stream`
as a dedicated SSE. Defer this decision to M3.

---

## 10. Risks

These are the three things most likely to bite us. They are also in the
top-of-file summary for the parent agent.

### Risk 1 — `metrics.json` staleness

`MetricsWriter.update()` is called on capture/index/dream edges. If the
provider is idle (no writes, no dreams) for hours, the counters tile shows
hours-old numbers. We surface a `stale_seconds` field in
`/api/memory/counters` and badge the tile when `stale > 600s`. Mitigation: a
lightweight scheduled refresh (every 5 min via cron) is cheap; we propose
adding it. **Open question: is the writer thread-safe for concurrent calls?**

### Risk 2 — `systemctl` shell-out from FastAPI

`/api/memory/dreamer/schedule` and `/api/memory/dreamer/run-now` shell out to
`systemctl`. On non-systemd hosts (macOS, NixOS minimal, containers), these
fail. We handle that gracefully (return `{"enabled": null, "supported": false}`)
and render a "Scheduling not available on this host" message. **No fallback
scheduler is in scope** — the dashboard does not pretend to schedule things
itself; it only reports what `systemd` is doing.

### Risk 3 — Privileged actions over a browser

A `POST /api/memory/dreamer/run-now` initiated from a stale tab on someone
else's laptop can spend real GPU minutes. Mitigations: (a) the dialog clearly
shows the resource cost, (b) actions are gated by the same `_SESSION_TOKEN`
auth as the rest of the dashboard, (c) we rate-limit `run-now` server-side to
one in flight + one queued. If David wants to harden further, a "double
confirm with typed phrase" mode for `force-reinit` is already specified.

### Bonus risk (lower likelihood) — Qdrant collection drift

If Qdrant restarts without persistent storage, the points-count drops to 0 and
the dashboard happily shows that, but it's catastrophic: facts in SQLite
reference vector IDs that no longer exist. Mitigation: cross-check `qdrant_points`
vs. `chunks_with_qdrant_id` from SQLite and warn loudly if they diverge by
>10%. Defer to M2.

---

## 11. Phase / Milestone Plan

See [MILESTONES.md](./MILESTONES.md) for the full breakdown. Summary:

| Milestone | Scope | Cost (subagent-hours) |
|---|---|---|
| **M1 — Read-only status** | Tiers 1, 2, 3 + nav integration + 5 endpoints | ~4–6 |
| **M2 — Dreamer interaction** | Tier 4 (full, including report drawer) + `run-now` + schedule + history view | ~6–8 |
| **M3 — Search & activity** | Tier 6 + Tier 7 + SSE wiring | ~8–10 |
| **M4 — Knowledge graph + advanced** | Tier 5 + per-entity drill-down + qdrant drift detector | ~12–16 |

M1 is the meaningful shippable. M2 is where the page becomes interactive. M3
is where it becomes a power tool. M4 is the long tail.

---

## 12. Decision Points for David

These are the three calls I want before implementation starts.

1.  **Tier 5 (knowledge graph) — defer to M4 as proposed, or pull into M1 as a
    minimal "top 10 entities" tile?** Cost: ~1 extra hour in M1; risk: the
    `entities` table schema may not exist yet (see §13). I lean *defer*.

2.  **Activity feed — SSE on `/api/events` (existing bus) or new dedicated
    `/api/memory/activity/stream`?** Lower risk = piggyback on the existing
    bus. Cleaner contract = new SSE. I lean *piggyback in M3, refactor later
    if noisy*.

3.  **`Run dream now` button — gate behind the typed-confirmation pattern
    (like force-reinit), or is the cost-disclosing dialog enough?** I lean
    *cost-dialog only*; running the dreamer is reversible (it just produces a
    report and updates facts) and ~$0.10 of GPU. Force-reinit is irreversible
    so it gets the typed gate.

---

## 13. Open questions (lower priority)

Captured here so they don't get forgotten:

-   Does `metrics.json` need a `provider` field so we can distinguish
    `hermes-local` metrics from a future provider's metrics? (Currently
    implicit.)
-   Does the `facts` table have an `entity` column today, or do we need to
    derive entities from `subject` / `object` fields? (M4 blocker if the
    schema is missing.)
-   Is `MetricsWriter` thread-safe under concurrent calls from
    `POST /api/memory/metrics/refresh`? (M1 hardening item.)
-   Should `/api/memory/search` budget-cap on max embed calls per minute? (M3
    rate-limit item.)
-   Do we want to expose dream **prompts** alongside dream reports? (Probably
    no — the prompt is templated and uninteresting; the report is the artifact.)
-   What happens to this page when the operator switches `memory.provider` to
    a different plugin (e.g. `holographic`)? Proposed: render the "inactive"
    empty state and link to `/plugins`. Confirmed in §7.

---

## 14. Acceptance criteria for M1

The implementing subagent considers M1 done when:

-   [ ] `/memory` route exists, nav item shows up with Brain icon, deep links work
-   [ ] Tier 1 banner reflects real backend status within 15s of a backend
      going down (manually verified by killing Qdrant)
-   [ ] Tier 2 counters match `cat ~/.hermes/memory/metrics.json` exactly
-   [ ] Tier 3 backends grid shows the correct model/endpoint/version for each
      backend
-   [ ] All five M1 endpoints have unit tests in `tests/test_web_server.py`
-   [ ] The page renders correctly when `hermes-local` is *not* the active
      provider (Tier 1 says "inactive", other tiers hide)
-   [ ] The page renders correctly with Qdrant killed (red Qdrant card, banner
      `DEGRADED`)
-   [ ] No new lint, type, or test regressions
-   [ ] One screenshot attached to the PR matching the wireframe

That's the bar. Anything beyond that bar moves to M2.

---

*End of SPEC.md.*
