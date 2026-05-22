# MemoryPage — Milestone Plan

**Sibling docs:** [SPEC.md](./SPEC.md) · [API.md](./API.md) · [WIREFRAMES.md](./WIREFRAMES.md)

This document breaks the MemoryPage build into four shippable milestones plus
a "Phase 0" review/decision gate that must close before M1 starts. Each
milestone is independently mergeable, independently shippable, and ends in a
testable exit criterion. No milestone depends on a milestone after it.

Hour estimates are honest **subagent-hours** — they assume a subagent that
already knows the codebase (has read `web/src/pages/PluginsPage.tsx`,
`hermes_cli/web_server.py`, and the SPEC). They do not include design review,
PR cycles, or David's read time.

Complexity baseline: `PluginsPage.tsx` is 583 lines, `AnalyticsPage.tsx` is
599 lines. A fresh, comparable page lands in ~6 focused hours from a cold
start. We use that as our calibration.

---

## Phase 0 — Review & decision gate (before M1)

Before any code lands, David needs to make three calls and the open SPEC
questions need to be resolved or explicitly deferred. This phase is hours of
**reading**, not coding.

### 0.1 Reading list (~45 min total)

In order:

1. `docs/design/memory-dashboard/SPEC.md` — start here; sets terminology.
2. `docs/design/memory-dashboard/API.md` — the contract being approved.
3. `docs/design/memory-dashboard/WIREFRAMES.md` — confirms the SPEC reads as a real page.
4. `docs/design/memory-dashboard/MILESTONES.md` (this file) — execution plan.

### 0.2 Decisions David owns

Three from SPEC §12, repeated here so they can't slip:

| # | Decision | SPEC default | David's call |
|---|---|---|---|
| D1 | Tier 5 (knowledge graph) — defer to M4 as proposed, or pull a "top 10 entities" tile into M1? | Defer | ☐ defer / ☐ pull |
| D2 | Activity feed transport — piggyback `/api/events` ws bus, or new dedicated `/api/memory/activity/stream` SSE? | Piggyback | ☐ piggyback / ☐ new SSE |
| D3 | `[Run now ⚠]` button gating — cost-disclosing dialog only, or also typed-confirm? | Dialog only | ☐ dialog / ☐ typed |

Two more flagged from SPEC §13 that block specific milestones:

| # | Question | Blocks | Resolution path |
|---|---|---|---|
| D4 | Does `facts` schema have an `entity` column, or do we derive from `subject`/`object`? | M4 endpoints `/entities`, `/contradictions` | Inspect schema; one line of SQL: `PRAGMA table_info(facts);` |
| D5 | Is `MetricsWriter` thread-safe under concurrent `POST /metrics/refresh`? | M1 endpoint hardening | Read `metrics_writer.py`; add `threading.Lock` if not. |

### 0.3 Approval signal

A `LGTM` comment on the SPEC PR is the green light. The implementing
subagent reads the approved SPEC + this MILESTONES, picks M1, and starts.

### 0.4 Exit criteria for Phase 0

- [ ] David has read all four design docs.
- [ ] D1, D2, D3 are decided in writing (PR comment or this file's table).
- [ ] D4, D5 have a resolution path agreed (resolve now vs. resolve when blocked).
- [ ] No outstanding "what does this mean?" questions on SPEC or API.

---

## M1 — Read-only status (the meaningful first ship)

**Goal:** the operator can navigate to `/memory` and see, at a glance, whether
the memory subsystem is alive. No interaction beyond `[Refresh]` and per-card
`[ping]`. This is the single most valuable milestone — it covers the "five
questions in five seconds" from SPEC §0.

### 1.1 What ships

**Panels:** Tier 1 health banner, Tier 2 counters strip, Tier 3 backends grid.
Plus the route registration and the nav item. M4's knowledge-graph card
ships as a static placeholder per SPEC §2 Tier 5 to keep the page from
feeling unfinished below the fold.

**Endpoints implemented** (all from API.md):

- `GET  /api/memory/status` (§1)
- `GET  /api/memory/backends` (§2)
- `GET  /api/memory/counters` (§3)
- `GET  /api/memory/metrics-json` (§4)
- `POST /api/memory/metrics/refresh` (§5)
- `POST /api/memory/backends/{name}/ping` (§6)

### 1.2 Files added / modified

| File | Action | Notes |
|---|---|---|
| `web/src/pages/MemoryPage.tsx` | **new** | Mirrors `PluginsPage.tsx` skeleton. ~350 LOC. |
| `web/src/components/memory/HealthBanner.tsx` | **new** | Tier 1. ~70 LOC. |
| `web/src/components/memory/CountersStrip.tsx` | **new** | Tier 2 stat tiles. ~90 LOC. |
| `web/src/components/memory/BackendsGrid.tsx` | **new** | Tier 3 grid + 4 cards. ~180 LOC. |
| `web/src/components/memory/useMemoryPoll.ts` | **new** | Visibility-gated polling hook (SPEC §9.4). ~40 LOC. |
| `web/src/lib/api.ts` | extend | Add `getMemoryStatus`, `getMemoryBackends`, `getMemoryCounters`, `refreshMemoryMetrics`, `pingMemoryBackend` typed wrappers. |
| `web/src/App.tsx` | extend | 3 edits (SPEC §6): import + route + nav item. |
| `web/src/i18n/en/index.ts` (+ other locales) | extend | Add `nav.memory`, `memoryPage.*` keys (SPEC §6). |
| `hermes_cli/web_server.py` | extend | Add the 6 endpoints above. ~250 LOC including caching + error envelope. |
| `tests/test_web_server.py` | extend | Happy path + auth + Qdrant-down case per endpoint. ~150 LOC. |

### 1.3 Dependencies

- **Code:** none. `hermes_memory_core.health.health_check()` already exists; `MetricsWriter` already exists; `~/.hermes/memory/metrics.json` is already written today. M1 is genuinely additive.
- **Infra:** none.
- **Phase 0:** D5 (MetricsWriter thread-safety) should be resolved before `POST /metrics/refresh` ships — but a `threading.Lock` is trivial if needed.

### 1.4 Estimated subagent-hours

**~5 hours.** Breakdown:

- Backend endpoints + caching + tests: 2h
- Frontend page scaffold + 3 components + nav wiring: 2h
- i18n + lint + manual QA against a live local provider: 0.5h
- Slack for "the existing health_check signature surprised me" buffer: 0.5h

### 1.5 Exit criteria

All from SPEC §14, repeated here:

- [ ] `/memory` route exists; Brain icon visible in nav; deep-linking works.
- [ ] Tier 1 banner flips to `DEGRADED` within 15s of `kill -9 qdrant`.
- [ ] Tier 2 counters numerically match `cat ~/.hermes/memory/metrics.json`.
- [ ] Tier 3 cards show real endpoint, model, last-success-at for each of the 4 backends.
- [ ] All 6 endpoints have unit tests (happy + auth + at least one error path).
- [ ] "Inactive provider" empty state renders correctly when `memory.provider != hermes-local`.
- [ ] No new `pnpm lint`, `pnpm type-check`, or `pytest` regressions.
- [ ] One screenshot attached to the PR matching W1 from WIREFRAMES.md.

### 1.6 Risks specific to M1

- **`metrics.json` staleness** (SPEC §10 Risk 1) is exposed but not yet
  fixed in M1. We surface the stale badge and document the limitation;
  the optional 5-min cron refresh is M2 scope.
- **Probe latency dominates `GET /backends`** when LMS is sluggish (500ms+).
  Caching at 10s/backend mitigates; if it still feels slow, add an
  `async.gather` over the 4 backends. Not expected to bite.

---

## M2 — Dreamer interaction

**Goal:** the operator can browse dream history, view individual dream
reports, and trigger a dreamer run manually (with confirmation). The Tier 4
panel becomes fully populated and the `[Run now ⚠]` action works end-to-end.

### 2.1 What ships

**Panels added:** Tier 4 (full — last-run block, schedule block, recent runs
table, `[Run now]` button with cost-disclosing confirm dialog).

**Subviews added:** the dream-report drawer (W3 in WIREFRAMES.md).

**Endpoints implemented:**

- `GET  /api/memory/dreamer/last` (§7)
- `GET  /api/memory/dreamer/runs` (§8)
- `GET  /api/memory/dreamer/runs/{id}` (§9)
- `GET  /api/memory/dreamer/runs/{id}/report` (§10)
- `GET  /api/memory/dreamer/schedule` (§11)
- `POST /api/memory/dreamer/run-now` (§12)

### 2.2 Files added / modified

| File | Action | Notes |
|---|---|---|
| `web/src/pages/MemoryPage.tsx` | extend | Mount `<DreamerPanel />`. ~30 LOC added. |
| `web/src/components/memory/DreamerPanel.tsx` | **new** | Tier 4. ~200 LOC. |
| `web/src/components/memory/DreamReportDrawer.tsx` | **new** | W3 drawer. Reuses existing markdown renderer. ~120 LOC. |
| `web/src/components/memory/DreamerRecentRuns.tsx` | **new** | Dense table. ~70 LOC. |
| `web/src/pages/MemoryDreamerPage.tsx` | **new** | `/memory/dreamer` paginated full-history page. ~150 LOC. |
| `web/src/lib/api.ts` | extend | 6 new wrappers. |
| `hermes_cli/web_server.py` | extend | 6 new endpoints. ~300 LOC including `systemctl` parsing + job tracking. |
| `tests/test_web_server.py` | extend | Per endpoint + job-status flow. ~200 LOC. |

### 2.3 Dependencies

- **M1** must be merged — `MemoryPage.tsx` scaffold, the polling hook, the
  page header refresh button, the i18n keys all come from M1.
- **`systemctl` access** — the gateway process must be able to call
  `systemctl --user list-timers` (or system-wide depending on how the timer
  is installed). On non-systemd hosts the endpoint returns
  `{"supported": false, ...}` cleanly (API §11).
- **Job-tracking infra** — `POST /run-now` reuses the existing
  `/api/actions/{name}/status` mechanism (referenced in SPEC §4.2). No new
  infra needed; verify by reading `gateway/restart` action.

### 2.4 Estimated subagent-hours

**~7 hours.** Breakdown:

- Backend endpoints (especially `systemctl` parsing & error handling): 2.5h
- Job-tracking integration for `run-now`: 1h
- Frontend `<DreamerPanel>` + drawer + confirm dialog: 2h
- `/memory/dreamer` full-history sub-page: 1h
- Tests + manual QA against a real dreamer run: 0.5h

### 2.5 Exit criteria

- [ ] `<DreamerPanel>` renders on `/memory` with real last-run + schedule + 5 most-recent rows.
- [ ] Clicking `[view report]` opens the drawer with the markdown rendered.
- [ ] Clicking `[see all 47 runs →]` deep-links to `/memory/dreamer` paginated table.
- [ ] Clicking `[Run now ⚠]` opens the cost-disclosing dialog (SPEC §8); clicking confirm calls `POST /dreamer/run-now` and streams status via `/api/actions/{job}/status`.
- [ ] Triggering the same `[Run now]` while a run is in flight produces a clean 409 and a toast — no double-fire.
- [ ] On a non-systemd host, the schedule block renders "Scheduling not available on this host" and does not crash.
- [ ] The dream-report drawer handles the report-file-deleted case (404 from `/report`).

### 2.6 Risks specific to M2

- **`systemctl` shell-out from FastAPI** (SPEC §10 Risk 2) — we depend on
  text parsing of `list-timers`. Format changes between systemd majors. We
  pin to a tested format and unit-test the parser with golden output.
- **Job collision** — if a `systemd`-timer-initiated dream is in flight when
  the operator clicks `[Run now]`, we must detect it. The `409` code carries
  the running run's `job_id` so the UI can attach to its status stream rather
  than refusing.
- **Privileged action over a browser** (SPEC §10 Risk 3) — mitigated by CSRF
  + cost-dialog + server-side rate limit. D3 may upgrade to typed-confirm.

---

## M3 — Search playground + activity feed

**Goal:** the operator can query the memory interactively (Tier 6) and see a
live stream of memory operations (Tier 7). This turns the page from a status
board into a power tool.

### 3.1 What ships

**Panels added:** Tier 6 (search playground, W2 in WIREFRAMES.md), Tier 7
(activity feed).

**Endpoints implemented:**

- `POST /api/memory/search` (§13)
- `GET  /api/memory/activity` (§14, paginated fallback)
- `GET  /api/memory/activity/stream` (§15, SSE) — **or** a `memory:event`
  filter on the existing `/api/events` ws, per D2.

### 3.2 Files added / modified

| File | Action | Notes |
|---|---|---|
| `web/src/pages/MemoryPage.tsx` | extend | Mount `<SearchPlayground />` + `<ActivityFeed />`. |
| `web/src/components/memory/SearchPlayground.tsx` | **new** | Tier 6. Form + results table + mode-disabled tooltips. ~180 LOC. |
| `web/src/components/memory/SearchResultsTable.tsx` | **new** | Hits table with score + source pills. ~90 LOC. |
| `web/src/components/memory/ActivityFeed.tsx` | **new** | Tier 7. SSE primary, polling fallback. ~150 LOC. |
| `web/src/lib/api.ts` | extend | `runMemorySearch`, `getMemoryActivity`, `subscribeMemoryActivity` (SSE). |
| `hermes_cli/web_server.py` | extend | 3 endpoints + SSE wiring or ws-bus filter. ~250 LOC. |
| `hermes_memory_core/audit_log.py` (if absent) | **maybe new** | Source for `/activity`. See risk below. |
| `tests/test_web_server.py` | extend | Search happy/empty/down-mode + activity pagination. |

### 3.3 Dependencies

- **M1** for the page scaffold and the Tier 3 backend status (the search
  playground reads it to disable `semantic`/`hybrid` modes when backends are
  red).
- **`audit_log` table** — currently unclear whether this exists or whether
  `/activity` must aggregate from `dream_runs` + a new event log. Resolve
  in Phase 0; this is the M3 equivalent of D4.
- **SSE infra** (D2) — if "piggyback on `/api/events`" wins, we need a
  `memory:` channel convention added to that bus. If "new dedicated SSE"
  wins, we wire `EventSourceResponse` from `sse-starlette`.

### 3.4 Estimated subagent-hours

**~9 hours.** Breakdown:

- Backend `/search` (mostly wrapping existing `query()`): 1h
- Backend `/activity` (depends heavily on audit_log status): 2–3h
- SSE / ws-bus wiring: 1.5h
- `<SearchPlayground>` + table + meta-line: 2h
- `<ActivityFeed>` + SSE reconnect + polling fallback: 2h
- Tests + rate-limit verification: 0.5h

The estimate spread comes from the audit_log uncertainty. If the table
exists with the schema we want, this drops to ~7h; if we have to add a
log-writer to capture pipeline edges, it climbs to ~10h.

### 3.5 Exit criteria

- [ ] Typing a query and hitting `[Search]` returns hits in < 500ms p95 on the local Qdrant.
- [ ] All four modes (`hybrid`, `semantic`, `keyword`, `facts`) work.
- [ ] When Qdrant is down, `semantic`/`hybrid` are visibly disabled with a tooltip carrying the `BACKEND_DOWN` reason.
- [ ] `<ActivityFeed>` shows new events within 2s of their server emission (SSE path).
- [ ] When SSE fails to connect (forced via devtools), the feed falls back to 10s polling without a visible glitch.
- [ ] Rate-limit on `/search` (6 req/min) returns 429 with `Retry-After`.

### 3.6 Risks specific to M3

- **`audit_log` may not exist.** TODO for Phase 0: confirm. If absent,
  proposed fallback is to derive the feed from a UNION of:
  - `dream_runs` (synthesized start/complete events)
  - a new lightweight `memory_events` table populated by `MemoryProvider`
    hooks (added in this milestone as an additive change).
- **SSE infra may not exist** at the dashboard level. We already have
  `/api/events` ws, so the piggyback path is concrete; the dedicated-SSE
  path needs `sse-starlette` added. Flagged as D2.
- **Embed budget drain.** A misbehaving operator could spam `[Search]` in
  `hybrid` mode and burn LMS GPU. The 6/min rate-limit is the floor; we may
  want to additionally meter against the existing LMS budget surface.

---

## M4 — Knowledge graph + advanced ops

**Goal:** the long tail. The operator can see top entities, browse
contradictions, trigger a backup, and (gated) re-init Qdrant collections.

### 4.1 What ships

**Panels added:** Tier 5 (real, no longer a placeholder).

**Subviews added:** entity drill-down (clicking a row in the entity table),
contradictions browser, backup management.

**Endpoints implemented:**

- `GET  /api/memory/entities` (§16)
- `GET  /api/memory/contradictions` (§17)
- `POST /api/memory/backup` (§18)
- `POST /api/memory/qdrant/reinit` (§19)

### 4.2 Files added / modified

| File | Action | Notes |
|---|---|---|
| `web/src/pages/MemoryPage.tsx` | extend | Mount `<KnowledgeGraphCard />`. |
| `web/src/components/memory/KnowledgeGraphCard.tsx` | **new** | Tier 5. Top-10 entities + sparkline. ~150 LOC. |
| `web/src/components/memory/EntityTable.tsx` | **new** | Full entity table for the drill-down sub-page. ~120 LOC. |
| `web/src/components/memory/ContradictionsList.tsx` | **new** | Side-by-side fact A vs. fact B. ~90 LOC. |
| `web/src/components/memory/BackupCard.tsx` | **new** | Backup trigger + history. ~80 LOC. |
| `web/src/pages/MemoryEntitiesPage.tsx` | **new** | `/memory/entities` paginated. ~120 LOC. |
| `web/src/lib/api.ts` | extend | 4 new wrappers. |
| `hermes_cli/web_server.py` | extend | 4 new endpoints. ~200 LOC. |
| `scripts/hermes_memory_backup.sh` | **new** | Snapshot SQLite + metrics.json + Qdrant dump. ~50 LOC of bash. |
| `hermes_memory_core/store/qdrant.py` | maybe extend | Confirm `init_collections(force=...)` signature. |
| `tests/test_web_server.py` | extend | Per endpoint + the typed-confirm gate for `force=true`. |

### 4.3 Dependencies

- **M1–M3** all merged.
- **D4** resolved — schema must have either an `entity` column on `facts`,
  or `subject`/`object` columns we can aggregate against. If neither, M4
  is blocked pending a schema migration (which is its own work item, not in
  M4's hour estimate).
- **`scripts/hermes_memory_backup.sh` must exist.** It does not today;
  writing it is part of M4. Confirm with David before assuming the shape
  (zstd-compressed tar vs. plain tar vs. SQL dump).
- **Viz library decision** — SPEC proposes `recharts` for sparklines but
  defers a graph-viz library. M4 needs to land on a sparkline lib only;
  full graph viz (node-link diagram of entities + relations) is **out of
  scope** for M4 and would be a hypothetical M5.

### 4.4 Estimated subagent-hours

**~13 hours.** Breakdown:

- Schema confirmation + entity aggregation query optimization: 1.5h
- Backend `/entities` + `/contradictions`: 2h
- `scripts/hermes_memory_backup.sh` + `/backup` endpoint: 2h
- `/qdrant/reinit` (especially the typed-confirm logic + `force=true` path): 1.5h
- `<KnowledgeGraphCard>` + sparkline integration: 2h
- `<ContradictionsList>` + `<BackupCard>`: 2h
- `/memory/entities` sub-page: 1.5h
- Tests + manual QA (especially the irreversible force-reinit): 0.5h

The spread here is bigger than M1–M3 because (a) the schema is the variable
of largest uncertainty, and (b) `force=true` Qdrant reinit deserves
careful manual verification.

### 4.5 Exit criteria

- [ ] Tier 5 card shows the top 10 entities by fact count, with avg trust.
- [ ] Clicking an entity opens `/memory/entities/{entity}` with paginated facts.
- [ ] `/contradictions` lists all detected contradictions, or empty-states cleanly.
- [ ] `[Trigger backup]` produces a `.tar.zst` (or chosen format) in `~/.hermes/memory/backups/` and the toast shows path + size.
- [ ] `[Re-init Qdrant]` (non-force) creates missing collections idempotently.
- [ ] `[Re-init Qdrant]` (force) refuses without `confirm: "RESET"`, accepts with it, and verifiably drops + recreates collections.
- [ ] Qdrant drift detector (SPEC §10 bonus risk) emits a warning in Tier 1 when `qdrant_points` and `chunks_with_qdrant_id` diverge > 10%.

### 4.6 Risks specific to M4

- **Schema drift** — if `facts.entity` doesn't exist, M4 needs an upstream
  migration. The dashboard can't ship before that. Flag early in Phase 0.
- **Backup script behavior on a hot WAL** — `sqlite3 .backup` is the right
  primitive (concurrent-safe), but the script writer must use it correctly.
  Naïve `cp` of `memory.sqlite` while writes are in flight will corrupt the
  backup.
- **Force-reinit is destructive.** All Qdrant vector data is lost. Even with
  the typed gate, an operator could click through. We document this loudly
  in the dialog copy and link to the recovery procedure (re-dream).
- **Sparkline library churn** — if we add `recharts`, it's a new build-time
  dependency. Confirm bundle-size impact (~50 KB gzipped) is acceptable.

---

## Cross-milestone summary

| Milestone | Subagent-hours | Endpoints | New panels | Files added | Risk |
|---|---|---|---|---|---|
| Phase 0 | reading only | — | — | — | low |
| **M1** | ~5 | 6 | Tiers 1, 2, 3 + placeholder Tier 5 | 5 new + 4 edited | low |
| **M2** | ~7 | 6 | Tier 4 + report drawer | 4 new + 3 edited | medium (`systemctl`) |
| **M3** | ~9 | 3 | Tiers 6, 7 | 4 new + 3 edited | medium-high (audit_log uncertain) |
| **M4** | ~13 | 4 | Tier 5 (real) + backups | 6 new + 4 edited | medium (schema, destructive ops) |
| **Total** | **~34** | 19 | 7 tiers | ~30 files touched | — |

That total assumes serial execution. Realistically M1 and the M2 backend
can overlap (different files), and once M1's `MemoryPage.tsx` scaffold is
merged, M2/M3 frontend work can also overlap. Two subagents in parallel
brings end-to-end calendar time to roughly 18–20 active hours.

---

## What is NOT in any milestone

Captured for completeness — these are not on the roadmap unless and until
David asks:

- **Multi-tenant memory views.** The page operates on the single
  `~/.hermes/memory` tree. No "switch user" affordance.
- **Editing facts by hand.** The dreamer is the editor. The dashboard is read-only on facts (M1–M3) and read-only-plus-delete (M4 contradiction
  resolution, if we ship it — currently unscoped).
- **Wipe-and-reset memory.** Out of scope (SPEC §8). The CLI remains the
  only path.
- **A node-link knowledge-graph viz.** SPEC §13 leaves this as an open
  hypothetical M5. Not committed.
- **Cross-provider comparison.** If `holographic` (or any future provider)
  arrives, this page handles "hermes-local is inactive" cleanly (SPEC §7)
  but does not render a side-by-side comparison.

---

*End of MILESTONES.md.*
