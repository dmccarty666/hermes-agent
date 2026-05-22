# MemoryPage — Wireframes

**Sibling docs:** [SPEC.md](./SPEC.md) · [API.md](./API.md) · [MILESTONES.md](./MILESTONES.md)

This file is the visual companion to SPEC.md. The seven information tiers
defined in SPEC §2 are rendered here as plaintext ASCII, with the click paths
into the two interactive subviews (search playground, single dream report)
and the narrow-viewport behavior.

ASCII intentionally — these are wireframes, not mocks. Layout, hierarchy and
labels are normative; pixel-exact spacing, colors and typography defer to the
existing `@nous-research/ui` tokens used by `PluginsPage.tsx` and
`AnalyticsPage.tsx`.

---

## W1 — Default landing view (`/memory`)

The page the operator sees the moment they click the **Memory** nav item. No
modals open, no subviews expanded. The seven tiers from SPEC §2 stack
top-to-bottom with `gap-8` spacing.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Hermes  ▸  Memory                                       [↻ Refresh]     │  page header (sticky)
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ╔════════════════════════════════════════════════════════════════════╗  │  Tier 1 — health banner
│  ║  ● OK   hermes-local · active · refreshed 2s ago         [↻]     ║  │  (sticky under header)
│  ╚════════════════════════════════════════════════════════════════════╝  │
│                                                                          │
│  ┌─Counters──────────────────────────────────────────────────────────┐   │  Tier 2 — stat tiles
│  │ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐         │   │
│  │ │ 76   │ │ 12   │ │ 12   │ │  0   │ │1,432 │ │ 3h ago   │         │   │
│  │ │facts │ │turns │ │chunks│ │pend. │ │qpts  │ │last dream│         │   │
│  │ │ ↑3   │ │  24h │ │  24h │ │      │ │ ↑15  │ │ ✓ ok     │         │   │
│  │ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────────┘         │   │
│  │                                       metrics.json · 2m stale [↻] │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─Backends──────────────────────────────────────────────────────────┐   │  Tier 3 — backend grid (2×2)
│  │ ┌─ SQLite ────────────────┐  ┌─ Qdrant ───────────────────────┐   │   │
│  │ │ ● ok                    │  │ ● ok                           │   │   │
│  │ │ ~/.hermes/…/memory.db   │  │ http://localhost:6333          │   │   │
│  │ │ wal · 4.2 MB            │  │ 3 collections · 1,432 points   │   │   │
│  │ │ last ok 2s ago          │  │ last ok 16s ago                │   │   │
│  │ │ [view tables]           │  │ [ping]   [re-init ⚠]           │   │   │
│  │ └─────────────────────────┘  └────────────────────────────────┘   │   │
│  │ ┌─ LMS embedding ─────────┐  ┌─ Dreamer LLM ──────────────────┐   │   │
│  │ │ ● ok                    │  │ ● ok                           │   │   │
│  │ │ 192.168.2.105:1235      │  │ 192.168.2.105:1234             │   │   │
│  │ │ Qwen3-Embed-8B · 1536   │  │ qwen3.6-35b-instruct           │   │   │
│  │ │ last ok 3s ago          │  │ p50 8.3s · last ok 3h ago      │   │   │
│  │ │ [ping]                  │  │ [ping]                         │   │   │
│  │ └─────────────────────────┘  └────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─Dreamer───────────────────────────────────────────────────────────┐   │  Tier 4 — dreamer activity
│  │  Last run                                                         │   │
│  │  ● completed · started 3h ago · duration 47s                      │   │
│  │  facts +5  ·  decisions +0  ·  open_qs +1  ·  turns processed 12  │   │
│  │  [view report ▸]                                                  │   │
│  │  ────────────────────────────────────────────────                 │   │
│  │  Next run                                                         │   │
│  │  hermes-memory-dream.timer · in 21h · enabled · active            │   │
│  │  [Run now ⚠]                                                      │   │
│  │  ────────────────────────────────────────────────                 │   │
│  │  Recent runs                                                      │   │
│  │  3h ago     47s     ✓ completed     +5  facts                     │   │
│  │  1d ago     52s     ✓ completed    +12  facts                     │   │
│  │  2d ago     —       ✗ error         extraction parse failure      │   │
│  │  3d ago     38s     ✓ completed     +9  facts                     │   │
│  │  4d ago     41s     ✓ completed     +7  facts                     │   │
│  │  [see all 47 runs →]                                              │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─Knowledge graph  (M4)─────────────────────────────────────────────┐   │  Tier 5 — placeholder until M4
│  │  Knowledge-graph view ships in M4.                                │   │
│  │  See docs/design/memory-dashboard/SPEC.md §2 Tier 5.               │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─Search playground  (M3)───────────────────────────────────────────┐   │  Tier 6 — collapsed default
│  │  query  [_______________________________]                         │   │
│  │  mode   [hybrid ▼ semantic   keyword   facts]                     │   │
│  │  limit  [10]   min_score  [0.0]              [Search]             │   │
│  │  (results render below after submit — see W2)                     │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─Recent activity  (M3)─────────────────────────────────────────────┐   │  Tier 7 — feed
│  │  2s ago    memory_query     agent     hybrid q="qdrant" → 3 hits  │   │
│  │  14s ago   sync_turn        agent     session abc12345 · turn 4   │   │
│  │  47s ago   memory_write     agent     3 chunks · session abc12345 │   │
│  │  3h ago    dream_run_done   dreamer   +5 facts                    │   │
│  │  3h ago    dream_run_start  dreamer   12 turns queued             │   │
│  │  …                                                                │   │
│  │  [load 50 more]                                  ● live (SSE)     │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Endpoint binding (per tier):**

- **Tier 1 banner** → `GET /api/memory/status` (API §1) · poll **15s** · pauses on `document.hidden`.
- **Tier 2 counters** → `GET /api/memory/counters` (API §3) · poll **60s** · `[↻]` calls `POST /api/memory/metrics/refresh` (API §5).
- **Tier 3 backends grid** → `GET /api/memory/backends` (API §2) · poll **30s** · per-card `[ping]` calls `POST /api/memory/backends/{name}/ping` (API §6).
- **Tier 4 dreamer panel** → `GET /api/memory/dreamer/last` (API §7) + `GET /api/memory/dreamer/runs?limit=5` (API §8) + `GET /api/memory/dreamer/schedule` (API §11) · fetch on mount + after `[Run now]` returns.
- **Tier 5 graph card** → no endpoint in M1–M3 (static placeholder). M4: `GET /api/memory/entities` (API §16).
- **Tier 6 search** → `POST /api/memory/search` (API §13) · **on submit only**.
- **Tier 7 activity** → `GET /api/memory/activity/stream` (API §15, SSE) primary; `GET /api/memory/activity?limit=50` (API §14) every **10s** as fallback when SSE fails to connect.

**Interactive elements on this view:**

- Page-header `[↻ Refresh]` — re-fetches every panel in parallel.
- Tier 1 inline `[↻]` — re-fetches `/status` only.
- Tier 2 `[↻]` (next to stale badge) — `POST /metrics/refresh` then re-renders.
- Each Tier 3 card `[ping]` — independent re-probe.
- Tier 3 Qdrant `[re-init ⚠]` — confirm dialog (SPEC §8), M4-gated.
- Tier 4 `[view report ▸]` — opens W3 (single-report subview as a drawer).
- Tier 4 `[Run now ⚠]` — cost-disclosing confirm dialog → `POST /api/memory/dreamer/run-now` (API §12).
- Tier 4 `[see all 47 runs →]` — routes to `/memory/dreamer` (paginated table).
- Tier 6 `[Search]` — submits the playground (W2).
- Tier 7 `[load 50 more]` — paginated GET fallback regardless of SSE state.

---

## W2 — Search playground (Tier 6, expanded after submit)

The operator types a query, picks a mode, and clicks Search. The card grows
to include a results table. The card never collapses on its own — the
operator scrolls past it.

```
┌─Search playground  (M3)─────────────────────────────────────────────────┐
│                                                                         │
│  query  [qdrant collection recovery_______________________]             │
│                                                                         │
│  mode   ( hybrid ●)( semantic )( keyword )( facts )                     │
│         hybrid: qdrant + sqlite FTS + facts, fused (default)            │
│                                                                         │
│  limit  [10 ]   min_score  [0.0]                            [Search]    │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────    │
│  142 ms · 1 embed call (38 ms) · 3 hits                                 │
│                                                                         │
│  ┌─#─┬─score─┬─source─────┬─snippet──────────────────────┬─session───┐  │
│  │ 1 │ 0.86  │ qdrant     │ …we lost the hermes_memory…  │ abc12345 →│  │
│  │ 2 │ 0.74  │ sqlite_fts │ …Qdrant restart procedure…   │ def67890 →│  │
│  │ 3 │ 0.61  │ facts      │ "Qdrant has no persistence…" │ ghi24680 →│  │
│  └───┴───────┴────────────┴──────────────────────────────┴───────────┘  │
│                                                                         │
│  No more hits above min_score 0.0.                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Endpoint binding:** `POST /api/memory/search` (API §13). One request per
click on `[Search]`. The response shape (`hits[]`, `elapsed_ms`,
`embed_calls`, `embed_ms`) populates the meta-line and the table 1:1.

**Refresh cadence:** none. Submit-only.

**Interactive elements:**

- `query` input — `Enter` submits, identical to clicking `[Search]`.
- `mode` segmented control — only 1 active at a time. When Qdrant or LMS is
  red in Tier 3, `semantic` and `hybrid` render disabled with a tooltip
  carrying the `BACKEND_DOWN` `code` from the API error envelope
  (API §1 "Standard error envelope").
- `limit`/`min_score` — numeric inputs, validated client-side. `limit` clamps
  to 1–100. `min_score` clamps to 0.0–1.0.
- Per-row session link — opens the existing `/sessions/{session_id}` page,
  scrolled to the matching turn (we already deep-link sessions today).
- The meta-line `1 embed call` line is informational only. Operators care
  about embed budget; we surface it for free.

**Mobile collapse rules:** the results table switches from 5 columns to a
single-column card stack (each hit becomes a small card with score badge,
source pill, snippet, session link). Width: see W4.

**Empty / edge:**

- 0 hits → "No hits. Try `mode=keyword` or lower `min_score`."
- 400 (empty query) → inline error under the query input.
- 503 (mode requires a down backend) → red banner inside the card with the
  `detail` text, and the offending mode pill is auto-switched off.

---

## W3 — Single dream report (drawer, opened from Tier 4 `[view report]`)

A right-side drawer (full height, ~640px wide on desktop) that slides over
the page. The page underneath is dimmed but not unmounted, so closing the
drawer is free.

```
                                            ┌─Dream report ─────────────────┐
                                            │   2026-05-22 11:00 UTC   [✕]  │
                                            ├───────────────────────────────┤
                                            │  Header                       │
                                            │  ──────                       │
                                            │  Run ID:  abc1234-5678        │
                                            │  Status:  ● completed         │
                                            │  Started: 11:00:00 UTC        │
                                            │  Finished:11:00:47 UTC        │
                                            │  Duration:47.3 s              │
                                            │                               │
                                            │  Summary                      │
                                            │  ───────                      │
                                            │  facts          +5            │
                                            │  decisions      +0            │
                                            │  open_questions +1            │
                                            │  turns processed 12           │
                                            │  embed calls     4            │
                                            │  llm tokens  in  8,432        │
                                            │             out  1,121        │
                                            │                               │
                                            │  Errors                       │
                                            │  ──────                       │
                                            │  (none)                       │
                                            │                               │
                                            │  Sessions touched (3)         │
                                            │  ─────────────────            │
                                            │  abc12345 — 7 turns →         │
                                            │  def67890 — 3 turns →         │
                                            │  ghi24680 — 2 turns →         │
                                            │                               │
                                            │  Raw report (markdown)        │
                                            │  ─────────────────────        │
                                            │  ┌──────────────────────────┐ │
                                            │  │ # Dream report           │ │
                                            │  │ ## New facts (5)         │ │
                                            │  │ - David runs Hermes…     │ │
                                            │  │ - Qdrant lost…           │ │
                                            │  │ - …                      │ │
                                            │  │ ## Open questions (1)    │ │
                                            │  │ - Should we…             │ │
                                            │  │ …                        │ │
                                            │  └──────────────────────────┘ │
                                            │                               │
                                            │  [open in editor]  [copy md]  │
                                            └───────────────────────────────┘
```

**Endpoint binding:**

- Header + Summary + Sessions touched + Errors → `GET /api/memory/dreamer/runs/{id}` (API §9) — the full row.
- Raw report → `GET /api/memory/dreamer/runs/{id}/report` (API §10) — markdown blob.

Both fire in parallel when the drawer opens. The drawer renders header data
as soon as `/runs/{id}` resolves; the markdown pane shows a skeleton until
`/report` resolves. If `/report` returns 404 (report file deleted),
the markdown pane shows: "Report file missing on disk — `{path}`."

**Refresh cadence:** none. The report is an immutable artifact for a completed
run. (For `status=running` we don't render the drawer at all — the `[view
report]` link is hidden.)

**Interactive elements:**

- `[✕]` close — also Esc.
- Per-session `→` link — routes to `/sessions/{id}` (existing page).
- `[open in editor]` — only present on desktop; emits a request to the
  existing `/api/files/open` action with the absolute report path.
- `[copy md]` — copies the raw markdown to clipboard. No backend round-trip.

**Mobile:** the drawer becomes a full-screen sheet rather than a right-rail
drawer. The Sessions-touched block and the Raw report block stack vertically
without their headers; we keep the header + summary above the fold.

---

## W4 — Mobile / narrow viewport (< 720px wide)

Mobile is not the primary use case — David runs Hermes from a laptop. But
the page must not embarrass itself on a phone, and on tablets in landscape
the page must remain readable.

```
┌──────────────────────────────────┐
│ ☰  Memory          [↻]           │  page header — hamburger nav
├──────────────────────────────────┤
│ ● OK · hermes-local · 2s ago [↻] │  Tier 1 (still sticky)
├──────────────────────────────────┤
│ ┌────────┐  ┌────────┐           │  Tier 2 — counters wrap 2/row
│ │  76    │  │  12    │           │
│ │ facts  │  │ turns  │           │
│ │  ↑3    │  │  24h   │           │
│ └────────┘  └────────┘           │
│ ┌────────┐  ┌────────┐           │
│ │  12    │  │   0    │           │
│ │chunks  │  │ pend.  │           │
│ └────────┘  └────────┘           │
│ ┌────────┐  ┌────────┐           │
│ │ 1,432  │  │ 3h     │           │
│ │ qpts   │  │ last d │           │
│ └────────┘  └────────┘           │
├──────────────────────────────────┤
│ Backends           [show 4 ▸]    │  Tier 3 — collapsed by default
├──────────────────────────────────┤
│ Dreamer                          │  Tier 4 — last-run block always
│ ● completed · 3h ago · 47s       │  visible; recent-runs collapsed
│ +5 facts · 12 turns              │
│ [view report]   [Run now ⚠]      │
│ ── Next: in 21h ──               │
│ Recent runs       [show 5 ▸]     │
├──────────────────────────────────┤
│ Knowledge graph (M4)             │
│ Ships in M4.                     │
├──────────────────────────────────┤
│ Search            [show ▸]       │  Tier 6 — collapsed
├──────────────────────────────────┤
│ Activity          [show ▸]       │  Tier 7 — collapsed
└──────────────────────────────────┘
```

**Mobile collapse rules (canonical):**

| Tier | < 720px behavior |
|---|---|
| Header | Hamburger replaces full nav. `[Refresh]` shrinks to icon. |
| 1 Banner | Always visible. Sticky under header. Refresh icon stays. |
| 2 Counters | Wrap from 6-across to 2-across. Deltas stay; "24h" label suppressed when no Δ. |
| 3 Backends | Collapsed by default behind `[show 4 ▸]`. Once expanded, each card is full-width and stacked. |
| 4 Dreamer | Last-run block + `[Run now]` always visible. Schedule + recent-runs hide behind `[show 5 ▸]`. |
| 5 Graph | Always visible (it's a 1-line placeholder anyway). |
| 6 Search | Collapsed behind `[show ▸]`. When expanded, the mode pills wrap 2×2 instead of 1×4. |
| 7 Activity | Collapsed behind `[show ▸]`. When expanded, each row becomes a 2-line cell (timestamp+kind on line 1, summary on line 2). |

**Sticky-header behavior:** the page header and Tier 1 banner share the
sticky region (combined ~96px tall). On scroll, both stay; everything else
scrolls beneath them. This matches `SessionsPage`'s current behavior.

**Reduced polling on mobile:** we already gate polling on
`document.visibilityState === "visible"` (SPEC §9.4); we additionally
halve all poll intervals when the viewport width is under 720px and the tab
is visible (15s→30s, 30s→60s, 60s→120s). Phones are typically on cellular
and the operator is making a quick check, not monitoring.

**What is *not* available on mobile:** the dream-report drawer's `[open in
editor]` button is hidden (no editor on a phone). Everything else works
identically — the privileged-action confirm dialog included, since pressing
`[Run now ⚠]` on a phone deserves the same friction as on a laptop.

---

*End of WIREFRAMES.md.*
