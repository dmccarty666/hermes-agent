# Storyboard — hermes-local memory

Read this if you want the narrative without running the HTML. Each section
maps 1:1 to a `<section class="frame">` in `index.html`.

Total runtime: **~75 seconds**.
Color palette throughout: dark navy backgrounds, teal (`#5eead4`) for user/data
flows, amber (`#fbbf24`) for system events, violet (`#a78bfa`) for the dreamer
/ semantic axis, rose (`#fb7185`) for PII / redaction signal.

---

## Frame 1 — "The Problem" (6s)
**Eyebrow:** `01 · the problem`
**Caption:** *Without memory, every conversation starts from scratch.*

**Focal element:** a glowing teal orb labeled `LLM` centered on the stage,
surrounded by four floating thought bubbles.

**Choreography:**
1. `t=0.0` — orb fades in and scales up (`power2.out`, 1.0s). Ring slowly rotates indefinitely.
2. `t=0.5..1.5` — four thought bubbles fade in around the orb, staggered 250ms each.
   Each contains a real-feeling memory: "my name is David", "i prefer concise replies",
   "yesterday we discussed Qdrant", "book launch is Nov 14".
3. `t=2.0..3.5` — each thought *pops* (translateY up, scale down, fade out) one
   by one — the visual metaphor for memory loss across turns.
4. Caption slides in from below at `t=0.2`.

**Emotional beat:** wistful → frustrating. The model is bright but forgetful.

---

## Frame 2 — "The Capture" (10s)
**Eyebrow:** `02 · capture`
**Caption:** *Every turn is captured — losslessly, with PII redaction.*

**Focal element:** three columns — chat stream (left), gradient flow arrow
(middle), database cylinder labeled `turns.jsonl` (right).

**Choreography:**
1. `t=0..2.5` — five chat bubbles cascade down the left column, alternating
   `user:` (teal-tinted) and `asst:` (amber-tinted). Inline PII (a name, a phone
   number, an email) is rendered in rose.
2. `t=2.2` — all PII tokens snap to redacted state simultaneously: text goes
   transparent, background becomes a translucent rose rectangle. (400ms ease.)
3. `t=1.0` — the flow arrow draws in, gradient sweeping teal→amber.
4. `t=1.4..2.0` — three DB cylinder slices stack into view bottom-up.
5. `t=2.5..4.5` — cylinder pulses with a teal glow (3 yoyos) to signal write-commits.

**Emotional beat:** mechanical reassurance. *"We saved it. We did not save what we shouldn't."*

---

## Frame 3 — "The Dreamer" (12s)
**Eyebrow:** `03 · the dreamer`
**Caption:** *Each night, a separate LLM extracts structured knowledge.*

**Focal element:** moon (left) + analog 03:00 clock (center) + violet "Qwen3 dreamer" brain (right).
Cards labeled FACT / DECISION / OPEN fly out from the brain into the right half.

**Background:** the grid switches to violet tint; 60 procedurally placed
twinkling stars fade in/out indefinitely at random delays.

**Choreography:**
1. `t=0..1.0` — moon rises (translateY down + scale up). Its glow halo
   yoyo-pulses for the duration of the frame.
2. `t=0.4` — clock face fades in, showing `03:00` in monospaced teal.
3. `t=0.8` — brain core fades + scales in. A violet pulse ring expands out of
   it on repeat (1.8s cycle), signaling active inference.
4. `t=2.0..` — four structured-knowledge cards fly out from the brain (one
   every 0.6s), tagged FACT (teal), DECISION (amber), OPEN (violet). Each
   gently floats (`yoyo` translateY of 6px) once landed.

**Emotional beat:** quiet, magical. *Something is working while you sleep.*

---

## Frame 4 — "Storage Tiers" (10s)
**Eyebrow:** `04 · storage`
**Caption:** *Stored three ways: structured rows, semantic vectors, raw events.*

**Focal element:** one source fact card at the top, three fan-out dashed lines
flowing down into three side-by-side storage panels.

**Choreography:**
1. `t=0..0.6` — source card fades in from above.
2. `t=0.6..1.2` — three dashed fan-out lines draw themselves from source to
   each tier (using `getTotalLength()` + `strokeDashoffset` trick).
3. `t=1.2..2.0` — three tier cards rise into view, staggered 200ms each:
   - **SQLite** (teal): a 3×3 row/key/value grid populates cell-by-cell.
   - **Qdrant** (violet): a 2D vector space with five dim violet dots and one
     bright amber dot (the matching fact) that scale-in and then pulse.
   - **JSONL** (amber): three monospace lines like `{"ts":"03:00","kind":"fact",…}`
     slide in left-to-right.

**Emotional beat:** "all the right answers." Three storage strategies, one truth.

---

## Frame 5 — "The Retrieval" (12s)
**Eyebrow:** `05 · retrieval`
**Caption:** *Hybrid retrieval: 4 backends, weighted merge, ranked output.*

**Focal element:** a query card at the top, splitter lines, four backend
cards (keyword/semantic/structural/recency), merger lines, ranked result rows.

**Choreography:**
1. `t=0..0.6` — query card "what did we decide about Qdrant?" lands.
2. `t=0.6..1.2` — four splitter lines fan downward, each in a different color
   (teal, violet, amber, rose) — one per backend.
3. `t=1.2..1.8` — backends slide up and fade in, staggered. Each shows engine
   name (FTS5 / Qdrant / HRR / Jaccard) and a score (0.81 / 0.92 / 0.74 / 0.66).
4. `t=1.6..2.2` — score values pulse with a brief text-shadow flare (each
   color matches its backend lane).
5. `t=2.4..2.7` — four lines converge into the merger.
6. `t=3.0..3.6` — three ranked result rows slide in from the left:
   #1 `prefers Qdrant over pgvector` (0.94)
   #2 `Qdrant collection: hermes_facts` (0.88)
   #3 `embedding model = nomic-embed` (0.71)

**Emotional beat:** systems-thinking, satisfying convergence. Four sources, one ranking.

---

## Frame 6 — "The Loop Closes" (8s)
**Eyebrow:** `06 · loop closes`
**Caption:** *The model now remembers — not as a transcript, but as structured knowledge.*

**Focal element:** three mini result chips on the left, an arrow, and a
brighter version of the Frame 1 orb on the right — this time with structured
thoughts INSIDE it.

**Choreography:**
1. `t=0..0.5` — three mini result chips fade in left-to-right.
2. `t=0.6` — teal arrow fades in pointing right.
3. `t=0.8` — orb fades in, this time with an amber+teal core (`back.out` ease)
   signaling enriched state.
4. `t=1.2..` — core gently scales 1.0 ↔ 1.15 (yoyo, sine.inOut, infinite).
5. `t=1.4` — three "remembered" lines fade in centered on the orb:
   - `∙ Qdrant chosen`
   - `∙ keep replies concise`
   - `∙ ⟨name⟩ redacted`

**Emotional beat:** payoff. The same orb from Frame 1, now full and luminous.

---

## Frame 7 — "The Stack" (10s)
**Eyebrow:** `07 · the stack`
**Caption:** *All local. No external API. Zero per-token charge.*

**Focal element:** 3×2 architecture grid of named nodes with dashed edges connecting them.

**Choreography:**
1. `t=0..0.6` — seven dashed edges draw themselves into view, staggered.
2. `t=0.4..1.4` — six nodes fade+scale in, top row → bottom row:
   - **Provider Plugin** (hermes-agent, teal)
   - **MemoryDB** (SQLite + FTS5, teal)
   - **Qdrant** (vector store, violet)
   - **LM Studio** (embeddings, amber)
   - **Dreamer** (nightly cron · 03:00, violet)
   - **Retriever** (hybrid · 4 backends, teal)
3. `t=1.4..2.0` — three pill badges fade in along the bottom: `all local`,
   `no external api`, `zero per-token cost`.

**Emotional beat:** confident architecture reveal. *This whole thing fits on one screen.*

---

## Frame 8 — "The Numbers" (7s)
**Eyebrow:** `08 · live`
**Caption:** *Running on David's box, right now.*

**Focal element:** 4 large statistic cards in a row.

**Choreography:**
1. `t=0..0.5` — four cards fade up into view, staggered 120ms.
2. `t=0.4..2.0` — each value counts up from 0 to its target using a GSAP
   tween on a scratch object + `onUpdate` writeback:
   - **Facts extracted:** 0 → 84
   - **Sessions:** 0 → 6
   - **Chunks indexed:** 0 → 9
   - **Facts in Qdrant:** 0 → 76
3. `t=1.8` — bottom cadence indicator fades in:
   `● dream runs · nightly @ 03:00` (the dot pulses indefinitely).

**Emotional beat:** evidence. Real numbers off a real machine.

---

## Transitions

All inter-frame transitions are 450ms crossfades (`power2.inOut`). The
outgoing frame's animations are killed on exit and `clearProps: 'all'` resets
inline transforms so re-entering a frame (via the prev button) replays its
timeline from the start.

## HUD

- Top: brand label (`hermes-local · memory`), thin teal→amber progress bar
  spanning all 8 frames, frame counter (`01 / 08`).
- Bottom-right: prev / play-pause / next circular controls (low-key, won't
  distract from the composition).

## Production notes

- **Why GSAP and not WAAPI?** GSAP timelines compose cleanly and `clearProps`
  is invaluable for restartable scenes. Web Animations API would have worked
  but required more bookkeeping per scene.
- **Why no React?** This is a single visual document. Reactivity offers
  nothing here, and a build step would only add friction.
- **Why a CDN for GSAP?** The HTML must work standalone in a browser. A CDN
  keeps the artifact at four files. If offline distribution matters, drop
  `gsap.min.js` next to `script.js` and update the `<script src>` in
  `index.html`.

## What was *not* implemented

Everything in the brief was implemented. If extending this:

- A real audio track (VO narration + ambient pad).
- Synchronized particle systems on the data-flow lines (currently dashed
  stroke draws — clean but static after they appear).
- Live data pull (the brief explicitly prohibited this — numbers are
  hardcoded representative values).
