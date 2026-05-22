# hermes-local memory — HyperFrames showcase

A polished, animated, self-contained HTML composition that narrates how the
**hermes-local** memory system works: capture → dreamer → storage → retrieval → loop.

8 frames, ~75 seconds runtime, GSAP-driven timelines, dark theme.

## quick view

```bash
# just open it
xdg-open index.html         # linux desktop
open index.html             # macOS

# or serve it (any static server works)
python3 -m http.server 8000
# then http://localhost:8000/docs/showcase/hermes-memory-hyperframes/
```

The HTML stands alone: GSAP is loaded from a CDN, everything else is local.
No build step. No bundler. No framework.

### keyboard

- `→` / `←` — next / previous frame
- `space` — play / pause

## with the HyperFrames CLI (optional)

[HyperFrames](https://github.com/heygen-com/hyperframes) is HeyGen's open-source
framework for declaring video timelines in HTML. This file uses the
`data-frame` / `data-duration` / `data-enter` / `data-exit` conventions so it
can be served and rendered by the CLI if you have it.

```bash
# preview (dev server)
npx -y hyperframes preview

# render to MP4 (1920×1080 @ 60fps by default)
npx -y hyperframes render out.mp4 --width 1920 --height 1080
```

If the npm package is unavailable on your system, the HTML file is fully
self-playing — it does not depend on the HyperFrames runtime to animate. The
controller in `script.js` reads the same `data-duration` attributes and
advances frames on a clock.

## file map

```
hermes-memory-hyperframes/
├── README.md         ← this file
├── STORYBOARD.md     ← frame-by-frame narrative + timing + palette
├── index.html        ← the composition (8 frames, semantic markup)
├── styles.css        ← dark theme, teal/amber accents, per-frame styling
└── script.js         ← GSAP timelines + frame controller (prev/next/play)
```

## the story

| # | frame      | duration | what it shows                                            |
|---|------------|----------|----------------------------------------------------------|
| 1 | problem    | 6s       | LLM with thought bubbles popping → memory loss           |
| 2 | capture    | 10s      | Chat turns streaming into a DB, PII tokens redacting     |
| 3 | dreamer    | 12s      | Moon + 03:00 clock + LLM brain → structured fact cards   |
| 4 | storage    | 10s      | One fact → SQLite rows / Qdrant vector / JSONL event     |
| 5 | retrieval  | 12s      | Query splits into 4 backends → merged ranked results     |
| 6 | loop       | 8s       | Results return to LLM, which "remembers" them            |
| 7 | stack      | 10s      | Architecture diagram of components, badges               |
| 8 | numbers    | 7s       | Live-feeling counters: 84 facts, 6 sessions, 9 chunks, 76 in Qdrant |

Total: **~75 seconds**.

## design notes

- **Type:** Inter (display), JetBrains Mono (data / code).
- **Palette:**
  - background: `#07090c` → `#0b0f15` gradient
  - teal `#5eead4` — data flows, user signals
  - amber `#fbbf24` — system events, scores, highlights
  - violet `#a78bfa` — the dreamer / semantic axis
  - rose `#fb7185` — PII / redaction signal
- **Animation:** GSAP 3.12, `power2.inOut` defaults, mechanical-precise (no spin/bounce except
  on the final orb and number cards where a slight `back.out` adds life).
- **Transitions:** 450ms crossfade between frames, intentionally short so the
  story keeps pace.

## credits & license

- Built for the **hermes-agent** project (Nous Research).
- Composition style adapted from [HyperFrames](https://github.com/heygen-com/hyperframes)
  by HeyGen, Apache 2.0. This file is compatible with the HyperFrames data
  conventions but does not embed any HyperFrames runtime code.
- [GSAP](https://gsap.com/) loaded from CDN under its standard license.

This file is a documentation artifact for the hermes-agent repository — see the
top-level repo for its license.
