/* hermes-local memory · HyperFrames composition controller
 *
 * The story is split into 8 frames. Each frame has:
 *   - a duration (read from data-duration)
 *   - an `enter` GSAP timeline that animates its contents in
 *
 * A master controller advances between frames on a clock, while
 * also exposing prev/next/play controls.
 */

(function () {
  const stage = document.getElementById('stage');
  const frames = Array.from(stage.querySelectorAll('.frame'));
  const hudBar = document.getElementById('hudBar');
  const hudFrame = document.getElementById('hudFrame');
  const btnPrev = document.getElementById('btnPrev');
  const btnNext = document.getElementById('btnNext');
  const btnPlay = document.getElementById('btnPlay');

  const total = frames.length;
  let index = 0;
  let playing = true;
  let frameStart = 0;
  let rafId = null;

  // --- helpers ---
  const dur = (frame) => Number(frame.dataset.duration || 6);

  function setHUD() {
    hudFrame.textContent = String(index + 1).padStart(2, '0') + ' / ' + String(total).padStart(2, '0');
  }

  function tickProgress() {
    if (!playing) { rafId = requestAnimationFrame(tickProgress); return; }
    const now = performance.now();
    const elapsed = (now - frameStart) / 1000;
    const d = dur(frames[index]);
    const pct = Math.min(1, elapsed / d);
    // overall progress = (frames done + current pct) / total
    const overall = (index + pct) / total;
    hudBar.style.width = (overall * 100).toFixed(2) + '%';
    if (pct >= 1 && playing) {
      next();
    }
    // ALWAYS reschedule the rAF — auto-advance must keep ticking on every
    // subsequent frame, not stop after the first transition.
    rafId = requestAnimationFrame(tickProgress);
  }

  // --- generate decorative stars for dreamer frame ---
  (function makeStars() {
    const host = document.getElementById('stars');
    if (!host) return;
    for (let i = 0; i < 60; i++) {
      const s = document.createElement('div');
      s.className = 'star';
      s.style.top = Math.random() * 100 + '%';
      s.style.left = Math.random() * 100 + '%';
      s.style.width = s.style.height = (Math.random() * 2 + 1) + 'px';
      s.dataset.delay = (Math.random() * 2).toFixed(2);
      host.appendChild(s);
    }
  })();

  // --- enter animations per frame ---
  const enterTL = {
    problem(el) {
      const tl = gsap.timeline();
      tl.fromTo(el.querySelector('.orb'),
        { scale: 0.6, opacity: 0 },
        { scale: 1, opacity: 1, duration: 1.0, ease: 'power2.out' });
      tl.fromTo(el.querySelector('.orb-ring'),
        { rotate: 0 },
        { rotate: 360, duration: 12, ease: 'none', repeat: -1 }, 0);
      const thoughts = el.querySelectorAll('.thought');
      thoughts.forEach((t, i) => {
        tl.fromTo(t,
          { opacity: 0, y: 10, scale: 0.9 },
          { opacity: 1, y: 0, scale: 1, duration: 0.5, ease: 'power2.out' },
          0.5 + i * 0.25);
        // each thought fades out, popping like memory loss
        tl.to(t,
          { opacity: 0, y: -20, scale: 0.7, duration: 0.5, ease: 'power2.in' },
          2.0 + i * 0.3);
      });
      tl.from(el.querySelector('.caption'),
        { opacity: 0, y: 20, duration: 0.6, ease: 'power2.out' }, 0.2);
      return tl;
    },

    capture(el) {
      const tl = gsap.timeline();
      const bubbles = el.querySelectorAll('.bubble');
      bubbles.forEach((b, i) => {
        tl.to(b, { opacity: 1, y: 0, duration: 0.45, ease: 'power2.out' }, i * 0.5);
      });
      // PII redaction — color shift, then background mask
      tl.to(el.querySelectorAll('.pii'),
        { color: 'transparent', backgroundColor: 'rgba(251, 113, 133, 0.25)', duration: 0.4, ease: 'power2.inOut' },
        2.2);
      // flow arrow draw
      tl.fromTo(el.querySelector('.flow-arrow svg'),
        { opacity: 0, x: -20 },
        { opacity: 1, x: 0, duration: 0.6, ease: 'power2.out' }, 1.0);
      // db pulse
      const cyls = el.querySelectorAll('.db-cyl');
      cyls.forEach((c, i) => {
        tl.fromTo(c,
          { opacity: 0, y: 10 },
          { opacity: 1, y: 0, duration: 0.4 }, 1.4 + i * 0.15);
      });
      // soft glow pulse on db
      tl.to(cyls, {
        boxShadow: '0 0 30px rgba(94, 234, 212, 0.5)',
        duration: 0.6,
        yoyo: true,
        repeat: 3
      }, 2.5);
      tl.from(el.querySelector('.caption'),
        { opacity: 0, y: 20, duration: 0.6 }, 0.2);
      return tl;
    },

    dreamer(el) {
      const tl = gsap.timeline();
      // stars twinkle
      const stars = el.querySelectorAll('.star');
      stars.forEach(s => {
        const d = Number(s.dataset.delay || 0);
        gsap.fromTo(s, { opacity: 0 }, {
          opacity: 0.7, duration: 1.2, delay: d, ease: 'power1.inOut',
          yoyo: true, repeat: -1
        });
      });
      tl.fromTo(el.querySelector('.moon'),
        { opacity: 0, scale: 0.6, y: -20 },
        { opacity: 1, scale: 1, y: 0, duration: 1.0, ease: 'power2.out' });
      tl.fromTo(el.querySelector('.moon-glow'),
        { opacity: 0 }, { opacity: 1, duration: 1.4, yoyo: true, repeat: -1, ease: 'sine.inOut' }, 0);
      tl.fromTo(el.querySelector('.clock-face'),
        { opacity: 0, scale: 0.8 },
        { opacity: 1, scale: 1, duration: 0.7, ease: 'power2.out' }, 0.4);
      tl.fromTo(el.querySelector('.brain-core'),
        { opacity: 0, scale: 0.6 },
        { opacity: 1, scale: 1, duration: 0.8, ease: 'power2.out' }, 0.8);
      tl.fromTo(el.querySelector('.brain-pulse'),
        { opacity: 0.6, scale: 1 },
        { opacity: 0, scale: 1.6, duration: 1.8, ease: 'power2.out', repeat: -1 }, 1.0);

      const cards = el.querySelectorAll('.card');
      cards.forEach((c, i) => {
        tl.fromTo(c,
          { opacity: 0, x: -60, scale: 0.9 },
          { opacity: 1, x: 0, scale: 1, duration: 0.6, ease: 'power3.out' },
          2.0 + i * 0.6);
        // gentle float
        gsap.to(c, {
          y: '+=6',
          duration: 2.4 + i * 0.2,
          yoyo: true,
          repeat: -1,
          ease: 'sine.inOut',
          delay: 3 + i * 0.2
        });
      });
      tl.from(el.querySelector('.caption'),
        { opacity: 0, y: 20, duration: 0.6 }, 0.2);
      return tl;
    },

    storage(el) {
      const tl = gsap.timeline();
      tl.fromTo(el.querySelector('.src-card'),
        { opacity: 0, y: -20, scale: 0.9 },
        { opacity: 1, y: 0, scale: 1, duration: 0.6, ease: 'power2.out' });

      // draw fan lines
      const fanLines = el.querySelectorAll('.fan-line');
      fanLines.forEach((l, i) => {
        const len = l.getTotalLength();
        gsap.set(l, { strokeDasharray: len, strokeDashoffset: len, opacity: 1 });
        tl.to(l, { strokeDashoffset: 0, duration: 0.7, ease: 'power2.inOut' }, 0.6 + i * 0.1);
      });

      const tiers = el.querySelectorAll('.tier');
      tiers.forEach((t, i) => {
        tl.fromTo(t,
          { opacity: 0, y: 30 },
          { opacity: 1, y: 0, duration: 0.6, ease: 'power2.out' },
          1.2 + i * 0.2);
      });

      // highlight the matching vec dot
      tl.fromTo(el.querySelector('.vd6'),
        { scale: 0.5, opacity: 0 },
        { scale: 1, opacity: 1, duration: 0.5, ease: 'back.out(2)' }, 2.4);
      tl.to(el.querySelector('.vd6'), {
        boxShadow: '0 0 30px var(--amber)',
        duration: 0.8,
        yoyo: true,
        repeat: -1,
        ease: 'sine.inOut'
      }, 2.8);

      // row cells stagger in
      tl.fromTo(el.querySelectorAll('.row-cell'),
        { opacity: 0 },
        { opacity: 1, duration: 0.3, stagger: 0.05 }, 2.0);

      // jsonl typewriter feel
      tl.fromTo(el.querySelectorAll('.jl'),
        { opacity: 0, x: -10 },
        { opacity: 1, x: 0, duration: 0.4, stagger: 0.15 }, 2.0);

      tl.from(el.querySelector('.caption'),
        { opacity: 0, y: 20, duration: 0.6 }, 0.2);
      return tl;
    },

    retrieval(el) {
      const tl = gsap.timeline();
      tl.fromTo(el.querySelector('.query-card'),
        { opacity: 0, y: -30, scale: 0.9 },
        { opacity: 1, y: 0, scale: 1, duration: 0.6, ease: 'power2.out' });

      const splitLines = el.querySelectorAll('.splitter .split-line');
      splitLines.forEach((l, i) => {
        const len = l.getTotalLength();
        gsap.set(l, { strokeDasharray: len, strokeDashoffset: len, opacity: 1 });
        tl.to(l, { strokeDashoffset: 0, duration: 0.6, ease: 'power2.inOut' }, 0.6 + i * 0.08);
      });

      const backends = el.querySelectorAll('.backend');
      backends.forEach((b, i) => {
        tl.fromTo(b,
          { opacity: 0, y: 20, scale: 0.95 },
          { opacity: 1, y: 0, scale: 1, duration: 0.5, ease: 'power2.out' },
          1.2 + i * 0.12);
      });

      // pulse the score on each backend
      backends.forEach((b, i) => {
        const score = b.querySelector('.be-score');
        tl.fromTo(score,
          { scale: 1.4, textShadow: '0 0 12px currentColor' },
          { scale: 1, textShadow: '0 0 0px currentColor', duration: 0.6, ease: 'power2.out' },
          1.6 + i * 0.12);
      });

      const mergeLines = el.querySelectorAll('.merger .split-line');
      mergeLines.forEach((l, i) => {
        const len = l.getTotalLength();
        gsap.set(l, { strokeDasharray: len, strokeDashoffset: len, opacity: 1 });
        tl.to(l, { strokeDashoffset: 0, duration: 0.5, ease: 'power2.inOut' }, 2.4 + i * 0.05);
      });

      const results = el.querySelectorAll('.result-row');
      results.forEach((r, i) => {
        tl.fromTo(r,
          { opacity: 0, x: -20 },
          { opacity: 1, x: 0, duration: 0.5, ease: 'power2.out' },
          3.0 + i * 0.2);
      });

      tl.from(el.querySelector('.caption'),
        { opacity: 0, y: 20, duration: 0.6 }, 0.2);
      return tl;
    },

    loop(el) {
      const tl = gsap.timeline();
      const mini = el.querySelectorAll('.result-mini');
      mini.forEach((m, i) => {
        tl.fromTo(m,
          { opacity: 0, x: -20 },
          { opacity: 1, x: 0, duration: 0.4, ease: 'power2.out' },
          i * 0.15);
      });
      tl.fromTo(el.querySelector('.loop-arrow'),
        { opacity: 0, x: -20 },
        { opacity: 1, x: 0, duration: 0.5, ease: 'power2.out' }, 0.6);
      tl.fromTo(el.querySelector('.orb-remembered'),
        { opacity: 0, scale: 0.6 },
        { opacity: 1, scale: 1, duration: 0.8, ease: 'back.out(1.4)' }, 0.8);
      tl.fromTo(el.querySelector('.orb-thought'),
        { opacity: 0, scale: 0.8 },
        { opacity: 1, scale: 1, duration: 0.5, ease: 'power2.out' }, 1.4);
      // glow pulse
      tl.to(el.querySelector('.orb-core-bright'), {
        scale: 1.15, duration: 0.8, yoyo: true, repeat: -1, ease: 'sine.inOut'
      }, 1.2);
      tl.from(el.querySelector('.caption'),
        { opacity: 0, y: 20, duration: 0.6 }, 0.2);
      return tl;
    },

    stack(el) {
      const tl = gsap.timeline();
      // edges draw first
      const edges = el.querySelectorAll('.edge');
      edges.forEach((e, i) => {
        const len = e.getTotalLength();
        gsap.set(e, { strokeDasharray: len, strokeDashoffset: len });
        tl.to(e, { strokeDashoffset: 0, duration: 0.5, ease: 'power2.inOut' }, i * 0.08);
      });
      const nodes = el.querySelectorAll('.node');
      nodes.forEach((n, i) => {
        tl.fromTo(n,
          { opacity: 0, scale: 0.85, y: 10 },
          { opacity: 1, scale: 1, y: 0, duration: 0.5, ease: 'power2.out' },
          0.4 + i * 0.1);
      });
      const badges = el.querySelectorAll('.badge');
      badges.forEach((b, i) => {
        tl.fromTo(b,
          { opacity: 0, y: 10 },
          { opacity: 1, y: 0, duration: 0.4, ease: 'power2.out' },
          1.4 + i * 0.15);
      });
      tl.from(el.querySelector('.caption'),
        { opacity: 0, y: 20, duration: 0.6 }, 0.2);
      return tl;
    },

    numbers(el) {
      const tl = gsap.timeline();
      const cards = el.querySelectorAll('.num-card');
      cards.forEach((c, i) => {
        tl.fromTo(c,
          { opacity: 0, y: 30 },
          { opacity: 1, y: 0, duration: 0.5, ease: 'power2.out' },
          i * 0.12);
      });
      // count up
      cards.forEach((c, i) => {
        const valEl = c.querySelector('.num-value');
        const target = Number(valEl.dataset.target);
        const obj = { v: 0 };
        tl.to(obj, {
          v: target,
          duration: 1.6,
          ease: 'power2.out',
          onUpdate() { valEl.textContent = Math.round(obj.v); }
        }, 0.4 + i * 0.12);
      });
      tl.fromTo(el.querySelector('.cadence'),
        { opacity: 0, y: 10 },
        { opacity: 1, y: 0, duration: 0.6 }, 1.8);
      tl.from(el.querySelector('.caption'),
        { opacity: 0, y: 20, duration: 0.6 }, 0.2);
      return tl;
    }
  };

  // active timelines per frame for cleanup
  const liveTLs = new Map();

  function enter(i) {
    const frame = frames[i];
    frame.classList.add('active');
    const key = frame.dataset.frame;
    const fn = enterTL[key];
    if (fn) {
      const tl = fn(frame);
      liveTLs.set(frame, tl);
    }
    frameStart = performance.now();
    setHUD();
  }

  function exit(i) {
    const frame = frames[i];
    const tl = liveTLs.get(frame);
    if (tl) {
      tl.kill();
      liveTLs.delete(frame);
    }
    // kill any leftover tweens on descendants
    gsap.killTweensOf(frame.querySelectorAll('*'));
    frame.classList.remove('active');
    // reset inline transforms so re-entry animates from scratch
    gsap.set(frame.querySelectorAll('*'), { clearProps: 'all' });
  }

  function go(to) {
    to = ((to % total) + total) % total;
    if (to === index) return;
    // crossfade
    const from = index;  // capture BEFORE reassign so the closure has the right value
    const out = frames[from];
    const into = frames[to];
    gsap.to(out, { opacity: 0, duration: 0.45, ease: 'power2.inOut', onComplete: () => exit(from) });
    gsap.fromTo(into,
      { opacity: 0 },
      { opacity: 1, duration: 0.45, ease: 'power2.inOut',
        onStart: () => { into.style.visibility = 'visible'; enter(to); } });
    index = to;
    // Reset frame timer immediately so the rAF clock doesn't see a huge
    // elapsed delta and re-fire next() on the very next tick.
    frameStart = performance.now();
  }

  function next() { go(index + 1); }
  function prev() { go(index - 1); }
  function togglePlay() {
    playing = !playing;
    btnPlay.textContent = playing ? '⏸' : '▶';
    if (playing) {
      // reset frame start so progress doesn't jump
      const dt = (performance.now() - frameStart) / 1000;
      const d = dur(frames[index]);
      frameStart = performance.now() - Math.min(dt, d - 0.1) * 1000;
    }
  }

  btnNext.addEventListener('click', next);
  btnPrev.addEventListener('click', prev);
  btnPlay.addEventListener('click', togglePlay);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') next();
    else if (e.key === 'ArrowLeft') prev();
    else if (e.key === ' ' || e.key === 'k') { e.preventDefault(); togglePlay(); }
  });

  // start
  enter(0);
  rafId = requestAnimationFrame(tickProgress);

  // ---- HyperFrames runtime compatibility ----
  // The HyperFrames CLI looks for `window.__timelines[compositionId]` and
  // expects a GSAP timeline it can scrub. We expose a no-op master timeline
  // whose duration equals the sum of frame durations so the CLI's render and
  // inspect tooling can find us. Animation is still driven by our own
  // controller above when the file is opened standalone.
  if (typeof window !== 'undefined') {
    window.__timelines = window.__timelines || {};
    const master = gsap.timeline({ paused: true });
    const totalDuration = frames.reduce((acc, f) => acc + dur(f), 0);
    // pad master so its duration matches the composition root
    master.to({}, { duration: totalDuration });
    window.__timelines['hermes-memory'] = master;
  }
})();
