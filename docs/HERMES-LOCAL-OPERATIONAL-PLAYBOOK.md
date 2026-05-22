# HERMES-LOCAL Operational Playbook

**Branch:** `recovery/phase-1-5-restore`
**Audience:** Operator flipping `memory.provider` from `holographic` → `hermes-local`
**Last verified:** 2026-05-22

This playbook walks the operator through a controlled switch to the hermes-local
memory provider, validates the system after switching, monitors logs and health,
manually exercises the dreamer, and rolls back in ≤30s if anything misbehaves.

Every section is copy-paste runnable. Expected output is shown after each command.

---

## Table of Contents

1. [Pre-switch verification](#1-pre-switch-verification)
2. [The switch procedure](#2-the-switch-procedure)
3. [Post-switch smoke test (interactive)](#3-post-switch-smoke-test-interactive)
4. [Log monitoring strategy](#4-log-monitoring-strategy)
5. [Health monitoring](#5-health-monitoring)
6. [Dreamer validation](#6-dreamer-validation)
7. [Canary rollback procedure](#7-canary-rollback-procedure)
8. [Known issues at switch time](#8-known-issues-at-switch-time)
9. [Failure modes — symptom → cause → fix](#9-failure-modes--symptom--cause--fix)

---

## 1. Pre-switch verification

Run these **while still on `holographic`**. Goal: confirm the hermes-local
plumbing is healthy and ready to take over before you touch `config.yaml`.

### 1a. Activate venv
```bash
cd ~/.hermes/hermes-agent
source venv/bin/activate
```

### 1b. Run the bundled smoke test
```bash
bash scripts/hermes_local_smoke_test.sh
```
**Success looks like:**
```
✓ module loads, ABC fully implemented
✓ is_available()=True with HERMES_MEMORY_PROVIDER=hermes-local
✓ all required tables present
✓ write persisted; query exercised (read path may flag known FTS5 bug)
✓ system_prompt_block() returned valid header + stats
✓ prefetch did not crash (graceful with backends partial)
SMOKE TEST PASSED
```
If any `✗` line appears, **do not switch**. Fix the failure first (see §9).

### 1c. Confirm DB schema + fact count
```bash
python3 -c "
import sqlite3
c = sqlite3.connect('$HOME/.hermes/memory/index/memory.sqlite')
for t in ('facts','turns','sessions','decisions','open_questions','dream_runs'):
    n = c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'{t:18s} {n}')"
```
**Expected:**
```
facts              76         # or higher, depending on prior smoke runs
turns              2
sessions           1
decisions          0
open_questions     0
dream_runs         0
```

### 1d. Confirm dreamer is importable (not crashed at module level)
```bash
python3 -c "from hermes_memory_core.dream import DreamWorker; print('DreamWorker OK')"
```
**Expected:** `DreamWorker OK`

### 1e. Confirm hermes-local plugin is listed and "available"
```bash
hermes memory status
```
**Expected (excerpt):**
```
Provider:  holographic
Installed plugins:
  • hermes-local  (no setup needed)
  • holographic   (local) ← active
```

### 1f. Confirm the recovered slash subcommands work
```bash
python3 -m hermes_cli.memory health 2>&1 | head -15
```
**Expected (excerpt):**
```
=== Hermes Local Memory — Health ===
hermes_home : /home/<you>/.hermes
provider     : holographic            # still holographic — that's correct pre-switch
qdrant       : connected              # or "unreachable" if Qdrant is down
lms          : unreachable            # known — LMS endpoint is currently offline
```

---

## 2. The switch procedure

### 2a. Back up `config.yaml` first
```bash
cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak-$(date +%Y%m%d-%H%M%S)
ls -la ~/.hermes/config.yaml.bak-*
```

### 2b. Back up the SQLite memory DB (safety net for rollback)
```bash
hermes memory backup
```
**Expected:** prints path to a new `~/.hermes/memory/backups/<timestamp>.tar.gz`.

### 2c. Edit `~/.hermes/config.yaml`
Add or update the `memory:` block. The key is `memory.provider`:
```yaml
memory:
  provider: hermes-local
```
Quick one-liner using Python (safe — preserves existing YAML structure):
```bash
python3 - <<'PY'
import yaml, pathlib
p = pathlib.Path.home() / ".hermes" / "config.yaml"
cfg = yaml.safe_load(p.read_text()) or {}
cfg.setdefault("memory", {})["provider"] = "hermes-local"
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("set memory.provider = hermes-local")
PY
```

### 2d. Verify the edit took
```bash
grep -A1 '^memory:' ~/.hermes/config.yaml
```
**Expected:**
```
memory:
  provider: hermes-local
```

### 2e. Confirm the agent now sees it
```bash
hermes memory status | head -5
```
**Expected:**
```
Memory status
──────────────────
  Built-in:  always active
  Provider:  hermes-local
```

---

## 3. Post-switch smoke test (interactive)

This is the 5-10 minute manual validation in `hermes chat`. Keep §4's log
watcher running in another terminal.

### 3a. Start a fresh chat
```bash
hermes chat
```

### 3b. Confirm system prompt block injection
In the chat, type:
```
/dump system
```
**Expected:** somewhere in the system prompt you should see a block like:
```
# Hermes Local Memory
Active. 77 facts, 0 decisions, 0 open questions.
Use memory_query to search ...
```
Fact count should match what `sqlite3 ~/.hermes/memory/index/memory.sqlite "SELECT COUNT(*) FROM facts;"` returns.

### 3c. Test `memory_query` (read path)
Ask:
> "What do you remember about the hermes-local smoke test?"

**Success:** the model invokes `memory_query`, returns at least one hit
referencing the smoketest marker (if step 4 of the smoke script ran today).
If `memory_query` returns zero results with the FTS5 bug present, ask:
> "Use mode='facts' to query for 'smoke'."

**Expected:** at least one hit (mode=facts bypasses the FTS5 read path).

### 3d. Test `memory_write` (write path)
Ask:
> "Please record a fact: 'David verified hermes-local switch on YYYY-MM-DD'. Use memory_write with type='fact', scope='general', project='ops', source_ref='playbook:section3d'."

**Success:** the model calls `memory_write`, returns `{"status":"ok", "id":"fact:..."}`.
Confirm at the shell:
```bash
sqlite3 ~/.hermes/memory/index/memory.sqlite \
  "SELECT fact_id, substr(fact_text,1,60) FROM facts WHERE source_refs_json LIKE '%playbook:section3d%';"
```
**Expected:** one row with your verification fact.

### 3e. Test `/new` (narrative thread)
In the chat:
```
/new
```
**Success:** new session starts. Then ask:
> "What was I just working on?"

The model should have access to a narrative-thread snippet from the previous
session (focus + recent exchanges). Look for log line `on_session_switch` in
the log watcher.

### 3f. Test `/resume` (context rehydration)
Exit chat (`/quit`), then:
```bash
hermes chat --continue
```
**Expected:** the agent reconnects to the most recent session. Type:
> "Summarize what we did in the last 3 turns."

The summary should reference the smoketest / memory_write activity from §3d.

---

## 4. Log monitoring strategy

### 4a. One-shot: tail with built-in filters
Start in a separate terminal **before** the switch:
```bash
bash scripts/hermes_local_log_watch.sh
```
This wraps `hermes logs --follow` and applies grep + colorization:
- **Green:** good signals (`Memory provider … initialized`, `narrative thread`, `Dream run complete`)
- **Yellow:** warnings (Qdrant/LMS offline, fallback, degraded)
- **Red:** errors (ERROR / CRITICAL / Traceback / fts5_search failed)

### 4b. Manual filter recipes
Just errors since the switch:
```bash
hermes logs --follow --level ERROR --since 10m
```
Memory-related events only (last hour):
```bash
hermes logs --since 1h | grep -iE "memory|hermes-local|qdrant|lms|fts5|dream"
```
Per-session debugging:
```bash
hermes logs --follow --session <session_id>
```
Errors log (separate file):
```bash
hermes logs errors --follow
```

### 4c. Good signal patterns (you want to see these)
| Pattern                                              | Meaning                                                     |
|------------------------------------------------------|-------------------------------------------------------------|
| `[hermes-local] Memory provider registered`          | Plugin discovered at startup                                |
| `[hermes-local] Initialized for session=<id>`        | Per-session init complete                                   |
| `[hermes-local] on_session_switch: <new>`            | `/new` or `/resume` fired the narrative-thread hook         |
| `MemoryStore connection closed`                      | Clean shutdown                                              |
| `Dream run complete` / `dream_status=done`           | Dreamer ran without crashing                                |
| `qdrant : connected`                                 | Qdrant reachable                                            |

### 4d. Bad signal patterns (immediate attention)
| Pattern                                              | Action                                                      |
|------------------------------------------------------|-------------------------------------------------------------|
| `ERROR` / `CRITICAL` / `Traceback`                   | Read full stack; check §9                                   |
| `handle_tool_call(memory_*) raised`                  | A tool handler crashed — capture args, check §9             |
| `sync_turn failed`                                   | Turn capture broken — DB may be locked or schema mismatched |
| `fts5_search: query failed -- no such column: project` | **Known** — FTS5 read-path bug, parallel agent fixing      |
| `MemoryProvider … failed to load`                    | Plugin import broke — rollback                              |

### 4e. Warning patterns (degraded but not fatal)
| Pattern                                              | Meaning                                                     |
|------------------------------------------------------|-------------------------------------------------------------|
| `LMS.*unreachable` / `connection refused`            | Embedding endpoint offline — semantic search disabled       |
| `Qdrant.*offline` / `Collection ... doesn't exist`   | Hybrid search disabled — falls back to FTS5                 |
| `hybrid prefetch failed: ... — falling back to FTS5` | Expected when Qdrant or LMS is down                         |

---

## 5. Health monitoring

### 5a. `hermes memory health` (slash subcommand)
```bash
python3 -m hermes_cli.memory health
```
**Expected (healthy hermes-local):**
```
hermes_home : /home/<you>/.hermes
config       : /home/<you>/.hermes/config.yaml
provider     : hermes-local
redaction    : OK   (or 'unavailable' — known API drift, see §8)
jsonl root   : /home/<you>/.hermes/memory/raw (...)
sqlite       : OK
qdrant       : connected
lms          : unreachable     # known — LMS offline at switch time
```

### 5b. `hermes memory db` (schema introspection)
```bash
python3 -m hermes_cli.memory db
```
**Expected:** lists tables, row counts, and schema_version.

### 5c. `hermes memory ls-sessions`
```bash
python3 -m hermes_cli.memory ls-sessions
```
**Expected:** at least one session row (your current chat) with timestamps.

### 5d. Metrics file
```bash
cat ~/.hermes/memory/metrics.json 2>/dev/null || echo "no metrics yet — written on first dream run"
```
**Expected fields when present:**
```json
{
  "captured_turns_24h": 12,
  "chunks_indexed_24h": 12,
  "facts_total": 77,
  "qdrant_points": 0,
  "last_dream_run_at": "2026-05-22T...",
  "redactions_24h": 0
}
```
If the file is missing: it's only created when a dream run completes — see §6.

### 5e. Quick raw-row sanity check
```bash
sqlite3 ~/.hermes/memory/index/memory.sqlite <<'SQL'
SELECT 'facts', COUNT(*) FROM facts;
SELECT 'turns_today', COUNT(*) FROM turns WHERE timestamp >= date('now');
SELECT 'last_session', session_id, started_at FROM sessions ORDER BY started_at DESC LIMIT 1;
SQL
```

---

## 6. Dreamer validation

The dreamer extracts facts/decisions/open_questions from captured turns. It
runs on a schedule, but you should trigger it once manually after the switch.

### 6a. Manual one-shot run (scope = since_last)
```bash
python3 -m hermes_memory_core.dream --once --scope since_last 2>&1 | tee /tmp/dream-$(date +%s).log
```
**Success looks like:**
```
[dream] starting run scope=since_last
[dream] reading turns since checkpoint=...
[dream] extracted N facts / M decisions / K open_questions
[dream] wrote report: /home/<you>/.hermes/memory/dreams/2026-05-22-HHMM.md
[dream] run complete
```

### 6b. Inspect the report
```bash
ls -lt ~/.hermes/memory/dreams/ | head -5
cat "$(ls -t ~/.hermes/memory/dreams/*.md | head -1)"
```
**Expected:** Markdown file with sections for new facts, decisions, open questions, contradictions, and a summary.

### 6c. Confirm facts landed
```bash
sqlite3 ~/.hermes/memory/index/memory.sqlite \
  "SELECT COUNT(*), MAX(created_at) FROM facts;"
```
The count should increase by the number extracted in §6a; `MAX(created_at)` should be ≈ now.

### 6d. Confirm dream_runs row was written
```bash
sqlite3 ~/.hermes/memory/index/memory.sqlite \
  "SELECT run_id, scope, status, started_at, ended_at FROM dream_runs ORDER BY started_at DESC LIMIT 3;"
```
**Expected:** at least one row with `status='done'`.

### 6e. Failure mode for dreamer
If the dream module crashes mid-run, you'll see:
- `[dream] run failed: <exception>` in the script output
- A `dream_runs` row with `status='failed'`
- No new `*.md` report

In that case: capture the traceback, **rollback** (§7), and file an issue. The
dreamer touching `hermes_memory_core/*` is being actively iterated on; runtime
errors in it are not blockers for keeping `hermes-local` as the read/write provider as long as captured turns persist.

---

## 7. Canary rollback procedure

If anything in §3-§6 misbehaves, revert to `holographic` in ≤30 seconds.

### 7a. Revert config.yaml
```bash
# Option A: restore from the .bak you made in §2a
cp ~/.hermes/config.yaml.bak-<timestamp> ~/.hermes/config.yaml

# Option B: programmatic (no .bak needed)
python3 - <<'PY'
import yaml, pathlib
p = pathlib.Path.home() / ".hermes" / "config.yaml"
cfg = yaml.safe_load(p.read_text()) or {}
if "memory" in cfg and "provider" in cfg["memory"]:
    cfg["memory"].pop("provider")
    if not cfg["memory"]:
        cfg.pop("memory")
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("reverted: memory.provider unset (default = holographic)")
PY
```

### 7b. Verify the rollback
```bash
hermes memory status | head -5
```
**Expected:**
```
Memory status
──────────────────
  Provider:  holographic
```

### 7c. Confirm in a fresh chat
```bash
hermes chat
# in chat: /dump system | grep -A2 "Memory"
```
You should now see the holographic memory block, not the hermes-local one.

### 7d. (Optional) restore the SQLite DB
The hermes-local DB is **additive**, not destructive — rolling back to
holographic does not require restoring memory.sqlite. But if you want to wipe
the writes that occurred during the canary:
```bash
tar -xzf ~/.hermes/memory/backups/<timestamp>.tar.gz -C ~/.hermes/memory/
```

Total wall-clock rollback time: **<30s**.

---

## 8. Known issues at switch time

These are **acknowledged limitations** that are NOT blockers for the switch.
Document them in your switch report so reviewers don't flag them as regressions.

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | **FTS5 read path bug** — `fts5_search: query failed -- no such column: project` | `memory_query` modes that hit FTS5 may return zero results; `mode='facts'`/`'decisions'` still works because they query base tables directly | Being fixed by a parallel subagent on `recovery/phase-1-5-restore`. Do not touch `hermes_memory_core/search/fts5.py`. |
| 2 | **LMS embedding endpoint offline** (`http://192.168.2.105:1235`) | Hybrid + semantic search degrade to keyword-only; prefetch returns empty when keyword search also misses | Being fixed by a separate parallel subagent. Tracked separately. |
| 3 | **Qdrant collection may be missing** (`hermes_memory_chunks_nomic_v15`) | Hybrid search degrades to FTS5 fallback. Logs will show `Qdrant search failed: ... 404`. | Run `python3 plugins/memory/hermes-local/cli.py qdrant-init` to create collections (only useful once LMS is back). |
| 4 | **9 pre-existing test failures** in `tests/.../test_hermes_local_plugin.py` (8 redaction API drift + 1 `dreams/prompts/` dir missing) | Test suite shows red, but functional paths work | Tracked; not blockers for the switch. |
| 5 | **`redaction : unavailable`** in `hermes memory health` (`cannot import name 'Redactor'`) | `_handle_memory_write` still works because it uses `redact()` (the lower-level function). Only the health-summary line is wrong. | Cosmetic; will be fixed alongside #4. |

---

## 9. Failure modes — symptom → cause → fix

| Symptom (what you see) | Likely cause | Fix command |
|------------------------|--------------|-------------|
| `hermes memory status` still says `Provider: holographic` after editing config | YAML indent wrong, or hermes was reading from a different `HERMES_HOME` | `grep -A1 '^memory:' ~/.hermes/config.yaml` + `echo $HERMES_HOME` |
| Chat shows holographic system-prompt block after switch | Same as above; or the chat session was started before the config edit | `/quit` and `hermes chat` again |
| `is_available()` returns `False` even with `provider: hermes-local` | DB exists but `facts` table missing (schema not initialized) | `python3 -m hermes_cli.memory db` then if needed `python3 plugins/memory/hermes-local/cli.py db init` |
| `sync_turn failed: database is locked` in logs | Concurrent dreamer + chat write contending on SQLite | Wait 5s and retry; if persistent: `pkill -f "hermes_memory_core.dream"` then retry |
| `handle_tool_call(memory_query) raised: ... no such column: project` | The FTS5 read-path bug (§8 #1) | Use `mode='facts'` / `'decisions'` until parallel agent's fix lands |
| `Qdrant search failed: ... 404` | Qdrant collection not created (§8 #3) | Tolerable — keyword fallback runs. Optional: `python3 plugins/memory/hermes-local/cli.py qdrant-init` (only once LMS is back) |
| `LMS connection error — retrying` (3x) | LMS embedding endpoint offline (§8 #2) | Tolerable — wait for parallel agent. Or set `embedding.endpoint` to a local fallback. |
| `Dream run failed: ...` | Dreamer crashed mid-run | Capture traceback to `/tmp/dream-fail.log`; rollback (§7) if it loops |
| `MemoryProvider … failed to load` at startup | Plugin import broke | Immediate rollback (§7). Then `python3 -c "import importlib; importlib.import_module('plugins.memory.hermes-local')"` to capture the import error |
| System prompt block says "Empty store" but `facts` table has rows | `system_prompt_block()` is reading a different DB | `echo $HERMES_HOME` — should match `~/.hermes` |
| `memory_write` returns `{"status":"ok"}` but row not in DB | Path mismatch between `MemoryStore` singleton and verification query | `python3 -c "from hermes_memory_core.store.sqlite import get_memory_store; print(get_memory_store()._db_path)"` and compare against the file you're querying |
| `narrative thread` log line never appears on `/new` | `agent_ref` was not passed at initialize time | Check `on_session_switch` log — `reset=True` means narrative is intentionally cleared. For `/resume`, the parent session must exist in `sessions` table |
| Metrics file (`~/.hermes/memory/metrics.json`) never appears | Metrics writer is fired by the dream run; no dream has run yet | Run §6a manually |

---

## Appendix A: Quick reference card

```bash
# Pre-switch
bash scripts/hermes_local_smoke_test.sh

# Switch
cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak-$(date +%Y%m%d-%H%M%S)
hermes memory backup
# edit ~/.hermes/config.yaml -> memory.provider: hermes-local
hermes memory status   # confirm Provider: hermes-local

# Watch
bash scripts/hermes_local_log_watch.sh           # term 1
hermes chat                                       # term 2

# Validate
python3 -m hermes_cli.memory health
python3 -m hermes_cli.memory db
python3 -m hermes_cli.memory ls-sessions
cat ~/.hermes/memory/metrics.json

# Dream
python3 -m hermes_memory_core.dream --once --scope since_last
ls -lt ~/.hermes/memory/dreams/ | head -3

# Rollback (≤30s)
# restore .bak OR: yaml-edit to remove memory.provider
hermes memory status   # confirm Provider: holographic
```
