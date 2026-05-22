# Hermes Infrastructure Reference

<!--
This file contains useful infrastructure and tool reference.
Not injected as system prompt — available via session_search or direct reference.
Last updated: 2026-05-13
-->

## Local Services

### LM Studio (Embedding Server)
- **Endpoint:** http://127.0.0.1:1235/v1
- **Model:** `text-embedding-nomic-embed-text-v1.5` (for Qdrant vector embeddings)
- **Use for:** Qdrant vector embeddings only

### ComfyUI (Image & Audio Generation)
- **URL:** http://localhost:8188
- **Location:** `/disk2/ComfyUI/`
- **Venv:** `/home/dmccarty/.comfyui-venv/`
- **GPU:** NVIDIA RTX 5070 Ti Laptop (11.7GB VRAM)
- **Start:** `cd /disk2/ComfyUI && /home/dmccarty/.comfyui-venv/bin/python3 main.py --force-fp16`
- **Image output:** `/disk2/ComfyUI/output/`
- **Audio output:** `/home/dmccarty/.openclaw/workspace/scratch/`
- **Models:**
  - Flux1 Dev GGUF (Q4_0) — images, ~4-5GB VRAM
  - MusicGen HF — audio (small/medium/large)

### n8n (Workflow Automation)
- **URL:** http://127.0.0.1:5678
- **Database:** ~/.n8n/database.sqlite
- **CLI:** ~/.openclaw/n8n-local/
- **Note:** JavaScript nodes must be edited in UI (CLI import has cache issues)

### Google Workspace CLI (gws)
- **Status:** Installed (v0.16.0), fully authenticated
- **Account:** james.woods.gmail@gmail.com
- **Working:** Drive, Docs, Sheets, Gmail
- **Credentials:** `/home/dmccarty/openclaw/handoff/google-james.txt`
- **Auth file:** `/home/dmccarty/.config/gws/credentials.enc`
- **Usage:** `gws drive files list`, `gws docs documents create`, `gws gmail users messages list`

### Chrome Remote Debugging
- **Profile:** `user` — attaches to David's real Chrome
- **Flag:** `--remote-debugging-port=0`
- **Status:** Working — can navigate, click, type, take snapshots

## Credentials

**Primary location:** `/home/dmccarty/openclaw/handoff/`
- `airtable.txt` — Airtable PAT
- `openrouter.txt` — OpenRouter API key
- `google-james.txt` — Google Workspace credentials

## Airtable

- **Base:** appKeQ7RXKJdeApL1
- **Table:** tbl0lUxKfgbfkxmD9 ("Intake Pipeline")
- **Fields:** Name, Description, Status, Type, Priority, Size, Suggested Workflow

## Workspace Paths

- **Main workspace:** `~/.openclaw/workspace/`
- **Daily memory:** `~/.openclaw/workspace/memory/YYYY-MM-DD.md`
- **Long-term memory:** `~/.openclaw/workspace/MEMORY.md`
- **Scratch:** `~/.openclaw/workspace/scratch/` (temporary work)
- **Projects:** `~/.openclaw/workspace/PROJECTS/`

## Orchestrator Pattern

When handling complex tasks, delegate rather than do everything directly:

| Task Type | Delegate? | Reason |
|-----------|-----------|--------|
| Web research | ✅ Yes | Isolated context |
| File analysis (>10 files) | ✅ Yes | Parallel processing |
| Coding/implementation | ✅ Yes | Specialized skills |
| Testing/QA | ✅ Yes | Clean environment |
| Simple file read | ❌ No | Trivial, fast |
| Quick status check | ❌ No | Trivial, fast |

Use `delegate_task` to spawn subagents with:
- Clear deliverables specified
- timeoutSeconds: 0 for real work (no timeout)
- Verification instruction

## Known Issues

1. n8n JavaScript nodes must be edited in UI (CLI import cache issues)
2. Browser CDP relay port conflict — use Playwright instead