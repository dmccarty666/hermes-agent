# LMS Embedding Endpoint Setup (spark2:1235)

This documents the embedding endpoint used by `hermes_memory_core.embed.LMSClient`
(default: `http://192.168.2.105:1235`).

## Summary

| Item                  | Value                                                |
| --------------------- | ---------------------------------------------------- |
| Host                  | spark2 (`192.168.2.105`)                             |
| Public port           | **1235**                                             |
| Backed by             | LM Studio server on `127.0.0.1:1234` (via socat fwd) |
| Model                 | `text-embedding-nomic-embed-text-v1.5`               |
| Architecture          | Nomic BERT (GGUF, ~84 MB)                            |
| Embedding dimension   | **768**                                              |
| API                   | OpenAI-compatible `POST /v1/embeddings`              |

## Architecture Note: Why a Port Forward?

LM Studio's CLI daemon (`llmster`) supports **only one OpenAI-compatible
server, on one port**, serving all loaded models (chat and embedding) on
that single port. The user-supplied "second LMS server on 1235" pattern is
not actually possible with vanilla `lms`.

To keep the convention that `hermes-local` calls `http://...:1235` while
also keeping the Qwen chat model reachable at `:1234`, port **1235 is
forwarded to 1234** by a tiny `socat` instance managed as a systemd user
service. Both ports hit the same LMS server; the *model* is selected by the
`"model"` field in each request body.

```
client → spark2:1235 ──socat──▶ spark2:1234 (llmster, OpenAI API)
                                  ├─ qwen/qwen3.6-35b-a3b        (chat)
                                  └─ text-embedding-nomic-embed-text-v1.5  (embed, 768d)
```

## Files installed on spark2

- `~/.config/systemd/user/lms-embed-proxy.service` — socat 0.0.0.0:1235 → 127.0.0.1:1234
- User linger enabled (`loginctl enable-linger dmccarty`) so the service
  survives reboots without an interactive login.

The LMS daemon itself is started by `~/.lmstudio/lmstudio-server-start.sh`
(pre-existing; not modified by this setup).

## Day-to-Day Operations

All commands run **on spark2** unless noted.

### Health check (from anywhere on the LAN)

```bash
curl -sX POST http://192.168.2.105:1235/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"text-embedding-nomic-embed-text-v1.5","input":"hello"}' \
  | python3 -c "import sys,json; v=json.load(sys.stdin)['data'][0]['embedding']; print(len(v),'dims')"
# expect: 768 dims
```

### Is the embedding model loaded?

```bash
lms ps
# expect to see both qwen/qwen3.6-35b-a3b AND text-embedding-nomic-embed-text-v1.5
```

If the embedding model is missing (e.g. after a reboot or `lms unload`):

```bash
lms load text-embedding-nomic-embed-text-v1.5 -y
```

### Is the 1235 proxy up?

```bash
systemctl --user status lms-embed-proxy.service
ss -tlnp | grep 1235     # should show socat listening on 0.0.0.0:1235
```

### Restart the proxy

```bash
systemctl --user restart lms-embed-proxy.service
```

### Stop / disable the proxy

```bash
systemctl --user stop lms-embed-proxy.service
systemctl --user disable lms-embed-proxy.service
```

## Recovery from a Cold Boot

If spark2 reboots and nothing on `:1235` answers:

1. **LMS server itself:**
   `bash ~/.lmstudio/lmstudio-server-start.sh &`
   (or whatever supervisor is configured for it — `pgrep -af llmster`
   should show the daemon.)
2. **Load the embedding model:**
   `lms load text-embedding-nomic-embed-text-v1.5 -y`
3. **Proxy** should auto-start via systemd user linger. If not:
   `systemctl --user start lms-embed-proxy.service`
4. Verify with the curl one-liner above.

## Client-side reference

`hermes_memory_core/embed/__init__.py` defaults:

```python
LMS_ENDPOINT = "http://192.168.2.105:1235"
EMBED_MODEL  = "text-embedding-nomic-embed-text-v1.5"
EMBED_DIM    = 768
```

Smoke test from the dev box:

```bash
cd /home/dmccarty/.hermes/hermes-agent
source venv/bin/activate
python3 -c "
from hermes_memory_core.embed import LMSClient
c = LMSClient()
v = c.embed('hello')
print('dim:', len(v))
"
# expect: dim: 768
```

## Caveats

- **`LMSClient.health_check()` cosmetic quirk:** the `model` field returned
  is whatever `/v1/models` lists first (currently `qwen/qwen3.6-35b-a3b`),
  not the embedding model name. The `dim` field is correct (768) because
  it comes from the actual embed call. Don't trust `health_check()['model']`
  as the embedding model id — trust `EMBED_MODEL` in code.
- **Single LMS daemon:** `lms server start --port 1235` would *move* the
  existing server off 1234 (killing chat). The socat proxy is the way
  around this until LM Studio supports multi-instance.
- **No firewall rule added here.** If spark2's UFW or nftables ever starts
  blocking 1235, allow inbound TCP/1235 on the LAN interface.
- **CPU-only embedding inference** (the model is tiny — 84 MB — and runs
  fast on CPU; we did not pin GPU layers).
