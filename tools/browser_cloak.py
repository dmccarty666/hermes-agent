"""
CloakBrowser tool — anti-detection browser automation for Hermes.

Uses CloakBrowser (source-level stealth-patched Chromium) instead of
stock Playwright/Chromium. CloakBrowser's 32 C++ fingerprint patches
pass Cloudflare, Facebook, and other antibot systems.

Threading model:
  Each task_id gets its OWN background thread running a Playwright sync
  browser. ALL browser operations (goto, click, evaluate, etc.) are
  dispatched to that thread via a queue. This keeps Playwright sync calls
  in the thread that created the browser (required by Playwright) while
  keeping Hermes's async tool handlers unblocked.

Setup (already done):
  ~/.hermes/hermes-agent/venv/bin/python -m pip install cloakbrowser

Agent configuration — enable in config.yaml:
  toolsets:
    enable:
      - browser         # standard Chromium via agent-browser
      - cloak_browser   # CloakBrowser stealth (anti-detection)
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cloakbrowser import launch

from tools.registry import registry

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Session management
# -----------------------------------------------------------------------
# One (browser, page) pair per task_id, each in its own dedicated thread.
# All Playwright calls for a session run in that session's thread via a queue.

_SESSIONS: Dict[str, Dict[str, Any]] = {}
_SESSIONS_LOCK = threading.RLock()


def _get_session(task_id: str) -> Dict[str, Any]:
    """Get or create a CloakBrowser session. All browser ops must run in returned thread."""
    with _SESSIONS_LOCK:
        if task_id in _SESSIONS:
            return _SESSIONS[task_id]

        cmd_queue: queue.Queue = queue.Queue()
        ready_event = threading.Event()
        launch_error: Dict[str, Any] = {}

        def thread_target():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            page = None
            try:
                browser = launch()
                context = browser.new_context()
                page = context.new_page()
                page.set_default_timeout(30000)
                ready_event.set()  # signal: ready for commands
                # Command loop — process operations until told to stop
                while True:
                    try:
                        cmd = cmd_queue.get(timeout=120)
                        if cmd is None:  # shutdown sentinel
                            break
                        fn, args, kwargs, reply_q = cmd
                        try:
                            kwargs["_page"] = page
                            reply_q.put(("ok", fn(*args, **kwargs)))
                        except Exception as e:
                            reply_q.put(("error", e))
                    except queue.Empty:
                        break
            except Exception as e:
                launch_error["msg"] = str(e)
                ready_event.set()
            finally:
                if page:
                    try:
                        page.context.close()
                    except Exception:
                        pass
                try:
                    loop.close()
                except Exception:
                    pass

        t = threading.Thread(target=thread_target, daemon=True, name=f"cloak-{task_id}")
        t.start()

        # Wait for thread to signal ready (or error)
        # Playwright browser launch can take 30-60s on first run
        ready_event.wait(timeout=120)
        if launch_error:
            raise RuntimeError(f"CloakBrowser launch failed: {launch_error['msg']}")
        if not ready_event.is_set():
            raise RuntimeError("CloakBrowser launch timed out after 120s")

        _SESSIONS[task_id] = {
            "queue": cmd_queue,
            "thread": t,
        }
        logger.info("CloakBrowser session ready for task_id=%s", task_id)
        return _SESSIONS[task_id]


def _session_call(task_id: str, fn, *args, _timeout: float = 60, **kwargs) -> Any:
    """
    Dispatch a blocking Playwright call to the session's background thread.
    Returns the result. Raises on error.
    """
    sess = _get_session(task_id)
    reply_q: queue.Queue = queue.Queue()
    sess["queue"].put((fn, args, kwargs, reply_q))
    status, val = reply_q.get(timeout=_timeout)
    if status == "error":
        raise val
    return val


def _close_session(task_id: str) -> None:
    with _SESSIONS_LOCK:
        if task_id not in _SESSIONS:
            return
        sess = _SESSIONS.pop(task_id)
    try:
        sess["queue"].put(None)  # signal shutdown
    except Exception:
        pass
    try:
        sess["thread"].join(timeout=5)
    except Exception:
        pass
    logger.info("CloakBrowser session closed for task_id=%s", task_id)


def _close_all() -> None:
    with _SESSIONS_LOCK:
        for tid in list(_SESSIONS.keys()):
            try:
                _SESSIONS[tid]["queue"].put(None)
            except Exception:
                pass
        _SESSIONS.clear()
    logger.info("All CloakBrowser sessions closed")

# -----------------------------------------------------------------------
# Screenshot helper
# -----------------------------------------------------------------------

_CLOAK_SS_DIR = Path.home() / ".hermes" / "cloak_screenshots"
_CLOAK_SS_DIR.mkdir(parents=True, exist_ok=True)


def _take_screenshot(page, task_id: str) -> str:
    screenshot_id = uuid.uuid4().hex[:8]
    path = _CLOAK_SS_DIR / f"cloak_{task_id}_{screenshot_id}.png"
    page.screenshot(path=str(path), full_page=False)
    return str(path)

# -----------------------------------------------------------------------
# JS-based accessibility tree builder (replaces Playwright's accessibility API
# which CloakBrowser doesn't expose)
# -----------------------------------------------------------------------

def _build_snapshot(page, full: bool = False) -> Dict[str, Any]:
    """
    Get a snapshot of interactive elements using JS evaluation.
    Returns a dict with 'elements' (list) and 'snapshot_text' (str).
    Interactive elements get @eN refs matching browser_tool convention.
    """
    try:
        elements_raw = page.evaluate("""
() => {
  const selectors = [
    'a[href]', 'button', 'input:not([type="hidden"])', 'select',
    'textarea', '[role="button"]', '[role="link"]', '[role="menuitem"]',
    '[role="tab"]', '[role="checkbox"]', '[role="radio"]',
    '[role="textbox"]', '[role="combobox"]', '[role="searchbox"]',
    '[contenteditable="true"]', '[tabindex]:not([tabindex="-1"])'
  ];
  const seen = new Set();
  const results = [];
  selectors.forEach(sel => {
    try {
      document.querySelectorAll(sel).forEach(el => {
        if (seen.has(el)) return;
        seen.add(el);
        const rect = el.getBoundingClientRect();
        if (rect.width < 4 || rect.height < 4) return;
        const role = el.getAttribute('role') || el.tagName.toLowerCase();
        const text = (el.innerText || el.value || el.placeholder || '').trim().slice(0, 120);
        const href = el.href || '';
        const name_attr = el.id || el.name || '';
        const type_attr = el.type || '';
        const is_checked = el.checked !== undefined ? el.checked : null;
        const is_disabled = el.disabled || el.getAttribute('aria-disabled') === 'true';
        results.push({
          role,
          text,
          href,
          name: name_attr,
          type: type_attr,
          checked: is_checked,
          disabled: is_disabled,
          x: Math.round(rect.x),
          y: Math.round(rect.y),
        });
      });
    } catch(e) {}
  });
  return results;
}
""")
    except Exception as e:
        logger.warning("JS element enumeration failed: %s", e)
        elements_raw = []

    if not elements_raw:
        return {
            "success": True,
            "url": page.url,
            "title": page.title(),
            "elements": [],
            "element_count": 0,
            "snapshot_text": f"URL: {page.url}\nTitle: {page.title()}\n\n(No interactive elements found)",
        }

    # Sort top-to-bottom, left-to-right
    elements_raw.sort(key=lambda el: (el["y"], el["x"]))

    INTERACTIVE = {
        "a", "button", "input", "select", "textarea",
        "link", "menuitem", "tab", "checkbox", "radio",
        "textbox", "combobox", "searchbox", "menuitemcheckbox", "menuitemradio"
    }

    elements = []
    lines = []
    for idx, el in enumerate(elements_raw):
        role = el["role"].lower()
        if role in ("img", "svg", "path", "rect", "circle", "line", "polygon") and not el["text"]:
            continue
        is_interactive = role in INTERACTIVE or el.get("disabled") or bool(el.get("href"))
        if not is_interactive:
            continue

        ref = f"@e{idx + 1}"
        el["ref"] = ref
        elements.append(el)

        attrs = []
        if el.get("type"):
            attrs.append(f"type={el['type']}")
        if el.get("checked") is not None:
            attrs.append(f"checked={el['checked']}")
        if el.get("disabled"):
            attrs.append("disabled")
        if el.get("href"):
            attrs.append("href")
        attr_str = f" ({', '.join(attrs)})" if attrs else ""
        lines.append(f"  {ref} [{role}] {el['text']!r}{attr_str}")

    body_text = ""
    if full:
        try:
            body_text = page.inner_text("body")
            if len(body_text) > 8000:
                body_text = body_text[:8000] + "\n... [truncated]"
        except Exception:
            pass

    lines_str = "\n".join(lines) if lines else "(no interactive elements)"
    result = {
        "success": True,
        "url": page.url,
        "title": page.title(),
        "elements": elements,
        "element_count": len(elements),
        "snapshot_text": f"URL: {page.url}\nTitle: {page.title()}\n\nInteractive elements ({len(elements)}):\n{lines_str}",
    }
    if full and body_text:
        result["content"] = body_text
        result["snapshot_text"] += f"\n\nPage content:\n{body_text}"
    return result

# -----------------------------------------------------------------------
# Tool functions — all dispatch to the session thread via _session_call
# -----------------------------------------------------------------------

def cloak_navigate(url: str, task_id: Optional[str] = None) -> str:
    """
    Navigate to a URL in CloakBrowser (stealth Chromium).
    Returns JSON with page snapshot and @eN interactive element refs.
    Anti-detection: CloakBrowser passes Cloudflare, Facebook, antibot systems.
    """
    tid = task_id or "default"
    try:
        # Step 1: create session (launches browser in bg thread)
        _get_session(tid)

        # Step 2: navigate in session thread
        snap = _session_call(tid, _nav_and_snapshot, url, _timeout=90)

        snap["stealth"] = True
        snap["cloakbrowser"] = True

        # Detect challenge pages
        try:
            body_text = snap.get("snapshot_text", "")
            challenge_kws = ["cloudflare", "checking your browser", "attention required",
                             "just a moment", "challenge", "cf-challenge"]
            snap["challenge_detected"] = any(kw in body_text.lower() for kw in challenge_kws)
        except Exception:
            snap["challenge_detected"] = False

        return json.dumps(snap)

    except Exception as e:
        logger.error("cloak_navigate(%s) failed: %s", url, e)
        return json.dumps({"success": False, "error": str(e), "url": url})


def _nav_and_snapshot(url: str, _page=None, **kwargs) -> Dict[str, Any]:
    """Must run in the session thread. _page injected by command loop."""
    if _page is None:
        raise RuntimeError("Session not initialized")
    _page.goto(url, wait_until="domcontentloaded")
    _page.wait_for_load_state("networkidle", timeout=15000)
    return _build_snapshot(_page, full=False)



def cloak_snapshot(task_id: Optional[str] = None, full: bool = False) -> str:
    """Refresh the accessibility snapshot of the current page."""
    tid = task_id or "default"
    try:
        _get_session(tid)  # ensure session exists
        snap = _session_call(tid, _do_snapshot, full, _timeout=30)
        return json.dumps(snap)
    except Exception as e:
        logger.error("cloak_snapshot failed: %s", e)
        return json.dumps({"success": False, "error": str(e)})


def _do_snapshot(full: bool, _page, **kwargs) -> Dict[str, Any]:
    """Must run in session thread. _page injected by command loop."""
    return _build_snapshot(_page, full=full)


def cloak_click(ref: str, task_id: Optional[str] = None) -> str:
    """Click an element by its @eN ref from the snapshot."""
    tid = task_id or "default"
    try:
        _get_session(tid)
        result = _session_call(tid, _do_click, ref, _timeout=30)
        return json.dumps({"success": True, "clicked_ref": ref, "details": result})
    except Exception as e:
        logger.error("cloak_click(%s) failed: %s", ref, e)
        return json.dumps({"success": False, "error": str(e), "ref": ref})


def _do_click(ref: str, _page, **kwargs) -> Dict[str, Any]:
    """Must run in session thread."""
    import re
    snap = _build_snapshot(_page, full=False)
    elements = snap.get("elements", [])
    match = re.match(r"@e(\d+)", ref.strip())
    if not match:
        raise ValueError(f"Invalid ref format: {ref}")
    idx = int(match.group(1)) - 1
    if idx < 0 or idx >= len(elements):
        raise ValueError(f"Ref {ref} out of range (have {len(elements)} elements)")
    el = elements[idx]
    role, name = el["role"], el["text"]
    locator = _page.locator(f"[role='{role}']").filter(has_text=name).first
    locator.click(timeout=5000)
    return {"role": role, "name": name}


def cloak_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """Type text into an input field by its @eN ref."""
    tid = task_id or "default"
    try:
        _get_session(tid)
        _session_call(tid, _do_type, ref, text, _timeout=30)
        return json.dumps({"success": True, "ref": ref, "text_length": len(text)})
    except Exception as e:
        logger.error("cloak_type(%s) failed: %s", ref, e)
        return json.dumps({"success": False, "error": str(e), "ref": ref})


def _do_type(ref: str, text: str, _page, **kwargs) -> None:
    """Must run in session thread."""
    import re
    snap = _build_snapshot(_page, full=False)
    elements = snap.get("elements", [])
    match = re.match(r"@e(\d+)", ref.strip())
    if not match:
        raise ValueError(f"Invalid ref: {ref}")
    idx = int(match.group(1)) - 1
    if idx < 0 or idx >= len(elements):
        raise ValueError(f"Ref {ref} out of range")
    el = elements[idx]
    locator = _page.locator(f"[role='{el['role']}']").filter(has_text=el["text"]).first
    locator.fill(text)


def cloak_scroll(direction: str = "down", task_id: Optional[str] = None) -> str:
    """Scroll the page up or down."""
    tid = task_id or "default"
    try:
        _get_session(tid)
        snap = _session_call(tid, _do_scroll, direction, _timeout=30)
        return json.dumps(snap)
    except Exception as e:
        logger.error("cloak_scroll failed: %s", e)
        return json.dumps({"success": False, "error": str(e)})


def _do_scroll(direction: str, _page, **kwargs) -> Dict[str, Any]:
    """Must run in session thread."""
    if direction == "up":
        _page.evaluate("window.scrollBy(0, -window.innerHeight)")
    else:
        _page.evaluate("window.scrollBy(0, window.innerHeight)")
    _page.wait_for_load_state("domcontentloaded", timeout=3000)
    return _build_snapshot(_page, full=False)


def cloak_back(task_id: Optional[str] = None) -> str:
    """Navigate back in browser history."""
    tid = task_id or "default"
    try:
        _get_session(tid)
        snap = _session_call(tid, _do_back, _timeout=30)
        return json.dumps(snap)
    except Exception as e:
        logger.error("cloak_back failed: %s", e)
        return json.dumps({"success": False, "error": str(e)})


def _do_back(_page, **kwargs) -> Dict[str, Any]:
    """Must run in session thread."""
    _page.go_back()
    _page.wait_for_load_state("domcontentloaded", timeout=10000)
    return _build_snapshot(_page, full=False)


def cloak_press(key: str, task_id: Optional[str] = None) -> str:
    """Press a keyboard key (Enter, Tab, Escape, ArrowDown, etc.)."""
    tid = task_id or "default"
    try:
        _get_session(tid)
        _session_call(tid, _do_press, key, _timeout=10)
        return json.dumps({"success": True, "key": key})
    except Exception as e:
        logger.error("cloak_press(%s) failed: %s", key, e)
        return json.dumps({"success": False, "error": str(e), "key": key})


def _do_press(key: str, _page, **kwargs) -> None:
    """Must run in session thread."""
    _page.keyboard.press(key)


def cloak_console(expression: Optional[str] = None, task_id: Optional[str] = None) -> str:
    """Evaluate JavaScript in the page context, or read console messages."""
    tid = task_id or "default"
    try:
        _get_session(tid)
        result = _session_call(tid, _do_console, expression, _timeout=15)
        return json.dumps({"success": True, "result": result})
    except Exception as e:
        logger.error("cloak_console failed: %s", e)
        return json.dumps({"success": False, "error": str(e)})


def _do_console(expression: Optional[str], _page, **kwargs) -> Any:
    """Must run in session thread."""
    if expression:
        return _page.evaluate(expression)
    try:
        return _page.evaluate("() => window.__hermes_console_logs || []")
    except Exception:
        return []


def cloak_screenshot(task_id: Optional[str] = None) -> str:
    """Take a PNG screenshot and return the file path."""
    tid = task_id or "default"
    try:
        _get_session(tid)
        path = _session_call(tid, _do_screenshot, tid, _timeout=30)
        return json.dumps({"success": True, "path": path})
    except Exception as e:
        logger.error("cloak_screenshot failed: %s", e)
        return json.dumps({"success": False, "error": str(e)})


def _do_screenshot(tid: str, _page, **kwargs) -> str:
    """Must run in session thread."""
    return _take_screenshot(_page, tid)


def cloak_get_images(task_id: Optional[str] = None) -> str:
    """List all images on the current page with URLs and alt text."""
    tid = task_id or "default"
    try:
        _get_session(tid)
        images = _session_call(tid, _do_get_images, _timeout=15)
        return json.dumps({"success": True, "images": images, "count": len(images)})
    except Exception as e:
        logger.error("cloak_get_images failed: %s", e)
        return json.dumps({"success": False, "error": str(e)})


def _do_get_images(_page, **kwargs) -> List[Dict[str, Any]]:
    """Must run in session thread."""
    return _page.evaluate("""
() => Array.from(document.images).map(img => ({
    src: img.src,
    alt: img.alt,
    width: img.naturalWidth,
    height: img.naturalHeight,
}))
""")


def cloak_vision(question: str, task_id: Optional[str] = None,
                  annotate: bool = False) -> str:
    """
    Take a screenshot of the current CloakBrowser page and save it.
    Returns the path — pass to vision_analyze separately for AI vision analysis.
    """
    tid = task_id or "default"
    try:
        _get_session(tid)
        path = _session_call(tid, _do_screenshot, tid, _timeout=30)
        return json.dumps({
            "success": True,
            "screenshot_path": path,
            "question": question,
            "note": "Pass screenshot_path to vision_analyze for AI analysis",
        })
    except Exception as e:
        logger.error("cloak_vision failed: %s", e)
        return json.dumps({"success": False, "error": str(e)})


# -----------------------------------------------------------------------
# Availability check
# -----------------------------------------------------------------------

def check_cloakbrowser_requirements() -> Optional[str]:
    try:
        from cloakbrowser import launch
        return None
    except ImportError:
        return "CloakBrowser not installed. Run: ~/.hermes/hermes-agent/venv/bin/python -m pip install cloakbrowser"


# -----------------------------------------------------------------------
# Tool schemas
# -----------------------------------------------------------------------

CLOAK_TOOL_SCHEMAS = [
    {
        "name": "cloak_navigate",
        "description": "Navigate to a URL in CloakBrowser (stealth Chromium). CloakBrowser passes Cloudflare, Facebook, and other antibot systems that block standard Playwright/Chromium. Returns a page snapshot with @eN interactive element refs.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "task_id": {"type": "string", "default": "default"},
            },
            "required": ["url"],
        }
    },
    {
        "name": "cloak_snapshot",
        "description": "Refresh the accessibility snapshot of the current CloakBrowser page. Use after interactions that change the page.",
        "parameters": {
            "type": "object",
            "properties": {
                "full": {"type": "boolean", "default": False},
                "task_id": {"type": "string", "default": "default"},
            },
            "required": [],
        }
    },
    {
        "name": "cloak_click",
        "description": "Click an element by its @eN ref from the snapshot.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "task_id": {"type": "string", "default": "default"},
            },
            "required": ["ref"],
        }
    },
    {
        "name": "cloak_type",
        "description": "Type text into an input field by its @eN ref.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "text": {"type": "string"},
                "task_id": {"type": "string", "default": "default"},
            },
            "required": ["ref", "text"],
        }
    },
    {
        "name": "cloak_scroll",
        "description": "Scroll the page up or down.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down"]},
                "task_id": {"type": "string", "default": "default"},
            },
            "required": ["direction"],
        }
    },
    {
        "name": "cloak_back",
        "description": "Navigate back in browser history.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "default": "default"},
            },
            "required": [],
        }
    },
    {
        "name": "cloak_press",
        "description": "Press a keyboard key (Enter, Tab, Escape, etc.).",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "task_id": {"type": "string", "default": "default"},
            },
            "required": ["key"],
        }
    },
    {
        "name": "cloak_console",
        "description": "Evaluate JavaScript in the page context.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "task_id": {"type": "string", "default": "default"},
            },
            "required": [],
        }
    },
    {
        "name": "cloak_screenshot",
        "description": "Take a PNG screenshot and return the file path.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "default": "default"},
            },
            "required": [],
        }
    },
    {
        "name": "cloak_get_images",
        "description": "List all images on the current page with URLs and alt text.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "default": "default"},
            },
            "required": [],
        }
    },
    {
        "name": "cloak_vision",
        "description": "Take a screenshot and return the path for vision analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "annotate": {"type": "boolean", "default": False},
                "task_id": {"type": "string", "default": "default"},
            },
            "required": ["question"],
        }
    },
]

# -----------------------------------------------------------------------
# Registry — register all tools under toolset "cloak_browser"
# -----------------------------------------------------------------------

for schema in CLOAK_TOOL_SCHEMAS:
    name = schema["name"]
    HANDLERS = {
        "cloak_navigate":     lambda a, **k: cloak_navigate(url=a.get("url", ""), task_id=k.get("task_id")),
        "cloak_snapshot":     lambda a, **k: cloak_snapshot(full=a.get("full", False), task_id=k.get("task_id")),
        "cloak_click":        lambda a, **k: cloak_click(ref=a.get("ref", ""), task_id=k.get("task_id")),
        "cloak_type":        lambda a, **k: cloak_type(ref=a.get("ref", ""), text=a.get("text", ""), task_id=k.get("task_id")),
        "cloak_scroll":       lambda a, **k: cloak_scroll(direction=a.get("direction", "down"), task_id=k.get("task_id")),
        "cloak_back":         lambda a, **k: cloak_back(task_id=k.get("task_id")),
        "cloak_press":        lambda a, **k: cloak_press(key=a.get("key", ""), task_id=k.get("task_id")),
        "cloak_console":      lambda a, **k: cloak_console(expression=a.get("expression"), task_id=k.get("task_id")),
        "cloak_screenshot":   lambda a, **k: cloak_screenshot(task_id=k.get("task_id")),
        "cloak_get_images":   lambda a, **k: cloak_get_images(task_id=k.get("task_id")),
        "cloak_vision":       lambda a, **k: cloak_vision(question=a.get("question", ""), annotate=a.get("annotate", False), task_id=k.get("task_id")),
    }
    registry.register(
        name=name,
        toolset="cloak_browser",
        schema=schema,
        handler=HANDLERS.get(name, lambda a, **k: json.dumps({"error": "no handler"})),
        check_fn=check_cloakbrowser_requirements,
        emoji="🕶",
    )
