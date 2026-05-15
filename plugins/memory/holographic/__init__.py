"""hermes-memory-store — holographic memory plugin using MemoryProvider interface.

Registers as a MemoryProvider plugin, giving the agent structured fact storage
with entity resolution, trust scoring, and HRR-based compositional retrieval.

Extends with Narrative Thread: a rolling SESSION-THREAD.md that survives
context compaction and session restarts. After each turn, the thread file is
updated. On session start, the previous thread is read and injected so Hermes
knows what was last discussed.

Original plugin by dusterbloom (PR #2351), adapted to the MemoryProvider ABC.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    hermes-memory-store:
      db_path: $HERMES_HOME/memory_store.db   # omit to use the default
      auto_extract: false
      default_trust: 0.5
      min_trust_threshold: 0.3
      temporal_decay_half_life: 0
      narrative_thread_enabled: true
      narrative_thread_file: $HERMES_HOME/SESSION-THREAD.md
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List
from pathlib import Path

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from .store import MemoryStore
from .retrieval import FactRetriever
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (unchanged from original PR)
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Deep structured memory with algebraic reasoning. "
        "Use alongside the memory tool — memory for always-on context, "
        "fact_store for deep recall and compositional queries.\n\n"
        "ACTIONS (simple → powerful):\n"
        "• add — Store a fact the user would expect you to remember.\n"
        "• search — Keyword lookup ('editor config', 'deploy process').\n"
        "• probe — Entity recall: ALL facts about a person/thing.\n"
        "• related — What connects to an entity? Structural adjacency.\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\n"
        "• contradict — Memory hygiene: find facts making conflicting claims.\n"
        "• update/remove/list — CRUD operations.\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names for 'reason'."},
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "enum": ["user_pref", "project", "tool", "general"]},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains the memory — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"]},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
        },
        "required": ["action", "fact_id"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "hermes-memory-store", default={}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Narrative Thread helpers
# ---------------------------------------------------------------------------

MAX_HISTORY = 5       # rolling window of exchanges in the thread file
SNIPPET_LEN = 120     # chars per exchange snippet in the thread file

def _truncate(s: str, max_chars: int) -> str:
    """Strip whitespace and truncate to max_chars."""
    if not s:
        return ""
    clean = s.replace("\n", " ").replace("  ", " ").strip()
    return (clean[:max_chars] + "…") if len(clean) > max_chars else clean


def _timestamp_cdt() -> str:
    """Locale-aware CDT timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M") + " CDT"


def _read_thread_file(path: Path) -> tuple[str, List[Dict[str, str]], int] | None:
    """Read an existing SESSION-THREAD.md.
    Returns (focus, exchanges, turn_count) or None if the file doesn't exist.
    """
    if not path.exists():
        return None
    try:
        content = path.read_text("utf-8")
    except Exception:
        return None

    focus = ""
    turn_count = 0
    exchanges: List[Dict[str, str]] = []

    current_exchange: Dict[str, str] | None = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("_Auto-updated") or line.startswith("# SESSION"):
            continue
        if line.startswith("## Last Updated:") or line.startswith("## Turns This Session:"):
            continue
        if line.startswith("## Session Started:"):
            continue
        if line.startswith("## Current Focus"):
            focus = line.split("## Current Focus", 1)[-1].strip()
            continue
        if line.startswith("## Exchange History"):
            continue
        if line.startswith("- **"):
            # Parse: "- **HH:MM** User: ... → ..."
            # Format: "- **TIME** User: TEXT\n  → TEXT"
            try:
                remainder = line[2:].strip()  # drop leading "- "
                time_part, rest = remainder.split("**", 1)[1].split("**", 1)
                rest = rest.strip()
                if rest.startswith("User:"):
                    _, user_part = rest.split("User:", 1)
                    current_exchange = {"time": time_part, "user": user_part.strip(), "ai": ""}
                elif current_exchange is not None and rest.startswith("→"):
                    current_exchange["ai"] = rest[1:].strip()
                    exchanges.append(current_exchange)
                    current_exchange = None
            except Exception:
                pass
        elif line.startswith("## Tools Used Recently"):
            continue
        elif line.startswith("--"):
            continue

    return focus, exchanges, turn_count


def _build_thread_content(
    focus: str,
    exchanges: List[Dict[str, str]],
    turn_count: int,
    session_start: str,
    tools_used: List[str],
) -> str:
    """Build the SESSION-THREAD.md markdown content."""
    tools_list = ", ".join(tools_used[-10:]) if tools_used else "none"
    focus_str = focus or "(system/heartbeat)"

    history_lines = ""
    for ex in exchanges:
        history_lines += f"- **{ex['time']}** User: {ex['user']}\n  → {ex['ai']}\n"

    return f"""# SESSION-THREAD.md — Working Memory
_Auto-updated by Narrative Thread after each exchange. Survives compaction._

## Last Updated: {_timestamp_cdt()}
## Turns This Session: {turn_count}
## Session Started: {session_start}

## Current Focus
{focus_str}

## Tools Used Recently
{tools_list}

## Exchange History
{history_lines or '_No exchanges yet this session._'}

---
_Write brief notes below this line if needed. They'll persist until next auto-update._
""".strip()


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HolographicMemoryProvider(MemoryProvider):
    """Holographic memory with structured facts, entity resolution, and HRR retrieval.
    
    Also maintains SESSION-THREAD.md — a rolling working memory file that survives
    context compaction and session restarts, giving Hermes instant context on resume.
    """

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store = None
        self._retriever = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))
        # Narrative Thread state
        self._nt_enabled = self._config.get("narrative_thread_enabled", True)
        self._nt_file: Path | None = None
        self._nt_prev_content: str = ""     # prior-session thread content for injection
        self._nt_prev_exists: bool = False  # whether a prior thread file existed
        self._nt_turn_count = 0
        self._nt_exchanges: List[Dict[str, str]] = []
        self._nt_tools: List[str] = []
        self._nt_pending_message: str = ""
        self._nt_session_start: str = ""
        self._nt_first_turn_done = False  # guard to inject prior content only once

    @property
    def name(self) -> str:
        return "holographic"

    def is_available(self) -> bool:
        return True  # SQLite is always available, numpy is optional

    def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.hermes-memory-store."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["hermes-memory-store"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memory_store.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts at session end", "default": "false", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
            {"key": "narrative_thread_enabled", "description": "Enable SESSION-THREAD.md rolling working memory", "default": "true", "choices": ["true", "false"]},
            {"key": "narrative_thread_file", "description": "Path to SESSION-THREAD.md", "default": f"{display_hermes_home()}/SESSION-THREAD.md"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = str(get_hermes_home())
        _default_db = _hermes_home + "/memory_store.db"
        db_path = self._config.get("db_path", _default_db)
        # Expand $HERMES_HOME in user-supplied paths so config values like
        # "$HERMES_HOME/memory_store.db" or "~/.hermes/memory_store.db" both
        # resolve to the active profile's directory.
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)
        default_trust = float(self._config.get("default_trust", 0.5))
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        hrr_weight = float(self._config.get("hrr_weight", 0.3))
        temporal_decay = int(self._config.get("temporal_decay_half_life", 0))

        self._store = MemoryStore(db_path=db_path, default_trust=default_trust, hrr_dim=hrr_dim)
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            hrr_weight=hrr_weight,
            hrr_dim=hrr_dim,
        )
        self._session_id = session_id

        # Narrative Thread: resolve thread file path
        if self._nt_enabled:
            nt_file = self._config.get("narrative_thread_file", f"{_hermes_home}/SESSION-THREAD.md")
            if isinstance(nt_file, str):
                nt_file = nt_file.replace("$HERMES_HOME", _hermes_home)
                nt_file = nt_file.replace("${HERMES_HOME}", _hermes_home)
            self._nt_file = Path(nt_file)
            self._nt_file.parent.mkdir(parents=True, exist_ok=True)
            self._nt_session_start = _timestamp_cdt()
            self._nt_turn_count = 0
            self._nt_exchanges = []
            self._nt_tools = []
            self._nt_pending_message = ""
            self._nt_first_turn_done = False

    def system_prompt_block(self) -> str:
        # Narrative Thread: inject prior session context on first turn of a resumed session
        if self._nt_enabled and self._nt_prev_exists and not self._nt_first_turn_done:
            self._nt_first_turn_done = True
            return (
                "# Prior Session Context (Narrative Thread)\n"
                "The following working memory was captured from the previous session:\n\n"
                + self._nt_prev_content
            )

        if not self._store:
            return ""
        try:
            total = self._store._conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
        except Exception:
            total = 0
        if total == 0:
            return (
                "# Holographic Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
                "Use fact_feedback to rate facts after using them (trains trust scores)."
            )
        return (
            f"# Holographic Memory\n"
            f"Active. {total} facts stored with entity resolution and trust scoring.\n"
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(query, min_trust=self._min_trust, limit=5)
            if not results:
                return ""
            lines = []
            for r in results:
                trust = r.get("trust_score", r.get("trust", 0))
                lines.append(f"- [{trust:.1f}] {r.get('content', '')}")
            return "## Holographic Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Holographic prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Narrative Thread: after each turn, append to rolling window and write thread file
        if self._nt_enabled:
            self._nt_turn_count += 1
            now = datetime.now(timezone.utc).strftime("%H:%M")
            self._nt_exchanges.append({
                "time": now,
                "user": _truncate(user_content, SNIPPET_LEN),
                "ai": _truncate(assistant_content, SNIPPET_LEN),
            })
            if len(self._nt_exchanges) > MAX_HISTORY:
                self._nt_exchanges = self._nt_exchanges[-MAX_HISTORY:]

            # Track tools from assistant response
            for pat in [r'"tool":\s*"(\w+)"', r"<tool_name>(\w+)</tool_name>", r"calling\s+(\w+)\s+tool"]:
                for m in re.finditer(pat, assistant_content):
                    tool = m.group(1)
                    if tool not in self._nt_tools:
                        self._nt_tools.append(tool)

            self._write_thread()

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Capture the pending user message so we can record it in the next sync_turn."""
        if self._nt_enabled:
            self._nt_pending_message = message

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return self._handle_fact_feedback(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._nt_enabled:
            if not self._config.get("auto_extract", False):
                return
            if not self._store or not messages:
                return
            self._auto_extract_facts(messages)
            return

        # Narrative Thread: write final thread state
        self._write_thread()
        logger.info("[Narrative Thread] Session ended, thread written to %s", self._nt_file)

        # Also run auto-extract if configured
        if self._config.get("auto_extract", False) and self._store and messages:
            self._auto_extract_facts(messages)

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, **kwargs) -> None:
        """Called when session_id changes — prepare thread state for the new session.

        For /resume and /branch (reset=False): the prior session's SESSION-THREAD.md
        already has its last state. Read it now so system_prompt_block can inject it as
        "Prior Session Context" for the first turn of the resumed conversation.

        For /reset and /new (reset=True): no prior context to carry forward.
        """
        if not self._nt_enabled:
            return

        if reset:
            # Genuine new conversation — don't carry forward any prior thread
            self._nt_turn_count = 0
            self._nt_exchanges = []
            self._nt_tools = []
            self._nt_pending_message = ""
            self._nt_prev_content = ""
            self._nt_prev_exists = False
            self._nt_first_turn_done = False
            self._nt_session_start = _timestamp_cdt()
        else:
            # /resume or /branch — read the prior session's thread for context injection
            if self._nt_file and self._nt_file.exists():
                try:
                    self._nt_prev_content = self._nt_file.read_text("utf-8")
                    self._nt_prev_exists = True
                except Exception:
                    self._nt_prev_content = ""
                    self._nt_prev_exists = False
            else:
                self._nt_prev_content = ""
                self._nt_prev_exists = False

            self._nt_turn_count = 0
            self._nt_exchanges = []
            self._nt_tools = []
            self._nt_pending_message = ""
            self._nt_first_turn_done = False
            self._nt_session_start = _timestamp_cdt()

        self._session_id = new_session_id

    def on_memory_write(self, action: str, target: str, content: str, metadata: dict | None = None) -> None:
        """Mirror built-in memory writes as facts."""
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                self._store.add_fact(content, category=category)
            except Exception as e:
                logger.debug("Holographic memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        self._store = None
        self._retriever = None

    # -- Narrative Thread private ------------------------------------------------

    def _write_thread(self) -> None:
        """Write the current session's rolling thread to SESSION-THREAD.md."""
        if not self._nt_enabled or not self._nt_file:
            return
        try:
            # Extract focus from the most recent user message
            focus = _truncate(self._nt_pending_message, 100) if self._nt_pending_message else "(system/heartbeat)"
            content = _build_thread_content(
                focus=focus,
                exchanges=self._nt_exchanges,
                turn_count=self._nt_turn_count,
                session_start=self._nt_session_start or _timestamp_cdt(),
                tools_used=self._nt_tools,
            )
            self._nt_file.write_text(content, "utf-8")
        except Exception as e:
            logger.debug("[Narrative Thread] Failed to write thread file: %s", e)

    # -- Tool handlers -----------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        try:
            action = args["action"]
            store = self._store
            retriever = self._retriever

            if action == "add":
                fact_id = store.add_fact(
                    args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                return json.dumps({"fact_id": fact_id, "status": "added"})

            elif action == "search":
                results = retriever.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "probe":
                results = retriever.probe(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "related":
                results = retriever.related(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "reason":
                entities = args.get("entities", [])
                if not entities:
                    return tool_error("reason requires 'entities' list")
                results = retriever.reason(
                    entities,
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "contradict":
                results = retriever.contradict(
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "update":
                updated = store.update_fact(
                    int(args["fact_id"]),
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                )
                return json.dumps({"updated": updated})

            elif action == "remove":
                removed = store.remove_fact(int(args["fact_id"]))
                return json.dumps({"removed": removed})

            elif action == "list":
                facts = store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"facts": facts, "count": len(facts)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_fact_feedback(self, args: dict) -> str:
        try:
            fact_id = int(args["fact_id"])
            helpful = args["action"] == "helpful"
            result = self._store.record_feedback(fact_id, helpful=helpful)
            return json.dumps(result)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Auto-extraction (on_session_end) ------------------------------------

    def _auto_extract_facts(self, messages: list) -> None:
        _PREF_PATTERNS = [
            re.compile(r'\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
            re.compile(r'\bI\s+(?:always|never|usually)\s+(.+)', re.IGNORECASE),
        ]
        _DECISION_PATTERNS = [
            re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
        ]

        extracted = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 10:
                continue

            for pattern in _PREF_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="user_pref")
                        extracted += 1
                    except Exception:
                        pass
                    break

            for pattern in _DECISION_PATTERNS:
                if pattern.search(content):
                    try:
                        self._store.add_fact(content[:400], category="project")
                        extracted += 1
                    except Exception:
                        pass
                    break

        if extracted:
            logger.info("Auto-extracted %d facts from conversation", extracted)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the holographic memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = HolographicMemoryProvider(config=config)
    ctx.register_memory_provider(provider)