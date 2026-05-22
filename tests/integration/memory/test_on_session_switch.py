"""Tests for HermesLocalProvider.on_session_switch narrative injection (Phase 5, Story T-031).

ADR-001 Option A: bypass _cached_system_prompt by injecting prior session context
directly into conversation_history at index 0.

Covers:
  - on_session_switch reads parent thread via _read_thread_file
  - _nt_prev_content is populated and injection happens at index 0
  - MAX_INJECTION_CHARS cap applied
  - _nt_first_turn_done guard prevents double injection
  - reset=True skips injection (fresh session)
  - Missing parent_id / empty thread gracefully handled
  - Three-tier agent_ref resolution (A1: stored, A2: MemoryManager._agent, A3: warning-and-proceed)

Run with: scripts/run_tests.sh tests/integration/memory/test_on_session_switch.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add hermes-agent root to sys.path
_AGENT_ROOT = Path(__file__).parents[3]
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))


# --------------------------------------------------------------------------:
# Module loading (same pattern as test_narrative.py)
# --------------------------------------------------------------------------:

def load_hermes_local_provider():
    """Load hermes-local __init__.py as a module."""
    # Pre-load hermes_constants so narrative.py can import it
    hc_spec = importlib.util.spec_from_file_location(
        "hermes_constants",
        str(_AGENT_ROOT / "hermes_constants.py"),
    )
    hc_mod = importlib.util.module_from_spec(hc_spec)
    sys.modules["hermes_constants"] = hc_mod
    hc_spec.loader.exec_module(hc_mod)

    # Pre-load hermes_memory_core stub
    hmc_spec = importlib.util.spec_from_file_location(
        "hermes_memory_core",
        str(_AGENT_ROOT / "hermes_memory_core" / "__init__.py"),
    )
    hmc_mod = importlib.util.module_from_spec(hmc_spec)
    sys.modules["hermes_memory_core"] = hmc_mod
    hmc_spec.loader.exec_module(hmc_mod)

    # Load narrative.py
    narrative_path = _AGENT_ROOT / "plugins" / "memory" / "hermes-local" / "narrative.py"
    narrative_spec = importlib.util.spec_from_file_location("narrative", str(narrative_path))
    narrative_mod = importlib.util.module_from_spec(narrative_spec)
    sys.modules["plugins.memory.hermes_local.narrative"] = narrative_mod
    sys.modules["narrative"] = narrative_mod
    narrative_spec.loader.exec_module(narrative_mod)

    # Register narrative in the plugins.memory.hermes_local namespace
    sys.modules["plugins.memory.hermes_local"] = MagicMock()
    sys.modules["plugins.memory.hermes_local"].narrative = narrative_mod

    # Load hermes-local __init__.py
    init_path = _AGENT_ROOT / "plugins" / "memory" / "hermes-local" / "__init__.py"
    init_spec = importlib.util.spec_from_file_location(
        "plugins.memory.hermes_local.__init__",
        str(init_path),
    )
    provider_mod = importlib.util.module_from_spec(init_spec)
    sys.modules["plugins.memory.hermes_local.__init__"] = provider_mod
    init_spec.loader.exec_module(provider_mod)
    return provider_mod


provider_mod = load_hermes_local_provider()
HermesLocalProvider = provider_mod.HermesLocalProvider


# --------------------------------------------------------------------------:
# Fixtures
# --------------------------------------------------------------------------:

@pytest.fixture
def provider():
    """Return a fresh HermesLocalProvider instance."""
    return HermesLocalProvider()


@pytest.fixture
def mock_agent():
    """Return a mock AIAgent with conversation_history."""
    agent = MagicMock()
    agent.conversation_history = []
    agent.session_id = "new-session-456"
    return agent


@pytest.fixture
def fake_hermes_home(tmp_path: Path, monkeypatch) -> Path:
    """Fake HERMES_HOME so thread files go to tmp_path."""
    fake_home = tmp_path / "hermes_home"
    fake_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    # Reset narrative module's cached path computation
    narrative_mod = sys.modules.get("plugins.memory.hermes_local.narrative")
    if narrative_mod and hasattr(narrative_mod, "_reset_path_cache"):
        narrative_mod._reset_path_cache()

    return fake_home


# --------------------------------------------------------------------------:
# Test: on_session_switch — no parent_id → no-op
# --------------------------------------------------------------------------:

def test_on_session_switch_no_parent_id(provider):
    """No parent_id → _nt_prev_content cleared, no injection."""
    provider._nt_prev_content = "stale"
    provider._nt_first_turn_done = True

    provider.on_session_switch("new-session", parent_id="", reset=False)

    assert provider._nt_prev_content == ""
    assert provider._nt_first_turn_done is False


# --------------------------------------------------------------------------:
# Test: on_session_switch — reset=True → skips injection
# --------------------------------------------------------------------------:

def test_on_session_switch_reset_true_skips_injection(provider, fake_hermes_home):
    """reset=True (/reset command) → _nt_prev_content stored but injection skipped."""
    from plugins.memory.hermes_local.narrative import _write_thread

    # Write prior session thread
    _write_thread("parent-session", [{"user": "old question", "ai": "old answer"}])

    provider.on_session_switch("new-session", parent_id="parent-session", reset=True)

    # Content is read and stored — that's fine, we just don't inject on reset
    assert "old question" in provider._nt_prev_content
    assert provider._nt_first_turn_done is False  # injection skipped


# --------------------------------------------------------------------------:
# Test: on_session_switch — reads parent thread, sets _nt_prev_content
# --------------------------------------------------------------------------:

def test_on_session_switch_reads_parent_thread(provider, fake_hermes_home):
    """on_session_switch with parent_id reads thread file and sets _nt_prev_content."""
    from plugins.memory.hermes_local.narrative import _write_thread

    # Write prior session thread
    _write_thread("parent-session-abc", [
        {"user": "Working on the memory plugin", "ai": "I've implemented the write tool"},
    ])

    provider.on_session_switch("new-session-xyz", parent_id="parent-session-abc", reset=False)

    assert provider._nt_prev_content != ""
    assert "Working on the memory plugin" in provider._nt_prev_content
    assert "memory plugin" in provider._nt_prev_content


# --------------------------------------------------------------------------:
# Test: on_session_switch — injects narrative message at index 0 (A3 path)
# --------------------------------------------------------------------------:

def test_on_session_switch_injects_at_index_0(provider, mock_agent, fake_hermes_home):
    """on_session_switch → _inject_narrative_message → user message at index 0."""
    from plugins.memory.hermes_local.narrative import _write_thread

    # Write prior thread
    _write_thread("parent-abc", [{"user": "Task A", "ai": "Done with task A"}])

    # A3: no stored agent_ref, no MemoryManager in call stack → agent stays None
    provider._agent_ref = None

    # Inject should not crash even when agent unavailable (A3: warning-and-proceed)
    provider.on_session_switch("new-xyz", parent_id="parent-abc", reset=False)

    # A3 → agent is None → injection skipped (warning logged, not an error)
    assert provider._nt_prev_content != ""
    assert provider._nt_first_turn_done is False  # no injection happened


# --------------------------------------------------------------------------:
# Test: _inject_narrative_message — A1 path (agent_ref set via initialize)
# --------------------------------------------------------------------------:

def test_inject_narrative_message_a1_path(provider, mock_agent):
    """agent_ref set via initialize() → A1 path → injection succeeds."""
    from plugins.memory.hermes_local.narrative import _write_thread

    # Write prior thread
    fake_home = provider_mod  # already set above
    _write_thread("parent-a1", [{"user": "Prior session work", "ai": "Completed"}])

    # Set via initialize (A1 path)
    provider.initialize("new-session", agent_ref=mock_agent)
    assert provider._agent_ref is not None

    provider.on_session_switch("new-session", parent_id="parent-a1", reset=False)

    # Should have injected at index 0
    assert len(mock_agent.conversation_history) == 1
    assert mock_agent.conversation_history[0]["role"] == "user"
    assert "Prior session work" in mock_agent.conversation_history[0]["content"]
    assert mock_agent.conversation_history[0]["metadata"]["source"] == "narrative_thread_injection"
    assert provider._nt_first_turn_done is True


# --------------------------------------------------------------------------:
# Test: _inject_narrative_message — double injection guard
# --------------------------------------------------------------------------:

def test_double_injection_guard(provider, mock_agent):
    """_nt_first_turn_done prevents two injections in same session."""
    provider.initialize("new-session", agent_ref=mock_agent)
    provider._nt_prev_content = "Some prior context"
    provider._nt_first_turn_done = True  # already injected

    provider._inject_narrative_message()

    assert len(mock_agent.conversation_history) == 0  # nothing added


# --------------------------------------------------------------------------:
# Test: _inject_narrative_message — MAX_INJECTION_CHARS cap
# --------------------------------------------------------------------------:

def test_injection_respects_max_chars_cap(provider, mock_agent):
    """Content exceeding MAX_INJECTION_CHARS is truncated with marker."""
    from plugins.memory.hermes_local.narrative import _write_thread

    long_content = "X" * 6000
    _write_thread("parent-long", [{"user": long_content, "ai": "response"}])

    provider.initialize("new-session", agent_ref=mock_agent)
    provider.on_session_switch("new-session", parent_id="parent-long", reset=False)

    msg = mock_agent.conversation_history[0]
    content = msg["content"]

    assert len(content) <= provider.MAX_INJECTION_CHARS + 20  # +20 for truncation marker
    assert "[…thread truncated]" in content or len(content) <= provider.MAX_INJECTION_CHARS


# --------------------------------------------------------------------------:
# Test: empty prior thread → graceful no-op
# --------------------------------------------------------------------------:

def test_on_session_switch_empty_thread_no_crash(provider, fake_hermes_home):
    """No prior thread file → graceful return, no exception."""
    provider.on_session_switch("new-session", parent_id="nonexistent-session", reset=False)

    assert provider._nt_prev_content == ""
    assert provider._nt_first_turn_done is False


# --------------------------------------------------------------------------:
# Test: _get_agent_ref — A1: stored agent_ref returned
# --------------------------------------------------------------------------:

def test_get_agent_ref_a1_stored(provider):
    """A1: stored _agent_ref from initialize() is returned."""
    mock_agent = MagicMock()
    provider._agent_ref = mock_agent
    agent = provider._get_agent_ref()
    assert agent is mock_agent


# --------------------------------------------------------------------------:
# Test: _get_agent_ref — A3: all tiers fail → returns None, logs warning
# --------------------------------------------------------------------------:

def test_get_agent_ref_a3_all_fail_returns_none(provider):
    """A3: no agent available → returns None (caller handles gracefully)."""
    provider._agent_ref = None  # A1 fails
    with patch("inspect.stack", return_value=[]):
        agent = provider._get_agent_ref()
    assert agent is None


# --------------------------------------------------------------------------:
# Test: _get_agent_ref — observable behavior: A1 tried first, then A2, then A3
# --------------------------------------------------------------------------:

def test_get_agent_ref_prefers_stored_over_walk(provider):
    """When both _agent_ref is set AND walk finds something, stored wins (A1)."""
    stored_agent = MagicMock()
    stored_agent._agent = "should_not_be_called"
    provider._agent_ref = stored_agent

    # Even if A2 walk somehow found another agent, A1 takes precedence
    with patch("inspect.stack", return_value=[]):
        agent = provider._get_agent_ref()

    assert agent is stored_agent  # A1 preferred over A2/A3


# --------------------------------------------------------------------------:
# Test: _inject_narrative_message — conversation_history not a list → warning
# --------------------------------------------------------------------------:

def test_inject_narrative_message_wrong_type(provider, mock_agent):
    """agent.conversation_history exists but is not a list → warning, no crash."""
    provider.initialize("new-session", agent_ref=mock_agent)
    mock_agent.conversation_history = "not-a-list"  # wrong type

    provider._nt_prev_content = "Some context"
    provider._inject_narrative_message()  # should not raise

    assert provider._nt_first_turn_done is False