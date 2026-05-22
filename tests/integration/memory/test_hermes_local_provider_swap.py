"""Story T-014 — Confirm holographic tools unloaded.

Integration tests that verify the hermes-local provider swap works correctly:
  - When memory.provider=hermes-local, fact_store/fact_feedback are NOT registered.
  - When memory.provider=hermes-local, memory_query IS registered.
  - When memory.provider=holographic, fact_store/fact_feedback ARE registered (no regression).

The tool swap is handled entirely by the MemoryManager — only the active provider's
get_tool_schemas() results are used. No plugin-level disable/enable mechanism exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------

PLUGIN_ROOT = Path(__file__).parents[4]  # hermes-agent/


def _load_provider(provider_name: str, fake_home: Path) -> "MemoryProvider":
    """Load and return a memory provider by name, simulating Hermes startup.

    This mimics what AIAgent.__init__ does:
      from plugins.memory import load_memory_provider
      _mp = load_memory_provider("hermes-local")
      if _mp and _mp.is_available():
          _memory_manager.add_provider(_mp)
    """
    # Patch HERMES_HOME for the duration so is_available() reads the right config
    import os
    old_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(fake_home)
    try:
        # Add hermes-agent root to sys.path so plugins.memory can be imported
        _agent_root = str(PLUGIN_ROOT)
        if _agent_root not in sys.path:
            sys.path.insert(0, _agent_root)

        from plugins.memory import load_memory_provider
        provider = load_memory_provider(provider_name)
        return provider
    finally:
        if old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = old_home


def _provider_tool_names(provider) -> set:
    """Return the set of tool names a provider registers."""
    if provider is None:
        return set()
    return {s["name"] for s in provider.get_tool_schemas()}


# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------

@pytest.fixture
def hermes_local_home(tmp_path, monkeypatch):
    """HERMES_HOME with memory.provider: hermes-local."""
    fake = tmp_path / "hermes_home_hermes_local"
    fake.mkdir(parents=True)
    (fake / "config.yaml").write_text("memory:\n  provider: hermes-local\n")
    monkeypatch.setenv("HERMES_HOME", str(fake))
    return fake


@pytest.fixture
def holographic_home(tmp_path, monkeypatch):
    """HERMES_HOME with memory.provider: holographic."""
    fake = tmp_path / "hermes_home_holographic"
    fake.mkdir(parents=True)
    (fake / "config.yaml").write_text("memory:\n  provider: holographic\n")
    monkeypatch.setenv("HERMES_HOME", str(fake))
    return fake


@pytest.fixture
def no_provider_home(tmp_path, monkeypatch):
    """HERMES_HOME with no memory.provider set."""
    fake = tmp_path / "hermes_home_no_provider"
    fake.mkdir(parents=True)
    (fake / "config.yaml").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(fake))
    return fake


# ---------------------------------------------------------------------------------------
# AC-1: fact_store absent when hermes-local active
# ---------------------------------------------------------------------------------------

def test_fact_store_absent_when_hermes_local_active(hermes_local_home):
    """Given memory.provider=hermes-local, fact_store must NOT be in the tool list."""
    provider = _load_provider("hermes-local", hermes_local_home)
    assert provider is not None, "hermes-local provider should load"
    assert provider.is_available(), "hermes-local should be available with correct config"

    tool_names = _provider_tool_names(provider)
    assert "fact_store" not in tool_names, (
        f"fact_store must NOT appear when hermes-local is active. "
        f"Got tools: {tool_names}"
    )


# ---------------------------------------------------------------------------------------
# AC-2: fact_feedback absent when hermes-local active
# ---------------------------------------------------------------------------------------

def test_fact_feedback_absent_when_hermes_local_active(hermes_local_home):
    """Given memory.provider=hermes-local, fact_feedback must NOT be in the tool list."""
    provider = _load_provider("hermes-local", hermes_local_home)
    assert provider is not None
    assert provider.is_available()

    tool_names = _provider_tool_names(provider)
    assert "fact_feedback" not in tool_names, (
        f"fact_feedback must NOT appear when hermes-local is active. "
        f"Got tools: {tool_names}"
    )


# ---------------------------------------------------------------------------------------
# AC-3: memory_query present when hermes-local active
# ---------------------------------------------------------------------------------------

def test_memory_query_present_when_hermes_local_active(hermes_local_home):
    """Given memory.provider=hermes-local, memory_query must be in the tool list."""
    provider = _load_provider("hermes-local", hermes_local_home)
    assert provider is not None
    assert provider.is_available()

    tool_names = _provider_tool_names(provider)
    assert "memory_query" in tool_names, (
        f"memory_query must appear when hermes-local is active. "
        f"Got tools: {tool_names}"
    )


# ---------------------------------------------------------------------------------------
# AC-3 follow-on: memory_get_source present when hermes-local active (T-013)
# ---------------------------------------------------------------------------------------

def test_memory_get_source_present_when_hermes_local_active(hermes_local_home):
    """Given memory.provider=hermes-local, memory_get_source must also be present."""
    provider = _load_provider("hermes-local", hermes_local_home)
    assert provider is not None
    assert provider.is_available()

    tool_names = _provider_tool_names(provider)
    assert "memory_get_source" in tool_names, (
        f"memory_get_source must appear when hermes-local is active. "
        f"Got tools: {tool_names}"
    )


# ---------------------------------------------------------------------------------------
# AC-4: holographic tools present when holographic is the active provider (no regression)
# ---------------------------------------------------------------------------------------

def test_fact_store_present_when_holographic_active(holographic_home):
    """Given memory.provider=holographic, fact_store must be in the tool list (no regression)."""
    provider = _load_provider("holographic", holographic_home)
    # holographic may not be available if its dependencies aren't installed —
    # that's fine, we only check that IF it loads, it returns fact_store
    if provider is None:
        pytest.skip("holographic provider failed to load (dependencies missing)")

    tool_names = _provider_tool_names(provider)
    assert "fact_store" in tool_names, (
        f"fact_store must appear when holographic is active. "
        f"Got tools: {tool_names}"
    )


def test_fact_feedback_present_when_holographic_active(holographic_home):
    """Given memory.provider=holographic, fact_feedback must be in the tool list (no regression)."""
    provider = _load_provider("holographic", holographic_home)
    if provider is None:
        pytest.skip("holographic provider failed to load (dependencies missing)")

    tool_names = _provider_tool_names(provider)
    assert "fact_feedback" in tool_names, (
        f"fact_feedback must appear when holographic is active. "
        f"Got tools: {tool_names}"
    )


# ---------------------------------------------------------------------------------------
# AC-3 cross-check: hermes-local tools vs holographic tools are mutually exclusive
# ---------------------------------------------------------------------------------------

def test_hermes_local_and_holographic_tools_are_disjoint(hermes_local_home, holographic_home):
    """hermes-local and holographic must not both register the same tool name.

    This is a structural guarantee from the MemoryManager (only one external provider
    is active at a time), but we verify it explicitly here.
    """
    hl_provider = _load_provider("hermes-local", hermes_local_home)
    ho_provider = _load_provider("holographic", holographic_home)

    if hl_provider is None:
        pytest.fail("hermes-local provider failed to load")
    if ho_provider is None:
        pytest.skip("holographic provider failed to load (dependencies missing)")

    hl_tools = _provider_tool_names(hl_provider)
    ho_tools = _provider_tool_names(ho_provider)

    overlap = hl_tools & ho_tools
    assert overlap == set(), (
        f"hermes-local and holographic must not share tool names. "
        f"Overlap: {overlap}"
    )


# ---------------------------------------------------------------------------------------
# MemoryManager integration: full tool list with hermes-local active
# ---------------------------------------------------------------------------------------

def test_memory_manager_tool_list_only_contains_hermes_local_tools(hermes_local_home, monkeypatch):
    """MemoryManager.get_all_tool_names() contains only hermes-local tools when that provider is active.

    This simulates what AIAgent does at startup: it calls load_memory_provider() and then
    MemoryManager.add_provider() — only one external provider gets added.
    """
    import os
    old_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(hermes_local_home)
    try:
        _agent_root = str(PLUGIN_ROOT)
        if _agent_root not in sys.path:
            sys.path.insert(0, _agent_root)

        from agent.memory_manager import MemoryManager
        from plugins.memory import load_memory_provider

        manager = MemoryManager()
        provider = load_memory_provider("hermes-local")

        if provider is None or not provider.is_available():
            pytest.fail("hermes-local provider should be available")

        manager.add_provider(provider)

        tool_names = manager.get_all_tool_names()

        # When hermes-local is active, fact_store and fact_feedback must not appear
        assert "fact_store" not in tool_names, (
            f"fact_store must not be in MemoryManager tool list with hermes-local. "
            f"Got: {tool_names}"
        )
        assert "fact_feedback" not in tool_names, (
            f"fact_feedback must not be in MemoryManager tool list with hermes-local. "
            f"Got: {tool_names}"
        )

        # memory_query must appear
        assert "memory_query" in tool_names, (
            f"memory_query must be in MemoryManager tool list with hermes-local. "
            f"Got: {tool_names}"
        )

        # memory_get_source must appear
        assert "memory_get_source" in tool_names, (
            f"memory_get_source must be in MemoryManager tool list with hermes-local. "
            f"Got: {tool_names}"
        )
    finally:
        if old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = old_home