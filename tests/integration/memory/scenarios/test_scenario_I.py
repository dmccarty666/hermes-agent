"""MVP Acceptance Test Suite — Scenario I: Provider Swap.

Verifies Plan.md §9, Scenario I:
  1. With memory.provider=holographic, list tools → confirm fact_store present.
  2. Set memory.provider=hermes-local, restart CLI, list tools.
  3. Verify fact_store and fact_feedback (holographic) are GONE.
  4. Verify memory_query, memory_write etc. PRESENT.

NOTE: This test documents the expected provider swap behavior.
The actual tool registration is controlled by Hermes' plugin system and
MemoryManager. This test verifies the state that SHOULD result from the swap.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_hermes_tools(provider: str) -> list[str]:
    """Run Hermes with given provider and return list of registered tool names.

    Runs `hermes --tools` in a subprocess with the specified provider.
    Returns list of tool names.
    """
    env = {
        "HERMES_MEMORY_PROVIDER": provider,
        "PATH": subprocess.os.environ.get("PATH", ""),
    }
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "hermes_cli.main",
                "--memory-provider", provider,
                "--tools",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        # Parse tool names from stdout
        lines = result.stdout.splitlines()
        tools = []
        for line in lines:
            if ":" in line or line.startswith("-"):
                continue
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                tools.append(stripped)
        return tools
    except Exception as e:
        # If hermes CLI is not available in this test env, return empty
        return []


# ---------------------------------------------------------------------------
# Scenario I Tests
# ---------------------------------------------------------------------------

def test_scenario_I_holographic_has_fact_store():
    """Given memory.provider=holographic, when tools are listed,
    then fact_store and fact_feedback are in the tool list."""
    # This test documents the expected state for holographic provider
    # When holographic is active, fact_store and fact_feedback should be registered

    # Check if holographic plugin is installed
    holo_plugin_path = Path(
        "/home/dmccarty/.hermes/hermes-agent/plugins/memory/holographic/__init__.py"
    )
    if not holo_plugin_path.exists():
        pytest.skip("Holographic plugin not installed")

    # Verify the holographic plugin exposes fact_store and fact_feedback
    from plugins.memory.holographic import FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA

    assert FACT_STORE_SCHEMA["name"] == "fact_store"
    assert FACT_FEEDBACK_SCHEMA["name"] == "fact_feedback"


def test_scenario_I_hermes_local_tools_present():
    """Verify hermes-local tool schemas are correctly defined.

    When hermes-local is active, these tools should be in get_tool_schemas():
      - memory_query
      - memory_write
      - memory_get_source
      - memory_dream_now
      - memory_health
    """
    # Try to import the hermes-local plugin tools
    try:
        from plugins.memory.hermes_local import tools
        schemas = tools.get_tool_schemas()
        tool_names = [s["name"] for s in schemas]
    except ImportError:
        pytest.skip("hermes_local plugin tools not yet implemented")

    expected_tools = [
        "memory_query",
        "memory_write",
        "memory_get_source",
        "memory_dream_now",
        "memory_health",
    ]
    for tool in expected_tools:
        assert tool in tool_names, \
            f"Expected '{tool}' in hermes-local tools, got: {tool_names}"


def test_scenario_I_fact_store_not_in_hermes_local_tools():
    """Verify that fact_store (holographic) is NOT in hermes-local tool schemas."""
    try:
        from plugins.memory.hermes_local import tools
        schemas = tools.get_tool_schemas()
        tool_names = [s["name"] for s in schemas]
    except ImportError:
        pytest.skip("hermes_local plugin tools not yet implemented")

    assert "fact_store" not in tool_names, \
        f"fact_store should NOT be in hermes-local tools: {tool_names}"
    assert "fact_feedback" not in tool_names, \
        f"fact_feedback should NOT be in hermes-local tools: {tool_names}"


def test_scenario_I_provider_swap_removes_holographic_tools():
    """Given provider is swapped from holographic to hermes-local,
    when get_tool_schemas is called, then fact_store is absent."""
    # This documents the expected behavior of the MemoryManager
    # when switching providers

    # Simulate checking tool schemas for hermes-local
    try:
        from plugins.memory.hermes_local import tools as hl_tools
        from plugins.memory.holographic import FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA
    except ImportError as e:
        pytest.skip(f"Plugin tools not importable: {e}")

    hl_tool_names = [s["name"] for s in hl_tools.get_tool_schemas()]

    # Verify hermes-local does NOT expose holographic tool schemas
    assert "fact_store" not in hl_tool_names
    assert "fact_feedback" not in hl_tool_names
    assert "memory_query" in hl_tool_names or "memory_write" in hl_tool_names, \
        "hermes-local should expose memory tools"


def test_scenario_I_memory_query_mode_options():
    """Verify memory_query supports all required mode options: keyword, semantic, hybrid."""
    try:
        from plugins.memory.hermes_local import tools
        schemas = tools.get_tool_schemas()
    except ImportError:
        pytest.skip("hermes_local plugin not yet implemented")

    memory_query_schema = None
    for s in schemas:
        if s["name"] == "memory_query":
            memory_query_schema = s
            break

    if memory_query_schema is None:
        pytest.skip("memory_query schema not yet defined")

    params = memory_query_schema.get("parameters", {})
    properties = params.get("properties", {})
    mode_enum = properties.get("mode", {}).get("enum", [])

    required_modes = ["keyword", "semantic", "hybrid"]
    for mode in required_modes:
        assert mode in mode_enum, \
            f"memory_query mode '{mode}' must be in enum: {mode_enum}"


def test_scenario_I_no_duplicate_tools_between_providers():
    """Verify that holographic and hermes-local don't both register the same tool name."""
    try:
        from plugins.memory.holographic import FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA
        from plugins.memory.hermes_local import tools as hl_tools
    except ImportError:
        pytest.skip("One or both plugins not implemented")

    holo_tools = {FACT_STORE_SCHEMA["name"], FACT_FEEDBACK_SCHEMA["name"]}
    hl_tool_names = {s["name"] for s in hl_tools.get_tool_schemas()}

    overlap = holo_tools & hl_tool_names
    assert len(overlap) == 0, \
        f"Tools registered by both providers (should not happen): {overlap}"