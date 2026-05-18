"""Test plugin scaffold (Story T-001).

Verifies:
  1. hermes-local plugin registers without error.
  2. is_available() returns False when memory.provider is NOT 'hermes-local'.
  3. hermes_memory_core imports cleanly.
  4. All required files exist in the plugin directory.
  5. hermes_memory_core sub-modules are importable.
  6. get_tool_schemas() returns [] (Phase 1 — no tools yet).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def plugin_dir() -> Path:
    """Path to the hermes-local plugin directory."""
    return Path(__file__).parents[3] / "plugins" / "memory" / "hermes-local"


def core_dir() -> Path:
    """Path to hermes_memory_core."""
    return Path(__file__).parents[3] / "hermes_memory_core"


# ---------------------------------------------------------------------------
# Test: Plugin files exist
# ---------------------------------------------------------------------------

PLUGIN_FILES = [
    "__init__.py",
    "plugin.yaml",
    "narrative.py",
    "tools.py",
    "prefetch.py",
    "README.md",
]

@pytest.mark.parametrize("filename", PLUGIN_FILES)
def test_plugin_file_exists(filename: str) -> None:
    p = plugin_dir() / filename
    assert p.exists(), f"Plugin file missing: {p}"
    assert p.stat().st_size > 0, f"Plugin file is empty: {p}"


# ---------------------------------------------------------------------------
# Test: hermes_memory_core imports cleanly
# ---------------------------------------------------------------------------

def test_hermes_memory_core_imports() -> None:
    """Importing hermes_memory_core must not raise."""
    # Add the hermes-agent root to sys.path so the import resolves
    import hermes_memory_core
    assert hasattr(hermes_memory_core, "__version__")


# ---------------------------------------------------------------------------
# Test: Core sub-modules importable
# ---------------------------------------------------------------------------

CORE_MODULES = [
    "hermes_memory_core.store.sqlite",
    "hermes_memory_core.store.qdrant",
    "hermes_memory_core.store.fs",
    "hermes_memory_core.search.hybrid",
    "hermes_memory_core.search.hrr",
    "hermes_memory_core.write.pipeline",
    "hermes_memory_core.write.redaction",
    "hermes_memory_core.source",
    "hermes_memory_core.embed",
    "hermes_memory_core.chunk",
    "hermes_memory_core.dream.worker",
]

@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_submodule_imports(module_name: str) -> None:
    """Every declared core submodule must be importable without error."""
    importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Test: Plugin __init__ register() exists and has correct signature
# ---------------------------------------------------------------------------

def test_plugin_register_exists() -> None:
    """register(ctx) must be defined in the plugin __init__."""
    # Import the plugin module
    spec = importlib.util.spec_from_file_location(
        "hermes_local_plugin",
        str(plugin_dir() / "__init__.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "register"), "register() not found in plugin __init__"
    assert callable(module.register)


def test_hermes_local_provider_class_exists() -> None:
    """HermesLocalProvider class must be defined in the plugin."""
    spec = importlib.util.spec_from_file_location(
        "hermes_local_plugin",
        str(plugin_dir() / "__init__.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "HermesLocalProvider")
    cls = module.HermesLocalProvider
    # Verify it has the key MemoryProvider methods
    assert hasattr(cls, "is_available")
    assert hasattr(cls, "initialize")
    assert hasattr(cls, "get_tool_schemas")
    assert hasattr(cls, "name")


# ---------------------------------------------------------------------------
# Test: get_tool_schemas returns [] (Phase 1 — no tools yet)
# ---------------------------------------------------------------------------

def test_get_tool_schemas_returns_memory_query_schema(tmp_path: Path, monkeypatch) -> None:
    """Phase 2: get_tool_schemas() must return the memory_query schema."""
    fake_home = tmp_path / "hermes_home"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("memory:\n  provider: hermes-local\n")
    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    spec = importlib.util.spec_from_file_location(
        "hermes_local_plugin",
        str(plugin_dir() / "__init__.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    provider = module.HermesLocalProvider()
    schemas = provider.get_tool_schemas()
    assert len(schemas) == 2, f"Expected 2 schemas for Phase 4, got {len(schemas)}: {schemas}"
    names = {s["name"] for s in schemas}
    assert "memory_query" in names, f"memory_query schema missing, got: {names}"
    assert "memory_get_source" in names, f"memory_get_source schema missing, got: {names}"


# ---------------------------------------------------------------------------
# Test: is_available() returns False without the config set
# ---------------------------------------------------------------------------

def test_is_available_false_without_config(tmp_path: Path, monkeypatch) -> None:
    """is_available() must return False when memory.provider != 'hermes-local'."""
    # Create a fake hermes home
    fake_home = tmp_path / "hermes_home"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("memory:\n  provider: holographic\n")

    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    spec = importlib.util.spec_from_file_location(
        "hermes_local_plugin",
        str(plugin_dir() / "__init__.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    provider = module.HermesLocalProvider()
    assert provider.is_available() is False, \
        "is_available() must be False when provider is not 'hermes-local'"


# ---------------------------------------------------------------------------
# Test: is_available() returns True with memory.provider=hermes-local
# ---------------------------------------------------------------------------

def test_is_available_true_with_config(tmp_path: Path, monkeypatch) -> None:
    """is_available() must return True when memory.provider == 'hermes-local'."""
    fake_home = tmp_path / "hermes_home"
    fake_home.mkdir()
    (fake_home / "config.yaml").write_text("memory:\n  provider: hermes-local\n")

    monkeypatch.setenv("HERMES_HOME", str(fake_home))

    spec = importlib.util.spec_from_file_location(
        "hermes_local_plugin",
        str(plugin_dir() / "__init__.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    provider = module.HermesLocalProvider()
    assert provider.is_available() is True, \
        "is_available() must be True when memory.provider is 'hermes-local'"


# ---------------------------------------------------------------------------
# Test: Core directory tree matches the DoD
# ---------------------------------------------------------------------------

CORE_DIRS = [
    "store",
    "search",
    "write",
    "chunk",
    "embed",
    "dream",
    "source",
]

@pytest.mark.parametrize("dirname", CORE_DIRS)
def test_core_dir_exists(dirname: str) -> None:
    p = core_dir() / dirname
    assert p.is_dir(), f"Core directory missing: {p}"
    assert any(p.iterdir()), f"Core directory is empty: {p}"


def test_dream_prompts_dir() -> None:
    """dream/prompts/ directory must exist."""
    p = core_dir() / "dream" / "prompts"
    assert p.is_dir(), f"dream/prompts directory missing: {p}"


# ---------------------------------------------------------------------------
# Test: redaction module has expected public API
# ---------------------------------------------------------------------------

def test_redaction_module_api() -> None:
    """Redaction module must expose scan() and RedactionResult."""
    from hermes_memory_core.write import redaction as red_mod
    assert hasattr(red_mod, "scan")
    assert callable(red_mod.scan)
    assert hasattr(red_mod, "RedactionResult")
    assert hasattr(red_mod, "Redactor")
    assert hasattr(red_mod, "default_redactor")


def test_redaction_scan_returns_redaction_result() -> None:
    """scan() must return a RedactionResult with the expected fields."""
    from hermes_memory_core.write.redaction import scan
    result = scan("Hello world")
    assert hasattr(result, "redacted_content")
    assert hasattr(result, "hits")
    assert hasattr(result, "fired")


def test_redaction_catches_aws_key() -> None:
    """scan() must detect and redact AWS access keys."""
    from hermes_memory_core.write.redaction import scan
    result = scan("My AWS key is AKIAFAKEFAKEFAKEFAKE and secret xyz")
    assert result.fired is True
    assert "AKIAFAKEFAKEFAKEFAKE" not in result.redacted_content
    assert "[REDACTED:aws_access_key]" in result.redacted_content


def test_redaction_catches_openai_key() -> None:
    """scan() must detect and redact OpenAI keys (sk-...)."""
    from hermes_memory_core.write.redaction import scan
    result = scan("OpenAI key: sk-test-AbCdEfGhIjKlMnOpQrStUvWxY")
    assert result.fired is True
    assert "sk-test-" not in result.redacted_content


def test_redaction_catches_github_token() -> None:
    """scan() must detect and redact GitHub tokens."""
    from hermes_memory_core.write.redaction import scan
    # ghp_ prefix + exactly 36 chars = 40-char token
    token = "ghp_" + "a" * 36
    result = scan(f"GitHub token: {token}")
    assert result.fired is True
    assert token not in result.redacted_content
    assert "[REDACTED:github_token]" in result.redacted_content


def test_redaction_catches_ssn() -> None:
    """scan() must detect and redact SSN patterns."""
    from hermes_memory_core.write.redaction import scan
    result = scan("SSN: 123-45-6789")
    assert result.fired is True
    assert "123-45-6789" not in result.redacted_content


def test_redaction_catches_credit_card() -> None:
    """scan() must detect and redact credit card numbers."""
    from hermes_memory_core.write.redaction import scan
    result = scan("Card: 4532015112830366")  # Luhn-valid test Visa
    assert result.fired is True
    assert "4532015112830366" not in result.redacted_content


def test_redaction_clean_content_no_hit() -> None:
    """scan() must not flag clean content."""
    from hermes_memory_core.write.redaction import scan
    result = scan("This is a normal conversation about Python programming.")
    assert result.fired is False
    assert result.hits == []
    assert result.redacted_content == "This is a normal conversation about Python programming."