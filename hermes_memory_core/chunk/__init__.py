"""Chunk package — re-export from parent-level chunk.py.

The hermes_memory_core/chunk/ directory shadows the hermes_memory_core/chunk.py
module. This __init__.py re-exports the parent-level module's symbols so that
`from hermes_memory_core.chunk import chunk_turns` continues to work.
"""

from __future__ import annotations

import os
import importlib.util

# Bypass the package shadowing by loading chunk.py from its explicit file path
_package_dir = os.path.dirname(os.path.abspath(__file__))
_chunk_py_path = os.path.join(_package_dir, "..", "chunk.py")

_spec = importlib.util.spec_from_file_location("hermes_memory_core._chunk", _chunk_py_path)
assert _spec is not None, f"Could not load spec for {_chunk_py_path}"
_chunk_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_chunk_mod)

chunk_turns = _chunk_mod.chunk_turns
Chunk = _chunk_mod.Chunk

__all__ = ["chunk_turns", "Chunk"]
