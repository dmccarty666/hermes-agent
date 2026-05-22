"""Chunk package — re-export from parent-level chunk.py.

The hermes_memory_core/chunk/ directory shadows hermes_memory_core/chunk.py.
This __init__.py re-exports the parent-level module's symbols so that
``from hermes_memory_core.chunk import chunk_turns`` continues to work.
"""

from __future__ import annotations

import os
import sys
import importlib.util

# Bypass package shadowing by loading chunk.py from its explicit file path.
_package_dir = os.path.dirname(os.path.abspath(__file__))
_chunk_py_path = os.path.join(_package_dir, "..", "chunk.py")

_module_name = "hermes_memory_core._chunk"
_spec = importlib.util.spec_from_file_location(_module_name, _chunk_py_path)
assert _spec is not None, f"Could not load spec for {_chunk_py_path}"
_chunk_mod = importlib.util.module_from_spec(_spec)
# Register in sys.modules BEFORE exec_module so @dataclass / typing helpers
# can look the module up by __module__ during class creation.
sys.modules[_module_name] = _chunk_mod
assert _spec.loader is not None
_spec.loader.exec_module(_chunk_mod)

# Re-export public symbols
chunk_turns = _chunk_mod.chunk_turns
Chunk = _chunk_mod.Chunk
_make_chunk_id = _chunk_mod._make_chunk_id
chunk_text_for_embedding = _chunk_mod.chunk_text_for_embedding

__all__ = ["chunk_turns", "Chunk", "_make_chunk_id", "chunk_text_for_embedding"]
