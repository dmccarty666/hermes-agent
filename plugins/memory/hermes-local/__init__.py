"""
Hermes Local Memory Provider — plugin entry point.

Provides memory_dream_now tool plus schema helpers for memory_query / memory_write.
"""

from plugins.memory.hermes_local.tools import (
    get_hermes_local_tool_schemas,
    handle_hermes_local_tool_call,
)

__all__ = [
    "get_hermes_local_tool_schemas",
    "handle_hermes_local_tool_call",
]
