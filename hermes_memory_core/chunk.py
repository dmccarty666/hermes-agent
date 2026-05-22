"""Chunker for Hermes Local Memory.

Splits conversation turns into smaller, embeddable chunks.

The chunker is **deterministic** and **idempotent**: the same input turns
produce the same chunks (and chunk IDs) on every run.

Public surface:
    - chunk_turns(turns, size=512, overlap=128, ...)         → List[dict]
    - chunk_text_for_embedding(text, size=512, overlap=128)  → List[str]
    - _make_chunk_id(...)                                    → 16-hex string
    - Chunk dataclass (kept for backwards compatibility)

Each chunk dict contains:
    chunk_id        16-hex deterministic id
    session_id      source session
    start_turn_id   first turn covered
    end_turn_id     last turn covered
    chunk_type      'conversation' (default) or 'tool_sequence'
    text            actual chunk text (role-labeled)
    text_hash       sha256 hex of `text`
    role_mix        sorted comma-separated unique roles
    turn_count      number of source turns covered
    embed_model     embed model name (passed through)
    chunker_version chunker version tag
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

# Embed model name (matches hermes_memory_core.embed.EMBED_MODEL).
# Imported lazily/defensively to avoid hard import dependency.
_DEFAULT_EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
_DEFAULT_CHUNKER_VERSION = "v1"


# ---------------------------------------------------------------------------
# Token encoder — use tiktoken cl100k_base if available, else cheap fallback
# ---------------------------------------------------------------------------

def _get_encoder():
    """Return a tiktoken encoder, or None for fallback."""
    try:
        import tiktoken  # type: ignore
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_ENC = _get_encoder()


def _encode(text: str) -> List[int]:
    if _ENC is not None:
        return _ENC.encode(text)
    # Fallback: ~4 chars per token approximation; emit integer codes per
    # 4-char window so we can slice/decode round-trip-style.
    return [hash(text[i : i + 4]) & 0xFFFF for i in range(0, len(text), 4)]


def _decode(tokens: Sequence[int]) -> str:
    if _ENC is not None:
        return _ENC.decode(list(tokens))
    # Fallback can't decode arbitrary token ids back to text; the caller
    # should not rely on _decode in that path (we only use it when the
    # tiktoken encoder is present).
    raise RuntimeError("tiktoken encoder unavailable — cannot decode tokens")


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, (len(text) + 3) // 4)


# ---------------------------------------------------------------------------
# Legacy Chunk dataclass (kept for backwards compatibility with any imports)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A discrete embeddable chunk of conversation content (legacy shape)."""
    chunk_id: str
    session_id: str
    start_turn_id: str
    end_turn_id: str
    text: str
    embed_model: str
    qdrant_point_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Deterministic chunk-id helper
# ---------------------------------------------------------------------------

def _make_chunk_id(
    session_id: str,
    start_turn_id: str,
    end_turn_id: str,
    text_hash: str,
    embed_model: str,
    chunker_version: str,
) -> str:
    """Build a stable 16-hex-char chunk id from its identifying fields.

    Inputs are joined with NUL separators (so distinct fields can't collide)
    and SHA-256'd; we return the first 16 hex characters (64 bits) which is
    ample for our scale and matches the existing test contract.
    """
    payload = "\0".join(
        [
            session_id or "",
            start_turn_id or "",
            end_turn_id or "",
            text_hash or "",
            embed_model or "",
            chunker_version or "",
        ]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Token-window splitter for raw text
# ---------------------------------------------------------------------------

def chunk_text_for_embedding(
    text: str,
    size: int = 512,
    overlap: int = 128,
) -> List[str]:
    """Split a raw text string into overlapping token windows.

    Returns a list of strings, each at most ``size`` tokens with ``overlap``
    tokens of carry-over between adjacent windows. Empty input → ``[]``.
    """
    if not text:
        return []
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")

    # Tiktoken path: precise token windows
    if _ENC is not None:
        toks = _ENC.encode(text)
        if len(toks) <= size:
            return [text]
        step = size - overlap
        spans: List[str] = []
        i = 0
        while i < len(toks):
            window = toks[i : i + size]
            spans.append(_ENC.decode(window))
            if i + size >= len(toks):
                break
            i += step
        return spans

    # Fallback: split by whitespace, approximate tokens = words
    words = text.split()
    if len(words) <= size:
        return [text]
    step = size - overlap
    spans = []
    i = 0
    while i < len(words):
        spans.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            break
        i += step
    return spans


# ---------------------------------------------------------------------------
# Turn → chunk builder
# ---------------------------------------------------------------------------

def _format_turn(turn: dict) -> str:
    """Render a turn as a role-labeled line for the chunk text."""
    role = (turn.get("role") or "user").strip()
    content = (turn.get("content") or "").strip()
    return f"{role}: {content}" if content else f"{role}:"


def _is_tool_turn(turn: dict) -> bool:
    role = (turn.get("role") or "").lower()
    return role == "tool" or bool(turn.get("tool_calls_json"))


def _build_chunk_dict(
    *,
    session_id: str,
    start_turn_id: str,
    end_turn_id: str,
    text: str,
    chunk_type: str,
    roles: Iterable[str],
    turn_count: int,
    embed_model: str,
    chunker_version: str,
) -> dict:
    text_hash = _sha256_hex(text)
    chunk_id = _make_chunk_id(
        session_id, start_turn_id, end_turn_id, text_hash, embed_model, chunker_version
    )
    role_mix = ",".join(sorted({(r or "").strip() for r in roles if r is not None}))
    return {
        "chunk_id": chunk_id,
        "session_id": session_id,
        "start_turn_id": start_turn_id,
        "end_turn_id": end_turn_id,
        "chunk_type": chunk_type,
        "text": text,
        "text_hash": text_hash,
        "role_mix": role_mix,
        "turn_count": turn_count,
        "embed_model": embed_model,
        "chunker_version": chunker_version,
    }


def _group_tool_sequences(turns: List[dict]) -> List[List[dict]]:
    """Group consecutive turns into atomic units.

    A *tool sequence* is the maximal run of turns starting at an assistant
    turn that has tool_calls_json and continuing through any immediately
    following tool/assistant turns until a plain user turn arrives. We keep
    these atomic so an assistant→tool→assistant exchange stays in one chunk.

    Returns a list of groups (each group is a list of turns, length ≥ 1).
    """
    groups: List[List[dict]] = []
    i = 0
    while i < len(turns):
        t = turns[i]
        # Tool sequence trigger: assistant turn with tool_calls_json, OR any
        # tool-role turn.
        triggers_tool = (
            (t.get("role") == "assistant" and t.get("tool_calls_json"))
            or t.get("role") == "tool"
        )
        if triggers_tool:
            group = [t]
            j = i + 1
            while j < len(turns):
                nxt = turns[j]
                if nxt.get("role") in ("tool", "assistant") and (
                    nxt.get("role") == "tool"
                    or nxt.get("tool_calls_json")
                    or (j == i + 1 and nxt.get("role") == "assistant")
                ):
                    group.append(nxt)
                    j += 1
                    continue
                break
            groups.append(group)
            i = j
        else:
            groups.append([t])
            i += 1
    return groups


def chunk_turns(
    turns: List[dict],
    size: int = 512,
    overlap: int = 128,
    *,
    max_tokens: Optional[int] = None,  # backwards-compatible alias
    prefer_boundaries: bool = True,
    embed_model: str = _DEFAULT_EMBED_MODEL,
    chunker_version: str = _DEFAULT_CHUNKER_VERSION,
    tool_atomic_max: int = 1024,
) -> List[dict]:
    """Split turns into embeddable chunks.

    Parameters
    ----------
    turns:
        Ordered list of turn dicts. Each turn must have at least
        ``turn_id``, ``role`` and ``content``; ``session_id`` and
        ``sequence`` are used when present.
    size:
        Target chunk size in tokens.
    overlap:
        Overlap between adjacent chunks in tokens.
    max_tokens:
        Back-compat alias for ``size`` (old caller signature). If provided,
        overrides ``size``.
    prefer_boundaries:
        When True (default), chunk boundaries align to turn boundaries
        whenever the next turn fits within ``size``. A single oversized turn
        is split mid-text into multiple sub-chunks (each pointing at the
        same start/end turn id).
    embed_model:
        Embedding model name; folded into chunk_id for stability across
        models.
    chunker_version:
        Chunker version tag; folded into chunk_id so re-chunking produces
        new ids.
    tool_atomic_max:
        Tool sequences (assistant+tool runs) up to this many tokens are
        kept atomic in one chunk (never split). Default 1024.

    Returns
    -------
    List[dict] — see module docstring for the dict shape.
    """
    if not turns:
        return []
    if max_tokens is not None:
        size = max_tokens
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        overlap = max(0, min(overlap, size - 1))

    # Sort by (session_id, sequence) for determinism. Stable sort preserves
    # input order within a session for ties.
    sorted_turns = sorted(
        enumerate(turns),
        key=lambda x: (
            x[1].get("session_id") or "",
            int(x[1].get("sequence") or 0),
            x[0],  # original index as tiebreaker
        ),
    )
    ordered = [t for _, t in sorted_turns]

    chunks: List[dict] = []

    # Group consecutive turns into atomic units (singletons or tool sequences),
    # then pack them under `size` tokens (or `tool_atomic_max` for tool runs).
    groups = _group_tool_sequences(ordered)

    # Pack groups into chunks
    buffer: List[dict] = []        # turns currently in the in-flight chunk
    buffer_token_count = 0

    def flush_buffer() -> None:
        nonlocal buffer, buffer_token_count
        if not buffer:
            return
        text = "\n".join(_format_turn(t) for t in buffer)
        roles = [t.get("role", "") for t in buffer]
        session_id = buffer[0].get("session_id") or ""
        start_turn_id = buffer[0].get("turn_id") or ""
        end_turn_id = buffer[-1].get("turn_id") or ""
        is_tool = any(_is_tool_turn(t) for t in buffer)
        chunk_type = "tool_sequence" if is_tool else "conversation"
        chunks.append(
            _build_chunk_dict(
                session_id=session_id,
                start_turn_id=start_turn_id,
                end_turn_id=end_turn_id,
                text=text,
                chunk_type=chunk_type,
                roles=roles,
                turn_count=len(buffer),
                embed_model=embed_model,
                chunker_version=chunker_version,
            )
        )
        buffer = []
        buffer_token_count = 0

    for group in groups:
        # Render this group as a single text block (for token counting + packing)
        group_text = "\n".join(_format_turn(t) for t in group)
        group_tokens = _count_tokens(group_text)
        is_tool_group = len(group) > 1 or _is_tool_turn(group[0])

        # CASE A: Tool sequence that fits in tool_atomic_max — keep atomic
        if is_tool_group and group_tokens <= tool_atomic_max:
            # If adding to buffer would exceed `size`, flush first
            if buffer and buffer_token_count + group_tokens > size:
                flush_buffer()
            # If even alone the tool sequence is > size but ≤ tool_atomic_max,
            # we still emit it as one chunk (atomicity wins).
            if not buffer and group_tokens > size:
                # Direct emit
                text = group_text
                roles = [t.get("role", "") for t in group]
                chunks.append(
                    _build_chunk_dict(
                        session_id=group[0].get("session_id") or "",
                        start_turn_id=group[0].get("turn_id") or "",
                        end_turn_id=group[-1].get("turn_id") or "",
                        text=text,
                        chunk_type="tool_sequence",
                        roles=roles,
                        turn_count=len(group),
                        embed_model=embed_model,
                        chunker_version=chunker_version,
                    )
                )
                continue
            # Otherwise append to current buffer
            buffer.extend(group)
            buffer_token_count += group_tokens
            continue

        # CASE B: Single turn that's bigger than `size` → split mid-text
        if len(group) == 1 and group_tokens > size:
            # Flush any in-flight buffer first
            flush_buffer()
            t = group[0]
            spans = chunk_text_for_embedding(
                _format_turn(t), size=size, overlap=overlap
            )
            for span in spans:
                chunks.append(
                    _build_chunk_dict(
                        session_id=t.get("session_id") or "",
                        start_turn_id=t.get("turn_id") or "",
                        end_turn_id=t.get("turn_id") or "",
                        text=span,
                        chunk_type="conversation",
                        roles=[t.get("role", "")],
                        turn_count=1,
                        embed_model=embed_model,
                        chunker_version=chunker_version,
                    )
                )
            continue

        # CASE C: Regular packing
        if buffer and buffer_token_count + group_tokens > size:
            # Need to flush. Optionally carry overlap turns into the next buffer
            # for token-level continuity, but we honor `prefer_boundaries` —
            # we restart from the next group rather than mid-turn.
            flush_buffer()
        buffer.extend(group)
        buffer_token_count += group_tokens

    # Final flush
    flush_buffer()

    # If overlap > 0, post-process: add overlap text from each chunk's tail
    # into the next chunk's head (token-level). This preserves the dict shape
    # while satisfying overlap-based recall heuristics.
    if overlap > 0 and len(chunks) >= 2 and _ENC is not None:
        for i in range(len(chunks) - 1):
            cur = chunks[i]
            nxt = chunks[i + 1]
            cur_toks = _ENC.encode(cur["text"])
            tail = cur_toks[-overlap:] if len(cur_toks) > overlap else cur_toks
            tail_text = _ENC.decode(tail)
            new_text = tail_text + "\n" + nxt["text"]
            # Recompute hash + id (deterministic)
            new_hash = _sha256_hex(new_text)
            new_id = _make_chunk_id(
                nxt["session_id"],
                nxt["start_turn_id"],
                nxt["end_turn_id"],
                new_hash,
                nxt["embed_model"],
                nxt["chunker_version"],
            )
            nxt["text"] = new_text
            nxt["text_hash"] = new_hash
            nxt["chunk_id"] = new_id

    return chunks
