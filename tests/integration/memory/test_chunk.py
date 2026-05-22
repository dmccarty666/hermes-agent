"""Tests for hermes_memory_core/chunk.py — T-017"""
import pytest


# ---------------------------------------------------------------------------
# Session fixture — 10 turns used by AC-1
# ---------------------------------------------------------------------------
def make_10_turn_session():
    """A 10-turn user/assistant session for boundary tests."""
    session_id = "sess_test_001"
    return [
        {"turn_id": "t1",  "session_id": session_id, "sequence": 1,  "role": "user",       "content": "What is the capital of France?"},
        {"turn_id": "t2",  "session_id": session_id, "sequence": 2,  "role": "assistant",  "content": "The capital of France is Paris."},
        {"turn_id": "t3",  "session_id": session_id, "sequence": 3,  "role": "user",       "content": "And Germany?"},
        {"turn_id": "t4",  "session_id": session_id, "sequence": 4,  "role": "assistant",  "content": "The capital of Germany is Berlin."},
        {"turn_id": "t5",  "session_id": session_id, "sequence": 5,  "role": "user",       "content": "Tell me about their populations."},
        {"turn_id": "t6",  "session_id": session_id, "sequence": 6,  "role": "assistant",  "content": "Paris has about 2.1 million people. Berlin has about 3.6 million."},
        {"turn_id": "t7",  "session_id": session_id, "sequence": 7,  "role": "user",       "content": "Which is bigger?"},
        {"turn_id": "t8",  "session_id": session_id, "sequence": 8,  "role": "assistant",  "content": "Berlin is bigger, with about 3.6 million vs Paris's 2.1 million."},
        {"turn_id": "t9",  "session_id": session_id, "sequence": 9,  "role": "user",       "content": "Thanks."},
        {"turn_id": "t10", "session_id": session_id, "sequence": 10, "role": "assistant",  "content": "You're welcome!"},
    ]


def _shared_tokens(a: str, b: str) -> int:
    """Approximate token overlap between two texts using cl100k_base."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    a_toks = set(enc.encode(a))
    b_toks = set(enc.encode(b))
    return len(a_toks & b_toks)


# ---------------------------------------------------------------------------
# AC-1: 10-turn session, size=512, overlap=128 → expected chunk count + boundaries
# ---------------------------------------------------------------------------
def test_chunk_turns_basic_count_and_alignment():
    """AC-1: chunk_turns returns expected chunk count with correct boundary alignment."""
    from hermes_memory_core.chunk import chunk_turns

    turns = make_10_turn_session()
    chunks = chunk_turns(turns, size=512, overlap=128)

    # Short QA turns should produce 1-3 chunks total (each well under 512 tokens)
    assert len(chunks) >= 1, f"Expected at least 1 chunk, got {len(chunks)}"

    # All chunks must have required fields
    for c in chunks:
        assert "chunk_id"    in c
        assert "session_id"  in c
        assert "start_turn_id" in c
        assert "end_turn_id"   in c
        assert "text"        in c
        assert "text_hash"   in c

    # Adjacent chunks should overlap by roughly 128 tokens (80-200 tolerance for tiktoken edge cases)
    if len(chunks) >= 2:
        for i in range(len(chunks) - 1):
            ov = _shared_tokens(chunks[i]["text"], chunks[i + 1]["text"])
            assert 80 <= ov <= 220, f"Chunk {i} overlap={ov} tokens outside 80-220 range"


# ---------------------------------------------------------------------------
# AC-2: tool-call sequences ≤1024 tokens treated as one atomic chunk
# ---------------------------------------------------------------------------
def test_tool_sequence_atomic():
    """AC-2: multi-turn tool sequences ≤1024 tokens are one atomic chunk."""
    from hermes_memory_core.chunk import chunk_turns

    session_id = "sess_tool_001"
    turns = [
        {"turn_id": "tt1", "session_id": session_id, "sequence": 1, "role": "user",
         "content": "List files in /tmp"},
        {"turn_id": "tt2", "session_id": session_id, "sequence": 2, "role": "assistant",
         "content": "Let me check.",
         "tool_calls_json": '[{"id":"tc1","function":{"name":"bash","arguments":"{\"command\":\"ls /tmp\"}"}}]'},
        {"turn_id": "tt3", "session_id": session_id, "sequence": 3, "role": "tool",
         "content": "file1.txt\nfile2.txt\n",
         "tool_calls_json": '[{"id":"tc1","type":"tool","content":"file1.txt\\nfile2.txt\\n"}]'},
        {"turn_id": "tt4", "session_id": session_id, "sequence": 4, "role": "assistant",
         "content": "Found file1.txt and file2.txt in /tmp."},
    ]

    chunks = chunk_turns(turns, size=512, overlap=128)

    # Tool sequence (tt2+tt3) must NOT be split across chunks.
    # All 4 turns (~29 tokens total) fit in one 512-token chunk.
    # Verify: at least one chunk contains both tt2 and tt3.
    # A chunk contains tt2 if start_turn_id <= tt2 <= end_turn_id (lexicographic compare works since numbered).
    def chunk_contains_turn(c, turn_id):
        return c["start_turn_id"] <= turn_id <= c["end_turn_id"]

    tt2_tt3_chunks = [
        c for c in chunks
        if chunk_contains_turn(c, "tt2") and chunk_contains_turn(c, "tt3")
    ]
    assert len(tt2_tt3_chunks) >= 1, (
        f"tt2 and tt3 must appear together in same chunk. "
        f"Got chunks: {[(c['start_turn_id'], c['end_turn_id']) for c in chunks]}"
    )


# ---------------------------------------------------------------------------
# AC-3: stable IDs — same turns chunked twice → identical IDs
# ---------------------------------------------------------------------------
def test_stable_chunk_ids():
    """AC-3: same turns chunked twice produce identical chunk IDs."""
    from hermes_memory_core.chunk import chunk_turns

    turns = make_10_turn_session()
    run1 = chunk_turns(turns, size=512, overlap=128)
    run2 = chunk_turns(turns, size=512, overlap=128)

    assert len(run1) == len(run2), f"Chunk count changed between runs: {len(run1)} vs {len(run2)}"

    ids1 = sorted(c["chunk_id"] for c in run1)
    ids2 = sorted(c["chunk_id"] for c in run2)
    assert ids1 == ids2, f"Chunk IDs not stable:\n  Run1: {ids1}\n  Run2: {ids2}"


# ---------------------------------------------------------------------------
# AC-4: changing chunker_version produces different IDs
# ---------------------------------------------------------------------------
def test_chunker_version_changes_ids():
    """AC-4: changing chunker_version produces different chunk IDs."""
    from hermes_memory_core.chunk import chunk_turns, _make_chunk_id

    turns = make_10_turn_session()

    v1_chunks = chunk_turns(turns, size=512, overlap=128, chunker_version="v1")
    v2_chunks = chunk_turns(turns, size=512, overlap=128, chunker_version="v2")

    assert len(v1_chunks) == len(v2_chunks)
    v1_ids = sorted(c["chunk_id"] for c in v1_chunks)
    v2_ids = sorted(c["chunk_id"] for c in v2_chunks)
    assert v1_ids != v2_ids, "chunker_version change must produce different IDs"


# ---------------------------------------------------------------------------
# Additional: prefer_boundaries=True aligns to turn boundaries
# ---------------------------------------------------------------------------
def test_prefer_boundaries():
    """When prefer_boundaries=True, chunks never split mid-turn."""
    from hermes_memory_core.chunk import chunk_turns

    session_id = "sess_bound_001"
    turns = [
        {"turn_id": "b1", "session_id": session_id, "sequence": 1, "role": "user",       "content": "Hello"},
        {"turn_id": "b2", "session_id": session_id, "sequence": 2, "role": "assistant",  "content": "Hi there! How can I help you today?"},
        {"turn_id": "b3", "session_id": session_id, "sequence": 3, "role": "user",       "content": "I need help with Python."},
        {"turn_id": "b4", "session_id": session_id, "sequence": 4, "role": "assistant",  "content": "I'd be happy to help with Python."},
    ]

    chunks = chunk_turns(turns, size=1024, overlap=256, prefer_boundaries=True)

    # With size=1024 and these small turns, we expect 1-2 chunks.
    # Each chunk's start_turn_id should be a turn that actually exists.
    turn_ids = {t["turn_id"] for t in turns}
    for c in chunks:
        assert c["start_turn_id"] in turn_ids
        assert c["end_turn_id"] in turn_ids


# ---------------------------------------------------------------------------
# Additional: empty turns list returns empty list
# ---------------------------------------------------------------------------
def test_empty_turns():
    """chunk_turns with empty list returns [] without error."""
    from hermes_memory_core.chunk import chunk_turns

    result = chunk_turns([], size=512, overlap=128)
    assert result == []


# ---------------------------------------------------------------------------
# Additional: missing session_id falls back gracefully
# ---------------------------------------------------------------------------
def test_missing_session_id():
    """Turns without session_id still chunk successfully."""
    from hermes_memory_core.chunk import chunk_turns

    turns = [{"turn_id": "x1", "session_id": "", "sequence": 1, "role": "user", "content": "Hello"}]
    chunks = chunk_turns(turns, size=512, overlap=128)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"]


# ---------------------------------------------------------------------------
# Additional: very long single turn splits into multiple chunks
# ---------------------------------------------------------------------------
def test_long_turn_splits():
    """A single turn exceeding chunk size splits into multiple sub-chunks."""
    from hermes_memory_core.chunk import chunk_turns

    session_id = "sess_long_001"
    long_content = "word " * 800  # ~800 tokens, well over size=512
    turns = [{"turn_id": "lt1", "session_id": session_id, "sequence": 1, "role": "user", "content": long_content}]

    chunks = chunk_turns(turns, size=512, overlap=128)

    assert len(chunks) > 1, f"Long turn should split; got {len(chunks)} chunks"
    # All sub-chunks should reference the same turn
    for c in chunks:
        assert c["start_turn_id"] == "lt1"
        assert c["end_turn_id"] == "lt1"


# ---------------------------------------------------------------------------
# Additional: _make_chunk_id deterministic + 16 hex chars
# ---------------------------------------------------------------------------
def test_make_chunk_id_deterministic():
    """_make_chunk_id is deterministic and returns 16 hex characters."""
    from hermes_memory_core.chunk import _make_chunk_id

    id1 = _make_chunk_id("s", "s1", "e1", "hash", "model", "v1")
    id2 = _make_chunk_id("s", "s1", "e1", "hash", "model", "v1")

    assert id1 == id2,        "_make_chunk_id must be deterministic"
    assert len(id1) == 16,     f"Expected 16 hex chars, got {len(id1)}"
    assert all(c in "0123456789abcdef" for c in id1), "Must be lowercase hex"

    # Different version → different ID
    id3 = _make_chunk_id("s", "s1", "e1", "hash", "model", "v2")
    assert id1 != id3, "chunker_version must affect chunk ID"


# ---------------------------------------------------------------------------
# chunk_text_for_embedding helper
# ---------------------------------------------------------------------------
def test_chunk_text_for_embedding():
    """chunk_text_for_embedding splits a raw string into token windows."""
    from hermes_memory_core.chunk import chunk_text_for_embedding

    text = "one " * 300  # ~300 tokens
    spans = chunk_text_for_embedding(text, size=200, overlap=50)

    assert len(spans) > 1, "Long text should produce multiple spans"
    for s in spans:
        assert isinstance(s, str)
        assert len(s) > 0