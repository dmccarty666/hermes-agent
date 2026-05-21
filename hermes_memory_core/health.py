"""Health check functions for hermes-local memory components.

Provides pure-check functions (no HTTP/FastAPI dependency) for:
  - SQLite connectivity and WAL status
  - Qdrant cluster health and collection count
  - LMS embedding endpoint reachability
  - Dreamer LLM reachability
  - Disk space check
  - Rolled-up overall status

These functions are used by both the gateway health routes and any direct
caller that needs component-level diagnostics.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any, TypedDict

import requests

# Default endpoints
DEFAULT_EMBEDDING_ENDPOINT = "http://192.168.2.105:1235"
DEFAULT_LLM_ENDPOINT = "http://192.168.2.105:1234"


class SQLiteHealth(TypedDict, total=False):
    status: str
    path: str
    journal_mode: str
    size_bytes: int
    message: str


class QdrantHealth(TypedDict, total=False):
    status: str
    collections_count: int
    collections: list[str]
    message: str


class EmbeddingHealth(TypedDict, total=False):
    status: str
    endpoint: str
    model: str
    message: str


class LLMHealth(TypedDict, total=False):
    status: str
    endpoint: str
    model: str
    message: str


class DiskHealth(TypedDict, total=False):
    status: str
    path: str
    free_bytes: int
    free_gb: float
    message: str


class RolledUpHealth(TypedDict, total=False):
    overall: str
    sqlite: SQLiteHealth
    qdrant: QdrantHealth
    embedding: EmbeddingHealth
    llm: LLMHealth
    disk: DiskHealth


def check_sqlite(db_path: Path | str) -> SQLiteHealth:
    """Check SQLite connectivity and WAL status.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        SQLiteHealth dict with status, path, journal_mode, size_bytes.
    """
    db_path = Path(db_path)
    result: SQLiteHealth = {
        "status": "error",
        "path": str(db_path),
        "journal_mode": "unknown",
        "size_bytes": 0,
    }
    try:
        if not db_path.exists():
            result["message"] = "Database file not found"
            return result
        conn = sqlite3.connect(str(db_path), timeout=3)
        try:
            cursor = conn.execute("PRAGMA journal_mode")
            row = cursor.fetchone()
            journal_mode = (row[0] or "unknown") if row else "unknown"
            cursor.close()
            cursor2 = conn.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
            row2 = cursor2.fetchone()
            size_bytes = (row2[0] or 0) if row2 else 0
            cursor2.close()
            result["status"] = "ok"
            result["journal_mode"] = journal_mode
            result["size_bytes"] = size_bytes
        finally:
            conn.close()
    except Exception as e:
        result["message"] = str(e)
    return result


def check_qdrant(client: Any) -> QdrantHealth:
    """Check Qdrant cluster health and collection count.

    Args:
        client: QdrantClient instance.

    Returns:
        QdrantHealth dict with status, collections_count, collections list.
    """
    result: QdrantHealth = {
        "status": "error",
        "collections_count": 0,
        "collections": [],
    }
    try:
        collections = client.get_collections()
        names = [c.name for c in collections.collections] if collections else []
        result["status"] = "ok"
        result["collections_count"] = len(names)
        result["collections"] = names
    except Exception as e:
        result["message"] = str(e)
    return result


def check_embedding(endpoint: str, timeout: int = 5) -> EmbeddingHealth:
    """Check LMS embedding endpoint reachability and model info.

    Args:
        endpoint: Base URL for the LMS embeddings API.
        timeout: Request timeout in seconds.

    Returns:
        EmbeddingHealth dict with status, endpoint, model.
    """
    result: EmbeddingHealth = {
        "status": "error",
        "endpoint": endpoint,
        "model": "unknown",
    }
    try:
        # LMS OpenAI-compatible /v1/models endpoint
        for path in ("/v1/models", "/models"):
            try:
                resp = requests.get(f"{endpoint.rstrip('/')}{path}", timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    # OpenAI-compatible: {"object": "list", "data": [{"id": "...", ...}]}
                    if isinstance(data, dict) and "data" in data:
                        models = data["data"]
                        if isinstance(models, list) and len(models) > 0:
                            result["model"] = models[0].get("id", "unknown")
                            result["status"] = "ok"
                            return result
                    # LMS may return a flat list of model strings or {"model": "..."}
                    if isinstance(data, dict) and "model" in data:
                        result["model"] = data["model"]
                        result["status"] = "ok"
                        return result
                    if isinstance(data, list) and len(data) > 0:
                        first = data[0]
                        result["model"] = first.get("id", first) if isinstance(first, dict) else str(first)
                        result["status"] = "ok"
                        return result
                    break
            except Exception:
                pass
        result["message"] = "No model info available"
    except requests.exceptions.ConnectTimeout:
        result["message"] = "Connection timeout"
    except requests.exceptions.ConnectionError:
        result["message"] = "Connection error"
    except Exception as e:
        result["message"] = str(e)
    return result


def check_llm(endpoint: str, timeout: int = 10) -> LLMHealth:
    """Check dreamer LLM (Qwen3.6-35B) reachability.

    Args:
        endpoint: Base URL for the LLM API.
        timeout: Request timeout in seconds.

    Returns:
        LLMHealth dict with status, endpoint, model.
    """
    result: LLMHealth = {
        "status": "error",
        "endpoint": endpoint,
        "model": "unknown",
    }
    try:
        # Try models endpoint first
        try:
            resp = requests.get(f"{endpoint.rstrip('/')}/models", timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    model = data[0].get("model", "unknown")
                elif isinstance(data, dict):
                    model = data.get("model", data.get("model_name", "unknown"))
                else:
                    model = "unknown"
                result["status"] = "ok"
                result["model"] = model
                return result
        except Exception:
            pass
        # Fallback: light chat completions probe
        resp = requests.post(
            f"{endpoint.rstrip('/')}/chat/completions",
            json={
                "model": "qwen3.6-35b-instruct",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0.1,
            },
            timeout=timeout,
        )
        if resp.status_code in (200, 201):
            result["status"] = "ok"
            # Try to extract model name from response
            try:
                data = resp.json()
                result["model"] = data.get("model", "qwen3.6-35b-instruct")
            except Exception:
                result["model"] = "qwen3.6-35b-instruct"
        else:
            result["message"] = f"HTTP {resp.status_code}"
    except requests.exceptions.ConnectTimeout:
        result["message"] = "Connection timeout"
    except requests.exceptions.ConnectionError:
        result["message"] = "Connection error"
    except Exception as e:
        result["message"] = str(e)
    return result


def check_disk(path: Path | str = Path.home() / ".hermes") -> DiskHealth:
    """Check disk space for the given path.

    Args:
        path: Path to check disk usage for.

    Returns:
        DiskHealth dict with status, path, free_bytes, free_gb.
    """
    result: DiskHealth = {
        "status": "error",
        "path": str(path),
        "free_bytes": 0,
        "free_gb": 0.0,
    }
    try:
        usage = shutil.disk_usage(str(path))
        result["status"] = "ok"
        result["free_bytes"] = usage.free
        result["free_gb"] = round(usage.free / (1024**3), 2)
    except Exception as e:
        result["message"] = str(e)
    return result


def health_check(
    db_path: Path | str | None = None,
    qdrant_client: Any = None,
    embedding_endpoint: str = DEFAULT_EMBEDDING_ENDPOINT,
    llm_endpoint: str = DEFAULT_LLM_ENDPOINT,
) -> RolledUpHealth:
    """Return rolled-up health of all memory components.

    Args:
        db_path: Path to hermes-local SQLite DB.
        qdrant_client: QdrantClient instance.
        embedding_endpoint: LMS embeddings base URL.
        llm_endpoint: LLM base URL.

    Returns:
        RolledUpHealth dict with overall status and per-component dicts.
    """
    if db_path is None:
        db_path = str(Path.home() / ".hermes" / "memory" / "index" / "memory.sqlite")

    # Run individual checks
    sqlite_result = check_sqlite(db_path)
    disk_result = check_disk(Path(db_path).parent.parent if db_path else Path.home() / ".hermes")

    qdrant_result: QdrantHealth = {"status": "error", "collections_count": 0, "collections": []}
    if qdrant_client is not None:
        qdrant_result = check_qdrant(qdrant_client)

    embedding_result = check_embedding(embedding_endpoint)
    llm_result = check_llm(llm_endpoint)

    # Determine overall status
    component_statuses = [
        sqlite_result["status"],
        qdrant_result["status"],
        embedding_result["status"],
        llm_result["status"],
        disk_result["status"],
    ]
    if all(s == "ok" for s in component_statuses):
        overall = "ok"
    elif any(s == "error" for s in component_statuses):
        overall = "error"
    else:
        overall = "degraded"

    result: RolledUpHealth = {
        "overall": overall,
        "sqlite": sqlite_result,  # type: ignore[assignment]
        "qdrant": qdrant_result,  # type: ignore[assignment]
        "embedding": embedding_result,  # type: ignore[assignment]
        "llm": llm_result,  # type: ignore[assignment]
        "disk": disk_result,  # type: ignore[assignment]
    }
    return result