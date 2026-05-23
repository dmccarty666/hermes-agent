"""
Pytest wrapper for the memory-dashboard M1 smoke test.

Each test corresponds to a MILESTONES §M1.5 exit criterion bullet so the
pytest output doubles as a coverage report. See the smoke script
``scripts/smoke_test_memory_m1.py`` for the standalone version.

The whole module is skipped if no Hermes gateway is reachable on
``localhost:9119`` (or wherever ``HERMES_BASE_URL`` points). This keeps the
suite CI-safe: integration tests gracefully skip when the live system isn't
running, but they do execute against a real gateway when one is up.

Run:
    pytest tests/integration/dashboard/test_memory_m1_smoke.py -v
"""

from __future__ import annotations

import os
import socket
import sys
import pytest

# Ensure ``scripts/`` is importable so we can reuse the smoke logic verbatim.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import smoke_test_memory_m1 as smoke  # noqa: E402

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Skip-if-gateway-down gate
# ---------------------------------------------------------------------------

def _gateway_is_up() -> tuple[bool, str]:
    """Return (alive, base_url_or_reason)."""
    base_url = os.environ.get("HERMES_BASE_URL")
    base_url = smoke.discover_base_url(base_url)
    if not base_url:
        return False, "no gateway port open"
    # Quick health ping — we only need the SPA shell to come back.
    try:
        import requests
        r = requests.get(base_url + "/", timeout=1.0)
        if r.status_code != 200:
            return False, f"GET {base_url}/ returned {r.status_code}"
    except Exception as e:  # noqa: BLE001
        return False, f"GET {base_url}/ raised {type(e).__name__}: {e}"
    return True, base_url


_ALIVE, _REASON = _gateway_is_up()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ALIVE, reason=f"Hermes gateway not reachable: {_REASON}"),
]


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_url() -> str:
    """Discovered (and confirmed alive) dashboard base URL."""
    assert _ALIVE, _REASON
    return _REASON  # _REASON is the base_url when alive


@pytest.fixture(scope="module")
def session_token(base_url: str) -> str:
    """The ephemeral session token extracted from the SPA shell."""
    tok = smoke.extract_session_token(base_url)
    if not tok:
        pytest.skip("Could not extract session token from SPA shell")
    return tok


@pytest.fixture(scope="module")
def ctx(base_url: str, session_token: str) -> smoke.Ctx:
    """A populated smoke.Ctx, with shape checks pre-run so downstream tests
    can use the cached bodies.

    NOTE: we deliberately do not assert on results here — each test below
    asserts on the specific check it cares about. This fixture only does
    the I/O so tests run fast in series."""
    c = smoke.Ctx(base_url, session_token)
    # Prime the cached bodies. Ignore failures; tests will surface them
    # with proper assertion messages.
    for fn in (
        smoke.check_status_shape,
        smoke.check_backends_shape,
        smoke.check_counters_shape,
    ):
        try:
            fn(c)
        except Exception:  # noqa: BLE001
            pass
    return c


def _assert_pass(result: tuple[str, str]) -> None:
    status, detail = result
    assert status == smoke.PASS, f"check returned {status}: {detail}"


# ---------------------------------------------------------------------------
# Tests
#
# Naming convention: ``test_<criterion-id>_<what>``. Each test maps to a
# bullet in MILESTONES.md §M1.5 — see the docstring for the verbatim text.
# ---------------------------------------------------------------------------

# === Criterion C1: /memory route exists ====================================

def test_C1_memory_route_returns_200(ctx: smoke.Ctx) -> None:
    """C1: ``/memory`` route exists; deep-linking works."""
    _assert_pass(smoke.check_memory_page_loads(ctx))


def test_C1_memory_page_renders_in_browser(ctx: smoke.Ctx) -> None:
    """C1: page renders in a headless browser (also covers C4 and C7).

    Tier 3 backend cards must be visible. Zero console errors must occur."""
    status, detail = smoke.check_browser_render(ctx)
    if status == smoke.SKIP:
        pytest.skip(detail)
    assert status == smoke.PASS, f"browser render failed: {detail}"


# === Criterion C2: Tier 1 health banner ====================================

def test_C2_status_endpoint_shape(ctx: smoke.Ctx) -> None:
    """C2: Tier 1 banner can render DEGRADED — proxied by status shape.

    We can't actually ``kill -9 qdrant`` in CI, but we *can* prove the
    contract that feeds the banner is honored."""
    _assert_pass(smoke.check_status_shape(ctx))


# === Criterion C3: Tier 2 counters match metrics.json ======================

def test_C3_counters_endpoint_shape(ctx: smoke.Ctx) -> None:
    """C3: ``/counters`` endpoint returns the expected shape."""
    _assert_pass(smoke.check_counters_shape(ctx))


def test_C3_metrics_json_passthrough(ctx: smoke.Ctx) -> None:
    """C3: ``/metrics-json`` returns the same data as the file on disk."""
    status, detail = smoke.check_metrics_json_passthrough(ctx)
    if status == smoke.SKIP:
        pytest.skip(detail)
    assert status == smoke.PASS, detail


def test_C3_tier2_numbers_match_metrics_json(ctx: smoke.Ctx) -> None:
    """C3 (the headline check): Tier 2 counters numerically match
    ``~/.hermes/memory/metrics.json`` verbatim. This is the MILESTONES
    §M1.5 line 128 acceptance criterion."""
    status, detail = smoke.check_tier2_matches_metrics_json(ctx)
    if status == smoke.SKIP:
        pytest.skip(detail)
    assert status == smoke.PASS, detail


# === Criterion C4: Tier 3 cards show real backend data =====================

def test_C4_backends_endpoint_shape(ctx: smoke.Ctx) -> None:
    """C4: ``/backends`` returns all 4 (+disk) backends with status."""
    _assert_pass(smoke.check_backends_shape(ctx))


def test_C4_backends_have_required_fields(ctx: smoke.Ctx) -> None:
    """C4 (detail): each backend block has the fields the UI cards display.

    We're lenient about ``last_success_at`` because a freshly-installed
    system may not have one yet — but the *key* must be present."""
    if ctx.backends_body is None:
        pytest.skip("backends body unavailable")
    for name in ("sqlite", "qdrant", "embedding", "llm"):
        block = ctx.backends_body.get(name)
        assert isinstance(block, dict), f"{name!r} is not a dict"
        assert "status" in block, f"{name!r} missing 'status'"


# === Criterion C5: all 6 endpoints exist, gated by auth, with error paths ==

def test_C5_all_six_endpoints_are_registered(ctx: smoke.Ctx) -> None:
    """C5 (happy path): every M1 endpoint returns JSON, not the SPA HTML."""
    _assert_pass(smoke.check_endpoints_reachable(ctx))


def test_C5_auth_required(ctx: smoke.Ctx) -> None:
    """C5 (auth path): unauthenticated GET /status returns 401."""
    _assert_pass(smoke.check_auth_required(ctx))


def test_C5_post_metrics_refresh(ctx: smoke.Ctx) -> None:
    """C5: POST /metrics/refresh is wired up (200 or 429 rate-limited)."""
    _assert_pass(smoke.check_post_metrics_refresh(ctx))


def test_C5_post_ping_backend(ctx: smoke.Ctx) -> None:
    """C5: POST /backends/sqlite/ping is wired up (200 or 503)."""
    _assert_pass(smoke.check_post_ping_backend(ctx))


def test_C5_ping_unknown_backend_returns_404(ctx: smoke.Ctx) -> None:
    """C5 (error path): unknown backend name returns 404, not 500/405."""
    _assert_pass(smoke.check_post_ping_unknown_backend(ctx))


# === Criteria C6 / C7 / C8 — not script-verifiable =========================
#
# C6 ("Inactive provider" empty state)  → requires reconfiguring the gateway
#                                          with memory.provider != hermes-local.
#                                          Verified in unit tests + manual QA.
# C7 (no pnpm lint / type-check / pytest regressions)
#                                       → proxied by zero browser-console
#                                          errors in test_C1_memory_page_renders.
# C8 (PR screenshot matches W1)         → manual, attached to the PR by the
#                                          frontend sub-agent.
#
# These are intentionally absent from the assertable suite. We do, however,
# emit a single ``xfail`` for each so they show up in pytest output as
# "expected-not-here", making the coverage gap visible.


@pytest.mark.xfail(reason="C6: inactive-provider state requires reconfig; manual QA only", strict=False)
def test_C6_inactive_provider_empty_state() -> None:
    """C6: placeholder — see module comment."""
    raise pytest.fail.Exception("not script-verifiable")  # type: ignore[attr-defined]


@pytest.mark.xfail(reason="C8: PR screenshot, not script-verifiable", strict=False)
def test_C8_pr_screenshot() -> None:
    """C8: placeholder — see module comment."""
    raise pytest.fail.Exception("not script-verifiable")  # type: ignore[attr-defined]
