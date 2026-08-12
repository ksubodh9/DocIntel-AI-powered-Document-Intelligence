"""
CORS preflight regression tests.

Background — the bug these guard against
----------------------------------------
The BYOK feature (frontend `src/lib/byok.js`) attaches per-request LLM
credential headers to EVERY API call:

    X-LLM-Use-Default, X-LLM-Provider, X-LLM-Api-Key, X-LLM-Model

A browser will not send those on a cross-origin request until its CORS
*preflight* (an `OPTIONS` carrying `Access-Control-Request-Headers`) succeeds.
If the header is missing from the backend's `allow_headers`, Starlette's
CORSMiddleware answers the preflight with **400 Disallowed CORS headers**, and
the browser reports it to the client as a generic "Network Error" — which is
exactly how the upload failure first surfaced.

These tests exercise the middleware directly (no auth needed — a preflight is
answered by CORSMiddleware before it ever reaches a route handler) and assert
that each BYOK header is permitted. If someone adds a new `X-LLM-*` header to
byok.js without updating `app/main.py`, the matching case here goes red.

`http://localhost:5173` is used as the origin because it is in the default
allowed-origins list baked into `app/main.py` (the Vite dev server).
"""

import pytest

API = "/api/v1"

ALLOWED_ORIGIN = "http://localhost:5173"

# Every custom header the BYOK client can send (byok.js::getByokHeaders).
BYOK_HEADERS = [
    "x-llm-use-default",
    "x-llm-provider",
    "x-llm-api-key",
    "x-llm-model",
]

# Endpoints that receive BYOK headers in normal use.
PREFLIGHTED_PATHS = [f"{API}/upload", f"{API}/documents", f"{API}/chat"]


def _preflight(client, path, request_headers, origin=ALLOWED_ORIGIN, method="POST"):
    """Send a CORS preflight (OPTIONS) the way a browser would."""
    return client.options(
        path,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": request_headers,
        },
    )


class TestByokPreflight:
    @pytest.mark.parametrize("path", PREFLIGHTED_PATHS)
    def test_upload_preflight_allows_all_byok_headers_together(self, client, path):
        """The full BYOK header set (as sent alongside Authorization + Content-Type)
        must pass preflight on every endpoint that uses it."""
        requested = "authorization,content-type," + ",".join(BYOK_HEADERS)
        r = _preflight(client, path, requested)

        assert r.status_code == 200, (
            f"Preflight for {path} rejected ({r.status_code}). A BYOK header is "
            f"missing from allow_headers in app/main.py. Body: {r.text!r}"
        )
        allowed = r.headers.get("access-control-allow-headers", "").lower()
        for h in BYOK_HEADERS:
            assert h in allowed, f"{h} not echoed in Access-Control-Allow-Headers"

    @pytest.mark.parametrize("header", BYOK_HEADERS)
    def test_each_byok_header_individually_allowed(self, client, header):
        """Pinpoint which specific header regressed if one is dropped."""
        r = _preflight(client, f"{API}/upload", f"authorization,content-type,{header}")
        assert r.status_code == 200, (
            f"Preflight rejected header {header!r} ({r.status_code}) — add it to "
            f"allow_headers in app/main.py to match frontend byok.js."
        )
        assert header in r.headers.get("access-control-allow-headers", "").lower()

    def test_allowed_origin_is_echoed(self, client):
        """Sanity: a permitted origin comes back in Access-Control-Allow-Origin."""
        r = _preflight(client, f"{API}/upload", "authorization,content-type")
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN


class TestPreflightNegativeControls:
    """These prove the assertions above are meaningful — i.e. the middleware
    really does reject things — so a passing suite isn't a false positive."""

    def test_unknown_header_is_rejected(self, client):
        r = _preflight(client, f"{API}/upload", "authorization,x-totally-made-up")
        assert r.status_code == 400  # Starlette: "Disallowed CORS headers"

    def test_disallowed_origin_not_echoed(self, client):
        r = _preflight(
            client,
            f"{API}/upload",
            "authorization,content-type",
            origin="http://evil.example.com",
        )
        # Either rejected, or answered without granting the origin — never echoed.
        assert r.headers.get("access-control-allow-origin") != "http://evil.example.com"
