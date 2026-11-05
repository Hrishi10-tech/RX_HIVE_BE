"""/api/health shape — including the LiveKit field.

Calls die silently when the SFU is down: the browser shows one generic toast and
nothing else anywhere says why. These tests pin the contract that health reports
LiveKit reachability *without* failing the endpoint, so messaging outages and
call outages stay distinguishable.
"""

import pytest

from app.core.config import Settings
from app.main import settings

LIVEKIT_STATES = {"connected", "unreachable", "not_configured"}


async def test_health_reports_livekit_and_core_dependencies(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    for field in ("status", "version", "service", "timestamp", "database", "redis", "livekit"):
        assert field in body, f"missing {field} in {body}"

    assert body["status"] == "healthy"
    assert body["database"] == "connected"
    assert body["redis"] == "connected"
    assert body["livekit"] in LIVEKIT_STATES
    assert isinstance(body["calls_available"], bool)
    assert body["calls_available"] is (body["livekit"] == "connected")


async def test_unreachable_livekit_is_reported_but_not_fatal(client, monkeypatch):
    """A dead SFU must be visible and must NOT 503 the API — messaging is fine."""
    # Port 1 refuses instantly: no reliance on the probe timeout.
    monkeypatch.setattr(settings, "livekit_health_url", "http://127.0.0.1:1")

    resp = await client.get("/api/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["livekit"] == "unreachable"
    assert body["calls_available"] is False
    assert body["status"] == "healthy"


async def test_path_style_livekit_url_reports_not_configured(client, monkeypatch):
    """ "/livekit" is resolvable by a browser against the site origin, never by
    this process — say so instead of falsely claiming the SFU is down."""
    monkeypatch.setattr(settings, "livekit_health_url", "")
    monkeypatch.setattr(settings, "livekit_url", "/livekit")

    body = (await client.get("/api/health")).json()
    assert body["livekit"] == "not_configured"
    assert body["calls_available"] is False


@pytest.mark.parametrize(
    ("livekit_url", "health_url", "expected"),
    [
        ("ws://localhost:7880", "", "http://localhost:7880"),
        ("wss://sfu.example.com", "", "https://sfu.example.com"),
        ("http://livekit:7880", "", "http://livekit:7880"),
        # Explicit override wins: the browser URL may be unreachable server-side.
        ("/livekit", "http://livekit:7880", "http://livekit:7880"),
        ("/livekit", "", None),
        ("", "", None),
    ],
)
def test_probe_url_translates_websocket_scheme(livekit_url, health_url, expected):
    probe = Settings(livekit_url=livekit_url, livekit_health_url=health_url).livekit_probe_url
    assert probe == expected
