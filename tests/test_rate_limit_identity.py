"""Who the rate limiter thinks you are, and the one endpoint it did not cover.

Every limiter in the product keys on one string: the client IP that
core/rate_limit._client_ip returns. The tests that existed proved a limiter
fires; none of them asked whether the caller gets to choose the bucket it fires
against. Behind a proxy, the caller did.

The other half is POST /api/notifications/subscribe, which is the only route
that hands a caller-chosen hostname to the blocking libc resolver. It carried a
deadline that could not fire and no limiter at all.
"""

import time

import anyio
import anyio.to_thread
import pytest

from app.core.config import get_settings
from app.core.rate_limit import _client_ip
from tests.conftest import login, make_user

# The address Caddy would append from the TCP peer: the real caller.
PEER = "198.51.100.7"


@pytest.fixture
def behind_a_proxy(monkeypatch):
    """Run the app as production runs it: RXHIVE_TRUST_PROXY=true.

    get_settings is lru_cached and read all over the app, so the cache is cleared
    on the way in AND on the way out — otherwise every later test in the session
    inherits a proxied app.
    """
    monkeypatch.setenv("RXHIVE_TRUST_PROXY", "true")
    get_settings.cache_clear()
    assert get_settings().trust_proxy is True
    yield
    monkeypatch.delenv("RXHIVE_TRUST_PROXY", raising=False)
    get_settings.cache_clear()
    assert get_settings().trust_proxy is False


def _forwarded(spoofed: str) -> dict:
    """The header as it ARRIVES at the app, not as the attacker sent it.

    A proxy appends, so Caddy turns the caller's `X-Forwarded-For: <spoofed>`
    into `<spoofed>, <peer>`. Reproducing the append is the whole point: a test
    that sent one hop would pass against either implementation, because with a
    single entry the first and last hop are the same string.
    """
    return {"x-forwarded-for": f"{spoofed}, {PEER}"}


class _FakeRequest:
    def __init__(self, headers: dict, peer: str | None):
        self.headers = headers

        class _Client:
            host = peer

        self.client = _Client() if peer is not None else None


def test_the_limiter_key_comes_from_the_hop_the_proxy_appended(behind_a_proxy):
    """Unit-level, because this one line decides every limiter in the product."""
    assert _client_ip(_FakeRequest(_forwarded("203.0.113.9"), "172.18.0.5")) == PEER
    # Padding, empty entries and odd spacing must not shift which hop is chosen.
    assert _client_ip(_FakeRequest({"x-forwarded-for": f" 203.0.113.9 ,, {PEER} "}, None)) == PEER
    # A single hop is the direct-client case: the proxy appended the only entry.
    assert _client_ip(_FakeRequest({"x-forwarded-for": PEER}, "172.18.0.5")) == PEER


def test_without_trust_proxy_the_header_is_ignored_entirely():
    """Default deployment: the peer is the client and X-Forwarded-For is noise."""
    assert _client_ip(_FakeRequest(_forwarded("203.0.113.9"), "192.0.2.44")) == "192.0.2.44"
    assert _client_ip(_FakeRequest({}, None)) == "unknown"


async def test_a_forwarded_header_cannot_mint_a_fresh_login_bucket(client, behind_a_proxy):
    """The finding: one header per request bought unlimited password guesses.

    _client_ip read the FIRST hop, which a proxy leaves entirely caller-supplied,
    so every request landed on `ratelimit:login:<whatever the attacker typed>` —
    a brand new counter that could never reach the limit. There is no account
    lockout anywhere in this product, so the login limiter is the ONLY brute-force
    control, and it was opt-in.
    """
    await make_user("brute@x.com")
    statuses = []
    for i in range(14):
        resp = await client.post(
            "/api/auth/login",
            json={"email": "brute@x.com", "password": "wrong-pass-1"},
            headers=_forwarded(f"203.0.113.{i}"),
        )
        statuses.append(resp.status_code)

    assert 429 in statuses, f"rotating X-Forwarded-For defeated the login limiter: {statuses}"
    assert statuses[-1] == 429


async def test_two_genuinely_different_clients_still_get_their_own_budget(client, behind_a_proxy):
    """The fix must not collapse the limiter into one global bucket.

    That is the failure mode of the obvious over-correction — ignoring the header
    and keying on the peer — which behind a proxy is one address for the whole
    internet, so the first ten failed sign-ins anywhere would lock out everyone.
    """
    await make_user("alice@x.com")
    for _ in range(11):
        await client.post(
            "/api/auth/login",
            json={"email": "alice@x.com", "password": "wrong-pass-1"},
            headers={"x-forwarded-for": "203.0.113.9, 198.51.100.1"},
        )
    # The noisy neighbour is now 429ed. A different peer must be unaffected.
    blocked = await client.post(
        "/api/auth/login",
        json={"email": "alice@x.com", "password": "wrong-pass-1"},
        headers={"x-forwarded-for": "203.0.113.9, 198.51.100.1"},
    )
    assert blocked.status_code == 429

    other = await client.post(
        "/api/auth/login",
        json={"email": "alice@x.com", "password": "wrong-pass-1"},
        headers={"x-forwarded-for": "203.0.113.9, 198.51.100.2"},
    )
    assert other.status_code == 401, "a separate client inherited another client's counter"


async def test_the_resolver_deadline_actually_returns_on_time(monkeypatch, client):
    """The SSRF budget was a comment, not a timeout.

    anyio.to_thread.run_sync defaults to abandon_on_cancel=False, and a cancelled
    scope then WAITS for the worker before the cancellation propagates — so
    move_on_after could not interrupt an uninterruptible getaddrinfo. Measured
    directly: a 1s deadline around a 4s blocking call returned after 4.01s with
    cancelled_caught False. The request held a connection and a DB session for the
    whole resolver timeout, which is exactly what the deadline was added to stop.

    The blocking sleep here stands in for getaddrinfo against a black-holing
    nameserver; it is well over the 2s budget, so an on-time return can only come
    from the deadline firing.
    """
    from app.api import notifications

    def never_resolves(_endpoint: str) -> bool:
        time.sleep(notifications.PUSH_ENDPOINT_RESOLVE_TIMEOUT * 4)
        return True

    monkeypatch.setattr("app.services.push.validate_push_endpoint", never_resolves)

    await make_user("push@x.com")
    await login(client, "push@x.com")

    started = time.monotonic()
    resp = await client.post(
        "/api/notifications/subscribe",
        json={"endpoint": "https://blackhole.example.com/x", "keys": {}},
    )
    elapsed = time.monotonic() - started

    assert resp.status_code == 400, resp.text
    budget = notifications.PUSH_ENDPOINT_RESOLVE_TIMEOUT
    assert elapsed < budget * 1.8, (
        f"the {budget}s resolve budget did not bound the request: returned after {elapsed:.2f}s"
    )


async def test_abandoned_resolver_threads_cannot_drain_the_shared_pool():
    """Why the dedicated limiter exists.

    abandon_on_cancel hands the request back on time but cannot stop the thread,
    which stays in getaddrinfo until libc gives up. Left on anyio's default
    limiter those abandoned threads would consume the same 40 tokens that serve
    every attachment upload, presign, thumbnail and push send, so a caller could
    stall all attachment I/O without ever holding a request open.
    """
    from app.api.notifications import _RESOLVE_LIMITER

    default_limiter = anyio.to_thread.current_default_thread_limiter()
    assert _RESOLVE_LIMITER is not default_limiter
    assert _RESOLVE_LIMITER.total_tokens < default_limiter.total_tokens


async def test_subscribe_is_rate_limited(monkeypatch, client):
    """Nothing bounded how many resolver threads one caller could start.

    The deadline bounds a single call. The limiter is what bounds the rate of
    them, and this route — the only one that hands a caller-chosen hostname to a
    blocking resolver — had none.
    """
    monkeypatch.setattr("app.services.push.validate_push_endpoint", lambda _endpoint: True)

    await make_user("spam@x.com")
    await login(client, "spam@x.com")

    limit = get_settings().rate_limit_push_subscribe
    statuses = []
    for i in range(limit + 4):
        resp = await client.post(
            "/api/notifications/subscribe",
            json={"endpoint": f"https://push.example.com/{i}", "keys": {}},
        )
        statuses.append(resp.status_code)

    assert 429 in statuses, f"subscribe accepted {len(statuses)} calls unthrottled: {statuses}"
    assert statuses[-1] == 429
