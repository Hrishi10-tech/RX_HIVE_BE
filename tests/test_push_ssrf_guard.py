"""The push send refuses a connection that lands on a non-public address.

validate_push_endpoint resolves the hostname when the browser subscribes;
pywebpush resolves it again on every send, days later, in a separate lookup. A
caller who controls DNS for their own hostname can answer publicly at subscribe
time and privately at send time, and because the endpoint is stored and reused
that is a POST into the private network on every message, indefinitely.

No amount of care at subscribe time closes that. What closes it is asking the
socket where it actually went — after connect, before anything is sent — which is
what these tests exercise against a real listening socket on loopback.
"""

import http.server
import socket
import threading

import pytest
import requests

import app.services.push as push_mod
from app.services.push import BlockedPushAddress, _push_session, is_public_address


@pytest.fixture
def loopback_listener():
    """A real listener, so the TCP connect SUCCEEDS and the guard is what stops it.

    Pointing at a closed port would prove nothing: the OS refuses first and the
    check never runs. An earlier version of this test did exactly that and passed
    for the wrong reason.
    """
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)

    def _accept():
        while True:
            try:
                conn, _ = srv.accept()
                conn.close()
            except OSError:
                return

    threading.Thread(target=_accept, daemon=True).start()
    yield srv.getsockname()[1]
    srv.close()


def _blocked_in_chain(exc: BaseException) -> bool:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, BlockedPushAddress):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def test_a_send_to_loopback_is_refused_at_connect(loopback_listener):
    session = _push_session()
    with pytest.raises(requests.exceptions.RequestException) as caught:
        session.post(f"https://127.0.0.1:{loopback_listener}/push", timeout=5)

    assert _blocked_in_chain(caught.value), (
        "the connection was not stopped by the address guard; it failed for some "
        f"other reason: {caught.value!r}"
    )


def test_the_guard_covers_http_too(loopback_listener):
    """A push service answering with a redirect to http:// would otherwise be
    served by the default adapter, skipping the guard on exactly the request
    worth guarding."""
    session = _push_session()
    with pytest.raises(requests.exceptions.RequestException) as caught:
        session.post(f"http://127.0.0.1:{loopback_listener}/push", timeout=5)

    assert _blocked_in_chain(caught.value), repr(caught.value)


def test_each_thread_gets_its_own_session():
    """requests.Session is not safe to share across threads, and _send_one runs in
    anyio's worker pool with up to _MAX_CONCURRENT_SENDS at once.

    The sessions themselves are collected, not their id()s. An earlier version of
    this test stored ids and was flaky: a thread that finishes early has its
    thread-local storage released, its session becomes collectable, and CPython is
    free to hand the same id to the next one — so `len(set(ids)) == 3` could fail
    with three genuinely distinct sessions, or pass with fewer. Holding the objects
    keeps them alive for the length of the comparison, which is what makes identity
    mean anything here.
    """
    sessions: list = []
    barrier = threading.Barrier(3)

    def _grab():
        barrier.wait()
        sessions.append(_push_session())

    threads = [threading.Thread(target=_grab) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(sessions) == 3
    assert len({id(s) for s in sessions}) == 3, "a session was shared between threads"
    # And within one thread it is reused, or the pooling is pointless.
    assert _push_session() is _push_session()


def test_the_session_guards_both_schemes():
    session = _push_session()
    for scheme in ("https://", "http://"):
        adapter = session.get_adapter(scheme + "push.example.test")
        assert type(adapter).__name__ == "_PublicOnlyAdapter", (scheme, adapter)


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("8.8.8.8", True),
        ("127.0.0.1", False),
        ("10.0.0.1", False),
        ("192.168.1.1", False),
        ("169.254.169.254", False),  # cloud metadata
        ("::1", False),
        ("::ffff:127.0.0.1", False),  # IPv4-mapped loopback
        ("0.0.0.0", False),
        ("224.0.0.1", False),  # multicast: is_global is True for these
        ("239.255.255.250", False),  # SSDP
        ("ff02::1", False),
        ("100.64.0.1", False),  # RFC 6598 shared address space (carrier-grade NAT)
        ("198.18.0.1", False),  # benchmarking range
        ("2001:db8::1", False),  # documentation range
        ("not-an-address", False),
    ],
)
def test_is_public_address(address, public):
    """One predicate, shared by the subscribe-time check and the connect-time one,
    so the two cannot drift."""
    assert is_public_address(address) is public


def test_environment_proxies_cannot_route_around_the_guard(monkeypatch):
    """A proxy would bypass the address check entirely.

    HTTPAdapter guards DIRECT connections through the connection classes above,
    but proxy_manager_for builds its own pools that do not use them. With
    HTTPS_PROXY set, delivery would go through the proxy and the peer we validate
    would be the proxy — the endpoint's real address never checked, which is the
    whole guarantee.

    Asserted on what requests would actually do with the environment, not merely
    on the flag: merge_environment_settings is the function that reads it.
    """
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:3128")

    session = _push_session()
    assert session.trust_env is False

    settings = session.merge_environment_settings("https://push.example.test/f/abc", {}, None, None, None)
    assert settings["proxies"] == {}, f"delivery would go through a proxy: {settings['proxies']}"


def test_a_redirect_to_a_blocked_address_is_stopped(monkeypatch):
    """The guard has to hold on the address a REDIRECT lands on, not just the first.

    The other scheme test proves the http adapter carries the guard. It does not
    prove requests re-selects an adapter, and re-runs the check, for the hop after
    a redirect — a push service answering 302 to an internal address is the case
    that matters, and "the adapter is mounted" is not the same claim.

    Both hops are loopback, because a hermetic test cannot reach a public host, and
    they are told apart by CALL ORDER rather than by address: the first connection
    is allowed, the second refused. Distinguishing them by address was the obvious
    way and is not portable — 127.0.0.2 is aliased on Linux but not on macOS, so
    that version passed in CI and failed on the machine it was written on.

    Both ends really listen. Pointing the redirect at a closed port would let the
    OS refuse first, and the test would pass without the guard ever running.
    """
    target = socket.socket()
    target.bind(("127.0.0.1", 0))
    target.listen(4)
    target_port = target.getsockname()[1]

    def _accept_target():
        while True:
            try:
                conn, _ = target.accept()
                conn.close()
            except OSError:
                return

    threading.Thread(target=_accept_target, daemon=True).start()

    class _Redirector(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target_port}/next")
            self.end_headers()

        def log_message(self, *args):
            pass

    redirector = http.server.HTTPServer(("127.0.0.1", 0), _Redirector)
    threading.Thread(target=redirector.serve_forever, daemon=True).start()

    seen: list[str] = []

    def _first_hop_only(addr: str) -> bool:
        seen.append(addr)
        return len(seen) == 1

    monkeypatch.setattr(push_mod, "is_public_address", _first_hop_only)

    try:
        session = _push_session()
        with pytest.raises(requests.exceptions.RequestException) as caught:
            session.post(f"http://127.0.0.1:{redirector.server_port}/start", timeout=5)
    finally:
        # shutdown() stops serve_forever; server_close() releases the socket.
        # Without the second the listener leaks for the rest of the session.
        redirector.shutdown()
        redirector.server_close()
        target.close()

    assert len(seen) == 2, (
        "the address guard did not run a second time, so the redirect hop was "
        f"never checked (checked: {seen})"
    )
    assert _blocked_in_chain(caught.value), (
        f"the redirect hop was not stopped by the address guard: {caught.value!r}"
    )
