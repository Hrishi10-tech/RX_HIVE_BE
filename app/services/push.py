"""Web Push delivery. Best-effort: a failed/expired subscription is pruned,
never blocks the request that triggered it. No-op when VAPID is unconfigured.
"""

import asyncio
import ipaddress
import json
import logging
import socket
import threading
import uuid
from typing import NamedTuple
from urllib.parse import urlsplit

import anyio
import requests
from requests.adapters import HTTPAdapter
from sqlalchemy import delete, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.poolmanager import PoolManager, pool_classes_by_scheme

from app.core.config import get_settings
from app.db.models import PushSubscription
from app.db.session import SessionLocal


def is_public_address(value: str) -> bool:
    """One definition of "safe to POST to", used by both checks below.

    Written twice would be the usual outcome — the subscribe-time check and the
    connect-time check are in different layers and ask the question at different
    moments — and the two copies would drift.
    """
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    # is_global AND not multicast. Both halves are load-bearing.
    #
    # The old test was a hand-written list — private, loopback, link-local,
    # reserved, multicast, unspecified — and it let 100.64.0.0/10 through. RFC 6598
    # shared address space is none of those by Python's predicates, and is
    # emphatically not somewhere to POST: cloud providers use it for carrier-grade
    # NAT and for internal services. is_global covers it, and tracks the IANA
    # special-purpose registry as Python updates it, so the next range like that
    # one is handled without anybody noticing it exists.
    #
    # But is_global is TRUE for multicast — 224.0.0.1, 239.255.255.250 (SSDP),
    # ff02::1 are all "global" by that definition — which the hand-written list got
    # right. Swapping one for the other would have closed the shared-address hole
    # and opened a multicast one. Verified against both.
    return ip.is_global and not ip.is_multicast


def validate_push_endpoint(endpoint: str) -> bool:
    """Reject SSRF-prone push endpoints: must be https to a public host, never
    an internal/loopback/link-local address. The server POSTs to this URL.

    This is a PRE-check and cannot be the only one. It resolves the hostname now;
    pywebpush resolves it again when the send happens, which may be days later and
    is a different lookup. A caller who controls DNS for their own hostname can
    answer with a public address here and a private one there — classic rebinding,
    and no amount of care at this end closes it. _PublicOnlyHTTPSConnection does,
    by checking the address actually connected to.

    Kept anyway, because refusing a bad endpoint at subscribe time is a far better
    experience than accepting it and silently failing every send, and because it
    rejects non-https and unresolvable hosts outright.
    """
    try:
        parts = urlsplit(endpoint)
    except ValueError:
        return False
    if parts.scheme != "https" or not parts.hostname:
        return False
    host = parts.hostname
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    return all(is_public_address(info[4][0]) for info in infos)


class BlockedPushAddress(OSError):
    """Raised when a push endpoint resolves to an address we must not POST to.

    OSError so urllib3 and requests treat it as a transport failure and wrap it
    the way they wrap a refused connection, rather than it escaping as something
    the retry machinery has no handling for.
    """


def _refuse_private_peer(sock: socket.socket) -> socket.socket:
    """Shared by both connection classes below. See _PublicOnlyHTTPSConnection."""
    try:
        peer = sock.getpeername()[0]
    except OSError:
        sock.close()
        raise
    if not is_public_address(peer):
        sock.close()
        raise BlockedPushAddress(f"refused push delivery to non-public address {peer}")
    return sock


class _PublicOnlyHTTPSConnection(HTTPSConnection):
    """Checks the address actually connected to, which is the only check that holds.

    validate_push_endpoint resolves the hostname when the browser subscribes.
    pywebpush resolves it again on every send, days later, in a separate lookup —
    so a caller who controls DNS for their own hostname can answer publicly at
    subscribe time and privately at send time. The endpoint is stored and reused,
    so that is not a one-shot trick: it is a POST into the private network on every
    message, indefinitely.

    There is no way to close that from the subscribe side. What closes it is asking
    the socket where it ACTUALLY went, after connect and before anything is sent.
    """

    def _new_conn(self) -> socket.socket:
        return _refuse_private_peer(super()._new_conn())


class _PublicOnlyHTTPConnection(HTTPConnection):
    """The http half, which is not decoration.

    Mounting the adapter for http:// was not enough on its own — the pool manager
    decides the connection class per scheme, so http kept urllib3's default and
    the guard was skipped for exactly the request worth guarding: a push service
    answering with a redirect to an http:// address. Caught by the test that
    covers the redirect case, having been missed here first.
    """

    def _new_conn(self) -> socket.socket:
        return _refuse_private_peer(super()._new_conn())


class _PublicOnlyHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _PublicOnlyHTTPSConnection


class _PublicOnlyHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _PublicOnlyHTTPConnection


class _PublicOnlyPoolManager(PoolManager):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool_classes_by_scheme = {
            **pool_classes_by_scheme,
            "http": _PublicOnlyHTTPConnectionPool,
            "https": _PublicOnlyHTTPSConnectionPool,
        }


class _PublicOnlyAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        self.poolmanager = _PublicOnlyPoolManager(
            num_pools=connections, maxsize=maxsize, block=block, **kwargs
        )


# One session per worker thread rather than one shared session.
#
# requests.Session is not safe to share across threads, and _send_one runs in
# anyio's worker pool with up to _MAX_CONCURRENT_SENDS of them at once. Per-thread
# keeps connection reuse — the point of a Session — without the sharing.
_sessions = threading.local()


def _push_session() -> "requests.Session":
    session = getattr(_sessions, "session", None)
    if session is None:
        session = requests.Session()
        # No environment proxies. HTTPAdapter guards direct connections through
        # the classes above, but proxy_manager_for builds its own pools that do
        # not use them — so an HTTP_PROXY or HTTPS_PROXY in the environment would
        # route delivery through the proxy and past the address check entirely.
        # Push goes straight out or not at all.
        session.trust_env = False
        # BOTH schemes, deliberately. A push service that answers with a redirect
        # to http:// would otherwise be served by the default adapter, and the
        # guard would be skipped for exactly the request worth guarding.
        session.mount("https://", _PublicOnlyAdapter())
        session.mount("http://", _PublicOnlyAdapter())
        _sessions.session = session
    return session


logger = logging.getLogger(__name__)

try:
    from pywebpush import WebPushException, webpush

    _HAS_WEBPUSH = True
except ImportError:  # optional dependency; feature simply disabled if absent
    _HAS_WEBPUSH = False


SEND_TIMEOUT_SECONDS = 5
_MAX_CONCURRENT_SENDS = 10

# asyncio keeps only a weak reference to a running task, so a fire-and-forget
# task with no other referent can be garbage-collected mid-flight. Hold a
# strong reference until it completes.
_background_tasks: set[asyncio.Task] = set()


def dispatch_push_to_users(user_ids, payload: dict) -> None:
    """Fan out Web Push off the caller's path, fire-and-forget.

    Awaited inline this put N remote HTTPS POSTs into the sender's latency and
    pinned the request's DB session (pool size 10) for their whole duration.
    """
    settings = get_settings()
    if not _HAS_WEBPUSH or not settings.vapid_private_key or not user_ids:
        return
    task = asyncio.create_task(_push_in_background(list(user_ids), payload))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


class _Target(NamedTuple):
    """What one send needs, detached from the ORM and from any session.

    The fan-out used to pass PushSubscription instances into worker threads and
    read `.endpoint` and `.keys` there. That only worked because the attributes
    happened to be loaded and unexpired: an ORM attribute access that has to go
    back to the database raises MissingGreenlet from a thread, and after the
    change below there is deliberately no session left to go back to. A plain
    tuple removes the whole class of problem rather than relying on it not
    happening.

    `id` and `user_id` are carried for the prune, not for the send — see _prune.
    """

    id: uuid.UUID
    user_id: uuid.UUID
    endpoint: str
    keys: dict


async def _load_targets(db: AsyncSession, user_ids) -> list[_Target]:
    rows = (
        (await db.execute(select(PushSubscription).where(PushSubscription.user_id.in_(list(user_ids)))))
        .scalars()
        .all()
    )
    return [_Target(id=r.id, user_id=r.user_id, endpoint=r.endpoint, keys=dict(r.keys or {})) for r in rows]


async def _prune(db: AsyncSession, dead: list[_Target]) -> None:
    """Delete the subscriptions the push service rejected — but only the exact rows
    that were loaded, still owned by whom they were loaded for.

    Deleting by endpoint alone was wrong, and reachable. Endpoints are globally
    unique and api/notifications.py RE-BINDS one to whoever presents it
    ("Endpoints are globally unique; re-subscribing re-binds to the caller"), so
    the row can change hands between _load_targets and a delayed 404/410 landing
    here. It is the shared-workstation case again: if a sign-out teardown does not
    complete, the browser keeps its subscription, the next person to sign in on
    that profile gets the SAME endpoint back from pushManager.subscribe(), and the
    route moves the row to them. A stale rejection from the previous user's
    in-flight fan-out would then delete a live subscription belonging to another
    user — in another organisation, in the general case — who silently stops
    receiving push and has no way to know why.

    Matching on (id, user_id) makes a re-bound row simply not match: same row, new
    owner, left alone.
    """
    await db.execute(
        delete(PushSubscription).where(
            tuple_(PushSubscription.id, PushSubscription.user_id).in_([(t.id, t.user_id) for t in dead])
        )
    )
    await db.commit()


async def _fan_out(targets: list[_Target], payload: dict) -> list[_Target]:
    """Send to every target and report the ones the push service rejected.

    Holds no database connection. That is the point — see _push_in_background.
    """
    settings = get_settings()
    data = json.dumps(payload)
    dead: list[_Target] = []
    # Concurrent, but bounded: a large group must not hand the whole worker
    # thread pool to one push fan-out.
    limiter = anyio.CapacityLimiter(_MAX_CONCURRENT_SENDS)
    async with anyio.create_task_group() as tg:
        for target in targets:
            tg.start_soon(_deliver, target, data, settings, limiter, dead)
    return dead


async def _push_in_background(user_ids, payload: dict) -> None:
    """Three phases, and the middle one holds no pooled connection.

    dispatch_push_to_users moved the fan-out off the request precisely because
    awaiting it inline "pinned the request's DB session (pool size 10) for their
    whole duration". Giving the background task its own session moved that
    pinning rather than removing it: the session was opened before the fan-out
    and closed after it, so one pooled connection sat checked out — and idle —
    for the entire run of remote HTTPS POSTs.

    The duration is not small. Sends are bounded to _MAX_CONCURRENT_SENDS (10) and
    each may spend SEND_TIMEOUT_SECONDS (5) against a black-holed endpoint, so the
    wall time scales with the number of subscriptions: 2,000 of them is up to
    ~17 minutes. And nothing caps that number — regular groups stop at
    MAX_GROUP_MEMBERS, but app/api/cross_org.py bounds only the MINIMUM size, so a
    cross-org group has no ceiling at all (services/messaging.py records the same
    asymmetry making a different tail reachable). Every message to such a group
    starts another one, so a handful of them concurrently exhausts a pool of 10
    and the API stops being able to serve anything that needs a session, which is
    every route.

    So the connection is held only for the two operations that need it, and the
    network happens in between with nothing checked out. Nothing may escape —
    push is best-effort, and an exception here has no caller left to surface it to.
    """
    try:
        settings = get_settings()
        if not _HAS_WEBPUSH or not settings.vapid_private_key or not user_ids:
            return
        async with SessionLocal() as db:
            targets = await _load_targets(db, user_ids)
        if not targets:
            return
        dead = await _fan_out(targets, payload)
        if dead:
            async with SessionLocal() as db:
                await _prune(db, dead)
    except Exception:
        logger.exception("background web push failed")


async def push_to_users(db: AsyncSession, user_ids, payload: dict) -> None:
    """Fan out using the CALLER's session, which it therefore pins for the whole
    duration. Prefer dispatch_push_to_users, which does not — see
    _push_in_background for why that matters. Kept for a caller that has no event
    loop of its own to hand the work to, such as a management script.
    """
    settings = get_settings()
    if not _HAS_WEBPUSH or not settings.vapid_private_key or not user_ids:
        return
    targets = await _load_targets(db, user_ids)
    if not targets:
        return
    dead = await _fan_out(targets, payload)
    if dead:
        await _prune(db, dead)


async def _deliver(
    sub: _Target, data: str, settings, limiter: anyio.CapacityLimiter, dead: list[_Target]
) -> None:
    """Deliver to one subscription. Must never raise: an exception out of a
    task-group child cancels its siblings, so one bad endpoint would take down
    the rest of the fan-out."""
    try:
        await anyio.to_thread.run_sync(_send_one, sub, data, settings, limiter=limiter)
    except WebPushException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (404, 410):
            # The whole target, not just its endpoint: _prune needs the identity
            # and the owner it was loaded with.
            dead.append(sub)
        else:
            logger.warning("web push failed: %s", exc)
    except Exception:
        logger.exception("web push error")


def _send_one(sub: _Target, data: str, settings) -> None:
    webpush(
        subscription_info={"endpoint": sub.endpoint, "keys": sub.keys},
        data=data,
        vapid_private_key=settings.vapid_private_key,
        vapid_claims={"sub": settings.vapid_subject},
        # The connect-time address check lives here. Without this the send uses
        # pywebpush's own session and re-resolves the hostname unguarded.
        requests_session=_push_session(),
        # pywebpush uses requests, which has no default timeout: a blackholed
        # push endpoint would hang this worker thread forever.
        timeout=SEND_TIMEOUT_SECONDS,
    )
