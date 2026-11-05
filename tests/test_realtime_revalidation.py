"""Account revalidation behind a live websocket.

WHAT IS AND IS NOT COVERED HERE

The websocket's periodic revalidation was moved out of the `ping` branch so that
EVERY frame revalidates. That was a real bypass: the 65s receive timeout resets
on any frame, so a client sending `typing_start` every 30s and never pinging
kept its socket open indefinitely and was never re-checked — a deactivated
account, a revoked mobile grant and an expired access token all survived for as
long as it kept typing.

That hoist is NOT covered end to end here. Driving the real socket needs
Starlette's synchronous TestClient, which runs its own event loop and cannot
share the async session engine these tests use — it fails in teardown with
asyncpg futures attached to a different loop, not on the assertion. Making it
work needs either a live-server fixture or reworking the engine's loop
ownership, both larger than this change.

What IS covered is the gate the hoist backstops: a deactivated user cannot send,
checked per message against the live row. Revalidation bounds how long a revoked
session keeps RECEIVING; the check below is what stops it SENDING, and it is the
security-relevant half.

The loop itself is driven since batch 73 through a stub socket, which needs no
second event loop (tests/test_must_change_password.py has the account checks).
The heartbeat tests at the bottom use it: a slow handler must not drop a client
whose frames are waiting in the buffer, and a silent client must still be dropped.
"""

import asyncio
import json
import time
import uuid

from httpx import ASGITransport, AsyncClient

from app.db.models import User
from app.db.session import SessionLocal
from app.main import app
from app.realtime import hub
from app.realtime.redis_bus import publish_to_users
from app.services.messaging import SendError, send_message
from tests.conftest import CSRF, login, make_org, make_user
from tests.test_must_change_password import _SilentSocketStub, _SocketStub, _token_for


async def _deactivate(user_id):
    async with SessionLocal() as db:
        row = await db.get(User, user_id)
        row.is_active = False
        await db.commit()


def test_revalidation_is_a_timer_not_a_frame_type():
    """Weak alone, but it pins the shape the fix depends on.

    REVALIDATE_SECONDS existing at module scope is what lets the check sit
    outside the ping branch and be gated by the wall clock rather than by which
    frame happened to arrive. It must also be shorter than the heartbeat
    timeout, or a socket could close before ever revalidating.
    """
    assert isinstance(hub.REVALIDATE_SECONDS, int)
    assert 0 < hub.REVALIDATE_SECONDS <= hub.HEARTBEAT_TIMEOUT


async def test_deactivated_sender_cannot_post_even_with_a_live_session(client):
    org = await make_org("WS Send Co")
    a = await make_user("a@wssend.com", org_id=org.id)
    b = await make_user("b@wssend.com", org_id=org.id)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c.headers.update(CSRF)
        await login(c, "a@wssend.com")
        conv = (await c.post("/api/conversations/direct", json={"participant_id": str(b.id)})).json()["_id"]

    await _deactivate(a.id)

    # The websocket send path re-loads the sender row per frame, so this is
    # exactly the object hub.py would hand to send_message after deactivation.
    async with SessionLocal() as db:
        sender = await db.get(User, a.id)
        try:
            await send_message(db, conversation_id=uuid.UUID(conv), sender=sender, content="should not land")
            raise AssertionError("a deactivated user was able to send")
        except SendError as exc:
            assert exc.status_code == 403
            assert "no longer active" in exc.detail


async def test_an_active_sender_is_unaffected(client):
    """Negative control: the new gate must not block ordinary sending."""
    org = await make_org("WS Active Co")
    await make_user("a@wsact.com", org_id=org.id)
    b = await make_user("b@wsact.com", org_id=org.id)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c.headers.update(CSRF)
        await login(c, "a@wsact.com")
        conv = (await c.post("/api/conversations/direct", json={"participant_id": str(b.id)})).json()["_id"]
        sent = await c.post(f"/api/conversations/{conv}/messages", json={"content": "fine"})
        assert sent.status_code == 200, sent.text


class _WedgedSocket:
    """A client that has stopped reading: its send never completes until released.

    Not artificial. uvicorn's ASGI websocket send awaits the transport drain, so a
    real socket behaves exactly like this once a non-reading peer's buffers fill —
    measured at ~0.8MB on loopback, and it stays pending until TCP gives up.
    """

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.entered = asyncio.Event()
        self.closed: tuple[int, str] | None = None

    async def send_text(self, data: str) -> None:
        self.entered.set()
        await self.release.wait()

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)


class _HealthySocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed: tuple[int, str] | None = None

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)


async def test_one_stalled_socket_does_not_block_delivery_to_other_users():
    """A worker's fan-out must not serialise behind its slowest socket.

    _reader is ONE task per worker. It used to `await ws.send_text(...)` inline, so a
    client that stopped reading stalled delivery for EVERY user that worker served —
    no messages, no typing, no presence, no call ring — until TCP gave up, which is
    minutes rather than seconds. Measured against the old code: the bystander below
    received 0 of its events during the stall and all of them only once the wedged
    socket was released, at 2.0s latency.

    Uses the real LocalRegistry and the real Redis pub/sub; only the sockets are
    stubs, because the behaviour under test is what the registry does with a send
    that does not return.
    """
    registry = hub.LocalRegistry()
    await registry.start()
    stalled, bystander = _WedgedSocket(), _HealthySocket()
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    try:
        await registry.add(user_a, "conn-a", stalled)
        await registry.add(user_b, "conn-b", bystander)
        await asyncio.sleep(0.3)  # let SUBSCRIBE take effect

        await publish_to_users([user_a], {"type": "wedge"})
        await asyncio.wait_for(stalled.entered.wait(), timeout=5)

        for i in range(5):
            await publish_to_users([user_b], {"type": "for_b", "n": i})
        for _ in range(40):  # up to 4s, but returns as soon as they land
            if len(bystander.sent) == 5:
                break
            await asyncio.sleep(0.1)

        assert len(bystander.sent) == 5, (
            f"only {len(bystander.sent)}/5 events reached a second user while one "
            "socket was stalled — the worker's fan-out is head-of-line blocked"
        )
        assert bystander.closed is None
    finally:
        stalled.release.set()
        await registry.stop()


async def test_a_socket_that_never_drains_is_dropped_rather_than_waited_for():
    """Overflow closes the wedged socket, and only it.

    Dropping frames silently would leave a client that looks connected and receives
    nothing. 1011 rather than 4001 because both clients treat 4001 as "refresh the
    session" and any other code as a plain disconnect to retry with backoff — so the
    close is what makes this self-heal via reconnect-and-refetch.
    """
    registry = hub.LocalRegistry()
    await registry.start()
    stalled, bystander = _WedgedSocket(), _HealthySocket()
    user_a, user_b = uuid.uuid4(), uuid.uuid4()
    try:
        await registry.add(user_a, "conn-a", stalled)
        await registry.add(user_b, "conn-b", bystander)
        await asyncio.sleep(0.3)

        overflow_by = 10
        total = hub.OUTBOX_MAX_FRAMES + overflow_by
        for i in range(total):
            await publish_to_users([user_a], {"n": i})
            await publish_to_users([user_b], {"n": i})
        for _ in range(60):  # up to 6s
            if stalled.closed is not None and len(bystander.sent) == total:
                break
            await asyncio.sleep(0.1)

        assert stalled.closed is not None, "a socket that never drains was never dropped"
        assert stalled.closed[0] == 1011, stalled.closed
        assert bystander.closed is None, "the healthy socket must not be collateral"
        assert len(bystander.sent) == total, f"{len(bystander.sent)}/{total} reached the healthy socket"
    finally:
        stalled.release.set()
        await registry.stop()


async def test_a_burst_larger_than_the_outbox_does_not_close_a_healthy_socket_over_real_redis():
    """Queue depth must measure socket slowness, not reader burstiness.

    INTEGRATION check, and deliberately not the guard: a pipeline batches the
    PUBLISH commands but does not synchronise the SUBSCRIBER socket with the last of
    them, so the burst may still arrive incrementally and let _reader suspend between
    frames. It therefore cannot be relied on to fail when the yield is removed — the
    deterministic version of this is the test below. This one is kept because it
    exercises the real pub/sub path end to end.
    """
    registry = hub.LocalRegistry()
    await registry.start()
    healthy = _HealthySocket()
    user = uuid.uuid4()
    try:
        await registry.add(user, "conn", healthy)
        await asyncio.sleep(0.3)

        # ONE pipeline, one round trip: Redis then hands the reader a buffered run of
        # messages, which is the condition that makes get_message stop suspending.
        # Publishing them one await at a time would yield between each and never
        # reproduce it — the mistake this test was written wrong with the first time.
        burst = hub.OUTBOX_MAX_FRAMES * 3
        await publish_to_users([user] * burst, {"burst": True})
        for _ in range(80):  # up to 8s
            if len(healthy.sent) == burst:
                break
            await asyncio.sleep(0.1)

        assert healthy.closed is None, f"a healthy socket was dropped: {healthy.closed}"
        assert len(healthy.sent) == burst, f"{len(healthy.sent)}/{burst} delivered"
    finally:
        await registry.stop()


class _PreloadedPubSub:
    """A pub/sub whose get_message NEVER suspends while frames remain.

    This is the condition the real broker only reaches sometimes: a run of messages
    already buffered, returned back to back. Awaiting a coroutine that returns
    immediately does not yield in asyncio, so a reader without an explicit yield
    drains the whole run in one tight loop.
    """

    def __init__(self, messages: list[dict]) -> None:
        self._messages = list(messages)

    # noqa signature, not style: it has to match the call _reader makes on the real
    # redis pubsub, so the keyword names and defaults are fixed by that contract.
    async def get_message(
        self,
        ignore_subscribe_messages: bool = True,
        timeout: float = 1.0,  # noqa: ASYNC109
    ):
        if self._messages:
            return self._messages.pop(0)
        await asyncio.sleep(0.02)
        return None

    async def subscribe(self, *_a) -> None: ...
    async def unsubscribe(self, *_a) -> None: ...
    async def aclose(self) -> None: ...


async def test_reader_yields_so_a_buffered_run_cannot_close_a_healthy_socket():
    """The deterministic guard for the reader's yield.

    No Redis and no timing: the pub/sub source hands _reader a buffered run of
    OUTBOX_MAX_FRAMES * 3 frames with no suspension between them. Without the yield
    the reader enqueues the whole run before any writer is scheduled, the outbox hits
    its ceiling and a perfectly healthy socket is closed with 1011.
    """
    user = uuid.uuid4()
    burst = hub.OUTBOX_MAX_FRAMES * 3
    frames = [{"channel": f"user:{user}", "data": json.dumps({"n": i})} for i in range(burst)]

    registry = hub.LocalRegistry()
    registry._pubsub = _PreloadedPubSub(frames)
    healthy = _HealthySocket()
    try:
        await registry.add(user, "conn", healthy)
        registry._pubsub_task = asyncio.create_task(registry._reader())
        for _ in range(200):  # up to 4s, returns as soon as the run is delivered
            if len(healthy.sent) == burst:
                break
            await asyncio.sleep(0.02)

        assert healthy.closed is None, f"a healthy socket was dropped: {healthy.closed}"
        assert len(healthy.sent) == burst, f"{len(healthy.sent)}/{burst} delivered"
    finally:
        await registry.stop()


async def test_send_to_reaches_the_socket_and_reports_an_unknown_one():
    """The endpoint's own frames now travel this way, including `pong`.

    pong is the heartbeat reply, so a send_to that silently failed would let clients
    time out at HEARTBEAT_TIMEOUT while the socket looked healthy from the server side.
    The False return matters for the same reason: a caller must never be told a frame
    was queued for a socket that has gone.
    """
    registry = hub.LocalRegistry()
    healthy = _HealthySocket()
    user, gone = uuid.uuid4(), uuid.uuid4()
    try:
        await registry.add(user, "conn", healthy)

        assert await registry.send_to(user, "conn", '{"type":"pong"}') is True
        for _ in range(50):
            if healthy.sent:
                break
            await asyncio.sleep(0.02)
        assert healthy.sent == ['{"type":"pong"}']

        assert await registry.send_to(user, "no-such-conn", "x") is False
        assert await registry.send_to(gone, "conn", "x") is False
    finally:
        await registry.stop()


# ---------------------------------------------------------------------------
# The heartbeat (batch 73 review)
# ---------------------------------------------------------------------------


class _LingeringSocketStub(_SocketStub):
    """Stays open for a moment after its last frame, so what the endpoint queued for
    it — the pong — is written by the socket's writer task before the disconnect
    tears that task down."""

    async def receive_text(self) -> str:
        """Wait briefly once the frames run out, then disconnect as the base stub does."""
        if not self._frames:
            await asyncio.sleep(0.05)  # well inside the heartbeat the tests set
        return await super().receive_text()


async def test_a_slow_handler_does_not_drop_a_client_whose_ping_is_buffered(client, monkeypatch):
    """The heartbeat used to be decided at the top of the loop, before any receive,
    from the time since the last frame — which includes however long that frame took
    to HANDLE. A handler that took HEARTBEAT_TIMEOUT dropped the client while its
    next ping sat in the buffer, unread. It is decided now only when a receive has
    actually come back empty."""
    user = await make_user("slow-handler@x.com")
    monkeypatch.setattr(hub, "HEARTBEAT_TIMEOUT", 0.2)

    async def _slow(_user, _data):
        """Take twice the heartbeat to handle a frame."""
        await asyncio.sleep(0.4)  # twice the heartbeat

    monkeypatch.setattr(hub, "_handle_inbound", _slow)
    frames = [json.dumps({"type": "typing_start", "conversation_id": str(uuid.uuid4())}), '{"type": "ping"}']
    ws = _LingeringSocketStub(_token_for(user), frames=frames)
    await asyncio.wait_for(hub.websocket_endpoint(ws), timeout=5)

    assert [f["type"] for f in ws.sent] == ["connected", "pong"], "the buffered ping was never answered"
    assert ws.closed is None


async def test_a_silent_client_is_still_dropped_at_the_heartbeat(client, monkeypatch):
    """The other half: deciding the heartbeat after the receive must not stop it
    deciding at all."""
    user = await make_user("silent-heartbeat@x.com")
    monkeypatch.setattr(hub, "HEARTBEAT_TIMEOUT", 0.2)
    ws = _SilentSocketStub(_token_for(user))
    started = time.monotonic()
    await asyncio.wait_for(hub.websocket_endpoint(ws), timeout=5)
    assert time.monotonic() - started >= 0.2
    assert [f["type"] for f in ws.sent] == ["connected"]
    # Dropped, not closed with a code: a client that stopped answering is not told why.
    assert ws.closed is None
    assert str(user.id) not in hub.registry.connections
