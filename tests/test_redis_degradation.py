"""Fail-open only works when the failure is an exception.

Every best-effort Redis call in the product runs inside
realtime.redis_bus.degrade_on_outage, and the module's promise is that a broker
outage costs a presence update or a rate-limit window rather than a request. That
promise had two holes, and both are about the SHAPE of the failure rather than
its presence:

  * A broker that stops ANSWERING raises nothing. With no socket deadline the
    command waits forever, so the guard never runs and the request never returns.
  * A bug inside a guarded block raises something that is not an outage at all,
    and was reported as one.
"""

import asyncio
import contextlib
import socket
import time
from collections.abc import AsyncIterator, Iterator

import pytest
import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.realtime import redis_bus
from app.realtime.redis_bus import (
    HEALTH_CHECK_INTERVAL_SECONDS,
    SOCKET_TIMEOUT_SECONDS,
    degrade_on_outage,
    get_redis,
)


@contextlib.contextmanager
def silent_broker() -> Iterator[int]:
    """A port that completes the TCP handshake and then never answers.

    A listening socket that never calls accept() is enough: the kernel finishes
    the handshake into the accept backlog, so connect() and the outbound write
    both succeed while the reply never comes. That is the failure under test — a
    partition, a dropped NAT mapping, a wedged broker — as opposed to a refusal,
    which is the only case the old configuration survived.

    A raw socket rather than asyncio.start_server deliberately: a Server with a
    live connection makes wait_closed() block until that connection closes, and
    nothing here ever closes it, so the teardown hangs instead of the assertion.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        yield listener.getsockname()[1]
    finally:
        listener.close()


@contextlib.asynccontextmanager
async def app_client_pointed_at(port: int, monkeypatch) -> AsyncIterator[aioredis.Redis]:
    """The application's OWN client, built by get_redis(), aimed at a dead port.

    Worth the setup: a test that constructs its own client with the right
    arguments proves redis-py honours them, not that this application passes
    them. Going through get_redis() is what ties the assertion to shipped code.

    _redis is a module global the whole session shares, so it is closed and
    cleared on the way in and on the way out, along with the lru_cached settings —
    otherwise every later test inherits a client aimed at a socket that no longer
    exists.

    close_redis() rather than monkeypatch.setattr on _redis: saving the reference
    and restoring it later only looks like isolation. The teardown then has a
    client it must discard anyway (its URL is a port that has stopped existing),
    so the session's original client would be orphaned with its connection pool
    still open. close_redis() is the module's own accessor for this — it closes
    and clears in one step — so both the incoming and outgoing client are shut
    down properly and the next get_redis() rebuilds lazily against the real URL.
    """
    await redis_bus.close_redis()
    monkeypatch.setenv("RXHIVE_REDIS_URL", f"redis://127.0.0.1:{port}/0")
    get_settings.cache_clear()
    client = redis_bus.get_redis()
    try:
        assert client.connection_pool.connection_kwargs["port"] == port
        yield client
    finally:
        await redis_bus.close_redis()
        monkeypatch.undo()
        get_settings.cache_clear()


async def test_the_shared_client_carries_a_read_and_connect_deadline():
    """redis-py defaults both to None, which is the whole finding.

    Built fresh rather than inspected wherever it happens to be. get_redis()
    caches into a module global, so reading it as-found asserts against whatever
    client an earlier test left there — including, if the ordering changed, one
    aimed at the dead port used below. The claim here is about what get_redis()
    CONSTRUCTS, so the cache is cleared on both sides and the client this test
    creates is not left behind for the next one.
    """
    await redis_bus.close_redis()
    try:
        kwargs = get_redis().connection_pool.connection_kwargs
        assert kwargs.get("socket_timeout") == SOCKET_TIMEOUT_SECONDS
        assert kwargs.get("socket_connect_timeout") == SOCKET_TIMEOUT_SECONDS
        assert kwargs.get("health_check_interval") == HEALTH_CHECK_INTERVAL_SECONDS
    finally:
        await redis_bus.close_redis()


async def test_the_deadline_stays_above_the_pubsub_polling_interval():
    """A deadline below hub._reader's own poll would break realtime outright.

    The reader calls get_message(timeout=1.0) in a loop. Were the socket deadline
    shorter than that, an IDLE subscription would trip it on every poll and the
    reader's own except branch would turn that into "pubsub reader error;
    reconnecting in 1s" forever — no events delivered, on a perfectly healthy
    broker. Pinned here because it is the one way this fix could be worse than
    the bug it fixes.
    """
    assert SOCKET_TIMEOUT_SECONDS > 1.0


async def test_a_broker_that_stops_answering_raises_instead_of_hanging(monkeypatch):
    """The finding. With no deadline this command never returns at all.

    Measured against this same silent socket before the fix, `incr` was still
    waiting when the harness capped it. A request in that state holds its DB
    session from a pool of ten for the whole time, so a silent broker took out
    endpoints that never touch Redis.

    wait_for is set well above the deadline so it cannot be what ends the call:
    if the socket deadline fails to fire, this raises asyncio.TimeoutError, which
    is not a RedisError, and pytest.raises fails — which is the regression.
    """
    with silent_broker() as port:
        async with app_client_pointed_at(port, monkeypatch) as client:
            started = time.monotonic()
            with pytest.raises(RedisError):
                await asyncio.wait_for(client.incr("ratelimit:probe"), timeout=SOCKET_TIMEOUT_SECONDS * 4)
            elapsed = time.monotonic() - started
            assert elapsed < SOCKET_TIMEOUT_SECONDS * 2, f"deadline was late: {elapsed:.2f}s"


async def test_that_timeout_is_something_degrade_on_outage_actually_catches(monkeypatch):
    """The two halves have to meet.

    A deadline that raised past the guard would turn a hang into a 500 on the
    sender's path — the exact failure the guard exists to prevent, since this work
    runs after the commit that already stored the message.
    """
    with silent_broker() as port:
        async with app_client_pointed_at(port, monkeypatch) as client:
            async with degrade_on_outage("silent broker"):
                await client.incr("ratelimit:probe")
            # Reaching here at all is the assertion: the guard absorbed it and the
            # caller carries on, which is what "best effort" is supposed to mean.


async def test_a_bug_in_a_guarded_block_is_not_reported_as_an_outage():
    """It was, and that is worse than it sounds.

    An outage recovers; a bug does not. Swallowing a TypeError here meant the
    guarded feature — presence, a rate-limit window, an event fan-out — silently
    did nothing on EVERY request from then on, while the only log line blamed the
    broker. The guard has to be narrow enough that a programming error still
    escapes to something that reports it as one.
    """
    with pytest.raises(AttributeError):
        async with degrade_on_outage("a typo, not an outage"):
            None.publish("user:1", "{}")  # type: ignore[attr-defined]

    with pytest.raises(TypeError):
        async with degrade_on_outage("a bad call, not an outage"):
            raise TypeError("wrong argument type")


async def test_a_real_outage_is_still_swallowed_in_full():
    """The narrowing must not cost the behaviour it was narrowed around."""
    for failure in (
        RedisError("generic redis failure"),
        aioredis.ConnectionError("connection lost"),
        aioredis.TimeoutError("timeout reading"),
        OSError("socket layer beneath redis-py"),
        ConnectionResetError("peer went away"),
    ):
        async with degrade_on_outage(f"outage: {type(failure).__name__}"):
            raise failure


async def test_publish_still_serializes_outside_the_guard():
    """An unserializable payload is a caller bug, and the module says so — it
    json.dumps BEFORE entering the guard on purpose. Kept honest here because
    narrowing the guard is exactly the moment someone might tidy that line
    inwards, where it would be swallowed again."""
    with pytest.raises(TypeError):
        # A non-string KEY: `default=str` rescues unserializable values, so a set
        # would quietly stringify and prove nothing.
        await redis_bus.publish_to_users(["u1"], {(1, 2): "tuple key"})
