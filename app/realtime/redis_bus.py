"""Shared async Redis client + pub/sub event bus.

Fan-out model: every realtime event is published to the per-user channel
`user:{user_id}` of each intended recipient. A worker runs one pub/sub reader
and subscribes to the channels of the users connected to *it*. Delivery is
therefore scoped at publish time — a socket can never receive an event for a
user or org it doesn't belong to, on any worker.
"""

import contextlib
import json
import logging
from collections.abc import AsyncIterator

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Read/connect deadline for every Redis command. redis-py defaults BOTH to None,
# which means no deadline at all — see get_redis().
#
# Above the pub/sub reader's own 1.0s get_message poll, deliberately: below it,
# an idle channel would trip this on every poll and hub._reader would spend its
# life logging "pubsub reader error; reconnecting in 1s" and delivering nothing.
# Verified against a real broker at both 2s and 5s — an idle subscription polls
# clean and publishes still arrive.
SOCKET_TIMEOUT_SECONDS = 2.0

# Ping a pooled connection that has been idle this long before reusing it. A
# broker restart leaves established connections in the pool that are dead but
# look fine; without this the next command on one fails, and because that failure
# lands inside degrade_on_outage it is swallowed rather than retried — one
# silently lost presence update or rate-limit hit per stale connection.
HEALTH_CHECK_INTERVAL_SECONDS = 30

_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """The shared client. Every Redis call in the app goes through here.

    THE TIMEOUTS ARE THE POINT. redis-py defaults socket_timeout and
    socket_connect_timeout to None, so a command against a broker that has
    stopped answering — a partition, a dropped NAT mapping, a wedged broker,
    anything that is not a clean refusal — waits forever. Measured against a
    server that completes the TCP handshake and then never replies, `incr` was
    still hanging when the harness gave up.

    That is what made the whole fail-open design in degrade_on_outage
    conditional: it can only degrade on an EXCEPTION, and silence raises
    nothing. So the outage story held for a broker that closes connections
    (restart, refusal) and inverted for one that just stops talking — every
    request touching Redis hung instead of continuing, holding its DB session
    from a pool of 10 the whole time, until the pool was gone and endpoints that
    never touch Redis stopped answering too. Login, refresh, presence, message
    fan-out and the call deadlines all sit on this client.

    With a deadline, silence becomes redis.exceptions.TimeoutError — a RedisError,
    which degrade_on_outage catches — and best-effort work is skipped as designed.
    """
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            get_settings().redis_url,
            decode_responses=True,
            socket_timeout=SOCKET_TIMEOUT_SECONDS,
            socket_connect_timeout=SOCKET_TIMEOUT_SECONDS,
            health_check_interval=HEALTH_CHECK_INTERVAL_SECONDS,
        )
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def user_channel(user_id) -> str:
    return f"user:{user_id}"


@contextlib.asynccontextmanager
async def degrade_on_outage(operation: str) -> AsyncIterator[None]:
    """Run best-effort Redis work, logging and continuing if Redis is down.

    Realtime fan-out and presence sit AFTER the database commit that is the real
    result of a request. Letting a Redis outage escape from there turned an
    already-persisted message into a 500: the sender was told the send failed for
    a message the server had stored, and clients resend on failure. Only wrap
    post-commit, best-effort work — a failed DB write must still surface as 500.

    Catches REDIS failures, not every failure. This was a bare `except Exception`,
    which is the same shape but a different promise: any bug inside a guarded
    block — a typo'd attribute, a bad argument, a None where a key was expected —
    was caught, logged as "Redis unavailable", and skipped. Not once, but on every
    request, forever, because a bug does not recover the way an outage does. The
    feature would simply never work again, and the one log line pointed at the
    broker. RedisError is the base of every redis-py exception (TimeoutError and
    ConnectionError included) and OSError covers the socket layer beneath it, so
    a real outage is still caught in full.
    """
    try:
        yield
    except (RedisError, OSError):
        logger.warning("Redis unavailable; skipped %s", operation, exc_info=True)


async def publish_to_users(user_ids, event: dict) -> None:
    """Publish one event to every recipient's channel. Best-effort: a broker
    outage costs the live event (clients refetch on reconnect), not the request."""
    # Serialized outside the guard: an unserializable payload is a bug in the
    # caller, not an outage, and must not be silently swallowed.
    payload = json.dumps(event, default=str)
    # An empty user_ids is free and needs no guard at the call sites. redis-py's
    # Pipeline.execute opens with `if not stack and not self.watching: return []`
    # (verified against the pinned 5.2.1), so no command is sent and no connection
    # is taken from the pool. That matters because callers legitimately pass an
    # empty list — conversation_recipients returns nothing for a revoked
    # conversation — and adding `if not user_ids: return` to each of them would be
    # a branch per caller for no saving.
    async with degrade_on_outage("publish_to_users"):
        redis = get_redis()
        pipe = redis.pipeline()
        for uid in user_ids:
            pipe.publish(user_channel(uid), payload)
        await pipe.execute()
