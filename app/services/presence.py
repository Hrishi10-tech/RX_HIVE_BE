"""Presence backed by Redis — works across any number of API workers.

A user is online while at least one worker holds a live WebSocket for them.
Each connection registers itself under presence:{user_id} with a TTL that the
connection's heartbeat keeps refreshing, so crashed workers can't leak
"online forever" state.
"""

import time
import uuid

from app.realtime.redis_bus import degrade_on_outage, get_redis

_TTL = 90  # seconds; heartbeats arrive at least every 60s


def _key(user_id) -> str:
    return f"presence:{user_id}"


# ---------------------------------------------------------------------------
# Online index
# ---------------------------------------------------------------------------
#
# "How many people are online" had no answer that did not reconstruct one. The
# superadmin dashboard SCANned the whole keyspace for presence:* — SCAN walks
# every key and MATCH filters server-side, so the cost was the size of the
# keyspace (ratelimit:, call:, direct: and user: entries included) rather than
# the number of people online. The org dashboard read every active user id in
# the tenant out of Postgres and then issued one EXISTS per user, per load.
#
# These sorted sets are that index: member is a user id, score is the last time
# they were seen. Counting is ZCARD after dropping everything older than the
# same TTL the presence keys use, which makes the count self-healing — a worker
# that dies without deregistering leaves entries that age out of the window and
# are trimmed by the next read, exactly as its presence keys expire.
#
# ADVISORY, and deliberately so. Nothing reads these to answer is_online or
# get_statuses; those still go to presence:{user_id}, which stays the single
# authority for whether one person is online. Every write here is best-effort
# and cannot fail the operation it accompanies. The worst a drifted index can do
# is put a wrong number on an admin tile — never a wrong green dot on a user.
_INDEX = "presence:index"


def _org_index(org_id) -> str:
    return f"presence:index:org:{org_id}"


async def _index_add(user_id, org_id) -> None:
    async with degrade_on_outage("presence.index_add"):
        now = time.time()
        pipe = get_redis().pipeline()
        pipe.zadd(_INDEX, {str(user_id): now})
        if org_id is not None:
            pipe.zadd(_org_index(org_id), {str(user_id): now})
        await pipe.execute()


# Compare-and-delete against the authority, in one round trip.
#
# mark_offline decides a user is gone from a SCARD, and only then removes them
# here — two awaits apart, which is room enough for that user's next socket to
# register in between. The removal would then land on a user who is online
# again: the presence key holds the new socket while the index has dropped them,
# and the count stays one short until their next heartbeat re-adds them.
#
# So the count is not what decides. This re-reads presence:{user_id} — the same
# key is_online answers from — inside the script, and removes only if it is
# genuinely gone. A reconnect that beat us here recreated that key, and the
# removal correctly does nothing.
_REMOVE_IF_GONE = """
if redis.call('EXISTS', KEYS[1]) == 1 then
  return 0
end
for i = 2, #KEYS do
  redis.call('ZREM', KEYS[i], ARGV[1])
end
return 1
"""


async def _index_remove(user_id, org_id) -> None:
    async with degrade_on_outage("presence.index_remove"):
        keys = [_key(user_id), _INDEX]
        if org_id is not None:
            keys.append(_org_index(org_id))
        await get_redis().eval(_REMOVE_IF_GONE, len(keys), *keys, str(user_id))


async def _count(key: str) -> int:
    """Trim the stale tail, then count what's left. Degrades to 0."""
    count = 0
    async with degrade_on_outage("presence.count"):
        pipe = get_redis().pipeline()
        pipe.zremrangebyscore(key, "-inf", time.time() - _TTL)
        pipe.zcard(key)
        _, total = await pipe.execute()
        count = int(total)
    return count


async def count_online() -> int:
    """Everyone online, across every tenant."""
    return await _count(_INDEX)


async def count_online_in_org(org_id) -> int:
    """Everyone online inside one organisation."""
    return await _count(_org_index(org_id))


async def mark_online(user_id: uuid.UUID, conn_id: str, *, org_id=None) -> bool:
    """Register a connection. Returns True if the user just came online.

    MULTI, because the caller uses the result as an edge trigger and separate
    round trips lose that edge: when a phone and a browser tab connect within the
    same few milliseconds both SADDs land before either SCARD, so both read
    count == 2, both return False, and nobody ever publishes the `online`
    presence event — the user stays grey to every conversation partner until
    something unrelated forces a refetch.
    """
    # Guarded like every read in this module. hub.py opens its try before this
    # call so a raise here is torn down cleanly, which makes this the least urgent
    # of the three — but raising still refuses a websocket Redis merely could not
    # be asked about, and the module's rule is that an outage costs a green dot.
    #
    # added/count stay 0 on an outage, so no edge is claimed: the caller treats
    # True as "just came online" and publishes a presence event from it.
    added, count, wrote = 0, 0, False
    async with degrade_on_outage("presence.mark_online"):
        pipe = get_redis().pipeline(transaction=True)
        key = _key(user_id)
        pipe.sadd(key, conn_id)
        pipe.expire(key, _TTL)
        pipe.scard(key)
        added, _, count = await pipe.execute()
        wrote = True
    # After the authoritative write, and best-effort: the index must never be
    # able to fail a connection that Redis has already accepted as online.
    #
    # Only if that write LANDED, though. Guarding the pipeline above opened a new
    # way for the two to disagree: a blip that fails the authoritative write and
    # then recovers would leave the index claiming this user is online while their
    # presence key holds no connection at all. Same drift refresh() was just fixed
    # for, arriving by the opposite route — and unlike a count ageing out of its
    # window, nothing reconciles this one.
    if wrote:
        await _index_add(user_id, org_id)
    return bool(added) and count == 1


async def refresh(user_id: uuid.UUID, conn_id: str, *, org_id=None) -> None:
    """Re-assert that this connection is alive. Best-effort, and that matters here.

    This is the websocket loop's `ping` branch (realtime/hub.py). That branch sits
    OUTSIDE the try/except wrapping _handle_inbound, and the enclosing try catches
    only WebSocketDisconnect — so a RedisError raised here left the endpoint
    altogether. One blip therefore dropped every connected socket on its next
    ping, within the 60-second heartbeat, all at the same moment: the outage
    turned into a reconnect storm on top of itself.

    SADD, not a bare EXPIRE, and it takes conn_id for that reason.

    EXPIRE on a missing key returns 0 and does nothing — measured, after asserting
    the opposite in an earlier version of this docstring. So once the key had aged
    out, which is exactly what an outage longer than _TTL causes, no number of
    pings could bring the user back: they stayed offline until their socket
    reconnected and mark_online ran. Worse, _index_add below ZADDs unconditionally,
    so every ping re-added them to the advisory index while the authority still
    said offline — count_online reported them online with no green dot anywhere to
    match, and the drift persisted instead of ageing out.

    Re-adding the member makes the heartbeat mean what it says: this connection is
    alive, so this user is online. It is idempotent for the ordinary case, since
    the member is already in the set.

    No `online` edge is published from here. A resurrection after an outage is not
    a state change anyone can be told about reliably — publish_to_users was
    degraded through the same outage — and peers refetch on reconnect.
    """
    wrote = False
    async with degrade_on_outage("presence.refresh"):
        key = _key(user_id)
        pipe = get_redis().pipeline(transaction=True)
        pipe.sadd(key, conn_id)
        pipe.expire(key, _TTL)
        await pipe.execute()
        wrote = True
    # Re-scores rather than merely extending: the index counts by recency, so a
    # connection that never re-scored would age out of the count while its
    # presence key was still being kept alive right beside it.
    #
    # Conditional for the same reason as in mark_online: re-scoring off the back of
    # a failed authoritative write is how the index and the authority come to
    # disagree with nothing to reconcile them.
    if wrote:
        await _index_add(user_id, org_id)


async def mark_offline(user_id: uuid.UUID, conn_id: str, *, org_id=None) -> bool | None:
    """Deregister a connection.

    True  — this was the last socket, so the user just went offline.
    False — other sockets remain.
    None  — Redis could not be asked, so the answer is UNKNOWN.

    None rather than False for the unknown case, because the two are not the same
    question and one caller cannot treat them alike. A client error does not prove
    the transaction had no effect: Redis can apply the MULTI/EXEC and lose the
    reply on the way back, in which case the SREM really did remove the last
    connection and this user really is offline. Returning False there told the
    caller "other sockets remain", which is a claim, not an absence of one.

    MULTI for the same reason as mark_online: the `offline` broadcast is an edge.
    """
    # Guarded because this runs in the websocket loop's `finally`. Raising from a
    # finally masks whatever actually ended the connection and skips every line
    # after it — including handle_user_link_down, the grace window that keeps a
    # call alive across a reconnect. hub.py's comment above its try lists that
    # skip as one of the consequences it moved the try earlier to prevent.
    #
    # The sentinel is -1, NOT 0, and that is the whole point of it: 0 is the value
    # that means "the last socket just closed". Initialising to 0 would make an
    # outage indistinguishable from a real disconnect and fire the offline edge —
    # greying out a user who may still hold a socket on another worker, which is
    # the one thing this module says it will never do.
    count = -1
    async with degrade_on_outage("presence.mark_offline"):
        pipe = get_redis().pipeline(transaction=True)
        key = _key(user_id)
        pipe.srem(key, conn_id)
        pipe.scard(key)
        _, count = await pipe.execute()
    if count < 0:
        return None
    if count == 0:
        # Only when the last socket goes. A user with a phone and a browser open
        # is still online after one of them closes, and dropping them from the
        # index on the first close would undercount them until their next
        # heartbeat re-added them.
        await _index_remove(user_id, org_id)
    return count == 0


async def is_online(user_id) -> bool:
    online = False
    async with degrade_on_outage("presence.is_online"):
        online = await get_redis().exists(_key(user_id)) > 0
    return online


async def get_statuses(user_ids: list) -> dict[str, str]:
    """Batch presence lookup: {user_id_str: "online"|"offline"}.

    Degrades to all-offline when Redis is unreachable. This lookup decorates
    responses whose real payload came from Postgres — including the login and
    send paths — so an outage must cost a green dot, not the whole request.
    """
    if not user_ids:
        return {}
    statuses = {str(uid): "offline" for uid in user_ids}
    async with degrade_on_outage("presence.get_statuses"):
        redis = get_redis()
        pipe = redis.pipeline()
        for uid in user_ids:
            pipe.exists(_key(uid))
        results = await pipe.execute()
        for uid, exists in zip(user_ids, results, strict=True):
            statuses[str(uid)] = "online" if exists else "offline"
    return statuses
