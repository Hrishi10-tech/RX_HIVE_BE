"""Presence during a Redis outage, against the promise the module already makes.

services/presence.py states the rule in get_statuses' own docstring — "an outage
must cost a green dot, not the whole request" — and keeps it in five places:
_index_add, _index_remove, _count, is_online and get_statuses all run their Redis
work inside degrade_on_outage.

Three functions did not, and two of those are reached on every connection:

  * refresh() is the ping branch of the websocket loop (realtime/hub.py). It sits
    OUTSIDE the try/except that wraps _handle_inbound, and the enclosing try
    catches only WebSocketDisconnect — so a RedisError there leaves the endpoint
    entirely. A blip drops every connected socket on its next ping, within the
    60s heartbeat, all at once.
  * mark_offline() runs in that loop's `finally`. Raising from a finally masks
    whatever actually ended the connection and skips the rest of the teardown —
    including handle_user_link_down, which is the grace window that keeps a call
    alive across a reconnect. hub.py's own comment lists that skip as one of the
    consequences it moved the `try` earlier to prevent.
  * mark_online() raises into a guarded region and is torn down correctly, so it
    is left alone; see the note on it below.
"""

import uuid

import pytest

# redis-py's ConnectionError, not the builtin. They are different exceptions:
# redis.exceptions.ConnectionError subclasses RedisError and NOT OSError, while
# the builtin subclasses OSError. degrade_on_outage catches (RedisError, OSError),
# so a double raising the builtin is handled through the OSError arm and never
# exercises the arm a real redis-py failure takes — a test that passes for a
# reason other than the one it names. Narrowing that guard to OSError alone would
# break every outage path in production while leaving these green.
from redis.exceptions import ConnectionError as RedisConnectionError

from app.realtime import hub
from app.realtime.redis_bus import get_redis
from app.services import presence
from tests.conftest import make_org, make_user

_REFUSED = "Error 61 connecting to localhost:6379. Connection refused."


class _DeadPipeline:
    """Buffers like the real one and dies on execute, as a broken client does."""

    def __getattr__(self, _name):
        def _queue(*_args, **_kwargs):
            return self

        return _queue

    async def execute(self, *_args, **_kwargs):
        raise RedisConnectionError(_REFUSED)


class _DeadRedis:
    def pipeline(self, *_args, **_kwargs):
        return _DeadPipeline()

    def __getattr__(self, _name):
        async def _raise(*_args, **_kwargs):
            raise RedisConnectionError(_REFUSED)

        return _raise


@pytest.fixture
def dead_presence_redis(monkeypatch):
    """Break Redis for presence only, so each function's own guard is what shows."""
    monkeypatch.setattr(presence, "get_redis", lambda: _DeadRedis())


async def test_refresh_degrades_instead_of_dropping_the_socket(client, dead_presence_redis):
    """The ping path. This raising is a disconnect for every user at once.

    hub.py calls this in the `ping` branch and then `continue`s, outside the
    try/except that guards _handle_inbound — and the enclosing try catches only
    WebSocketDisconnect.
    """
    user = await make_user("pres-refresh@x.com")
    await presence.refresh(user.id, "conn-1", org_id=user.org_id)


async def test_mark_offline_degrades_instead_of_raising_from_finally(client, dead_presence_redis):
    """Raising here masks what actually ended the connection.

    It also skips the teardown after it, which includes the call grace window —
    the exact consequence hub.py's comment says moving the `try` earlier fixed.
    """
    user = await make_user("pres-offline@x.com")
    went_offline = await presence.mark_offline(user.id, "conn-1", org_id=user.org_id)
    assert went_offline is None, (
        "an unreachable Redis is UNKNOWN, not False. A client error does not prove "
        "the transaction had no effect — Redis can apply the EXEC and lose the "
        "reply — so False would be a claim rather than the absence of one, and the "
        "call path reads it as 'other sockets remain'"
    )


async def test_mark_online_degrades_and_claims_no_edge(client, dead_presence_redis):
    """Guarded by the caller, but it must not claim an edge it cannot know about.

    hub.py opens its try before this call precisely so a raise here is torn down
    cleanly, so this is the least urgent of the three. It still must not return
    True on an outage: the caller treats True as "just came online" and publishes
    a presence event from it.
    """
    user = await make_user("pres-online@x.com")
    came_online = await presence.mark_online(user.id, "conn-1", org_id=user.org_id)
    assert came_online is False


async def test_the_read_paths_already_degraded(client, dead_presence_redis):
    """Control. These five were already guarded, and the fix must not change them."""
    user = await make_user("pres-read@x.com")
    assert await presence.is_online(user.id) is False
    assert await presence.get_statuses([user.id]) == {str(user.id): "offline"}
    assert await presence.count_online() == 0
    assert await presence.count_online_in_org(uuid.uuid4()) == 0


# ---------------------------------------------------------------------------
# Recovery after an outage longer than the TTL
# ---------------------------------------------------------------------------


async def test_a_heartbeat_brings_a_user_back_after_the_key_aged_out(client):
    """EXPIRE on a missing key does nothing, so a bare TTL bump could not recover.

    An outage longer than _TTL is exactly what ages the key out. Measured before
    the fix: EXPIRE returned 0, the key stayed gone, and no number of pings brought
    the user back — they were offline until their socket reconnected and
    mark_online ran. Meanwhile _index_add ZADDs unconditionally, so every ping
    re-added them to the advisory index while the authority still said offline:
    count_online reported them online with no green dot anywhere to match.
    """
    # An org, because the org index is only written when org_id is not None and
    # the point of this test is that the two views agree again afterwards.
    org = await make_org("Presence Revive Co")
    user = await make_user("pres-revive@x.com", org_id=org.id)
    redis = get_redis()

    await presence.mark_online(user.id, "conn-1", org_id=user.org_id)
    assert await presence.is_online(user.id) is True

    # What an outage longer than the TTL leaves behind.
    await redis.delete(presence._key(user.id))
    assert await presence.is_online(user.id) is False

    await presence.refresh(user.id, "conn-1", org_id=user.org_id)

    assert await presence.is_online(user.id) is True, (
        "the heartbeat did not re-register the connection, so the user stays "
        "offline until their socket reconnects"
    )
    assert await presence.get_statuses([user.id]) == {str(user.id): "online"}
    # And the two agree again, which is the drift this closes.
    assert await presence.count_online_in_org(user.org_id) == 1


async def test_a_heartbeat_is_idempotent_for_a_live_connection(client):
    """The ordinary case: the member is already in the set, so nothing changes."""
    user = await make_user("pres-idem@x.com")
    await presence.mark_online(user.id, "conn-1", org_id=user.org_id)

    for _ in range(3):
        await presence.refresh(user.id, "conn-1", org_id=user.org_id)

    members = await get_redis().smembers(presence._key(user.id))
    assert {m.decode() if isinstance(m, bytes) else m for m in members} == {"conn-1"}


# ---------------------------------------------------------------------------
# A blip that recovers between the two writes
# ---------------------------------------------------------------------------


class _RecoveringRedis:
    """Fails the first pipeline, then behaves like the real client.

    Models the interleaving that guarding the authoritative write opened up: the
    pipeline fails, degrade_on_outage swallows it, and by the time _index_add runs
    Redis is answering again. Every other attribute delegates, so the index write
    lands for real and the assertion is about the index's actual contents.
    """

    def __init__(self, real):
        self._real = real
        self._failed = False

    def pipeline(self, *args, **kwargs):
        if not self._failed:
            self._failed = True
            return _DeadPipeline()
        return self._real.pipeline(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def redis_that_recovers(monkeypatch):
    """ONE instance, deliberately.

    `lambda: _RecoveringRedis(get_redis())` builds a fresh double per call, so its
    "first pipeline" counter resets and EVERY pipeline dies — which made the two
    tests below pass for the wrong reason: they were exercising a total outage, not
    a blip that recovers between the authoritative write and the index write.
    Caught by mutation: removing the `if wrote:` guards did not fail them.
    """
    double = _RecoveringRedis(get_redis())
    monkeypatch.setattr(presence, "get_redis", lambda: double)


async def test_a_failed_authoritative_write_does_not_still_index_the_user(client, redis_that_recovers):
    """The index must not claim someone is online whose presence key is empty.

    presence.py's own comment allows the advisory index to put a wrong number on an
    admin tile, on the grounds that a drifted entry ages out of the window. This
    drift would not: mark_online would keep being called on each new socket and
    keep re-adding an entry for a user with no connection recorded.
    """
    org = await make_org("Presence Blip Co")
    user = await make_user("pres-blip@x.com", org_id=org.id)

    came_online = await presence.mark_online(user.id, "conn-1", org_id=org.id)

    assert came_online is False
    assert await presence.is_online(user.id) is False, "the authoritative write failed"
    assert await presence.count_online_in_org(org.id) == 0, (
        "the index recorded a user the authority has no connection for"
    )


async def test_a_failed_heartbeat_does_not_still_rescore_the_index(client, redis_that_recovers):
    """Same rule on the refresh path, which re-scores rather than adds."""
    org = await make_org("Presence Blip Refresh Co")
    user = await make_user("pres-blip-refresh@x.com", org_id=org.id)

    await presence.refresh(user.id, "conn-1", org_id=org.id)

    assert await presence.is_online(user.id) is False
    assert await presence.count_online_in_org(org.id) == 0


# ---------------------------------------------------------------------------
# The call grace window's own decision (issue #89)
# ---------------------------------------------------------------------------


async def test_mark_offline_reports_the_three_outcomes_distinctly(client):
    """True, False and None are three different answers and the caller needs all."""
    org = await make_org("Presence Tri Co")
    user = await make_user("pres-tri@x.com", org_id=org.id)

    await presence.mark_online(user.id, "c1", org_id=org.id)
    await presence.mark_online(user.id, "c2", org_id=org.id)

    assert await presence.mark_offline(user.id, "c1", org_id=org.id) is False, (
        "one of two sockets closed — the user is still online"
    )
    assert await presence.mark_offline(user.id, "c2", org_id=org.id) is True, "the last socket closed"


async def test_mark_offline_is_unknown_not_false_when_redis_cannot_be_asked(client, dead_presence_redis):
    user = await make_user("pres-unknown@x.com")
    assert await presence.mark_offline(user.id, "c1", org_id=user.org_id) is None


async def test_an_unknown_result_asks_the_authority_before_skipping_the_grace_window(client):
    """Issue #89. Redis can apply the EXEC and lose the reply on the way back.

    The SREM really removed the last connection, so the user really has no live
    transport — but mark_offline could not read the SCARD and returns None.
    Treating that as "other sockets remain" skipped handle_user_link_down and lost
    a call that would have survived a reconnect.
    """
    org = await make_org("Grace Unknown Co")
    user = await make_user("grace-unknown@x.com", org_id=org.id)

    # No presence key: exactly the state an applied-but-unacknowledged SREM leaves.
    assert await presence.is_online(user.id) is False
    assert await hub._lost_last_transport(user.id, None) is True, (
        "an unknown result plus an authority that says the user is gone must open the grace window"
    )


async def test_an_unknown_result_does_not_open_a_window_for_a_user_still_connected(client):
    """The converse, and the reason this is not just an unconditional call.

    Someone closing one of two tabs must not arm a grace deadline or show their
    peers "Connecting…" for a call that never lost its transport.
    """
    org = await make_org("Grace Still Here Co")
    user = await make_user("grace-still@x.com", org_id=org.id)
    await presence.mark_online(user.id, "other-tab", org_id=org.id)

    assert await presence.is_online(user.id) is True
    assert await hub._lost_last_transport(user.id, None) is False


async def test_a_known_result_is_taken_at_its_word(client):
    """A definite answer is not second-guessed with an extra round trip."""
    org = await make_org("Grace Known Co")
    user = await make_user("grace-known@x.com", org_id=org.id)
    # Online, but mark_offline already said True: the known answer wins.
    await presence.mark_online(user.id, "c1", org_id=org.id)

    assert await hub._lost_last_transport(user.id, True) is True
    assert await hub._lost_last_transport(user.id, False) is False
