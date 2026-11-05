"""The online index: counting who is online without reconstructing the answer.

`presence:{user_id}` stays the authority on whether ONE person is online. These
sorted sets exist only to answer HOW MANY, which previously meant either walking
the whole Redis keyspace (the superadmin tile) or reading a tenant's entire user
list out of Postgres and issuing one EXISTS per row (the org tile).

The index is advisory by design, and the last test here is the one that matters
most: every write into it is best-effort, so a drifted or unavailable index can
put a wrong number on an admin tile but can never take a socket down or turn a
user's dot the wrong colour.
"""

import time
import uuid

from app.realtime.redis_bus import get_redis
from app.services import presence


async def test_the_index_counts_people_not_sockets():
    """A phone and a browser are one person online, and closing one keeps them."""
    user, org = uuid.uuid4(), uuid.uuid4()

    assert await presence.mark_online(user, "conn-phone", org_id=org) is True
    # Second socket for the same user: not a new arrival, not a second body.
    assert await presence.mark_online(user, "conn-browser", org_id=org) is False
    assert await presence.count_online() == 1
    assert await presence.count_online_in_org(org) == 1

    # One socket closes. They are still here, so they must still be counted —
    # dropping them here would undercount until their next heartbeat re-added them.
    assert await presence.mark_offline(user, "conn-phone", org_id=org) is False
    assert await presence.count_online() == 1
    assert await presence.count_online_in_org(org) == 1
    assert await presence.is_online(user) is True

    assert await presence.mark_offline(user, "conn-browser", org_id=org) is True
    assert await presence.count_online() == 0
    assert await presence.count_online_in_org(org) == 0
    assert await presence.is_online(user) is False


async def test_the_index_is_org_scoped():
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    alice, bob, carol = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    await presence.mark_online(alice, "c1", org_id=org_a)
    await presence.mark_online(bob, "c2", org_id=org_a)
    await presence.mark_online(carol, "c3", org_id=org_b)

    assert await presence.count_online_in_org(org_a) == 2
    assert await presence.count_online_in_org(org_b) == 1
    assert await presence.count_online() == 3
    # An org with nobody online has no key at all, which must read as 0, not error.
    assert await presence.count_online_in_org(uuid.uuid4()) == 0


async def test_an_orgless_user_counts_globally_and_nowhere_else():
    """Superadmins carry no org_id. They are online; they are in no tenant."""
    root = uuid.uuid4()
    await presence.mark_online(root, "c1", org_id=None)
    assert await presence.count_online() == 1
    assert await presence.mark_offline(root, "c1", org_id=None) is True
    assert await presence.count_online() == 0


async def test_a_worker_that_died_ages_out_of_the_count():
    """Self-healing, on the same clock the presence keys use.

    A worker killed mid-connection never runs mark_offline, so its entries stay
    in the index. They must fall out on their own, exactly as the presence key
    they shadow expires, or the count only ever climbs.
    """
    org = uuid.uuid4()
    ghost, live = uuid.uuid4(), uuid.uuid4()
    await presence.mark_online(ghost, "c1", org_id=org)
    await presence.mark_online(live, "c2", org_id=org)
    assert await presence.count_online_in_org(org) == 2

    # Backdate the ghost past the TTL, as if nothing had refreshed it.
    stale = time.time() - presence._TTL - 1
    await get_redis().zadd(presence._INDEX, {str(ghost): stale})
    await get_redis().zadd(presence._org_index(org), {str(ghost): stale})

    assert await presence.count_online_in_org(org) == 1
    assert await presence.count_online() == 1
    # ...and the read trimmed it rather than merely skipping it, so the set does
    # not grow without bound.
    assert await get_redis().zscore(presence._INDEX, str(ghost)) is None


async def test_a_heartbeat_rescores_so_a_long_call_is_not_aged_out():
    """refresh() has to re-score, not just extend the presence key's TTL.

    Counting is by recency, so a connection whose index entry was never re-scored
    would drop out of the count while its presence key was still being kept alive
    right beside it — someone on a two-hour call vanishing from the tile.
    """
    org = uuid.uuid4()
    user = uuid.uuid4()
    await presence.mark_online(user, "c1", org_id=org)

    stale = time.time() - presence._TTL - 1
    await get_redis().zadd(presence._INDEX, {str(user): stale})
    await get_redis().zadd(presence._org_index(org), {str(user): stale})

    # The connection this test registered above. refresh() now SADDs, so passing a
    # different id would add a SECOND connection rather than heart-beating the
    # live one — the test would stay green while measuring something else.
    await presence.refresh(user, "c1", org_id=org)

    assert await presence.count_online() == 1
    assert await presence.count_online_in_org(org) == 1
    # The member set, not just the count: refresh() SADDs, so heart-beating the
    # wrong id would add a SECOND connection and leave both counts at 1 anyway.
    # Without this the test cannot tell a heartbeat from a new connection.
    members = await get_redis().smembers(presence._key(user))
    assert {m.decode() if isinstance(m, bytes) else m for m in members} == {"c1"}


async def test_a_stale_disconnect_cannot_evict_a_user_who_reconnected():
    """The interleaving that the count alone could not survive.

    mark_offline decides a user is gone from a SCARD and removes them from the
    index two awaits later. In between, that user's next socket can register —
    a refresh, a handover, a flaky network — and the removal would then land on
    someone who is online again, leaving the presence key holding the new socket
    while the index has dropped them.

    The removal re-reads the presence key instead of trusting that count, so it
    no-ops when a reconnect beat it. Driven in the exact order here rather than
    raced, so it fails deterministically if that guard goes.
    """
    org = uuid.uuid4()
    user = uuid.uuid4()

    await presence.mark_online(user, "old-socket", org_id=org)
    # The old socket drops: index entry goes, as it should.
    assert await presence.mark_offline(user, "old-socket", org_id=org) is True
    assert await presence.count_online() == 0

    # The new socket lands.
    await presence.mark_online(user, "new-socket", org_id=org)
    assert await presence.count_online() == 1

    # ...and only now does the old disconnect's removal get its turn.
    await presence._index_remove(user, org)

    assert await presence.is_online(user) is True
    assert await presence.count_online() == 1, "a stale disconnect evicted a live user"
    assert await presence.count_online_in_org(org) == 1


async def test_the_index_is_advisory_and_cannot_break_presence(monkeypatch):
    """Every index write is best-effort, and the counts degrade to 0.

    This is the property that makes the whole thing safe to add: is_online and
    get_statuses never read these sets, and nothing that writes them can raise
    into the connection lifecycle. A broken index costs an admin tile its number.
    """

    def _redis_is_down():
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr("app.services.presence.get_redis", _redis_is_down)

    # Neither direction raises...
    await presence._index_add(uuid.uuid4(), uuid.uuid4())
    await presence._index_remove(uuid.uuid4(), uuid.uuid4())
    # ...and the readers answer 0 rather than 500ing the dashboard around them.
    assert await presence.count_online() == 0
    assert await presence.count_online_in_org(uuid.uuid4()) == 0
