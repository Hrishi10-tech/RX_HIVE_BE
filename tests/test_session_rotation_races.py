"""Rotation under concurrency, and logging out a session that has moved on.

test_refresh_rotation.py pins down what rotation decides. These pin down that the
decision is actually reached: both the single-use rule and logout were read-then-
write on an unlocked row, so each could be stepped over by an ordinary second
request.

Measured before the fix, with four simultaneous refreshes presenting the SAME
token: four 200s and three live sessions. The extra sessions are the smaller half.
Every one of those requests read the token as unrevoked, so none entered the reuse
branch — the theft detection, and the family revocation it triggers, are exactly
what a captured cookie racing the real client would have walked past.
"""

import asyncio
import datetime as dt

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.api import auth as api_auth
from app.core.security import ACCESS_COOKIE, REFRESH_COOKIE, hash_refresh_token
from app.db.models import RefreshToken
from app.db.session import SessionLocal
from app.main import app
from tests.conftest import CSRF, login, make_user
from tests.test_refresh_rotation import _live_tokens, _refresh_with, _row_for


def _fresh_client() -> AsyncClient:
    """A client with no cookie jar of its own, so it presents only what we hand it."""
    c = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    c.headers.update(CSRF)
    return c


async def _logout_with(raw: str, access: str):
    async with _fresh_client() as c:
        return await c.post(
            "/api/auth/logout",
            headers={"Cookie": f"{REFRESH_COOKIE}={raw}; {ACCESS_COOKIE}={access}"},
        )


async def test_concurrent_refreshes_cannot_multiply_live_sessions(client):
    """One single-use token must never yield more than one live session.

    Without a row lock the branch that spots reuse is unreachable under
    concurrency: each request reads `revoked_at` as NULL, mints its own successor
    and commits. Four simultaneous refreshes produced three live sessions and not
    one 401.
    """
    user = await make_user("race-refresh@x.com")
    await login(client, "race-refresh@x.com")
    raw = client.cookies.get(REFRESH_COOKIE)

    async def one():
        async with _fresh_client() as c:
            return await _refresh_with(c, raw)

    results = await asyncio.gather(*[one() for _ in range(4)])
    codes = sorted(r.status_code for r in results)

    live = await _live_tokens(user.id)
    assert len(live) <= 1, f"one token yielded {len(live)} live sessions; codes={codes}"
    assert 401 in codes, (
        f"every request succeeded ({codes}), so the single-use rule was never "
        "enforced and the reuse branch was never reached"
    )


async def test_logging_out_a_rotated_token_ends_the_session_it_became(client):
    """Logout has to end the session, not the row it was handed.

    A client whose rotation response never arrived holds a spent token — the case
    `_undelivered_successor` exists for. Presenting it here used to revoke an
    already-revoked row, clear the caller's cookies and answer "Logged out
    successfully", while the live successor kept working until it expired days
    later. Anyone else holding that successor — one good reason the response went
    missing — kept a working session.
    """
    user = await make_user("logout-chain@x.com")
    await login(client, "logout-chain@x.com")
    lost = client.cookies.get(REFRESH_COOKIE)

    assert (await client.post("/api/auth/refresh")).status_code == 200
    successor = client.cookies.get(REFRESH_COOKIE)
    access = client.cookies.get(ACCESS_COOKIE)

    resp = await _logout_with(lost, access)
    assert resp.status_code == 200, resp.text

    assert (await _row_for(successor)).revoked_at is not None, "the successor outlived a successful logout"
    assert await _live_tokens(user.id) == []
    # And it is really dead, not merely marked: presenting it is refused.
    async with _fresh_client() as c:
        assert (await _refresh_with(c, successor)).status_code == 401


async def test_logout_ends_the_chain_and_leaves_other_sessions_alone(client):
    """The chain, not the family.

    Without this, revoking every live token for the user would satisfy the test
    above — and would sign this person out of the other browser they are working
    in. `_revoke_session_family` is deliberately wider, because theft implies other
    stolen cookies; a logout implies nothing about the user's other sessions.
    """
    await make_user("logout-scope@x.com")
    async with _fresh_client() as first, _fresh_client() as second:
        await login(first, "logout-scope@x.com")
        await login(second, "logout-scope@x.com")
        keep = second.cookies.get(REFRESH_COOKIE)

        going = first.cookies.get(REFRESH_COOKIE)
        assert (await first.post("/api/auth/refresh")).status_code == 200
        rotated = first.cookies.get(REFRESH_COOKIE)

        resp = await _logout_with(going, first.cookies.get(ACCESS_COOKIE))
        assert resp.status_code == 200

        assert (await _row_for(rotated)).revoked_at is not None, "the logged-out chain must die"
        assert (await _row_for(keep)).revoked_at is None, "the other session must survive"
        assert (await _refresh_with(second, keep)).status_code == 200


async def test_a_cyclic_rotation_link_does_not_hang_logout(client):
    """The walk follows a link written by an earlier request, so it guards itself.

    Specifically it needs the `seen` set, not the length bound: a set does not grow
    when the walk revisits a row, so on A -> B -> A the bound never trips. And
    `db.get` serves an identity-mapped row without awaiting anything, so the loop
    never yields — no request or task timeout can interrupt it, and one bad row
    pegs a worker while holding a row lock.

    Measured by removing the `seen` guard: this does not fail, it hangs the test
    process outright, which is what the same loop would do to a worker.
    """
    await make_user("logout-cycle@x.com")
    await login(client, "logout-cycle@x.com")
    first = client.cookies.get(REFRESH_COOKIE)
    assert (await client.post("/api/auth/refresh")).status_code == 200
    second = client.cookies.get(REFRESH_COOKIE)
    access = client.cookies.get(ACCESS_COOKIE)

    async with SessionLocal() as db:
        a = (
            await db.execute(select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(first)))
        ).scalar_one()
        b = (
            await db.execute(
                select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(second))
            )
        ).scalar_one()
        b.replaced_by_id = a.id  # close the loop
        a.revoked_at = None  # and make both look live, so neither is skipped
        await db.commit()

    resp = await asyncio.wait_for(_logout_with(first, access), timeout=15)
    assert resp.status_code == 200
    assert (await _row_for(first)).revoked_at is not None
    assert (await _row_for(second)).revoked_at is not None


async def test_logging_out_a_live_token_revokes_exactly_it(client):
    """The ordinary case still behaves: no chain, nothing else touched."""
    user = await make_user("logout-plain@x.com")
    await login(client, "logout-plain@x.com")
    raw = client.cookies.get(REFRESH_COOKIE)
    access = client.cookies.get(ACCESS_COOKIE)

    assert (await _logout_with(raw, access)).status_code == 200
    row = await _row_for(raw)
    assert row.revoked_at is not None
    assert row.replaced_by_id is None
    assert await _live_tokens(user.id) == []


async def test_logout_does_not_revoke_another_users_token(client):
    """The user_id check is still load-bearing after the rewrite."""
    await make_user("logout-mine@x.com")
    victim = await make_user("logout-theirs@x.com")
    async with _fresh_client() as mine, _fresh_client() as theirs:
        await login(mine, "logout-mine@x.com")
        await login(theirs, "logout-theirs@x.com")
        theirs_raw = theirs.cookies.get(REFRESH_COOKIE)

        # My access cookie, their refresh cookie.
        resp = await _logout_with(theirs_raw, mine.cookies.get(ACCESS_COOKIE))
        assert resp.status_code == 200
        assert (await _row_for(theirs_raw)).revoked_at is None
        assert len(await _live_tokens(victim.id)) == 1


async def test_an_expired_successor_still_gets_revoked_by_logout(client):
    """Revocation is not conditional on the successor still being useful.

    A row that is expired but unrevoked is exactly what an audit of live sessions
    reads as live, and skipping it would leave the chain half-dead for no reason.
    """
    await make_user("logout-expired@x.com")
    await login(client, "logout-expired@x.com")
    first = client.cookies.get(REFRESH_COOKIE)
    assert (await client.post("/api/auth/refresh")).status_code == 200
    second = client.cookies.get(REFRESH_COOKIE)
    access = client.cookies.get(ACCESS_COOKIE)

    async with SessionLocal() as db:
        row = (
            await db.execute(
                select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(second))
            )
        ).scalar_one()
        row.expires_at = row.expires_at - dt.timedelta(days=365)
        await db.commit()

    assert (await _logout_with(first, access)).status_code == 200
    assert (await _row_for(second)).revoked_at is not None


async def test_logout_sees_a_rotation_that_committed_while_it_was_deciding(client):
    """The chain walk is only as good as the row it starts from.

    Read the token without locking and the walk starts from a snapshot taken before
    the rotation: `replaced_by_id` is still NULL in memory, so logout revokes the
    predecessor of a session that now exists and reports success. The lock makes
    the read wait for the rotation to commit, so the walk sees where the session
    went.

    Driven with a real second transaction rather than a monkeypatch, because what
    is under test is whether the read takes the lock at all.
    """
    user = await make_user("logout-race@x.com")
    await login(client, "logout-race@x.com")
    presented = client.cookies.get(REFRESH_COOKIE)
    access = client.cookies.get(ACCESS_COOKIE)

    holding = asyncio.Event()
    release = asyncio.Event()
    successor_id: list = []

    async def rotator() -> None:
        """Rotate the presented token the way /refresh does, and hold it uncommitted."""
        async with SessionLocal() as db:
            row = (
                await db.execute(
                    select(RefreshToken)
                    .where(RefreshToken.token_hash == hash_refresh_token(presented))
                    .with_for_update()
                )
            ).scalar_one()
            issued = RefreshToken(
                user_id=row.user_id,
                token_hash=hash_refresh_token("successor-of-logout-race"),
                client=row.client,
                expires_at=row.expires_at,
            )
            db.add(issued)
            await db.flush()
            row.revoked_at = dt.datetime.now(dt.UTC)
            row.replaced_by_id = issued.id
            successor_id.append(issued.id)
            holding.set()
            await release.wait()
            await db.commit()

    task = asyncio.create_task(rotator())
    await asyncio.wait_for(holding.wait(), timeout=5)

    request = asyncio.create_task(_logout_with(presented, access))
    await asyncio.sleep(0.5)
    blocked = not request.done()
    release.set()
    await task
    resp = await request

    assert blocked, (
        "logout completed while a rotation held the token row, so its chain walk "
        "started from a row that did not yet know about the successor"
    )
    assert resp.status_code == 200, resp.text
    async with SessionLocal() as db:
        assert (await db.get(RefreshToken, successor_id[0])).revoked_at is not None, (
            "the session the token was rotated into outlived the logout"
        )
    assert await _live_tokens(user.id) == []


async def test_logout_follows_a_rotation_that_happens_further_down_the_chain(client):
    """Locking the presented token is not enough — every node needs the lock.

    Lock only the row the caller presented and the rest of the chain stays open: a
    refresh can rotate successor B into C after the walk has read B, and the B it
    read still says `replaced_by_id IS NULL`, so the walk stops there and C stays
    live. That is the same defect as logging out a rotated token, one link further
    along, and it survives the fix for the first one.
    """
    user = await make_user("logout-deep-race@x.com")
    await login(client, "logout-deep-race@x.com")
    presented = client.cookies.get(REFRESH_COOKIE)

    assert (await client.post("/api/auth/refresh")).status_code == 200
    middle = client.cookies.get(REFRESH_COOKIE)
    access = client.cookies.get(ACCESS_COOKIE)

    holding = asyncio.Event()
    release = asyncio.Event()
    tail_id: list = []

    async def rotate_the_middle() -> None:
        """Rotate B -> C while holding B, so only a locking read of B can see C."""
        async with SessionLocal() as db:
            row = (
                await db.execute(
                    select(RefreshToken)
                    .where(RefreshToken.token_hash == hash_refresh_token(middle))
                    .with_for_update()
                )
            ).scalar_one()
            issued = RefreshToken(
                user_id=row.user_id,
                token_hash=hash_refresh_token("tail-of-deep-race"),
                client=row.client,
                expires_at=row.expires_at,
            )
            db.add(issued)
            await db.flush()
            row.revoked_at = dt.datetime.now(dt.UTC)
            row.replaced_by_id = issued.id
            tail_id.append(issued.id)
            holding.set()
            await release.wait()
            await db.commit()

    task = asyncio.create_task(rotate_the_middle())
    await asyncio.wait_for(holding.wait(), timeout=5)

    request = asyncio.create_task(_logout_with(presented, access))
    await asyncio.sleep(0.5)
    blocked = not request.done()
    release.set()
    await task
    resp = await request

    assert blocked, (
        "the walk read the middle of the chain without locking it, so it could not "
        "see the rotation that was in flight there"
    )
    assert resp.status_code == 200, resp.text
    async with SessionLocal() as db:
        assert (await db.get(RefreshToken, tail_id[0])).revoked_at is not None, (
            "the session at the end of the chain outlived the logout"
        )
    assert await _live_tokens(user.id) == []


async def test_a_grace_replay_waits_for_a_rotation_of_its_successor(client):
    """The grace path reads the successor to decide, then rotates from it.

    Read it unlocked and a concurrent /refresh can lock and spend that successor in
    between, leaving the replay to honour a grace window against state that no
    longer holds and fork a second live branch off it. Blocking means the existing
    reuse policy is applied to the successor's committed state — which, once the
    successor has been spent, is the theft branch rather than the grace branch.
    """
    user = await make_user("grace-race@x.com")
    await login(client, "grace-race@x.com")
    lost = client.cookies.get(REFRESH_COOKIE)

    assert (await client.post("/api/auth/refresh")).status_code == 200
    successor = client.cookies.get(REFRESH_COOKIE)

    holding = asyncio.Event()
    release = asyncio.Event()

    async def spend_the_successor() -> None:
        """Rotate the successor while holding it, so only a locking read sees it spent."""
        async with SessionLocal() as db:
            row = (
                await db.execute(
                    select(RefreshToken)
                    .where(RefreshToken.token_hash == hash_refresh_token(successor))
                    .with_for_update()
                )
            ).scalar_one()
            issued = RefreshToken(
                user_id=row.user_id,
                token_hash=hash_refresh_token("tail-of-grace-race"),
                client=row.client,
                expires_at=row.expires_at,
            )
            db.add(issued)
            await db.flush()
            row.revoked_at = dt.datetime.now(dt.UTC)
            row.replaced_by_id = issued.id
            holding.set()
            await release.wait()
            await db.commit()

    task = asyncio.create_task(spend_the_successor())
    await asyncio.wait_for(holding.wait(), timeout=5)

    async def replay():
        async with _fresh_client() as c:
            return await _refresh_with(c, lost)

    request = asyncio.create_task(replay())
    await asyncio.sleep(0.5)
    blocked = not request.done()
    release.set()
    await task
    resp = await request

    assert blocked, (
        "the replay read its successor without locking it, so it decided the "
        "grace question against a successor that was being spent"
    )
    # The successor was spent while this waited, so the committed state is reuse of
    # a token that has already moved on: the theft branch, not the grace branch.
    assert resp.status_code == 401, resp.text
    assert await _live_tokens(user.id) == [], "the family should have been burned"


async def _extend_chain(raw: str, links: int) -> list:
    """Append `links` rotated successors after the row holding `raw`.

    Built directly, because reaching this length through the endpoint would be a
    few hundred HTTP round trips to prove something about one SQL walk. The shape
    is what /refresh produces: each row revoked and pointing at the next, the last
    one live.
    """
    ids = []
    async with SessionLocal() as db:
        node = (
            await db.execute(select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(raw)))
        ).scalar_one()
        for i in range(links):
            issued = RefreshToken(
                user_id=node.user_id,
                token_hash=hash_refresh_token(f"{raw}-link-{i}"),
                client=node.client,
                expires_at=node.expires_at,
            )
            db.add(issued)
            await db.flush()
            node.revoked_at = dt.datetime.now(dt.UTC)
            node.replaced_by_id = issued.id
            ids.append(issued.id)
            node = issued
        await db.commit()
    return ids


async def test_a_lineage_longer_than_the_old_cap_is_revoked_to_its_end(client):
    """A cap on the walk was a cap on correctness, and 64 was reachable.

    A web session refreshing on a 15-minute access token rotates four times an
    hour, so an ordinary long-lived session passes 64 links inside a day. Stopping
    there let logout answer "Logged out successfully" with live descendants still
    reachable — the defect this whole function exists to fix, back again past an
    arbitrary line.
    """
    user = await make_user("logout-long-chain@x.com")
    await login(client, "logout-long-chain@x.com")
    head = client.cookies.get(REFRESH_COOKIE)
    access = client.cookies.get(ACCESS_COOKIE)

    ids = await _extend_chain(head, 70)
    assert len(await _live_tokens(user.id)) == 1, "only the tail of the chain is live"

    assert (await _logout_with(head, access)).status_code == 200

    async with SessionLocal() as db:
        unrevoked = [str(i) for i in ids if (await db.get(RefreshToken, i)).revoked_at is None]
    assert unrevoked == [], f"{len(unrevoked)} descendant(s) survived the logout"
    assert await _live_tokens(user.id) == []


async def test_hitting_the_runaway_limit_does_not_report_a_quiet_success(client, monkeypatch):
    """The limit is a budget, not a correctness boundary.

    If the walk ever cannot finish, the one outcome that must not happen is a 200
    with a live descendant. Falling back to the family is wider than the lineage —
    not what a logout should normally do — but it cannot leave one behind.
    """
    user = await make_user("logout-runaway@x.com")
    await login(client, "logout-runaway@x.com")
    head = client.cookies.get(REFRESH_COOKIE)
    access = client.cookies.get(ACCESS_COOKIE)

    ids = await _extend_chain(head, 8)
    monkeypatch.setattr(api_auth, "_ROTATION_CHAIN_RUNAWAY_LIMIT", 3)

    assert (await _logout_with(head, access)).status_code == 200

    async with SessionLocal() as db:
        unrevoked = [str(i) for i in ids if (await db.get(RefreshToken, i)).revoked_at is None]
    assert unrevoked == [], f"{len(unrevoked)} descendant(s) survived past the limit"
    assert await _live_tokens(user.id) == []


async def test_the_runaway_fallback_still_spares_another_client(client, monkeypatch):
    """The fallback is the family, which stops at the client boundary.

    So even the pathological path cannot sign a person out of the browser they are
    working in on the strength of a phone logout — the property
    refresh_tokens.client exists for.
    """
    user = await make_user("logout-runaway-scope@x.com", mobile_access=True)
    async with _fresh_client() as web, _fresh_client() as phone:
        await login(web, "logout-runaway-scope@x.com")
        keep = web.cookies.get(REFRESH_COOKIE)
        await phone.post(
            "/api/auth/login",
            json={
                "email": "logout-runaway-scope@x.com",
                "password": "TestPass1234",
                "client": "mobile",
            },
        )
        mobile_raw = phone.cookies.get(REFRESH_COOKIE)
        assert mobile_raw is not None, "the mobile session is the whole point of this test"

        await _extend_chain(mobile_raw, 8)
        monkeypatch.setattr(api_auth, "_ROTATION_CHAIN_RUNAWAY_LIMIT", 3)
        assert (await _logout_with(mobile_raw, phone.cookies.get(ACCESS_COOKIE))).status_code == 200

    assert (await _row_for(keep)).revoked_at is None, "the web session must survive"
    assert [t.id for t in await _live_tokens(user.id)] == [(await _row_for(keep)).id]
