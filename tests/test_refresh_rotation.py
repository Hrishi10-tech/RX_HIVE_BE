"""Refresh rotation: delivery safety on one side, reuse detection on the other.

Rotation used to be a pure single-use rule, which quietly assumed the rotation
response always arrives. It does not — the mobile client gives up after 30s, a
proxy can 502 after the commit, iOS suspends the app mid-request — and the client
was then holding a token the server had spent, so the next refresh signed the user
out of a healthy session.

The rules these tests pin down:
  1. A normal rotation still spends the predecessor, and records its successor.
  2. Replaying a rotated token inside the grace window, while its successor is
     unspent, is an undelivered rotation: 200, and the new pair works.
  3. Anything else is reuse of a spent token — successor already used, or outside
     the window — and burns that client's whole session family, not just the row
     presented. Stricter than the previous build, which failed only that token.
  4. Family revocation stops at the client boundary: a stolen mobile token must
     not sign the same person out of the browser they are working in.
  5. A Redis outage on the limiter in front of /refresh does not end sessions.
  6. A revoked token that was never rotated — no successor, so a logout, a reset,
     a password change or a deactivation ended it — is refused and nothing else:
     no family burn, no theft warning (batch 73 review). A device signed out by a
     change made elsewhere must not sign out the device that made it.
"""

import datetime as dt
import logging

import pytest
from httpx import ASGITransport, AsyncClient
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import select

from app.api.auth import MOBILE_NOT_APPROVED_CODE
from app.core import rate_limit
from app.core.security import REFRESH_COOKIE, hash_refresh_token
from app.db.models import RefreshToken, User
from app.db.session import SessionLocal
from app.main import app
from tests.conftest import CSRF, login, make_user


async def _refresh_with(client: AsyncClient, raw: str):
    """Present exactly this refresh token.

    An explicit Cookie header suppresses the jar (http.cookiejar leaves a header
    the caller already set alone), so a replay cannot accidentally also carry the
    jar's current token and pass for the wrong reason.
    """
    return await client.post("/api/auth/refresh", headers={"Cookie": f"{REFRESH_COOKIE}={raw}"})


async def _row_for(raw: str) -> RefreshToken:
    async with SessionLocal() as db:
        stmt = select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(raw))
        return (await db.execute(stmt)).scalar_one()


async def _age_revocation(raw: str, seconds: int) -> None:
    """Backdate a revocation so the replay lands outside the grace window without
    the test sleeping through it."""
    async with SessionLocal() as db:
        stmt = select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(raw))
        row = (await db.execute(stmt)).scalar_one()
        row.revoked_at = row.revoked_at - dt.timedelta(seconds=seconds)
        await db.commit()


async def _live_tokens(user_id, client: str | None = None) -> list[RefreshToken]:
    async with SessionLocal() as db:
        stmt = select(RefreshToken).where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        if client is not None:
            stmt = stmt.where(RefreshToken.client == client)
        return list((await db.execute(stmt)).scalars().all())


# ---------------------------------------------------------------------------
# Normal rotation
# ---------------------------------------------------------------------------


async def test_rotation_revokes_predecessor_and_records_successor(client):
    user = await make_user("rot1@x.com")
    await login(client, "rot1@x.com")
    first = client.cookies.get(REFRESH_COOKIE)

    assert (await client.post("/api/auth/refresh")).status_code == 200
    second = client.cookies.get(REFRESH_COOKIE)
    assert second != first

    old, new = await _row_for(first), await _row_for(second)
    assert old.revoked_at is not None
    assert old.replaced_by_id == new.id
    # The live row is the end of the chain, not a fork off it.
    assert new.revoked_at is None and new.replaced_by_id is None
    assert [t.id for t in await _live_tokens(user.id)] == [new.id]


# ---------------------------------------------------------------------------
# The undelivered-response case
# ---------------------------------------------------------------------------


async def test_replay_inside_grace_window_returns_a_working_pair(client):
    """The defect this exists for: the client never received the rotation, so its
    jar still holds the spent token."""
    user = await make_user("rot2@x.com")
    await login(client, "rot2@x.com")
    lost = client.cookies.get(REFRESH_COOKIE)

    assert (await client.post("/api/auth/refresh")).status_code == 200
    undelivered = client.cookies.get(REFRESH_COOKIE)

    resp = await _refresh_with(client, lost)
    assert resp.status_code == 200
    assert resp.json()["user"]["email"] == "rot2@x.com"

    # A brand-new pair chained from the successor, which is itself now spent.
    replacement = client.cookies.get(REFRESH_COOKIE)
    assert replacement not in (lost, undelivered)
    successor = await _row_for(undelivered)
    assert successor.revoked_at is not None
    assert successor.replaced_by_id == (await _row_for(replacement)).id
    assert [t.id for t in await _live_tokens(user.id)] == [(await _row_for(replacement)).id]

    # And the pair actually works — access cookie and all.
    assert (await client.get("/api/auth/me")).status_code == 200
    assert (await client.post("/api/auth/refresh")).status_code == 200


async def test_replay_inside_grace_is_still_gated_on_the_mobile_grant(client):
    """The grace path must not become a way around the re-checks: it substitutes
    which row is rotated, nothing else."""
    user = await make_user("rot3@x.com", mobile_access=True)
    await client.post(
        "/api/auth/login", json={"email": "rot3@x.com", "password": "TestPass1234", "client": "mobile"}
    )
    lost = client.cookies.get(REFRESH_COOKIE)
    assert (await client.post("/api/auth/refresh")).status_code == 200

    async with SessionLocal() as db:
        (await db.get(User, user.id)).mobile_access = False
        await db.commit()

    resp = await _refresh_with(client, lost)
    assert resp.status_code == 403
    assert "Mobile access has not been enabled" in resp.json()["detail"]
    assert resp.json()["code"] == MOBILE_NOT_APPROVED_CODE
    assert await _live_tokens(user.id) == []


# ---------------------------------------------------------------------------
# Reuse detection
# ---------------------------------------------------------------------------


async def test_replay_after_successor_was_used_is_theft(client):
    """Inside the window, but the successor has been spent — so the client did get
    it, and two parties are now presenting tokens from one chain."""
    user = await make_user("rot4@x.com")
    await login(client, "rot4@x.com")
    stolen = client.cookies.get(REFRESH_COOKIE)

    assert (await client.post("/api/auth/refresh")).status_code == 200
    assert (await client.post("/api/auth/refresh")).status_code == 200

    resp = await _refresh_with(client, stolen)
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid refresh token"
    assert await _live_tokens(user.id) == []
    # The live token the legitimate client was holding is gone too — that is the
    # point of family revocation, and why it is scoped per client.
    assert (await client.post("/api/auth/refresh")).status_code == 401


async def test_replay_outside_grace_window_is_theft(client):
    user = await make_user("rot5@x.com")
    await login(client, "rot5@x.com")
    stolen = client.cookies.get(REFRESH_COOKIE)
    assert (await client.post("/api/auth/refresh")).status_code == 200
    await _age_revocation(stolen, 600)

    resp = await _refresh_with(client, stolen)
    assert resp.status_code == 401
    assert await _live_tokens(user.id) == []


async def test_unknown_and_pre_migration_tokens_are_still_rejected(client):
    """A token with no successor recorded (revoked by logout, or written before the
    column existed) has nothing to fall back on and must fail closed."""
    user = await make_user("rot6@x.com")
    await login(client, "rot6@x.com")
    logged_out = client.cookies.get(REFRESH_COOKIE)
    assert (await client.post("/api/auth/logout")).status_code == 200
    assert (await _row_for(logged_out)).replaced_by_id is None

    assert (await _refresh_with(client, logged_out)).status_code == 401
    assert (await _refresh_with(client, "not-a-real-token")).status_code == 401
    assert await _live_tokens(user.id) == []


async def test_a_device_signed_out_by_a_password_change_does_not_sign_out_the_one_that_changed_it(
    client, caplog
):
    """Rule 6. change-password revokes every OTHER session, so device B's next
    refresh presents a revoked row with no successor. That used to land in the theft
    branch, which burned the user's live web sessions — A's included, the one that
    had just changed the password — and logged a theft warning for an account
    nobody had attacked."""
    user = await make_user("rot10@x.com")
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as a,
        AsyncClient(transport=transport, base_url="http://test") as b,
    ):
        a.headers.update(CSRF)
        b.headers.update(CSRF)
        await login(a, "rot10@x.com")
        await login(b, "rot10@x.com")
        resp = await a.post(
            "/api/auth/change-password",
            json={"current_password": "TestPass1234", "new_password": "BrandNewPass99"},
        )
        assert resp.status_code == 200, resp.text

        with caplog.at_level(logging.INFO, logger="app.api.auth"):
            resp = await b.post("/api/auth/refresh")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid refresh token"
        assert "Refresh token reuse" not in caplog.text
        assert "never-rotated" in caplog.text

        assert (await a.post("/api/auth/refresh")).status_code == 200
        assert len(await _live_tokens(user.id)) == 1


async def test_family_revocation_spares_the_other_client(client):
    """One user, a live web session and a live phone session. Reuse on the phone
    chain must not sign the browser out."""
    user = await make_user("rot7@x.com", mobile_access=True)
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as web,
        AsyncClient(transport=transport, base_url="http://test") as phone,
    ):
        web.headers.update(CSRF)
        phone.headers.update(CSRF)
        await login(web, "rot7@x.com")
        assert (
            await phone.post(
                "/api/auth/login",
                json={"email": "rot7@x.com", "password": "TestPass1234", "client": "mobile"},
            )
        ).status_code == 200

        stolen = phone.cookies.get(REFRESH_COOKIE)
        assert (await phone.post("/api/auth/refresh")).status_code == 200
        await _age_revocation(stolen, 600)

        assert (await _refresh_with(phone, stolen)).status_code == 401
        assert await _live_tokens(user.id, "mobile") == []

        web_live = await _live_tokens(user.id, "web")
        assert len(web_live) == 1
        assert (await web.post("/api/auth/refresh")).status_code == 200
        assert (await web.get("/api/auth/me")).status_code == 200


# ---------------------------------------------------------------------------
# The limiter must not be able to end a session
# ---------------------------------------------------------------------------


_REFUSED = "Error 111 connecting to redis:6379. Connection refused."


class _DeadPipeline:
    """Queueing is local, so only execute() can fail — same as redis-py.

    This stub used to be missing entirely, and that mattered: the limiter moved
    from two round-trips to one MULTI/EXEC in 6dcc4db, so `incr`/`expire` on the
    client below stopped being called and `pipeline()` started being called
    instead. The stub was never updated, so these tests raised AttributeError
    rather than a connection failure — and passed anyway, because
    degrade_on_outage caught bare Exception. They certified fail-open while
    exercising a hole in the guard instead of the outage path. Now that the guard
    only catches Redis and socket errors, the stub has to be honest.
    """

    def incr(self, *_args, **_kwargs):
        return self

    def expire(self, *_args, **_kwargs):
        return self

    async def execute(self):
        raise RedisConnectionError(_REFUSED)


# redis-py's ConnectionError rather than the builtin, for the reason recorded in
# test_presence_outage.py: the builtin is an OSError and would exercise the wrong
# arm of degrade_on_outage, which is what rate_limit.py runs its Redis work in.
class _DeadRedis:
    """Every call raises, the way a client with no reachable server does."""

    def pipeline(self, *_args, **_kwargs):
        return _DeadPipeline()

    async def incr(self, *_args, **_kwargs):
        raise RedisConnectionError(_REFUSED)

    async def expire(self, *_args, **_kwargs):
        raise RedisConnectionError(_REFUSED)

    async def ttl(self, *_args, **_kwargs):
        raise RedisConnectionError(_REFUSED)


@pytest.fixture
def dead_limiter_redis(monkeypatch):
    """Break Redis for the limiter only. Presence and the event bus keep the real
    client, so the test isolates the limiter's own failure mode."""
    monkeypatch.setattr(rate_limit, "get_redis", lambda: _DeadRedis())


async def test_refresh_survives_a_redis_outage_on_the_limiter(client, dead_limiter_redis):
    """A 500 here is a signed-out user on both clients, so the limiter fails open."""
    await make_user("rot8@x.com")
    await login(client, "rot8@x.com")

    resp = await client.post("/api/auth/refresh")
    assert resp.status_code == 200, resp.text
    assert resp.json()["user"]["email"] == "rot8@x.com"
    assert (await client.get("/api/auth/me")).status_code == 200


async def test_login_survives_a_redis_outage_on_the_limiter(client, dead_limiter_redis):
    await make_user("rot9@x.com")
    resp = await client.post("/api/auth/login", json={"email": "rot9@x.com", "password": "TestPass1234"})
    assert resp.status_code == 200, resp.text
