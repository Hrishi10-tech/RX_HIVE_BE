"""must_change_password is enforced (batch 73).

Both admin resets set users.must_change_password and show the admin the temporary
password, change-password cleared it, and nothing else read it. So a reset account
stayed fully usable with a password somebody else had been shown, for as long as
nobody happened to change it.

The contract these tests pin, shared with the web and iOS clients:

  * login, refresh and me answer 200 and carry user.must_change_password;
  * every other authenticated route answers 403 PASSWORD_CHANGE_REQUIRED, except
    logout, change-password and DELETE /api/notifications/subscribe;
  * the refusals that existed before keep their 401 / mobile 403, ahead of it;
  * the websocket refuses with 4403 (never 4001, which both clients answer with a
    refresh that succeeds here and would loop);
  * change-password refuses an unchanged password and re-issues a session the
    reset revoked (never one a later logout revoked); both resets delete push
    subscriptions; an org admin cannot reset themselves;
  * the reset ends every session from before it (users.sessions_valid_after): an
    older access token is refused with 401 on every gated route and at the socket,
    and stays refused after the owner changes the password (batch 73 review);
  * the socket decides that from the token's own claims, by the rule HTTP uses,
    and re-checks the account before every frame that acts — every frame but a
    ping or a hang-up — closing with the same codes as its clock revalidation;
  * a stale device's refresh after a reset or a password change is refused without
    burning the session the user has just made; a refresh row created before the
    reset is refused; a login or change-password racing a reset is refused, and a
    reset waits for a sign-in that holds the account and then ends its session; a
    pre-reset device that signs out is never re-issued a session afterwards.

The route walk at the bottom is the guard against the next route: it fails the
moment one is added that skips the gate, or that picks up the un-gated twin.
"""

import asyncio
import contextlib
import datetime as dt
import json
import time
import uuid
from types import SimpleNamespace

import pytest
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from starlette.websockets import WebSocketDisconnect

from app.api import auth as api_auth
from app.api.auth import MOBILE_NOT_APPROVED_CODE, lock_user_row
from app.core import deps
from app.core.deps import (
    PASSWORD_CHANGE_REQUIRED_CODE,
    PASSWORD_CHANGE_REQUIRED_DETAIL,
    PasswordChangeRequiredError,
    get_current_user_ws,
)
from app.core.security import (
    ACCESS_COOKIE,
    REFRESH_COOKIE,
    create_access_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from app.db.models import Department, PushSubscription, RefreshToken, User, UserRole
from app.db.session import SessionLocal
from app.main import app
from app.realtime import hub
from app.services import messaging
from app.utils import now_utc
from tests.conftest import CSRF, login, make_org, make_user

REFUSAL = {"detail": PASSWORD_CHANGE_REQUIRED_DETAIL, "code": PASSWORD_CHANGE_REQUIRED_CODE}
TEMP = "TempPass1234"
NEW = "BrandNewPass99"

# The only APIRoutes that authenticate nobody. Every other route must reach
# get_current_user, which is where the gate lives.
PUBLIC_ROUTES = {
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/refresh"),
    ("POST", "/api/livekit/webhook"),
    ("GET", "/api/health"),
}
# The only routes a reset account may still reach, through the un-gated twin.
ALLOWED_WHILE_PENDING = {
    ("POST", "/api/auth/logout"),
    ("GET", "/api/auth/me"),
    ("POST", "/api/auth/change-password"),
    ("DELETE", "/api/notifications/subscribe"),
}


@contextlib.asynccontextmanager
async def _as(email: str, password: str = "TestPass1234", *, client: str = "web"):
    """Yield a client signed in as `email` on the given client type, asserting the login succeeded."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c.headers.update(CSRF)
        resp = await c.post("/api/auth/login", json={"email": email, "password": password, "client": client})
        assert resp.status_code == 200, resp.text
        yield c


async def _row(user_id) -> User:
    """Read a user's row fresh, in a session of its own."""
    async with SessionLocal() as db:
        return await db.get(User, user_id)


async def _set(user_id, **fields) -> None:
    """Write the given fields onto a user's row directly and commit, bypassing the API."""
    async with SessionLocal() as db:
        row = await db.get(User, user_id)
        for name, value in fields.items():
            setattr(row, name, value)
        await db.commit()


async def _subscribe(user_id, endpoint: str) -> None:
    """Insert a push subscription for the user at `endpoint`."""
    async with SessionLocal() as db:
        db.add(PushSubscription(user_id=user_id, endpoint=endpoint, keys={"p256dh": "x", "auth": "y"}))
        await db.commit()


async def _endpoints_of(user_id) -> set[str]:
    """The push endpoints currently subscribed for the user."""
    async with SessionLocal() as db:
        rows = await db.execute(select(PushSubscription.endpoint).where(PushSubscription.user_id == user_id))
        return set(rows.scalars().all())


async def _sessions_of(user_id) -> list[RefreshToken]:
    """Every refresh-token row the user has, live or revoked."""
    async with SessionLocal() as db:
        rows = await db.execute(select(RefreshToken).where(RefreshToken.user_id == user_id))
        return list(rows.scalars().all())


async def _org_member(org_name: str, email: str, **kwargs) -> tuple:
    """Create an org with one department and a user in it, and return (org, dept, user).

    Extra keyword arguments go to make_user.
    """
    org = await make_org(org_name)
    async with SessionLocal() as db:
        dept = Department(org_id=org.id, name="Ward")
        db.add(dept)
        await db.commit()
        await db.refresh(dept)
    user = await make_user(email, org_id=org.id, dept_id=dept.id, **kwargs)
    return org, dept, user


async def _until_blocked_on_a_lock(query_like: str) -> None:
    """Wait until a statement matching `query_like` is waiting on a lock in Postgres.

    Proof that the concurrent request under test has reached the lock, rather than a
    sleep that hopes it has: the races below are only races once one side is
    actually waiting on the other."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        async with SessionLocal() as db:
            waiting = (
                await db.execute(
                    text(
                        "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
                        "AND datname = current_database() AND query ILIKE :q"
                    ),
                    {"q": query_like},
                )
            ).scalar_one()
        if waiting:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"no statement like {query_like!r} ever waited on a lock")


def _assert_refused(resp) -> None:
    """Assert the response is the 403 PASSWORD_CHANGE_REQUIRED refusal, body and all."""
    assert resp.status_code == 403, resp.text
    assert resp.json() == REFUSAL


def test_the_refusal_is_pinned_verbatim():
    """REFUSAL above is built from the constants, so it cannot notice them changing.
    The web and iOS clients match the code literally; the detail is the agreed text."""
    assert PASSWORD_CHANGE_REQUIRED_CODE == "PASSWORD_CHANGE_REQUIRED"
    assert PASSWORD_CHANGE_REQUIRED_DETAIL == "You must change your password before continuing."


# ---------------------------------------------------------------------------
# What a reset account can still reach
# ---------------------------------------------------------------------------


async def test_login_refresh_and_me_answer_200_and_carry_the_flag(client):
    """A flagged account can log in, refresh and read /me, each carrying the flag inside user."""
    org, _, _ = await _org_member("Flag Co", "flag@x.com", password=TEMP, must_change_password=True)

    resp = await client.post("/api/auth/login", json={"email": "flag@x.com", "password": TEMP})
    assert resp.status_code == 200, resp.text
    assert set(resp.json()) == {"user"}, "the flag belongs inside user, never at the top level"
    assert resp.json()["user"]["must_change_password"] is True

    resp = await client.post("/api/auth/refresh")
    assert resp.status_code == 200, resp.text
    assert resp.json()["user"]["must_change_password"] is True

    resp = await client.get("/api/auth/me")
    assert resp.status_code == 200, resp.text
    me = resp.json()
    assert me["must_change_password"] is True
    # The full payload, not a cut-down one: the client renders this screen with it.
    assert me["org_name"] == org.name and "avatar_url" in me and "mobile_access" in me


async def test_the_flag_is_false_for_an_ordinary_account_of_every_role(client):
    """/me reports must_change_password false for an unflagged member and superadmin."""
    org = await make_org("Plain Co")
    await make_user("plain@x.com", org_id=org.id)
    await make_user("root@x.com", role=UserRole.superadmin)
    for email in ("plain@x.com", "root@x.com"):
        async with _as(email) as c:
            me = (await c.get("/api/auth/me")).json()
            assert me["must_change_password"] is False, email


async def test_logout_and_push_unsubscribe_stay_reachable(client):
    """A flagged account can still delete its push subscription and log out, revoking its sessions."""
    _, _, user = await _org_member("Leaving Co", "leaving@x.com", password=TEMP, must_change_password=True)
    endpoint = "https://push.example.test/fcm/send/leaving"
    await _subscribe(user.id, endpoint)
    await login(client, "leaving@x.com", TEMP)

    resp = await client.request("DELETE", "/api/notifications/subscribe", json={"endpoint": endpoint})
    assert resp.status_code == 200, resp.text
    assert await _endpoints_of(user.id) == set()

    resp = await client.post("/api/auth/logout")
    assert resp.status_code == 200, resp.text
    assert all(t.revoked_at is not None for t in await _sessions_of(user.id))


# ---------------------------------------------------------------------------
# What it cannot: one representative per dependency family
# ---------------------------------------------------------------------------


async def test_get_current_user_routes_refuse_with_the_code(client, monkeypatch):
    """Routes behind get_current_user, POST push subscribe included, answer a flagged account with the 403."""
    await _org_member("Gate Co", "gate@x.com", password=TEMP, must_change_password=True)
    monkeypatch.setattr("app.services.push.validate_push_endpoint", lambda _endpoint: True)
    await login(client, "gate@x.com", TEMP)

    _assert_refused(await client.get("/api/conversations"))
    # POST stays gated while DELETE does not: adding a push channel is exactly what
    # a reset account must not be able to do.
    _assert_refused(
        await client.post(
            "/api/notifications/subscribe",
            json={"endpoint": "https://push.example.test/fcm/send/gate", "keys": {}},
        )
    )


async def test_get_tenant_routes_refuse_with_the_code(client):
    """Tenant-scoped routes answer a flagged account with the 403 refusal."""
    await _org_member("Tenant Co", "tenant@x.com", password=TEMP, must_change_password=True)
    await login(client, "tenant@x.com", TEMP)
    _assert_refused(await client.get("/api/users/contacts"))
    _assert_refused(await client.get("/api/search", params={"q": "anything"}))


async def test_org_admin_routes_refuse_with_the_code(client):
    """Org-admin routes answer a flagged org admin with the 403 refusal."""
    await _org_member(
        "Admin Gate Co", "oa@x.com", role=UserRole.org_admin, password=TEMP, must_change_password=True
    )
    await login(client, "oa@x.com", TEMP)
    _assert_refused(await client.get("/api/org-admin/users"))


async def test_superadmin_routes_refuse_with_the_code(client):
    """Superadmin routes that served a superadmin answer the 403 refusal once its row is flagged."""
    root = await make_user("root@x.com", role=UserRole.superadmin)
    await login(client, "root@x.com")
    assert (await client.get("/api/admin/stats")).status_code == 200

    # Flagged on the row directly: nothing in the API resets a superadmin, and the
    # gate must not depend on how the flag got there.
    await _set(root.id, must_change_password=True)
    _assert_refused(await client.get("/api/admin/stats"))
    _assert_refused(await client.get("/api/metrics"))


# ---------------------------------------------------------------------------
# Refusals that existed before keep their answer
# ---------------------------------------------------------------------------


async def test_a_deactivated_flagged_account_gets_401_not_403(client):
    """A deactivated account that is also flagged gets the plain 401, on gated and un-gated routes."""
    _, _, user = await _org_member("Gone Co", "gone@x.com", password=TEMP, must_change_password=True)
    await login(client, "gone@x.com", TEMP)
    await _set(user.id, is_active=False)

    for resp in (await client.get("/api/conversations"), await client.get("/api/auth/me")):
        assert resp.status_code == 401, resp.text
        assert resp.json() == {"detail": "Not authenticated"}


async def test_flagged_mobile_login_without_the_grant_still_gets_the_mobile_403(client):
    """A flagged account without the mobile grant still gets MOBILE_NOT_APPROVED at mobile login."""
    await _org_member("No Mobile Co", "nomob@x.com", password=TEMP, must_change_password=True)
    resp = await client.post(
        "/api/auth/login", json={"email": "nomob@x.com", "password": TEMP, "client": "mobile"}
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == MOBILE_NOT_APPROVED_CODE


async def test_a_revoked_mobile_grant_still_answers_401_ahead_of_the_flag(client):
    """Revoking a flagged account's mobile grant makes its phone session answer 401, not the 403."""
    _, _, user = await _org_member(
        "Revoked Co", "rev@x.com", password=TEMP, must_change_password=True, mobile_access=True
    )
    async with _as("rev@x.com", TEMP, client="mobile") as phone:
        assert (await phone.get("/api/auth/me")).status_code == 200
        await _set(user.id, mobile_access=False)
        resp = await phone.get("/api/conversations")
        assert resp.status_code == 401, resp.text


# ---------------------------------------------------------------------------
# Getting out: change-password
# ---------------------------------------------------------------------------


async def test_change_password_clears_the_flag_and_the_same_session_carries_on(client):
    """Changing the password clears the flag and unlocks gated routes on the same, unreissued session."""
    _, _, user = await _org_member("Change Co", "change@x.com", password=TEMP, must_change_password=True)
    await login(client, "change@x.com", TEMP)
    refresh_before = client.cookies.get(REFRESH_COOKIE)
    _assert_refused(await client.get("/api/conversations"))

    resp = await client.post(
        "/api/auth/change-password", json={"current_password": TEMP, "new_password": NEW}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"message": "Password changed"}
    # A live session is kept, not replaced: re-issuing it would be harmless here
    # but is the branch reserved for a session the reset revoked.
    assert "set-cookie" not in resp.headers

    assert (await _row(user.id)).must_change_password is False
    assert (await client.get("/api/conversations")).status_code == 200
    assert (await client.get("/api/auth/me")).json()["must_change_password"] is False
    assert client.cookies.get(REFRESH_COOKIE) == refresh_before
    assert (await client.post("/api/auth/refresh")).status_code == 200


async def test_an_unchanged_password_is_refused_and_the_flag_stays(client):
    """Changing to the same password is refused with 400 and leaves the hash and the flag unchanged."""
    _, _, user = await _org_member("Same Co", "same@x.com", password=TEMP, must_change_password=True)
    await login(client, "same@x.com", TEMP)
    hash_before = (await _row(user.id)).password_hash

    resp = await client.post(
        "/api/auth/change-password", json={"current_password": TEMP, "new_password": TEMP}
    )
    assert resp.status_code == 400, resp.text
    assert resp.json() == {"detail": "Choose a password different from your current one."}

    row = await _row(user.id)
    assert row.must_change_password is True
    assert row.password_hash == hash_before
    _assert_refused(await client.get("/api/conversations"))


async def test_an_unchanged_password_is_refused_for_an_ordinary_account_too(client):
    """For every account, not only reset ones: the rule is about the password."""
    await make_user("ordinary@x.com")
    await login(client, "ordinary@x.com")
    resp = await client.post(
        "/api/auth/change-password",
        json={"current_password": "TestPass1234", "new_password": "TestPass1234"},
    )
    assert resp.status_code == 400, resp.text
    assert resp.json() == {"detail": "Choose a password different from your current one."}


async def test_a_wrong_temporary_password_is_401_and_the_flag_stays(client):
    """A wrong current password is 401 even when the new password repeats it, and the flag stays set."""
    _, _, user = await _org_member("Wrong Co", "wrong@x.com", password=TEMP, must_change_password=True)
    await login(client, "wrong@x.com", TEMP)

    resp = await client.post(
        "/api/auth/change-password", json={"current_password": "NotTheTemp99", "new_password": NEW}
    )
    assert resp.status_code == 401, resp.text
    assert resp.json() == {"detail": "Current password is incorrect"}

    # The unchanged-password rule runs only once the current password has verified,
    # so a wrong guess is a wrong guess whatever it is paired with.
    resp = await client.post(
        "/api/auth/change-password",
        json={"current_password": "NotTheTemp99", "new_password": "NotTheTemp99"},
    )
    assert resp.status_code == 401, resp.text
    assert (await _row(user.id)).must_change_password is True


@pytest.mark.parametrize("session_client", ["web", "mobile"])
async def test_a_session_from_before_the_reset_can_finish_the_change(client, session_client):
    """The reset revokes every refresh token, so a client signed in before it is
    finishing the change on its access cookie alone. Without a fresh session it
    would be signed out when that cookie lapsed, minutes after doing as asked."""
    org, dept, user = await _org_member("Before Co", "before@x.com", mobile_access=True)
    await make_user("root@x.com", role=UserRole.superadmin)

    async with _as("before@x.com", client=session_client) as victim:
        refresh_before = victim.cookies.get(REFRESH_COOKIE)
        async with _as("root@x.com") as root:
            resp = await root.post(f"/api/admin/users/{user.id}/reset-password")
            assert resp.status_code == 200, resp.text
            temp = resp.json()["temporary_password"]
        assert all(t.revoked_at is not None for t in await _sessions_of(user.id))

        # Still signed in on the access cookie, and told what to do.
        assert (await victim.get("/api/auth/me")).json()["must_change_password"] is True

        resp = await victim.post(
            "/api/auth/change-password", json={"current_password": temp, "new_password": NEW}
        )
        assert resp.status_code == 200, resp.text
        set_cookie = " ".join(resp.headers.get_list("set-cookie"))
        assert ACCESS_COOKIE in set_cookie and REFRESH_COOKIE in set_cookie
        assert victim.cookies.get(REFRESH_COOKIE) != refresh_before

        live = [t for t in await _sessions_of(user.id) if t.revoked_at is None]
        assert len(live) == 1
        # Re-issued for the client the access token was minted for, so a phone
        # keeps a phone session and its mobile-grant checks keep applying.
        assert live[0].client == session_client

        resp = await victim.post("/api/auth/refresh")
        assert resp.status_code == 200, resp.text
        assert resp.json()["user"]["must_change_password"] is False
        assert (await victim.get("/api/conversations")).status_code == 200


# ---------------------------------------------------------------------------
# The resets
# ---------------------------------------------------------------------------


async def test_superadmin_reset_ends_the_targets_live_session_and_drops_its_push(client):
    """A superadmin reset ends the target's session (401), flags it, and deletes its push, nobody else's."""
    org, dept, target = await _org_member("Reset Co", "target@x.com")
    bystander = await make_user("bystander@x.com", org_id=org.id, dept_id=dept.id)
    await make_user("root@x.com", role=UserRole.superadmin)
    await _subscribe(target.id, "https://push.example.test/fcm/send/target")
    await _subscribe(bystander.id, "https://push.example.test/fcm/send/bystander")

    async with _as("target@x.com") as victim:
        assert (await victim.get("/api/conversations")).status_code == 200
        async with _as("root@x.com") as root:
            resp = await root.post(f"/api/admin/users/{target.id}/reset-password")
            assert resp.status_code == 200, resp.text
        # The SAME access token, minted before the reset, is refused at once — not
        # at its expiry fifteen minutes later — and as a session that has ENDED
        # (401), not one waiting on a password change: the reset ended it.
        assert (await victim.get("/api/conversations")).status_code == 401

    assert (await _row(target.id)).must_change_password is True
    assert await _endpoints_of(target.id) == set()
    assert await _endpoints_of(bystander.id) == {"https://push.example.test/fcm/send/bystander"}


async def test_org_admin_reset_sets_the_flag_ends_the_targets_session_and_drops_its_push(client):
    """An org-admin reset flags the target, stamps its epoch, ends its session and deletes its push."""
    org, dept, target = await _org_member("Org Reset Co", "member@x.com")
    await make_user("oa@x.com", org_id=org.id, dept_id=dept.id, role=UserRole.org_admin)
    await _subscribe(target.id, "https://push.example.test/fcm/send/member")

    async with _as("member@x.com") as victim:
        assert (await victim.get("/api/conversations")).status_code == 200
        async with _as("oa@x.com") as admin:
            resp = await admin.post(f"/api/org-admin/users/{target.id}/reset-password")
            assert resp.status_code == 200, resp.text
        assert (await victim.get("/api/conversations")).status_code == 401
        assert (await _row(target.id)).sessions_valid_after is not None

    assert (await _row(target.id)).must_change_password is True
    assert await _endpoints_of(target.id) == set()


async def test_a_token_from_before_the_reset_stays_dead_after_the_owner_changes_the_password(client):
    """The batch 73 review's attack. Someone holding the account's access cookie from
    before the reset was refused only while the flag was set; once the owner chose
    a new password the same cookie worked again for the rest of its life — long
    enough to register a push subscription, which never expires."""
    org, dept, user = await _org_member("Epoch Co", "epoch@x.com")
    await make_user("root@x.com", role=UserRole.superadmin)

    async with _as("epoch@x.com") as stale:
        async with _as("root@x.com") as root:
            resp = await root.post(f"/api/admin/users/{user.id}/reset-password")
            assert resp.status_code == 200, resp.text
            temp = resp.json()["temporary_password"]
        async with _as("epoch@x.com", temp) as owner:
            resp = await owner.post(
                "/api/auth/change-password", json={"current_password": temp, "new_password": NEW}
            )
            assert resp.status_code == 200, resp.text
            assert (await owner.get("/api/conversations")).status_code == 200  # control

        assert (await _row(user.id)).must_change_password is False
        assert (await stale.get("/api/conversations")).status_code == 401
        # Nor the un-gated routes, now that nothing is pending: their exemption is
        # for finishing the change, and /me would otherwise go on handing the
        # full profile to a token copied before the reset (CodeRabbit, 1fbc1de).
        assert (await stale.get("/api/auth/me")).status_code == 401
        resp = await stale.post(
            "/api/notifications/subscribe",
            json={
                "endpoint": "https://push.example.test/fcm/send/stale",
                "keys": {"p256dh": "x", "auth": "y"},
            },
        )
        assert resp.status_code == 401, resp.text
        assert (await stale.post("/api/auth/refresh")).status_code == 401
    assert await _endpoints_of(user.id) == set()


async def test_change_password_does_not_reissue_a_session_a_logout_ended(client):
    """Sign-out pressed while the change was in flight: logout revoked the caller's
    row first, and finding no live row the route used to hand a fresh 30-day
    session to the device that had just signed out (batch 73 review)."""
    org, dept, user = await _org_member("Race Co", "race@x.com")
    await make_user("root@x.com", role=UserRole.superadmin)
    async with _as("root@x.com") as root:
        resp = await root.post(f"/api/admin/users/{user.id}/reset-password")
        temp = resp.json()["temporary_password"]

    async with _as("race@x.com", temp) as device:
        # What logout does to this device's row, landing after the reset.
        async with SessionLocal() as db:
            for row in (
                await db.execute(select(RefreshToken).where(RefreshToken.user_id == user.id))
            ).scalars():
                if row.revoked_at is None:
                    row.revoked_at = now_utc()
            await db.commit()
        resp = await device.post(
            "/api/auth/change-password", json={"current_password": temp, "new_password": NEW}
        )
        assert resp.status_code == 200, resp.text
        set_cookie = " ".join(resp.headers.get_list("set-cookie"))
        assert REFRESH_COOKIE not in set_cookie and ACCESS_COOKIE not in set_cookie
    assert [t for t in await _sessions_of(user.id) if t.revoked_at is None] == []


async def test_change_password_does_not_reissue_for_an_account_no_reset_touched(client):
    """Change-password re-issues no session for a revoked refresh row on an account never reset."""
    await make_user("plain@x.com")
    async with _as("plain@x.com") as device:
        async with SessionLocal() as db:
            for row in (await db.execute(select(RefreshToken))).scalars():
                row.revoked_at = now_utc()
            await db.commit()
        resp = await device.post(
            "/api/auth/change-password", json={"current_password": "TestPass1234", "new_password": NEW}
        )
        assert resp.status_code == 200, resp.text
        assert REFRESH_COOKIE not in " ".join(resp.headers.get_list("set-cookie"))


async def test_a_pre_reset_device_that_signs_out_is_not_reissued_a_session_afterwards(client):
    """The batch 73 review's R6. The reset had already revoked this device's refresh
    row, so logout's revocation changed nothing, the row still read as "ended by the
    reset", and a change-password already in flight re-issued a 30-day session into
    the device the user had just signed out of."""
    org, dept, user = await _org_member("Signout Co", "signout@x.com")
    await make_user("root@x.com", role=UserRole.superadmin)

    async with _as("signout@x.com") as device:
        async with _as("root@x.com") as root:
            resp = await root.post(f"/api/admin/users/{user.id}/reset-password")
            temp = resp.json()["temporary_password"]
        # What the in-flight change-password carries: the cookies from before sign-out.
        before_signout = (
            f"{ACCESS_COOKIE}={device.cookies.get(ACCESS_COOKIE)}; "
            f"{REFRESH_COOKIE}={device.cookies.get(REFRESH_COOKIE)}"
        )
        assert (await device.post("/api/auth/logout")).status_code == 200

        resp = await device.post(
            "/api/auth/change-password",
            json={"current_password": temp, "new_password": NEW},
            headers={"Cookie": before_signout},
        )
        assert resp.status_code == 200, resp.text
        set_cookie = " ".join(resp.headers.get_list("set-cookie"))
        assert REFRESH_COOKIE not in set_cookie and ACCESS_COOKIE not in set_cookie
    assert [t for t in await _sessions_of(user.id) if t.revoked_at is None] == []


async def test_logout_and_change_password_at_the_same_moment_neither_deadlock_nor_reissue(
    client, monkeypatch
):
    """The same sign-out, truly concurrent. Logout used to lock the refresh row and
    write the users row only at commit, while change-password writes the users row
    and then locks the refresh row: opposite orders, so the two deadlocked and one of
    them failed. Both take the account's lock first now (lock_user_row), and the
    change, queued behind the logout, sees the row the logout re-stamped."""
    org, dept, user = await _org_member("Both Co", "both@x.com")
    await make_user("root@x.com", role=UserRole.superadmin)

    async with _as("both@x.com") as device:
        async with _as("root@x.com") as root:
            resp = await root.post(f"/api/admin/users/{user.id}/reset-password")
            temp = resp.json()["temporary_password"]
        cookies = {
            "Cookie": f"{ACCESS_COOKIE}={device.cookies.get(ACCESS_COOKIE)}; "
            f"{REFRESH_COOKIE}={device.cookies.get(REFRESH_COOKIE)}"
        }
        real_chain = api_auth._revoke_rotation_chain
        holding, release = asyncio.Event(), asyncio.Event()

        async def _revoke_then_wait(db, token):
            """Run the real chain revocation, then hold logout's transaction open until released."""
            revoked = await real_chain(db, token)
            holding.set()
            await release.wait()
            return revoked

        monkeypatch.setattr(api_auth, "_revoke_rotation_chain", _revoke_then_wait)
        logout = asyncio.create_task(device.post("/api/auth/logout", headers=cookies))
        await asyncio.wait_for(holding.wait(), timeout=5)
        change = asyncio.create_task(
            device.post(
                "/api/auth/change-password",
                json={"current_password": temp, "new_password": NEW},
                headers=cookies,
            )
        )
        await _until_blocked_on_a_lock("%")
        release.set()

        assert (await logout).status_code == 200
        resp = await change
        assert resp.status_code == 200, resp.text
        assert REFRESH_COOKIE not in " ".join(resp.headers.get_list("set-cookie"))
    assert [t for t in await _sessions_of(user.id) if t.revoked_at is None] == []


# ---------------------------------------------------------------------------
# Refresh after a reset: the stale device, the new session, and the race
# ---------------------------------------------------------------------------


async def test_a_stale_device_refreshing_after_a_reset_does_not_sign_out_the_new_session(client):
    """A revoked refresh row with no successor used to be read as a stolen cookie, and
    the theft branch burns every live session the user has on that client. So the
    stale device's refresh — which both clients now force, to learn whether their
    session survived — signed the owner out of the session they had just signed in
    to with the temporary password (batch 73 review)."""
    org, dept, user = await _org_member("Stale Co", "stale@x.com")
    await make_user("root@x.com", role=UserRole.superadmin)

    async with _as("stale@x.com") as stale:
        async with _as("root@x.com") as root:
            resp = await root.post(f"/api/admin/users/{user.id}/reset-password")
            temp = resp.json()["temporary_password"]
        async with _as("stale@x.com", temp) as owner:
            resp = await stale.post("/api/auth/refresh")
            assert resp.status_code == 401, resp.text
            assert resp.json() == {"detail": "Invalid refresh token"}

            resp = await owner.post("/api/auth/refresh")
            assert resp.status_code == 200, resp.text
            assert resp.json()["user"]["must_change_password"] is True
    assert len([t for t in await _sessions_of(user.id) if t.revoked_at is None]) == 1


async def test_a_stale_device_refreshing_does_not_sign_out_the_session_change_password_reissued(client):
    """A pre-reset device's refresh is refused without ending the session change-password re-issued."""
    org, dept, user = await _org_member("Reissue Co", "reissue@x.com")
    await make_user("root@x.com", role=UserRole.superadmin)

    async with _as("reissue@x.com") as finisher, _as("reissue@x.com") as stale:
        async with _as("root@x.com") as root:
            resp = await root.post(f"/api/admin/users/{user.id}/reset-password")
            temp = resp.json()["temporary_password"]
        resp = await finisher.post(
            "/api/auth/change-password", json={"current_password": temp, "new_password": NEW}
        )
        assert resp.status_code == 200, resp.text
        assert REFRESH_COOKIE in " ".join(resp.headers.get_list("set-cookie")), "control: re-issued"

        assert (await stale.post("/api/auth/refresh")).status_code == 401
        resp = await finisher.post("/api/auth/refresh")
        assert resp.status_code == 200, resp.text
        assert resp.json()["user"]["must_change_password"] is False


async def test_refresh_refuses_a_session_created_before_the_last_reset(client):
    """What a rotation racing a reset leaves behind, set up directly: a refresh row
    still live, created before the epoch the reset stamped."""
    _, _, user = await _org_member("Epoch Refresh Co", "eprefresh@x.com")
    await login(client, "eprefresh@x.com")
    await asyncio.sleep(0.005)
    await _set(user.id, sessions_valid_after=now_utc())

    resp = await client.post("/api/auth/refresh")
    assert resp.status_code == 401, resp.text
    assert resp.json() == {"detail": "Invalid refresh token"}
    # Revoked, not merely refused, so it cannot be retried.
    assert [t for t in await _sessions_of(user.id) if t.revoked_at is None] == []


async def test_refresh_rotates_a_session_created_after_the_last_reset(client):
    """A refresh row created after the session epoch still rotates."""
    _, _, user = await _org_member("Epoch After Co", "epafter@x.com")
    await _set(user.id, sessions_valid_after=now_utc())
    await asyncio.sleep(0.005)
    await login(client, "epafter@x.com")
    resp = await client.post("/api/auth/refresh")
    assert resp.status_code == 200, resp.text


async def test_a_rotation_racing_a_reset_does_not_leave_a_live_session(client, monkeypatch):
    """The race itself. The refresh holds its row lock while the reset runs; the
    reset's revocation waits on that row, and by the time it runs the successor the
    rotation inserted is invisible to it (READ COMMITTED re-checks only the row it
    waited on). The successor came out of the reset live, for 30 days."""
    org, dept, user = await _org_member("Rotation Race Co", "rotrace@x.com")
    await make_user("root@x.com", role=UserRole.superadmin)
    real_issue = api_auth._issue_session
    holding, release = asyncio.Event(), asyncio.Event()

    async def _issue_once_the_reset_is_waiting(*args, **kwargs):
        """Hold the first rotation's session issue until released, so the reset runs during the refresh."""
        if kwargs.get("replaces") is not None and not release.is_set():
            holding.set()
            await release.wait()
        return await real_issue(*args, **kwargs)

    monkeypatch.setattr(api_auth, "_issue_session", _issue_once_the_reset_is_waiting)

    async with _as("rotrace@x.com") as device, _as("root@x.com") as root:
        refresh = asyncio.create_task(device.post("/api/auth/refresh"))
        await asyncio.wait_for(holding.wait(), timeout=5)
        reset = asyncio.create_task(root.post(f"/api/admin/users/{user.id}/reset-password"))
        await _until_blocked_on_a_lock("%update refresh_tokens%")
        release.set()
        assert (await refresh).status_code == 200, "the rotation committed first"
        assert (await reset).status_code == 200

        live = [t for t in await _sessions_of(user.id) if t.revoked_at is None]
        assert len(live) == 1, "the precondition: the successor the reset could not see"

        resp = await device.post("/api/auth/refresh")
        assert resp.status_code == 401, resp.text
        assert [t for t in await _sessions_of(user.id) if t.revoked_at is None] == []
        # And the access token that rotation minted is from before the epoch too.
        assert (await device.get("/api/conversations")).status_code == 401


# ---------------------------------------------------------------------------
# Sign-in and change-password racing a reset
# ---------------------------------------------------------------------------


async def test_login_is_refused_when_the_password_changes_while_it_is_verified(client, monkeypatch):
    """A reset committing while login's bcrypt round ran revoked every refresh row
    that existed — not the one the login then inserted — so a sign-in with the
    password the reset had just replaced kept a live session (batch 73 review)."""
    _, _, user = await _org_member("Login Race Co", "loginrace@x.com")
    real_verify = api_auth.verify_password
    replaced = await hash_password("SomeoneElses99")

    async def _verify_while_a_reset_lands(password, hashed):
        """Verify the password and, if it matched, replace the stored hash as a reset landing then would."""
        ok = await real_verify(password, hashed)
        if ok:
            await _set(user.id, password_hash=replaced)
        return ok

    monkeypatch.setattr(api_auth, "verify_password", _verify_while_a_reset_lands)
    resp = await client.post("/api/auth/login", json={"email": "loginrace@x.com", "password": "TestPass1234"})
    assert resp.status_code == 401, resp.text
    assert resp.json() == {"detail": "Invalid email or password"}
    assert await _sessions_of(user.id) == []


async def test_change_password_is_refused_when_a_reset_lands_while_it_is_verified(client, monkeypatch):
    """The same race on change-password: it overwrote the reset's temporary password
    with one chosen by whoever held the session the reset was meant to end, cleared
    the flag, and — the reset having revoked that session's refresh row — re-issued
    them a fresh one."""
    _, _, user = await _org_member("Change Race Co", "changerace@x.com")
    await login(client, "changerace@x.com")
    real_verify = api_auth.verify_password
    temp_hash = await hash_password(TEMP)

    async def _verify_while_a_reset_lands(password, hashed):
        """Verify the password and, if it matched, apply a reset's temporary hash and flag at that moment."""
        ok = await real_verify(password, hashed)
        if ok:
            await _set(user.id, password_hash=temp_hash, must_change_password=True)
        return ok

    monkeypatch.setattr(api_auth, "verify_password", _verify_while_a_reset_lands)
    resp = await client.post(
        "/api/auth/change-password", json={"current_password": "TestPass1234", "new_password": NEW}
    )
    assert resp.status_code == 401, resp.text
    assert resp.json() == {"detail": "Current password is incorrect"}
    row = await _row(user.id)
    assert row.must_change_password is True
    assert await verify_password(TEMP, row.password_hash), "the reset's password must stand"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("org", (404, "User not found")),
        ("role", (403, "Only a superadmin can manage another admin's account")),
    ],
)
async def test_an_org_admin_reset_rechecks_its_scope_on_the_locked_row(client, monkeypatch, change, expected):
    """A superadmin moving the target out of the org, or promoting it, while the reset hashes its temporary
    password must not leave the org admin holding a password for an account outside their reach (CodeRabbit,
    review of 70495a4)."""
    from app.api import org_admin as org_admin_api

    org, dept, target = await _org_member("Scope Co", "scoped@x.com")
    await make_user("scope-oa@x.com", org_id=org.id, dept_id=dept.id, role=UserRole.org_admin)
    elsewhere = await make_org("Elsewhere Co")
    hash_before = (await _row(target.id)).password_hash
    real_hash = org_admin_api.hash_password

    async def _hash_while_the_target_moves(password):
        """Hash as the route does, while a superadmin changes the target in another session."""
        if change == "org":
            await _set(target.id, org_id=elsewhere.id, dept_id=None)
        else:
            await _set(target.id, role=UserRole.org_admin)
        return await real_hash(password)

    monkeypatch.setattr(org_admin_api, "hash_password", _hash_while_the_target_moves)
    async with _as("scope-oa@x.com") as admin:
        resp = await admin.post(f"/api/org-admin/users/{target.id}/reset-password")
    assert (resp.status_code, resp.json()["detail"]) == expected, resp.text
    assert "temporary_password" not in resp.json()
    row = await _row(target.id)
    assert row.password_hash == hash_before
    assert row.must_change_password is False


@pytest.mark.parametrize("route", ["admin", "org-admin"])
async def test_a_reset_waits_for_a_sign_in_holding_the_account_and_then_ends_its_session(client, route):
    """The other order of the login race: the sign-in locked the account first and is
    about to insert its session. The reset must wait for it and revoke what it
    produced, not revoke what existed before and leave the new session live."""
    org, dept, user = await _org_member("Lock Co", "lock@x.com")
    await make_user("root@x.com", role=UserRole.superadmin)
    await make_user("oa@x.com", org_id=org.id, dept_id=dept.id, role=UserRole.org_admin)
    admin_email = "root@x.com" if route == "admin" else "oa@x.com"

    async with _as(admin_email) as admin:
        async with SessionLocal() as signing_in:
            await lock_user_row(signing_in, user.id)
            reset = asyncio.create_task(admin.post(f"/api/{route}/users/{user.id}/reset-password"))
            await _until_blocked_on_a_lock("%users%")
            signing_in.add(
                RefreshToken(
                    user_id=user.id,
                    token_hash=hash_refresh_token(f"signing-in-{route}"),
                    client="web",
                    expires_at=now_utc() + dt.timedelta(days=30),
                    created_at=now_utc(),
                )
            )
            await signing_in.commit()
        resp = await reset
        assert resp.status_code == 200, resp.text

    sessions = await _sessions_of(user.id)
    assert sessions and all(t.revoked_at is not None for t in sessions)


def test_session_client_reads_the_claims_the_request_was_authenticated_with():
    """Never a second decode: change-password asks after two bcrypt rounds, by which
    time the token may have expired, and an expired token decodes to nothing — so a
    phone was re-issued a WEB session, outside every mobile-grant check."""

    def _request(claims, token=None):
        """A request stand-in with the given verified claims on its state and an optional access cookie."""
        cookies = {ACCESS_COOKIE: token} if token else {}
        return SimpleNamespace(
            state=SimpleNamespace(**({"auth_claims": claims} if claims else {})), cookies=cookies, headers={}
        )

    assert deps.session_client(_request({"client": "mobile"})) == "mobile"
    assert deps.session_client(_request(None)) == "web"
    # A perfectly decodable mobile token on the request changes nothing: only the
    # claims the dependency verified count.
    token = create_access_token(uuid.uuid4(), "member", None, client="mobile")
    assert deps.session_client(_request(None, token)) == "web"


async def test_an_org_admin_cannot_reset_their_own_password(client):
    """An org admin's reset of their own password is a 400 that changes nothing about the account."""
    org, dept, admin = await _org_member("Self Co", "self@x.com", role=UserRole.org_admin)
    await _subscribe(admin.id, "https://push.example.test/fcm/send/self")
    hash_before = (await _row(admin.id)).password_hash

    await login(client, "self@x.com")
    resp = await client.post(f"/api/org-admin/users/{admin.id}/reset-password")
    assert resp.status_code == 400, resp.text
    assert resp.json() == {"detail": "Use Change Password to change your own password."}

    row = await _row(admin.id)
    assert row.password_hash == hash_before
    assert row.must_change_password is False
    assert await _endpoints_of(admin.id) == {"https://push.example.test/fcm/send/self"}
    assert all(t.revoked_at is None for t in await _sessions_of(admin.id))
    assert (await client.get("/api/org-admin/users")).status_code == 200


async def test_created_accounts_are_not_flagged(client):
    """Only a reset flags an account. e2e seeding signs straight in as the users
    it creates, so a create path that flagged them would wall off every run."""
    org, dept, _ = await _org_member("Create Co", "oa@create.com", role=UserRole.org_admin)
    await make_user("root@x.com", role=UserRole.superadmin)

    async with _as("root@x.com") as root:
        resp = await root.post(
            "/api/admin/users",
            json={
                "org_id": str(org.id),
                "dept_id": str(dept.id),
                "email": "bysuper@create.com",
                "display_name": "By Super",
                "password": "GoodPass1234",
                "role": "member",
            },
        )
        assert resp.status_code == 200, resp.text
    async with _as("oa@create.com") as admin:
        resp = await admin.post(
            "/api/org-admin/users",
            json={
                "dept_id": str(dept.id),
                "email": "byadmin@create.com",
                "display_name": "By Admin",
                "password": "GoodPass1234",
            },
        )
        assert resp.status_code == 200, resp.text

    for email in ("bysuper@create.com", "byadmin@create.com"):
        async with _as(email, "GoodPass1234") as c:
            assert (await c.get("/api/auth/me")).json()["must_change_password"] is False
            assert (await c.get("/api/conversations")).status_code == 200


# ---------------------------------------------------------------------------
# Messaging: the gate a socket opened before the reset still meets
# ---------------------------------------------------------------------------


async def test_a_flagged_sender_cannot_send_or_mutate(client):
    """Messaging refuses a flagged sender's send and reaction with the coded 403, not a 404."""
    org, dept, a = await _org_member("Send Co", "a@send.com")
    b = await make_user("b@send.com", org_id=org.id, dept_id=dept.id)
    async with _as("a@send.com") as c:
        conv = (await c.post("/api/conversations/direct", json={"participant_id": str(b.id)})).json()["_id"]
        sent = await c.post(f"/api/conversations/{conv}/messages", json={"content": "before the reset"})
        assert sent.status_code == 200, sent.text
        message_id = sent.json()["_id"]

    await _set(a.id, must_change_password=True)

    # The hub re-loads the sender row per frame, so this is what it hands over.
    async with SessionLocal() as db:
        sender = await db.get(User, a.id)
        with pytest.raises(messaging.PasswordChangeRequired) as caught:
            await messaging.send_message(db, conversation_id=uuid.UUID(conv), sender=sender, content="after")
    assert caught.value.status_code == 403
    assert caught.value.code == PASSWORD_CHANGE_REQUIRED_CODE

    # The mutation paths collapse conversation refusals into 404 "Message not
    # found". This is an account refusal and must come through as one.
    async with SessionLocal() as db:
        actor = await db.get(User, a.id)
        with pytest.raises(messaging.PasswordChangeRequired):
            await messaging.toggle_reaction(db, message_id=uuid.UUID(message_id), actor=actor, emoji="👍")


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


class _HandshakeStub:
    """Exactly what get_current_user_ws reads off a WebSocket."""

    def __init__(self, token: str) -> None:
        """Carry `token` as the access cookie, with no query parameters."""
        self.cookies = {ACCESS_COOKIE: token}
        self.query_params: dict[str, str] = {}


def _token_for(user: User, client: str = "web") -> str:
    """Mint an access token for `user` as the API would for the given client."""
    return create_access_token(user.id, user.role.value, user.org_id, client=client)


async def test_ws_auth_tells_a_flagged_account_apart_from_a_failed_one(client):
    """get_current_user_ws raises for a flagged account and returns None for a failed one.

    A revoked mobile grant and a deactivated account still count as failed, flag or no flag.
    """
    _, _, user = await _org_member("WS Auth Co", "wsauth@x.com", mobile_access=True)

    async with SessionLocal() as db:
        auth = await get_current_user_ws(_HandshakeStub(_token_for(user)), db)
    assert auth is not None and auth[0].id == user.id  # control

    await _set(user.id, must_change_password=True)
    async with SessionLocal() as db:
        with pytest.raises(PasswordChangeRequiredError):
            await get_current_user_ws(_HandshakeStub(_token_for(user)), db)

    # A revoked mobile grant keeps its None (4001), flag or no flag.
    await _set(user.id, mobile_access=False)
    async with SessionLocal() as db:
        assert await get_current_user_ws(_HandshakeStub(_token_for(user, "mobile")), db) is None

    # So does a deactivated account.
    await _set(user.id, is_active=False)
    async with SessionLocal() as db:
        assert await get_current_user_ws(_HandshakeStub(_token_for(user)), db) is None


async def test_ws_handshake_refuses_a_token_from_before_the_reset(client):
    """The socket handshake treats a token issued before the session epoch as a failed authentication."""
    _, _, user = await _org_member("WS Epoch Co", "wsepoch@x.com")
    token = _token_for(user)
    # The epoch lands after the token's iat second, as a reset after sign-in does.
    await _set(user.id, sessions_valid_after=now_utc() + dt.timedelta(seconds=2))
    async with SessionLocal() as db:
        assert await get_current_user_ws(_HandshakeStub(token), db) is None


async def test_the_session_epoch_tells_tokens_apart_within_the_same_second(client):
    """`iat` is whole seconds, so at that precision a token minted moments before a
    reset could not be told from one minted moments after it. `iat_ms` can, and
    these three tokens all fall in the same second (batch 73 review)."""
    import jwt

    from app.core.config import get_settings
    from app.core.security import JWT_ALGORITHM

    _, _, user = await _org_member("Epoch Ms Co", "epochms@x.com")
    before = _token_for(user)
    await asyncio.sleep(0.005)
    epoch = now_utc()
    await _set(user.id, sessions_valid_after=epoch)
    await asyncio.sleep(0.005)
    after = _token_for(user)

    # A token minted before iat_ms existed, inside the reset's own second: whole
    # seconds are all it carries, and within that second it is refused.
    legacy_claims = jwt.decode(after, get_settings().secret_key, algorithms=[JWT_ALGORITHM])
    legacy_claims.pop("iat_ms")
    legacy = jwt.encode(legacy_claims, get_settings().secret_key, algorithm=JWT_ALGORITHM)

    async with SessionLocal() as db:
        assert await get_current_user_ws(_HandshakeStub(before), db) is None
        assert (await get_current_user_ws(_HandshakeStub(after), db))[0].id == user.id
        if int(legacy_claims["iat"]) == int(epoch.timestamp()):
            assert await get_current_user_ws(_HandshakeStub(legacy), db) is None


class _SocketStub:
    """Enough of a WebSocket to drive websocket_endpoint without Starlette's
    TestClient, which cannot share this suite's event loop (see the docstring of
    test_realtime_revalidation.py). `frames` are what the client sends; running out
    reads as a disconnect, so a missing close ends the loop instead of hanging it."""

    def __init__(self, token: str, frames=(), before_each_frame=None) -> None:
        """Hold the access cookie, the frames to deliver and an optional hook awaited before each one."""
        self.cookies = {ACCESS_COOKIE: token}
        self.query_params: dict[str, str] = {}
        self.accepted = False
        self.closed: tuple[int, str] | None = None
        self.sent: list[dict] = []
        self._frames = list(frames)
        self._before_each_frame = before_each_frame

    async def accept(self) -> None:
        """Record that the socket was accepted."""
        self.accepted = True

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Record the close code and reason."""
        self.closed = (code, reason)

    async def send_text(self, data: str) -> None:
        """Record an outbound frame, parsed from JSON."""
        self.sent.append(json.loads(data))

    async def receive_text(self) -> str:
        """Await the hook and deliver the next frame, or raise WebSocketDisconnect once they run out."""
        if not self._frames:
            raise WebSocketDisconnect(code=1000)
        if self._before_each_frame is not None:
            await self._before_each_frame()
        return self._frames.pop(0)


def test_the_ws_close_code_is_4403_and_not_4001():
    """Both clients answer 4001 by refreshing and reconnecting, which succeeds for a
    reset account, so a 4001 here would loop. The clients match on 4403."""
    assert hub.WS_CLOSE_PASSWORD_CHANGE_REQUIRED == 4403
    assert hub.WS_CLOSE_PASSWORD_CHANGE_REQUIRED_REASON == "Password change required"


async def test_ws_handshake_refuses_a_flagged_account_before_registering_anything(client, monkeypatch):
    """A flagged account's socket is closed 4403 before `connected`, registration, presence or calls."""
    _, _, user = await _org_member("WS Hand Co", "wshand@x.com", must_change_password=True)
    reached: list[str] = []

    async def _spy(name):
        """Record that a stubbed step was reached."""
        reached.append(name)

    async def _add(*_a, **_k):
        """Stand in for registry.add and record the call."""
        await _spy("registry.add")

    async def _online(*_a, **_k):
        """Stand in for presence.mark_online, record the call and report the user newly online."""
        await _spy("presence.mark_online")
        return True

    async def _resume(*_a, **_k):
        """Stand in for resume_calls_for and record the call."""
        await _spy("resume_calls_for")

    monkeypatch.setattr(hub.registry, "add", _add)
    monkeypatch.setattr(hub.presence, "mark_online", _online)
    monkeypatch.setattr("app.services.calls.resume_calls_for", _resume)

    ws = _SocketStub(_token_for(user), frames=['{"type": "ping"}'])
    await hub.websocket_endpoint(ws)

    assert ws.accepted
    assert ws.closed == (4403, "Password change required")
    assert ws.sent == [], "a refused socket must not even be told `connected`"
    assert reached == []


async def test_ws_handshake_keeps_4001_for_a_deactivated_flagged_account(client):
    """A deactivated account that is also flagged is closed 4001 at the handshake, not 4403."""
    _, _, user = await _org_member("WS Gone Co", "wsgone@x.com", must_change_password=True)
    await _set(user.id, is_active=False)
    ws = _SocketStub(_token_for(user))
    await hub.websocket_endpoint(ws)
    assert ws.closed == (4001, "Invalid token")


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"must_change_password": True}, (4403, "Password change required")),
        # Ordering: an account that is also deactivated keeps the existing 4001.
        ({"must_change_password": True, "is_active": False}, (4001, "Account inactive")),
        # What a real reset writes: the epoch ends the socket's session, ahead of the flag.
        ({"must_change_password": True, "sessions_valid_after": "now"}, (4001, "Session ended")),
    ],
)
async def test_ws_revalidation_closes_a_socket_the_reset_lands_on(client, monkeypatch, change, expected):
    """Revalidation closes an open socket with the code for what landed and drops it from the registry.

    4403 for the flag alone, 4001 when the account is also deactivated or the epoch ends the session.
    """
    _, _, user = await _org_member("WS Live Co", "wslive@x.com")
    # Revalidate on the first frame instead of waiting thirty seconds.
    monkeypatch.setattr(hub, "REVALIDATE_SECONDS", -1)

    async def _reset_lands():
        """Apply the parametrised change to the user's row, "now" standing for the current time."""
        await _set(user.id, **{k: (now_utc() if v == "now" else v) for k, v in change.items()})

    ws = _SocketStub(_token_for(user), frames=['{"type": "ping"}'], before_each_frame=_reset_lands)
    await hub.websocket_endpoint(ws)

    assert ws.sent and ws.sent[0]["type"] == "connected", "the socket was open before the reset"
    assert ws.closed == expected
    assert str(user.id) not in hub.registry.connections


class _SilentSocketStub(_SocketStub):
    """A client that never sends anything, and is never told to hurry."""

    async def receive_text(self) -> str:
        """Block for an hour, so only the endpoint's own timers end the wait."""
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


async def test_ws_revalidation_runs_on_the_clock_for_a_socket_that_sends_nothing(client, monkeypatch):
    """Revalidation used to run only when a frame arrived, so a silent socket kept
    receiving for up to REVALIDATE_SECONDS + HEARTBEAT_TIMEOUT after a reset."""
    _, _, user = await _org_member("WS Quiet Co", "wsquiet@x.com")
    monkeypatch.setattr(hub, "REVALIDATE_SECONDS", 0.05)
    ws = _SilentSocketStub(_token_for(user))
    task = asyncio.create_task(hub.websocket_endpoint(ws))
    for _ in range(200):
        if ws.sent:
            break
        await asyncio.sleep(0.01)
    assert ws.sent and ws.sent[0]["type"] == "connected"
    await _set(user.id, must_change_password=True)
    await asyncio.wait_for(task, timeout=5)
    assert ws.closed == (4403, "Password change required")


async def test_ws_junk_frames_do_not_skip_revalidation(client, monkeypatch):
    """An unparseable frame `continue`d past the checks, so a client sending junk
    every minute was never re-checked at all (batch 73 review)."""
    _, _, user = await _org_member("WS Junk Co", "wsjunk@x.com")
    monkeypatch.setattr(hub, "REVALIDATE_SECONDS", -1)

    async def _reset_lands():
        """Flag the user's row as a reset would."""
        await _set(user.id, must_change_password=True)

    ws = _SocketStub(_token_for(user), frames=["not json"] * 3, before_each_frame=_reset_lands)
    await hub.websocket_endpoint(ws)
    assert ws.closed == (4403, "Password change required")


class _ResetDuringHandshakeStub(_SocketStub):
    """An administrator's reset that lands inside the handshake: after
    get_current_user_ws has accepted the token, before the socket's loop starts."""

    def __init__(self, token: str, user_id, **kwargs) -> None:
        """Keep the id of the user whose row the reset lands on."""
        super().__init__(token, **kwargs)
        self._user_id = user_id

    async def accept(self) -> None:
        """Accept, then flag the user and stamp the session epoch just after the token's iat_ms."""
        await super().accept()
        await asyncio.sleep(0.005)  # past the token's iat_ms millisecond
        await _set(self._user_id, must_change_password=True, sessions_valid_after=now_utc())


async def test_ws_a_reset_inside_the_handshake_ends_the_socket_even_after_the_change(client, monkeypatch):
    """The epoch used to be compared with the moment the socket connected, which is
    later than the moment its token was checked. A reset between the two left the
    socket "newer" than the epoch, so once the owner changed the password nothing
    closed it — while HTTP refused the very same token (batch 73 review)."""
    _, _, user = await _org_member("WS Hand Race Co", "wshandrace@x.com")
    monkeypatch.setattr(hub, "REVALIDATE_SECONDS", -1)

    async def _owner_changes_the_password(_user_id):
        """Stand in for resume_calls_for and clear the flag, as the owner changing the password would."""
        # The handshake's last await before the loop, so the flag is already clear
        # when the first revalidation runs and only the epoch can close the socket.
        await _set(user.id, must_change_password=False)

    monkeypatch.setattr("app.services.calls.resume_calls_for", _owner_changes_the_password)
    token = _token_for(user)
    ws = _ResetDuringHandshakeStub(token, user.id, frames=['{"type": "ping"}'])
    await hub.websocket_endpoint(ws)

    assert ws.sent and ws.sent[0]["type"] == "connected", "the socket did open"
    assert (await _row(user.id)).must_change_password is False
    assert ws.closed == (4001, "Session ended")
    # The rule HTTP applies to the same token.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c.headers.update(CSRF)
        resp = await c.get("/api/conversations", headers={"Cookie": f"{ACCESS_COOKIE}={token}"})
        assert resp.status_code == 401


_HANG_UPS = ("call:end", "call:cancel", "call:decline", "call:leave")
_REFUSALS = [
    pytest.param({"must_change_password": True}, (4403, "Password change required"), id="flagged"),
    pytest.param({"is_active": False}, (4001, "Account inactive"), id="inactive"),
    # A token from before a reset once the owner has changed the password: the flag
    # is clear and only the epoch is left to refuse it.
    pytest.param({"sessions_valid_after": "now"}, (4001, "Session ended"), id="pre-reset-token"),
]


def _once(user_id, change: dict):
    """A before_each_frame hook that applies `change` to the user's row, once."""
    done = False

    async def _apply() -> None:
        """Apply the change on the first call only, just after the token's iat_ms."""
        nonlocal done
        if done:
            return
        done = True
        await asyncio.sleep(0.005)  # past the token's iat_ms millisecond
        await _set(user_id, **{k: (now_utc() if v == "now" else v) for k, v in change.items()})

    return _apply


@pytest.mark.parametrize("starter", ["call:initiate", "call:group_initiate", "call:accept", "call:join"])
@pytest.mark.parametrize(("change", "expected"), _REFUSALS)
async def test_ws_a_refused_account_can_hang_up_but_not_start_a_call(
    client, monkeypatch, starter, change, expected
):
    """Every frame but a ping or a hang-up is re-checked against the live row before
    it is dispatched (batch 73 review). The hang-ups go through, because both
    clients send them on their way into the change-password or sign-in screen; the
    frame that would start or join a call closes the socket instead."""
    _, _, user = await _org_member("WS Call Co", "wscall@x.com")
    # Far beyond the test, so only the per-frame check can act.
    monkeypatch.setattr(hub, "REVALIDATE_SECONDS", 3600)
    dispatched: list[str] = []

    async def _dispatch(_user, data):
        """Record the call frame's type instead of handling it."""
        dispatched.append(data["type"])

    monkeypatch.setattr("app.services.calls.handle_call_ws_message", _dispatch)
    frames = [json.dumps({"type": t, "call_id": "c1"}) for t in (*_HANG_UPS, starter)]
    ws = _SocketStub(_token_for(user), frames=frames, before_each_frame=_once(user.id, change))
    await hub.websocket_endpoint(ws)

    assert ws.sent and ws.sent[0]["type"] == "connected"
    assert dispatched == list(_HANG_UPS)
    assert ws.closed == expected


async def test_ws_the_per_frame_check_lets_a_healthy_account_through(client, monkeypatch):
    """Control for the test above: nothing changes, so everything is dispatched."""
    _, _, user = await _org_member("WS Call Ok Co", "wscallok@x.com")
    monkeypatch.setattr(hub, "REVALIDATE_SECONDS", 3600)
    dispatched: list[str] = []

    async def _dispatch(_user, data):
        """Record the call frame's type instead of handling it."""
        dispatched.append(data["type"])

    monkeypatch.setattr("app.services.calls.handle_call_ws_message", _dispatch)
    frames = [json.dumps({"type": t, "call_id": "c1"}) for t in ("call:initiate", *_HANG_UPS)]
    ws = _SocketStub(_token_for(user), frames=frames)
    await hub.websocket_endpoint(ws)
    assert dispatched == ["call:initiate", *_HANG_UPS]
    assert ws.closed is None


@pytest.mark.parametrize("frame_type", ["message", "typing_start", "read_receipt"])
async def test_ws_a_flagged_socket_is_closed_by_its_next_frame_that_acts(client, monkeypatch, frame_type):
    """A frame that acts, from a socket flagged since its handshake, closes it 4403 undispatched."""
    _, _, user = await _org_member("WS Frame Co", "wsframe@x.com")
    monkeypatch.setattr(hub, "REVALIDATE_SECONDS", 3600)
    reached: list[str] = []

    async def _handle(_user, data):
        """Record the frame's type instead of handling it."""
        reached.append(data["type"])

    monkeypatch.setattr(hub, "_handle_inbound", _handle)
    frame = {"type": frame_type, "conversation_id": str(uuid.uuid4()), "content": "after", "temp_id": "t1"}
    ws = _SocketStub(
        _token_for(user),
        frames=[json.dumps(frame)],
        before_each_frame=_once(user.id, {"must_change_password": True}),
    )
    await hub.websocket_endpoint(ws)
    assert ws.closed == (4403, "Password change required")
    assert reached == [], "refused before it was dispatched"


# ---------------------------------------------------------------------------
# Structural guard: every route is gated, and the exceptions are exactly these
# ---------------------------------------------------------------------------


def _reached(dependant, found: set) -> set:
    """Add every dependency callable reachable from `dependant`, recursively, to `found` and return it."""
    for sub in dependant.dependencies:
        # RateLimiter dependencies are instances, not functions, and carry no
        # __qualname__; identity against the two functions below is all that matters.
        found.add(sub.call)
        _reached(sub, found)
    return found


def test_every_route_is_gated_except_the_ones_that_must_not_be():
    """Every APIRoute reaches get_current_user except the public ones and the four allowed while pending.

    None reaches both get_current_user and its un-gated twin.
    """
    gated, twin = deps.get_current_user, deps.get_current_user_pending_password_change
    neither, via_twin, both = set(), set(), set()
    for route in app.routes:
        # Only APIRoutes: the dev-only docs Routes authenticate nobody by design and
        # the websocket authenticates itself (tested above).
        if not isinstance(route, APIRoute):
            continue
        calls = _reached(route.dependant, set())
        for method in route.methods:
            key = (method, route.path)
            if gated in calls and twin in calls:
                both.add(key)
            elif twin in calls:
                via_twin.add(key)
            elif gated not in calls:
                neither.add(key)

    assert neither == PUBLIC_ROUTES, (
        "a route that reaches neither get_current_user nor its un-gated twin authenticates nobody — "
        "if that is deliberate, add it to PUBLIC_ROUTES"
    )
    assert via_twin == ALLOWED_WHILE_PENDING, (
        "the un-gated twin lets an account an administrator reset past the password-change gate; "
        "it belongs on logout, me, change-password and push unsubscribe only"
    )
    # Reaching both would gate the route anyway, which means the twin on it is dead
    # weight at best and a reset account locked out of the way out at worst.
    assert both == set()
