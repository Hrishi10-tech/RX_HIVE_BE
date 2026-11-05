"""Super-admin & org-admin portal contracts on Postgres (P1 exit gate)."""

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditLog, RefreshToken, User, UserRole
from app.db.session import SessionLocal
from tests.conftest import CSRF, login, make_user


async def _superadmin_client(client):
    await make_user("root@x.com", role=UserRole.superadmin)
    await login(client, "root@x.com")
    return client


async def test_org_department_user_crud_flow(client):
    await _superadmin_client(client)

    # create org
    resp = await client.post("/api/admin/organizations", json={"name": "Acme Corp"})
    assert resp.status_code == 200, resp.text
    org = resp.json()
    assert org["slug"] == "acme-corp" and org["is_active"] is True
    org_id = org["_id"]

    # list uses _id + envelope
    resp = await client.get("/api/admin/organizations")
    body = resp.json()
    assert body["total"] == 1 and body["data"][0]["_id"] == org_id

    # name-availability validator
    resp = await client.get("/api/admin/validate/org-name", params={"name": "Acme Corp"})
    assert resp.json()["available"] is False
    resp = await client.get("/api/admin/validate/org-name", params={"name": "Totally New"})
    assert resp.json()["available"] is True

    # department
    resp = await client.post("/api/admin/departments", json={"org_id": org_id, "name": "Engineering"})
    assert resp.status_code == 200
    dept_id = resp.json()["_id"]

    # duplicate department rejected
    resp = await client.post("/api/admin/departments", json={"org_id": org_id, "name": "Engineering"})
    assert resp.status_code == 400

    # user with weak password rejected (400 policy, or 422 schema min_length)
    resp = await client.post(
        "/api/admin/users",
        json={
            "org_id": org_id,
            "dept_id": dept_id,
            "email": "weak@acme.com",
            "display_name": "Weak",
            "password": "short",
        },
    )
    assert resp.status_code in (400, 422)

    # user created ok
    resp = await client.post(
        "/api/admin/users",
        json={
            "org_id": org_id,
            "dept_id": dept_id,
            "email": "sam@acme.com",
            "display_name": "Sam Rivera",
            "password": "GoodPass1234",
            "role": "member",
        },
    )
    assert resp.status_code == 200, resp.text
    user = resp.json()
    assert "password_hash" not in user and user["org_name"] == "Acme Corp"
    user_id = user["_id"]

    # duplicate email (case-insensitive) rejected
    resp = await client.post(
        "/api/admin/users",
        json={
            "org_id": org_id,
            "dept_id": dept_id,
            "email": "SAM@acme.com",
            "display_name": "Dup",
            "password": "GoodPass1234",
        },
    )
    assert resp.status_code == 400

    # reset password returns a temp password + flags must_change
    resp = await client.post(f"/api/admin/users/{user_id}/reset-password")
    assert resp.status_code == 200 and len(resp.json()["temporary_password"]) >= 10
    async with SessionLocal() as db:
        u = await db.get(User, __import__("uuid").UUID(user_id))
        assert u.must_change_password is True


async def test_deactivation_revokes_refresh_tokens(client):
    await _superadmin_client(client)
    resp = await client.post("/api/admin/organizations", json={"name": "RevokeCo"})
    org_id = resp.json()["_id"]
    resp = await client.post("/api/admin/departments", json={"org_id": org_id, "name": "Ops"})
    dept_id = resp.json()["_id"]
    resp = await client.post(
        "/api/admin/users",
        json={
            "org_id": org_id,
            "dept_id": dept_id,
            "email": "victim@revoke.com",
            "display_name": "Victim",
            "password": "GoodPass1234",
        },
    )
    user_id = resp.json()["_id"]

    # victim logs in (gets a refresh token)
    victim = AsyncClient(
        transport=ASGITransport(app=__import__("app.main", fromlist=["app"]).app), base_url="http://test"
    )
    victim.headers.update(CSRF)
    async with victim:
        await login(victim, "victim@revoke.com", "GoodPass1234")
        assert (await victim.get("/api/auth/me")).status_code == 200

        # admin deactivates the user
        resp = await client.put(f"/api/admin/users/{user_id}", json={"is_active": False})
        assert resp.status_code == 200

        # every refresh token for the user is revoked
        import uuid as _uuid

        async with SessionLocal() as db:
            tokens = (
                (await db.execute(select(RefreshToken).where(RefreshToken.user_id == _uuid.UUID(user_id))))
                .scalars()
                .all()
            )
            assert tokens and all(t.revoked_at is not None for t in tokens)

        # victim can no longer refresh
        assert (await victim.post("/api/auth/refresh")).status_code == 401


async def test_bulk_change_dept_skips_cross_org(client):
    await _superadmin_client(client)
    a = (await client.post("/api/admin/organizations", json={"name": "OrgA bulk"})).json()["_id"]
    b = (await client.post("/api/admin/organizations", json={"name": "OrgB bulk"})).json()["_id"]
    da = (await client.post("/api/admin/departments", json={"org_id": a, "name": "DA"})).json()["_id"]
    db_ = (await client.post("/api/admin/departments", json={"org_id": b, "name": "DB"})).json()["_id"]
    ua = (
        await client.post(
            "/api/admin/users",
            json={
                "org_id": a,
                "dept_id": da,
                "email": "ua@a.com",
                "display_name": "UA",
                "password": "GoodPass1234",
            },
        )
    ).json()["_id"]

    # moving OrgA's user into OrgB's dept must be skipped (0 moved)
    resp = await client.post(
        "/api/admin/users/bulk-action",
        json={"user_ids": [ua], "action": "change_dept", "dept_id": db_},
    )
    assert resp.status_code == 200
    assert "0" in resp.json()["message"]


async def test_org_state_decisions_take_a_row_lock(client, monkeypatch):
    """The suspension guards decide from Organization.is_active, so they have to
    read it under FOR UPDATE.

    The engine sets no isolation level, so it runs READ COMMITTED: an unlocked
    read lets a suspension commit between the read and the write, and the
    cascade in _deactivate_org_users only touches users that existed when it
    ran. Both sides of that race — the suspension and the create/activate —
    therefore take the same lock. This asserts the lock is actually requested,
    because losing it would be silent and the resulting race is timing-dependent.
    """
    await _superadmin_client(client)
    org_id = (await client.post("/api/admin/organizations", json={"name": "Locking Co"})).json()["_id"]
    dept_id = (await client.post("/api/admin/departments", json={"org_id": org_id, "name": "Ops"})).json()[
        "_id"
    ]

    seen: list[str] = []
    original = AsyncSession.execute

    async def _recording(self, statement, *args, **kwargs):
        seen.append(str(statement))
        return await original(self, statement, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "execute", _recording)

    def _locked_org_read() -> bool:
        return any("FROM organizations" in s and "FOR UPDATE" in s for s in seen)

    # the create path reads is_active to decide, so it locks
    seen.clear()
    resp = await client.post(
        "/api/admin/users",
        json={
            "org_id": org_id,
            "dept_id": dept_id,
            "email": "locked@locking.co",
            "display_name": "Locked",
            "password": "GoodPass1234",
        },
    )
    assert resp.status_code == 200, resp.text
    assert _locked_org_read(), "create_user read Organization.is_active without FOR UPDATE"

    # the suspension path writes it, and must take the same lock
    seen.clear()
    resp = await client.put(f"/api/admin/organizations/{org_id}", json={"is_active": False})
    assert resp.status_code == 200, resp.text
    assert _locked_org_read(), "update_organization suspended an org without FOR UPDATE"


async def test_admin_stats_survives_a_redis_outage(client, monkeypatch):
    """The superadmin dashboard was the one presence reader that did not degrade.

    get_statuses and is_online both sit inside degrade_on_outage, so a Redis
    outage costs a green dot. _count_online_users did not, so the same outage
    500'd /api/admin/stats — including the four tiles that come from Postgres.

    The counting moved into app/services/presence.py behind the online index, so
    the outage is now injected there; the guarantee under test is unchanged.
    """
    await _superadmin_client(client)
    # real rows behind the Postgres-backed tiles, so "still answers" means
    # something (total_users deliberately excludes superadmins)
    org_id = (await client.post("/api/admin/organizations", json={"name": "Outage Co"})).json()["_id"]
    dept_id = (await client.post("/api/admin/departments", json={"org_id": org_id, "name": "Ops"})).json()[
        "_id"
    ]
    resp = await client.post(
        "/api/admin/users",
        json={
            "org_id": org_id,
            "dept_id": dept_id,
            "email": "someone@outage.co",
            "display_name": "Someone",
            "password": "GoodPass1234",
        },
    )
    assert resp.status_code == 200, resp.text

    def _redis_is_down():
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr("app.services.presence.get_redis", _redis_is_down)

    resp = await client.get("/api/admin/stats")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # presence degrades to nobody-online...
    assert body["active_today"] == 0
    # ...while the Postgres-backed tiles still answer for real
    assert body["total_users"] == 1
    assert body["total_orgs"] == 1 and body["total_depts"] == 1
    assert body["recent_activity"]


async def test_bulk_action_audit_record_is_attributed_and_accurate(client):
    """A bulk action is the only admin write that never set audit_logs.org_id.

    The org-admin activity feed filters on that column, so the one operation
    that can deactivate a whole organisation at once was also the one its
    admins could not see. The details blob had the matching problem: it listed
    every id asked for, while `count` beside it reported only those affected.
    """
    import uuid as _uuid

    await _superadmin_client(client)
    org_id = (await client.post("/api/admin/organizations", json={"name": "Bulk Audit Co"})).json()["_id"]
    dept_id = (await client.post("/api/admin/departments", json={"org_id": org_id, "name": "Ops"})).json()[
        "_id"
    ]
    member = (
        await client.post(
            "/api/admin/users",
            json={
                "org_id": org_id,
                "dept_id": dept_id,
                "email": "member@bulkaudit.co",
                "display_name": "Member",
                "password": "GoodPass1234",
            },
        )
    ).json()["_id"]

    # a stranger id that matches no user: requested, but never affected
    ghost = str(_uuid.uuid4())
    resp = await client.post(
        "/api/admin/users/bulk-action", json={"user_ids": [member, ghost], "action": "deactivate"}
    )
    assert resp.status_code == 200
    assert "1 users" in resp.json()["message"]

    async with SessionLocal() as db:
        log = (
            (
                await db.execute(
                    select(AuditLog)
                    .where(AuditLog.action == "bulk_deactivate")
                    .order_by(AuditLog.created_at.desc())
                )
            )
            .scalars()
            .first()
        )
        assert log is not None
        # attributed to the org, so the org-admin activity feed can find it
        assert log.org_id == _uuid.UUID(org_id)
        assert log.details["org_ids"] == [org_id]
        # and it records what actually happened, not what was asked for
        assert log.details["user_ids"] == [member]
        assert log.details["count"] == 1
        assert ghost not in log.details["user_ids"]


async def test_org_suspension_survives_user_create_and_activate(client):
    """A suspended org must stay suspended.

    Nothing on the login path reads Organization.is_active — the suspension is
    carried entirely by the users.is_active cascade — so any route that creates
    or reactivates an account inside a suspended org silently restores login to
    a tenant that is supposed to be shut off.
    """
    import uuid as _uuid

    await _superadmin_client(client)
    org_id = (await client.post("/api/admin/organizations", json={"name": "Suspend Co"})).json()["_id"]
    dept_id = (await client.post("/api/admin/departments", json={"org_id": org_id, "name": "Ward"})).json()[
        "_id"
    ]

    async def _new_user(email: str):
        return await client.post(
            "/api/admin/users",
            json={
                "org_id": org_id,
                "dept_id": dept_id,
                "email": email,
                "display_name": email.split("@")[0],
                "password": "GoodPass1234",
            },
        )

    first = (await _new_user("first@suspend.co")).json()["_id"]
    second = (await _new_user("second@suspend.co")).json()["_id"]

    async def _assert_active(expected: bool):
        async with SessionLocal() as db:
            for uid in (first, second):
                assert (await db.get(User, _uuid.UUID(uid))).is_active is expected

    async def _login_first():
        return await client.post(
            "/api/auth/login", json={"email": "first@suspend.co", "password": "GoodPass1234"}
        )

    async def _set_org_active(active: bool):
        resp = await client.put(f"/api/admin/organizations/{org_id}", json={"is_active": active})
        assert resp.status_code == 200, resp.text

    # suspending the org cascades to its members, who can no longer sign in
    await _set_org_active(False)
    await _assert_active(False)
    assert (await _login_first()).status_code == 401

    # a fresh account may not be minted inside the suspended org
    resp = await _new_user("backdoor@suspend.co")
    assert resp.status_code == 400, resp.text

    # nor may an existing one be reactivated, one at a time...
    resp = await client.put(f"/api/admin/users/{first}", json={"is_active": True})
    assert resp.status_code == 400, resp.text

    # ...nor in bulk, which is the call that could restore a whole tenant at once.
    # Mixed batch: a deactivated user in a HEALTHY org rides along, to prove the
    # suspended pair is skipped rather than the whole request being refused.
    other_org = (await client.post("/api/admin/organizations", json={"name": "Bystander Co"})).json()["_id"]
    other_dept = (
        await client.post("/api/admin/departments", json={"org_id": other_org, "name": "Clinic"})
    ).json()["_id"]
    bystander = (
        await client.post(
            "/api/admin/users",
            json={
                "org_id": other_org,
                "dept_id": other_dept,
                "email": "bystander@ok.co",
                "display_name": "Bystander",
                "password": "GoodPass1234",
            },
        )
    ).json()["_id"]
    assert (await client.put(f"/api/admin/users/{bystander}", json={"is_active": False})).status_code == 200

    resp = await client.post(
        "/api/admin/users/bulk-action",
        json={"user_ids": [first, second, bystander], "action": "activate"},
    )
    assert resp.status_code == 200
    # exactly one changed: the bystander, not the two in the suspended org
    assert "1 users" in resp.json()["message"]
    await _assert_active(False)
    async with SessionLocal() as db:
        assert (await db.get(User, _uuid.UUID(bystander))).is_active is True
    assert (await _login_first()).status_code == 401

    # the guard tracks the org's current state rather than freezing the accounts:
    # restoring the org makes both activation paths work again.
    await _set_org_active(True)
    resp = await client.put(f"/api/admin/users/{first}", json={"is_active": True})
    assert resp.status_code == 200, resp.text
    resp = await client.post(
        "/api/admin/users/bulk-action", json={"user_ids": [second], "action": "activate"}
    )
    assert resp.status_code == 200
    assert "1 users" in resp.json()["message"]
    await _assert_active(True)
    assert (await _login_first()).status_code == 200
