"""What an org admin may do to accounts.

One rule, and it needs closing in two directions to mean anything:

  1. An admin creates MEMBERS, never admins. POST /users is the obvious route;
     PUT /users/{id} with role=admin is the same grant reached through a member
     who already exists.
  2. An admin cannot take over an admin that already exists. Blocking creation
     while leaving seizure open would protect nothing — reset-password returns
     the new password in its own response body, so one call against a peer hands
     the actor a working login for an account they do not own.

Department scoping used to live here too, and is gone with the access-control
feature: an org admin manages their whole organisation again. What is left is
the role boundary, which never depended on departments.
"""

import contextlib

from httpx import ASGITransport, AsyncClient

from app.db.models import Department, User, UserRole
from app.db.session import SessionLocal
from app.main import app
from tests.conftest import CSRF, login, make_org, make_user


@contextlib.asynccontextmanager
async def _as(email):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c.headers.update(CSRF)
        await login(c, email)
        yield c


async def _dept(org_id, name):
    async with SessionLocal() as db:
        d = Department(org_id=org_id, name=name)
        db.add(d)
        await db.commit()
        await db.refresh(d)
        return d


async def _role_of(email) -> UserRole:
    async with SessionLocal() as db:
        return (await db.execute(User.__table__.select().where(User.email == email))).one().role


async def _password_hash(email) -> str:
    async with SessionLocal() as db:
        return (await db.execute(User.__table__.select().where(User.email == email))).one().password_hash


async def _fixture(name: str):
    """An org with two departments and an admin."""
    org = await make_org(name)
    ward = await _dept(org.id, "Ward")
    pharmacy = await _dept(org.id, "Pharmacy")
    admin = await make_user(
        f"adm@{name.replace(' ', '')}.com", org_id=org.id, role=UserRole.org_admin, dept_id=ward.id
    )
    return org, ward, pharmacy, admin


# ---------------------------------------------------------------------------
# Rule 1: members only
# ---------------------------------------------------------------------------


async def test_admin_cannot_create_another_admin(client):
    org, ward, _, _ = await _fixture("Members Only Co")

    async with _as("adm@MembersOnlyCo.com") as c:
        resp = await c.post(
            "/api/org-admin/users",
            json={
                "dept_id": str(ward.id),
                "email": "new.admin@members.com",
                "display_name": "New Admin",
                "password": "TestPass1234",
                "role": "admin",
            },
        )
    assert resp.status_code == 403, resp.text
    assert "members only" in resp.json()["detail"].lower()

    # Refused, not quietly downgraded to a member — an admin who asked for an
    # admin and silently got a member would hand out the wrong credentials.
    async with SessionLocal() as db:
        assert (
            await db.execute(User.__table__.select().where(User.email == "new.admin@members.com"))
        ).one_or_none() is None


async def test_admin_can_still_create_members_in_any_department(client):
    """The positive control, and the department half of the removal.

    Ward's admin creating someone in Pharmacy is the case that used to 404. It
    must work now: delegation is gone, so an org admin covers the whole org.
    """
    org, ward, pharmacy, _ = await _fixture("Happy Path Co")

    async with _as("adm@HappyPathCo.com") as c:
        mine = await c.post(
            "/api/org-admin/users",
            json={
                "dept_id": str(ward.id),
                "email": "nurse@happy.com",
                "display_name": "Nurse Joy",
                "password": "TestPass1234",
                "role": "member",
            },
        )
        other = await c.post(
            "/api/org-admin/users",
            json={
                "dept_id": str(pharmacy.id),
                "email": "pharmacist@happy.com",
                "display_name": "Pharmacist",
                "password": "TestPass1234",
                "role": "member",
            },
        )
        depts = await c.get("/api/org-admin/departments")

    assert mine.status_code == 200, mine.text
    assert other.status_code == 200, other.text
    assert mine.json()["role"] == "member"
    assert await _role_of("nurse@happy.com") is UserRole.member
    assert sorted(d["name"] for d in depts.json()) == ["Pharmacy", "Ward"]


async def test_admin_cannot_promote_a_member_to_admin(client):
    """The second route to the same grant."""
    org, ward, _, _ = await _fixture("No Promotion Co")
    member = await make_user("member@noprom.com", org_id=org.id, dept_id=ward.id)

    async with _as("adm@NoPromotionCo.com") as c:
        resp = await c.put(f"/api/org-admin/users/{member.id}", json={"role": "admin"})
    assert resp.status_code == 403, resp.text
    assert await _role_of("member@noprom.com") is UserRole.member


async def test_promotion_is_refused_without_applying_the_rest_of_the_edit(client):
    """A rejected role must not leave a half-applied request behind.

    The route mutates the ORM object field by field, so a 403 raised after
    display_name was already assigned would still be rolled back by the session
    teardown — but only because nothing commits first. Worth pinning: someone
    moving the role check below the commit would break it silently.
    """
    org, ward, _, _ = await _fixture("Atomic Co")
    member = await make_user("atomic@members.com", org_id=org.id, dept_id=ward.id, display_name="Before")

    async with _as("adm@AtomicCo.com") as c:
        resp = await c.put(
            f"/api/org-admin/users/{member.id}", json={"display_name": "After", "role": "admin"}
        )
        assert resp.status_code == 403
        again = await c.get(f"/api/org-admin/users/{member.id}")
    assert again.json()["display_name"] == "Before"


async def test_editing_a_member_still_works(client):
    """The no-op-role case, and the positive control for the route above.

    The drawer sends every field on save, so an ordinary rename arrives with
    role="member" alongside it. If the check keyed on the requested role rather
    than on whether it is a CHANGE, renaming anyone would 403.
    """
    org, ward, pharmacy, _ = await _fixture("Rename Co")
    member = await make_user("rename@members.com", org_id=org.id, dept_id=ward.id)

    async with _as("adm@RenameCo.com") as c:
        resp = await c.put(
            f"/api/org-admin/users/{member.id}",
            json={"display_name": "Renamed", "role": "member", "dept_id": str(pharmacy.id)},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["display_name"] == "Renamed"
    assert resp.json()["dept_name"] == "Pharmacy"


# ---------------------------------------------------------------------------
# Rule 2: hands off other admins
# ---------------------------------------------------------------------------


async def test_admin_cannot_reset_a_peer_admins_password(client):
    """The takeover that made rule 1 decorative.

    reset-password returns the new password in its own response body, so without
    this an admin who cannot MINT an admin can simply TAKE one: reset a peer,
    read the password out of the 200, sign in as them.
    """
    org, ward, _, _ = await _fixture("No Takeover Co")
    peer = await make_user("peer@notakeover.com", org_id=org.id, dept_id=ward.id, role=UserRole.org_admin)
    before = await _password_hash("peer@notakeover.com")

    async with _as("adm@NoTakeoverCo.com") as c:
        resp = await c.post(f"/api/org-admin/users/{peer.id}/reset-password")

    assert resp.status_code == 403, resp.text
    assert "temporary_password" not in resp.json()
    # The password must be untouched, not merely undisclosed: a reset that
    # happened but was not reported would still have locked the peer out.
    assert await _password_hash("peer@notakeover.com") == before


async def test_admin_cannot_deactivate_or_demote_a_peer_admin(client):
    """Every single-user mutation resolves through _load_org_user, so one check
    covers them all. Asserted across three routes rather than one, because the
    guard living in the shared helper is exactly what a future route would
    inherit — or lose."""
    org, ward, pharmacy, _ = await _fixture("Hands Off Co")
    peer = await make_user("peer@handsoff.com", org_id=org.id, dept_id=ward.id, role=UserRole.org_admin)

    async with _as("adm@HandsOffCo.com") as c:
        deactivate = await c.put(f"/api/org-admin/users/{peer.id}", json={"is_active": False})
        demote = await c.put(f"/api/org-admin/users/{peer.id}", json={"role": "member"})
        move = await c.put(f"/api/org-admin/users/{peer.id}", json={"dept_id": str(pharmacy.id)})
        read = await c.get(f"/api/org-admin/users/{peer.id}")

    assert deactivate.status_code == 403, deactivate.text
    assert demote.status_code == 403, demote.text
    assert move.status_code == 403, move.text
    assert read.status_code == 403, read.text
    assert await _role_of("peer@handsoff.com") is UserRole.org_admin


async def test_an_admin_can_still_manage_their_own_account(client):
    """The guard is about OTHER admins. Locking an admin out of their own row
    would break the ordinary case of correcting your own display name."""
    org, _, _, admin = await _fixture("Self Serve Co")

    async with _as("adm@SelfServeCo.com") as c:
        resp = await c.put(f"/api/org-admin/users/{admin.id}", json={"display_name": "My New Name"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["display_name"] == "My New Name"


async def test_peer_admins_are_still_listed(client):
    """Blocked from MUTATING, not from seeing.

    An admin needs to know who else administers the organisation — that is who
    they escalate to. Hiding peers would also make the console look like it had
    lost people.
    """
    org, ward, _, _ = await _fixture("Visible Co")
    await make_user(
        "peer@visible.com", org_id=org.id, dept_id=ward.id, role=UserRole.org_admin, display_name="Peer Admin"
    )

    async with _as("adm@VisibleCo.com") as c:
        listed = (await c.get("/api/org-admin/users")).json()["data"]

    assert "Peer Admin" in [u["display_name"] for u in listed]
