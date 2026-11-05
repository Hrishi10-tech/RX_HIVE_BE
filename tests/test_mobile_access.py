"""The per-user mobile-app grant: who may sign in from the native client.

Three rules, each with a test that fails if the rule regresses:
  1. Web sign-in is untouched — the grant gates the mobile client only.
  2. Superadmins can never sign in on mobile, grant or no grant.
  3. The grant is revocable, and revoking it ends the phone's session without
     ending the same user's web session.
"""

from sqlalchemy import select

from app.api.auth import MOBILE_NOT_APPROVED_CODE, SUPERADMIN_MOBILE_DENIED_CODE
from app.db.models import RefreshToken, User, UserRole
from app.db.session import SessionLocal
from tests.conftest import login, make_org, make_user


def _mobile_login(email: str, password: str = "TestPass1234") -> dict:
    return {"email": email, "password": password, "client": "mobile"}


# ---------------------------------------------------------------------------
# Login gate
# ---------------------------------------------------------------------------


async def test_member_without_grant_cannot_login_on_mobile(client):
    await make_user("nomob@x.com")
    resp = await client.post("/api/auth/login", json=_mobile_login("nomob@x.com"))
    assert resp.status_code == 403
    # Must not read as a credential problem, or the user retypes forever.
    assert "Mobile access has not been enabled" in resp.json()["detail"]
    # The code is what the client routes on; the sentence above is only for reading.
    assert resp.json()["code"] == MOBILE_NOT_APPROVED_CODE


async def test_member_with_grant_can_login_on_mobile(client):
    await make_user("yesmob@x.com", mobile_access=True)
    resp = await client.post("/api/auth/login", json=_mobile_login("yesmob@x.com"))
    assert resp.status_code == 200
    assert resp.json()["user"]["email"] == "yesmob@x.com"


async def test_org_admin_needs_the_grant_too(client):
    """The grant is per-account, not per-role: being an org admin is not a bypass."""
    org = await make_org("Mobile Org")
    await make_user("oa@x.com", role=UserRole.org_admin, org_id=org.id)
    resp = await client.post("/api/auth/login", json=_mobile_login("oa@x.com"))
    assert resp.status_code == 403

    async with SessionLocal() as db:
        target = (await db.execute(select(User).where(User.email == "oa@x.com"))).scalar_one()
        target.mobile_access = True
        await db.commit()

    resp = await client.post("/api/auth/login", json=_mobile_login("oa@x.com"))
    assert resp.status_code == 200


async def test_superadmin_cannot_login_on_mobile_even_with_grant(client):
    await make_user("root2@x.com", role=UserRole.superadmin, mobile_access=True)
    resp = await client.post("/api/auth/login", json=_mobile_login("root2@x.com"))
    assert resp.status_code == 403
    assert "only sign in on the web" in resp.json()["detail"]
    assert resp.json()["code"] == SUPERADMIN_MOBILE_DENIED_CODE


async def test_the_two_mobile_denials_are_distinguishable_without_reading_the_prose(client):
    """The reason this code exists at all.

    Both sentences mention a super admin, so the wording cannot separate them — and
    the client has to, because one denial has a remedy and the other never will.
    """
    await make_user("prose1@x.com")  # member, no grant
    await make_user("prose2@x.com", role=UserRole.superadmin)

    member = await client.post("/api/auth/login", json=_mobile_login("prose1@x.com"))
    superadmin = await client.post("/api/auth/login", json=_mobile_login("prose2@x.com"))

    assert "super admin" in member.json()["detail"].lower()
    assert "super admin" in superadmin.json()["detail"].lower()
    assert member.json()["code"] != superadmin.json()["code"]


async def test_an_ordinary_refusal_keeps_the_plain_detail_envelope(client):
    """`code` is added only where a client must branch. Every other error response
    keeps FastAPI's exact shape, so nothing else on the wire moved."""
    await make_user("plain@x.com", mobile_access=True)
    resp = await client.post("/api/auth/login", json=_mobile_login("plain@x.com", "wrong-pass-9"))
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Invalid email or password"}


async def test_web_login_is_unaffected_by_the_grant(client):
    """No client field, and client="web", both behave exactly as before."""
    await make_user("webonly@x.com")  # mobile_access defaults to False
    assert (
        await client.post("/api/auth/login", json={"email": "webonly@x.com", "password": "TestPass1234"})
    ).status_code == 200
    assert (
        await client.post(
            "/api/auth/login", json={"email": "webonly@x.com", "password": "TestPass1234", "client": "web"}
        )
    ).status_code == 200


async def test_superadmin_web_login_still_works(client):
    await make_user("root3@x.com", role=UserRole.superadmin)
    resp = await client.post("/api/auth/login", json={"email": "root3@x.com", "password": "TestPass1234"})
    assert resp.status_code == 200


async def test_bad_password_is_401_not_403_for_unapproved_account(client):
    """Approval state must not leak to someone who cannot authenticate."""
    await make_user("secret@x.com")  # not approved for mobile
    resp = await client.post("/api/auth/login", json=_mobile_login("secret@x.com", "wrong-pass-9"))
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid email or password"


async def test_unknown_client_value_is_rejected(client):
    await make_user("weird@x.com", mobile_access=True)
    resp = await client.post(
        "/api/auth/login",
        json={"email": "weird@x.com", "password": "TestPass1234", "client": "android-tv"},
    )
    assert resp.status_code == 422  # Literal["web","mobile"]


# ---------------------------------------------------------------------------
# Revocation
# ---------------------------------------------------------------------------


async def test_revoking_grant_kills_the_mobile_access_token_immediately(client):
    """Not "at next refresh" — the access token carries a signed client claim, so
    every request re-checks the grant."""
    user = await make_user("revoke1@x.com", mobile_access=True)
    resp = await client.post("/api/auth/login", json=_mobile_login("revoke1@x.com"))
    assert resp.status_code == 200
    assert (await client.get("/api/auth/me")).status_code == 200

    async with SessionLocal() as db:
        target = await db.get(User, user.id)
        target.mobile_access = False
        await db.commit()

    assert (await client.get("/api/auth/me")).status_code == 401


async def test_revoking_grant_blocks_mobile_refresh_and_burns_the_token(client):
    user = await make_user("revoke2@x.com", mobile_access=True)
    await client.post("/api/auth/login", json=_mobile_login("revoke2@x.com"))

    async with SessionLocal() as db:
        target = await db.get(User, user.id)
        target.mobile_access = False
        await db.commit()

    resp = await client.post("/api/auth/refresh")
    assert resp.status_code == 403
    assert resp.json()["code"] == MOBILE_NOT_APPROVED_CODE
    async with SessionLocal() as db:
        tokens = (
            (await db.execute(select(RefreshToken).where(RefreshToken.user_id == user.id))).scalars().all()
        )
        assert tokens and all(t.revoked_at is not None for t in tokens)


async def test_superadmin_mobile_refresh_is_denied_as_the_web_only_case(client):
    """A mobile session whose account is a superadmin must be refused as the
    superadmin case, not as an ungranted one.

    The two 403s drive different screens on the phone: one says "ask a super admin
    for the grant", the other says "the portal is web-only". Reporting the first to a
    superadmin tells them to ask themselves for something no one can grant.

    Setting the role directly is the point, not a shortcut: no endpoint can promote to
    superadmin (`UserUpdate.role` is Literal["admin", "member"], and the update
    endpoint refuses superadmin targets), and the login gate never issues a mobile
    session to one — so out-of-band promotion is the only way in, and the branch is
    still expected to be exact.
    """
    user = await make_user("promoted@x.com", mobile_access=True)
    assert (await client.post("/api/auth/login", json=_mobile_login("promoted@x.com"))).status_code == 200

    async with SessionLocal() as db:
        (await db.get(User, user.id)).role = UserRole.superadmin
        await db.commit()

    resp = await client.post("/api/auth/refresh")
    assert resp.status_code == 403
    assert resp.json()["code"] == SUPERADMIN_MOBILE_DENIED_CODE
    assert "only sign in on the web" in resp.json()["detail"]

    # Still burned, exactly as the ungranted case is: a refused mobile session must
    # not be retryable.
    async with SessionLocal() as db:
        tokens = (
            (await db.execute(select(RefreshToken).where(RefreshToken.user_id == user.id))).scalars().all()
        )
        assert tokens and all(t.revoked_at is not None for t in tokens)


async def test_an_ungranted_mobile_refresh_still_reports_the_member_case(client):
    """The other half of the split: a plain member keeps MOBILE_NOT_APPROVED, so the
    phone still offers the remedy that actually exists for them."""
    user = await make_user("stillmember@x.com", mobile_access=True)
    await client.post("/api/auth/login", json=_mobile_login("stillmember@x.com"))

    async with SessionLocal() as db:
        (await db.get(User, user.id)).mobile_access = False
        await db.commit()

    resp = await client.post("/api/auth/refresh")
    assert resp.status_code == 403
    assert resp.json()["code"] == MOBILE_NOT_APPROVED_CODE


async def test_mobile_session_refresh_keeps_working_while_granted(client):
    await make_user("keepmob@x.com", mobile_access=True)
    await client.post("/api/auth/login", json=_mobile_login("keepmob@x.com"))
    resp = await client.post("/api/auth/refresh")
    assert resp.status_code == 200
    assert (await client.get("/api/auth/me")).status_code == 200

    # The rotated session must still be marked mobile, or the gate stops applying
    # after the first refresh.
    async with SessionLocal() as db:
        live = (
            (await db.execute(select(RefreshToken).where(RefreshToken.revoked_at.is_(None)))).scalars().all()
        )
        assert live and all(t.client == "mobile" for t in live)


async def test_web_sessions_are_recorded_as_web(client):
    await make_user("webclient@x.com")
    await login(client, "webclient@x.com")
    async with SessionLocal() as db:
        tokens = (await db.execute(select(RefreshToken))).scalars().all()
        assert tokens and all(t.client == "web" for t in tokens)


# ---------------------------------------------------------------------------
# Admin portal surface
# ---------------------------------------------------------------------------


async def _superadmin_client(client):
    await make_user("sa@x.com", role=UserRole.superadmin)
    await login(client, "sa@x.com")


async def test_admin_user_row_exposes_mobile_access(client):
    org = await make_org("Grant Org")
    await make_user("m1@x.com", org_id=org.id, mobile_access=True)
    await make_user("m2@x.com", org_id=org.id)
    await _superadmin_client(client)

    rows = (await client.get("/api/admin/users")).json()["data"]
    by_email = {r["email"]: r for r in rows}
    assert by_email["m1@x.com"]["mobile_access"] is True
    assert by_email["m2@x.com"]["mobile_access"] is False


async def test_admin_can_grant_and_revoke_mobile_access(client):
    org = await make_org("Toggle Org")
    target = await make_user("toggle@x.com", org_id=org.id)
    await _superadmin_client(client)

    resp = await client.put(f"/api/admin/users/{target.id}", json={"mobile_access": True})
    assert resp.status_code == 200
    assert resp.json()["mobile_access"] is True

    resp = await client.put(f"/api/admin/users/{target.id}", json={"mobile_access": False})
    assert resp.status_code == 200
    assert resp.json()["mobile_access"] is False


async def test_revoking_via_admin_ends_mobile_session_but_not_web_session(client):
    """The whole point of refresh_tokens.client: one user, two live sessions, only
    the phone gets signed out."""
    org = await make_org("Two Session Org")
    target = await make_user("dual@x.com", org_id=org.id, mobile_access=True)

    # Two independent cookie jars for the same account.
    from httpx import ASGITransport, AsyncClient

    from app.main import app
    from tests.conftest import CSRF

    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as web,
        AsyncClient(transport=transport, base_url="http://test") as phone,
    ):
        web.headers.update(CSRF)
        phone.headers.update(CSRF)
        assert (
            await web.post("/api/auth/login", json={"email": "dual@x.com", "password": "TestPass1234"})
        ).status_code == 200
        assert (await phone.post("/api/auth/login", json=_mobile_login("dual@x.com"))).status_code == 200

        await _superadmin_client(client)
        assert (
            await client.put(f"/api/admin/users/{target.id}", json={"mobile_access": False})
        ).status_code == 200

        # Phone: signed out. 401 rather than 403 because the admin endpoint burns
        # the mobile refresh token outright, so the session is already gone before
        # refresh gets as far as re-checking the grant. (The 403 path covers the
        # other route in: the flag flipped without going through the endpoint.)
        assert (await phone.post("/api/auth/refresh")).status_code == 401
        assert (await phone.get("/api/auth/me")).status_code == 401

        # Web: still live, and still live after rotating.
        assert (await web.post("/api/auth/refresh")).status_code == 200
        assert (await web.get("/api/auth/me")).status_code == 200


async def test_bulk_grant_and_revoke(client):
    org = await make_org("Bulk Org")
    a = await make_user("b1@x.com", org_id=org.id)
    b = await make_user("b2@x.com", org_id=org.id)
    await _superadmin_client(client)

    resp = await client.post(
        "/api/admin/users/bulk-action",
        json={"user_ids": [str(a.id), str(b.id)], "action": "grant_mobile"},
    )
    assert resp.status_code == 200
    async with SessionLocal() as db:
        for uid in (a.id, b.id):
            assert (await db.get(User, uid)).mobile_access is True

    resp = await client.post(
        "/api/admin/users/bulk-action",
        json={"user_ids": [str(a.id), str(b.id)], "action": "revoke_mobile"},
    )
    assert resp.status_code == 200
    async with SessionLocal() as db:
        for uid in (a.id, b.id):
            assert (await db.get(User, uid)).mobile_access is False


async def test_admin_users_mobile_filter(client):
    org = await make_org("Filter Org")
    await make_user("f1@x.com", org_id=org.id, mobile_access=True)
    await make_user("f2@x.com", org_id=org.id)
    await _superadmin_client(client)

    granted = (await client.get("/api/admin/users", params={"mobile": "granted"})).json()
    assert [r["email"] for r in granted["data"]] == ["f1@x.com"]

    denied = (await client.get("/api/admin/users", params={"mobile": "denied"})).json()
    # The superadmin performing the call is excluded from /api/admin/users.
    assert [r["email"] for r in denied["data"]] == ["f2@x.com"]


async def test_admin_create_user_defaults_to_no_mobile_access(client):
    from app.db.models import Department

    org = await make_org("Create Org")
    async with SessionLocal() as db:
        dept = Department(org_id=org.id, name="Ops")
        db.add(dept)
        await db.commit()
        await db.refresh(dept)

    await _superadmin_client(client)
    body = {
        "org_id": str(org.id),
        "dept_id": str(dept.id),
        "email": "fresh@x.com",
        "display_name": "Fresh Hire",
        "password": "TestPass1234",
        "role": "member",
    }
    resp = await client.post("/api/admin/users", json=body)
    assert resp.status_code == 200
    assert resp.json()["mobile_access"] is False

    body["email"] = "fresh2@x.com"
    body["mobile_access"] = True
    resp = await client.post("/api/admin/users", json=body)
    assert resp.status_code == 200
    assert resp.json()["mobile_access"] is True


async def test_org_admin_cannot_grant_mobile_access(client):
    """OrgUpdateUser has no mobile_access field, so the value must be ignored
    rather than silently applied by a permissive model."""
    org = await make_org("No Grant Org")
    await make_user("oadmin@x.com", role=UserRole.org_admin, org_id=org.id)
    target = await make_user("victim@x.com", org_id=org.id)

    await login(client, "oadmin@x.com")
    resp = await client.put(f"/api/org-admin/users/{target.id}", json={"mobile_access": True})
    assert resp.status_code == 200  # the request succeeds; the field is not honoured
    async with SessionLocal() as db:
        assert (await db.get(User, target.id)).mobile_access is False


async def test_me_reports_mobile_access(client):
    await make_user("meflag@x.com", mobile_access=True)
    await login(client, "meflag@x.com")
    assert (await client.get("/api/auth/me")).json()["mobile_access"] is True
