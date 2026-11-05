"""Regression tests for the closed security-review findings."""

import datetime as dt
import ipaddress
import socket
import uuid

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.security import REFRESH_COOKIE
from app.services.push import validate_push_endpoint
from app.utils import now_utc
from tests.conftest import login, make_user


def test_production_rejects_placeholder_secrets():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            environment="production",
            secret_key="change-me-openssl-rand-hex-32",
            livekit_api_secret="x" * 40,
            s3_secret_key="y" * 40,
        )


def test_production_rejects_short_secret():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            environment="production",
            secret_key="tooshort",
            livekit_api_secret="x" * 40,
            s3_secret_key="y" * 40,
        )


def test_production_accepts_strong_secrets_and_forces_secure_cookies():
    s = Settings(
        _env_file=None,
        environment="production",
        secret_key="a" * 48,
        livekit_api_secret="b" * 40,
        s3_secret_key="c" * 40,
    )
    assert s.cookies_secure is True
    assert s.is_production is True


def test_dev_uses_insecure_cookies_by_default():
    assert Settings(_env_file=None).cookies_secure is False


def test_explicit_cookie_secure_override_wins():
    assert (
        Settings(
            _env_file=None,
            cookie_secure=False,
            environment="production",
            secret_key="a" * 48,
            livekit_api_secret="b" * 40,
            s3_secret_key="c" * 40,
        ).cookies_secure
        is False
    )


@pytest.mark.parametrize(
    "endpoint,ok",
    [
        ("https://push.example.test/fcm/send/abc", True),
        ("http://push.example.test/fcm/send/abc", False),  # not https
        ("https://localhost/x", False),  # loopback
        ("https://127.0.0.1/x", False),  # loopback ip
        ("https://10.0.0.5/x", False),  # private
        ("https://169.254.169.254/x", False),  # link-local (cloud metadata)
        ("ftp://example.com/x", False),
        ("not a url", False),
    ],
)
def test_push_endpoint_ssrf_validation(endpoint, ok, monkeypatch):
    """The resolver is stubbed, and the positive case used to depend on live DNS.

    This asserted validate_push_endpoint("https://fcm.googleapis.com/...") is True,
    which needs a real lookup of a real Google hostname. When DNS is slow or
    unavailable the function correctly returns False and the test fails — reported
    as an SSRF-validation regression, pointing at the code rather than at the
    network. Observed twice in one sitting on 2026-09-03.

    Every negative case still exercises the real logic. Literal IPs ARE resolved —
    validate_push_endpoint calls getaddrinfo for them too — so the stub returns
    each literal as its own answer, exactly as the real resolver does, and
    is_public_address is what rejects them. Raising for literals instead would have
    made them fail on the lookup rather than on the address, which is what the
    first version of this stub did. The scheme and unparseable-URL rejections
    happen before any lookup at all.
    """

    def _resolve(host, *args, **kwargs):
        # Literal IPs resolve to THEMSELVES, exactly as the real resolver does.
        #
        # Raising for them instead made every rejected literal — loopback,
        # private, link-local — return False because the lookup failed, not
        # because is_public_address refused the address. A regression that started
        # permitting 127.0.0.1 would have passed this test unnoticed.
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            return [(family, socket.SOCK_STREAM, 6, "", (host, 443))]
        if host == "push.example.test":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        raise socket.gaierror(f"unexpected lookup of {host!r} in a test")

    monkeypatch.setattr(socket, "getaddrinfo", _resolve)
    assert validate_push_endpoint(endpoint) is ok


async def test_push_subscribe_rejects_ssrf_endpoint(client, two_orgs_with_users):
    await login(client, "alice@a.com")
    resp = await client.post(
        "/api/notifications/subscribe",
        json={"endpoint": "https://127.0.0.1:9200/push", "keys": {"p256dh": "x", "auth": "y"}},
    )
    assert resp.status_code == 400


async def test_change_password_revokes_other_sessions(client):
    """Every OTHER session dies; the one doing the changing survives.

    This asserted that *every* token was revoked, the caller's included, which is
    what the endpoint used to do — so the test's name and its assertion disagreed
    and the surprise sign-out that behaviour caused was locked in rather than
    caught. The caller's session surviving is the contract the name describes.
    """
    from sqlalchemy import select

    from app.core.security import hash_refresh_token
    from app.db.models import RefreshToken
    from app.db.session import SessionLocal

    user = await make_user("carla@x.com")
    await login(client, "carla@x.com")
    # A second, older session that must not survive the password change.
    stale = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token("some-other-devices-token"),
        client="web",
        expires_at=now_utc() + dt.timedelta(days=30),
    )
    async with SessionLocal() as db:
        db.add(stale)
        await db.commit()

    current_raw = client.cookies.get(REFRESH_COOKIE)
    assert current_raw
    resp = await client.post(
        "/api/auth/change-password",
        json={"current_password": "TestPass1234", "new_password": "BrandNewPass99"},
    )
    assert resp.status_code == 200

    current_hash = hash_refresh_token(current_raw)
    async with SessionLocal() as db:
        tokens = (
            (await db.execute(select(RefreshToken).where(RefreshToken.user_id == user.id))).scalars().all()
        )
    by_hash = {t.token_hash: t for t in tokens}
    assert by_hash[current_hash].revoked_at is None, "the caller's own session was revoked"
    assert all(t.revoked_at is not None for h, t in by_hash.items() if h != current_hash)

    # And the surviving session still works — no silent sign-out later.
    assert (await client.post("/api/auth/refresh")).status_code == 200


async def test_push_subscribe_unsubscribe_round_trip(client, two_orgs_with_users, monkeypatch):
    """The frontend's subscribe/unsubscribe pair must match the routes the API
    actually exposes — a mismatch here silently 404s in the browser."""
    from sqlalchemy import select

    from app.db.models import PushSubscription
    from app.db.session import SessionLocal

    await login(client, "alice@a.com")
    # validate_push_endpoint stubbed rather than resolved, matching
    # tests/test_rate_limit_identity.py. The subject here is the subscribe and
    # unsubscribe round trip, not whether Google's DNS answers.
    monkeypatch.setattr("app.services.push.validate_push_endpoint", lambda _endpoint: True)
    endpoint = "https://push.example.test/fcm/send/round-trip-test"

    resp = await client.post(
        "/api/notifications/subscribe",
        json={"endpoint": endpoint, "keys": {"p256dh": "x", "auth": "y"}},
    )
    assert resp.status_code == 200, resp.text

    async with SessionLocal() as db:
        rows = (await db.execute(select(PushSubscription))).scalars().all()
        assert [r.endpoint for r in rows] == [endpoint]

    resp = await client.request("DELETE", "/api/notifications/subscribe", json={"endpoint": endpoint})
    assert resp.status_code == 200, resp.text

    async with SessionLocal() as db:
        assert (await db.execute(select(PushSubscription))).scalars().all() == []


async def test_require_superadmin_refuses_everyone_else(client, two_orgs_with_users):
    """The guard behind 31 admin routes had no negative test of its own.

    Every existing admin-portal test signs in AS a superadmin, so the suite proved
    the routes work and never once proved they are shut. A guard that returned the
    user unconditionally would have passed the whole suite.
    """
    from app.db.models import UserRole

    org_a = two_orgs_with_users["org_a"]
    await make_user("orgadmin@a.com", role=UserRole.org_admin, org_id=org_a.id)

    # A plain member and an org admin are both refused, and told 403 (they are
    # authenticated — this is authorization), not 401.
    for email in ("alice@a.com", "orgadmin@a.com"):
        await login(client, email)
        resp = await client.get("/api/admin/organizations")
        assert resp.status_code == 403, f"{email} reached a superadmin route: {resp.text}"
        assert resp.json()["detail"] == "Super admin access required"

    # And the guard is not simply refusing everyone.
    await make_user("root@x.com", role=UserRole.superadmin)
    await login(client, "root@x.com")
    assert (await client.get("/api/admin/organizations")).status_code == 200


async def test_require_org_admin_refuses_a_plain_member(client, two_orgs_with_users):
    """Same gap one tier down: the org-admin portal guard."""
    await login(client, "alice@a.com")
    resp = await client.get("/api/org-admin/users")
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Admin access required"


async def test_deactivating_a_conversation_revokes_reading_its_history(client, two_orgs_with_users):
    """Archiving or deleting must cut off the history, not just hide the row.

    Deactivation leaves the ConversationParticipant rows in place — it is the only
    revocation lever a superadmin has for a cross-org group — and nothing on the
    read side checked the flag. The conversation disappeared from the list while
    every message in it stayed readable to anyone who had kept the id.
    """
    from app.db.models import Conversation
    from app.db.session import SessionLocal

    await login(client, "alice@a.com")
    conv_id = (
        await client.post(
            "/api/conversations/direct", json={"participant_id": str(two_orgs_with_users["bob"].id)}
        )
    ).json()["_id"]
    assert (
        await client.post(f"/api/conversations/{conv_id}/messages", json={"content": "secret"})
    ).status_code == 200
    assert (await client.get(f"/api/conversations/{conv_id}/messages")).status_code == 200

    async with SessionLocal() as db:
        conv = await db.get(Conversation, uuid.UUID(conv_id))
        conv.is_active = False
        await db.commit()

    # Every read surface, not just the list.
    assert (await client.get(f"/api/conversations/{conv_id}/messages")).status_code == 404
    search = await client.post(f"/api/conversations/{conv_id}/messages/search", json={"q": "secret"})
    assert search.status_code == 404
    assert (await client.get(f"/api/conversations/{conv_id}/starred")).status_code == 404
    assert (await client.get(f"/api/conversations/{conv_id}/pinned")).status_code == 404
    assert (await client.get(f"/api/conversations/{conv_id}/media?type=image")).status_code == 404
