"""Auth hardening: cookies, rotation, revocation, rate limit, password policy."""

import jwt
from sqlalchemy import select

from app.core.config import get_settings
from app.core.security import ACCESS_COOKIE, JWT_ALGORITHM, REFRESH_COOKIE
from app.db.models import RefreshToken, User, UserRole
from app.db.session import SessionLocal
from app.realtime.redis_bus import get_redis
from tests.conftest import login, make_user


async def test_login_sets_httponly_cookies_and_returns_user(client):
    await make_user("dana@x.com", display_name="Dana")
    resp = await client.post("/api/auth/login", json={"email": "dana@x.com", "password": "TestPass1234"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"user"}  # no tokens in the body — cookies only
    assert body["user"]["name"] == "Dana"
    set_cookie = " ".join(resp.headers.get_list("set-cookie"))
    assert ACCESS_COOKIE in set_cookie and REFRESH_COOKIE in set_cookie
    assert set_cookie.count("HttpOnly") == 2


async def test_wrong_password_and_unknown_email_are_401(client):
    await make_user("erin@x.com")
    for payload in (
        {"email": "erin@x.com", "password": "nope-nope-1"},
        {"email": "ghost@x.com", "password": "TestPass1234"},
    ):
        resp = await client.post("/api/auth/login", json=payload)
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid email or password"


async def test_email_login_is_case_insensitive(client):
    await make_user("frank@x.com")
    resp = await client.post("/api/auth/login", json={"email": "FRANK@X.com", "password": "TestPass1234"})
    assert resp.status_code == 200


async def test_refresh_rotates_and_old_token_is_single_use(client):
    await make_user("gina@x.com")
    await login(client, "gina@x.com")
    old_refresh = client.cookies.get(REFRESH_COOKIE)

    resp = await client.post("/api/auth/refresh")
    assert resp.status_code == 200
    assert client.cookies.get(REFRESH_COOKIE) != old_refresh

    # A consumed token buys at most one replay, and only while the successor it
    # carried is still unspent — the bounded delivery grace, covered in full by
    # test_refresh_rotation.py. Once the successor has been used the client
    # demonstrably received it, so replaying its predecessor is reuse and fails.
    assert (await client.post("/api/auth/refresh")).status_code == 200
    resp = await client.post("/api/auth/refresh", headers={"Cookie": f"{REFRESH_COOKIE}={old_refresh}"})
    assert resp.status_code == 401


async def test_deactivated_user_cannot_refresh(client):
    user = await make_user("hank@x.com")
    await login(client, "hank@x.com")

    async with SessionLocal() as db:
        target = await db.get(User, user.id)
        target.is_active = False
        await db.commit()

    resp = await client.post("/api/auth/refresh")
    assert resp.status_code == 401
    # and the token was revoked server-side, not just rejected
    async with SessionLocal() as db:
        tokens = (
            (await db.execute(select(RefreshToken).where(RefreshToken.user_id == user.id))).scalars().all()
        )
        assert tokens and all(t.revoked_at is not None for t in tokens)


async def test_deactivated_user_access_token_stops_working_immediately(client):
    user = await make_user("iris@x.com")
    await login(client, "iris@x.com")
    assert (await client.get("/api/auth/me")).status_code == 200

    async with SessionLocal() as db:
        target = await db.get(User, user.id)
        target.is_active = False
        await db.commit()

    # get_current_user re-checks is_active in the DB on every request
    assert (await client.get("/api/auth/me")).status_code == 401


async def test_superadmin_gets_refresh_cookie_too(client):
    await make_user("root@x.com", role=UserRole.superadmin)
    resp = await client.post("/api/auth/login", json={"email": "root@x.com", "password": "TestPass1234"})
    set_cookie = " ".join(resp.headers.get_list("set-cookie"))
    assert REFRESH_COOKIE in set_cookie  # the Mongo build's superadmin 15-min logout bug

    await client.post("/api/auth/refresh")
    assert (await client.get("/api/auth/me")).status_code == 200


async def test_login_rate_limiter_fires(client):
    await make_user("judy@x.com")
    statuses = []
    for _ in range(12):
        resp = await client.post("/api/auth/login", json={"email": "judy@x.com", "password": "wrong-pass-1"})
        statuses.append(resp.status_code)
    assert 429 in statuses
    assert statuses[-1] == 429


async def test_rate_limit_counter_always_carries_a_ttl(client):
    """A counter that loses its TTL never rolls over: it climbs past the limit and
    then 429s that IP out of sign-in permanently, with Retry-After promising a
    retry in one second forever. INCR and EXPIRE were two round-trips, so a
    connection lost between them left exactly that. They now go in one MULTI/EXEC,
    and EXPIRE NX also repairs a counter that already lost its TTL."""
    await make_user("owen@x.com")
    redis = get_redis()

    await client.post("/api/auth/login", json={"email": "owen@x.com", "password": "wrong-pass-1"})
    keys = [k async for k in redis.scan_iter(match="ratelimit:login:*")]
    assert keys, "the login limiter recorded no counter"
    for key in keys:
        assert await redis.ttl(key) > 0

    # Strip the TTL to stand in for the lost EXPIRE, then show the next request
    # restores it rather than counting up against an immortal key.
    for key in keys:
        await redis.persist(key)
        assert await redis.ttl(key) == -1
    await client.post("/api/auth/login", json={"email": "owen@x.com", "password": "wrong-pass-1"})
    for key in keys:
        assert await redis.ttl(key) > 0


async def test_access_token_with_unusable_subject_is_401_not_500(client):
    """decode_access_token verifies the signature and the token type, not the shape
    of the claims, so a signed token whose `sub` is missing or not a UUID reached
    uuid.UUID() unguarded and answered 500 to what is only a failed
    authentication.

    The two guarded cases are the reachable ones: a `sub` that is a string but
    not a UUID (ValueError) and an absent `sub` (KeyError). The non-string cases
    below never reach uuid.UUID() at all — PyJWT 2.10 rejects a numeric, null or
    structured `sub` inside jwt.decode with InvalidSubjectError, which is a
    PyJWTError and so already returns None from decode_access_token. They are
    here to pin that down: if the pin ever moves below 2.10 that validation
    disappears, uuid.UUID() starts seeing those values, and TypeError /
    AttributeError become live 500s that deps.py does not catch."""
    secret = get_settings().secret_key
    for claims in (
        {"sub": "not-a-uuid", "type": "access"},
        {"type": "access"},
        {"sub": None, "type": "access"},
        {"sub": 12345, "type": "access"},
        {"sub": ["not", "a", "uuid"], "type": "access"},
    ):
        token = jwt.encode(claims, secret, algorithm=JWT_ALGORITHM)
        resp = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401, f"{claims} answered {resp.status_code}"
        assert resp.json()["detail"] == "Not authenticated"


async def test_csrf_header_required_for_cookie_mutations(client):
    await make_user("kyle@x.com", role=UserRole.superadmin)
    await login(client, "kyle@x.com")
    resp = await client.post(
        "/api/admin/organizations",
        json={"name": "No CSRF Org"},
        headers={"X-Requested-With": ""},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Missing CSRF header"


async def test_password_policy_enforced_on_change(client):
    await make_user("lena@x.com")
    await login(client, "lena@x.com")
    resp = await client.post(
        "/api/auth/change-password",
        json={"current_password": "TestPass1234", "new_password": "short1"},
    )
    assert resp.status_code == 400
    resp = await client.post(
        "/api/auth/change-password",
        json={"current_password": "TestPass1234", "new_password": "LongEnoughPass99"},
    )
    assert resp.status_code == 200


async def test_password_longer_than_bcrypt_can_hash_is_rejected(client):
    """bcrypt hashes only the first 72 bytes and discards the rest silently, so
    without this cap a long password and its own 72-byte prefix are one credential."""
    await make_user("mira@x.com")
    await login(client, "mira@x.com")
    resp = await client.post(
        "/api/auth/change-password",
        json={"current_password": "TestPass1234", "new_password": "A1" + "x" * 71},
    )
    assert resp.status_code == 400
    assert "72 bytes" in resp.json()["detail"]
    # The 72-byte boundary itself is still allowed.
    resp = await client.post(
        "/api/auth/change-password",
        json={"current_password": "TestPass1234", "new_password": "A1" + "x" * 70},
    )
    assert resp.status_code == 200


async def test_change_password_is_rate_limited(client):
    """This route verifies current_password, so it is an online guessing surface
    and is throttled like the admin reset it mirrors."""
    await make_user("nate@x.com")
    await login(client, "nate@x.com")
    statuses = []
    for _ in range(8):
        resp = await client.post(
            "/api/auth/change-password",
            json={"current_password": "WrongPass1234", "new_password": "LongEnoughPass99"},
        )
        statuses.append(resp.status_code)
    assert 401 in statuses  # the guesses themselves are rejected
    assert 429 in statuses  # and the limiter cuts the attempt off
    assert statuses[-1] == 429
