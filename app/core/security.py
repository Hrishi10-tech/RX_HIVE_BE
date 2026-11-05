"""Password hashing, JWT minting/verification, refresh-token helpers."""

import asyncio
import datetime as dt
import hashlib
import re
import secrets
import uuid

import bcrypt
import jwt

from app.core.config import get_settings

JWT_ALGORITHM = "HS256"

BCRYPT_ROUNDS = 12

# bcrypt hashes at most the first 72 bytes of a password and silently discards
# the rest, so without a cap a 100-character password and its own 72-byte prefix
# are the same credential — the extra length is security theatre. Capping at the
# boundary keeps every stored hash valid, which pre-hashing (the other usual fix)
# would not: that would invalidate every password already on file.
BCRYPT_MAX_PASSWORD_BYTES = 72

ACCESS_COOKIE = "rx_access"
REFRESH_COOKIE = "rx_refresh"

# Which client a session belongs to. Lives here, in a core module, rather than in
# api/auth.py so the realtime hub can compare against it without the realtime
# layer importing the API layer.
WEB_CLIENT = "web"
MOBILE_CLIENT = "mobile"


def _hash_password_sync(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode()


def _verify_password_sync(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


# bcrypt at cost 12 is ~170ms of uninterruptible CPU. Called straight from an
# `async def` handler that is 170ms the event loop cannot serve anyone else, so a
# burst of sign-ins stalls every unrelated request — including WebSocket pings and
# the health probe — behind the queue. Both helpers therefore run in the default
# thread pool, where the GIL is released for the duration of the C hash.
async def hash_password(password: str) -> str:
    return await asyncio.to_thread(_hash_password_sync, password)


async def verify_password(password: str, password_hash: str) -> bool:
    return await asyncio.to_thread(_verify_password_sync, password, password_hash)


class PasswordPolicyError(ValueError):
    pass


def enforce_password_policy(password: str) -> None:
    """Raise PasswordPolicyError unless the password meets the org policy.

    Called on every path that sets a password: registration, admin create/reset,
    self-service change.
    """
    settings = get_settings()
    if len(password) < settings.password_min_length:
        raise PasswordPolicyError(f"Password must be at least {settings.password_min_length} characters")
    if len(password.encode()) > BCRYPT_MAX_PASSWORD_BYTES:
        # Rejected rather than truncated: silently accepting a longer password
        # and only honouring its prefix is the behaviour this guards against.
        raise PasswordPolicyError(f"Password must be at most {BCRYPT_MAX_PASSWORD_BYTES} bytes")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise PasswordPolicyError("Password must contain both letters and numbers")


def create_access_token(
    user_id: uuid.UUID,
    role: str,
    org_id: uuid.UUID | None,
    client: str = WEB_CLIENT,
) -> str:
    """Mint a signed access JWT for a user, valid for access_token_minutes.

    It carries the role, the org, the client it was minted for, and `iat_ms` beside `iat` so the session epoch
    can be compared to the millisecond.
    """
    settings = get_settings()
    now = dt.datetime.now(dt.UTC)
    payload = {
        "sub": str(user_id),
        "role": role,
        "org_id": str(org_id) if org_id else None,
        "iat": now,
        # The same instant in milliseconds. `iat` is whole seconds (PyJWT encodes
        # a NumericDate as an integer), which is too coarse for the session epoch an
        # admin reset stamps (core/deps.py): a token minted in the same second as
        # the reset could not be told from one minted just after it (batch 73
        # review). Unregistered claims are ignored by every JWT consumer.
        "iat_ms": int(now.timestamp() * 1000),
        # Which client this token was minted for. Signed, so it is trustworthy —
        # which is what lets every request re-check the mobile grant instead of
        # only the refresh path, so a revoked grant does not stay usable for the
        # remaining lifetime of an already-issued access token.
        "client": client,
        "exp": now + dt.timedelta(minutes=settings.access_token_minutes),
        "type": "access",
    }
    return jwt.encode(payload, settings.secret_key, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    if payload.get("type") != "access":
        return None
    return payload


def new_refresh_token() -> tuple[str, str]:
    """Return (raw_token, sha256_hash). Only the hash is stored server-side."""
    raw = secrets.token_urlsafe(48)
    return raw, hash_refresh_token(raw)


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()
