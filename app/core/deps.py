"""FastAPI dependencies: current user, role guards, and the tenant guard.

The tenant guard is the single place multi-tenant isolation is enforced.
Every org-scoped route depends on `TenantContext`; every query made through it
is scoped to the caller's organization. Closing the cross-tenant IDOR class by
construction — not per-endpoint vigilance.
"""

import datetime as dt
import uuid

from fastapi import Depends, HTTPException, Request, WebSocket, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import CodedHTTPException
from app.core.security import ACCESS_COOKIE, WEB_CLIENT, decode_access_token
from app.db.models import Conversation, ConversationParticipant, ConversationType, User, UserRole
from app.db.session import get_db

_credentials_error = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

# The refusal for an account an administrator has reset (batch 73). Both admin
# resets set users.must_change_password and hand the admin the temporary password,
# and nothing read the flag: the account stayed fully usable with a password
# somebody else had been shown. The code is the wire contract both clients branch
# on to enter their forced change-password screen; the sentence is free to change.
# Defined here rather than in api/auth.py so the realtime hub and the messaging
# service can raise and recognise it without importing the API layer.
PASSWORD_CHANGE_REQUIRED_CODE = "PASSWORD_CHANGE_REQUIRED"
PASSWORD_CHANGE_REQUIRED_DETAIL = "You must change your password before continuing."


class PasswordChangeRequiredError(CodedHTTPException):
    """403: the caller authenticated, and must choose a new password before anything else.

    A 403 and not a 401 on purpose. Both clients answer a 401 by refreshing the
    session, and a refresh succeeds for this account — it has to, or the user could
    never reach change-password — so a 401 here would be an endless
    refresh-and-retry loop rather than a prompt.
    """

    def __init__(self) -> None:
        """Build the 403 with the fixed detail and the PASSWORD_CHANGE_REQUIRED code."""
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=PASSWORD_CHANGE_REQUIRED_DETAIL,
            code=PASSWORD_CHANGE_REQUIRED_CODE,
        )


def _token_from_request(request: Request) -> str | None:
    # httpOnly cookie is the canonical transport; Authorization header is
    # accepted for API tooling and tests.
    token = request.cookies.get(ACCESS_COOKIE)
    if token:
        return token
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.removeprefix("Bearer ")
    return None


def _mobile_grant_revoked(payload: dict, user: User) -> bool:
    """True if this token was minted for mobile but the account may no longer use it.

    Checked on every request, not just refresh, so a superadmin revoking mobile
    access ends that user's mobile session now rather than whenever their access
    token happens to expire. Tokens minted before the claim existed have no
    "client" and are treated as web, which is what they were.
    """
    if payload.get("client") != "mobile":
        return False
    return user.role == UserRole.superadmin or not user.mobile_access


def _issued_before_session_epoch(payload: dict, user: User) -> bool:
    """True if this access token predates the last administrator reset of the account.

    An access token is a signed JWT the server cannot revoke, so revoking every
    refresh token — what both resets did — left the target's already-issued access
    tokens working for the rest of their lifetime. Batch 73 refused them only while
    must_change_password was set, so the moment the owner chose a new password a
    token minted before the reset came back to life, long enough to register a push
    subscription that never expires (batch 73 review). The reset now stamps
    users.sessions_valid_after, and a token issued before it is a session that
    reset ended.

    Compared in milliseconds, from the `iat_ms` claim this server mints beside
    `iat`. `iat` alone is whole seconds, and at that precision a token minted in
    the same second as the reset could not be told from one minted just after it:
    either the pre-reset token survived that second, or a sign-in with the
    temporary password moments after the reset was refused. A token from before
    `iat_ms` existed falls back to `iat` in whole seconds, which errs towards
    refusing within the reset's own second — those tokens predate this check, so
    none of them belongs to a session minted after a reset. A token with neither
    counts as issued at 0, i.e. before any reset.

    The realtime hub's revalidation calls this too, on the claims its socket was
    opened with, so a socket and a request carrying the same token always get the
    same answer. It had its own rule, keyed on when the socket connected, and a
    reset landing inside the handshake fell between the two (batch 73 review).
    """
    epoch = user.sessions_valid_after
    if epoch is None:
        return False
    if epoch.tzinfo is None:
        # The column is timezone-aware, but an in-memory value assigned without a
        # zone would otherwise be read as local time by timestamp() below.
        epoch = epoch.replace(tzinfo=dt.UTC)
    try:
        issued_ms = int(payload["iat_ms"])
    except (KeyError, TypeError, ValueError):
        try:
            issued_ms = int(payload.get("iat") or 0) * 1000
        except (TypeError, ValueError):
            issued_ms = 0
    return issued_ms < int(epoch.timestamp() * 1000)


async def _authenticate(
    token: str | None, db: AsyncSession, *, allow_password_change_pending: bool = False
) -> tuple[User, dict]:
    """The account checks behind every authenticated surface, HTTP and WebSocket alike.

    Returns (user, verified claims). Raises `_credentials_error` (401) for a missing,
    invalid or expired token, a deactivated account, a revoked mobile grant or a
    token issued before an administrator's reset ended the account's sessions, and
    `PasswordChangeRequiredError` (403) for an account that must change its password.
    The last two are skipped for the few routes that exist to get out of a reset
    (`allow_password_change_pending`).

    ONE helper on purpose. The WebSocket handshake used to be a second, hand-kept
    copy of these checks, and copies drift — the HTTP path once answered a malformed
    `sub` with a 500 that the WebSocket copy already caught. A new account condition
    added here reaches both transports, which is the point of batch 73's flag: added
    to one copy only, the socket would have kept a reset account receiving messages.

    The password-change gate is checked LAST, so every refusal that existed before it
    keeps its own answer: a deactivated account that also happens to be flagged is
    still told it cannot sign in, not invited to change a password it cannot use.
    """
    if not token:
        raise _credentials_error
    payload = decode_access_token(token)
    if not payload:
        raise _credentials_error
    # `decode_access_token` verifies the signature and the token type, not the
    # shape of the claims, so a token carrying a missing or malformed `sub`
    # reached `uuid.UUID()` unguarded and left this path answering 500 to what is
    # simply a failed authentication. The WebSocket handshake already caught
    # exactly this; the HTTP path is the one that drifted.
    try:
        user = await db.get(User, uuid.UUID(payload["sub"]))
    except (ValueError, KeyError):
        raise _credentials_error from None
    if not user or not user.is_active:
        raise _credentials_error
    if _mobile_grant_revoked(payload, user):
        raise _credentials_error
    # The session epoch (batch 73 review): a token issued before the last admin
    # reset belongs to a session that reset ended, so it is a failed authentication
    # — 401, which both clients answer with a refresh that fails, because the reset
    # revoked the refresh token too, and so they sign out. Ahead of the
    # password-change gate, so a pre-reset token is told its session is over rather
    # than invited to change a password. Relaxed on the un-gated routes ONLY while
    # the change is still pending, so a device signed in before the reset can
    # finish the change it is being asked to make (me, change-password, logout and
    # push unsubscribe accept it, and change-password re-issues it a session of its
    # own). Once the flag clears the exemption has no purpose, and kept, it let a
    # token copied before the reset go on reading the full profile from /me for the
    # rest of its life (CodeRabbit, review of 1fbc1de).
    pending_change = allow_password_change_pending and user.must_change_password
    if not pending_change and _issued_before_session_epoch(payload, user):
        raise _credentials_error
    if user.must_change_password and not allow_password_change_pending:
        raise PasswordChangeRequiredError()
    return user, payload


async def _authenticate_request(
    request: Request, db: AsyncSession, *, allow_password_change_pending: bool = False
) -> User:
    """`_authenticate` for an HTTP request, keeping the verified claims on request.state.

    Stashed so `session_client` can read the claims this request was authenticated
    with instead of decoding the token a second time (batch 73 review). The second
    decode ran after change-password's two bcrypt rounds, and a token that expired
    in between decoded to nothing, fell back to "web", and handed a phone a WEB
    session — one the mobile-grant checks never look at.
    """
    user, claims = await _authenticate(
        _token_from_request(request), db, allow_password_change_pending=allow_password_change_pending
    )
    request.state.auth_claims = claims
    return user


async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    """The authenticated user behind an HTTP request; the dependency every gated route uses.

    Answers 401 for a failed authentication, including a token issued before the account's last reset, and 403
    PASSWORD_CHANGE_REQUIRED while must_change_password is set.
    """
    return await _authenticate_request(request, db)


async def get_current_user_pending_password_change(
    request: Request, db: AsyncSession = Depends(get_db)
) -> User:
    """`get_current_user` without the must-change-password gate or the session epoch.

    Every other check stands: a deactivated account or a revoked mobile grant is
    refused here exactly as everywhere else.

    For the handful of routes a reset account must still reach to get out of that
    state, and nothing else: logout, me (the client needs the flag to know which
    screen to show), change-password itself, and DELETE /api/notifications/subscribe
    (signing out unsubscribes the browser first, and a refusal there would leave a
    push subscription behind for an account that is meant to be signing out). The
    session epoch is skipped for the same reason (batch 73 review): a device signed
    in before the reset can still reach these four with its old access token, so it
    can finish the change — and change-password then re-issues it a fresh session.
    tests/test_must_change_password.py walks every route and pins that exact set, so
    a new route cannot pick this up by accident.
    """
    return await _authenticate_request(request, db, allow_password_change_pending=True)


def session_client(request: Request) -> str:
    """The client the access token on this request was minted for, "web" if unknown.

    Only meaningful after an auth dependency has accepted the token: it reads the
    signed claims that dependency verified, never anything the caller says about
    itself. NOT a fresh decode (batch 73 review): change-password calls this after
    two bcrypt rounds, by which time the token may have expired, and an expired
    token decodes to nothing — so a phone finishing the change was re-issued a web
    session, outside every mobile-grant check. The claims were verified when the
    request was authenticated, which is the moment that decides whose session it is.
    """
    claims = getattr(request.state, "auth_claims", None) or {}
    return str(claims.get("client") or WEB_CLIENT)


async def get_current_user_ws(websocket: WebSocket, db: AsyncSession) -> tuple[User, int, str, dict] | None:
    """Cookie-authenticated WebSocket handshake.

    Returns (user, token_exp, client, claims) so the connection can enforce the
    access-token lifetime and force a re-auth on expiry, so its periodic liveness
    check knows whether to re-verify the mobile grant, and so that check can apply
    the session epoch to the token itself. The claims are the ones verified here,
    and the hub keeps them for the life of the socket (batch 73 review): it used to
    compare the epoch with the moment the socket connected instead, so a reset
    landing between this check and that moment left the socket newer than the
    epoch, and it survived once the owner changed the password while HTTP refused
    the very same token. The query-param token fallback exists only outside
    production.

    Returns None for every failed authentication, which the hub closes 4001 and the
    clients answer by refreshing — including a token issued before an administrator's
    reset ended the account's sessions, whose refresh then fails because the reset
    revoked that too (batch 73 review). Raises `PasswordChangeRequiredError` instead for an
    account that authenticated but must change its password: that is not a failed
    authentication, a refresh would succeed and reconnect into the same refusal, so
    the hub has to be able to tell the two apart and close with its own code."""
    from app.core.config import get_settings

    token = websocket.cookies.get(ACCESS_COOKIE)
    if token is None and not get_settings().is_production:
        token = websocket.query_params.get("token")
    try:
        user, payload = await _authenticate(token, db)
    except PasswordChangeRequiredError:
        raise
    except HTTPException:
        return None
    return user, int(payload.get("exp", 0)), str(payload.get("client") or WEB_CLIENT), payload


async def require_superadmin(user: User = Depends(get_current_user)) -> User:
    if user.role != UserRole.superadmin:
        raise HTTPException(status_code=403, detail="Super admin access required")
    return user


async def require_org_admin(user: User = Depends(get_current_user)) -> User:
    if user.role not in (UserRole.superadmin, UserRole.org_admin):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


class TenantContext:
    """The caller's identity + org scope. All org-scoped data access goes
    through helpers that take this context, so org filtering can't be forgotten."""

    def __init__(self, user: User, db: AsyncSession):
        self.user = user
        self.db = db

    @property
    def org_id(self) -> uuid.UUID | None:
        return self.user.org_id

    @property
    def is_superadmin(self) -> bool:
        return self.user.role == UserRole.superadmin

    def check_org(self, org_id: uuid.UUID | None) -> None:
        """Assert an object belongs to the caller's org (superadmin bypasses)."""
        if self.is_superadmin:
            return
        if org_id is None or self.org_id is None or org_id != self.org_id:
            raise HTTPException(status_code=404, detail="Not found")

    async def require_membership(self, conversation_id: uuid.UUID) -> Conversation:
        """Return the conversation iff the caller is a participant. 404 otherwise
        (existence is not revealed across tenants)."""
        conv = await self.db.get(Conversation, conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        membership = await self.db.get(ConversationParticipant, (conversation_id, self.user.id))
        if membership is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        # Org check lives here rather than in each route: every participant row the
        # API can create today already matches the conversation's org, so this is
        # defence in depth — but the message read paths checked it and the pin /
        # mute / delete / clear / export routes did not, and only enforcing it in
        # the shared guard keeps a future route from drifting the same way.
        # cross_org conversations belong to no single org and are exempt by design.
        if conv.type != ConversationType.cross_org and conv.org_id != self.user.org_id:
            raise HTTPException(status_code=403, detail="Access denied")
        return conv

    async def org_user(self, user_id: uuid.UUID) -> User:
        """Load a user constrained to the caller's org (superadmin: any user)."""
        target = await self.db.get(User, user_id)
        if target is None:
            raise HTTPException(status_code=404, detail="User not found")
        if not self.is_superadmin and target.org_id != self.org_id:
            raise HTTPException(status_code=404, detail="User not found")
        return target

    async def org_users_query(self):
        stmt = select(User).where(User.is_active.is_(True))
        if not self.is_superadmin:
            stmt = stmt.where(User.org_id == self.org_id)
        return stmt


async def get_tenant(
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> TenantContext:
    return TenantContext(user, db)
