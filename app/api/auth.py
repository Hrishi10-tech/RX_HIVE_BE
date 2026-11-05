"""Auth: httpOnly-cookie sessions with a persisted, revocable refresh table.

Changes vs the Mongo build (all deliberate, coordinated with the frontend):
- Tokens ride in httpOnly cookies, never in response bodies or localStorage.
- Refresh tokens are opaque, hashed at rest, rotated on every refresh, and
  revoked on logout/deactivation. is_active is re-checked on refresh.
- Superadmin sessions refresh like everyone else (fixes the 15-min logout bug).
- The rate limiter is a dependency wired here — not dead code in main.py.
- must_change_password is enforced (batch 73). Both admin resets set it and show
  the admin the temporary password, and nothing read it, so a reset account stayed
  fully usable with a password someone else knew. Every authenticated route now
  answers 403 PASSWORD_CHANGE_REQUIRED until it is cleared; login, refresh, me,
  logout and change-password stay reachable so the user can get out of that state.
  change-password refuses a new password equal to the current one, and re-issues
  the caller's session when the one it is using was revoked by the reset.
- A reset ends every session from before it (batch 73 review). The reset stamps
  users.sessions_valid_after: core/deps.py refuses an access token issued before
  it, and /refresh refuses a refresh row created before it. Login, logout,
  change-password and both resets lock the users row before any refresh_tokens
  row, in that order everywhere, so a sign-in racing a reset cannot come out of it
  holding a session the reset never saw.

Rotation is now delivery-safe as well as single-use. Revoking the presented token
commits before its replacement can reach the client, so a lost response used to
leave a healthy client holding a token the server had already spent, and its next
refresh signed the user out of a session that was fine. A rotated token may now be
replayed once, within a short grace window, while the successor it never received
is still unused. Every other reuse of a ROTATED token is treated as a stolen
cookie and burns that client's whole session family — stricter than failing the
one token, which is what the previous build did.

A revoked token that was never rotated is a different thing, and is simply refused
(batch 73 review). Rotation is the only thing that records a successor, so a
revoked row without one was ended on purpose — by a reset, a logout, a password
change, a deactivation — and presenting it is what a device signed out from
somewhere else does on its next refresh. Burning the family for it signed the user
out of the session they had just made: the one they signed in to with the
temporary password, or the one change-password had just re-issued.
"""

import datetime as dt
import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.deps import get_current_user_pending_password_change, session_client
from app.core.errors import CodedHTTPException
from app.core.rate_limit import login_limiter, password_limiter, refresh_limiter
from app.core.security import (
    ACCESS_COOKIE,
    MOBILE_CLIENT,
    REFRESH_COOKIE,
    WEB_CLIENT,
    create_access_token,
    hash_refresh_token,
    new_refresh_token,
    verify_password,
)
from app.db.models import Department, Organization, RefreshToken, User, UserRole
from app.db.session import get_db
from app.services import presence
from app.utils import iso_z, now_utc

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# A fixed bcrypt hash used to burn constant time on unknown-email logins.
_DUMMY_HASH = "$2b$12$C6UzMDM.H6dfI/f/IKcEeO0000000000000000000000000000000000"


# Distinct from the generic 401 on purpose. A member whose account is fine but
# who has not been granted mobile access must not be told "invalid password" —
# they would retype it forever. These are 403s: authentication succeeded, the
# client is what is refused.
SUPERADMIN_MOBILE_DENIED = "Super admin accounts can only sign in on the web app"
MOBILE_NOT_APPROVED = (
    "Mobile access has not been enabled for this account. Ask your super admin to approve mobile sign-in."
)

# The sentences above are what the user reads; these are what the client branches
# on. The two denials need different screens — one offers a remedy, the other says
# the portal is web-only and always will be — and the sentences cannot tell them
# apart reliably: MOBILE_NOT_APPROVED names the super admin as the remedy, so any
# test for that phrase matches both. Reword the prose freely; these are the wire
# contract, decoded verbatim by the iOS client (`MobileDenialKind`).
SUPERADMIN_MOBILE_DENIED_CODE = "SUPERADMIN_MOBILE_DENIED"
MOBILE_NOT_APPROVED_CODE = "MOBILE_NOT_APPROVED"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    # Which client is signing in. Absent means web, so the existing frontend and
    # every API consumer keep working untouched.
    client: Literal["web", "mobile"] = WEB_CLIENT


def _assert_mobile_allowed(user: User) -> None:
    """Gate a mobile sign-in. Raises 403 unless this account may use mobile.

    Called on login AND on refresh, so a revoked grant ends the session at the
    next refresh instead of lasting as long as the refresh token would. The
    superadmin exclusion is unconditional and not overridable by the flag: the
    admin portal has no mobile UI, and a phone is the wrong place to hold the
    keys to every tenant.
    """
    if user.role == UserRole.superadmin:
        raise CodedHTTPException(
            status_code=403, detail=SUPERADMIN_MOBILE_DENIED, code=SUPERADMIN_MOBILE_DENIED_CODE
        )
    if not user.mobile_access:
        raise CodedHTTPException(status_code=403, detail=MOBILE_NOT_APPROVED, code=MOBILE_NOT_APPROVED_CODE)


def wire_role(role: UserRole) -> str:
    # The frontend contract uses "admin" for org admins.
    return "admin" if role == UserRole.org_admin else role.value


def _user_payload(user: User) -> dict:
    """The user object login, refresh and me return, with must_change_password for every role.

    Organization and department ids are included for every role but superadmin.
    """
    payload = {
        "id": str(user.id),
        "email": user.email,
        "name": user.display_name,
        "role": wire_role(user.role),
        # Every role, superadmin included, and on login, refresh and me alike: this
        # is how a client learns to show the change-password screen before it makes
        # a request that would be refused with PASSWORD_CHANGE_REQUIRED (batch 73).
        "must_change_password": bool(user.must_change_password),
    }
    if user.role != UserRole.superadmin:
        payload["org_id"] = str(user.org_id) if user.org_id else None
        payload["dept_id"] = str(user.dept_id) if user.dept_id else None
    return payload


def set_auth_cookies(response: Response, access: str, refresh: str) -> None:
    settings = get_settings()
    common = {
        "httponly": True,
        "secure": settings.cookies_secure,
        "samesite": "lax",
        "domain": settings.cookie_domain,
        "path": "/",
    }
    response.set_cookie(ACCESS_COOKIE, access, max_age=settings.access_token_minutes * 60, **common)
    response.set_cookie(REFRESH_COOKIE, refresh, max_age=settings.refresh_token_days * 86400, **common)


def clear_auth_cookies(response: Response) -> None:
    settings = get_settings()
    for name in (ACCESS_COOKIE, REFRESH_COOKIE):
        response.delete_cookie(name, domain=settings.cookie_domain, path="/")


async def lock_user_row(db: AsyncSession, user_id: uuid.UUID) -> User | None:
    """Lock one users row for the rest of the transaction, and return it as it is now.

    The lock every path that both decides on an account's credentials and writes
    its sessions takes FIRST — login, logout and change-password here, and both
    admin resets — so they serialise against each other on the account rather than
    racing (batch 73 review). Always the users row before any refresh_tokens row
    those paths lock, so two of them can wait on each other but never deadlock.
    /refresh takes no explicit users lock at all, for the same reason.

    FOR NO KEY UPDATE, not FOR UPDATE, and that is load-bearing. Every INSERT into
    refresh_tokens checks its foreign key by taking FOR KEY SHARE on the users row,
    and /refresh inserts its successor while holding the presented refresh row's
    lock — so it does reach the users row after a refresh_tokens row, implicitly.
    FOR UPDATE conflicts with FOR KEY SHARE, and a reset holding it while its
    revocation waited on that refresh row deadlocked against the rotation: measured,
    by tests/test_must_change_password.py's rotation-race test, before this was
    changed. FOR NO KEY UPDATE is the lock an ordinary UPDATE of this row takes: it
    conflicts with itself and with any write to the row, so these paths still queue
    on each other, and it lets a foreign-key check through.

    populate_existing, because the caller has usually loaded this row already, and
    without it the identity map hands back that copy as it was read — before a
    bcrypt round, before the lock — rather than the row as it is now.
    """
    return (
        await db.execute(
            select(User)
            .where(User.id == user_id)
            .with_for_update(key_share=True)  # FOR NO KEY UPDATE on PostgreSQL
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()


async def _issue_session(
    db: AsyncSession,
    user: User,
    response: Response,
    request: Request,
    client: str = WEB_CLIENT,
    replaces: RefreshToken | None = None,
) -> None:
    """Mint an access token and a new refresh session for the user, commit, and set both cookies.

    With `replaces`, the rotated row is linked to the new one in the same transaction, so a later replay of
    the rotated token can be recognised as a rotation that never arrived.
    """
    settings = get_settings()
    access = create_access_token(user.id, wire_role(user.role), user.org_id, client=client)
    raw_refresh, token_hash = new_refresh_token()
    issued = RefreshToken(
        user_id=user.id,
        token_hash=token_hash,
        user_agent=(request.headers.get("user-agent") or "")[:300],
        client=client,
        expires_at=now_utc() + dt.timedelta(days=settings.refresh_token_days),
        # Stamped here, on the application clock, not left to the column's server
        # default (batch 73 review). /refresh compares it with
        # users.sessions_valid_after, which the resets stamp with now_utc(), so the
        # comparison has one clock on both sides. The server default is Postgres'
        # now() — the database's clock, read at the start of the transaction — and
        # any skew between the two would refuse a session minted just after a reset
        # at its first refresh, or pass one minted just before it.
        created_at=now_utc(),
    )
    db.add(issued)
    if replaces is not None:
        # Flushed first so the rotated row can point at a real id in the same
        # transaction: that link is the only way a later replay of the rotated
        # token can be recognised as a response this client never received.
        await db.flush()
        replaces.replaced_by_id = issued.id
    await db.commit()
    set_auth_cookies(response, access, raw_refresh)


@router.post("/login")
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    _rl: None = Depends(login_limiter),
):
    """Sign in with email and password and set the session cookies.

    Refuses with 401 an unknown email, a wrong password, a password a reset replaced while it was being
    verified, and a deactivated account; a mobile sign-in must also hold the mobile grant. An account that
    must change its password is let in, and the flag in the returned payload sends the client to
    change-password.
    """
    # users.email is CITEXT with a UNIQUE btree index, so `==` is already
    # case-insensitive. Wrapping it in lower() made the comparison lower(email)
    # instead of email, which that index cannot serve: every sign-in — including
    # every failed one — sequentially scanned the whole users table.
    stmt = select(User).where(User.email == body.email)
    user = (await db.execute(stmt)).scalar_one_or_none()
    if user is None:
        # Equalize timing so a missing account can't be distinguished from a
        # wrong password by response latency (user-enumeration defense).
        await verify_password(body.password, _DUMMY_HASH)
        raise HTTPException(status_code=401, detail="Invalid email or password")
    verified_hash = user.password_hash
    if not await verify_password(body.password, verified_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    # Re-read the account under its row lock, held until the session below is
    # issued and committed, and refuse if the password changed while bcrypt was
    # running (batch 73 review). Unlocked, a reset committing in that window revoked
    # every refresh row that existed at the time — not the one this login was about
    # to insert — so a sign-in with the password the reset had just replaced came
    # out of it holding a live 30-day session. Both resets take the same lock before
    # they revoke anything, so whichever request locks first wins outright: this
    # login sees the reset's new hash and is refused, or the reset waits for this
    # commit and then revokes the session it produced.
    locked = await lock_user_row(db, user.id)
    if locked is None or locked.password_hash != verified_hash:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user = locked
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account is deactivated")
    # Ordered after the password check so the mobile-approval state of an account
    # is never disclosed to someone who cannot authenticate as it.
    if body.client == MOBILE_CLIENT:
        _assert_mobile_allowed(user)
    # must_change_password is deliberately NOT refused here. The temporary password
    # is the only credential a reset account has, and changing it needs a session:
    # refusing the sign-in would leave no way out. The session issued is gated
    # everywhere else by get_current_user, and the flag in the payload below tells
    # the client to go straight to the change-password screen.

    user.last_seen_at = now_utc()
    await _issue_session(db, user, response, request, client=body.client)
    return {"user": _user_payload(user)}


def _as_utc(value: dt.datetime) -> dt.datetime:
    # Rows written before the timezone-aware columns landed can still read naive.
    return value if value.tzinfo else value.replace(tzinfo=dt.UTC)


async def _revoke_session_family(db: AsyncSession, token: RefreshToken) -> None:
    """Burn every live session the reused token could belong to, and commit.

    Scoped to the user AND the client that opened the reused session. Walking the
    replaced_by chain would only reach rotations descended from this one row, while
    whoever holds a stolen cookie plausibly holds others taken with it, so the
    family is read as that user's live sessions for that client. Wider than the
    chain on purpose — and still narrow enough that a stolen phone token cannot
    sign the same person out of the browser they are working in, which is the
    property refresh_tokens.client exists to protect.
    """
    await db.execute(
        update(RefreshToken)
        .where(
            RefreshToken.user_id == token.user_id,
            RefreshToken.client == (token.client or WEB_CLIENT),
            RefreshToken.revoked_at.is_(None),
        )
        .values(revoked_at=now_utc())
        .execution_options(synchronize_session=False)
    )
    await db.commit()


# A rotation chain is one session moving forward, so it is short in practice. The
# walk below is guarded twice anyway, because it follows a link written by an
# earlier request and the two failures are not the same:
#
#   * `seen` stops a CYCLE. The length bound cannot: a set does not grow when the
#     walk revisits a row, so on A -> B -> A `len(seen)` sits at 2 forever, and the
#     walk goes round issuing the same two locking SELECTs until something kills
#     the request. Measured against the earlier `db.get` version, which served the
#     identity-mapped row without awaiting anything: removing `seen` hung the test
#     process outright, since the loop never yielded and no timeout could reach it.
#     The locking read below does yield, so the same bug is now a hot loop against
#     the database rather than a wedged worker — still worth not having.
#   * the runaway limit stops an acyclic chain of absurd length from turning one
#     logout into unbounded round trips. It is NOT a functional bound on how long
#     a lineage may be, and the earlier value of 64 was wrong to treat it as one:
#     a web session refreshing on a 15-minute access token rotates four times an
#     hour, so an ordinary long-lived session passes 64 links inside a day, and
#     stopping there let logout answer "Logged out successfully" with live
#     descendants still reachable. That is precisely the defect this function
#     exists to fix, back again past an arbitrary line. Set high enough that no
#     real lineage reaches it, and hitting it is handled below rather than
#     silently reported as success.
_ROTATION_CHAIN_RUNAWAY_LIMIT = 10_000


async def _revoke_rotation_chain(db: AsyncSession, token: RefreshToken) -> int:
    """Revoke this token and every session it was rotated into.

    Logout used to revoke exactly the row it was handed, which cannot end a session
    that has since moved on. The case is not hypothetical, and it is the one this
    module already builds for elsewhere: a client whose rotation response never
    arrived holds a spent token (see `_undelivered_successor`). Presenting it here
    revoked a row that was already revoked, cleared the caller's cookies and
    answered "Logged out successfully" — while the live successor kept working
    until it expired, days later. Anyone else holding that successor, which is one
    good reason the response went missing, kept a working session.

    Same shape reached by a race: a refresh that commits its rotation while a
    logout is deciding leaves the logout revoking the predecessor of a session that
    now exists. The FOR UPDATE in the caller serializes the two, and this walks to
    whatever the rotation produced.

    The CHAIN, not the family. `_revoke_session_family` is deliberately wider
    because theft implies other stolen cookies; a logout implies nothing about the
    user's other sessions, so this revokes exactly the lineage of the token
    presented and leaves a second browser signed in.

    The one thing it must never do is report success while a descendant is still
    live, so the runaway limit is not allowed to end the walk quietly: reaching it
    falls back to `_revoke_session_family`, which is wider than this lineage but a
    guaranteed superset of it. That trades the scope property for the security one
    in a case no real lineage reaches — the limit is 10,000 links, where a busy
    session accumulates a few thousand in a year — and it is logged at ERROR
    because reaching it means something is wrong with rotation, not with logout.

    ONE REQUIREMENT ON ANYONE WHO ADDS A RETENTION JOB. `replaced_by_id` is
    `ON DELETE SET NULL`, so deleting an ancestor row silently clears the link to
    its successor and a logout presenting that ancestor can no longer reach the
    live descendant — the defect this function exists to fix, reintroduced by a
    tidy-up. Nothing prunes refresh_tokens today. Anything that starts to must
    either keep a chain whole for as long as any row in it can still be presented,
    or revoke the reachable descendants before deleting an ancestor.
    """
    revoked = 0
    seen: set[uuid.UUID] = set()
    node: RefreshToken | None = token
    while node is not None and node.id not in seen:
        if len(seen) >= _ROTATION_CHAIN_RUNAWAY_LIMIT:
            # Whatever is left of the lineage is unreachable in the budget we are
            # willing to spend, and answering 200 with a live descendant is the one
            # outcome this function exists to prevent. The family is wider than the
            # lineage and revoking it is not what a logout should normally do, but
            # it cannot leave a descendant behind.
            logger.error(
                "Rotation chain for user %s exceeded %s links; revoking the session "
                "family so the logout cannot leave a live descendant",
                token.user_id,
                _ROTATION_CHAIN_RUNAWAY_LIMIT,
            )
            await _revoke_session_family(db, token)
            return revoked
        seen.add(node.id)
        if node.revoked_at is None:
            node.revoked_at = now_utc()
            revoked += 1
        if node.replaced_by_id is None:
            break
        # FOR UPDATE on every node, not just the one the caller presented. Locking
        # only the presented token leaves the rest of the chain open: a refresh can
        # rotate successor B into C after this walk has read B, and since the B it
        # read still says `replaced_by_id IS NULL`, the walk stops there and C stays
        # live. Locking each node as it is reached makes the walk see whatever the
        # rotation produced, one link at a time.
        node = (
            await db.execute(
                select(RefreshToken).where(RefreshToken.id == node.replaced_by_id).with_for_update()
            )
        ).scalar_one_or_none()
    return revoked


async def _undelivered_successor(db: AsyncSession, token: RefreshToken) -> RefreshToken:
    """Resolve a replay of an already-rotated token, or raise 401 as theft.

    Rotation commits the revocation of the presented token before its replacement
    can possibly reach the client, so any response lost in flight — the iOS
    client's 30s budget expiring, a proxy 502 after the commit, the app being
    suspended, a reset connection — leaves a perfectly healthy client holding a
    token the server considers spent. Refusing that is what ended sessions that
    were fine, so within a bounded window the replay is honoured by rotating from
    the successor the client never saw. It buys exactly one rotation: that
    successor is spent by the rotation below, so a second replay lands in the
    theft branch.

    Every other reuse is a token someone has already spent being presented again,
    which is what a captured cookie looks like, so the family is burned rather than
    just this row.

    Only for a token that WAS rotated. /refresh refuses a revoked token with no
    successor before it gets here, without the burn: nothing spent that token, it
    was ended on purpose, and burning the family for it signed users out of the
    session a reset or a password change had just given them (batch 73 review).
    """
    grace = dt.timedelta(seconds=get_settings().refresh_reuse_grace_seconds)
    # FOR UPDATE, for the same reason the logout walk takes it on every node. This
    # reads the successor's `revoked_at` to decide whether the rotation was
    # undelivered, and then rotates from it — read it unlocked and a concurrent
    # /refresh can lock and spend that successor in between, leaving this request
    # to honour a grace window against state that no longer holds and fork a second
    # live branch off it. Blocking here instead means the existing reuse policy is
    # applied to the successor's committed state, whichever request got there first.
    successor = (
        (
            await db.execute(
                select(RefreshToken).where(RefreshToken.id == token.replaced_by_id).with_for_update()
            )
        ).scalar_one_or_none()
        if token.replaced_by_id
        else None
    )
    undelivered = (
        successor is not None
        and successor.revoked_at is None
        and _as_utc(successor.expires_at) > now_utc()
        and now_utc() - _as_utc(token.revoked_at) <= grace
    )
    if not undelivered:
        logger.warning(
            "Refresh token reuse for user %s (%s session): revoking that client's session family",
            token.user_id,
            token.client or WEB_CLIENT,
        )
        await _revoke_session_family(db, token)
        # Same detail as an unknown token: which of the two it was is not the
        # caller's business, and saying so would help a thief probe the window.
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    logger.info(
        "Rotation response never reached user %s; replaying onto the undelivered successor",
        token.user_id,
    )
    return successor


@router.post("/refresh")
async def refresh(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    _rl: None = Depends(refresh_limiter),
):
    """Rotate the presented refresh token into a new session and set fresh cookies.

    Refuses with 401 a token that is unknown, expired, revoked without ever being rotated, created before the
    account's last reset, or held by a deactivated account, and re-checks the mobile grant for a mobile
    session. A rotated token replayed within the grace window is honoured once; any other replay burns that
    client's session family.
    """
    raw = request.cookies.get(REFRESH_COOKIE)
    if not raw:
        raise HTTPException(status_code=401, detail="Refresh token required")
    # No explicit users-row lock on this path: it locks a refresh_tokens row first,
    # the opposite of the order the credential paths keep, so it must never wait on
    # an account lock they hold. The foreign-key check its insert makes does touch
    # the users row, which lock_user_row's FOR NO KEY UPDATE is chosen to let
    # through. A rotation racing a reset is caught by the session-epoch check below
    # instead (batch 73 review).
    #
    # FOR UPDATE. Rotation is a read-check-write on this row — `revoked_at` is
    # read at the branch below and written at the end — and without the lock those
    # are two transactions, so the single-use rule is not enforced under
    # concurrency at all. Measured before this line existed: four simultaneous
    # refreshes presenting the SAME token all returned 200 and left three live
    # sessions. Worse than the extra sessions, every one of them read the token as
    # unrevoked, so none of them entered the reuse branch — the theft detection
    # below, and the family revocation it triggers, are exactly what a captured
    # cookie racing the real client would have slipped past.
    token = (
        await db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(raw)).with_for_update()
        )
    ).scalar_one_or_none()
    if token is None:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    if token.revoked_at is not None and token.replaced_by_id is None:
        # Revoked and never rotated — rotation is the only writer of replaced_by_id
        # (_issue_session(replaces=...), in the same transaction as the revocation)
        # — so this session was ended on purpose: by a reset, a logout, a password
        # change, a deactivation or a revoked mobile grant. Refused, and that is all
        # (batch 73 review). It used to fall into the theft branch below, which
        # burns every live session this user has on this client, so a stale device
        # refreshing after an admin reset, or after the owner changed the password
        # elsewhere, signed the user out of the session they had just made — the
        # temporary-password sign-in, or the one change-password had re-issued.
        # Both clients now force a refresh precisely to learn whether their session
        # survived a change made somewhere else, so this is the ordinary answer to
        # an ordinary question, not evidence of a stolen cookie. Info, not a
        # warning, for the same reason; same detail as an unknown token.
        logger.info(
            "Refresh with a revoked, never-rotated token for user %s (%s session); refusing it",
            token.user_id,
            token.client or WEB_CLIENT,
        )
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    if token.revoked_at is not None:
        # Substituting the successor rather than returning early keeps every check
        # below — expiry, is_active, the session epoch, the mobile grant — applied
        # in the same order, to the row that is actually the live session.
        token = await _undelivered_successor(db, token)
    expires = _as_utc(token.expires_at)
    if expires < now_utc():
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user = await db.get(User, token.user_id)
    if user is None or not user.is_active:
        # Deactivated users are revoked HERE — the gap the old build shipped.
        token.revoked_at = now_utc()
        await db.commit()
        raise HTTPException(status_code=401, detail="Account is deactivated")

    # The session epoch, for refresh rows (batch 73 review). A reset revokes every
    # refresh row it can see and then stamps users.sessions_valid_after, but a
    # rotation that holds the presented row's lock while the reset runs inserts its
    # successor in a transaction the reset's revocation cannot see, so that
    # successor came out of the reset live — a session the reset was meant to end,
    # good for another 30 days. Any row created before the epoch belongs to a
    # session from before the last reset, so it is revoked here and refused, the
    # way core/deps.py refuses an access token issued before it.
    epoch = user.sessions_valid_after
    if epoch is not None and _as_utc(token.created_at) < _as_utc(epoch):
        token.revoked_at = now_utc()
        await db.commit()
        logger.info("Refresh with a session from before the last reset for user %s; refusing it", user.id)
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    # A mobile session outlives the grant that created it unless the grant is
    # re-checked here — same reasoning as is_active directly above. The session is
    # burned before raising so a revoked user cannot keep retrying this token.
    #
    # The gate itself is `_assert_mobile_allowed`, not a copy of its condition. The
    # copy that used to live here collapsed both cases into MOBILE_NOT_APPROVED, so a
    # superadmin holding a mobile session was told to ask a super admin to approve
    # mobile sign-in — advice addressed to themselves, and the phone picked the
    # remedy screen over the web-only one on the strength of that code. The two
    # refusals stay distinct because the remedies are: a grant can be given, a
    # superadmin's web-only portal cannot.
    session_client = token.client or WEB_CLIENT
    if session_client == MOBILE_CLIENT:
        try:
            _assert_mobile_allowed(user)
        except HTTPException:
            # Broader than CodedHTTPException on purpose: burning the token is the
            # security-relevant half, and it must not stop happening if the gate ever
            # grows a refusal that is not coded.
            token.revoked_at = now_utc()
            await db.commit()
            raise

    # Nor is must_change_password refused here, for the reason login gives. Burning
    # or failing the refresh of a reset account would sign the user out of the very
    # session they need to change the password from, fifteen minutes in.
    token.revoked_at = now_utc()  # rotation: old token is single-use
    await _issue_session(db, user, response, request, client=session_client, replaces=token)
    return {"user": _user_payload(user)}


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    # Un-gated: a reset account must always be able to sign out, and refusing it
    # would leave its refresh session live on the device it is leaving.
    user: User = Depends(get_current_user_pending_password_change),
):
    """Revoke the presented refresh token and every session it was rotated into, then clear the cookies.

    Reachable while a password change is pending, so a reset account can always sign out. Takes the account's
    row lock first, and re-stamps a row the reset had already revoked so change-password will not re-issue a
    session to this device.
    """
    # The account first, then its refresh row: the lock order lock_user_row
    # documents. This route always wrote the users row (last_seen_at, below), but
    # at commit — AFTER locking the refresh row — which is the opposite order to
    # change-password and could deadlock against it now that change-password locks
    # the presented row too. Refreshed, so the session epoch read below is current.
    await lock_user_row(db, user.id)
    raw = request.cookies.get(REFRESH_COOKIE)
    if raw:
        # FOR UPDATE for the same reason /refresh takes it: this reads the row and
        # then writes it, and a rotation committing in between would leave the
        # successor untouched.
        token = (
            await db.execute(
                select(RefreshToken)
                .where(RefreshToken.token_hash == hash_refresh_token(raw))
                .with_for_update()
            )
        ).scalar_one_or_none()
        if token and token.user_id == user.id:
            # A device signed in before an administrator's reset presents a row the
            # reset already revoked, and revoking it again — the walk below only
            # writes revoked_at where it is NULL — left it reading as "ended by the
            # reset" (revoked_at <= the epoch). change-password re-issues a session
            # for exactly that, so a change in flight when the user pressed sign-out
            # still handed this device a fresh 30-day session after it had signed
            # out (batch 73 review). Re-stamped, the row reads as ended AFTER the
            # reset, by this logout, and _reissue_after_password_change refuses it.
            epoch = user.sessions_valid_after
            if (
                token.revoked_at is not None
                and epoch is not None
                and _as_utc(token.revoked_at) <= _as_utc(epoch)
            ):
                token.revoked_at = now_utc()
            _revoked = await _revoke_rotation_chain(db, token)
            if _revoked > 1:
                logger.info(
                    "Logout for user %s revoked %s rotated sessions from the presented token",
                    user.id,
                    _revoked,
                )
    user.last_seen_at = now_utc()
    await db.commit()
    clear_auth_cookies(response)
    return {"message": "Logged out successfully"}


@router.get("/me")
async def me(
    # Un-gated, and the full payload: the client reads must_change_password here
    # before and after a password change to decide which screen it is on.
    user: User = Depends(get_current_user_pending_password_change),
    db: AsyncSession = Depends(get_db),
):
    """The signed-in user's profile, including must_change_password.

    For every role but superadmin it adds avatar, about, presence status, last seen, mobile access, and the
    resolved organization and department names.
    """
    payload = _user_payload(user)
    if user.role != UserRole.superadmin:
        statuses = await presence.get_statuses([user.id])
        payload["avatar_url"] = user.avatar_url
        payload["about"] = user.about
        payload["status"] = statuses.get(str(user.id), "offline")
        payload["last_seen"] = iso_z(user.last_seen_at)
        payload["mobile_access"] = user.mobile_access
        # Resolved NAMES, not just the ids _user_payload already carries. The
        # profile drawer renders org_name and dept_name, which this endpoint
        # never returned — so those two rows read "N/A" for every user, always,
        # regardless of their actual organization and department.
        payload["org_name"] = (await db.get(Organization, user.org_id)).name if user.org_id else None
        payload["dept_name"] = (await db.get(Department, user.dept_id)).name if user.dept_id else None
    return payload


def _reissue_after_password_change(
    raw_refresh: str | None, presented: RefreshToken | None, user: User
) -> bool:
    """Whether change-password should hand the caller a new session.

    Only when the session it is changing the password from was ended by an
    administrator's reset — and that has to be proven, not inferred from "no live
    row". The inference was the batch 73 review's finding: pressing sign-out while
    the change was in flight let logout revoke the caller's row first, and this
    route, finding no live row, re-issued a fresh 30-day session into the device
    that had just signed out, on web and iOS alike.

    So, re-issue when:
      * no refresh cookie came with the request at all — a caller that never held a
        refresh session (API tooling on a bearer token) keeps today's behaviour; or
      * the presented row is this user's, is revoked, and was revoked no later than
        users.sessions_valid_after. Both resets stamp the epoch AFTER revoking every
        refresh row, so revoked_at <= epoch means the reset (or a rotation before
        it) ended this session. A logout, or any other revocation after the last
        reset, lands after the epoch and is respected — including a logout from a
        device whose row the reset had already revoked, which re-stamps that row
        for exactly this reason (batch 73 review).

    Never for a live row (it is kept, and a second session would be pointless), an
    unknown cookie, or a revoked row on an account no reset has touched.
    """
    if not raw_refresh:
        return True
    if presented is None or presented.revoked_at is None:
        return False
    epoch = user.sessions_valid_after
    return epoch is not None and _as_utc(presented.revoked_at) <= _as_utc(epoch)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.post("/change-password")
async def change_password(
    body: ChangePasswordRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    # Un-gated: this is the route a reset account is being sent to.
    user: User = Depends(get_current_user_pending_password_change),
    _rl: None = Depends(password_limiter),
):
    """Change the caller's password after verifying the current one, and clear must_change_password.

    Refuses with 400 a new password equal to the current one or failing the policy, and with 401 a wrong
    current password or one a reset replaced while it was being verified. Revokes every other refresh session
    and keeps the caller's, issuing a fresh one when an administrator's reset ended it or the request carried
    no refresh cookie.
    """
    # Rate-limited like the admin reset it mirrors: this route verifies
    # current_password, so an attacker sitting on a hijacked session could
    # otherwise brute-force it offline-fast and take the account outright.
    from app.core.security import PasswordPolicyError, enforce_password_policy, hash_password

    verified_hash = user.password_hash
    if not await verify_password(body.current_password, verified_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    # After the verify, so it reveals nothing to a caller who does not already know
    # the current password. For every account, not only reset ones: "changing" a
    # password to itself clears must_change_password while leaving the account on
    # the temporary password the administrator was shown, which is the defect this
    # flag exists to close (batch 73).
    if body.new_password == body.current_password:
        raise HTTPException(status_code=400, detail="Choose a password different from your current one.")
    try:
        enforce_password_policy(body.new_password)
    except PasswordPolicyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    new_hash = await hash_password(body.new_password)
    # The account's row lock, taken after both bcrypt rounds so it is held only for
    # the writes, and before any refresh_tokens row (lock_user_row). Refused if the
    # password changed since it was verified (batch 73 review): an administrator's
    # reset committing in that window would otherwise be overwritten by a password
    # chosen by whoever was holding the session the reset was meant to end, and —
    # since the reset had just revoked that session's refresh row — this route
    # would then re-issue them a fresh one. Same answer as a wrong password: the
    # password they proved is not the account's password any more.
    locked = await lock_user_row(db, user.id)
    if locked is None or locked.password_hash != verified_hash:
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    user.password_hash = new_hash
    user.must_change_password = False
    # Revoke every OTHER session on password change — the caller keeps the one
    # they are changing it from, which is the whole point of "other".
    #
    # Revoking the caller's own token too (what this did) signed the user out of
    # the tab they had just used, silently: no client re-authenticates here, so
    # the session died at the next refresh, up to access_token_minutes later, with
    # no visible connection to the password change. Worse, that refresh presented
    # a revoked token with no successor, which /refresh then treated as a replay,
    # so every password change also logged a "token reuse — revoking that client's
    # session family" theft warning against an account that was never attacked.
    # (/refresh now refuses such a token quietly, without the burn — batch 73
    # review — but signing the caller out would still be wrong.)
    raw_refresh = request.cookies.get(REFRESH_COOKIE)
    # The row the caller presented, looked up whatever its state: whether it is live,
    # and if not who revoked it, decides below whether this request may hand out a
    # new session. Only this user's own row counts — a refresh cookie belonging to
    # another account says nothing about this one's session.
    #
    # FOR UPDATE, so the decision is made on the row as committed (batch 73
    # review). Logout re-stamps a row the reset had revoked so that it stops reading
    # as "ended by the reset", and a decision made on the row as it was before that
    # commit re-issues a session into a device that has just signed out. Logout
    # takes the account's lock first too, so the two already queue there; the row
    # lock is what still holds if either route's account lock ever moves, and
    # against /refresh, which takes no account lock at all.
    presented = None
    if raw_refresh:
        presented = (
            await db.execute(
                select(RefreshToken)
                .where(RefreshToken.token_hash == hash_refresh_token(raw_refresh))
                .with_for_update()
            )
        ).scalar_one_or_none()
        if presented is not None and presented.user_id != user.id:
            presented = None
    tokens = (
        await db.execute(
            select(RefreshToken).where(RefreshToken.user_id == user.id, RefreshToken.revoked_at.is_(None))
        )
    ).scalars()
    for t in tokens:
        if presented is not None and t.id == presented.id:
            continue
        t.revoked_at = now_utc()
    if _reissue_after_password_change(raw_refresh, presented, user):
        # The caller's session was ended by an administrator's reset, so a client
        # that was signed in before it is finishing the change on its access cookie
        # alone. Answering 200 and leaving it there signs the user out when that
        # cookie lapses, at most access_token_minutes later and with no visible
        # connection to the password they just chose. A fresh session in the same
        # transaction as the change, for the client this access token was minted
        # for, is what "finished" means (batch 73).
        await _issue_session(db, user, response, request, client=session_client(request))
    else:
        await db.commit()
    return {"message": "Password changed"}
