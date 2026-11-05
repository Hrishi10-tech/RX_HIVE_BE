"""Notifications (net-new vs the Mongo build): in-app list + Web Push subs."""

import anyio
import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.deps import get_current_user, get_current_user_pending_password_change
from app.core.rate_limit import push_subscribe_limiter
from app.db.models import Notification, PushSubscription, User
from app.db.session import get_db
from app.utils import iso_z

router = APIRouter(prefix="/api/notifications", tags=["notifications"])

# Wall-clock budget for the SSRF check's DNS lookup. Generous for a real push
# service (Google/Mozilla resolve in single-digit ms) and short enough that a
# deliberately slow nameserver costs one request rather than the worker.
PUSH_ENDPOINT_RESOLVE_TIMEOUT = 2.0

# Concurrent resolver threads, bounded separately from anyio's shared pool.
#
# Needed because of what makes the deadline below work: abandon_on_cancel hands
# the request back on time but cannot stop the thread, which stays in
# getaddrinfo until libc gives up — for as long as resolv.conf allows. Those
# abandoned threads have to be capped somewhere, and it must not be the default
# limiter, because that is the same 40 tokens serving every attachment upload,
# presign, thumbnail and web push send (services/storage.py, services/push.py).
# Filling it would stall all of them.
#
# 4 is generous for the real traffic: a browser calls this once per subscription
# change, and a real push service resolves in single-digit milliseconds.
_RESOLVE_LIMITER = anyio.CapacityLimiter(4)


@router.get("/vapid-key")
async def vapid_key(_: User = Depends(get_current_user)):
    """Public VAPID key for the browser Push subscription (empty if unconfigured)."""
    return {"public_key": get_settings().vapid_public_key}


def _serialize_notification(n: Notification) -> dict:
    return {
        "_id": str(n.id),
        "type": n.type,
        "payload": n.payload or {},
        "is_read": n.is_read,
        "created_at": iso_z(n.created_at),
    }


@router.get("")
async def list_notifications(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    rows = (
        (
            await db.execute(
                select(Notification)
                .where(Notification.user_id == user.id)
                .order_by(Notification.created_at.desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )
    return {"data": [_serialize_notification(n) for n in rows]}


@router.post("/read-all")
async def read_all(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await db.execute(
        update(Notification)
        .where(Notification.user_id == user.id, Notification.is_read.is_(False))
        .values(is_read=True)
    )
    await db.commit()
    return {"message": "Marked as read"}


class SubscribeRequest(BaseModel):
    endpoint: str
    keys: dict = Field(default_factory=dict)


class UnsubscribeRequest(BaseModel):
    endpoint: str


@router.post("/subscribe")
async def subscribe(
    body: SubscribeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _rl: None = Depends(push_subscribe_limiter),
):
    from app.services.push import validate_push_endpoint

    endpoint = (body.endpoint or "").strip()
    if not endpoint:
        raise HTTPException(status_code=400, detail="Endpoint required")
    # SSRF guard: the server will POST to this URL, so it must be a public HTTPS
    # push service — never an internal/loopback address.
    #
    # Off the event loop and on a deadline. The check resolves the hostname with
    # socket.getaddrinfo, which is the blocking libc resolver and takes no timeout
    # of its own: the hostname here is entirely caller-chosen, so any authenticated
    # user could point it at a black-holing nameserver and freeze this worker —
    # every HTTP request, every WebSocket, and the call-deadline sweeper — for the
    # tens of seconds resolv.conf allows.
    #
    # abandon_on_cancel is what makes that deadline real, and it is not the
    # default. anyio.to_thread.run_sync defaults to abandon_on_cancel=False, which
    # means a cancelled scope WAITS for the worker to return before the
    # cancellation propagates — so move_on_after could not interrupt an
    # uninterruptible getaddrinfo, and this budget bounded nothing at all. It read
    # as a timeout and behaved as a comment: measured, a 1s deadline around a 4s
    # blocking call returned after 4.01s with cancelled_caught FALSE, so even the
    # 400 below was unreachable by that path.
    #
    # True does not stop the thread — nothing can — it stops the REQUEST waiting
    # on it, which is the part that was holding a connection and a DB session. The
    # thread is then bounded by _RESOLVE_LIMITER instead of by the shared pool.
    endpoint_ok = False
    with anyio.move_on_after(PUSH_ENDPOINT_RESOLVE_TIMEOUT) as scope:
        endpoint_ok = await anyio.to_thread.run_sync(
            validate_push_endpoint,
            endpoint,
            abandon_on_cancel=True,
            limiter=_RESOLVE_LIMITER,
        )
    if scope.cancelled_caught or not endpoint_ok:
        raise HTTPException(status_code=400, detail="Invalid push endpoint")
    existing = (
        await db.execute(select(PushSubscription).where(PushSubscription.endpoint == endpoint))
    ).scalar_one_or_none()
    if existing is not None:
        # Endpoints are globally unique; re-subscribing re-binds to the caller.
        existing.user_id = user.id
        existing.keys = dict(body.keys or {})
    else:
        db.add(PushSubscription(user_id=user.id, endpoint=endpoint, keys=dict(body.keys or {})))
    await db.commit()
    return {"message": "Subscribed"}


@router.delete("/subscribe")
async def unsubscribe(
    body: UnsubscribeRequest,
    # Un-gated, unlike POST above (batch 73). Both clients unsubscribe on the way to
    # signing out, and an account waiting to change its password must still be able
    # to stop push arriving on a device it is leaving. Removing a subscription
    # grants nothing; adding one stays refused until the password is changed.
    user: User = Depends(get_current_user_pending_password_change),
    db: AsyncSession = Depends(get_db),
):
    """Delete the caller's push subscription for the given endpoint.

    Not gated on must_change_password, so an account signing out after an administrator's reset can still stop
    push reaching the device it is leaving.
    """
    await db.execute(
        delete(PushSubscription).where(
            PushSubscription.endpoint == (body.endpoint or "").strip(),
            PushSubscription.user_id == user.id,
        )
    )
    await db.commit()
    return {"message": "Unsubscribed"}
