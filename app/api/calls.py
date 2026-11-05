"""Call history, links, scheduling, and the LiveKit webhook.

REST surface of the LiveKit-backed calling stack (lifecycle itself lives in
app/services/calls.py). Wire shapes preserve the RxHivexx contract
(docs/reference/api-uploads-calls-misc.md): serialize_call docs use `_id`
string keys and ISO-Z datetimes.

Fixes vs the Mongo build: /schedule 400s on malformed datetimes instead of
500ing, validates participants against the caller's org, and inserts the
CallLink row for the generated code (the old build left the code dangling);
/ended verifies call membership via the calls service.
"""

import contextlib
import datetime as dt
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from livekit import api as lk_api
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.deps import get_current_user
from app.db.models import Call, CallLink, CallParticipant, CallStatus, CallType, ScheduledCall, User
from app.db.session import get_db
from app.realtime.redis_bus import publish_to_users
from app.services import calls as calls_service
from app.services.calls import handle_call_ws_message, mint_token, serialize_call
from app.utils import generate_call_code, iso_z, now_utc, parse_uuid, sanitize_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/calls", tags=["calls"])

# The LiveKit webhook lives outside the /api/calls prefix; mounted separately.
webhook_router = APIRouter(tags=["calls"])

_ACTIVE_STATUSES = (CallStatus.ringing, CallStatus.connected)


def _not_initiated_by(user_id: uuid.UUID):
    # Mongo `$ne` semantics: docs with a missing initiator also match.
    return or_(Call.initiated_by.is_(None), Call.initiated_by != user_id)


@router.get("/history")
async def call_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    filter_: str = Query("all", alias="filter"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    base = (
        select(Call)
        .join(CallParticipant, CallParticipant.call_id == Call.id)
        .where(CallParticipant.user_id == user.id)
    )
    if filter_ == "missed":
        base = base.where(Call.status == CallStatus.missed, _not_initiated_by(user.id))
    elif filter_ == "incoming":
        base = base.where(_not_initiated_by(user.id))
    elif filter_ == "outgoing":
        base = base.where(Call.initiated_by == user.id)

    total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    calls = (
        (await db.execute(base.order_by(Call.started_at.desc()).offset((page - 1) * limit).limit(limit)))
        .scalars()
        .all()
    )
    data = [await serialize_call(db, call, user.id) for call in calls]
    return {"data": data, "total": total, "page": page, "limit": limit}


@router.get("/missed-count")
async def missed_count(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    stmt = (
        select(func.count())
        .select_from(CallParticipant)
        .join(Call, Call.id == CallParticipant.call_id)
        .where(
            CallParticipant.user_id == user.id,
            CallParticipant.seen_at.is_(None),
            Call.status == CallStatus.missed,
            _not_initiated_by(user.id),
        )
    )
    count = (await db.execute(stmt)).scalar_one()
    return {"count": count}


@router.post("/mark-seen")
async def mark_seen(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Clear the missed-call badge for calls that have actually finished.

    In-flight calls are excluded. The CallParticipant row is written when the call
    is initiated, so it already exists while the phone is ringing: opening the
    Calls tab during an incoming ring used to stamp seen_at on that row, and when
    the ring then timed out into `missed` the badge stayed at zero — the one
    missed call the user most needed to see was the one silently marked read.
    """
    in_flight = select(Call.id).where(Call.status.in_(_ACTIVE_STATUSES))
    await db.execute(
        update(CallParticipant)
        .where(
            CallParticipant.user_id == user.id,
            CallParticipant.seen_at.is_(None),
            CallParticipant.call_id.notin_(in_flight),
        )
        .values(seen_at=now_utc())
    )
    await db.commit()
    return {"message": "Marked as seen"}


class CreateLinkRequest(BaseModel):
    call_type: str = "video"


@router.post("/create-link")
async def create_link(
    body: CreateLinkRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ctype = CallType.voice if body.call_type == "voice" else CallType.video
    link = CallLink(org_id=user.org_id, creator_id=user.id, code=generate_call_code(), call_type=ctype)
    db.add(link)
    await db.commit()
    return {"code": link.code, "url": f"/call/{link.code}", "call_type": ctype.value}


@router.get("/link/{code}")
async def get_link(code: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    _ = user  # any authenticated user may resolve a meet-style link
    link = (
        await db.execute(select(CallLink).where(CallLink.code == code, CallLink.is_active.is_(True)))
    ).scalar_one_or_none()
    if link is None:
        raise HTTPException(status_code=404, detail="Call link not found")
    return {
        "_id": str(link.id),
        "code": link.code,
        "creator_id": str(link.creator_id) if link.creator_id else None,
        "call_type": link.call_type.value,
        "created_at": iso_z(link.created_at),
        "is_active": link.is_active,
    }


class ScheduleCallRequest(BaseModel):
    # ScheduledCall.title is String(300); longer titles 500'd on commit.
    title: str = Field(max_length=300)
    description: str | None = None
    start_time: str
    end_time: str | None = None
    call_type: str = "video"
    reminder_minutes: int = 15
    participant_ids: list[str] = Field(default_factory=list)


def _parse_iso(value: str, field: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field} format") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed


def _serialize_scheduled(call: ScheduledCall) -> dict:
    return {
        "_id": str(call.id),
        "title": call.title,
        "description": call.description,
        "call_type": call.call_type.value,
        "call_link_code": call.call_link_code,
        "start_time": iso_z(call.start_time),
        "end_time": iso_z(call.end_time),
        "reminder_minutes": call.reminder_minutes,
        "creator_id": str(call.creator_id) if call.creator_id else None,
        "participant_ids": [str(p) for p in (call.participant_ids or [])],
        "org_id": str(call.org_id) if call.org_id else None,
        "status": call.status,
        "created_at": iso_z(call.created_at),
    }


@router.post("/schedule")
async def schedule_call(
    body: ScheduleCallRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    start = _parse_iso(body.start_time, "start_time")
    end = _parse_iso(body.end_time, "end_time") if body.end_time else None

    # Keep only requested participants that exist, are active, and share the
    # caller's org; invalid/foreign ids are silently skipped (contract).
    requested = list(dict.fromkeys(u for u in (parse_uuid(p) for p in body.participant_ids) if u is not None))
    valid_ids: list[str] = []
    if requested:
        rows = (
            (
                await db.execute(
                    select(User.id).where(
                        User.id.in_(requested),
                        User.org_id == user.org_id,
                        User.is_active.is_(True),
                    )
                )
            )
            .scalars()
            .all()
        )
        keep = set(rows)
        valid_ids = [str(u) for u in requested if u in keep]

    ctype = CallType.voice if body.call_type == "voice" else CallType.video
    code = generate_call_code()
    now = now_utc()
    scheduled = ScheduledCall(
        org_id=user.org_id,
        creator_id=user.id,
        title=sanitize_text(body.title).strip(),
        description=sanitize_text(body.description) if body.description else None,
        call_type=ctype,
        call_link_code=code,
        start_time=start,
        end_time=end,
        reminder_minutes=body.reminder_minutes,
        participant_ids=valid_ids,
        status="scheduled",
        created_at=now,
    )
    db.add(scheduled)
    # The Mongo build generated a code but never inserted the link — fixed here.
    db.add(CallLink(org_id=user.org_id, creator_id=user.id, code=code, call_type=ctype))
    await db.commit()
    return _serialize_scheduled(scheduled)


@router.get("/scheduled")
async def list_scheduled(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    stmt = (
        select(ScheduledCall)
        .where(
            ScheduledCall.status == "scheduled",
            ScheduledCall.start_time >= now_utc(),
            or_(
                ScheduledCall.creator_id == user.id,
                ScheduledCall.participant_ids.contains([str(user.id)]),
            ),
        )
        .order_by(ScheduledCall.start_time.asc())
    )
    calls = (await db.execute(stmt)).scalars().all()
    return [_serialize_scheduled(c) for c in calls]


class CallEndedBeacon(BaseModel):
    call_id: str
    user_id: str


@router.post("/ended")
async def call_ended(body: CallEndedBeacon, user: User = Depends(get_current_user)):
    if body.user_id != str(user.id):
        raise HTTPException(status_code=403, detail="Cannot end call for another user")
    # The service verifies the caller is actually a call participant.
    await handle_call_ws_message(user, {"type": "call:end", "call_id": body.call_id})
    return {"message": "OK"}


@router.get("/active")
async def active_call(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The call this client should currently be showing, if any.

    Every `call:*` frame is a fire-and-forget publish, so anything sent while a
    socket was reconnecting is gone. This is how a client recovers from that:
    fetched on connect, on reconnect, and on returning to the foreground. Without
    it a ring delivered during a two-second Wi-Fi handover was simply never seen,
    even though the server rang on for another forty seconds.

    `{"call": null}` is the normal, common answer — not an error.
    """
    return {"call": await calls_service.active_call_state(db, user.id)}


class CallInviteRequest(BaseModel):
    # Capped well below GROUP_CALL_CAP: this is one tap of a picker, not a bulk import,
    # and an unbounded list would let one request ring an entire org.
    user_ids: list[uuid.UUID] = Field(min_length=1, max_length=32)


@router.post("/{call_id}/invite")
async def invite_to_call(
    call_id: uuid.UUID,
    body: CallInviteRequest,
    user: User = Depends(get_current_user),
):
    """Add people to a group call that is already running.

    REST rather than a `call:*` socket frame, because the caller needs the ANSWER: a
    per-invitee outcome, so the UI can say "Priya is in another call" instead of
    reporting a flat success for a partial result. Every other call action is
    fire-and-forget and belongs on the socket; this one is a request.

    Authorisation is entirely in `services/calls.invite_to_call` — only someone already
    in the call may invite, group calls only, and each invitee is org-checked against
    the inviter.
    """
    result = await calls_service.invite_to_call(user, call_id, body.user_ids)
    if error := result.get("error"):
        raise HTTPException(
            status_code=403 if error == "not_a_participant" else 400,
            detail={
                "not_joinable": "That call is no longer active",
                "not_a_group_call": "People can only be added to a group call",
                "not_a_participant": "You are not in this call",
            }.get(error, "This call cannot be added to"),
        )
    return result


class CallTokenRequest(BaseModel):
    # Stable per install/tab. Makes the LiveKit identity unique per client so a
    # second device of the same user is additive rather than evicting the first;
    # see services/calls.identity_for. Optional so an older client still works —
    # it just gets a random suffix per join instead of a stable one.
    device_id: str | None = Field(default=None, max_length=64)


@router.post("/{call_id}/token")
async def call_token(
    call_id: uuid.UUID,
    body: CallTokenRequest | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # FOR UPDATE, because everything below decides on `call.status` and then
    # writes. Without the lock the guard and the write are two transactions: a call
    # ended in between still gets a six-hour token, and the participant this marks
    # present keeps `left_at IS NULL` — the predicate the roster and the
    # last-one-out check both read, so a call that has ended acquires a member who
    # never leaves. The commit below releases it, so the SFU and Redis calls that
    # follow are outside the lock.
    call = (await db.execute(select(Call).where(Call.id == call_id).with_for_update())).scalar_one_or_none()
    member = await db.get(CallParticipant, (call_id, user.id)) if call else None
    if call is None or member is None:
        raise HTTPException(status_code=404, detail="Call not found")
    if call.status not in _ACTIVE_STATUSES:
        raise HTTPException(status_code=400, detail="Call is not active")
    # A ringing call is joinable only by the person who placed it. Without this a
    # callee could POST here while their phone was still ringing and be published
    # into the room without ever accepting — the caller would hear a room they
    # believed was still ringing. All three clients only ask for a token after
    # `call:accepted` / `call:group_participants`, so this closes a hole rather
    # than changing any real flow.
    if call.status == CallStatus.ringing and call.initiated_by != user.id:
        raise HTTPException(status_code=400, detail="Call has not been answered yet")
    if member.joined_at is None:
        member.joined_at = now_utc()
    # A rejoin after a drop has to clear `left_at`, or the participant stays
    # "gone" for every subsequent roster query — including the last-one-out check
    # that ends a group call, which would then tear the room down under someone
    # who had just come back.
    member.left_at = None
    await db.commit()
    device_id = body.device_id if body else None
    identity = calls_service.identity_for(user.id, device_id)
    # One person, one audio leg. A second client of the same user joining would put
    # two microphones and two speakers into the call, which the other side hears as
    # echo and their own voice repeating with a delay. Newest join wins; the older
    # leg is removed here rather than left for the SFU to kill silently.
    await calls_service.evict_other_devices(call, user.id, identity)
    with contextlib.suppress(Exception):
        await calls_service._set_in_call([user.id], call.id)
        # Taking a token is proof of life: cancel any grace window opened by an
        # earlier socket drop, so a client that reconnects and rejoins is never
        # resolved out of its own call a moment later.
        await calls_service.handle_user_link_up(user.id)
    settings = get_settings()
    return {
        "token": mint_token(call, user, identity=identity),
        "url": settings.livekit_url,
        "room": call.room_name,
        "identity": identity,
        # Echoed so a client never has to hardcode the server's grace policy.
        "reconnect_grace_seconds": calls_service.RECONNECT_GRACE_SECONDS,
    }


async def _handle_webhook_event(db: AsyncSession, event) -> None:
    room = event.room.name or ""
    if not room.startswith("call_"):
        return
    call_id = parse_uuid(room.removeprefix("call_"))
    if call_id is None:
        return
    # FOR UPDATE for the same reason the token endpoint takes it: every branch below
    # reads `call.status` and then writes. Under READ COMMITTED an unlocked read
    # leaves the interleaving where this handler sees an active call, `_finalize`
    # commits a terminal status, and this then commits `joined_at` / `left_at = None`
    # — a participant marked present on a call that has ended, having sailed past
    # the terminal branch that would have torn the room down. `_finalize` must
    # UPDATE this row, so holding it here serializes the two.
    call = (await db.execute(select(Call).where(Call.id == call_id).with_for_update())).scalar_one_or_none()
    if call is None:
        return

    name = event.event
    if name in ("participant_joined", "participant_left") and call.status not in _ACTIVE_STATUSES:
        # Admission, enforced where it can actually be enforced.
        #
        # A join token is a JWT with a six-hour TTL and nothing can revoke it, so
        # every participant of a finished call keeps a working credential. Deleting
        # the room on teardown does not close that: LiveKit creates a room on join,
        # so a holder can walk back into a room that was deleted and — with a second
        # holder — hold a conversation with no call record, no history and nobody
        # able to see them leave.
        #
        # The token cannot carry this check, but LiveKit tells us the moment someone
        # uses one. Deleting the room again disconnects everyone in it, and does so
        # for every rejoiner rather than one identity at a time. Without this the
        # handler below would also clear `left_at`, putting a live participant back
        # onto a call that had already ended.
        if name == "participant_joined":
            logger.warning(
                "call.join_after_end call_id=%s status=%s identity=%s — "
                "stale token used, tearing the room down again",
                call.id,
                call.status.value,
                event.participant.identity,
            )
            # Release the row lock before the SFU round trip: the decision is made
            # and nothing below writes, so there is no reason to hold the call row
            # for the timeout's duration. `expire_on_commit=False` keeps `call`
            # readable afterwards.
            await db.commit()
            await calls_service._delete_room(call)
        return

    if name in ("participant_joined", "participant_left"):
        # Identities are `{user_id}#{device_id}` so one user may hold more than one
        # connection without the SFU evicting them as a duplicate
        # (services/calls.identity_for). `user_id_from_identity` also accepts a bare
        # user id, so a webhook queued by an older build still resolves.
        member_id = parse_uuid(calls_service.user_id_from_identity(event.participant.identity))
        if member_id is None:
            return
        member = await db.get(CallParticipant, (call.id, member_id))
        if member is None:
            return
        # Note: one user holding two room connections produces two joins and two
        # leaves, so a leave here can name someone still present on another device.
        # Deliberately not guarded against — both clients treat the ROOM, not this
        # frame, as the authority on the roster (see LiveKitSession.syncFromRoom and
        # livekitClient._syncParticipant), so a premature `participant_left` is
        # corrected on the next room sync, and `POST /{id}/token` clears `left_at`
        # on any rejoin.
        others = [u for u in await calls_service._call_participant_ids(db, call.id) if u != member_id]
        if name == "participant_joined":
            if member.joined_at is None:
                member.joined_at = now_utc()
            member.left_at = None
            await db.commit()
            joined_user = await db.get(User, member_id)
            if joined_user is not None:
                await publish_to_users(
                    others,
                    {
                        "type": "call:participant_joined",
                        "call_id": str(call.id),
                        "participant": calls_service._brief(joined_user),
                    },
                )
        else:
            member.left_at = now_utc()
            await db.commit()
            await publish_to_users(
                others,
                {
                    "type": "call:participant_left",
                    "call_id": str(call.id),
                    "participant_id": str(member_id),
                },
            )
    elif name == "room_finished" and call.status == CallStatus.connected:
        ids = await calls_service._call_participant_ids(db, call.id)
        await calls_service._finalize(db, call, CallStatus.answered)
        await publish_to_users(
            ids,
            {
                "type": "call:ended",
                "call_id": str(call.id),
                "reason": "room_finished",
                "duration": calls_service._duration_of(call),
            },
        )


@webhook_router.post("/api/livekit/webhook")
async def livekit_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """Server-to-server LiveKit events. Always 200: webhooks must not 500-loop.

    Unverifiable payloads are acknowledged but never processed.
    """
    try:
        body = (await request.body()).decode("utf-8")
        auth = request.headers.get("Authorization", "")
        settings = get_settings()
        verifier = lk_api.TokenVerifier(settings.livekit_api_key, settings.livekit_api_secret)
        event = lk_api.WebhookReceiver(verifier).receive(body, auth)
    except Exception:
        logger.warning("Rejected LiveKit webhook: verification failed")
        return {"message": "ignored"}
    try:
        await _handle_webhook_event(db, event)
    except Exception:
        logger.exception("LiveKit webhook handling failed")
    return {"message": "ok"}
