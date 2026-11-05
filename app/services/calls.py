"""Calling on LiveKit: the backend mints scoped room tokens and tracks
lifecycle/history; media never touches this process (SFU's job).

Replaces the hand-rolled 6-person WebRTC mesh + public TURN + blind signaling
relay. Client-facing WS events keep the RxHivexx names (call:incoming,
call:accepted, …) so the salvaged UI flow survives; SDP/ICE relay events are
gone — the LiveKit client SDK talks to the SFU directly.

Cross-worker state (who is in which call) lives in Redis.

## Reachability and survivability

Three properties this module is responsible for, because every client depends on
them and none of them can be fixed client-side:

**A ring is not gated on a live socket.** A callee whose WebSocket happens to be
absent at the instant of dialling — an iOS app that just backgrounded, a laptop
mid-Wi-Fi-handover, a tab whose 15-minute access cookie just lapsed — used to be
refused immediately with `call:unavailable`, so the phone never rang at all. The
call now rings for its whole window regardless, a Web Push is dispatched to wake a
sleeping web client, and `resume_calls_for` re-delivers `call:incoming` the
moment that user's socket reappears. "Offline" is reported as a *hint* on
`call:ringing_started`, never as a refusal.

**A dropped link is a pause, not a hang-up.** `handle_user_link_down` no longer
ends the call; it opens a grace window, tells the peers so they can show
"Connecting…", and only resolves the call if the user is still missing when the
window closes. This is what makes a call survive a lift-doors dead spot — and it
is also what stops the routine 15-minute token-expiry reconnect from killing every
call longer than the remaining cookie lifetime.

**Timers are durable and worker-agnostic.** Ring timeouts and grace windows are
Redis deadlines swept by any worker (`app.services.call_deadlines`), not asyncio
tasks in a per-process dict. See that module for what the old scheme lost.
"""

import contextlib
import datetime as dt
import logging
import uuid

import aiohttp
from livekit import api as lk_api
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import (
    Call,
    CallParticipant,
    CallStatus,
    CallType,
    Conversation,
    ConversationParticipant,
    User,
)
from app.db.session import SessionLocal
from app.realtime.redis_bus import get_redis, publish_to_users
from app.services import call_deadlines, enrich, presence
from app.utils import iso_z, now_utc

logger = logging.getLogger(__name__)

# How long a callee's device has to ring before the call is recorded as missed.
#
# Longer than the 30s the Mongo build used, on purpose. A phone that has to wake a
# suspended socket (or be woken by a push) can easily need ten seconds before it is
# even able to display the ringer, and a window that expires while the device is
# still waking is indistinguishable, to the user, from a call that was never
# delivered.
RING_TIMEOUT_SECONDS = 45

# How long a participant whose link dropped mid-call may be absent before the call
# is resolved without them.
#
# Sized against the things that actually cause it: a cellular-to-Wi-Fi handover
# (1–5s), a client refreshing an expired access cookie and reopening its socket
# (1–3s), a tunnel or lift dead spot (5–20s), and an iOS app returning from a brief
# backgrounding. 40s covers all of those while still resolving a genuinely dead
# client well inside the time a user would wait before giving up and redialling.
RECONNECT_GRACE_SECONDS = 40

# How long a deadline waits before being tried again after its handler raised.
# Claiming is destructive — claim_due ZREMs before the handler runs — so without
# putting it back a failed fire is a lost fire, and the row it was going to
# resolve stays mid-state forever. Short, because everything a deadline resolves
# is something a user is actively waiting on.
DEADLINE_RETRY_SECONDS = 5

GROUP_CALL_CAP = 32
_CALL_TTL = 24 * 3600
# Long enough to outlive any grace window, short enough that a leaked key expires
# on its own rather than pinning a user "in a call" forever.
_LINK_TTL = 10 * 60

# Every distinct state a client may be told a peer's link is in. Kept as constants
# because all three clients switch on the exact strings.
LINK_UP = "up"
LINK_DOWN = "down"


def _user_key(user_id) -> str:
    return f"call:user:{user_id}"


def _conv_key(conversation_id) -> str:
    return f"call:conv:{conversation_id}"


def _link_key(call_id, user_id) -> str:
    return f"call:link:{call_id}:{user_id}"


def _grace_member(call_id, user_id) -> str:
    return f"{call_id}:{user_id}"


def room_name_for(call_id) -> str:
    return f"call_{call_id}"


async def _set_in_call(user_ids, call_id) -> None:
    redis = get_redis()
    pipe = redis.pipeline()
    for uid in user_ids:
        pipe.set(_user_key(uid), str(call_id), ex=_CALL_TTL)
    await pipe.execute()


async def _clear_in_call(user_ids, call_id) -> None:
    redis = get_redis()
    for uid in user_ids:
        current = await redis.get(_user_key(uid))
        if current == str(call_id):
            await redis.delete(_user_key(uid))


async def current_call_of(user_id) -> str | None:
    return await get_redis().get(_user_key(user_id))


def _brief(user: User) -> dict:
    return {
        "id": str(user.id),
        "display_name": user.display_name,
        "avatar_url": user.avatar_url,
    }


async def _load_call(db: AsyncSession, call_id: uuid.UUID) -> Call | None:
    stmt = select(Call).where(Call.id == call_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def _call_participant_ids(db: AsyncSession, call_id: uuid.UUID) -> list[uuid.UUID]:
    return (
        (await db.execute(select(CallParticipant.user_id).where(CallParticipant.call_id == call_id)))
        .scalars()
        .all()
    )


# ---------------------------------------------------------------------------
# LiveKit identity
# ---------------------------------------------------------------------------

# A LiveKit identity must be unique per *connection*, and ours used to be the bare
# user id — so a user with two clients in one room had the second connection evict
# the first as a duplicate identity, silently, with no error surfaced anywhere.
# Both clients grew elaborate "only the device that pressed Accept may join"
# guards to work around it (see the comments in websocket.js and CallStore.swift),
# and those guards are load-bearing only because of this. Suffixing the device
# makes a second client ADDITIVE instead of destructive: the guards remain, but
# now they only decide which device stops ringing, which is a UX question rather
# than the difference between a working call and a dead one.
IDENTITY_SEPARATOR = "#"

# Control-plane calls to the SFU take single-digit milliseconds when it is healthy.
# The LiveKit client's default is a 60-second total timeout, which on these two
# paths is not a timeout so much as a stall: an SFU that accepts a connection and
# then stops answering would hold POST /{id}/token for a minute before the caller
# got a token, and delay every participant's `call:ended` by the same, leaving
# clients sitting in a call the server had already finished ending. Both callers
# are best-effort, so giving up quickly is strictly better than waiting.
_SFU_TIMEOUT = aiohttp.ClientTimeout(total=5)


def identity_for(user_id, device_id: str | None) -> str:
    """`{user_id}#{device_id}` — unique per client, resolvable back to the user."""
    suffix = (device_id or uuid.uuid4().hex)[:64]
    return f"{user_id}{IDENTITY_SEPARATOR}{suffix}"


def user_id_from_identity(identity: str | None) -> str:
    """The user half of a room identity.

    Tolerates a bare user id so a client (or a LiveKit webhook replay) from before
    the suffix existed still resolves.
    """
    return (identity or "").split(IDENTITY_SEPARATOR, 1)[0]


async def evict_other_devices(call: Call, user_id, keep_identity: str) -> int:
    """Remove this user's OTHER connections from the room before they join again.

    One person must be exactly one audio leg. When the identity was the bare user
    id the SFU enforced that for us, by evicting the earlier connection as a
    duplicate — badly (silently, and it killed whichever client happened to connect
    first), but it did enforce it. Per-device identities removed that side effect,
    and with it the guarantee: a user signed in on two clients could put two
    microphones and two speakers into one call.

    That does not sound like a duplicate participant to the person on the other end.
    It sounds like **echo, and their own voice coming back several times with a
    delay** — one copy per extra leg, each on its own jitter path. If the two clients
    are in the same room (or on the same machine, sharing a microphone) it becomes a
    feedback loop that no amount of echo cancellation can fix, because each device's
    AEC only knows about its own output, never the other's.

    So the rule is now enforced here, deliberately and visibly: newest join wins,
    older legs are removed by the server and logged. Best-effort — a failure to
    reach the SFU must not stop a legitimate join, since the common case is a first
    join with nothing to evict.
    """
    settings = get_settings()
    base = settings.livekit_probe_url
    if not base:
        return 0
    removed = 0
    try:
        client = lk_api.LiveKitAPI(
            base, settings.livekit_api_key, settings.livekit_api_secret, timeout=_SFU_TIMEOUT
        )
        try:
            existing = await client.room.list_participants(
                lk_api.ListParticipantsRequest(room=call.room_name)
            )
            for p in existing.participants:
                if p.identity == keep_identity:
                    continue
                if user_id_from_identity(p.identity) != str(user_id):
                    continue
                logger.info(
                    "call.evicting_stale_leg call_id=%s user_id=%s identity=%s",
                    call.id,
                    user_id,
                    p.identity,
                )
                # Per leg, and at WARNING. The outer catch below is written for the
                # benign case — a room that does not exist yet — and once a stale
                # leg has actually been identified that reading is gone: this user
                # has two microphones in the call and the other side is hearing
                # their own voice come back. Sharing one DEBUG line with the
                # first-join case made a broken echo invariant invisible.
                #
                # Scoped per leg for the second reason too: a single leg that
                # refuses to go used to abort the loop, so a user with three stale
                # legs kept two of them.
                try:
                    await client.room.remove_participant(
                        lk_api.RoomParticipantIdentity(room=call.room_name, identity=p.identity)
                    )
                    removed += 1
                except Exception:
                    logger.warning(
                        "call.evict_failed call_id=%s user_id=%s identity=%s — "
                        "duplicate audio leg remains, expect echo",
                        call.id,
                        user_id,
                        p.identity,
                        exc_info=True,
                    )
        finally:
            await client.aclose()
    except Exception:
        # Reaching the SFU, or listing the room. A room that does not exist yet is
        # the normal first-join case and raises here; nothing to evict either way,
        # which is why this stays at DEBUG while a failed removal does not.
        logger.debug("evict_other_devices skipped", exc_info=True)
    return removed


async def _delete_room(call: Call) -> bool:
    """Tear down the media room. A call is over when the ROOM is gone, not the row.

    Ending a call used to be a database-and-Redis operation only: status set,
    `in_call` keys cleared, timers cancelled — and the SFU room left running. That
    matters because a room is the only thing a client can actually observe. Tokens
    are JWTs with a six-hour TTL and nothing can revoke them, so every participant
    of a call that just ended holds a working credential for a room the server
    still serves. Rejoining it recreates a live call that no longer exists in the
    database: no ring, no roster, no history, and no participant the other side can
    see leave.

    LiveKit does eventually reap an empty room by itself, but `empty_timeout` only
    starts once the room is empty — a client that rejoins keeps resetting it.

    Best-effort, like eviction: a failure here must not stop a call from ending.
    Logged at WARNING rather than DEBUG, because a room that outlived its call is a
    condition someone should be able to see in production, and the failure is
    invisible to every client until one of them rejoins.
    """
    settings = get_settings()
    base = settings.livekit_probe_url
    if not base or not call.room_name:
        return False
    try:
        client = lk_api.LiveKitAPI(
            base, settings.livekit_api_key, settings.livekit_api_secret, timeout=_SFU_TIMEOUT
        )
        try:
            await client.room.delete_room(lk_api.DeleteRoomRequest(room=call.room_name))
            logger.info("call.room_deleted call_id=%s room=%s", call.id, call.room_name)
            return True
        finally:
            await client.aclose()
    except Exception:
        logger.warning(
            "call.room_delete_failed call_id=%s room=%s — room may outlive the call",
            call.id,
            call.room_name,
            exc_info=True,
        )
        return False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def _finalize(
    db: AsyncSession,
    call: Call,
    status: CallStatus,
    *,
    system_note: str | None = None,
) -> None:
    call.status = status
    call.ended_at = now_utc()
    await db.commit()
    # Before the Redis bookkeeping: the room is the part a client can still act on,
    # and every other line here only affects what the server believes.
    await _delete_room(call)
    ids = await _call_participant_ids(db, call.id)
    await _clear_in_call(ids, call.id)
    if call.conversation_id:
        await get_redis().delete(_conv_key(call.conversation_id))

    # Drop every timer this call owns. Both are idempotent and both are durable, so
    # unlike the asyncio-task scheme this replaced there is no way to cancel the
    # task we are currently running on and silently abort the rest of this function.
    await call_deadlines.cancel(call_deadlines.RING, str(call.id))
    redis = get_redis()
    for uid in ids:
        await call_deadlines.cancel(call_deadlines.GRACE, _grace_member(call.id, uid))
        with contextlib.suppress(Exception):
            await redis.delete(_link_key(call.id, uid))

    if system_note and call.conversation_id:
        from app.services.messaging import send_system_message

        with contextlib.suppress(Exception):
            await send_system_message(db, call.conversation_id, system_note)


def _duration_of(call: Call) -> int:
    if not call.answered_at or not call.ended_at:
        return 0
    a = call.answered_at if call.answered_at.tzinfo else call.answered_at.replace(tzinfo=dt.UTC)
    e = call.ended_at if call.ended_at.tzinfo else call.ended_at.replace(tzinfo=dt.UTC)
    return max(0, int((e - a).total_seconds()))


def _fmt_duration(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def _call_emoji(call: Call) -> str:
    return "📹" if call.type == CallType.video else "📞"


async def _ring_timeout(call_id: uuid.UUID) -> None:
    """The ring window closed. Idempotent: a no-op unless the call is still ringing."""
    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        if call is None or call.status != CallStatus.ringing:
            return
        ids = await _call_participant_ids(db, call_id)
        caller_id = call.initiated_by
        callee_ids = [u for u in ids if u != caller_id]
        caller = await db.get(User, caller_id) if caller_id else None
        await _finalize(db, call, CallStatus.missed, system_note=f"{_call_emoji(call)} Missed call")
        logger.info("call.ring_timeout call_id=%s caller_id=%s", call_id, caller_id)
        if caller_id:
            await publish_to_users([caller_id], {"type": "call:missed", "call_id": str(call_id)})
        await publish_to_users(
            callee_ids,
            {
                "type": "call:missed",
                "call_id": str(call_id),
                "caller": _brief(caller) if caller else None,
                "call_type": call.type.value,
            },
        )


async def initiate_direct_call(
    user: User, callee_id: uuid.UUID, call_type: str, conversation_id: str | None
) -> None:
    async with SessionLocal() as db:
        caller = await db.get(User, user.id)
        callee = await db.get(User, callee_id)
        if callee is None or not callee.is_active or callee.org_id != caller.org_id:
            await publish_to_users([caller.id], {"type": "call:error", "message": "User not available"})
            return
        if await current_call_of(caller.id):
            await publish_to_users([caller.id], {"type": "call:error", "message": "Already in a call"})
            return

        from fastapi import HTTPException

        from app.services.conversations import get_or_create_direct

        try:
            # Calls are a second reachability channel with their own org check,
            # so a chat-only policy would have let a blocked pair ring each other
            # over LiveKit. Routing through the shared helper closes that.
            #
            # Caught rather than allowed to propagate: this runs in a background
            # task and reports failure over the socket, so an escaping
            # HTTPException would be logged and the caller's phone would simply
            # ring forever with no error — the same shape as the two guards above.
            conv = await get_or_create_direct(db, caller, callee)
        except HTTPException:
            await publish_to_users(
                [caller.id],
                {"type": "call:error", "message": "You are not permitted to call this person"},
            )
            return

        ctype = CallType.video if call_type == "video" else CallType.voice
        call = Call(
            org_id=caller.org_id,
            conversation_id=conv.id,
            initiated_by=caller.id,
            type=ctype,
            status=CallStatus.ringing,
            is_group=False,
            room_name="",
        )
        db.add(call)
        await db.flush()
        call.room_name = room_name_for(call.id)
        db.add_all(
            [
                CallParticipant(call_id=call.id, user_id=caller.id),
                CallParticipant(call_id=call.id, user_id=callee.id),
            ]
        )
        await db.commit()
        call_id = call.id

        if await current_call_of(callee.id):
            await _finalize(db, call, CallStatus.busy)
            await publish_to_users([caller.id], {"type": "call:busy", "call_id": str(call_id)})
            return

        # Deliberately NOT gated on presence any more.
        #
        # This used to finalise the call as `no_answer` and answer `call:unavailable`
        # the instant `presence.is_online` was false, which made an absent socket —
        # not an absent person — the thing that refused the call. Every client has
        # windows where that is true through no fault of the user: iOS tears its
        # socket down on backgrounding, the server closes every socket 4001 when the
        # 15-minute access cookie lapses, a Wi-Fi handover costs a couple of seconds,
        # and `is_online` degrades to False on a Redis blip. Each of those presented
        # as "the call was never received".
        #
        # So: ring for the full window regardless. Presence is now a *hint* on
        # `call:ringing_started` (so the caller's UI can say "they may be offline"
        # rather than lie), a push is dispatched to wake a sleeping web client, and
        # `resume_calls_for` re-delivers the ring the moment the callee's socket
        # reappears — which is the case that used to be unrecoverable.
        callee_online = await presence.is_online(callee.id)

        await _set_in_call([caller.id, callee.id], call_id)
        incoming = {
            "type": "call:incoming",
            "call_id": str(call_id),
            "caller": _brief(caller),
            "call_type": ctype.value,
            "conversation_id": str(conv.id),
        }
        await publish_to_users([callee.id], incoming)
        await publish_to_users(
            [caller.id],
            {
                "type": "call:ringing_started",
                "call_id": str(call_id),
                "callee_id": str(callee.id),
                "callee_online": callee_online,
                "ring_timeout": RING_TIMEOUT_SECONDS,
            },
        )
        await call_deadlines.schedule(call_deadlines.RING, str(call_id), RING_TIMEOUT_SECONDS)
        if not callee_online:
            _dispatch_call_push(callee.id, caller, ctype, call_id)
        logger.info(
            "call.initiated call_id=%s caller_id=%s callee_id=%s type=%s callee_online=%s",
            call_id,
            caller.id,
            callee.id,
            ctype.value,
            callee_online,
        )


def _dispatch_call_push(callee_id, caller: User, ctype: CallType, call_id) -> None:
    """Wake a web client that has no live socket.

    Best-effort and fire-and-forget by design (`services/push.py`), and a no-op
    when VAPID is unconfigured. It is the only channel that reaches a callee whose
    socket is gone, and until this existed a call to such a user was silently
    undeliverable — the ring frame was published to a channel nobody was
    subscribed to and simply evaporated.

    iOS has no equivalent: waking a suspended iOS app for a call needs an APNs VoIP
    push (PushKit + CallKit), and `push_subscriptions` stores Web Push endpoints
    only. See docs/CALLS.md for what that would take.
    """
    with contextlib.suppress(Exception):
        from app.services.push import dispatch_push_to_users

        label = "Incoming video call" if ctype == CallType.video else "Incoming voice call"
        dispatch_push_to_users(
            [callee_id],
            {
                "title": caller.display_name,
                "body": label,
                # A distinct tag so a ring never collapses into a message
                # notification, and `kind` so sw.js can make it persistent.
                "tag": f"call:{call_id}",
                "kind": "call",
                "call_id": str(call_id),
                "url": "/chat",
            },
        )


async def initiate_group_call(user: User, conversation_id: uuid.UUID, call_type: str) -> None:
    async with SessionLocal() as db:
        caller = await db.get(User, user.id)
        conv = await db.get(Conversation, conversation_id)
        me = await db.get(ConversationParticipant, (conversation_id, caller.id))
        if conv is None or me is None or not conv.is_active:
            await publish_to_users([caller.id], {"type": "call:error", "message": "Conversation not found"})
            return
        if conv.type.value != "cross_org" and conv.org_id != caller.org_id:
            await publish_to_users([caller.id], {"type": "call:error", "message": "Access denied"})
            return

        redis = get_redis()
        existing = await redis.get(_conv_key(conversation_id))
        if existing:
            await publish_to_users(
                [caller.id],
                {
                    "type": "call:group_already_active",
                    "call_id": existing,
                    "conversation_id": str(conversation_id),
                },
            )
            return
        if await current_call_of(caller.id):
            await publish_to_users([caller.id], {"type": "call:error", "message": "Already in a call"})
            return

        # The ROSTER, not just the caller. The caller's org is checked above; these
        # rows were not, and the same asymmetry that ran through api/messages.py ran
        # here — except this is worse than a fan-out audience. A CallParticipant row
        # is created for every id below, so a foreign-org participant row in an
        # ordinary conversation was handed the caller's identity, the conversation id
        # and the group name, AND a seat it could use to join the audio.
        #
        # enrich.tenant_participants is the canonical filter: it admits a foreign org
        # only for a cross_org conversation, which is exactly where a group call is
        # meant to span tenants.
        member_ids = [p.user_id for p in await enrich.tenant_participants(db, conversation_id)]

        ctype = CallType.video if call_type == "video" else CallType.voice
        now = now_utc()
        call = Call(
            org_id=conv.org_id or caller.org_id,
            conversation_id=conversation_id,
            initiated_by=caller.id,
            type=ctype,
            status=CallStatus.connected,
            is_group=True,
            room_name="",
            answered_at=now,
        )
        db.add(call)
        await db.flush()
        call.room_name = room_name_for(call.id)
        for uid in member_ids:
            db.add(CallParticipant(call_id=call.id, user_id=uid, joined_at=now if uid == caller.id else None))
        await db.commit()
        call_id = call.id

        await redis.set(_conv_key(conversation_id), str(call_id), ex=_CALL_TTL)
        await _set_in_call([caller.id], call_id)

        others = [u for u in member_ids if u != caller.id]
        await publish_to_users(
            others,
            {
                "type": "call:incoming",
                "call_id": str(call_id),
                "caller": _brief(caller),
                "call_type": ctype.value,
                "conversation_id": str(conversation_id),
                "is_group": True,
                "group_name": conv.name or "Group call",
            },
        )
        await publish_to_users(
            others,
            {
                "type": "call:group_active",
                "call_id": str(call_id),
                "conversation_id": str(conversation_id),
                "participants": [_brief(caller)],
                "call_type": ctype.value,
            },
        )
        await publish_to_users(
            [caller.id],
            {
                "type": "call:group_started",
                "call_id": str(call_id),
                "conversation_id": str(conversation_id),
            },
        )
        logger.info(
            "call.group_initiated call_id=%s caller_id=%s conversation_id=%s members=%d",
            call_id,
            caller.id,
            conversation_id,
            len(member_ids),
        )


async def invite_to_call(user: User, call_id: uuid.UUID, invitee_ids: list[uuid.UUID]) -> dict:
    """Pull people into a group call that is already running.

    The missing half of "group calling". `_join` admits only users who already hold a
    `CallParticipant` row, and those rows are created once, from the conversation's
    membership, at the moment the call starts. So before this existed a group call was
    a closed set decided at second zero: anyone who was not in the conversation could
    not be added, and the "Add people" buttons on the call screens had nothing to call
    and were wired to nothing.

    Rules, and why:

      * **Only someone already in the call may invite.** Otherwise a call becomes a way
        to make anyone's phone ring by knowing a call id.
      * **Group calls only.** Adding a third party to a 1:1 would silently change what
        both people agreed to be in; promoting a direct call to a group is a different
        feature with its own consent question.
      * **Same org**, checked per invitee against the inviter, exactly as
        `initiate_direct_call` does — a call must not become a second reachability
        channel that skips tenant isolation.
      * **Ringing, not conscription.** An invitee gets `call:incoming` and chooses. They
        are added to the roster as an invited-but-not-joined participant, which is the
        same state a conversation member starts in, so `_join` then accepts them
        through the ordinary path.

    Returns a per-invitee outcome so the caller's UI can say which ones it could not
    reach, rather than reporting a bare success for a partial result.
    """
    outcome: dict[str, str] = {}
    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        if call is None or call.status != CallStatus.connected:
            return {"error": "not_joinable"}
        if not call.is_group:
            return {"error": "not_a_group_call"}
        if await db.get(CallParticipant, (call_id, user.id)) is None:
            return {"error": "not_a_participant"}

        inviter = await db.get(User, user.id)
        existing = set(await _call_participant_ids(db, call_id))
        joined_count = len(
            (
                await db.execute(
                    select(CallParticipant.user_id).where(
                        CallParticipant.call_id == call_id,
                        CallParticipant.joined_at.is_not(None),
                        CallParticipant.left_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )

        # Loaded before the loop because the org rule below turns on the
        # conversation's type, not only on the two users. Same string comparison
        # initiate_group_call uses, so no new import.
        conv = await db.get(Conversation, call.conversation_id) if call.conversation_id else None
        is_cross_org = conv is not None and conv.type.value == "cross_org"

        added: list[User] = []
        for invitee_id in invitee_ids:
            row = await db.get(CallParticipant, (call_id, invitee_id))
            if row is not None and row.joined_at is not None and row.left_at is None:
                # In the room right now. Nothing to do, and saying so is more useful to
                # the inviter than a bare success would be.
                outcome[str(invitee_id)] = "already_invited"
                continue
            invitee = await db.get(User, invitee_id)
            if invitee is None or not invitee.is_active:
                outcome[str(invitee_id)] = "unavailable"
                continue
            # Same org — except in a cross-org conversation, where the roster already
            # spans organizations: initiate_group_call builds it from every
            # conversation member and exempts cross_org from this very check, exactly
            # as deps.require_membership does. A flat comparison here refused to
            # (re-)invite a participant from the other org who had not answered or had
            # left — the single case this function exists to serve, and one whose only
            # other way back is a Join chip in a conversation an outside invitee does
            # not have. Conversation membership stands in for the org check, so this
            # still cannot ring anyone who was never in the conversation.
            if is_cross_org:
                if await db.get(ConversationParticipant, (conv.id, invitee_id)) is None:
                    outcome[str(invitee_id)] = "different_org"
                    continue
            elif invitee.org_id != inviter.org_id:
                outcome[str(invitee_id)] = "different_org"
                continue
            # The cap counts people IN the call, plus the ones being added — an invite
            # that would overfill the room has to be refused now rather than at their
            # `_join`, when they are already looking at a ringing screen.
            if joined_count + len(added) + 1 > GROUP_CALL_CAP:
                outcome[str(invitee_id)] = "call_full"
                continue
            if row is None:
                db.add(CallParticipant(call_id=call_id, user_id=invitee.id))
            else:
                # On the roster but not in the room: they never answered, or they left.
                # Ringing them again is the whole point of inviting them — reporting
                # `already_invited` and doing nothing (which is what this did) left the
                # only way back into a call you had left as a Join chip in the
                # conversation, which someone invited from outside it does not have.
                # `left_at` is cleared so the roster stops counting them as gone before
                # they answer; `_join` sets `joined_at` when they actually arrive.
                row.left_at = None
            added.append(invitee)
            outcome[str(invitee_id)] = "invited"

        if not added:
            await db.commit()
            return {"invited": [], "outcome": outcome}
        await db.commit()

        group_name = (conv.name if conv else None) or "Group call"
        roster = [
            _brief(u)
            for u in (await db.execute(select(User).where(User.id.in_(existing | {u.id for u in added}))))
            .scalars()
            .all()
        ]
        ids_now = list(existing | {u.id for u in added})

    ring = {
        "type": "call:incoming",
        "call_id": str(call_id),
        "caller": _brief(inviter),
        "call_type": call.type.value,
        "conversation_id": str(call.conversation_id) if call.conversation_id else None,
        "is_group": True,
        "group_name": group_name,
        "invited": True,
    }
    for invitee in added:
        await publish_to_users([invitee.id], ring)
        # Also offered as a joinable call in the conversation, so a "Join call" chip
        # is there for them after the ringing screen has gone.
        await publish_to_users(
            [invitee.id],
            {
                "type": "call:group_active",
                "call_id": str(call_id),
                "conversation_id": str(call.conversation_id) if call.conversation_id else None,
                "participants": roster,
                "call_type": call.type.value,
            },
        )
        if not await presence.is_online(invitee.id):
            _dispatch_call_push(invitee.id, inviter, call.type, call_id)

    # Everyone already in the call learns who is on the way, so the roster shows a
    # placeholder tile rather than someone appearing from nowhere when they answer.
    await publish_to_users(
        [uid for uid in ids_now if uid not in {u.id for u in added}],
        {
            "type": "call:participants_invited",
            "call_id": str(call_id),
            "invited_by": str(inviter.id),
            "participants": [_brief(u) for u in added],
        },
    )
    logger.info(
        "call.invited call_id=%s inviter_id=%s added=%d outcome=%s",
        call_id,
        inviter.id,
        len(added),
        outcome,
    )
    return {"invited": [str(u.id) for u in added], "outcome": outcome}


async def _accept(user: User, call_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        # Every rejection path answers the accepter.
        #
        # These were bare `return`s, which is why "the call never connects" had no
        # error anywhere: a client that pressed Accept on a call the server had
        # already finalised — rung out, cancelled, or ended — got no frame at all
        # and sat on "Connecting" indefinitely. It could not tell a dead call from
        # a dropped frame. `call:error` gives the client something to act on.
        if call is None:
            await publish_to_users(
                [user.id],
                {"type": "call:error", "call_id": str(call_id), "reason": "not_found"},
            )
            return
        member = await db.get(CallParticipant, (call_id, user.id))
        if member is None or user.id == call.initiated_by:
            await publish_to_users(
                [user.id],
                {"type": "call:error", "call_id": str(call_id), "reason": "not_a_participant"},
            )
            return
        if call.status == CallStatus.connected:
            # Already accepted — almost always this device retrying after a socket
            # blip swallowed the first `call:accept`, or a second Accept tap. Replay
            # the acceptance instead of refusing it: refusing left the answering
            # client on "Connecting…" for a call that was in fact live and joinable.
            logger.info("call.accept.replay call_id=%s user_id=%s", call_id, user.id)
            await publish_to_users(
                [user.id],
                {"type": "call:accepted", "call_id": str(call_id), "accepter_id": str(user.id)},
            )
            return
        if call.status != CallStatus.ringing:
            logger.info(
                "call.accept.stale call_id=%s user_id=%s status=%s",
                call_id,
                user.id,
                call.status.value,
            )
            await publish_to_users(
                [user.id],
                {
                    "type": "call:error",
                    "call_id": str(call_id),
                    "reason": "no_longer_ringing",
                    "status": call.status.value,
                },
            )
            return
        call.status = CallStatus.connected
        call.answered_at = now_utc()
        await db.commit()
        await call_deadlines.cancel(call_deadlines.RING, str(call_id))
        logger.info(
            "call.accepted call_id=%s accepter_id=%s caller_id=%s",
            call_id,
            user.id,
            call.initiated_by,
        )
        # Both sides get this: the caller needs to know who answered, and the
        # accepter needs the server's confirmation that the call really moved to
        # connected before joining the SFU room. Sending it only to the caller
        # left the callee never joining — a connected-looking but silent call.
        #
        # It reaches every SOCKET of both users, because the bus is keyed per user
        # (redis_bus.publish_to_users -> hub fan-out). That is deliberate — the
        # accepter's other devices must stop ringing. Since `mint_token` now issues
        # a per-device identity, a second device that joins anyway no longer EVICTS
        # the first; the client-side guards keyed on `accepter_id` remain so that
        # only the answering device enters the call, but a slip there now costs an
        # extra tile rather than a silently dead call.
        await publish_to_users(
            [call.initiated_by, user.id],
            {"type": "call:accepted", "call_id": str(call_id), "accepter_id": str(user.id)},
        )


async def _decline(user: User, call_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        if call is None:
            return
        member = await db.get(CallParticipant, (call_id, user.id))
        if member is None or user.id == call.initiated_by:
            return
        if call.is_group:
            # Declining a GROUP call used to be a complete no-op.
            #
            # A group call is created `connected` (initiate_group_call), never
            # `ringing`, so it fell straight through the status check below: nobody was
            # told, and the decliner kept a roster row marked as somebody still
            # expected to arrive. That is what a "Ringing…" placeholder reads from, so
            # it would have sat there for the rest of the call, and the last-one-out
            # check counted a person who had said no.
            #
            # One person declining must not end a call the others are in — so this is a
            # per-participant answer, not a `_finalize`. Somebody who has already
            # joined cannot decline; that is a leave, and pressing hang up sends one.
            if call.status != CallStatus.connected or member.joined_at is not None:
                return
            member.left_at = now_utc()
            await db.commit()
            await call_deadlines.cancel(call_deadlines.GRACE, _grace_member(call_id, user.id))
            ids = await _call_participant_ids(db, call_id)
            decliner = await db.get(User, user.id)
            logger.info("call.group.declined call_id=%s user_id=%s", call_id, user.id)
            await publish_to_users(
                [u for u in ids if u != user.id],
                {
                    "type": "call:participant_declined",
                    "call_id": str(call_id),
                    "participant_id": str(user.id),
                    "participant": _brief(decliner) if decliner else None,
                },
            )
            return
        if call.status != CallStatus.ringing:
            return
        await _finalize(db, call, CallStatus.declined, system_note=f"{_call_emoji(call)} Call declined")
        await publish_to_users([call.initiated_by], {"type": "call:declined", "call_id": str(call_id)})


async def _cancel(user: User, call_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        if call is None or call.status != CallStatus.ringing or call.initiated_by != user.id:
            return
        ids = await _call_participant_ids(db, call_id)
        await _finalize(db, call, CallStatus.cancelled)
        await publish_to_users(
            [u for u in ids if u != user.id], {"type": "call:cancelled", "call_id": str(call_id)}
        )


async def _end(user: User, call_id: uuid.UUID, reason: str = "normal") -> None:
    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        if call is None:
            return
        member = await db.get(CallParticipant, (call_id, user.id))
        if member is None:
            return
        if call.status not in (CallStatus.ringing, CallStatus.connected):
            return
        if call.status == CallStatus.ringing and call.initiated_by == user.id:
            await _cancel(user, call_id)
            return
    # In a group call, an ordinary participant hanging up is a LEAVE — it must
    # not tear down the room for everyone. Only the initiator (or the last one
    # out, handled in _leave) ends the whole call.
    if call.is_group and call.initiated_by != user.id:
        await _leave(user, call_id)
        return
    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        if call is None or call.status not in (CallStatus.ringing, CallStatus.connected):
            return
        member = await db.get(CallParticipant, (call_id, user.id))
        if member is None:
            return
        member.left_at = now_utc()
        ids = await _call_participant_ids(db, call_id)
        final = CallStatus.answered if call.answered_at else CallStatus.cancelled
        duration_note = None
        if call.answered_at:
            call.ended_at = now_utc()
            duration_note = (
                f"{_call_emoji(call)} "
                f"{'Video' if call.type == CallType.video else 'Voice'} call · "
                f"{_fmt_duration(_duration_of(call))}"
            )
        await _finalize(db, call, final, system_note=duration_note)
        logger.info("call.ended call_id=%s by=%s reason=%s", call_id, user.id, reason)
        await publish_to_users(
            [u for u in ids if u != user.id],
            {
                "type": "call:ended",
                "call_id": str(call_id),
                "reason": reason,
                "duration": _duration_of(call),
            },
        )


async def _join(user: User, call_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        if call is None or call.status != CallStatus.connected or not call.is_group:
            # A join that cannot be honoured has to say so. Returning silently left
            # the joiner's UI on "Connecting…" with nothing to act on, which is the
            # same dead end the accept path used to have.
            await publish_to_users(
                [user.id],
                {
                    "type": "call:error",
                    "call_id": str(call_id),
                    "reason": "not_joinable",
                    "status": call.status.value if call else "not_found",
                },
            )
            return
        member = await db.get(CallParticipant, (call_id, user.id))
        if member is None:
            # only invited conversation members may join
            await publish_to_users(
                [user.id],
                {"type": "call:error", "call_id": str(call_id), "reason": "not_a_participant"},
            )
            return
        joined = (
            (
                await db.execute(
                    select(CallParticipant).where(
                        CallParticipant.call_id == call_id,
                        CallParticipant.joined_at.is_not(None),
                        CallParticipant.left_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        if len(joined) >= GROUP_CALL_CAP and not any(p.user_id == user.id for p in joined):
            await publish_to_users(
                [user.id],
                {
                    "type": "call:full",
                    "call_id": str(call_id),
                    "message": f"Call is full ({GROUP_CALL_CAP}/{GROUP_CALL_CAP} participants)",
                },
            )
            return
        member.joined_at = member.joined_at or now_utc()
        member.left_at = None
        await db.commit()
        await _set_in_call([user.id], call_id)
        await _mark_link(call_id, user.id, LINK_UP)

        users = {}
        joined_ids = [p.user_id for p in joined if p.user_id != user.id]
        if joined_ids:
            rows = (await db.execute(select(User).where(User.id.in_(joined_ids)))).scalars().all()
            users = {u.id: u for u in rows}
        me = await db.get(User, user.id)
        await publish_to_users(
            [user.id],
            {
                "type": "call:group_participants",
                "call_id": str(call_id),
                "participants": [_brief(users[uid]) for uid in joined_ids if uid in users],
            },
        )
        await publish_to_users(
            joined_ids,
            {
                "type": "call:participant_joined",
                "call_id": str(call_id),
                "participant": _brief(me),
            },
        )


async def _leave(user: User, call_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        if call is None:
            return
        member = await db.get(CallParticipant, (call_id, user.id))
        if member is None:
            return
        if not call.is_group:
            await _end(user, call_id, reason="normal")
            return
        member.left_at = now_utc()
        await db.commit()
        await _clear_in_call([user.id], call_id)
        await call_deadlines.cancel(call_deadlines.GRACE, _grace_member(call_id, user.id))
        remaining = (
            (
                await db.execute(
                    select(CallParticipant.user_id).where(
                        CallParticipant.call_id == call_id,
                        CallParticipant.joined_at.is_not(None),
                        CallParticipant.left_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )
        ids = await _call_participant_ids(db, call_id)
        await publish_to_users(
            [u for u in ids if u != user.id],
            {"type": "call:participant_left", "call_id": str(call_id), "participant_id": str(user.id)},
        )
        if not remaining and call.status == CallStatus.connected:
            duration_note = (
                f"{_call_emoji(call)} Group call · {_fmt_duration(_duration_of(call))}"
                if call.answered_at
                else None
            )
            call.ended_at = now_utc()
            await _finalize(db, call, CallStatus.answered, system_note=duration_note)
            await publish_to_users(
                ids,
                {
                    "type": "call:group_ended",
                    "call_id": str(call_id),
                    "conversation_id": str(call.conversation_id) if call.conversation_id else None,
                },
            )


async def _toggle_media(user: User, call_id: uuid.UUID, media_type: str, enabled: bool) -> None:
    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        # Gated on the call being live. Without this a stale client could keep
        # relaying mute state into a finished call, flipping icons on a UI that has
        # already moved on to the next one.
        if call is None or call.status not in (CallStatus.ringing, CallStatus.connected):
            return
        member = await db.get(CallParticipant, (call_id, user.id))
        if member is None:
            return
        ids = await _call_participant_ids(db, call_id)
    await publish_to_users(
        [u for u in ids if u != user.id],
        {
            "type": "call:media_toggle",
            "call_id": str(call_id),
            "user_id": str(user.id),
            "media_type": "audio" if media_type == "audio" else "video",
            "enabled": bool(enabled),
        },
    )


# ---------------------------------------------------------------------------
# Link state: "this user's connection is struggling", relayed to their peers
# ---------------------------------------------------------------------------

_PEER_STATES = {"connected", "reconnecting"}
_PEER_QUALITIES = {"excellent", "good", "poor", "unknown"}


async def _relay_peer_state(user: User, call_id: uuid.UUID, state: str | None, quality: str | None) -> None:
    """Tell the other participants how this client's own media link is doing.

    The SFU knows a peer is reconnecting but does not tell the *other* peers, and
    LiveKit's connection-quality events are local-only. So a user whose network is
    failing looked completely normal from the other side — frozen video with a
    running duration timer and no explanation. This is the one frame that closes
    that gap, and it is what lets both ends show "Connecting…" for the same event.
    """
    state = state if state in _PEER_STATES else None
    quality = quality if quality in _PEER_QUALITIES else None
    if state is None and quality is None:
        return
    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        if call is None or call.status not in (CallStatus.ringing, CallStatus.connected):
            return
        if await db.get(CallParticipant, (call_id, user.id)) is None:
            return
        ids = await _call_participant_ids(db, call_id)
    payload = {
        "type": "call:peer_state",
        "call_id": str(call_id),
        "user_id": str(user.id),
    }
    if state is not None:
        payload["state"] = state
    if quality is not None:
        payload["quality"] = quality
    await publish_to_users([u for u in ids if u != user.id], payload)


async def _mark_link(call_id, user_id, state: str) -> None:
    redis = get_redis()
    with contextlib.suppress(Exception):
        if state == LINK_UP:
            await redis.delete(_link_key(call_id, user_id))
        else:
            await redis.set(_link_key(call_id, user_id), state, ex=_LINK_TTL)


async def _link_states(call_id, user_ids) -> dict[str, str]:
    """Which participants are currently absent, for a resuming client's UI."""
    states = {str(uid): LINK_UP for uid in user_ids}
    with contextlib.suppress(Exception):
        redis = get_redis()
        pipe = redis.pipeline()
        for uid in user_ids:
            pipe.get(_link_key(call_id, uid))
        for uid, value in zip(user_ids, await pipe.execute(), strict=True):
            if value:
                states[str(uid)] = LINK_DOWN
    return states


# ---------------------------------------------------------------------------
# Resume: what a (re)connecting client needs to rebuild its call UI
# ---------------------------------------------------------------------------


async def active_call_state(db: AsyncSession, user_id: uuid.UUID) -> dict | None:
    """The live call this user belongs to, in a shape a client can resume from.

    Read from Postgres rather than from the Redis `call:user:*` marker, so it is
    still correct after a Redis flush or a failover — the call row is the only
    authority on whether a call is live.

    This is the piece that was missing entirely. Every `call:*` frame is a
    fire-and-forget publish to a channel; anything sent while a client's socket was
    reconnecting is gone for good. Without a way to ask "what am I in?", a callee
    whose socket blipped during the ring never saw the ringer even though the server
    rang for another 40 seconds, and a client that reconnected mid-call had no way
    to discover it should still be in a room.
    """
    stmt = (
        select(Call)
        .join(CallParticipant, CallParticipant.call_id == Call.id)
        .where(
            CallParticipant.user_id == user_id,
            Call.status.in_((CallStatus.ringing, CallStatus.connected)),
        )
        .order_by(Call.started_at.desc())
        .limit(1)
    )
    call = (await db.execute(stmt)).scalars().first()
    if call is None:
        return None

    rows = (
        await db.execute(
            select(CallParticipant, User)
            .join(User, User.id == CallParticipant.user_id)
            .where(CallParticipant.call_id == call.id)
        )
    ).all()

    participants = []
    joined: list[str] = []
    me_joined = False
    caller = None
    for cp, u in rows:
        participants.append(_brief(u))
        if cp.joined_at is not None and cp.left_at is None:
            joined.append(str(u.id))
            if u.id == user_id:
                me_joined = True
        if u.id == call.initiated_by:
            caller = _brief(u)

    conv_name = None
    if call.is_group and call.conversation_id:
        conv = await db.get(Conversation, call.conversation_id)
        conv_name = (conv.name if conv else None) or "Group call"

    ids = [cp.user_id for cp, _ in rows]
    return {
        "call_id": str(call.id),
        "status": call.status.value,
        "call_type": call.type.value,
        "is_group": call.is_group,
        "conversation_id": str(call.conversation_id) if call.conversation_id else None,
        "room": call.room_name,
        "initiated_by": str(call.initiated_by) if call.initiated_by else None,
        "is_initiator": call.initiated_by == user_id,
        "caller": caller,
        "group_name": conv_name,
        "participants": participants,
        "joined": joined,
        # Whether *this* client should already be in the SFU room. A resuming client
        # rejoins only when this is true, so a callee that never accepted is put back
        # on the ringer rather than dropped straight into a live room.
        "self_joined": me_joined,
        "started_at": iso_z(call.started_at),
        "answered_at": iso_z(call.answered_at),
        "ring_expires_in": (
            await call_deadlines.remaining(call_deadlines.RING, str(call.id))
            if call.status == CallStatus.ringing
            else None
        ),
        "peer_links": await _link_states(call.id, ids),
    }


async def resume_calls_for(user_id: uuid.UUID) -> None:
    """Everything a freshly-registered socket needs, in one pass.

    Called from the hub the moment a socket is added — before presence is published,
    so the frames cannot race the client's own resume fetch. Two jobs, sharing the
    single state read:

      * close any grace window a previous drop opened, so a reconnect inside the
        window resumes the call rather than being resolved out of it, and tell the
        peers to stop showing "Connecting…";
      * re-deliver the ring, or the full connected state, that was published to a
        channel with no subscriber while this client was away.

    The second is the one that used to be impossible. `call:*` frames are
    fire-and-forget publishes: a ring sent during a two-second Wi-Fi handover
    evaporated, and the phone showed nothing while the server rang on for another
    forty seconds.
    """
    async with SessionLocal() as db:
        state = await active_call_state(db, user_id)
    if state is None:
        return
    call_id = _parse_uuid(state["call_id"])
    if call_id is None:
        return
    # Only announce a recovery for someone actually engaged in the call.
    #
    # Every member of a group conversation has a CallParticipant row, so a 30-person
    # group call would otherwise have each member's reconnect publish a
    # `call:peer_state` to the other 29 — thirty times the traffic, for people who
    # were never in the call and whose link nobody was waiting on.
    engaged = state["self_joined"] or state["is_initiator"] or state["status"] == CallStatus.ringing.value
    if engaged:
        await _mark_link_up(call_id, user_id, participants=state["participants"])
    await _replay(user_id, state)


async def _replay(user_id: uuid.UUID, state: dict) -> None:
    if state["status"] == CallStatus.ringing.value and not state["is_initiator"]:
        # The ring itself, replayed in the same shape the original had, so the client
        # needs no separate code path for a resumed ring.
        await publish_to_users(
            [user_id],
            {
                "type": "call:incoming",
                "call_id": state["call_id"],
                "caller": state["caller"],
                "call_type": state["call_type"],
                "conversation_id": state["conversation_id"],
                "is_group": state["is_group"],
                "group_name": state["group_name"],
                "replayed": True,
                "ring_expires_in": state["ring_expires_in"],
            },
        )
        return
    # Connected (or our own outgoing ring): hand back the whole state and let the
    # client reconcile. `call:resume` is additive — a client already showing this
    # call ignores it.
    await publish_to_users([user_id], {"type": "call:resume", "call": state})


# ---------------------------------------------------------------------------
# Link down / link up
# ---------------------------------------------------------------------------


async def handle_user_link_down(user: User) -> None:
    """Every socket this user held has gone. Open a grace window, do not hang up.

    This function used to end the call outright, and that single decision accounted
    for most reports of a call "dropping for no reason":

      * the server closes every socket with 4001 the moment the 15-minute access
        cookie lapses, so **every call that outlived the remaining cookie lifetime
        died on the spot**, mid-sentence, with `reason="disconnected"`;
      * iOS tears its socket down on backgrounding, so glancing at another app
        ended the call;
      * a Wi-Fi/cellular handover, a laptop lid, or a sleeping tab did the same.

    None of those mean the person hung up. They mean the signalling link is absent
    for a moment while the media session on the SFU is very often still perfectly
    alive. So: mark the link down, tell the peers so both ends can show
    "Connecting…", and let `_grace_expired` decide — but only once the window has
    actually closed with the user still missing.
    """
    call_id = _parse_uuid(await current_call_of(user.id))
    if call_id is None:
        return
    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        if call is None or call.status not in (CallStatus.ringing, CallStatus.connected):
            return
        ids = await _call_participant_ids(db, call_id)

    await _mark_link(call_id, user.id, LINK_DOWN)
    await call_deadlines.schedule(
        call_deadlines.GRACE, _grace_member(call_id, user.id), RECONNECT_GRACE_SECONDS
    )
    logger.info("call.link_down call_id=%s user_id=%s grace=%ss", call_id, user.id, RECONNECT_GRACE_SECONDS)
    await publish_to_users(
        [u for u in ids if u != user.id],
        {
            "type": "call:peer_state",
            "call_id": str(call_id),
            "user_id": str(user.id),
            "state": "reconnecting",
            "grace_seconds": RECONNECT_GRACE_SECONDS,
        },
    )


async def handle_user_link_up(user_id: uuid.UUID) -> None:
    """A socket (or a fresh room token) for this user is live again.

    Closes any grace window a previous drop opened, so a client that reconnects
    inside the window resumes its call instead of being resolved out of it a moment
    later, and tells the peers to stop showing "Connecting…".
    """
    call_id = _parse_uuid(await current_call_of(user_id))
    if call_id is None:
        # Redis may have forgotten (flush, failover, TTL); the call row has not.
        async with SessionLocal() as db:
            state = await active_call_state(db, user_id)
        if state is None:
            return
        call_id = _parse_uuid(state["call_id"])
        if call_id is None:
            return
        await _set_in_call([user_id], call_id)
        await _mark_link_up(call_id, user_id, participants=state["participants"])
        return

    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        if call is None or call.status not in (CallStatus.ringing, CallStatus.connected):
            return
        ids = await _call_participant_ids(db, call_id)
    await _mark_link_up(call_id, user_id, peer_ids=[u for u in ids if u != user_id])


async def _mark_link_up(
    call_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    peer_ids: list | None = None,
    participants: list[dict] | None = None,
) -> None:
    """Cancel the grace window and announce the recovery. Idempotent."""
    await call_deadlines.cancel(call_deadlines.GRACE, _grace_member(call_id, user_id))
    await _mark_link(call_id, user_id, LINK_UP)
    if peer_ids is None:
        peer_ids = [p["id"] for p in (participants or []) if p["id"] != str(user_id)]
    if not peer_ids:
        return
    logger.info("call.link_up call_id=%s user_id=%s", call_id, user_id)
    await publish_to_users(
        peer_ids,
        {
            "type": "call:peer_state",
            "call_id": str(call_id),
            "user_id": str(user_id),
            "state": "connected",
        },
    )


async def _grace_expired(call_id: uuid.UUID, user_id: uuid.UUID) -> None:
    """The grace window closed. Resolve the call without this participant.

    Re-checks everything: the deadline is durable and may fire on a worker that
    knows nothing about the original disconnect, and the user may have reconnected
    in the meantime (in which case the link key is gone and this is a no-op).
    """
    if not await get_redis().exists(_link_key(call_id, user_id)):
        return  # came back
    async with SessionLocal() as db:
        call = await _load_call(db, call_id)
        if call is None or call.status not in (CallStatus.ringing, CallStatus.connected):
            return
        user = await db.get(User, user_id)
        if user is None:
            return
        is_initiator = call.initiated_by == user_id
        status = call.status
        is_group = call.is_group

    logger.info(
        "call.grace_expired call_id=%s user_id=%s status=%s initiator=%s",
        call_id,
        user_id,
        status.value,
        is_initiator,
    )

    if status == CallStatus.ringing:
        if is_initiator:
            await _cancel(user, call_id)
            return
        # The callee never came back. Leave the RING deadline to record the miss
        # rather than pre-empting it — the caller may still be listening to ringback,
        # and cutting it short here would report a miss the callee could still have
        # answered had they returned a second later.
        #
        # The link key is deliberately left DOWN: they really are still absent, and
        # `handle_user_link_up` clears it if they return. Nothing re-fires either way —
        # `claim_due` removed this deadline before calling us.
        return

    # Resolving them out of the call, so their link record goes with it. `_finalize`
    # clears every participant's key when the whole call ends; a group `_leave` leaves
    # the call running, so this is the only thing that clears the leaver's.
    await _mark_link(call_id, user_id, LINK_UP)
    if is_group:
        await _leave(user, call_id)
    else:
        await _end(user, call_id, reason="disconnected")


# Kept under its original name: `api/calls.py` and the tests both reference it, and
# a socket teardown is exactly a link going down.
handle_user_disconnect = handle_user_link_down


# ---------------------------------------------------------------------------
# Sweeper
# ---------------------------------------------------------------------------


async def sweep_due_deadlines() -> int:
    """Fire every ring/grace deadline that has come due. Safe to call from any
    worker, as often as you like. Returns how many fired, for the tests.

    A handler that raises gets its deadline put back rather than dropped.
    claim_due is destructive by design — it ZREMs in the same Lua script that
    reads, so exactly one worker owns each firing — which means an exception
    after the claim is a deadline that no longer exists anywhere. That is the
    third failure mode call_deadlines was written to eliminate, reached through
    the error path instead of a restart: nothing left to sweep, and a call stuck
    ringing until someone fixes the row by hand.

    Re-firing is safe by construction, which is what makes putting it back the
    right move rather than a risk: every handler re-reads its row and no-ops
    unless the state it acts on is still true, so a retry against a call that was
    answered in the meantime does nothing.

    Retries are not bounded, deliberately. There is no attempt counter to bound
    them with — the sorted set holds a score and nothing else — and the failures
    this recovers are transient by nature (a database blip, an exhausted pool,
    Redis refusing the link read in _grace_expired). A handler that keeps raising
    means the infrastructure under it is still down, which is exactly when the
    deadline should still be waiting.

    What this does NOT recover is a Redis outage: the deadline sets live in
    Redis, so schedule() degrades to a no-op in the same outage that made the
    handler fail, and the deadline is gone regardless.
    """
    fired = 0
    for member in await call_deadlines.claim_due(call_deadlines.RING):
        call_id = _parse_uuid(member)
        if call_id is None:
            # Unparseable members can never succeed; dropping is the fix, and
            # claim_due has already done it.
            continue
        try:
            await _ring_timeout(call_id)
        except Exception:
            logger.exception("ring timeout failed, retrying call_id=%s", call_id)
            await call_deadlines.schedule(call_deadlines.RING, member, DEADLINE_RETRY_SECONDS)
            continue
        fired += 1
    for member in await call_deadlines.claim_due(call_deadlines.GRACE):
        call_part, _, user_part = member.partition(":")
        call_id, user_id = _parse_uuid(call_part), _parse_uuid(user_part)
        if call_id is None or user_id is None:
            continue
        try:
            await _grace_expired(call_id, user_id)
        except Exception:
            logger.exception("grace expiry failed, retrying call_id=%s user_id=%s", call_id, user_id)
            await call_deadlines.schedule(call_deadlines.GRACE, member, DEADLINE_RETRY_SECONDS)
            continue
        fired += 1
    return fired


# ---------------------------------------------------------------------------
# Inbound frame dispatch
# ---------------------------------------------------------------------------


async def handle_call_ws_message(user: User, data: dict) -> None:
    msg_type = data.get("type")
    call_id = _parse_uuid(data.get("call_id"))

    if msg_type == "call:initiate":
        callee = _parse_uuid(data.get("callee_id"))
        if callee:
            await initiate_direct_call(
                user, callee, data.get("call_type", "voice"), data.get("conversation_id")
            )
    elif msg_type == "call:group_initiate":
        conv = _parse_uuid(data.get("conversation_id"))
        if conv:
            await initiate_group_call(user, conv, data.get("call_type", "video"))
    elif call_id is None:
        return
    elif msg_type == "call:accept":
        await _accept(user, call_id)
    elif msg_type == "call:decline":
        await _decline(user, call_id)
    elif msg_type == "call:cancel":
        await _cancel(user, call_id)
    elif msg_type == "call:end":
        await _end(user, call_id)
    elif msg_type == "call:join":
        await _join(user, call_id)
    elif msg_type == "call:leave":
        await _leave(user, call_id)
    elif msg_type == "call:toggle_media":
        await _toggle_media(user, call_id, data.get("media_type", "audio"), data.get("enabled", True))
    elif msg_type == "call:link_state":
        await _relay_peer_state(user, call_id, data.get("state"), data.get("quality"))


def mint_token(call: Call, user: User, device_id: str | None = None, *, identity: str | None = None) -> str:
    """A room-scoped join token.

    `device_id` makes the identity unique per client. Callers that omit both
    arguments get a random suffix, which is still strictly better than the bare
    user id this used to issue — see `identity_for` for why that mattered. Pass
    `identity` when the caller needs to know the exact value that went into the
    token (the token endpoint returns it to the client).
    """
    settings = get_settings()
    token = (
        lk_api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity or identity_for(user.id, device_id))
        .with_name(user.display_name)
        .with_grants(
            lk_api.VideoGrants(
                room_join=True,
                room=call.room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_ttl(dt.timedelta(hours=6))
    )
    return token.to_jwt()


def _parse_uuid(value) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


async def serialize_call(db: AsyncSession, call: Call, for_user: uuid.UUID) -> dict:
    parts = (
        await db.execute(
            select(CallParticipant, User)
            .join(User, User.id == CallParticipant.user_id)
            .where(CallParticipant.call_id == call.id)
        )
    ).all()
    participants = []
    initiator = None
    seen_by = []
    for cp, u in parts:
        entry = {"user_id": str(u.id), "display_name": u.display_name, "avatar_url": u.avatar_url}
        participants.append(entry)
        if u.id == call.initiated_by:
            initiator = entry
        if cp.seen_at is not None:
            seen_by.append(str(u.id))
    direction = "outgoing" if call.initiated_by == for_user else "incoming"
    other = next((p for p in participants if p["user_id"] != str(for_user)), None)
    return {
        "_id": str(call.id),
        "call_id": str(call.id),
        "call_type": call.type.value,
        "is_group": call.is_group,
        "status": call.status.value,
        "conversation_id": str(call.conversation_id) if call.conversation_id else None,
        "initiator": initiator,
        "participants": participants,
        "started_at": iso_z(call.started_at),
        "answered_at": iso_z(call.answered_at),
        "ended_at": iso_z(call.ended_at),
        "duration": _duration_of(call),
        "org_id": str(call.org_id) if call.org_id else None,
        "created_at": iso_z(call.started_at),
        "seen_by": seen_by,
        "direction": direction,
        "other_participant": other,
    }
