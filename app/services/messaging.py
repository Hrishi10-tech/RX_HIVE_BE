"""Unified messaging service: ONE path that persists and broadcasts.

Every message — REST-sent, WebSocket-sent, system — goes through send_message,
so persistence and realtime fan-out can never diverge (the defining defect of
the Mongo build: HTTP sends were stored but never broadcast).
"""

import datetime as dt
import re
import uuid

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import PasswordChangeRequiredError
from app.db.models import (
    Conversation,
    ConversationParticipant,
    Message,
    MessageAttachment,
    MessagePin,
    MessageReaction,
    MessageStar,
    MessageType,
    ParticipantRole,
    Upload,
    User,
)
from app.realtime.redis_bus import publish_to_users
from app.services import enrich, presence
from app.utils import now_utc, sanitize_text

_MEDIA_TYPES = {"image", "video", "audio", "file"}
_UPLOAD_URL_RE = re.compile(r"/api/media/up/([0-9a-f-]{36})")

# Longer than a uuid4, short enough that the unique index stays cheap. Clients
# send uuid4; anything longer is not a key we minted and is dropped rather than
# truncated, because a truncated key could collide with an unrelated send.
_MAX_CLIENT_MSG_ID = 64

# Pins are conversation-wide and the whole set is fetched on conversation open,
# so they must stay bounded — otherwise one member can pin every message and
# make the hot path unbounded for everyone. Keep in sync with the /pinned limit.
MAX_PINS_PER_CONVERSATION = 50


def _push_preview(doc: dict) -> str:
    t = doc.get("type")
    if t == "image":
        return "📷 Photo"
    if t == "video":
        return "🎬 Video"
    if t == "audio":
        return "🎤 Voice message"
    if t == "file":
        return f"📄 {doc.get('filename') or 'Document'}"
    body = doc.get("content") or ""
    return body[:120]


class SendError(HTTPException):
    pass


class AccountInactive(SendError):
    """The actor's own account is gone, as distinct from the conversation.

    Its own type so a caller can remap the conversation-scoped refusals to 404 —
    the mutation paths deliberately keep a foreign message indistinguishable from a
    nonexistent one — without restating the account condition to tell them apart.
    Restating it is how this batch's defects happened.
    """


class PasswordChangeRequired(AccountInactive, PasswordChangeRequiredError):
    """The actor's account was reset by an administrator and has not chosen a new password.

    An AccountInactive so `_require_message_access` passes it through as the account
    refusal it is, rather than collapsing it into a 404 "Message not found" the way
    it does every plain SendError (batch 73). Also the deps-layer
    PasswordChangeRequiredError, so it carries the same 403, detail and
    PASSWORD_CHANGE_REQUIRED code wherever it surfaces — over HTTP the coded
    handler serialises the code, which is what tells a client to show the
    change-password screen instead of a generic failure.
    """


def assert_conversation_access(conv: Conversation, user: User, *, is_member: bool) -> None:
    """May this user interact with this conversation AT ALL. Raises SendError.

    Public and shared on purpose. These three conditions were previously written
    out at each call site, and the copies drifted — which is the whole reason this
    exists. `_member_attachment` in api/media.py records the same lesson from the
    other direction: "media was missed when that gate was added", so the bytes kept
    serving out of a group whose is_active flag had been cleared while /messages
    404'd. Batch 53 found three more paths still on membership alone — typing
    indicators, presence fan-out and read receipts — so a tenant cut off from a
    cross-org group kept signalling into it. A fourth copy would have been a fourth
    thing to miss.

    `is_active = False` IS the revocation: archiving or deleting a group sets that
    flag and leaves every ConversationParticipant row in place (cross_org.py
    delete_group writes deleted_at and is_active and nothing else), so membership
    alone can never express whether access still stands.
    """
    # The user's own account, checked per call. HTTP refuses a deactivated user at
    # the auth dependency, but the websocket path arrives with a User row loaded
    # when the socket was opened — so without this a session that was live at
    # deactivation could keep acting until the socket's periodic revalidation
    # caught up. That revalidation is the backstop; this is the actual gate.
    if not user.is_active:
        raise AccountInactive(status_code=403, detail="Your account is no longer active")
    # The same reasoning for a password reset (batch 73): HTTP refuses the account at
    # the auth dependency, but a socket opened before the reset carries a User loaded
    # back then. The hub now re-checks the account before it dispatches any frame
    # that acts, and closes the socket rather than reaching this (batch 73 review),
    # so this is defence in depth on both transports — kept because it sits on the
    # code path itself, where a caller added later cannot forget it.
    if user.must_change_password:
        raise PasswordChangeRequired()
    if not is_member or not conv.is_active:
        raise SendError(status_code=404, detail="Conversation not found")
    # Org isolation: non-cross-org conversations must match the user's org.
    if conv.type.value != "cross_org" and conv.org_id != user.org_id:
        raise SendError(status_code=403, detail="Access denied")


async def conversation_recipients(
    db: AsyncSession, conversation_id: uuid.UUID, *, exclude: uuid.UUID | None = None
) -> list[uuid.UUID]:
    """Who may be TOLD anything about this conversation.

    The mirror of `assert_conversation_access`, which decides who may ACT. Both
    exist for the same reason: a ConversationParticipant row carries no org, so
    nothing in the schema stops one pointing at a conversation in another tenant,
    and the predicate that excludes such a row on the way IN has to exclude it on
    the way OUT too. Gating the actor and not the audience would have left the
    tenant boundary open in the direction that actually discloses data.

    There were eight copies of the unfiltered version of this query —
    `select(ConversationParticipant.user_id).where(conversation_id == ...)`, five of
    them byte-identical — feeding new_message, message_edited, reaction_update,
    message_pin_update, the system-message batch, messages_read and typing. A
    foreign-org row in an ordinary conversation received all of it, message bodies
    included. Adding the org predicate to eight call sites would have been eight
    more places for it to go stale, which is the defect this batch is about.
    """
    # A revoked conversation has no recipients. `is_active` belongs on THIS
    # question rather than on enrich.tenant_participants, which answers a different
    # one — tenant_participants supplies the rows serialize_message turns into
    # read_by/delivered_to, and history should still render its receipts.
    #
    # It matters because two callers reach here with no actor gate in front of
    # them: send_system_message and send_system_messages take a conversation id and
    # broadcast. Group events and call records go through them, so a call ending on
    # a group that was archived mid-call, or an admin action on an archived one,
    # pushed `new_message` to every former participant of a conversation they had
    # been cut off from. The row is still written — that is history, and no read
    # path will serve it — but nothing is delivered.
    conv = await db.get(Conversation, conversation_id)
    if conv is None or not conv.is_active:
        return []
    rows = await enrich.tenant_participants(db, conversation_id)
    return [p.user_id for p in rows if exclude is None or p.user_id != exclude]


async def _require_send_access(
    db: AsyncSession, conv: Conversation, sender: User
) -> list[ConversationParticipant]:
    participants = (
        (
            await db.execute(
                select(ConversationParticipant).where(ConversationParticipant.conversation_id == conv.id)
            )
        )
        .scalars()
        .all()
    )
    me = next((p for p in participants if p.user_id == sender.id), None)
    assert_conversation_access(conv, sender, is_member=me is not None)

    # No per-pair reachability check any more: membership of the conversation
    # IS the permission, and the org check above is what keeps tenants apart.
    if (
        conv.admin_only_messages
        and conv.type.value in ("group", "cross_org")
        and me.role not in (ParticipantRole.creator, ParticipantRole.admin)
    ):
        raise SendError(status_code=403, detail="Only admins can send messages")
    return participants


async def _claim_upload(db: AsyncSession, media_url: str, sender: User) -> Upload:
    m = _UPLOAD_URL_RE.search(media_url or "")
    if not m:
        raise SendError(status_code=400, detail="Invalid media_url")
    upload = await db.get(Upload, uuid.UUID(m.group(1)))
    if upload is None or upload.uploader_id != sender.id:
        raise SendError(status_code=400, detail="Invalid media_url")
    return upload


def _clean_client_msg_id(temp_id: str | None) -> str | None:
    """The sender's id for this message, or None if it cannot serve as a key."""
    if not temp_id:
        return None
    key = str(temp_id).strip()
    if not key or len(key) > _MAX_CLIENT_MSG_ID:
        return None
    return key


async def _find_by_client_msg_id(
    db: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    sender_id: uuid.UUID,
    client_msg_id: str,
) -> Message | None:
    return (
        (
            await db.execute(
                select(Message).where(
                    Message.conversation_id == conversation_id,
                    Message.sender_id == sender_id,
                    Message.client_msg_id == client_msg_id,
                )
            )
        )
        .scalars()
        .first()
    )


async def _replay_after_conflict(
    db: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    sender_id: uuid.UUID,
    client_msg_id: str | None,
    temp_id: str | None,
) -> dict | None:
    """Turn the loser of an idempotency race into the replay it should have had.

    Returns None when the conflict was not ours to swallow — a different
    constraint — so the caller re-raises.

    Both ids are taken as values, not as ORM instances, because the rollback below
    expires every object in the session and re-reading an attribute off an expired
    instance needs a lazy refresh async SQLAlchemy cannot perform.
    """
    await db.rollback()
    if client_msg_id is None:
        return None
    conv = await db.get(Conversation, conversation_id)
    sender = await db.get(User, sender_id)
    if conv is None or sender is None:
        return None
    winner = await _find_by_client_msg_id(
        db,
        conversation_id=conversation_id,
        sender_id=sender_id,
        client_msg_id=client_msg_id,
    )
    if winner is None:
        return None
    # Called for the refusal, not the return value: a replay must still be gated
    # the same way the original send was. Do not fold this away because nothing
    # reads the list any more.
    await _require_send_access(db, conv, sender)
    return await _replay_doc(db, winner, sender=sender, temp_id=temp_id)


async def _replay_doc(
    db: AsyncSession,
    msg: Message,
    *,
    sender: User,
    temp_id: str | None,
) -> dict:
    """Answer a repeat send with the message it already created.

    Deliberately silent: the recipients were fanned out to, and pushed to, on the
    first attempt. Only the sender — who never learned the outcome — is answered.

    Unlike the fresh path this cannot assume the message is unstarred and
    unpinned, because an arbitrary amount of time may have passed since it was
    stored, so it pays the two lookups instead of passing empty sets.

    Took a `participants` list until the serializer's participant set had to be
    tenant-filtered; it now reads the filtered rows itself, so passing the caller's
    unfiltered list in would have been a way to reintroduce the leak.
    """
    loaded = await enrich.load_message(db, msg.id)
    visible = await enrich.tenant_participants(db, msg.conversation_id)
    doc = await enrich.serialize_message(
        db,
        loaded,
        conv_participants=visible,
        sender=sender,
        include_reply=True,
        for_user=sender.id,
    )
    doc["temp_id"] = temp_id
    doc["status"] = "sent"
    return doc


async def send_message(
    db: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    sender: User,
    content: str = "",
    msg_type: str = "text",
    reply_to: str | None = None,
    temp_id: str | None = None,
    media_url: str | None = None,
    media_urls: list[str] | None = None,
    duration: float | None = None,
) -> dict:
    """Persist a message and fan it out. Returns the serialized message
    (with temp_id + status echoed for the sender's optimistic UI)."""
    conv = await db.get(Conversation, conversation_id)
    if conv is None:
        raise SendError(status_code=404, detail="Conversation not found")
    participants = await _require_send_access(db, conv, sender)

    # Idempotency, checked AFTER the access gate so a repeat send is never a way
    # to read a message back out of a conversation the caller has been removed
    # from. Scoped to this sender, so one account cannot probe or block another's.
    client_msg_id = _clean_client_msg_id(temp_id)
    if client_msg_id is not None:
        already = await _find_by_client_msg_id(
            db,
            conversation_id=conv.id,
            sender_id=sender.id,
            client_msg_id=client_msg_id,
        )
        if already is not None:
            return await _replay_doc(db, already, sender=sender, temp_id=temp_id)

    if msg_type not in _MEDIA_TYPES and msg_type != "text":
        msg_type = "text"  # clients may not mint system/other types
    content = sanitize_text(content).strip()
    if msg_type == "text" and not content:
        raise SendError(status_code=400, detail="Message content required")

    urls = [u for u in ([media_url] if media_url else []) + list(media_urls or []) if u]
    if msg_type in _MEDIA_TYPES and not urls:
        raise SendError(status_code=400, detail="media_url required for media messages")

    reply_to_id = None
    if reply_to:
        try:
            candidate = uuid.UUID(str(reply_to))
        except ValueError:
            candidate = None
        if candidate:
            replied = await db.get(Message, candidate)
            # Replies must reference a message in the same conversation.
            if replied and replied.conversation_id == conv.id:
                reply_to_id = replied.id

    now = now_utc()
    msg = Message(
        conversation_id=conv.id,
        sender_id=sender.id,
        type=MessageType(msg_type),
        content=content,
        client_msg_id=client_msg_id,
        reply_to_id=reply_to_id,
        created_at=now,
    )
    db.add(msg)
    # Captured before the flush: the recovery below rolls back, which expires every
    # instance in the session.
    conversation_pk = conv.id
    sender_pk = sender.id
    try:
        await db.flush()
    except IntegrityError:
        # THE point at which a duplicate is detected, and it was unguarded.
        #
        # uq_messages_client_msg_id is a partial unique INDEX, not a constraint, so
        # it cannot be deferred and PostgreSQL enforces it when the INSERT runs —
        # here, at the flush. The guard that exists for exactly this race was
        # attached to db.commit() instead, and Session.commit() flushes first, so
        # by the time it ran this flush had already emitted the INSERT and there
        # was nothing left for the commit to raise. The backstop was therefore
        # unreachable for the one constraint it names, and the loser of a genuine
        # race got an IntegrityError out of the endpoint — a 500 where the design
        # promises a replay.
        #
        # It only ever looked like it worked because the check-then-act lookup
        # above catches the common case: when the winner commits before the loser
        # reaches that lookup, the loser replays and never inserts. Two sends that
        # genuinely overlap skip it.
        replay = await _replay_after_conflict(
            db,
            conversation_id=conversation_pk,
            sender_id=sender_pk,
            client_msg_id=client_msg_id,
            temp_id=temp_id,
        )
        if replay is None:
            raise
        return replay

    seen_keys: set[str] = set()
    for url in urls:
        upload = await _claim_upload(db, url, sender)
        if upload.storage_key in seen_keys:
            continue
        seen_keys.add(upload.storage_key)
        upload.claimed = True
        db.add(
            MessageAttachment(
                message_id=msg.id,
                storage_key=upload.storage_key,
                thumbnail_key=upload.thumbnail_key,
                filename=upload.filename,
                mime_type=upload.mime_type,
                file_size=upload.file_size,
                # Only meaningful for audio/video, and only the first attachment
                # can carry it — a voice note is always a single file.
                duration_seconds=duration if not seen_keys or len(seen_keys) == 1 else None,
                # PDF page count, carried across the upload -> attachment hop.
                page_count=upload.page_count,
            )
        )

    conv.last_message_at = now
    # Captured before the commit: a rollback expires every instance in the
    # session, and re-reading `sender.id` off an expired object in async
    # SQLAlchemy would need a lazy refresh it cannot perform.
    sender_id = sender.id
    try:
        await db.commit()
    except IntegrityError:
        # The lookup above is check-then-act, so two genuinely concurrent copies
        # of one send — the socket and the HTTP fallback racing, or a retry that
        # overlaps the attempt it is retrying — can both reach here. The partial
        # unique index is what actually enforces the guarantee; this turns the
        # loser of that race into the same replay a sequential repeat would get.
        #
        # Kept, and now sharing the flush path's implementation. It is not the
        # branch that catches the client_msg_id race — the flush above is — but it
        # stays for anything that can still fail at commit, and so that removing
        # the explicit flush one day cannot silently take the recovery with it.
        replay = await _replay_after_conflict(
            db,
            conversation_id=conversation_id,
            sender_id=sender_id,
            client_msg_id=client_msg_id,
            temp_id=temp_id,
        )
        if replay is None:
            raise
        return replay

    loaded = await enrich.load_message(db, msg.id)
    # Filtered rows, not the full membership list: conv_participants becomes
    # read_by/delivered_to in the fanned-out document, so an unfiltered list would
    # disclose a foreign-org participant's user id to everyone legitimately here even
    # though nothing is delivered to them.
    visible = await enrich.tenant_participants(db, msg.conversation_id)
    doc = await enrich.serialize_message(
        db,
        loaded,
        conv_participants=visible,
        sender=sender,
        include_reply=True,
        # Brand-new message: it cannot be starred or pinned yet, so hand the
        # serializer empty sets instead of paying two lookups on the send path.
        starred_ids=set(),
        pinned_ids=set(),
    )

    # Not `[p.user_id for p in participants]`: that list is every membership row,
    # and a foreign-org one would have been handed the message body. `participants`
    # stays as loaded only for the mute set below, which subtracts from this list
    # and so cannot widen it. The serializer takes `visible` above — an earlier
    # version of this comment still credited `participants` for that, and describing
    # the behaviour a fix removed is the kind of stale comment this batch keeps
    # finding.
    others = await conversation_recipients(db, msg.conversation_id, exclude=sender.id)
    statuses = await presence.get_statuses(others)
    online = [uid for uid in others if statuses.get(str(uid)) == "online"]

    broadcast_doc = {**doc, "status": "delivered"}
    await publish_to_users(others, {"type": "new_message", "message": broadcast_doc})

    if online:
        await publish_to_users(
            [sender.id],
            {
                "type": "message_status",
                "message_id": str(msg.id),
                "status": "delivered",
                "delivered_to": [str(u) for u in online],
            },
        )

    # Web Push to recipients who are offline. Dispatched, never awaited: these
    # are remote HTTPS POSTs to third-party push services, and a slow or dead
    # endpoint must not delay — or fail — the sender's send.
    #
    # online_set is hoisted, and that is the whole point rather than a tidy-up.
    # `set(online)` written inside the comprehension's condition is rebuilt for
    # every element of `others` — CPython does not hoist a loop-invariant call out
    # of a comprehension — so the offline list cost n*m element insertions per
    # message sent. Measured with half the group online: 1.65ms at 256 recipients,
    # 114.50ms at 2,000, 703.88ms at 5,000, against 0.023 / 0.195 / 0.461ms hoisted.
    #
    # That is CPU on the event loop, so it is not the sender's latency alone — it is
    # the worker unable to serve anything else for the duration. Regular groups are
    # capped at MAX_GROUP_MEMBERS (256) so they stayed inside a couple of
    # milliseconds, but cross-org groups have no size cap at all (app/api/cross_org.py
    # bounds only the minimum), which is what made the tail reachable.
    online_set = set(online)
    offline = [uid for uid in others if uid not in online_set]
    # Respect each recipient's mute. is_muted has been on the participant row and
    # served on every conversation all along, and NOTHING read it — so "Mute
    # notifications" silenced neither the in-app path nor push. Filter here rather
    # than inside push_to_users so the mute is evaluated against this
    # conversation's participants, which we already hold.
    muted = {p.user_id for p in participants if p.is_muted}
    offline = [uid for uid in offline if uid not in muted]
    if offline:
        from app.services.push import dispatch_push_to_users

        preview = _push_preview(doc)
        dispatch_push_to_users(
            offline,
            {
                "title": sender.display_name,
                "body": preview,
                "tag": str(conv.id),
                # conversation_id lets sw.js tell a focused tab WHICH chat to open;
                # the url is the fallback for when it has to open a new window.
                "conversation_id": str(conv.id),
                "url": f"/chat?c={conv.id}",
            },
        )

    doc["temp_id"] = temp_id
    doc["status"] = "sent"
    return doc


async def send_system_message(
    db: AsyncSession, conversation_id: uuid.UUID, content: str, *, broadcast: bool = True
) -> dict:
    """System messages (group events, call records). Not user-attributable."""
    conv = await db.get(Conversation, conversation_id)
    now = now_utc()
    msg = Message(
        conversation_id=conversation_id,
        sender_id=None,
        type=MessageType.system,
        content=content,
        created_at=now,
    )
    db.add(msg)
    if conv is not None:
        conv.last_message_at = now
    await db.commit()
    loaded = await enrich.load_message(db, msg.id)
    doc = await enrich.serialize_message(db, loaded, include_reply=False, starred_ids=set(), pinned_ids=set())
    if broadcast and conv is not None:
        participants = await conversation_recipients(db, conversation_id)
        await publish_to_users(
            participants, {"type": "new_message", "message": {**doc, "status": "delivered"}}
        )
    return doc


async def send_system_messages(
    db: AsyncSession, conversation_id: uuid.UUID, contents: list[str], *, broadcast: bool = True
) -> None:
    """Several system messages, ONE commit. Returns nothing — see below.

    Four call sites emitted these one per member in a loop, and send_system_message
    commits, re-loads and serializes on every call. Measured against the real
    schema: 7 queries per message, so 256 members cost 1,792 queries and 573ms of
    request time — 0.6s of a superadmin's request spent entirely on rows nobody
    was waiting for. Until Batch 41 capped cross-org groups there was no ceiling
    on that at all.

    Every one of those loops discarded the returned document, so the load and the
    serialize were bought and thrown away. This returns None to make that
    explicit: a caller that wants the document should send one message and use
    send_system_message.

    Timestamps are separated by a microsecond each rather than sharing one `now`.
    History is ordered by created_at, and a batch that all landed on the same
    instant would be free to come back in any order — "Admin added Bob" before
    "Admin added Alice" — which is the kind of thing that only shows up in
    production.
    """
    if not contents:
        return
    conv = await db.get(Conversation, conversation_id)
    now = now_utc()
    msgs = [
        Message(
            conversation_id=conversation_id,
            sender_id=None,
            type=MessageType.system,
            content=content,
            created_at=now + dt.timedelta(microseconds=i),
        )
        for i, content in enumerate(contents)
    ]
    db.add_all(msgs)
    if conv is not None:
        conv.last_message_at = msgs[-1].created_at
    await db.commit()

    if not broadcast or conv is None:
        return
    # Read once, not once per message. The frames themselves are still sent
    # individually: collapsing them would change what a client receives, and this
    # is meant to cost less, not to do less.
    participants = await conversation_recipients(db, conversation_id)
    for msg in msgs:
        loaded = await enrich.load_message(db, msg.id)
        doc = await enrich.serialize_message(
            db, loaded, include_reply=False, starred_ids=set(), pinned_ids=set()
        )
        await publish_to_users(
            participants, {"type": "new_message", "message": {**doc, "status": "delivered"}}
        )


async def mark_read(
    db: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    reader: User,
    last_read_message_id: str | None = None,
) -> None:
    """Set last_read_at and broadcast messages_read to other participants."""
    me = await db.get(ConversationParticipant, (conversation_id, reader.id))
    # Membership alone was the whole gate here, so a reader cut off from a
    # conversation could still stamp last_read_at on it and fan `messages_read` out
    # to the members of a group they had been revoked from — while their own
    # message list 404'd. Same rule as sending, deliberately the same function.
    conv = await db.get(Conversation, conversation_id)
    if conv is None:
        raise SendError(status_code=404, detail="Conversation not found")
    assert_conversation_access(conv, reader, is_member=me is not None)

    read_time = now_utc()
    anchor_id = None
    if last_read_message_id:
        try:
            anchor = await db.get(Message, uuid.UUID(str(last_read_message_id)))
        except ValueError:
            anchor = None
        if anchor and anchor.conversation_id == conversation_id:
            anchor_id = str(anchor.id)
            read_time = max(
                read_time,
                anchor.created_at if anchor.created_at.tzinfo else anchor.created_at.replace(tzinfo=dt.UTC),
            )
    if me.last_read_at is None or read_time > me.last_read_at:
        me.last_read_at = read_time
    await db.commit()

    others = await conversation_recipients(db, conversation_id, exclude=reader.id)
    await publish_to_users(
        others,
        {
            "type": "messages_read",
            "conversation_id": str(conversation_id),
            "reader_id": str(reader.id),
            "last_read_message_id": anchor_id,
        },
    )


async def _require_message_access(db: AsyncSession, conversation_id: uuid.UUID, actor: User) -> Conversation:
    """The actor gate for mutating a message, as opposed to receiving one.

    Membership alone was the whole check on reactions, edits, stars, pins,
    deletions and forwards, so a participant row on a conversation that had been
    revoked — or one filed under another org, which the schema permits — could
    still MUTATE message state. Filtering the recipients of the resulting broadcast
    stops it disclosing anything; it does not stop the write.

    Refusals about the CONVERSATION collapse to 404 "Message not found", keeping the
    posture toggle_reaction and _member_message already documented: a non-member
    must not be able to tell a foreign message from a nonexistent one. A dead
    account is not information about the conversation, so AccountInactive passes
    through as the 403 the send path gives.
    """
    conv = await db.get(Conversation, conversation_id)
    me = await db.get(ConversationParticipant, (conversation_id, actor.id))
    if conv is None:
        raise SendError(status_code=404, detail="Message not found")
    try:
        assert_conversation_access(conv, actor, is_member=me is not None)
    except AccountInactive:
        raise
    except SendError as exc:
        raise SendError(status_code=404, detail="Message not found") from exc
    return conv


async def toggle_reaction(db: AsyncSession, *, message_id: uuid.UUID, actor: User, emoji: str) -> list[dict]:
    """Toggle a reaction. Requires conversation membership (closes the IDOR)."""
    msg = await enrich.load_message(db, message_id)
    if msg is None:
        raise SendError(status_code=404, detail="Message not found")
    conv = await _require_message_access(db, msg.conversation_id, actor)

    emoji = (emoji or "").strip()[:32]
    if not emoji:
        raise SendError(status_code=400, detail="Emoji required")

    existing = await db.get(MessageReaction, (message_id, actor.id, emoji))
    if existing:
        await db.delete(existing)
    else:
        db.add(MessageReaction(message_id=message_id, user_id=actor.id, emoji=emoji))
    await db.commit()

    msg = await enrich.load_message(db, message_id)
    user_ids = {r.user_id for r in msg.reactions}
    users = {}
    if user_ids:
        rows = (await db.execute(select(User).where(User.id.in_(user_ids)))).scalars().all()
        users = {u.id: u for u in rows}
    reactions = enrich.enriched_reactions(list(msg.reactions), users, conv)

    participants = await conversation_recipients(db, msg.conversation_id)
    await publish_to_users(
        participants,
        {
            "type": "reaction_update",
            "message_id": str(message_id),
            "conversation_id": str(msg.conversation_id),
            "reactions": reactions,
        },
    )
    return reactions


async def _member_message(db: AsyncSession, message_id: uuid.UUID, actor: User) -> Message:
    """Load a message iff the actor participates in its conversation.

    404 either way — a non-member must not be able to tell a foreign message
    from a nonexistent one (same posture as toggle_reaction).
    """
    msg = await db.get(Message, message_id)
    if msg is None:
        raise SendError(status_code=404, detail="Message not found")
    await _require_message_access(db, msg.conversation_id, actor)
    return msg


async def toggle_star(db: AsyncSession, *, message_id: uuid.UUID, actor: User) -> bool:
    """Star/unstar for the actor only. Private: never broadcast, never visible
    to other participants."""
    await _member_message(db, message_id, actor)
    existing = await db.get(MessageStar, (message_id, actor.id))
    if existing is not None:
        await db.delete(existing)
        starred = False
    else:
        db.add(MessageStar(message_id=message_id, user_id=actor.id))
        starred = True
    await db.commit()
    return starred


async def toggle_pin(db: AsyncSession, *, message_id: uuid.UUID, actor: User) -> bool:
    """Pin/unpin for the whole conversation, then tell every participant."""
    msg = await _member_message(db, message_id, actor)
    existing = await db.get(MessagePin, message_id)
    if existing is not None:
        await db.delete(existing)
        pinned = False
    else:
        pin_count = (
            await db.execute(
                select(func.count())
                .select_from(MessagePin)
                .where(MessagePin.conversation_id == msg.conversation_id)
            )
        ).scalar_one()
        if pin_count >= MAX_PINS_PER_CONVERSATION:
            raise SendError(
                status_code=400,
                detail=(
                    f"This conversation already has the maximum of {MAX_PINS_PER_CONVERSATION} "
                    "pinned messages. Unpin one first."
                ),
            )
        db.add(
            MessagePin(
                message_id=message_id,
                conversation_id=msg.conversation_id,
                pinned_by=actor.id,
            )
        )
        pinned = True
    await db.commit()

    participants = await conversation_recipients(db, msg.conversation_id)
    await publish_to_users(
        participants,
        {
            "type": "message_pin_update",
            "message_id": str(message_id),
            "conversation_id": str(msg.conversation_id),
            "is_pinned": pinned,
        },
    )
    return pinned


# Message deletion was REMOVED as a product feature. `delete_message` and the
# DELETE /messages/{id} route that called it are gone, so nothing can tombstone a
# message (Message.deleted_at) any more.
#
# NOTE: delete-for-me as a MECHANISM is still live and still reachable — deleting
# a conversation (DELETE /conversations/{id} -> _hide_all_messages_for_caller)
# writes MessageDeletion rows for the caller. That is a separate feature and was
# left alone; only the per-message Delete for me / Delete for everyone actions
# were removed. So the MessageDeletion read filters below are load-bearing, not
# legacy. ("Clear chat" used to be the other writer and was itself removed later;
# rows it created before then are still out there and must stay hidden.)
#
# The READ side is deliberately untouched, because rows already exist:
#   * Message.deleted_at stays on the model and in the schema. Dropping it would
#     need a destructive migration, and any message deleted before this change
#     would lose its tombstone and reappear with its original content — the one
#     outcome a user who deleted a message must never get.
#   * enrich.serialize_message keeps emitting is_deleted and blanking content, so
#     historical tombstones still render as "This message was deleted".
#   * MessageDeletion rows stay filtered out of every read path — required both
#     for history hidden before this change and for conversation deletion, which
#     still writes them.
#   * edit_message and forward_message keep refusing to operate on a tombstone.
# The result is that removing the feature changes what users can DO without
# changing what they SEE.


async def edit_message(db: AsyncSession, *, message_id: uuid.UUID, actor: User, content: str) -> dict:
    """Message editing — absent from the Mongo build, built new here."""
    msg = await enrich.load_message(db, message_id)
    if msg is None:
        raise SendError(status_code=404, detail="Message not found")
    await _require_message_access(db, msg.conversation_id, actor)
    if msg.sender_id != actor.id:
        raise SendError(status_code=403, detail="Can only edit your own messages")
    if msg.deleted_at is not None:
        raise SendError(status_code=400, detail="Cannot edit a deleted message")
    if msg.type.value not in ("text", "image", "video", "audio", "file"):
        raise SendError(status_code=400, detail="Cannot edit this message")

    content = sanitize_text(content).strip()
    if msg.type.value == "text" and not content:
        raise SendError(status_code=400, detail="Message content required")
    msg.content = content
    msg.edited_at = now_utc()
    await db.commit()

    msg = await enrich.load_message(db, message_id)
    doc = await enrich.serialize_message(db, msg, for_user=actor.id)
    participants = await conversation_recipients(db, msg.conversation_id)
    await publish_to_users(
        participants,
        {
            "type": "message_edited",
            "message_id": str(message_id),
            "conversation_id": str(msg.conversation_id),
            "content": doc["content"],
            "edited_at": doc["edited_at"],
        },
    )
    return doc


async def forward_message(
    db: AsyncSession,
    *,
    actor: User,
    message_id: uuid.UUID,
    conversation_ids: list[str],
    contact_ids: list[str],
) -> list[str]:
    """Forward a message. The actor must be a member of the SOURCE conversation
    (closes the exfiltration IDOR) and of every target conversation."""
    original = await enrich.load_message(db, message_id)
    if original is None:
        raise SendError(status_code=404, detail="Message not found")
    # The SOURCE conversation, on the same terms as any other message mutation. A
    # revoked source keeps its participant rows, so membership alone let a former
    # member copy a message out of a group they had been cut off from into a live
    # one — laundering the content past the revocation.
    await _require_message_access(db, original.conversation_id, actor)
    if original.deleted_at is not None:
        raise SendError(status_code=400, detail="Cannot forward a deleted message")

    forwarded_to: list[str] = []

    async def _copy_into(conv: Conversation, participants: list[ConversationParticipant]):
        now = now_utc()
        copy = Message(
            conversation_id=conv.id,
            sender_id=actor.id,
            type=original.type,
            content=original.content,
            is_forwarded=True,
            created_at=now,
        )
        db.add(copy)
        await db.flush()
        for a in original.attachments:
            db.add(
                MessageAttachment(
                    message_id=copy.id,
                    storage_key=a.storage_key,
                    thumbnail_key=a.thumbnail_key,
                    filename=a.filename,
                    mime_type=a.mime_type,
                    file_size=a.file_size,
                    # Carried, or a forwarded voice note loses its length and the
                    # bubble falls back to reading it out of the file.
                    duration_seconds=a.duration_seconds,
                    # Or a forwarded PDF loses its preview and page count, the
                    # same way a forwarded voice note would lose its duration.
                    page_count=a.page_count,
                )
            )
        conv.last_message_at = now
        await db.commit()
        loaded = await enrich.load_message(db, copy.id)
        visible = await enrich.tenant_participants(db, conv.id)
        doc = await enrich.serialize_message(
            db,
            loaded,
            conv_participants=visible,
            sender=actor,
            include_reply=False,
            starred_ids=set(),  # the forwarded copy is new — no stars, no pin
            pinned_ids=set(),
        )
        others = [p.user_id for p in visible if p.user_id != actor.id]
        await publish_to_users(others, {"type": "new_message", "message": {**doc, "status": "delivered"}})
        forwarded_to.append(str(conv.id))

    for cid in conversation_ids or []:
        try:
            conv_uuid = uuid.UUID(str(cid))
        except ValueError:
            continue
        conv = await db.get(Conversation, conv_uuid)
        if conv is None or not conv.is_active:
            continue
        # Was: hand-rolled membership + org checks, and nothing else. That made
        # forwarding a way around admin_only_messages — a plain member of an
        # admin-only group could post into it by forwarding, because only
        # send_message consulted that flag. Routing through the shared gate
        # closes it and means every future send rule applies here automatically
        # instead of having to be remembered twice.
        #
        # Skip rather than raise, preserving forward's best-effort semantics:
        # one unreachable target must not fail the whole fan-out.
        try:
            participants = await _require_send_access(db, conv, actor)
        except SendError:
            continue
        await _copy_into(conv, participants)

    from app.services.conversations import get_or_create_direct  # local import: avoids cycle

    for uid in contact_ids or []:
        try:
            contact_uuid = uuid.UUID(str(uid))
        except ValueError:
            continue
        contact = await db.get(User, contact_uuid)
        if contact is None or contact.org_id != actor.org_id or not contact.is_active:
            continue
        try:
            # get_or_create_direct now enforces reachability and raises 403 for a
            # pair that may not converse. Caught so an unreachable contact is
            # skipped like any other bad target, rather than failing the fan-out
            # to everyone else.
            conv = await get_or_create_direct(db, actor, contact, notify=True)
        except HTTPException:
            continue
        participants = (
            (
                await db.execute(
                    select(ConversationParticipant).where(ConversationParticipant.conversation_id == conv.id)
                )
            )
            .scalars()
            .all()
        )
        await _copy_into(conv, participants)

    return forwarded_to
