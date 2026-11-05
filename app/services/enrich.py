"""Wire-format serialization of conversations and messages.

Shapes mirror the RxHivexx contract the salvaged frontend expects
(docs/reference/api-conversations-messages.md): serialized docs use `_id`
string keys, ISO-Z datetimes, participants carry display data + per-user
unread counts, messages carry sender enrichment + reply preview.

Unread counts and read receipts are DERIVED from conversation_participants.
last_read_at instead of stored counters — one source of truth, no drift.
"""

import datetime as dt
import uuid

from sqlalchemy import func, literal_column, or_, select, true
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import (
    Conversation,
    ConversationParticipant,
    ConversationType,
    Message,
    MessageAttachment,
    MessageDeletion,
    MessagePin,
    MessageReaction,
    MessageStar,
    MessageType,
    User,
)
from app.services import presence
from app.utils import iso_z


def media_url_for(attachment_id) -> str:
    return f"/api/media/{attachment_id}"


def thumb_url_for(attachment_id) -> str:
    return f"/api/media/{attachment_id}/thumb"


def serialize_user_brief(user: User, status: str = "offline") -> dict:
    return {
        "user_id": str(user.id),
        "display_name": user.display_name,
        "avatar_url": user.avatar_url,
        "status": status,
        "last_seen": iso_z(user.last_seen_at),
    }


async def unread_counts(
    db: AsyncSession, conversation_ids: list[uuid.UUID], user_id: uuid.UUID
) -> dict[uuid.UUID, int]:
    """Unread messages per conversation for one user: messages newer than the
    user's last_read_at, not sent by them, not deleted-for-them, not system.

    Counted per conversation through a LATERAL, and the cursor is a COALESCE
    rather than an OR. BOTH halves are load-bearing and neither works alone —
    measured on a seeded 2.1M-message tenant, one sidebar page of 30 conversations:

        hash join + OR (as it was)   96-102ms   58,059 buffers
        LATERAL, OR kept                199ms   13,744   <- worse
        hash join + COALESCE          no change  33,746   <- still a Join Filter
        LATERAL + COALESCE (this)       1.1ms      512

    last_read_at arrives from the joined participant row, so with a hash join the
    cursor can only be applied AFTER the join, as a Join Filter — which is why this
    read the whole messages table and threw away 99.8% of it to count 1,747 rows.
    The LATERAL makes last_read_at a per-conversation constant. Separately,
    (last_read_at IS NULL OR created_at > last_read_at) is not indexable at all:
    the IS NULL disjunct means created_at cannot bound the scan, so even inside the
    LATERAL Postgres still read every message in the conversation. COALESCE to
    -infinity is the same predicate with one branch, so it becomes an index
    condition on ix_messages_conversation_created and the scan starts at the cursor.

    The old cost tracked the tenant's TOTAL history: adding a million messages to
    conversations this user is not a member of took it from 33,746 buffers to
    58,059, while this version went from 510 to 512.

    sender_id != user_id is kept EXACTLY as it was, NOT made NULL-safe. System
    messages have a NULL sender, so `NULL != user_id` is NULL rather than true and
    they are excluded — which is what "not system" above means. IS DISTINCT FROM
    would silently start counting every system message as unread.
    """
    if not conversation_ids:
        return {}
    cp = ConversationParticipant
    hidden = (
        select(MessageDeletion.message_id)
        .where(
            MessageDeletion.message_id == Message.id,
            MessageDeletion.user_id == user_id,
        )
        .exists()
    )
    unread = (
        select(func.count(Message.id).label("n"))
        .where(
            Message.conversation_id == cp.conversation_id,
            Message.sender_id != user_id,
            Message.deleted_at.is_(None),
            Message.created_at > func.coalesce(cp.last_read_at, literal_column("'-infinity'::timestamptz")),
            ~hidden,
        )
        .lateral("unread")
    )
    # n > 0 keeps the returned dict IDENTICAL to the old GROUP BY, which only
    # emitted conversations having at least one unread row. Both call sites read it
    # with .get(conv_id, 0), so explicit zeros would be harmless — but identical is
    # cheaper to review than equivalent.
    stmt = (
        select(cp.conversation_id, unread.c.n)
        .select_from(cp)
        .join(unread, true())
        .where(
            cp.user_id == user_id,
            cp.conversation_id.in_(conversation_ids),
            unread.c.n > 0,
        )
    )
    rows = (await db.execute(stmt)).all()
    return {conv_id: count for conv_id, count in rows}


async def last_messages(
    db: AsyncSession, conversation_ids: list[uuid.UUID], for_user: uuid.UUID
) -> dict[uuid.UUID, dict]:
    """Latest visible message per conversation (excludes delete-for-me).

    One index descent per conversation, via LATERAL — deliberately NOT
    row_number() OVER (PARTITION BY conversation_id), and not DISTINCT ON. Those
    are the obvious spellings of "newest per group" and both make Postgres read
    and sort EVERY message in these conversations to hand back one row each.
    Measured on a seeded 1.08M-message tenant, for one sidebar page of 30
    conversations holding 340k messages between them: 252ms, 33,896 buffers
    (~265MB) and a 75MB external merge sort spilled to disk, to return 30 rows.
    The same page through this LATERAL is 0.6ms and 343 buffers, because
    ORDER BY created_at DESC LIMIT 1 inside a per-conversation scope is a backward
    walk of ix_messages_conversation_created that stops at the first row it can
    return. DISTINCT ON is not the fix either: ORDER BY conversation_id ASC,
    created_at DESC cannot be served by a single-direction btree, so it sorts too.

    The window version's cost scaled with the tenant's total history rather than
    with the page, so it got worse every day the product was used.

    NOT EXISTS rather than a LEFT JOIN ... IS NULL, so the hidden-row test stays
    inside the per-conversation scan: when the newest messages are deleted-for-me
    the walk simply continues backwards to the newest one that is not.
    """
    if not conversation_ids:
        return {}
    hidden = (
        select(MessageDeletion.message_id)
        .where(
            MessageDeletion.message_id == Message.id,
            MessageDeletion.user_id == for_user,
        )
        .exists()
    )
    convs = (
        select(Conversation.id.label("conversation_id"))
        .where(Conversation.id.in_(conversation_ids))
        .subquery()
    )
    newest = (
        select(Message)
        .where(Message.conversation_id == convs.c.conversation_id, ~hidden)
        .order_by(Message.created_at.desc())
        .limit(1)
        .lateral("newest")
    )
    # An INNER lateral join: a conversation whose every message is hidden for this
    # user yields no row and is therefore absent from the dict, which is exactly
    # what the window version did — serialize_conversation renders that as
    # last_message: null.
    stmt = (
        select(newest, User.display_name)
        .select_from(convs)
        .join(newest, true())
        .outerjoin(User, User.id == newest.c.sender_id)
    )
    rows = (await db.execute(stmt)).all()
    out: dict[uuid.UUID, dict] = {}
    for row in rows:
        m = row._mapping
        out[m["conversation_id"]] = {
            "content": "" if m["deleted_at"] else (m["content"] or ""),
            "sender_id": str(m["sender_id"]) if m["sender_id"] else None,
            "sender_name": row.display_name or "System",
            "created_at": iso_z(m["created_at"]),
            "type": m["type"].value if hasattr(m["type"], "value") else (m["type"] or "text"),
        }
    return out


async def load_conversation_with_participants(
    db: AsyncSession, conversation_id: uuid.UUID
) -> Conversation | None:
    stmt = (
        select(Conversation)
        .options(selectinload(Conversation.participants).selectinload(ConversationParticipant.user))
        .where(Conversation.id == conversation_id)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def serialize_conversation(
    db: AsyncSession,
    conv: Conversation,
    for_user: uuid.UUID,
    *,
    statuses: dict[str, str] | None = None,
    unread: dict[uuid.UUID, int] | None = None,
    last_msgs: dict[uuid.UUID, dict] | None = None,
) -> dict:
    """Full enriched conversation object. Batch inputs (statuses/unread/last
    messages) may be supplied by list endpoints; otherwise fetched here."""
    participants = conv.participants
    if statuses is None:
        statuses = await presence.get_statuses([p.user_id for p in participants])
    if unread is None:
        unread = await unread_counts(db, [conv.id], for_user)
    if last_msgs is None:
        last_msgs = await last_messages(db, [conv.id], for_user)

    me = next((p for p in participants if p.user_id == for_user), None)
    is_group = conv.type.value in ("group", "cross_org")

    part_dicts = []
    for p in participants:
        if p.user is None:
            continue
        d = serialize_user_brief(p.user, statuses.get(str(p.user_id), "offline"))
        d["unread_count"] = unread.get(conv.id, 0) if p.user_id == for_user else 0
        d["role"] = p.role.value
        if is_group:
            d["joined_at"] = iso_z(p.joined_at)
        part_dicts.append(d)

    doc = {
        "_id": str(conv.id),
        "type": conv.type.value,
        "org_id": str(conv.org_id) if conv.org_id else None,
        "name": conv.name,
        "avatar_url": conv.avatar_url,
        "cross_org": conv.type.value == "cross_org",
        "allowed_org_ids": list(conv.allowed_org_ids or []),
        "created_by": str(conv.created_by) if conv.created_by else None,
        "created_at": iso_z(conv.created_at),
        "last_message_at": iso_z(conv.last_message_at),
        "pinned_by": [str(p.user_id) for p in participants if p.is_pinned],
        "is_active": conv.is_active,
        "participants": part_dicts,
        "last_message": last_msgs.get(conv.id),
        "unread_count": unread.get(conv.id, 0),
        "is_pinned": bool(me and me.is_pinned),
        # My explicit position within the pinned block, or null when unordered.
        # Sent so the client's own re-sort in bumpConversation can reproduce the
        # server's ORDER BY instead of keeping pins in arrival order.
        "pin_order": me.pin_order if me else None,
        # Mute was stored and toggleable but never sent back, so every client read
        # it as False after a reload — and since PUT /mute is a blind flip, the
        # menu offered "Mute" on an already-muted chat and unmuted it instead.
        "is_muted": bool(me and me.is_muted),
    }
    if is_group:
        doc["description"] = conv.description
        doc["admin_only_messages"] = conv.admin_only_messages
    if conv.type.value == "cross_org":
        doc["purpose_tag"] = conv.purpose_tag
    return doc


def serialize_permissions(conv: Conversation) -> dict:
    """The group policy the settings UI reads/writes: one boolean, because one is
    all the server enforces.

    `send_messages` is `admin_only_messages` INVERTED — one stored source of truth,
    and the message routes honour it.

    `edit_info` and `add_members` used to be here too, described as the enforced
    ones. They were not enforced by anything: `perm_edit_info` and
    `perm_add_members` appeared nowhere outside the model and this function, while
    PUT /{id}/group and POST /{id}/members both gate on _require_group_admin alone.
    Since both columns default to TRUE, the permissive setting was the one that did
    nothing — a member shown "Edit group settings: enabled" got a 403.

    Withdrawn rather than implemented, which is the rule this module and
    groups.PermissionsRequest already apply to perm_send_history,
    perm_invite_via_link and perm_approve_new_members: nothing enforces them, so
    serving them advertises a policy the server does not apply. Implementing them
    instead would have loosened access on every existing group, because that TRUE
    default was a migration backfill rather than anyone's choice.

    The columns stay in the schema, inert, exactly as the other three do. Dropping
    them would be a destructive migration for no gain.
    """
    return {"send_messages": not conv.admin_only_messages}


def attachment_thumb_url(a: MessageAttachment) -> str | None:
    # Public, and called from api/media.py's gallery route as well as from here.
    # It was private, and conversation_media hand-rolled a stricter rule that
    # gated on thumbnail_key alone — so a PDF awaiting its lazy first render got
    # a preview everywhere except the Media/Links/Docs drawer, which also never
    # triggered the render that would have healed it. One function, one rule.
    """Thumbnail URL, or None when there will never be one.

    Emitted for a PDF that has no thumbnail_key YET but has never been processed
    (page_count IS NULL): /thumb renders page 1 lazily on first view, so offering
    the URL is what triggers the backfill for anything the batch job missed.

    Not emitted once page_count is 0, which is the "tried and could not render"
    marker — otherwise every view would re-parse a file already known to be
    un-renderable. Non-PDF documents never have a thumbnail at all.
    """
    if a.thumbnail_key:
        return thumb_url_for(a.id)
    if a.mime_type == "application/pdf" and a.page_count is None:
        return thumb_url_for(a.id)
    return None


def _attachment_fields(attachments: list[MessageAttachment]) -> dict:
    """Legacy top-level media fields from the first attachment + full list."""
    out: dict = {
        "media_url": None,
        "thumbnail_url": None,
        "file_size": None,
        "filename": None,
        "duration": None,
        "page_count": None,
        "attachments": [],
    }
    for a in attachments:
        item = {
            "id": str(a.id),
            "media_url": media_url_for(a.id),
            "thumbnail_url": attachment_thumb_url(a),
            "filename": a.filename,
            "mime_type": a.mime_type,
            "file_size": a.file_size,
            # Recorded length in seconds for audio/video. Sent so a voice-note
            # bubble can render "0:34" without an authenticated /api/media
            # redirect plus a presigned S3 GET just to read the container.
            "duration": a.duration_seconds,
            # PDF page count. NULL/absent means "not a PDF, or predates
            # previews" — the bubble falls back to the plain icon.
            "page_count": a.page_count,
        }
        out["attachments"].append(item)
    if attachments:
        first = attachments[0]
        out["media_url"] = media_url_for(first.id)
        out["thumbnail_url"] = attachment_thumb_url(first)
        out["file_size"] = first.file_size
        out["filename"] = first.filename
        out["duration"] = first.duration_seconds
        out["page_count"] = first.page_count
    return out


def receipts_for_message(
    msg_created_at: dt.datetime,
    sender_id,
    participants: list[ConversationParticipant],
) -> tuple[list[dict], list[dict]]:
    """Derive read_by/delivered_to arrays from participants' last_read_at."""
    read_by, delivered_to = [], []
    for p in participants:
        if p.user_id == sender_id:
            continue
        if p.last_read_at and msg_created_at and p.last_read_at >= msg_created_at:
            ts = iso_z(p.last_read_at)
            read_by.append({"user_id": str(p.user_id), "read_at": ts})
            delivered_to.append({"user_id": str(p.user_id), "delivered_at": ts})
    return read_by, delivered_to


async def starred_message_ids(
    db: AsyncSession, message_ids: list[uuid.UUID], user_id: uuid.UUID
) -> set[uuid.UUID]:
    """Which of these messages the given user has starred (one batched query)."""
    if not message_ids:
        return set()
    rows = await db.execute(
        select(MessageStar.message_id).where(
            MessageStar.message_id.in_(message_ids), MessageStar.user_id == user_id
        )
    )
    return set(rows.scalars().all())


async def pinned_message_ids(db: AsyncSession, message_ids: list[uuid.UUID]) -> set[uuid.UUID]:
    """Which of these messages are pinned (conversation-wide, one batched query)."""
    if not message_ids:
        return set()
    rows = await db.execute(select(MessagePin.message_id).where(MessagePin.message_id.in_(message_ids)))
    return set(rows.scalars().all())


async def reply_targets(db: AsyncSession, messages: list[Message]) -> dict[uuid.UUID, Message]:
    """Reply previews for a whole page, in at most one query.

    Two problems, one line. Serializing a page used to re-read each row's reply
    target individually, which cost three queries per replying message once
    selectinload followed up for its sender and its attachments: a measured 152
    queries for a 50-row page of replies, and roughly 600 for a 200-row /starred.

    The second problem is worse than the first. That per-row read passed
    populate_existing=True with only sender/attachments in its options, so whenever
    the target was ALSO a member of the page it refreshed that member and EXPIRED
    the reactions collection MESSAGE_LOAD_OPTIONS had eagerly loaded for it. Then
    the loop reached that row, touched msg.reactions, and lazy-loaded — which under
    asyncio is MissingGreenlet, a 500. /messages escaped it because a reply target
    is always older and that endpoint reverses to oldest-first, so the clobbered row
    had already been serialized. /starred and /pinned order newest-first and do not
    reverse, so they reached it afterwards and failed. Starring or pinning a message
    together with a reply to it was the entire reproduction, and /pinned is fetched
    on every conversation open — where the client's catch turns the 500 into a
    permanently blank pinned banner rather than an error anyone would see.

    So a target already in the page is taken FROM the page and never re-read: it
    arrived with the full load options, and refreshing it IS the bug. Only genuinely
    absent targets are queried, and those keep populate_existing, because
    list_messages resolves `before`/`around` anchors with db.get(), which leaves a
    bare Message in the identity map that would otherwise satisfy this query with no
    sender and no attachments loaded and lazy-load later. Those rows are not page
    members, so refreshing them expires nothing the loop will touch.
    """
    in_page = {m.id: m for m in messages}
    wanted = {m.reply_to_id for m in messages if m.reply_to_id}
    found = {mid: in_page[mid] for mid in wanted if mid in in_page}
    missing = wanted - found.keys()
    if missing:
        rows = (
            (
                await db.execute(
                    select(Message)
                    .options(selectinload(Message.sender), selectinload(Message.attachments))
                    .where(Message.id.in_(missing))
                    .execution_options(populate_existing=True)
                )
            )
            .scalars()
            .all()
        )
        found.update({m.id: m for m in rows})
    return found


async def tenant_participants(db: AsyncSession, conversation_id: uuid.UUID) -> list[ConversationParticipant]:
    """Participant rows that pass the conversation's tenant rule.

    Lives here rather than in messaging because this is the lower layer and it
    already owned the unfiltered version of this query — so the serializer's own
    fallback is correct by construction instead of being a second copy.

    The rows matter as well as the ids: `receipts_for_message` turns them into
    `read_by` and `delivered_to`, which go into the document fanned out to every
    recipient. So an unfiltered list does not merely widen delivery, it discloses a
    foreign-org participant's user id to everyone legitimately in the conversation,
    even once delivery itself is filtered.

    A participant row carries no org and nothing in the schema ties it to the
    conversation's, so this predicate is the only thing that excludes such a row. A
    cross_org conversation spans tenants by design and is the explicit exception.
    """
    return list(
        (
            await db.execute(
                select(ConversationParticipant)
                .join(Conversation, Conversation.id == ConversationParticipant.conversation_id)
                .join(User, User.id == ConversationParticipant.user_id)
                .where(
                    ConversationParticipant.conversation_id == conversation_id,
                    or_(
                        Conversation.type == ConversationType.cross_org,
                        Conversation.org_id == User.org_id,
                    ),
                )
            )
        )
        .scalars()
        .all()
    )


def visible_reactions(
    reactions: list[MessageReaction],
    conv: Conversation | None,
    users: dict[uuid.UUID, User] | None = None,
) -> list[MessageReaction]:
    """Reaction rows whose author belongs in this conversation's tenant.

    `conv_participants` stopped `read_by` and `delivered_to` disclosing a
    foreign-org row, but reactions are stored rows rather than derived ones: a
    reaction created before the mutation gate landed outlives it, and serializing it
    hands every legitimate member that user's id and display name. Gating new
    reactions does not remove the old ones.

    Filters on the AUTHOR's org against the conversation's, not on current
    membership — a member who has since left legitimately reacted while they were
    there, and dropping their reaction would rewrite history to fix a leak. A
    cross_org conversation spans tenants by design and keeps every author.

    An author that cannot be resolved is dropped for an ordinary conversation.
    message_reactions.user_id is ON DELETE CASCADE, so a row cannot outlive its
    user; an unresolvable author means the relationship was not loaded, and
    disclosing an id we cannot place is the thing being prevented.
    """
    if conv is None or conv.type == ConversationType.cross_org:
        return list(reactions)
    kept = []
    for r in reactions:
        author = users.get(r.user_id) if users is not None else r.user
        if author is not None and author.org_id == conv.org_id:
            kept.append(r)
    return kept


async def serialize_message(
    db: AsyncSession,
    msg: Message,
    *,
    conv_participants: list[ConversationParticipant] | None = None,
    sender: User | None = None,
    include_reply: bool = True,
    for_user: uuid.UUID | None = None,
    starred_ids: set[uuid.UUID] | None = None,
    pinned_ids: set[uuid.UUID] | None = None,
    reply_targets: dict[uuid.UUID, Message] | None = None,
) -> dict:
    """Serialize one message.

    `is_starred` is the REQUESTING user's star, so it needs `for_user`; with no
    user context it is False. `client_msg_id` is likewise scoped to `for_user` —
    a caller that omits it gets a document safe to fan out to anyone, which is
    what the send path wants. List endpoints pass pre-computed `starred_ids` /
    `pinned_ids` sets so a page costs two queries instead of two per message, and
    `reply_targets` for the same reason — see that function for why passing it also
    keeps this off a 500.
    """
    if sender is None and msg.sender_id:
        sender = await db.get(User, msg.sender_id)
    conv = await db.get(Conversation, msg.conversation_id)
    if conv_participants is None:
        conv_participants = await tenant_participants(db, msg.conversation_id)

    read_by, delivered_to = receipts_for_message(msg.created_at, msg.sender_id, conv_participants)

    if starred_ids is not None:
        is_starred = msg.id in starred_ids
    elif for_user is not None:
        is_starred = await db.get(MessageStar, (msg.id, for_user)) is not None
    else:
        is_starred = False
    if pinned_ids is not None:
        is_pinned = msg.id in pinned_ids
    else:
        is_pinned = await db.get(MessagePin, msg.id) is not None

    reactions = []
    for r in visible_reactions(list(msg.reactions), conv):
        reactions.append(
            {
                "user_id": str(r.user_id),
                # user_name is what the client's reaction tooltip reads. Only the
                # react endpoint used to send it, so tooltips were blank on load and
                # filled in only for whoever had just reacted. The reactor is
                # batch-loaded with the message (MESSAGE_LOAD_OPTIONS), not per row.
                "user_name": r.user.display_name if r.user else "Unknown",
                "emoji": r.emoji,
                "created_at": iso_z(r.created_at),
            }
        )

    doc = {
        "_id": str(msg.id),
        "conversation_id": str(msg.conversation_id),
        "sender_id": str(msg.sender_id) if msg.sender_id else None,
        "type": msg.type.value,
        "content": "" if msg.deleted_at else (msg.content or ""),
        # The sender's own id for this message, on every serialization of it TO
        # THAT SENDER — history included. Without it a client refetching a
        # conversation cannot tell which fetched rows its own unresolved bubbles
        # became, so a send whose outcome was never learned showed twice: once as
        # the stored message and once as the local bubble still waiting on it.
        #
        # Withheld from everyone else, who have nothing to reconcile it against:
        # it describes one device's local state, and it is the only field here
        # that is client-supplied text never passed through sanitize_text
        # (_clean_client_msg_id trims and caps the length, nothing more).
        # for_user is None on the send path, so a recipient's fanned-out copy is
        # covered too; the sender's own response reconciles on temp_id instead.
        "client_msg_id": msg.client_msg_id if for_user is not None and msg.sender_id == for_user else None,
        "reply_to": str(msg.reply_to_id) if msg.reply_to_id else None,
        "reactions": reactions,
        "read_by": read_by,
        "delivered_to": delivered_to,
        "is_deleted": msg.deleted_at is not None,
        "is_forwarded": msg.is_forwarded,
        "is_starred": is_starred,
        "is_pinned": is_pinned,
        "created_at": iso_z(msg.created_at),
        "edited_at": iso_z(msg.edited_at),
        "sender_name": sender.display_name if sender else "System",
        "sender_avatar": sender.avatar_url if sender else None,
    }
    doc.update(_attachment_fields([] if msg.deleted_at else list(msg.attachments)))

    if include_reply and msg.reply_to_id:
        if reply_targets is not None:
            reply = reply_targets.get(msg.reply_to_id)
        else:
            # The single-message callers — send, edit, react — have no page to batch
            # across. Explicit select (not db.get): an identity-map hit would skip
            # the eager-load options and later lazy-loads would blow up in async.
            # populate_existing is safe on this path for the reason it was NOT safe
            # on the page path: there is no sibling row here whose own eager loads
            # it could expire.
            reply = (
                await db.execute(
                    select(Message)
                    .options(selectinload(Message.sender), selectinload(Message.attachments))
                    .where(Message.id == msg.reply_to_id)
                    .execution_options(populate_existing=True)
                )
            ).scalar_one_or_none()
        if reply:
            reply_media = _attachment_fields([] if reply.deleted_at else list(reply.attachments))
            doc["reply_to_message"] = {
                "_id": str(reply.id),
                "sender_id": str(reply.sender_id) if reply.sender_id else None,
                "sender_name": reply.sender.display_name if reply.sender else "System",
                "content": "" if reply.deleted_at else (reply.content or ""),
                "type": reply.type.value,
                "media_url": reply_media["media_url"],
                "is_deleted": reply.deleted_at is not None,
            }
        else:
            doc["reply_to_message"] = None
    return doc


MESSAGE_LOAD_OPTIONS = [
    # The nested load is what keeps serialize_message's reactor names off an N+1:
    # one IN query for the reactions of the page, one more for their users.
    selectinload(Message.reactions).selectinload(MessageReaction.user),
    selectinload(Message.attachments),
    selectinload(Message.sender),
]


async def load_message(db: AsyncSession, message_id: uuid.UUID) -> Message | None:
    # populate_existing refreshes an identity-map hit so freshly committed
    # reactions/attachments are re-read instead of served from a stale collection.
    stmt = (
        select(Message)
        .options(*MESSAGE_LOAD_OPTIONS)
        .where(Message.id == message_id)
        .execution_options(populate_existing=True)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


def enriched_reactions(
    msg_reactions: list[MessageReaction],
    users: dict[uuid.UUID, User],
    conv: Conversation | None = None,
) -> list[dict]:
    """The react endpoint's shape: [{user_id, user_name, emoji}].

    Takes the conversation so the same tenant filter applies here as in
    serialize_message: this list is broadcast on every reaction_update, so an
    unfiltered one re-discloses a stale foreign author each time anyone reacts.
    """
    out = []
    for r in visible_reactions(msg_reactions, conv, users):
        user = users.get(r.user_id)
        out.append(
            {
                "user_id": str(r.user_id),
                "user_name": user.display_name if user else "Unknown",
                "emoji": r.emoji,
            }
        )
    return out


def is_system_type(t) -> bool:
    return (t.value if hasattr(t, "value") else t) == MessageType.system.value
