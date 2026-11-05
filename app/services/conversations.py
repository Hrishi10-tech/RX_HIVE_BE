"""Conversation lifecycle: direct find-or-create, group create/update helpers."""

import uuid

from fastapi import HTTPException
from sqlalchemy import and_, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Conversation,
    ConversationParticipant,
    ConversationType,
    Message,
    MessageType,
    ParticipantRole,
    User,
    UserRole,
)
from app.realtime.redis_bus import publish_to_users
from app.services import enrich
from app.utils import now_utc


async def find_direct(db: AsyncSession, user_a: uuid.UUID, user_b: uuid.UUID) -> Conversation | None:
    cp1 = ConversationParticipant.__table__.alias("cp1")
    cp2 = ConversationParticipant.__table__.alias("cp2")
    stmt = (
        select(Conversation)
        .join(cp1, and_(cp1.c.conversation_id == Conversation.id, cp1.c.user_id == user_a))
        .join(cp2, and_(cp2.c.conversation_id == Conversation.id, cp2.c.user_id == user_b))
        .where(
            Conversation.type == ConversationType.direct,
            Conversation.is_active.is_(True),
        )
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def get_or_create_direct(
    db: AsyncSession, me: User, other: User, *, notify: bool = False
) -> Conversation:
    # THE tenant choke point. Three separate paths manufacture a direct
    # conversation — POST /conversations/direct, forward-to-contacts, and call
    # initiation — and all three land here. Guarding any one endpoint instead
    # would leave the other two as ways to reach someone in another tenant.
    #
    # Anyone may message anyone inside their own organisation; this is the only
    # thing left standing between them. It used to be the first clause of
    # access.can_converse, which is gone, so it is restated here rather than
    # inherited — deleting the rules engine must not quietly open direct
    # conversations BETWEEN organisations, which is a different feature
    # (cross-org groups) with its own membership model.
    #
    # Superadmins are exempt, as they were before: org_id is NULL for them and
    # they drive cross-org group machinery. They have no chat surface of their
    # own, so this is not a way in.
    if (
        me.role != UserRole.superadmin
        and other.role != UserRole.superadmin
        and (me.org_id is None or other.org_id is None or me.org_id != other.org_id)
    ):
        raise HTTPException(status_code=403, detail="You are not permitted to message this person")

    # Serialize the check-then-act below, per pair of users. Nothing in the schema
    # constrains "one direct conversation per pair" — conversations has only an
    # org_id index — so two requests that interleave between the lookup and the
    # commit both create one. That is reachable in ordinary use, not just under
    # load: POST /api/conversations/direct, forward_message's contact fan-out and
    # initiate_direct_call all land here, so Alice tapping call on Bob at the
    # moment Bob opens a chat with her is enough. The duplicates are permanent and
    # user-visible — two threads with the same person, each holding half the
    # history — and neither client has any way to merge them.
    #
    # An advisory lock rather than a unique index because the index would need a
    # migration over existing data that may already contain duplicates. The lock
    # is transaction-scoped, so the commit below releases it.
    lo, hi = sorted((me.id, other.id), key=str)
    await db.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:pair, 0))").bindparams(pair=f"direct:{lo}:{hi}")
    )

    existing = await find_direct(db, me.id, other.id)
    if existing is not None:
        return existing

    now = now_utc()
    conv = Conversation(
        org_id=me.org_id,
        type=ConversationType.direct,
        created_by=me.id,
        last_message_at=now,
        created_at=now,
    )
    db.add(conv)
    await db.flush()
    db.add_all(
        [
            ConversationParticipant(conversation_id=conv.id, user_id=me.id, role=ParticipantRole.member),
            ConversationParticipant(conversation_id=conv.id, user_id=other.id, role=ParticipantRole.member),
        ]
    )
    db.add(
        Message(
            conversation_id=conv.id,
            sender_id=None,
            type=MessageType.system,
            content="Conversation started",
            created_at=now,
        )
    )
    await db.commit()

    if notify:
        loaded = await enrich.load_conversation_with_participants(db, conv.id)
        for uid in (me.id, other.id):
            doc = await enrich.serialize_conversation(db, loaded, uid)
            await publish_to_users([uid], {"type": "conversation_created", "conversation": doc})
    return conv


async def participant_ids(db: AsyncSession, conversation_id: uuid.UUID) -> list[uuid.UUID]:
    """Raw membership, unfiltered — the honest answer to "who is a member".

    Deliberately NOT tenant-filtered, because its callers ask two different
    questions and only one of them is about disclosure. cross_org.py uses it to
    build dedup and emptiness sets, where a foreign-org row must still count as
    present or an add would collide with the primary key. Anything that FANS OUT
    must use enrich.tenant_participants instead — see broadcast_conversation_event
    directly below, which used this and should not have.
    """
    return (
        (
            await db.execute(
                select(ConversationParticipant.user_id).where(
                    ConversationParticipant.conversation_id == conversation_id
                )
            )
        )
        .scalars()
        .all()
    )


async def broadcast_conversation_event(
    db: AsyncSession, conversation_id: uuid.UUID, event: dict, *, exclude: uuid.UUID | None = None
) -> None:
    # Tenant-filtered, unlike participant_ids above. Every caller of this is in
    # api/groups.py — ordinary group conversations, not cross_org ones — and the
    # events are icon changes, renames, member removals and leaves. A foreign-org
    # participant row in one of those groups was receiving all of it.
    rows = await enrich.tenant_participants(db, conversation_id)
    targets = [p.user_id for p in rows if p.user_id != exclude]
    await publish_to_users(targets, event)


async def notify_conversation_created(
    db: AsyncSession, conversation_id: uuid.UUID, *, only: list[uuid.UUID] | None = None
) -> None:
    """Send each participant a conversation_created carrying THEIR enriched view."""
    loaded = await enrich.load_conversation_with_participants(db, conversation_id)
    if loaded is None:
        return
    # `loaded.participants` is the raw relationship, so the default audience needs
    # the same filter as everything else — this one hands over a fully serialized
    # conversation (name, participant list, last message), which is the largest
    # single disclosure of the set. An explicit `only` is the caller naming its own
    # recipients and is left alone.
    if only is not None:
        targets = only
    else:
        targets = [p.user_id for p in await enrich.tenant_participants(db, conversation_id)]
    for uid in targets:
        doc = await enrich.serialize_conversation(db, loaded, uid)
        await publish_to_users([uid], {"type": "conversation_created", "conversation": doc})


async def count_group_participants(db: AsyncSession, conversation_id: uuid.UUID) -> int:
    return (
        await db.execute(
            select(func.count())
            .select_from(ConversationParticipant)
            .where(ConversationParticipant.conversation_id == conversation_id)
        )
    ).scalar_one()
