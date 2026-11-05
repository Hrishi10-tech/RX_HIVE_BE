"""Global search + profile update — GET /api/search, PUT /api/users/profile.

The profile route lives here to mirror the RxHivexx layout the frontend was
built against (docs/reference/api-uploads-calls-misc.md).

Deliberate fixes vs the Mongo build:
- Search input is escaped + parameterized (was raw $regex — injection/ReDoS).
- The conversations bucket enforces org visibility: org match OR a cross_org
  group whose allowed_org_ids contains the caller's org (the duplicate "$or"
  key in the Mongo query silently dropped this scoping).
- Message search excludes delete-for-me rows and batches enrichment (no N+1).
- avatar_url accepts only relative /api/media/ URLs or empty/null (was
  unvalidated — javascript:/data:/external URLs went straight to the DB).
- Null org/dept no longer 500s the profile response (superadmin accounts).
- Profile changes broadcast a `profile_updated` event to conversation partners
  (the old build left other users' cached names/avatars stale).
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.api.auth import wire_role
from app.core.deps import TenantContext, get_tenant
from app.db.models import (
    Conversation,
    ConversationParticipant,
    ConversationType,
    Department,
    Message,
    MessageDeletion,
    MessageType,
    Organization,
    User,
)
from app.realtime.redis_bus import publish_to_users
from app.services import presence
from app.utils import iso_z, sanitize_text

router = APIRouter(prefix="/api", tags=["search"])

_BUCKET_LIMIT = 5
_AVATAR_PREFIX = "/api/media/"
_AVATAR_MAX_LENGTH = 500


def _like_pattern(value: str) -> str:
    """Escape LIKE wildcards so user input is matched literally."""
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


async def _direct_partners(
    db: AsyncSession, conversation_ids: list[uuid.UUID], me_id: uuid.UUID
) -> dict[uuid.UUID, User]:
    """Batch-load the other participant of each direct conversation."""
    if not conversation_ids:
        return {}
    rows = (
        await db.execute(
            select(ConversationParticipant.conversation_id, User)
            .join(User, User.id == ConversationParticipant.user_id)
            .where(
                ConversationParticipant.conversation_id.in_(conversation_ids),
                ConversationParticipant.user_id != me_id,
            )
        )
    ).all()
    return {conv_id: partner for conv_id, partner in rows}


async def _conversation_partner_ids(db: AsyncSession, user_id: uuid.UUID) -> list[uuid.UUID]:
    """Distinct users sharing at least one conversation with user_id.

    Mirrors app.realtime.hub._conversation_partner_ids — kept local so the
    HTTP router does not import the websocket module.
    """
    mine = select(ConversationParticipant.conversation_id).where(ConversationParticipant.user_id == user_id)
    stmt = (
        select(ConversationParticipant.user_id)
        .where(
            ConversationParticipant.conversation_id.in_(mine),
            ConversationParticipant.user_id != user_id,
        )
        .distinct()
    )
    return list((await db.execute(stmt)).scalars().all())


@router.get("/search")
async def global_search(
    q: str = Query("", min_length=1),
    types: str = Query("conversations,contacts,messages"),
    tenant: TenantContext = Depends(get_tenant),
) -> dict:
    """Search conversations, contacts, and messages — max 5 hits per bucket.

    All three keys are always present. Callers without an org (superadmins)
    and requests without a q get empty buckets.
    """
    results: dict = {"conversations": [], "contacts": [], "messages": []}
    user = tenant.user
    if not q or user.org_id is None:
        return results

    db = tenant.db
    requested = {t.strip() for t in types.split(",") if t.strip()}
    pattern = _like_pattern(q)

    conv_rows: list[Conversation] = []
    msg_rows: list = []

    if "conversations" in requested:
        other_cp = aliased(ConversationParticipant)
        partner = aliased(User)
        partner_name_match = (
            select(other_cp.user_id)
            .join(partner, partner.id == other_cp.user_id)
            .where(
                other_cp.conversation_id == Conversation.id,
                other_cp.user_id != user.id,
                partner.display_name.ilike(pattern, escape="\\"),
            )
            .exists()
        )
        stmt = (
            select(Conversation)
            .join(
                ConversationParticipant,
                and_(
                    ConversationParticipant.conversation_id == Conversation.id,
                    ConversationParticipant.user_id == user.id,
                ),
            )
            .where(
                Conversation.is_active.is_(True),
                or_(
                    Conversation.org_id == user.org_id,
                    and_(
                        Conversation.type == ConversationType.cross_org,
                        Conversation.allowed_org_ids.contains([str(user.org_id)]),
                    ),
                ),
                or_(
                    and_(
                        Conversation.type != ConversationType.direct,
                        Conversation.name.ilike(pattern, escape="\\"),
                    ),
                    and_(Conversation.type == ConversationType.direct, partner_name_match),
                ),
            )
            .order_by(Conversation.last_message_at.desc().nulls_last())
            .limit(_BUCKET_LIMIT)
        )
        conv_rows = list((await db.execute(stmt)).scalars().all())

    if "contacts" in requested:
        stmt = (
            select(User, Department.name)
            .outerjoin(Department, Department.id == User.dept_id)
            .where(
                User.org_id == user.org_id,
                User.is_active.is_(True),
                User.id != user.id,
                or_(
                    User.display_name.ilike(pattern, escape="\\"),
                    User.email.ilike(pattern, escape="\\"),
                ),
            )
            .order_by(User.display_name.asc())
            .limit(_BUCKET_LIMIT)
        )
        for contact, dept_name in (await db.execute(stmt)).all():
            results["contacts"].append(
                {
                    "id": str(contact.id),
                    "display_name": contact.display_name,
                    "email": contact.email,
                    "avatar_url": contact.avatar_url,
                    "department": dept_name or "",
                }
            )

    if "messages" in requested:
        sender = aliased(User)
        stmt = (
            select(
                Message.id,
                Message.conversation_id,
                Message.content,
                Message.created_at,
                Conversation.type,
                Conversation.name,
                sender.display_name,
            )
            .join(Conversation, Conversation.id == Message.conversation_id)
            .join(
                ConversationParticipant,
                and_(
                    ConversationParticipant.conversation_id == Message.conversation_id,
                    ConversationParticipant.user_id == user.id,
                ),
            )
            .outerjoin(sender, sender.id == Message.sender_id)
            .outerjoin(
                MessageDeletion,
                and_(
                    MessageDeletion.message_id == Message.id,
                    MessageDeletion.user_id == user.id,
                ),
            )
            .where(
                # The same revocation gate _require_org_access applies to the
                # message read paths (api/messages.py) and the media drawer
                # (api/media.py). Deactivation is the only lever a superadmin has
                # to cut a tenant off from a cross-org group — DELETE and the
                # archive route both just set is_active = False and leave the
                # participant rows in place — so without this the join on
                # ConversationParticipant still matched and full message content
                # from a revoked group kept coming back here. The conversations
                # bucket above has always filtered it; the two halves of one
                # response disagreed, and this was the half that leaked.
                Conversation.is_active.is_(True),
                # The org half of the same gate. Mirrors _require_org_access, not
                # the conversations bucket above: a cross_org conversation passes
                # on membership, because a member's org is appended to
                # allowed_org_ids when they are added and never removed
                # (api/cross_org.py), so membership already implies scope. Testing
                # allowed_org_ids here would make a snippet vanish for a caller
                # whose message list still renders it.
                or_(
                    Conversation.type == ConversationType.cross_org,
                    Conversation.org_id == user.org_id,
                ),
                Message.type == MessageType.text,
                Message.deleted_at.is_(None),
                MessageDeletion.message_id.is_(None),
                Message.content.ilike(pattern, escape="\\"),
            )
            .order_by(Message.created_at.desc())
            .limit(_BUCKET_LIMIT)
        )
        msg_rows = (await db.execute(stmt)).all()

    # One batch query resolves direct-chat display names for both buckets.
    direct_ids = {c.id for c in conv_rows if c.type == ConversationType.direct}
    direct_ids |= {row.conversation_id for row in msg_rows if row.type == ConversationType.direct}
    partners = await _direct_partners(db, list(direct_ids), user.id)

    for conv in conv_rows:
        if conv.type == ConversationType.direct:
            p = partners.get(conv.id)
            name = p.display_name if p else "Unknown"
            avatar = p.avatar_url if p else None
        else:
            name = conv.name
            avatar = conv.avatar_url
        results["conversations"].append(
            {
                "id": str(conv.id),
                "name": name,
                "type": conv.type.value,
                "avatar_url": avatar,
            }
        )

    for row in msg_rows:
        if row.type == ConversationType.direct:
            p = partners.get(row.conversation_id)
            conv_name = p.display_name if p else "Unknown"
        else:
            conv_name = row.name or ""
        results["messages"].append(
            {
                "message_id": str(row.id),
                "conversation_id": str(row.conversation_id),
                "conversation_name": conv_name,
                "content_snippet": (row.content or "")[:200],
                "sender_name": row.display_name or "System",
                "created_at": iso_z(row.created_at),
            }
        )

    return results


class ProfileUpdate(BaseModel):
    display_name: str | None = None
    about: str | None = None
    avatar_url: str | None = None


@router.put("/users/profile")
async def update_profile(
    body: ProfileUpdate,
    tenant: TenantContext = Depends(get_tenant),
) -> dict:
    user = tenant.user
    db = tenant.db
    updated = False

    if body.display_name is not None:
        name = sanitize_text(body.display_name).strip()
        if not 1 <= len(name) <= 50:
            raise HTTPException(status_code=400, detail="Display name must be 1-50 characters")
        user.display_name = name
        updated = True

    if body.about is not None:
        about = sanitize_text(body.about).strip()
        if len(about) > 140:
            raise HTTPException(status_code=400, detail="About must be 140 characters or less")
        user.about = about
        updated = True

    if "avatar_url" in body.model_fields_set:
        avatar = (body.avatar_url or "").strip()
        if avatar and not (avatar.startswith(_AVATAR_PREFIX) and len(avatar) <= _AVATAR_MAX_LENGTH):
            raise HTTPException(status_code=400, detail="Invalid avatar URL")
        user.avatar_url = avatar or None
        updated = True

    if not updated:
        raise HTTPException(status_code=400, detail="No fields to update")

    await db.commit()

    org_name = ""
    dept_name = ""
    if user.org_id is not None:
        row = await db.execute(select(Organization.name).where(Organization.id == user.org_id))
        org_name = row.scalar_one_or_none() or ""
    if user.dept_id is not None:
        row = await db.execute(select(Department.name).where(Department.id == user.dept_id))
        dept_name = row.scalar_one_or_none() or ""

    statuses = await presence.get_statuses([user.id])

    partners = await _conversation_partner_ids(db, user.id)
    if partners:
        await publish_to_users(
            partners,
            {
                "type": "profile_updated",
                "user_id": str(user.id),
                "display_name": user.display_name,
                "avatar_url": user.avatar_url,
                "about": user.about or "",
            },
        )

    return {
        "id": str(user.id),
        "email": user.email,
        "display_name": user.display_name,
        "about": user.about or "",
        "avatar_url": user.avatar_url,
        "role": wire_role(user.role),
        "status": statuses.get(str(user.id), "offline"),
        "org_name": org_name,
        "dept_name": dept_name,
        "org_id": str(user.org_id) if user.org_id else None,
        "dept_id": str(user.dept_id) if user.dept_id else None,
    }
