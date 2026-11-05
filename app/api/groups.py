"""Group conversation routes: create, update, membership, roles, leave.

Wire contract: docs/reference/api-conversations-messages.md (groups.py section).
Deliberate fixes vs the Mongo build, per that doc's BUGS notes: org isolation on
every route, all-or-nothing member validation before any write, target-must-be-a-
participant checks, the 256-member cap enforced on add, sanitized names and
descriptions, and a role_changed broadcast for creator succession on leave.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.deps import TenantContext, get_tenant
from app.db.models import (
    Conversation,
    ConversationParticipant,
    ConversationType,
    ParticipantRole,
    User,
)
from app.realtime.redis_bus import publish_to_users
from app.services import conversations as conv_service
from app.services import enrich, messaging
from app.utils import now_utc, parse_uuid, sanitize_text

router = APIRouter(prefix="/api/conversations", tags=["groups"])

MAX_GROUP_MEMBERS = 256
_ADMIN_ROLES = (ParticipantRole.creator, ParticipantRole.admin)


# Conversation.name is String(200); anything longer reached Postgres and came
# back as a value-too-long DataError, i.e. a 500 for what is plainly bad input.
# description and avatar_url are unbounded TEXT columns and need no cap here.
_GROUP_NAME_MAX = 200


class CreateGroupRequest(BaseModel):
    name: str = Field(max_length=_GROUP_NAME_MAX)
    description: str | None = None
    avatar_url: str | None = None
    member_ids: list[str]


class UpdateGroupRequest(BaseModel):
    name: str | None = Field(default=None, max_length=_GROUP_NAME_MAX)
    description: str | None = None
    avatar_url: str | None = None
    admin_only_messages: bool | None = None


class AddMembersRequest(BaseModel):
    user_ids: list[str]


class PermissionsRequest(BaseModel):
    """Every field optional — a PUT applies only the keys actually sent."""

    # send_history / invite_via_link / approve_new_members were intentionally gone:
    # nothing enforced them, so accepting them was a success response for a no-op.
    #
    # edit_info and add_members now join them, for the same reason and after the
    # same check: neither perm_edit_info nor perm_add_members was read anywhere
    # outside the model, so this route was storing and broadcasting a policy that
    # update_group and add_members never consulted. Extra keys are ignored rather
    # than rejected (pydantic's default), so a client that has not been updated
    # gets the same 200 it always did — it simply no longer gets a policy back
    # that the server was never applying.
    send_messages: bool | None = None


class ChangeRoleRequest(BaseModel):
    role: str


def _parse_conv_id(conv_id: str) -> uuid.UUID:
    parsed = parse_uuid(conv_id)
    if parsed is None:
        raise HTTPException(status_code=400, detail="Invalid conversation ID")
    return parsed


async def _load_group(tenant: TenantContext, conv_id: uuid.UUID) -> Conversation:
    """Load a group conversation scoped to the caller's org (404 outside it)."""
    conv = await tenant.db.get(Conversation, conv_id)
    if conv is None or conv.type != ConversationType.group:
        raise HTTPException(status_code=404, detail="Group not found")
    if not tenant.is_superadmin and (tenant.org_id is None or conv.org_id != tenant.org_id):
        raise HTTPException(status_code=404, detail="Group not found")
    return conv


async def _require_group_admin(
    tenant: TenantContext, conv_id: uuid.UUID, action: str
) -> tuple[Conversation, ConversationParticipant]:
    """Load the group and assert the caller is a participant with role creator/admin."""
    conv = await _load_group(tenant, conv_id)
    me = await tenant.db.get(ConversationParticipant, (conv.id, tenant.user.id))
    if me is None or me.role not in _ADMIN_ROLES:
        raise HTTPException(status_code=403, detail=f"Only group admins can {action}")
    return conv, me


async def _serialized(db, conv_id: uuid.UUID, for_user: uuid.UUID) -> dict:
    loaded = await enrich.load_conversation_with_participants(db, conv_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Group not found")
    return await enrich.serialize_conversation(db, loaded, for_user)


@router.post("/group")
async def create_group(body: CreateGroupRequest, tenant: TenantContext = Depends(get_tenant)):
    db, user = tenant.db, tenant.user
    if user.org_id is None:
        raise HTTPException(status_code=403, detail="You must belong to an organization to create a group")
    name = sanitize_text(body.name).strip()
    if not name:
        raise HTTPException(status_code=400, detail="Group name is required")
    description = sanitize_text(body.description) if body.description is not None else None

    member_ids: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for raw in body.member_ids:
        member_id = parse_uuid(raw)
        if member_id is None:
            raise HTTPException(status_code=400, detail="Invalid member ID")
        if member_id == user.id or member_id in seen:
            continue
        seen.add(member_id)
        member_ids.append(member_id)
    if len(member_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 members are required")
    # Against the shared constant, not a literal. This read `> 255` — correct only
    # because member_ids excludes the creator, and silently decoupled from
    # MAX_GROUP_MEMBERS, which the add path below does use. Two caps for one rule,
    # either of which could be retuned without the other.
    if len(member_ids) + 1 > MAX_GROUP_MEMBERS:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_GROUP_MEMBERS} members")

    rows = (await db.execute(select(User).where(User.id.in_(member_ids)))).scalars().all()
    by_id = {u.id: u for u in rows}
    members: list[User] = []
    for member_id in member_ids:
        member = by_id.get(member_id)
        if member is None or not member.is_active:
            raise HTTPException(status_code=404, detail="Member not found")
        if member.org_id != user.org_id:
            raise HTTPException(status_code=403, detail="Cannot add members from another organization")
        members.append(member)

    now = now_utc()
    conv = Conversation(
        org_id=user.org_id,
        type=ConversationType.group,
        name=name,
        description=description,
        avatar_url=body.avatar_url,
        created_by=user.id,
        last_message_at=now,
        created_at=now,
    )
    db.add(conv)
    await db.flush()
    db.add(
        ConversationParticipant(
            conversation_id=conv.id, user_id=user.id, role=ParticipantRole.creator, joined_at=now
        )
    )
    for member in members:
        db.add(
            ConversationParticipant(
                conversation_id=conv.id, user_id=member.id, role=ParticipantRole.member, joined_at=now
            )
        )
    await db.commit()

    # One commit for the whole batch. Per-message this cost 7 queries each, all of
    # them producing a document this path discards.
    await messaging.send_system_messages(
        db,
        conv.id,
        [
            f"You created group '{name}'",
            *(f"{user.display_name} added {member.display_name}" for member in members),
        ],
        broadcast=False,
    )
    await conv_service.notify_conversation_created(db, conv.id)
    return await _serialized(db, conv.id, user.id)


@router.put("/{conv_id}/group")
async def update_group(conv_id: str, body: UpdateGroupRequest, tenant: TenantContext = Depends(get_tenant)):
    db, user = tenant.db, tenant.user
    group_id = _parse_conv_id(conv_id)
    conv, _me = await _require_group_admin(tenant, group_id, "update the group")

    data = body.model_dump(exclude_unset=True)
    updates: dict = {}
    if data.get("name") is not None:
        name = sanitize_text(data["name"]).strip()
        if name:
            conv.name = name
            updates["name"] = name
    if "description" in data:
        description = sanitize_text(data["description"]) if data["description"] is not None else None
        conv.description = description
        updates["description"] = description
    if "avatar_url" in data:
        conv.avatar_url = data["avatar_url"]
        updates["avatar_url"] = data["avatar_url"]
    if data.get("admin_only_messages") is not None:
        conv.admin_only_messages = bool(data["admin_only_messages"])
        updates["admin_only_messages"] = conv.admin_only_messages
    await db.commit()

    if "name" in updates:
        text = f"{user.display_name} changed the group name to '{updates['name']}'"
        await messaging.send_system_message(db, conv.id, text)
    if "avatar_url" in updates:
        await messaging.send_system_message(db, conv.id, f"{user.display_name} changed the group icon")
    if updates:
        await conv_service.broadcast_conversation_event(
            db,
            conv.id,
            {"type": "conversation_updated", "conversation_id": str(conv.id), "updates": updates},
        )
    return await _serialized(db, conv.id, user.id)


@router.get("/{conv_id}/permissions")
async def get_permissions(conv_id: str, tenant: TenantContext = Depends(get_tenant)):
    """Any participant of a live in-org group may read the permission set (404 otherwise)."""
    group_id = _parse_conv_id(conv_id)
    # require_membership alone let cross-org groups through: their members are
    # participants, but those conversations are superadmin-managed and belong to no
    # single org. _load_group pins this to a group inside the caller's own org.
    conv = await _load_group(tenant, group_id)
    if not conv.is_active:
        raise HTTPException(status_code=404, detail="Group not found")
    if await tenant.db.get(ConversationParticipant, (conv.id, tenant.user.id)) is None:
        raise HTTPException(status_code=404, detail="Group not found")
    return enrich.serialize_permissions(conv)


@router.put("/{conv_id}/permissions")
async def update_permissions(
    conv_id: str, body: PermissionsRequest, tenant: TenantContext = Depends(get_tenant)
):
    """Creator/admin only (403 for plain members, 404 outside the conversation)."""
    db = tenant.db
    # Gated like every other mutating route here: membership alone let a user who a
    # superadmin made admin of a cross-org group silence participants in other orgs.
    conv, _me = await _require_group_admin(tenant, _parse_conv_id(conv_id), "change permissions")
    if not conv.is_active:
        raise HTTPException(status_code=404, detail="Group not found")

    data = body.model_dump(exclude_unset=True, exclude_none=True)
    if "send_messages" in data:
        # send_messages is admin_only_messages inverted — one stored source of truth.
        conv.admin_only_messages = not bool(data["send_messages"])
    await db.commit()

    permissions = enrich.serialize_permissions(conv)
    if data:
        await conv_service.broadcast_conversation_event(
            db,
            conv.id,
            {
                "type": "permissions_updated",
                "conversation_id": str(conv.id),
                "permissions": permissions,
            },
        )
    return permissions


@router.post("/{conv_id}/members")
async def add_members(conv_id: str, body: AddMembersRequest, tenant: TenantContext = Depends(get_tenant)):
    db, user = tenant.db, tenant.user
    group_id = _parse_conv_id(conv_id)
    conv, _me = await _require_group_admin(tenant, group_id, "add members")
    if not conv.is_active:
        raise HTTPException(status_code=404, detail="Group not found")

    # Lock the conversation row BEFORE reading the membership. The cap is a
    # read-check-insert, and under PostgreSQL's default READ COMMITTED two
    # concurrent adds each see the same pre-insert count: at 255 members both
    # observe one free slot, both pass, and the group ends at 257. Neither request
    # is wrong on its own, which is why the check cannot be made correct by
    # arithmetic — the reads have to be serialized.
    #
    # FOR UPDATE on the parent row rather than on the participants: there is no
    # row to lock for a member who does not exist yet, so the invariant belongs to
    # the conversation. The lock is held to commit, so the second request reads the
    # membership only after the first has finished writing it.
    await db.execute(select(Conversation.id).where(Conversation.id == conv.id).with_for_update())

    stmt = select(ConversationParticipant).where(ConversationParticipant.conversation_id == conv.id)
    existing = (await db.execute(stmt)).scalars().all()
    existing_ids = {p.user_id for p in existing}

    # Validate everything BEFORE any write: the reference persisted system
    # messages mid-loop and could 403 after a partial write.
    candidates: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set(existing_ids)
    for raw in body.user_ids:
        candidate = parse_uuid(raw)
        if candidate is None or candidate in seen:
            continue  # invalid ids and existing members are skipped silently
        seen.add(candidate)
        candidates.append(candidate)

    new_members: list[User] = []
    if candidates:
        rows = (await db.execute(select(User).where(User.id.in_(candidates)))).scalars().all()
        by_id = {u.id: u for u in rows}
        for candidate in candidates:
            member = by_id.get(candidate)
            if member is None or not member.is_active:
                continue  # nonexistent/inactive users are skipped silently
            if member.org_id != conv.org_id:
                raise HTTPException(status_code=403, detail="Cannot add members from another organization")
            new_members.append(member)

    if len(existing) + len(new_members) > MAX_GROUP_MEMBERS:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_GROUP_MEMBERS} members")

    if new_members:
        now = now_utc()
        for member in new_members:
            db.add(
                ConversationParticipant(
                    conversation_id=conv.id, user_id=member.id, role=ParticipantRole.member, joined_at=now
                )
            )
        await db.commit()
        # Still one frame per member — the batching is in the commit, not in what
        # subscribers receive.
        await messaging.send_system_messages(
            db,
            conv.id,
            [f"{user.display_name} added {member.display_name}" for member in new_members],
        )
        loaded = await enrich.load_conversation_with_participants(db, conv.id)
        for uid in existing_ids:
            doc = await enrich.serialize_conversation(db, loaded, uid)
            await publish_to_users(
                [uid],
                {"type": "member_added", "conversation_id": str(conv.id), "conversation": doc},
            )
        await conv_service.notify_conversation_created(db, conv.id, only=[m.id for m in new_members])
    return await _serialized(db, conv.id, user.id)


@router.delete("/{conv_id}/members/{member_id}")
async def remove_member(conv_id: str, member_id: str, tenant: TenantContext = Depends(get_tenant)):
    db, user = tenant.db, tenant.user
    group_id = _parse_conv_id(conv_id)
    target_id = parse_uuid(member_id)
    if target_id is None:
        raise HTTPException(status_code=400, detail="Invalid member ID")
    conv, me = await _require_group_admin(tenant, group_id, "remove members")

    target = await db.get(ConversationParticipant, (conv.id, target_id))
    if target is None:
        raise HTTPException(status_code=404, detail="Member not found")
    if target.role == ParticipantRole.creator:
        raise HTTPException(status_code=403, detail="Cannot remove the group creator")
    if me.role == ParticipantRole.admin and target.role == ParticipantRole.admin:
        raise HTTPException(status_code=403, detail="Admins cannot remove other admins")

    target_user = await db.get(User, target_id)
    target_name = target_user.display_name if target_user else "Unknown"
    await db.delete(target)
    await messaging.send_system_message(db, conv.id, f"{user.display_name} removed {target_name}")
    await conv_service.broadcast_conversation_event(
        db,
        conv.id,
        {
            "type": "member_removed",
            "conversation_id": str(conv.id),
            "user_id": str(target_id),
            "removed_by": str(user.id),
        },
    )
    await publish_to_users(
        [target_id], {"type": "removed_from_conversation", "conversation_id": str(conv.id)}
    )
    return {"message": f"{target_name} removed from group"}


@router.put("/{conv_id}/members/{member_id}/role")
async def change_member_role(
    conv_id: str,
    member_id: str,
    body: ChangeRoleRequest,
    tenant: TenantContext = Depends(get_tenant),
):
    db, user = tenant.db, tenant.user
    group_id = _parse_conv_id(conv_id)
    target_id = parse_uuid(member_id)
    if target_id is None:
        raise HTTPException(status_code=400, detail="Invalid member ID")
    if body.role not in ("admin", "member"):
        raise HTTPException(status_code=400, detail="Invalid role")

    conv = await _load_group(tenant, group_id)
    me = await db.get(ConversationParticipant, (conv.id, user.id))
    if me is None or me.role != ParticipantRole.creator:
        raise HTTPException(status_code=403, detail="Only the group creator can change roles")
    if target_id == user.id:
        raise HTTPException(status_code=400, detail="Cannot change your own role")

    target = await db.get(ConversationParticipant, (conv.id, target_id))
    if target is None:
        raise HTTPException(status_code=404, detail="Member not found")
    target_user = await db.get(User, target_id)
    target_name = target_user.display_name if target_user else "Unknown"

    target.role = ParticipantRole(body.role)
    if body.role == "admin":
        text = f"{user.display_name} made {target_name} an admin"
    else:
        text = f"{user.display_name} removed {target_name} as admin"
    await messaging.send_system_message(db, conv.id, text)
    await conv_service.broadcast_conversation_event(
        db,
        conv.id,
        {
            "type": "role_changed",
            "conversation_id": str(conv.id),
            "user_id": str(target_id),
            "new_role": body.role,
        },
    )
    return {"message": text}


@router.post("/{conv_id}/leave")
async def leave_group(conv_id: str, tenant: TenantContext = Depends(get_tenant)):
    db, user = tenant.db, tenant.user
    group_id = _parse_conv_id(conv_id)
    conv = await _load_group(tenant, group_id)

    participants = (
        (
            await db.execute(
                select(ConversationParticipant)
                .where(ConversationParticipant.conversation_id == conv.id)
                .order_by(ConversationParticipant.joined_at)
            )
        )
        .scalars()
        .all()
    )
    me = next((p for p in participants if p.user_id == user.id), None)
    if me is None:
        raise HTTPException(status_code=403, detail="You are not a member of this group")

    remaining = [p for p in participants if p.user_id != user.id]
    promoted: ConversationParticipant | None = None
    if me.role == ParticipantRole.creator and remaining:
        # Succession: earliest-joined admin, else earliest-joined member.
        promoted = next((p for p in remaining if p.role == ParticipantRole.admin), remaining[0])
        promoted.role = ParticipantRole.creator
    if not remaining:
        conv.is_active = False
    await db.delete(me)

    await messaging.send_system_message(db, conv.id, f"{user.display_name} left the group")
    await conv_service.broadcast_conversation_event(
        db, conv.id, {"type": "member_left", "conversation_id": str(conv.id), "user_id": str(user.id)}
    )
    if promoted is not None:
        await conv_service.broadcast_conversation_event(
            db,
            conv.id,
            {
                "type": "role_changed",
                "conversation_id": str(conv.id),
                "user_id": str(promoted.user_id),
                "new_role": "creator",
            },
        )
    return {"message": "Left group"}
