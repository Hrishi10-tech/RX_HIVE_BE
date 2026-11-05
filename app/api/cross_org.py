"""Superadmin cross-org group management.

Cross-org groups are conversations of type `cross_org` with org_id NULL and an
allowed_org_ids JSONB list. Wire shapes preserve the RxHivexx contract
(docs/reference/api-uploads-calls-misc.md): `_id` string keys, ISO-Z
datetimes, admin-enriched participant/organization/last_message blocks.

Fixes vs the Mongo build: the >=2-members invariant is enforced AFTER
filtering invalid member ids (the old build counted raw input, so a group
could be created with one real participant); list search uses escaped
parameterized ILIKE instead of raw-interpolated $regex; removing a user who
is not actually a member is a no-op (no phantom system messages/audit rows).
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.groups import MAX_GROUP_MEMBERS
from app.core.deps import require_superadmin
from app.db.models import (
    Conversation,
    ConversationParticipant,
    ConversationType,
    Message,
    Organization,
    ParticipantRole,
    User,
)
from app.db.session import get_db
from app.realtime.redis_bus import publish_to_users
from app.services import presence
from app.services.audit import log_audit
from app.services.conversations import notify_conversation_created, participant_ids
from app.services.messaging import send_system_message, send_system_messages
from app.utils import iso_z, now_utc, parse_uuid, sanitize_text

router = APIRouter(prefix="/api/admin/cross-org-groups", tags=["cross-org"])


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _participant_role(role: str | None) -> ParticipantRole:
    return ParticipantRole.admin if role == "admin" else ParticipantRole.member


async def _last_messages(db: AsyncSession, conv_ids: list[uuid.UUID]) -> dict[uuid.UUID, dict]:
    """Latest message per conversation, projected to the contract's 4 keys."""
    if not conv_ids:
        return {}
    ranked = (
        select(
            Message.conversation_id,
            Message.content,
            Message.deleted_at,
            Message.type,
            Message.created_at,
            Message.sender_id,
            func.row_number()
            .over(partition_by=Message.conversation_id, order_by=Message.created_at.desc())
            .label("rn"),
        )
        .where(Message.conversation_id.in_(conv_ids))
        .subquery()
    )
    stmt = (
        select(ranked, User.display_name)
        .outerjoin(User, User.id == ranked.c.sender_id)
        .where(ranked.c.rn == 1)
    )
    rows = (await db.execute(stmt)).all()
    out: dict[uuid.UUID, dict] = {}
    for row in rows:
        m = row._mapping
        mtype = m["type"]
        out[m["conversation_id"]] = {
            "content": "" if m["deleted_at"] else (m["content"] or ""),
            "sender_name": row.display_name or "System",
            "created_at": iso_z(m["created_at"]),
            "type": mtype.value if hasattr(mtype, "value") else (mtype or "text"),
        }
    return out


async def _enrich_groups(db: AsyncSession, convs: list[Conversation]) -> list[dict]:
    """Admin-enriched group docs, batched across the whole page (no N+1)."""
    if not convs:
        return []
    conv_ids = [c.id for c in convs]
    rows = (
        await db.execute(
            select(ConversationParticipant, User)
            .join(User, User.id == ConversationParticipant.user_id)
            .where(ConversationParticipant.conversation_id.in_(conv_ids))
        )
    ).all()

    org_ids: set[uuid.UUID] = set()
    for conv in convs:
        for oid in conv.allowed_org_ids or []:
            parsed = parse_uuid(oid)
            if parsed:
                org_ids.add(parsed)
    for _, member in rows:
        if member.org_id:
            org_ids.add(member.org_id)
    orgs: dict[uuid.UUID, Organization] = {}
    if org_ids:
        org_rows = (
            (await db.execute(select(Organization).where(Organization.id.in_(org_ids)))).scalars().all()
        )
        orgs = {o.id: o for o in org_rows}

    statuses = await presence.get_statuses(list({member.id for _, member in rows}))
    last = await _last_messages(db, conv_ids)

    by_conv: dict[uuid.UUID, list[tuple[ConversationParticipant, User]]] = {}
    for cp, member in rows:
        by_conv.setdefault(cp.conversation_id, []).append((cp, member))

    docs = []
    for conv in convs:
        parts = []
        for cp, member in by_conv.get(conv.id, []):
            org = orgs.get(member.org_id) if member.org_id else None
            parts.append(
                {
                    "user_id": str(member.id),
                    "display_name": member.display_name,
                    "email": member.email,
                    "avatar_url": member.avatar_url,
                    "role": cp.role.value,
                    "org_name": org.name if org else "Unknown",
                    "org_id": str(member.org_id) if member.org_id else None,
                    "status": statuses.get(str(member.id), "offline"),
                }
            )
        organizations = []
        for oid in conv.allowed_org_ids or []:
            parsed = parse_uuid(oid)
            if parsed and parsed in orgs:
                organizations.append({"id": str(parsed), "name": orgs[parsed].name})
        docs.append(
            {
                "_id": str(conv.id),
                "type": "cross_org",
                "org_id": None,
                "cross_org": True,
                "allowed_org_ids": [str(o) for o in (conv.allowed_org_ids or [])],
                "name": conv.name,
                "description": conv.description,
                "avatar_url": conv.avatar_url,
                "purpose_tag": conv.purpose_tag,
                "created_by": str(conv.created_by) if conv.created_by else None,
                "created_at": iso_z(conv.created_at),
                "last_message_at": iso_z(conv.last_message_at),
                "pinned_by": [str(cp.user_id) for cp, _ in by_conv.get(conv.id, []) if cp.is_pinned],
                "is_active": conv.is_active,
                "admin_only_messages": conv.admin_only_messages,
                "participants": parts,
                "organizations": organizations,
                "member_count": len(parts),
                "last_message": last.get(conv.id),
            }
        )
    return docs


async def _enrich_group(db: AsyncSession, conv: Conversation) -> dict:
    return (await _enrich_groups(db, [conv]))[0]


async def _load_group(
    db: AsyncSession,
    group_id: str,
    *,
    active_only: bool = False,
    invalid_detail: str = "Invalid group ID",
) -> Conversation:
    gid = parse_uuid(group_id)
    if gid is None:
        raise HTTPException(status_code=400, detail=invalid_detail)
    # deleted_at is checked here rather than in each route: a deleted group is
    # gone, so archive, update, member changes and a second delete all 404
    # instead of operating on it. Archiving used to be the loud one — it flips
    # is_active, which a delete also set, so the toggle read a deleted group as
    # archived and restored it — but every route that mutates conv had the same
    # hole through it.
    stmt = select(Conversation).where(
        Conversation.id == gid,
        Conversation.type == ConversationType.cross_org,
        Conversation.deleted_at.is_(None),
    )
    if active_only:
        stmt = stmt.where(Conversation.is_active.is_(True))
    conv = (await db.execute(stmt)).scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="Group not found")
    return conv


@router.get("")
async def list_groups(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: str = Query(""),
    status: str = Query("active"),
    actor: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    _ = actor
    # Deleted groups are excluded from every status, "archived" included: they
    # share is_active = False with archived ones, so listing them there offered
    # the admin an unarchive button for a group they had destroyed.
    stmt = select(Conversation).where(
        Conversation.type == ConversationType.cross_org, Conversation.deleted_at.is_(None)
    )
    if status == "active":
        stmt = stmt.where(Conversation.is_active.is_(True))
    elif status == "archived":
        stmt = stmt.where(Conversation.is_active.is_(False))
    if search:
        pattern = f"%{_escape_like(search)}%"
        stmt = stmt.where(Conversation.name.ilike(pattern, escape="\\"))
    total = (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar_one()
    convs = (
        (
            await db.execute(
                stmt.order_by(Conversation.created_at.desc()).offset((page - 1) * limit).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return {
        "data": await _enrich_groups(db, list(convs)),
        "total": total,
        "page": page,
        "limit": limit,
    }


class CreateCrossOrgGroup(BaseModel):
    name: str
    description: str | None = None
    avatar_url: str | None = None
    purpose_tag: str = "Project"
    org_ids: list[str]
    members: list[dict]


@router.post("")
async def create_group(
    body: CreateCrossOrgGroup,
    actor: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    name = sanitize_text(body.name).strip()
    if not name:
        raise HTTPException(status_code=400, detail="Group name required")
    # De-duplicated BEFORE the >=2 check rather than after it. The check counted
    # the raw request, so one organization sent twice satisfied it and then
    # collapsed to a group spanning a single org — a cross-org group crossing
    # nothing. Same rule as the members invariant below: count what survives
    # filtering, not what was sent. The map keeps each id's original spelling for
    # the error messages.
    by_uuid: dict[uuid.UUID, str] = {}
    for oid in body.org_ids:
        parsed = parse_uuid(oid)
        if parsed is None:
            raise HTTPException(status_code=400, detail=f"Invalid org ID: {oid}")
        by_uuid.setdefault(parsed, oid)
    org_uuids = list(by_uuid)
    if len(org_uuids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 organizations required")

    orgs: dict[uuid.UUID, Organization] = {
        o.id: o
        for o in (
            (await db.execute(select(Organization).where(Organization.id.in_(org_uuids)))).scalars().all()
        )
    }
    for parsed, oid in by_uuid.items():
        org = orgs.get(parsed)
        if org is None or not org.is_active:
            raise HTTPException(status_code=404, detail=f"Organization not found: {oid}")

    if len(body.members) < 2:
        raise HTTPException(status_code=400, detail="At least 2 members required")
    specs: list[tuple[uuid.UUID, str]] = []
    seen: set[uuid.UUID] = set()
    for m in body.members:
        uid = parse_uuid(m.get("user_id"))
        if uid is None or uid in seen:
            continue  # invalid/duplicate member ids are silently skipped
        seen.add(uid)
        specs.append((uid, "admin" if m.get("role") == "admin" else "member"))
    users_map: dict[uuid.UUID, User] = {}
    if seen:
        user_rows = (
            (await db.execute(select(User).where(User.id.in_(seen), User.is_active.is_(True))))
            .scalars()
            .all()
        )
        users_map = {u.id: u for u in user_rows}

    allowed_set = set(org_uuids)
    valid: list[tuple[User, str]] = []
    for uid, role in specs:
        member = users_map.get(uid)
        if member is None:
            continue  # unknown/inactive member ids are silently skipped
        if member.org_id not in allowed_set:
            raise HTTPException(
                status_code=400,
                detail=f"User {member.display_name} does not belong to a selected organization",
            )
        valid.append((member, role))
    # Enforced AFTER filtering — the Mongo build counted the raw input.
    if len(valid) < 2:
        raise HTTPException(status_code=400, detail="At least 2 members required")
    # A MAXIMUM, which this file had never had. Every other size rule here is a
    # floor — two orgs, two members, one admin, one member per org — and nothing
    # anywhere bounded the ceiling, while a same-org group has been capped at
    # MAX_GROUP_MEMBERS all along.
    #
    # That asymmetry is not theoretical; it is cited as the reason two separate
    # tails were reachable. services/messaging.py measured a per-recipient set
    # rebuild at 703ms of event-loop time for 5,000 recipients and notes regular
    # groups "stayed inside a couple of milliseconds" only because of their cap.
    # services/push.py notes the fan-out's duration scales with the subscription
    # count and that nothing caps it. Both fixed their own symptom; this is the
    # cause. It also bounds the per-member loops below, which commit once each.
    if len(valid) > MAX_GROUP_MEMBERS:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_GROUP_MEMBERS} members")
    if not any(role == "admin" for _, role in valid):
        raise HTTPException(status_code=400, detail="At least 1 admin required")
    for org_uuid in org_uuids:
        if not any(member.org_id == org_uuid for member, _ in valid):
            raise HTTPException(
                status_code=400,
                detail=f"At least 1 member required from {orgs[org_uuid].name}",
            )

    now = now_utc()
    conv = Conversation(
        org_id=None,
        type=ConversationType.cross_org,
        name=name,
        description=sanitize_text(body.description) if body.description else None,
        avatar_url=body.avatar_url,
        purpose_tag=sanitize_text(body.purpose_tag) if body.purpose_tag else None,
        allowed_org_ids=[str(o) for o in org_uuids],
        admin_only_messages=False,
        is_active=True,
        created_by=actor.id,
        last_message_at=now,
        created_at=now,
    )
    db.add(conv)
    await db.flush()
    for member, role in valid:
        db.add(
            ConversationParticipant(conversation_id=conv.id, user_id=member.id, role=_participant_role(role))
        )
    await db.commit()

    await send_system_messages(
        db,
        conv.id,
        [
            f"Cross-org group {name} was created",
            *(f"Admin added {member.display_name}" for member, _role in valid),
        ],
        broadcast=False,
    )

    await notify_conversation_created(db, conv.id)

    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="cross_org_group_created",
        target=name,
        details={
            "group_id": str(conv.id),
            "org_ids": [str(o) for o in org_uuids],
            "members": [str(member.id) for member, _ in valid],
        },
    )
    await db.commit()
    return await _enrich_group(db, conv)


@router.get("/{group_id}")
async def get_group(
    group_id: str,
    actor: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    _ = actor
    conv = await _load_group(db, group_id)
    return await _enrich_group(db, conv)


class UpdateCrossOrgGroup(BaseModel):
    name: str | None = None
    description: str | None = None
    avatar_url: str | None = None
    purpose_tag: str | None = None
    is_active: bool | None = None


@router.put("/{group_id}")
async def update_group(
    group_id: str,
    body: UpdateCrossOrgGroup,
    actor: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    conv = await _load_group(db, group_id)
    was_active = conv.is_active

    updates: dict = {}
    if body.name is not None:
        new_name = sanitize_text(body.name).strip()
        if not new_name:
            # create_group refuses a blank name; this path applied one. A name that
            # sanitizes away left a nameless row in every member's sidebar — and
            # broadcast a rename system message announcing it.
            raise HTTPException(status_code=400, detail="Group name required")
        updates["name"] = new_name
    if body.description is not None:
        updates["description"] = sanitize_text(body.description)
    if body.avatar_url is not None:
        updates["avatar_url"] = body.avatar_url
    if body.purpose_tag is not None:
        updates["purpose_tag"] = sanitize_text(body.purpose_tag)
    if body.is_active is not None:
        updates["is_active"] = body.is_active

    name_changed = "name" in updates and updates["name"] != conv.name
    for key, value in updates.items():
        setattr(conv, key, value)
    await db.commit()

    if name_changed:
        await send_system_message(db, conv.id, f"Group name changed to '{updates['name']}'", broadcast=False)

    member_ids = await participant_ids(db, conv.id)
    await publish_to_users(
        member_ids,
        {"type": "conversation_updated", "conversation_id": str(conv.id), "updates": updates},
    )
    if "is_active" in updates:
        if was_active and not conv.is_active:
            await publish_to_users(
                member_ids,
                {"type": "removed_from_conversation", "conversation_id": str(conv.id)},
            )
        elif not was_active and conv.is_active:
            await notify_conversation_created(db, conv.id)

    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="cross_org_group_updated",
        target=str(conv.id),
        details={"updates": {k: str(v) for k, v in updates.items()}},
    )
    await db.commit()
    return await _enrich_group(db, conv)


class AddMembersRequest(BaseModel):
    members: list[dict]


@router.post("/{group_id}/members")
async def add_members(
    group_id: str,
    body: AddMembersRequest,
    actor: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    conv = await _load_group(db, group_id, active_only=True)
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
    existing_ids = set(await participant_ids(db, conv.id))

    specs: list[tuple[uuid.UUID, str]] = []
    seen: set[uuid.UUID] = set()
    for m in body.members:
        uid = parse_uuid(m.get("user_id"))
        if uid is None or uid in seen or uid in existing_ids:
            continue  # invalid or already-present members are silently skipped
        seen.add(uid)
        specs.append((uid, "admin" if m.get("role", "member") == "admin" else "member"))

    users_map: dict[uuid.UUID, User] = {}
    if seen:
        user_rows = (
            (await db.execute(select(User).where(User.id.in_(seen), User.is_active.is_(True))))
            .scalars()
            .all()
        )
        users_map = {u.id: u for u in user_rows}

    allowed = [str(o) for o in (conv.allowed_org_ids or [])]
    # Resolved BEFORE anything is inserted, so the cap below is checked against
    # what would actually be added rather than against what was asked for —
    # unknown, inactive and already-present ids are all dropped above.
    resolved: list[tuple[User, str]] = []
    for uid, role in specs:
        member = users_map.get(uid)
        if member is None or member.org_id is None:
            continue
        resolved.append((member, role))

    # The same ceiling as creation and as a same-org group. Counted against the
    # CURRENT membership, so a group already at or over the cap cannot grow —
    # without breaking one that predates this check, which can still be read,
    # messaged and pruned, just not extended.
    if resolved and len(existing_ids) + len(resolved) > MAX_GROUP_MEMBERS:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_GROUP_MEMBERS} members")

    added: list[User] = []
    for member, role in resolved:
        # A new member's org is auto-added to the group's org scope (contract).
        if str(member.org_id) not in allowed:
            allowed = [*allowed, str(member.org_id)]
        db.add(
            ConversationParticipant(conversation_id=conv.id, user_id=member.id, role=_participant_role(role))
        )
        added.append(member)

    if not added:
        return await _enrich_group(db, conv)

    conv.allowed_org_ids = allowed
    conv.last_message_at = now_utc()
    await db.commit()

    await send_system_messages(
        db,
        conv.id,
        [f"Admin added {member.display_name}" for member in added],
        broadcast=False,
    )

    enriched = await _enrich_group(db, conv)
    await notify_conversation_created(db, conv.id, only=[member.id for member in added])
    await publish_to_users(
        list(existing_ids),
        {"type": "member_added", "conversation_id": str(conv.id), "conversation": enriched},
    )

    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="cross_org_member_added",
        target=str(conv.id),
        details={"added": [str(member.id) for member in added]},
    )
    await db.commit()
    return enriched


@router.delete("/{group_id}/members/{user_id}")
async def remove_member(
    group_id: str,
    user_id: str,
    actor: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    uid = parse_uuid(user_id)
    if uid is None:
        raise HTTPException(status_code=400, detail="Invalid ID")
    conv = await _load_group(db, group_id, invalid_detail="Invalid ID")

    target = await db.get(User, uid)
    display_name = target.display_name if target else "Unknown"

    membership = await db.get(ConversationParticipant, (conv.id, uid))
    if membership is not None:
        await db.delete(membership)
        conv.last_message_at = now_utc()
        await db.commit()

        await send_system_message(db, conv.id, f"Admin removed {display_name}", broadcast=False)
        await publish_to_users([uid], {"type": "removed_from_conversation", "conversation_id": str(conv.id)})
        remaining = await participant_ids(db, conv.id)
        await publish_to_users(
            remaining,
            {
                "type": "member_removed",
                "conversation_id": str(conv.id),
                "user_id": str(uid),
                "removed_by": str(actor.id),
            },
        )
        await log_audit(
            db,
            actor_id=actor.id,
            actor_type="superadmin",
            action="cross_org_member_removed",
            target=str(conv.id),
            details={"removed": str(uid)},
        )
        await db.commit()
    return {"message": f"{display_name} removed"}


@router.post("/{group_id}/archive")
async def toggle_archive(
    group_id: str,
    actor: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    conv = await _load_group(db, group_id)
    conv.is_active = not conv.is_active
    await db.commit()
    member_ids = await participant_ids(db, conv.id)
    if conv.is_active:
        await notify_conversation_created(db, conv.id)
    else:
        await publish_to_users(
            member_ids, {"type": "removed_from_conversation", "conversation_id": str(conv.id)}
        )
    # Archiving drops the group out of every member's list — the same state change
    # PUT /{group_id} makes and audits, and the same one delete_group audits. Only
    # this route left no record of who did it.
    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="cross_org_group_unarchived" if conv.is_active else "cross_org_group_archived",
        target=str(conv.id),
        details={"name": conv.name},
    )
    await db.commit()
    return {"is_active": conv.is_active}


@router.delete("/{group_id}")
async def delete_group(
    group_id: str,
    actor: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db),
):
    conv = await _load_group(db, group_id)
    # Both columns: deleted_at is the durable state that separates this from an
    # archive, and is_active stays False so the member-facing read paths, which
    # only know about is_active, keep the group out of every list as before.
    conv.deleted_at = now_utc()
    conv.is_active = False
    await db.commit()
    member_ids = await participant_ids(db, conv.id)
    await publish_to_users(member_ids, {"type": "removed_from_conversation", "conversation_id": str(conv.id)})
    await log_audit(
        db,
        actor_id=actor.id,
        actor_type="superadmin",
        action="cross_org_group_deleted",
        target=str(conv.id),
        details={"name": conv.name},
    )
    await db.commit()
    return {"message": "Group deleted"}
