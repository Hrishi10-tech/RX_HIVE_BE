"""Contacts roster — GET /api/users/contacts.

Wire contract preserved from RxHivexx (docs/reference/api-uploads-calls-misc.md):
bare array response with the `department_name` key (the /api/search contacts
bucket calls the same concept `department`). Fixes vs the Mongo build: the
search input is escaped + parameterized (was raw $regex — injection/ReDoS),
and departments are batch-joined instead of one lookup per user.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import selectinload

from app.core.deps import TenantContext, get_tenant
from app.db.models import (
    Conversation,
    ConversationParticipant,
    ConversationType,
    Department,
    User,
)
from app.services import enrich, presence
from app.utils import iso_z, parse_uuid

router = APIRouter(prefix="/api/users", tags=["contacts"])

# The contact panel renders this list in full — the wire contract has no cursor —
# so the query must cap itself: a user who shares hundreds of groups with a
# colleague would otherwise serialize every one of them on each profile open.
_GROUPS_IN_COMMON_LIMIT = 50


def _like_pattern(value: str) -> str:
    """Escape LIKE wildcards so user input is matched literally."""
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


@router.get("/contacts")
async def list_contacts(
    search: str = "",
    limit: int = Query(200, ge=1, le=500),
    tenant: TenantContext = Depends(get_tenant),
) -> list[dict]:
    """Active users in the caller's org except self, display_name asc, CAPPED.

    Callers without an org (superadmins) have no roster: returns [].

    The cap is the point. This used to return the entire org roster on every
    contact-picker open: measured against a seeded 25,000-user tenant, one call
    was 24,999 rows and 5.5 MB, took ~430ms, and spent ~130ms of that inside a
    single Redis pipeline issuing 24,999 EXISTS — all on the event loop, so it
    stalled every other request the worker was serving. Nothing consumed more
    than a screenful of it.

    Every picker in the app already narrows with `search`, which is what makes a
    cap safe: the roster is reachable, it is just no longer shipped whole. The
    ceiling is 500 rather than unbounded so a client cannot opt back into the
    old behaviour by passing a large number.
    """
    user = tenant.user
    if user.org_id is None:
        return []

    stmt = (
        select(User, Department.name)
        .outerjoin(Department, Department.id == User.dept_id)
        .where(
            User.org_id == user.org_id,
            User.is_active.is_(True),
            User.id != user.id,
        )
        .order_by(User.display_name.asc())
    )
    if search:
        pattern = _like_pattern(search)
        stmt = stmt.where(
            or_(
                User.display_name.ilike(pattern, escape="\\"),
                User.email.ilike(pattern, escape="\\"),
            )
        )

    rows = (await tenant.db.execute(stmt.limit(limit))).all()

    statuses = await presence.get_statuses([u.id for u, _ in rows])

    return [_contact_row(u, dept_name, statuses) for u, dept_name in rows]


def _contact_row(u: User, dept_name: str | None, statuses: dict) -> dict:
    """One directory row. Shared so the list and the single lookup below cannot
    drift apart — the client stores both in the same contacts map."""
    return {
        "id": str(u.id),
        "display_name": u.display_name,
        "email": u.email,
        "avatar_url": u.avatar_url,
        "department_name": dept_name or "Unknown",
        "status": statuses.get(str(u.id), "offline"),
        "last_seen": iso_z(u.last_seen_at),
    }


@router.get("/directory/{user_id}")
async def get_contact(
    user_id: uuid.UUID,
    tenant: TenantContext = Depends(get_tenant),
) -> dict:
    """One directory record, by id.

    Exists because the contact panel wanted exactly this and had no way to ask
    for it: it fetched the ENTIRE org roster and did `rows.find(c => c.id === …)`
    in the browser. That was tolerable only while /contacts was unbounded, and
    capping it would have silently broken the panel for anyone outside the first
    page — the lookup would simply have found nothing and rendered blank.

    Scoped to the caller's own org, so this is not a cross-tenant directory read:
    an id from another org 404s exactly as an unknown id does, and the two are
    deliberately indistinguishable to the caller.
    """
    user = tenant.user
    if user.org_id is None:
        raise HTTPException(status_code=404, detail="User not found")

    row = (
        await tenant.db.execute(
            select(User, Department.name)
            .outerjoin(Department, Department.id == User.dept_id)
            .where(
                User.id == user_id,
                User.org_id == user.org_id,
                User.is_active.is_(True),
            )
        )
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")

    u, dept_name = row
    statuses = await presence.get_statuses([u.id])
    return _contact_row(u, dept_name, statuses)


@router.get("/{user_id}/groups-in-common")
async def groups_in_common(user_id: str, tenant: TenantContext = Depends(get_tenant)):
    """Group conversations BOTH the caller and the target user belong to.

    The target is resolved through the tenant guard, so a foreign-org user id
    is a 404 — the caller can't probe another org's roster or memberships.
    """
    target_id = parse_uuid(user_id)
    if target_id is None:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    target = await tenant.org_user(target_id)
    if target.id == tenant.user.id:
        return {"data": []}

    mine = ConversationParticipant.__table__.alias("mine")
    theirs = ConversationParticipant.__table__.alias("theirs")
    stmt = (
        select(Conversation)
        .join(mine, and_(mine.c.conversation_id == Conversation.id, mine.c.user_id == tenant.user.id))
        .join(theirs, and_(theirs.c.conversation_id == Conversation.id, theirs.c.user_id == target.id))
        .options(selectinload(Conversation.participants).selectinload(ConversationParticipant.user))
        .where(
            Conversation.is_active.is_(True),
            Conversation.type.in_((ConversationType.group, ConversationType.cross_org)),
            # The same tenancy predicate list_conversations applies. Redundant
            # today rather than load-bearing, and kept deliberately: the `mine`
            # join already restricts this to conversations the CALLER belongs to,
            # and both write paths make membership imply org scope — groups.py
            # refuses a member whose org differs from the group's (on create and
            # on add), and a cross-org group auto-appends a new member's org to
            # allowed_org_ids and never removes one. So there is no reachable id
            # this excludes. It is here because that invariant lives in two other
            # files, and a reader of this query cannot see it from here.
            or_(
                Conversation.org_id == tenant.user.org_id,
                and_(
                    Conversation.type == ConversationType.cross_org,
                    Conversation.allowed_org_ids.contains([str(tenant.user.org_id)]),
                ),
            ),
        )
        .order_by(Conversation.last_message_at.desc().nulls_last())
        .limit(_GROUPS_IN_COMMON_LIMIT)
    )
    convs = (await tenant.db.execute(stmt)).scalars().all()

    # Batched like the conversation LIST endpoint: serializing one at a time made
    # this three queries PER group (unread, last message, presence) issued
    # sequentially, so the panel's latency grew linearly with shared groups.
    db = tenant.db
    conv_ids = [c.id for c in convs]
    unread = await enrich.unread_counts(db, conv_ids, tenant.user.id)
    last_msgs = await enrich.last_messages(db, conv_ids, tenant.user.id)
    statuses = await presence.get_statuses(list({p.user_id for c in convs for p in c.participants}))

    data = [
        await enrich.serialize_conversation(
            db, c, tenant.user.id, statuses=statuses, unread=unread, last_msgs=last_msgs
        )
        for c in convs
    ]
    return {"data": data}
