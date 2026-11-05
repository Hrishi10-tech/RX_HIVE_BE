"""Inviting into a group call that runs on a cross-org conversation.

initiate_group_call exempts cross_org conversations from the org check and builds
the call roster from every conversation member, so such a call legitimately spans
organizations. invite_to_call has to hold the same line, and conversation
membership — not org equality — is what keeps it from becoming a way to ring
someone who was never in the conversation.
"""

import contextlib
import uuid

from httpx import ASGITransport, AsyncClient

from app.db.models import UserRole
from app.main import app
from app.services import calls as calls_service
from tests.conftest import CSRF, login, make_user


@contextlib.asynccontextmanager
async def _superadmin():
    await make_user("root@x.com", role=UserRole.superadmin)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c.headers.update(CSRF)
        await login(c, "root@x.com")
        yield c


async def _cross_org_group(users) -> uuid.UUID:
    async with _superadmin() as root:
        resp = await root.post(
            "/api/admin/cross-org-groups",
            json={
                "name": "Joint Clinic",
                "org_ids": [str(users["org_a"].id), str(users["org_b"].id)],
                "members": [
                    {"user_id": str(users["alice"].id), "role": "admin"},
                    {"user_id": str(users["carol"].id), "role": "member"},
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        return uuid.UUID(resp.json()["_id"])


async def test_cross_org_group_call_can_invite_the_other_org(client, two_orgs_with_users):
    """Carol is in the conversation and on the call roster from second zero, but she
    is in the other organization. The flat org comparison reported "different_org"
    for a participant of the very call being invited into, so the one route back
    into a cross-org call for someone who had not answered was closed."""
    users = two_orgs_with_users
    conv_id = await _cross_org_group(users)

    await calls_service.initiate_group_call(users["alice"], conv_id, "voice")
    call_id = await calls_service.current_call_of(users["alice"].id)
    assert call_id, "the group call was never registered for the caller"

    result = await calls_service.invite_to_call(users["alice"], uuid.UUID(call_id), [users["carol"].id])
    assert "error" not in result, result
    assert result["outcome"][str(users["carol"].id)] == "invited"


async def test_cross_org_invite_still_refuses_a_non_member(client, two_orgs_with_users):
    """The org check is replaced by a membership check, not removed. Someone from
    the other organization who was never in the conversation must stay unreachable
    — otherwise the exemption would turn a call id into a way to ring anyone."""
    users = two_orgs_with_users
    outsider = await make_user("dave@b.com", org_id=users["org_b"].id, display_name="Dave")
    conv_id = await _cross_org_group(users)

    await calls_service.initiate_group_call(users["alice"], conv_id, "voice")
    call_id = await calls_service.current_call_of(users["alice"].id)

    result = await calls_service.invite_to_call(users["alice"], uuid.UUID(call_id), [outsider.id])
    assert result["outcome"][str(outsider.id)] == "different_org"


async def test_single_org_group_call_still_refuses_another_org(client, two_orgs_with_users):
    """An ordinary (non-cross-org) group call keeps the plain org comparison."""
    users = two_orgs_with_users
    erin = await make_user("erin@a.com", org_id=users["org_a"].id, display_name="Erin")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as alice:
        alice.headers.update(CSRF)
        await login(alice, "alice@a.com")
        resp = await alice.post(
            "/api/conversations/group",
            json={"name": "Org A only", "member_ids": [str(users["bob"].id), str(erin.id)]},
        )
        assert resp.status_code == 200, resp.text
        conv_id = uuid.UUID(resp.json()["_id"])

    await calls_service.initiate_group_call(users["alice"], conv_id, "voice")
    call_id = await calls_service.current_call_of(users["alice"].id)

    result = await calls_service.invite_to_call(users["alice"], uuid.UUID(call_id), [users["carol"].id])
    assert result["outcome"][str(users["carol"].id)] == "different_org"
