"""THE P1 exit gate: org A can provably never read org B.

Every historical IDOR from the security punch-list gets a regression test:
message react, forward, delete, info, conversation reads, contacts, search.
"""

from httpx import ASGITransport, AsyncClient

from app.main import app
from tests.conftest import CSRF, login


async def _fresh_client() -> AsyncClient:
    c = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    c.headers.update(CSRF)
    return c


async def _setup_conversation_with_message(client, users):
    """Alice (org A) messages Bob (org A); returns conv id + message id."""
    await login(client, "alice@a.com")
    resp = await client.post("/api/conversations/direct", json={"participant_id": str(users["bob"].id)})
    assert resp.status_code == 200, resp.text
    conv_id = resp.json()["_id"]
    resp = await client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"content": "org A secret", "type": "text", "temp_id": "t1"},
    )
    assert resp.status_code == 200, resp.text
    return conv_id, resp.json()["_id"]


async def test_cross_org_message_actions_all_404(client, two_orgs_with_users):
    conv_id, msg_id = await _setup_conversation_with_message(client, two_orgs_with_users)

    carol = await _fresh_client()
    async with carol:
        await login(carol, "carol@b.com")

        # read messages
        resp = await carol.get(f"/api/conversations/{conv_id}/messages")
        assert resp.status_code == 404
        # react (was CRITICAL IDOR)
        resp = await carol.post(f"/api/conversations/messages/{msg_id}/react", json={"emoji": "x"})
        assert resp.status_code == 404
        # forward (was exfiltration IDOR)
        resp = await carol.post(
            "/api/conversations/messages/forward",
            json={"message_id": msg_id, "conversation_ids": [], "contact_ids": []},
        )
        assert resp.status_code == 404
        # Message deletion was removed as a feature, so DELETE on this path is no
        # longer routable at all — Starlette answers 405 because the path exists
        # (PUT edits a message) but the method does not. Still asserted: a
        # non-member must not be able to reach a delete, and now nobody can.
        resp = await carol.delete(f"/api/conversations/messages/{msg_id}")
        assert resp.status_code == 405
        # message info
        resp = await carol.get(f"/api/conversations/messages/{msg_id}/info")
        assert resp.status_code == 404
        # membership actions
        resp = await carol.put(f"/api/conversations/{conv_id}/pin")
        assert resp.status_code == 404
        resp = await carol.put(f"/api/conversations/{conv_id}/read")
        assert resp.status_code == 404
        # sending into the conversation
        resp = await carol.post(
            f"/api/conversations/{conv_id}/messages",
            json={"content": "intrusion", "type": "text"},
        )
        assert resp.status_code in (403, 404)


async def test_conversation_list_and_search_are_org_scoped(client, two_orgs_with_users):
    conv_id, _ = await _setup_conversation_with_message(client, two_orgs_with_users)

    carol = await _fresh_client()
    async with carol:
        await login(carol, "carol@b.com")
        resp = await carol.get("/api/conversations")
        assert resp.json() == {"data": [], "has_more": False}

        resp = await carol.get("/api/search", params={"q": "secret"})
        body = resp.json()
        assert body["conversations"] == [] and body["messages"] == [] and body["contacts"] == []

        resp = await carol.get("/api/users/contacts")
        names = [c["display_name"] for c in resp.json()]
        assert "Alice" not in names and "Bob" not in names


async def test_directory_lookup_is_scoped_to_the_callers_org(client, two_orgs_with_users):
    """GET /api/users/directory/{id} must not become a cross-tenant read.

    It exists so the contact panel can ask for one record instead of pulling the
    whole roster — which is exactly the shape that invites an IDOR, because the
    id now comes from the URL rather than from a list the server chose.

    Carol (org B) asking for Alice (org A) must get the same answer she gets for
    an id that does not exist at all: the two are deliberately indistinguishable,
    so the endpoint cannot be used to probe which ids are real.
    """
    users = two_orgs_with_users

    await login(client, "alice@a.com")
    mine = await client.get(f"/api/users/directory/{users['bob'].id}")
    assert mine.status_code == 200, mine.text
    assert mine.json()["display_name"] == "Bob"

    carol = await _fresh_client()
    async with carol:
        await login(carol, "carol@b.com")

        cross = await carol.get(f"/api/users/directory/{users['alice'].id}")
        assert cross.status_code == 404, cross.text

        missing = await carol.get("/api/users/directory/00000000-0000-0000-0000-000000000000")
        assert missing.status_code == 404
        # Same status AND same body: a different message would leak existence.
        assert cross.json() == missing.json()


async def test_direct_conversation_across_orgs_rejected(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    resp = await client.post("/api/conversations/direct", json={"participant_id": str(users["carol"].id)})
    assert resp.status_code in (403, 404)


async def test_org_admin_portal_is_org_scoped(client, two_orgs_with_users):
    users = two_orgs_with_users
    from app.db.models import User, UserRole
    from app.db.session import SessionLocal

    async with SessionLocal() as db:
        alice = await db.get(User, users["alice"].id)
        alice.role = UserRole.org_admin
        await db.commit()

    await login(client, "alice@a.com")
    resp = await client.get("/api/org-admin/users")
    emails = [u["email"] for u in resp.json()["data"]]
    assert "carol@b.com" not in emails
    assert "bob@a.com" in emails

    # org admin cannot manage a foreign-org user
    resp = await client.put(f"/api/org-admin/users/{users['carol'].id}", json={"is_active": False})
    assert resp.status_code == 404

    resp = await client.post(f"/api/org-admin/users/{users['carol'].id}/reset-password")
    assert resp.status_code == 404


async def test_org_stats_counts_only_its_own_tenant_as_online(client, two_orgs_with_users):
    """active_today is org-scoped, and now reads the presence index.

    It used to load every active user id in the tenant and issue one Redis EXISTS
    per row. The index answers it directly, so this pins the thing that actually
    matters through the endpoint: a neighbouring tenant's traffic never lands on
    this org's tile.
    """
    from app.db.models import User, UserRole
    from app.db.session import SessionLocal
    from app.services import presence

    users = two_orgs_with_users
    async with SessionLocal() as db:
        alice = await db.get(User, users["alice"].id)
        alice.role = UserRole.org_admin
        await db.commit()
    await login(client, "alice@a.com")

    assert (await client.get("/api/org-admin/stats")).json()["active_today"] == 0

    # Bob connects in this org; Carol connects in the other one.
    await presence.mark_online(users["bob"].id, "c-bob", org_id=users["org_a"].id)
    await presence.mark_online(users["carol"].id, "c-carol", org_id=users["org_b"].id)

    body = (await client.get("/api/org-admin/stats")).json()
    assert body["active_today"] == 1, "the other tenant's session was counted here"
    assert body["total_users"] == 2  # alice + bob, unchanged by any of this

    await presence.mark_offline(users["bob"].id, "c-bob", org_id=users["org_a"].id)
    assert (await client.get("/api/org-admin/stats")).json()["active_today"] == 0


async def test_superadmin_rejected_from_org_admin_portal(client, two_orgs_with_users):
    from app.db.models import UserRole
    from tests.conftest import make_user

    await make_user("root@x.com", role=UserRole.superadmin)
    await login(client, "root@x.com")
    resp = await client.get("/api/org-admin/stats")
    assert resp.status_code == 403
