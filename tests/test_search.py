"""Search: parameterized FTS/ILIKE, org-scoped, injection-safe."""

from tests.conftest import login


async def _seed(client, users):
    await login(client, "alice@a.com")
    conv = (
        await client.post("/api/conversations/direct", json={"participant_id": str(users["bob"].id)})
    ).json()["_id"]
    await client.post(
        f"/api/conversations/{conv}/messages",
        json={"content": "quarterly financials review", "type": "text", "temp_id": "s1"},
    )
    return conv


async def test_search_is_org_scoped_and_finds_messages(client, two_orgs_with_users):
    users = two_orgs_with_users
    await _seed(client, users)

    # message search
    resp = await client.get("/api/search", params={"q": "quarterly"})
    assert any("quarterly" in m["content_snippet"] for m in resp.json()["messages"])

    # contact search (a term that matches the contact's name)
    resp = await client.get("/api/search", params={"q": "Bob"})
    assert any(c["display_name"] == "Bob" for c in resp.json()["contacts"])


async def test_search_special_characters_do_not_crash_or_injection(client, two_orgs_with_users):
    users = two_orgs_with_users
    await _seed(client, users)

    # ReDoS / LIKE-wildcard / SQL-ish payloads must be treated as literals
    for payload in ["(a+)+$", "100%", "under_score", "'; DROP TABLE messages;--", "%"]:
        resp = await client.get("/api/search", params={"q": payload})
        assert resp.status_code == 200
        # a literal "%" must NOT match everything
        if payload == "%":
            assert resp.json()["messages"] == []


async def test_conversation_message_search_escapes_wildcards(client, two_orgs_with_users):
    users = two_orgs_with_users
    conv = await _seed(client, users)
    resp = await client.post(f"/api/conversations/{conv}/messages/search", params={"q": "%"})
    assert resp.status_code == 200
    # "%" is a literal here — the "quarterly financials" message has no percent sign
    assert resp.json()["total"] == 0


async def _revoke(conv_id: str) -> None:
    """Cut the caller off exactly the way the product does.

    DELETE /api/admin/cross-org-groups/{id} and the archive route both just set
    is_active = False and leave the participant rows in place — that flag is the
    whole revocation, which is why every read path has to consult it.
    """
    import uuid as _uuid

    from app.db.models import Conversation
    from app.db.session import SessionLocal

    async with SessionLocal() as db:
        conv = await db.get(Conversation, _uuid.UUID(conv_id))
        conv.is_active = False
        await db.commit()


async def test_search_stops_at_a_revoked_conversation(client, two_orgs_with_users):
    """Global search kept serving the history of a group the caller was cut off from.

    _require_org_access gates the per-conversation reads and media.py gates the
    drawer, but /api/search sweeps across conversations and never consulted the
    flag. Its own conversations bucket did, so one half of the response hid the
    group while the other half returned its messages.
    """
    users = two_orgs_with_users
    conv = await _seed(client, users)

    # findable while the conversation is live
    resp = await client.get("/api/search", params={"q": "quarterly"})
    assert resp.json()["messages"], "precondition: the message is searchable"

    await _revoke(conv)

    resp = await client.get("/api/search", params={"q": "quarterly"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["messages"] == [], "revoked conversation still leaked message content"
    # the bucket that was already correct stays correct
    assert body["conversations"] == []


async def test_message_info_stops_at_a_revoked_conversation(client, two_orgs_with_users):
    """Same lever, smaller payload: no content, but every other participant's
    name and the times they read it."""
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = (
        await client.post("/api/conversations/direct", json={"participant_id": str(users["bob"].id)})
    ).json()["_id"]
    msg_id = (
        await client.post(
            f"/api/conversations/{conv}/messages",
            json={"content": "receipts please", "type": "text", "temp_id": "i1"},
        )
    ).json()["_id"]

    assert (await client.get(f"/api/conversations/messages/{msg_id}/info")).status_code == 200

    await _revoke(conv)

    resp = await client.get(f"/api/conversations/messages/{msg_id}/info")
    assert resp.status_code == 404, "read receipts survived the revocation"
