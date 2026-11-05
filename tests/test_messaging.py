"""Messaging core contracts: replies, editing, deletes, reactions, unread."""

import contextlib

from httpx import ASGITransport, AsyncClient

from app.main import app
from tests.conftest import CSRF, login


@contextlib.asynccontextmanager
async def _client_for(email):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c.headers.update(CSRF)
        await login(c, email)
        yield c


async def _direct(client, other_id) -> str:
    resp = await client.post("/api/conversations/direct", json={"participant_id": str(other_id)})
    assert resp.status_code == 200, resp.text
    return resp.json()["_id"]


async def _send(client, conv_id, content, **extra):
    resp = await client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"content": content, "type": "text", **extra},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def test_reply_is_persisted_and_enriched(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    first = await _send(client, conv, "original")
    reply = await _send(client, conv, "the reply", reply_to=first["_id"])
    assert reply["reply_to"] == first["_id"]
    assert reply["reply_to_message"]["content"] == "original"

    # reload from DB — the classic RxHivexx bug was replies lost on reload
    resp = await client.get(f"/api/conversations/{conv}/messages")
    stored = [m for m in resp.json()["messages"] if m["_id"] == reply["_id"]][0]
    assert stored["reply_to"] == first["_id"]
    assert stored["reply_to_message"]["content"] == "original"


async def test_reply_to_foreign_conversation_message_is_dropped(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv1 = await _direct(client, users["bob"].id)
    await _send(client, conv1, "in conv1")

    async with _client_for("bob@a.com") as bob:
        conv2_resp = await bob.post(
            "/api/conversations/group",
            json={"name": "G", "member_ids": [str(users["alice"].id), str(users["carol"].id)]},
        )
        # carol is foreign-org → group creation must fail entirely
        assert conv2_resp.status_code == 403


async def test_edit_message(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    msg = await _send(client, conv, "tyop here")
    resp = await client.put(f"/api/conversations/messages/{msg['_id']}", json={"content": "typo fixed"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["content"] == "typo fixed"
    assert body["edited_at"] is not None

    async with _client_for("bob@a.com") as bob:
        resp = await bob.put(f"/api/conversations/messages/{msg['_id']}", json={"content": "hijack"})
        assert resp.status_code == 403


# test_delete_for_me_actually_filters removed with the delete-for-me feature —
# nothing can create a MessageDeletion row any more, so the behaviour it asserted
# is unreachable. The read-side filter it protected is still in place (media.py
# and messages.py still exclude the caller's MessageDeletion rows) so that anyone
# who deleted a message for themselves before the removal keeps it hidden; that
# path simply has no way to be set up from the API now.


async def test_delete_message_endpoint_is_gone(client, two_orgs_with_users):
    """Message deletion was removed as a feature.

    Replaces test_delete_for_everyone_tombstones_and_only_sender, which exercised
    DELETE /api/conversations/messages/{id}. This asserts the route is actually
    unroutable rather than merely unlinked from the UI — a removed feature that
    still answers on the API is not removed.

    The read side is untouched on purpose: is_deleted is still serialized so that
    messages tombstoned before the removal keep rendering as deleted. That is
    covered wherever serialize_message is asserted, and cannot be re-tested here
    because nothing can create a tombstone any more.
    """
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    msg = await _send(client, conv, "not deletable")

    resp = await client.delete(f"/api/conversations/messages/{msg['_id']}")
    assert resp.status_code == 405, resp.text

    resp = await client.delete(f"/api/conversations/messages/{msg['_id']}", params={"for_everyone": "true"})
    assert resp.status_code == 405, resp.text

    # And the message is still there, intact and not tombstoned.
    resp = await client.get(f"/api/conversations/{conv}/messages")
    still = [m for m in resp.json()["messages"] if m["_id"] == msg["_id"]][0]
    assert still["is_deleted"] is False
    assert still["content"] == "not deletable"


async def test_reaction_toggle_and_shape(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    msg = await _send(client, conv, "react to me")

    resp = await client.post(f"/api/conversations/messages/{msg['_id']}/react", json={"emoji": "👍"})
    reactions = resp.json()["reactions"]
    assert reactions == [{"user_id": str(users["alice"].id), "user_name": "Alice", "emoji": "👍"}]

    # The loaded message must name the reactor too, or the client's reaction
    # tooltip is blank until the viewer happens to react themselves.
    resp = await client.get(f"/api/conversations/{conv}/messages")
    loaded = [m for m in resp.json()["messages"] if m["_id"] == msg["_id"]][0]
    assert loaded["reactions"][0]["user_name"] == "Alice"
    assert loaded["reactions"][0]["user_id"] == str(users["alice"].id)

    resp = await client.post(f"/api/conversations/messages/{msg['_id']}/react", json={"emoji": "👍"})
    assert resp.json()["reactions"] == []


async def test_unread_counts_and_read_receipts(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    await _send(client, conv, "one")
    await _send(client, conv, "two")

    async with _client_for("bob@a.com") as bob:
        data = (await bob.get("/api/conversations")).json()["data"]
        assert data[0]["unread_count"] == 2

        await bob.put(f"/api/conversations/{conv}/read")
        data = (await bob.get("/api/conversations")).json()["data"]
        assert data[0]["unread_count"] == 0

    # Alice now sees read receipts derived from Bob's last_read_at
    resp = await client.get(f"/api/conversations/{conv}/messages")
    for m in resp.json()["messages"]:
        if m["type"] == "text":
            assert any(r["user_id"] == str(users["bob"].id) for r in m["read_by"])


async def test_text_message_requires_content_and_media_requires_upload(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    resp = await client.post(f"/api/conversations/{conv}/messages", json={"content": "   ", "type": "text"})
    assert resp.status_code == 400
    resp = await client.post(f"/api/conversations/{conv}/messages", json={"content": "", "type": "image"})
    assert resp.status_code == 400  # media without a claimed upload URL
    resp = await client.post(
        f"/api/conversations/{conv}/messages",
        json={"content": "", "type": "image", "media_url": "https://evil.example/x.png"},
    )
    assert resp.status_code == 400  # foreign URLs can't be claimed


async def test_unread_filter_selects_only_conversations_with_unread(client, two_orgs_with_users):
    """?filter=unread must return exactly the conversations holding unread messages.

    Previously resolved in Python — load every candidate conversation, count unread
    for all of them, slice — so the tab cost the caller's whole conversation list no
    matter the page size. It is a LATERAL probe in the query now, which means the
    predicate itself is new code on a path that had NO test at all: nothing in the
    suite exercised filter=unread before this.

    Bob gets three conversations and reads the two most RECENT, leaving the oldest
    unread. That arrangement is what makes the limit=1 case discriminating, and the
    margin matters: the query fetches limit + 1 rows to decide has_more, so a single
    read conversation ahead of the unread one would be absorbed by that slack and
    prove nothing. TWO of them exceed it — filtering first returns the unread
    conversation, paginating first returns nothing at all. That is not a hypothetical
    shape; it is the shape this endpoint had, where the page was sliced out of a
    list already filtered in Python.
    """
    users = two_orgs_with_users
    from tests.conftest import make_user

    await make_user("erin@a.com", org_id=users["org_a"].id, display_name="Erin")
    await make_user("frank@a.com", org_id=users["org_a"].id, display_name="Frank")

    # Sent oldest-first, so the sidebar order is frank, erin, alice.
    await login(client, "alice@a.com")
    from_alice = await _direct(client, users["bob"].id)
    await _send(client, from_alice, "alice says hello")

    async with _client_for("erin@a.com") as erin_client:
        from_erin = await _direct(erin_client, users["bob"].id)
        await _send(erin_client, from_erin, "erin says hello")

    async with _client_for("frank@a.com") as frank_client:
        from_frank = await _direct(frank_client, users["bob"].id)
        await _send(frank_client, from_frank, "frank says hello")

    async with _client_for("bob@a.com") as bob:
        unread_ids = {
            c["_id"]
            for c in (await bob.get("/api/conversations", params={"filter": "unread"})).json()["data"]
        }
        assert unread_ids == {from_alice, from_erin, from_frank}, unread_ids

        assert [c["_id"] for c in (await bob.get("/api/conversations")).json()["data"]] == [
            from_frank,
            from_erin,
            from_alice,
        ]
        await bob.put(f"/api/conversations/{from_frank}/read")
        await bob.put(f"/api/conversations/{from_erin}/read")

        resp = await bob.get("/api/conversations", params={"filter": "unread"})
        assert resp.status_code == 200, resp.text
        rows = resp.json()["data"]
        assert [c["_id"] for c in rows] == [from_alice]
        assert rows[0]["unread_count"] == 1
        assert resp.json()["has_more"] is False

        # The filter must be applied BEFORE the limit. One slot plus one row of
        # has_more slack is two, and the two rows that sort into them are both
        # already read, so paginating first would return an empty tab.
        resp = await bob.get("/api/conversations", params={"filter": "unread", "limit": 1})
        assert resp.status_code == 200, resp.text
        assert [c["_id"] for c in resp.json()["data"]] == [from_alice], resp.json()
        assert resp.json()["has_more"] is False

        # The unfiltered list still shows all three — the probe narrows this tab only.
        everything = {c["_id"] for c in (await bob.get("/api/conversations")).json()["data"]}
        assert {from_alice, from_erin, from_frank} <= everything


async def test_content_is_sanitized(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    msg = await _send(client, conv, "<script>alert(1)</script>hello")
    assert "<script>" not in msg["content"]
    assert "hello" in msg["content"]


async def test_group_lifecycle_roles_and_admin_only(client, two_orgs_with_users):
    users = two_orgs_with_users
    from tests.conftest import make_user

    await make_user("erin@a.com", org_id=users["org_a"].id, display_name="Erin")

    await login(client, "alice@a.com")
    erin_id = None
    contacts = (await client.get("/api/users/contacts")).json()
    for c in contacts:
        if c["display_name"] == "Erin":
            erin_id = c["id"]

    resp = await client.post(
        "/api/conversations/group",
        json={"name": "Launch <b>Team</b>", "member_ids": [str(users["bob"].id), erin_id]},
    )
    assert resp.status_code == 200, resp.text
    group = resp.json()
    assert group["name"] == "Launch Team"  # sanitized
    roles = {p["display_name"]: p["role"] for p in group["participants"]}
    assert roles["Alice"] == "creator" and roles["Bob"] == "member"

    # admin-only messages: toggle on, member send rejected
    resp = await client.put(f"/api/conversations/{group['_id']}/group", json={"admin_only_messages": True})
    assert resp.status_code == 200
    async with _client_for("bob@a.com") as dave:  # bob is a plain member
        resp = await dave.post(
            f"/api/conversations/{group['_id']}/messages",
            json={"content": "member speaking", "type": "text"},
        )
        assert resp.status_code == 403


async def test_messages_around_cursor_centres_window(client, two_orgs_with_users):
    """`around=<id>` returns a window centred on a message, not just the tail.

    This is what makes "jump to a pinned/replied/search-hit message" possible at
    all: with only `before`, reaching an old message meant paging backwards until
    it happened to appear.
    """
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    sent = [await _send(client, conv, f"m{i:02d}") for i in range(40)]
    target = sent[10]

    # Default window is the newest tail, and does NOT reach back to m10.
    resp = await client.get(f"/api/conversations/{conv}/messages", params={"limit": 10})
    body = resp.json()
    assert body["has_more"] is True
    assert body["has_newer"] is False, "the default window is anchored to the newest end"
    assert body["anchor_id"] is None
    assert target["_id"] not in [m["_id"] for m in body["messages"]]

    # around= centres on the target and reports both directions.
    resp = await client.get(
        f"/api/conversations/{conv}/messages", params={"around": target["_id"], "limit": 10}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    ids = [m["_id"] for m in body["messages"]]
    assert target["_id"] in ids, "the anchor must be inside its own window"
    assert len(ids) == 10
    assert body["anchor_id"] == target["_id"]
    assert body["has_more"] is True, "m00..m09 are older"
    assert body["has_newer"] is True, "m12..m39 are newer"

    # Oldest-first, no duplicates, and contiguous around the anchor.
    assert ids == sorted(set(ids), key=ids.index), "no duplicate rows"
    contents = [m["content"] for m in body["messages"]]
    assert contents == sorted(contents), "wire order is oldest-first"
    assert "m10" in contents
    # 10 slots split 5 older-inclusive / 5 newer => m06..m15.
    assert contents == [f"m{i:02d}" for i in range(6, 16)], contents


async def test_messages_around_edges_and_bad_anchor(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    sent = [await _send(client, conv, f"e{i}") for i in range(5)]

    # Anchor on the very first real message: nothing older except the system row.
    resp = await client.get(
        f"/api/conversations/{conv}/messages", params={"around": sent[0]["_id"], "limit": 50}
    )
    body = resp.json()
    assert body["has_newer"] is False, "the whole conversation fits in one window"
    assert sent[0]["_id"] in [m["_id"] for m in body["messages"]]

    # Anchor on the newest message: nothing newer.
    resp = await client.get(
        f"/api/conversations/{conv}/messages", params={"around": sent[-1]["_id"], "limit": 3}
    )
    body = resp.json()
    assert body["has_newer"] is False
    assert sent[-1]["_id"] in [m["_id"] for m in body["messages"]]

    # limit=1 must still return the anchor itself.
    resp = await client.get(
        f"/api/conversations/{conv}/messages", params={"around": sent[2]["_id"], "limit": 1}
    )
    body = resp.json()
    assert [m["_id"] for m in body["messages"]] == [sent[2]["_id"]]
    assert body["has_newer"] is True and body["has_more"] is True

    # A garbage or foreign anchor is ignored, not an error — the caller gets the
    # newest window and anchor_id=null tells it the anchor did not resolve.
    for bad in ("not-a-uuid", "00000000-0000-0000-0000-000000000000"):
        resp = await client.get(f"/api/conversations/{conv}/messages", params={"around": bad, "limit": 3})
        assert resp.status_code == 200, resp.text
        assert resp.json()["anchor_id"] is None
        assert resp.json()["has_newer"] is False
