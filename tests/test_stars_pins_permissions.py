"""Stars, pins, mute, group permissions, groups-in-common, delete-for-me, links.

Every one of these endpoints is membership-scoped, so the last test in this
module re-runs the whole surface as a foreign-org caller and demands a 404.
"""

import contextlib
import uuid

from httpx import ASGITransport, AsyncClient

from app.db.models import Conversation, UserRole
from app.db.session import SessionLocal
from app.main import app
from tests.conftest import CSRF, login, make_user


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


async def _send(client, conv_id, content, **extra) -> dict:
    resp = await client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"content": content, "type": "text", **extra},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _group_with_bob_and_erin(client, users) -> str:
    erin = await make_user("erin@a.com", org_id=users["org_a"].id, display_name="Erin")
    resp = await client.post(
        "/api/conversations/group",
        json={"name": "Ops", "member_ids": [str(users["bob"].id), str(erin.id)]},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["_id"]


async def _messages(client, conv_id) -> list[dict]:
    resp = await client.get(f"/api/conversations/{conv_id}/messages")
    assert resp.status_code == 200, resp.text
    return resp.json()["messages"]


async def test_star_toggle_round_trip_and_starred_list(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    msg = await _send(client, conv, "worth keeping")

    resp = await client.post(f"/api/conversations/messages/{msg['_id']}/star")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"starred": True}

    stored = [m for m in await _messages(client, conv) if m["_id"] == msg["_id"]][0]
    assert stored["is_starred"] is True and stored["is_pinned"] is False

    resp = await client.get(f"/api/conversations/{conv}/starred")
    assert [m["_id"] for m in resp.json()["data"]] == [msg["_id"]]

    # Stars are private: Bob sees the same message unstarred and an empty list.
    async with _client_for("bob@a.com") as bob:
        theirs = [m for m in await _messages(bob, conv) if m["_id"] == msg["_id"]][0]
        assert theirs["is_starred"] is False
        assert (await bob.get(f"/api/conversations/{conv}/starred")).json() == {"data": []}

    resp = await client.post(f"/api/conversations/messages/{msg['_id']}/star")
    assert resp.json() == {"starred": False}
    assert (await client.get(f"/api/conversations/{conv}/starred")).json() == {"data": []}


async def test_pin_toggle_round_trip_is_conversation_wide(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    msg = await _send(client, conv, "pin this")

    resp = await client.post(f"/api/conversations/messages/{msg['_id']}/pin")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"pinned": True}

    resp = await client.get(f"/api/conversations/{conv}/pinned")
    assert [m["_id"] for m in resp.json()["data"]] == [msg["_id"]]

    # Pins are shared state — Bob sees the pin too.
    async with _client_for("bob@a.com") as bob:
        theirs = [m for m in await _messages(bob, conv) if m["_id"] == msg["_id"]][0]
        assert theirs["is_pinned"] is True
        assert [m["_id"] for m in (await bob.get(f"/api/conversations/{conv}/pinned")).json()["data"]] == [
            msg["_id"]
        ]

    resp = await client.post(f"/api/conversations/messages/{msg['_id']}/pin")
    assert resp.json() == {"pinned": False}
    assert (await client.get(f"/api/conversations/{conv}/pinned")).json() == {"data": []}


async def test_mute_toggle(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    assert (await client.put(f"/api/conversations/{conv}/mute")).json() == {"is_muted": True}
    assert (await client.put(f"/api/conversations/{conv}/mute")).json() == {"is_muted": False}


async def test_conversation_pin_order_and_isolation(client, two_orgs_with_users):
    """Conversation pinning: round-trip, ordering, per-user isolation, cap.

    There was no functional test for this at all — only a tenant-isolation 404.
    """
    users = two_orgs_with_users
    erin = await make_user("erin@a.com", org_id=users["org_a"].id, display_name="Erin")

    await login(client, "alice@a.com")
    conv_bob = await _direct(client, users["bob"].id)
    conv_erin = await _direct(client, erin.id)

    # Give conv_bob the more recent message, so recency alone would rank it first.
    await _send(client, conv_erin, "older")
    await _send(client, conv_bob, "newer")

    async def order():
        resp = await client.get("/api/conversations")
        assert resp.status_code == 200, resp.text
        return [c["_id"] for c in resp.json()["data"]]

    assert (await order())[0] == conv_bob, "recency should rank conv_bob first"

    # Pinning the OLDER chat must lift it above the newer one.
    resp = await client.put(f"/api/conversations/{conv_erin}/pin")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"is_pinned": True, "pin_order": 0}
    assert (await order())[0] == conv_erin

    # A second pin goes ABOVE the first (newest pin takes min-1).
    resp = await client.put(f"/api/conversations/{conv_bob}/pin")
    assert resp.json() == {"is_pinned": True, "pin_order": -1}
    assert (await order())[:2] == [conv_bob, conv_erin]

    # The chosen order must SURVIVE a new message in the lower-ranked pin —
    # this is what a boolean alone could not express.
    await _send(client, conv_erin, "bump me")
    assert (await order())[:2] == [conv_bob, conv_erin]

    # Unpin clears the ordinal so a stale position cannot resurrect later.
    resp = await client.put(f"/api/conversations/{conv_bob}/pin")
    assert resp.json() == {"is_pinned": False, "pin_order": None}
    assert (await order())[0] == conv_erin, "conv_erin is still pinned"

    # is_pinned/pin_order are MINE: Bob sees his own state, not Alice's.
    async with _client_for("bob@a.com") as bob:
        resp = await bob.get("/api/conversations")
        mine = [c for c in resp.json()["data"] if c["_id"] == conv_erin]
        # Bob is not a participant of Alice<->Erin, so it must not appear at all.
        assert mine == []
        bobs = [c for c in resp.json()["data"] if c["_id"] == conv_bob][0]
        assert bobs["is_pinned"] is False
        assert bobs["pin_order"] is None


async def test_conversation_pin_cap(client, two_orgs_with_users):
    from app.api.conversations import MAX_PINNED_CONVERSATIONS

    users = two_orgs_with_users
    await login(client, "alice@a.com")

    convs = []
    for i in range(MAX_PINNED_CONVERSATIONS):
        peer = await make_user(f"cap{i}@a.com", org_id=users["org_a"].id, display_name=f"Cap{i}")
        conv = await _direct(client, peer.id)
        convs.append(conv)
        resp = await client.put(f"/api/conversations/{conv}/pin")
        assert resp.status_code == 200, resp.text

    one_too_many = await _direct(
        client, (await make_user("capX@a.com", org_id=users["org_a"].id, display_name="CapX")).id
    )
    resp = await client.put(f"/api/conversations/{one_too_many}/pin")
    assert resp.status_code == 400, resp.text
    assert str(MAX_PINNED_CONVERSATIONS) in resp.json()["detail"]

    # Unpinning one frees a slot.
    assert (await client.put(f"/api/conversations/{convs[0]}/pin")).status_code == 200
    assert (await client.put(f"/api/conversations/{one_too_many}/pin")).status_code == 200


async def test_permissions_read_write_and_non_admin_403(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    group = await _group_with_bob_and_erin(client, users)

    resp = await client.get(f"/api/conversations/{group}/permissions")
    assert resp.status_code == 200, resp.text
    # ONE key on the wire: the only permission the server enforces. edit_info and
    # add_members joined send_history / invite_via_link / approve_new_members for
    # the same reason — nothing honoured them.
    assert resp.json() == {"send_messages": True}

    resp = await client.put(
        f"/api/conversations/{group}/permissions",
        json={"send_messages": False},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"send_messages": False}

    # send_messages=False is admin_only_messages=True: a plain member is refused.
    async with _client_for("bob@a.com") as bob:
        resp = await bob.post(f"/api/conversations/{group}/messages", json={"content": "hi", "type": "text"})
        assert resp.status_code == 403

        # Members may read permissions but never write them.
        assert (await bob.get(f"/api/conversations/{group}/permissions")).status_code == 200
        resp = await bob.put(f"/api/conversations/{group}/permissions", json={"send_messages": True})
        assert resp.status_code == 403

    resp = await client.get(f"/api/conversations/{group}/permissions")
    assert resp.json()["send_messages"] is False  # bob's 403 changed nothing


async def test_the_withdrawn_permission_keys_are_accepted_and_ignored(client, two_orgs_with_users):
    """A client that has not been updated still gets its 200, and still no policy.

    edit_info and add_members were stored and broadcast while nothing read
    perm_edit_info or perm_add_members, so the PUT was a success response for a
    no-op. Extra keys are ignored rather than rejected, which is what keeps an old
    web or iOS build working — but they must not come back on the wire, or the
    server is advertising a policy again.
    """
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    group = await _group_with_bob_and_erin(client, users)

    resp = await client.put(
        f"/api/conversations/{group}/permissions",
        json={"edit_info": False, "add_members": False, "send_messages": True},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"send_messages": True}

    assert (await client.get(f"/api/conversations/{group}/permissions")).json() == {"send_messages": True}

    # Read the COLUMNS, not the serializer. Now that serialize_permissions omits
    # both keys unconditionally, the two assertions above cannot tell "ignored"
    # from "written and no longer shown" — they would pass just as happily if the
    # PUT had set perm_edit_info=False. The stored row is the only thing that
    # separates them, and "accepted and ignored" is what this test claims.
    async with SessionLocal() as db:
        conv = await db.get(Conversation, uuid.UUID(group))
        assert conv is not None
        assert conv.perm_edit_info is True, "the withdrawn key was written after all"
        assert conv.perm_add_members is True, "the withdrawn key was written after all"


async def test_group_info_and_membership_stay_admin_only(client, two_orgs_with_users):
    """The behaviour the withdrawn toggles misdescribed, pinned.

    perm_edit_info and perm_add_members both default to TRUE, and the settings UI
    read that as "members may". The server never did: both routes gate on
    _require_group_admin. This test is what makes re-enforcing those columns a
    deliberate act rather than an accident — implementing them would loosen access
    on every existing group, and it would have to change this test to do it.
    """
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    group = await _group_with_bob_and_erin(client, users)

    # Same org, not yet a member: so a refusal can only be the admin gate, not the
    # tenant gate. carol would have 404'd on org scoping and proved nothing.
    frank = await make_user("frank@a.com", org_id=users["org_a"].id, display_name="Frank")

    async with _client_for("bob@a.com") as bob:
        renamed = await bob.put(f"/api/conversations/{group}/group", json={"name": "Bob's rename"})
        assert renamed.status_code == 403, renamed.text

        added = await bob.post(
            f"/api/conversations/{group}/members",
            json={"user_ids": [str(frank.id)]},
        )
        assert added.status_code == 403, added.text

    # The admin can do both, so the 403s above are about role and nothing else.
    assert (
        await client.put(f"/api/conversations/{group}/group", json={"name": "Ops renamed"})
    ).status_code == 200
    assert (
        await client.post(f"/api/conversations/{group}/members", json={"user_ids": [str(frank.id)]})
    ).status_code == 200


async def test_cross_org_group_admin_gets_404_from_permissions(client, two_orgs_with_users):
    """A cross-org participant with ParticipantRole.admin is not a group admin.

    These routes used to gate on membership alone, so this user could read and flip
    a superadmin-managed group's permissions — silencing another organization's
    participants, unaudited. Both verbs must 404 like every other /group route.
    """
    users = two_orgs_with_users
    await make_user("root@x.com", role=UserRole.superadmin)
    async with _client_for("root@x.com") as root:
        resp = await root.post(
            "/api/admin/cross-org-groups",
            json={
                "name": "Joint Programme",
                "org_ids": [str(users["org_a"].id), str(users["org_b"].id)],
                "members": [
                    {"user_id": str(users["alice"].id), "role": "admin"},
                    {"user_id": str(users["carol"].id), "role": "member"},
                ],
            },
        )
        assert resp.status_code == 200, resp.text
        group = resp.json()["_id"]

    await login(client, "alice@a.com")
    assert (await client.get(f"/api/conversations/{group}/permissions")).status_code == 404
    resp = await client.put(f"/api/conversations/{group}/permissions", json={"send_messages": False})
    assert resp.status_code == 404

    # Nothing was silenced: Carol's org can still post in the cross-org group.
    async with _client_for("carol@b.com") as carol:
        resp = await carol.post(
            f"/api/conversations/{group}/messages", json={"content": "hi", "type": "text"}
        )
        assert resp.status_code == 200, resp.text


async def test_groups_in_common(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    await _direct(client, users["bob"].id)  # direct chats are not "groups in common"
    group = await _group_with_bob_and_erin(client, users)

    resp = await client.get(f"/api/users/{users['bob'].id}/groups-in-common")
    assert resp.status_code == 200, resp.text
    assert [c["_id"] for c in resp.json()["data"]] == [group]

    # Bob sees the same shared group from his side.
    async with _client_for("bob@a.com") as bob:
        resp = await bob.get(f"/api/users/{users['alice'].id}/groups-in-common")
        assert [c["_id"] for c in resp.json()["data"]] == [group]


async def test_export_and_clear_endpoints_are_gone(client, two_orgs_with_users):
    """Export chat and Clear chat were removed as product features.

    Asserted explicitly rather than by deleting the old tests: both routes were
    also called by the shipped iOS client, so a silent reappearance is exactly
    the kind of thing that should fail here.
    """
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    assert (await client.get(f"/api/conversations/{conv}/export")).status_code == 404
    assert (await client.post(f"/api/conversations/{conv}/clear")).status_code == 404


async def test_delete_conversation_is_delete_for_me_only(client, two_orgs_with_users):
    """Delete-for-me still works and still touches only the caller's rows.

    Previously driven through POST /clear. That endpoint is gone, so DELETE
    /{conv_id} is now the ONLY thing that writes MessageDeletion rows — which is
    precisely why the model, the table and every read-side exclusion filter had
    to survive the removal.

    It also pins the sidebar preview, which is derived by a different query than
    the message list and was previously unasserted.
    """
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    await _send(client, conv, "keep me for bob")

    resp = await client.delete(f"/api/conversations/{conv}")
    assert resp.status_code == 200, resp.text
    assert await _messages(client, conv) == []

    # The sidebar preview comes from enrich.last_messages, which applies the same
    # exclusion — so Alice's row must LOSE its preview rather than keep the message
    # she just hid. This is the only assertion that pins the delete-for-me filter
    # inside that query, and it is the one a "newest per conversation" rewrite can
    # break while every message-list assertion above stays green.
    mine = [c for c in (await client.get("/api/conversations")).json()["data"] if c["_id"] == conv]
    assert mine and mine[0]["last_message"] is None

    async with _client_for("bob@a.com") as bob:
        contents = [m["content"] for m in await _messages(bob, conv)]
        assert "keep me for bob" in contents
        his = [c for c in (await bob.get("/api/conversations")).json()["data"] if c["_id"] == conv]
        assert his and his[0]["last_message"]["content"] == "keep me for bob"


async def test_starred_and_pinned_survive_a_reply_to_another_row_in_the_page(client, two_orgs_with_users):
    """A page whose reply target is ALSO in the page must not expire that row.

    serialize_message used to re-read each reply target with populate_existing and
    only sender/attachments in its options, which refreshed the target whenever it
    was a page member and expired the reactions collection MESSAGE_LOAD_OPTIONS had
    loaded for it. /messages escaped it — a reply target is always older and that
    endpoint reverses to oldest-first, so the clobbered row was already serialized.
    /starred and /pinned order newest-first and do not reverse, so they reached it
    afterwards, lazy-loaded under asyncio and raised MissingGreenlet: a 500.

    Starring or pinning a message together with a reply to it is the whole
    reproduction, and /pinned is fetched on every conversation open, where the
    client's catch turns the failure into a permanently blank banner rather than
    anything a user could report.
    """
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    original = await _send(client, conv, "the original")
    reply = await _send(client, conv, "a reply to it", reply_to=original["_id"])

    for msg_id in (original["_id"], reply["_id"]):
        assert (await client.post(f"/api/conversations/messages/{msg_id}/star")).status_code == 200
        assert (await client.post(f"/api/conversations/messages/{msg_id}/pin")).status_code == 200

    starred = await client.get(f"/api/conversations/{conv}/starred")
    assert starred.status_code == 200, starred.text
    starred_doc = next(d for d in starred.json()["data"] if d["_id"] == reply["_id"])
    assert starred_doc["reply_to_message"]["_id"] == original["_id"]
    assert starred_doc["reply_to_message"]["content"] == "the original"

    pinned = await client.get(f"/api/conversations/{conv}/pinned")
    assert pinned.status_code == 200, pinned.text
    assert {d["_id"] for d in pinned.json()["data"]} == {original["_id"], reply["_id"]}
    # Assert the preview on THIS endpoint too, not just on /starred. The two paths
    # order differently and reach reply_targets with different pages, so a status
    # code alone would let a null preview through here while /starred stayed right.
    pinned_doc = next(d for d in pinned.json()["data"] if d["_id"] == reply["_id"])
    assert pinned_doc["reply_to_message"]["_id"] == original["_id"]
    assert pinned_doc["reply_to_message"]["content"] == "the original"


async def test_media_type_link_extracts_urls_without_fetching(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    await _send(client, conv, "specs at https://docs.example.com/a?b=1 and https://cdn.example.org/x.pdf.")
    await _send(client, conv, "no links here")

    resp = await client.get(f"/api/conversations/{conv}/media", params={"type": "link"})
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert [item["url"] for item in data] == [
        "https://docs.example.com/a?b=1",
        "https://cdn.example.org/x.pdf",  # trailing sentence period is not part of the URL
    ]
    assert [item["domain"] for item in data] == ["docs.example.com", "cdn.example.org"]
    assert all(item["sender_name"] == "Alice" for item in data)
    assert all(item["title"] and item["created_at"] for item in data)


async def test_cross_tenant_404_on_every_new_endpoint(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    group = await _group_with_bob_and_erin(client, users)
    msg = await _send(client, group, "org A only")

    async with _client_for("carol@b.com") as carol:
        assert (await carol.post(f"/api/conversations/messages/{msg['_id']}/star")).status_code == 404
        assert (await carol.post(f"/api/conversations/messages/{msg['_id']}/pin")).status_code == 404
        assert (await carol.get(f"/api/conversations/{group}/starred")).status_code == 404
        assert (await carol.get(f"/api/conversations/{group}/pinned")).status_code == 404
        assert (await carol.put(f"/api/conversations/{group}/mute")).status_code == 404
        assert (await carol.get(f"/api/conversations/{group}/permissions")).status_code == 404
        resp = await carol.put(f"/api/conversations/{group}/permissions", json={"send_messages": False})
        assert resp.status_code == 404
        # DELETE replaces the old /export and /clear assertions here. Those two
        # routes no longer exist, so asserting 404 on them would have kept
        # passing while testing nothing — a missing route and a tenant rejection
        # are indistinguishable by status code.
        assert (await carol.delete(f"/api/conversations/{group}")).status_code == 404
        resp = await carol.get(f"/api/conversations/{group}/media", params={"type": "link"})
        assert resp.status_code == 404
        resp = await carol.get(f"/api/users/{users['alice'].id}/groups-in-common")
        assert resp.status_code == 404

    # Nothing leaked and nothing changed: Alice's view is untouched.
    assert (await client.get(f"/api/conversations/{group}/pinned")).json() == {"data": []}
    assert (await client.get(f"/api/conversations/{group}/permissions")).json()["send_messages"] is True


async def test_mute_suppresses_push_dispatch(client, two_orgs_with_users, monkeypatch):
    """A muted recipient must not be pushed.

    is_muted lived on the participant row and was served on every conversation,
    and nothing anywhere read it — so "Mute notifications" silenced nothing. This
    asserts the send path now filters muted recipients out of the push fan-out.
    """
    from app.services import messaging as messaging_mod

    users = two_orgs_with_users
    dispatched: list[list] = []

    def fake_dispatch(user_ids, payload):
        dispatched.append(list(user_ids))

    # push is imported lazily inside send_message, so patch it on its own module.
    import app.services.push as push_mod

    monkeypatch.setattr(push_mod, "dispatch_push_to_users", fake_dispatch)

    # Bob must look OFFLINE or he is never a push candidate in the first place.
    async def fake_statuses(user_ids):
        return {str(u): "offline" for u in user_ids}

    monkeypatch.setattr(messaging_mod.presence, "get_statuses", fake_statuses)

    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    await _send(client, conv, "should push")
    assert dispatched, "an offline recipient should have been pushed"
    assert users["bob"].id in dispatched[-1]

    # Bob mutes the conversation.
    async with _client_for("bob@a.com") as bob:
        resp = await bob.put(f"/api/conversations/{conv}/mute")
        assert resp.json() == {"is_muted": True}

    dispatched.clear()
    await _send(client, conv, "should NOT push")
    assert dispatched == [] or users["bob"].id not in dispatched[-1], (
        "a muted recipient must be excluded from the push fan-out"
    )

    # And the payload carries the conversation id for deep-linking.
    async with _client_for("bob@a.com") as bob:
        await bob.put(f"/api/conversations/{conv}/mute")  # unmute

    captured: list[dict] = []

    def capture(user_ids, payload):
        captured.append(payload)

    monkeypatch.setattr(push_mod, "dispatch_push_to_users", capture)
    await _send(client, conv, "with conv id")
    assert captured, "expected a push payload"
    assert captured[-1]["conversation_id"] == conv
    assert conv in captured[-1]["url"]
