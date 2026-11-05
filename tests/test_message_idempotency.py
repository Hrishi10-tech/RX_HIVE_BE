"""Retry safety: a repeated send must not create a second message.

send_message commits before the sender learns anything, so a lost response, a
timeout or a 502 leaves the client unable to tell "never arrived" from "arrived,
answer lost". Retrying was how that uncertainty became two copies of one message.
"""

import asyncio
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


async def _history(client, conv_id):
    """Sent messages only — opening a direct conversation writes a system row."""
    resp = await client.get(f"/api/conversations/{conv_id}/messages")
    assert resp.status_code == 200, resp.text
    return [m for m in resp.json()["messages"] if m["type"] != "system"]


async def test_resend_with_same_temp_id_returns_the_same_message(client, two_orgs_with_users):
    """The uncertain case: the client never learned the first send landed."""
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    first = await _send(client, conv, "only once", temp_id="key-abc")
    again = await _send(client, conv, "only once", temp_id="key-abc")

    assert again["_id"] == first["_id"], "the retry must resolve to the original row"
    assert again["temp_id"] == "key-abc", "the sender's key is still echoed"
    assert again["status"] == "sent"

    body = [m for m in await _history(client, conv) if m["content"] == "only once"]
    assert len(body) == 1, f"expected one stored message, got {len(body)}"


async def test_history_exposes_the_client_key_for_reconciliation(client, two_orgs_with_users):
    """The client cannot drop a stale local bubble without this on every row."""
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    await _send(client, conv, "reconcile me", temp_id="key-xyz")

    stored = [m for m in await _history(client, conv) if m["content"] == "reconcile me"][0]
    assert stored["client_msg_id"] == "key-xyz"


async def test_the_client_key_is_not_shown_to_the_other_participant(client, two_orgs_with_users):
    """It is one device's local state, so only that device is told it.

    Bob has nothing to reconcile alice's key against — his own unresolved bubbles
    carry keys he generated — and it is the one field on a message that is
    client-supplied text with no sanitising beyond a length cap.
    """
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    await _send(client, conv, "not bob's business", temp_id="key-private")

    async with _client_for("bob@a.com") as bob:
        seen = [m for m in await _history(bob, conv) if m["content"] == "not bob's business"][0]

    assert seen["client_msg_id"] is None, "alice's key must not reach bob"
    assert seen["_id"], "the message itself is still served to him"


async def test_a_changed_body_under_the_same_key_does_not_create_a_second_row(client, two_orgs_with_users):
    """The key identifies the send, so a repeat is the original — not an edit."""
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    first = await _send(client, conv, "the real one", temp_id="key-dup")
    again = await _send(client, conv, "a different body", temp_id="key-dup")

    assert again["_id"] == first["_id"]
    assert again["content"] == "the real one", "the stored message wins, nothing is overwritten"
    assert len(await _history(client, conv)) == 1


async def test_distinct_keys_still_create_distinct_messages(client, two_orgs_with_users):
    """The guard must not collapse two genuinely different sends."""
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    a = await _send(client, conv, "same words", temp_id="key-1")
    b = await _send(client, conv, "same words", temp_id="key-2")

    assert a["_id"] != b["_id"]
    assert len(await _history(client, conv)) == 2


async def test_sends_without_a_key_are_never_collapsed(client, two_orgs_with_users):
    """No temp_id means no idempotency claim — every send is its own message."""
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    await _send(client, conv, "no key here")
    await _send(client, conv, "no key here")

    stored = await _history(client, conv)
    assert len(stored) == 2
    assert all(m["client_msg_id"] is None for m in stored)


async def test_system_messages_carry_no_client_key(client, two_orgs_with_users):
    """They are not client-originated, so they stay outside the partial index."""
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    resp = await client.get(f"/api/conversations/{conv}/messages")
    system = [m for m in resp.json()["messages"] if m["type"] == "system"]
    assert system, "expected the conversation-started row"
    assert all(m["client_msg_id"] is None for m in system)


async def test_one_users_key_does_not_block_another(client, two_orgs_with_users):
    """The key is scoped per sender, so it cannot be used to suppress a peer."""
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    mine = await _send(client, conv, "from alice", temp_id="shared-key")

    async with _client_for("bob@a.com") as bob:
        theirs = await _send(bob, conv, "from bob", temp_id="shared-key")

    assert theirs["_id"] != mine["_id"], "bob's send must not resolve to alice's message"
    assert len(await _history(client, conv)) == 2


async def test_a_known_key_is_not_a_cross_tenant_path(client, two_orgs_with_users):
    """Carol is in the other org, so knowing the key and the conversation id
    buys her nothing.

    Two independent things have to hold for that, and this pins the pair: the
    access gate refuses her at all, and the refusal does not come back carrying
    the message she aimed the key at.

    Asserted as 404, not 403: the membership check in _require_send_access fires
    before the org check, so a non-participant never reaches the "Access denied"
    branch. That ordering is the right way round and is the part worth pinning —
    403 would confirm to an outsider that this conversation exists. Neutering the
    membership check turns this into a 403 and fails here, which is the regression
    this test is for.

    What it does NOT pin is the gate-before-idempotency ordering, despite that
    being the same feature's other cross-tenant concern: the lookup is scoped to
    sender_id, so carol misses alice's row whichever side of the gate it runs on
    (verified — hoisting the lookup above the gate leaves this test green). That
    ordering protects a REMOVED PARTICIPANT, who does share the sender_id, and
    proving it needs a test that removes someone from a conversation.
    """
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    mine = await _send(client, conv, "org A only", temp_id="shared-key")

    async with _client_for("carol@b.com") as carol:
        resp = await carol.post(
            f"/api/conversations/{conv}/messages",
            json={"content": "intrusion", "type": "text", "temp_id": "shared-key"},
        )
        assert resp.status_code == 404, resp.text
        # Refused, and not refused with a copy of what she was reaching for: a
        # replay answers with the stored message, which here would hand a foreign
        # tenant both the id and the body.
        assert mine["_id"] not in resp.text, "the refusal must not replay alice's message"
        assert "org A only" not in resp.text
        assert (await carol.get(f"/api/conversations/{conv}/messages")).status_code == 404

    after = await _history(client, conv)
    assert [m["_id"] for m in after] == [mine["_id"]], "nothing created, nothing replaced"


async def test_concurrent_identical_sends_store_one_message(client, two_orgs_with_users):
    """The check-then-act lookup can be lost; the unique index is the backstop."""
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    async def send():
        async with _client_for("alice@a.com") as c:
            resp = await c.post(
                f"/api/conversations/{conv}/messages",
                json={"content": "race", "type": "text", "temp_id": "key-race"},
            )
            return resp

    results = await asyncio.gather(*(send() for _ in range(4)), return_exceptions=True)
    assert all(not isinstance(r, Exception) for r in results), results
    assert all(r.status_code == 200 for r in results), [r.status_code for r in results]

    ids = {r.json()["_id"] for r in results}
    assert len(ids) == 1, f"concurrent sends resolved to {len(ids)} different messages"

    body = [m for m in await _history(client, conv) if m["content"] == "race"]
    assert len(body) == 1, f"expected one stored message, got {len(body)}"


async def test_a_send_that_loses_the_insert_race_replays_instead_of_erroring(
    client, two_orgs_with_users, monkeypatch
):
    """The idempotency backstop was guarding the wrong statement.

    uq_messages_client_msg_id is a partial unique INDEX, not a constraint, so it
    cannot be deferred and PostgreSQL enforces it when the INSERT runs — at
    db.flush(). The `except IntegrityError` written for exactly this race was
    attached to db.commit(). Session.commit() flushes first, but the explicit
    flush had already emitted the INSERT, so there was nothing left for the commit
    to raise: the backstop was unreachable for the one constraint its own comment
    names, and the loser of a genuine race got an IntegrityError out of the
    endpoint.

    THE RACE IS FORCED, deliberately. test_concurrent_identical_sends_store_one_message
    passes or fails depending on what ran before it in the same session — the
    check-then-act lookup catches the common case, where the winner commits before
    the loser gets that far, and only two genuinely overlapping sends skip it. This
    holds both senders at the lookup until both have seen nothing, so both go on to
    insert every time.
    """
    from app.services import messaging as messaging_mod

    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    original_find = messaging_mod._find_by_client_msg_id
    lookups = {"n": 0}
    both_have_looked = asyncio.Event()

    async def barrier_find(db, **kwargs):
        lookups["n"] += 1
        nth = lookups["n"]
        result = await original_find(db, **kwargs)
        # Only the two PRE-CHECKS are held. Later calls are the loser's recovery
        # lookup, which must not be delayed — and must be able to see the winner.
        if nth == 1:
            # NOT suppressed. If the second sender never reaches this lookup the
            # race was not forced, and the assertions below would then be satisfied
            # by an ordinary sequential replay — passing without ever exercising the
            # flush-conflict recovery they exist to cover. Generous bound, because
            # the point is to fail on "could not force it", not on a slow runner.
            await asyncio.wait_for(both_have_looked.wait(), timeout=15.0)
        elif nth == 2:
            both_have_looked.set()
        return result

    monkeypatch.setattr(messaging_mod, "_find_by_client_msg_id", barrier_find)

    async def send():
        async with _client_for("alice@a.com") as c:
            return await c.post(
                f"/api/conversations/{conv}/messages",
                json={"content": "forced race", "type": "text", "temp_id": "key-forced"},
            )

    results = await asyncio.gather(send(), send(), return_exceptions=True)

    assert all(not isinstance(r, Exception) for r in results), results
    assert all(r.status_code == 200 for r in results), [r.status_code for r in results]
    # Both senders were told about the same message, which is what a replay means.
    ids = {r.json()["_id"] for r in results}
    assert len(ids) == 1, f"the two sends resolved to {len(ids)} different messages"

    stored = [m for m in await _history(client, conv) if m["content"] == "forced race"]
    assert len(stored) == 1, f"expected one stored message, got {len(stored)}"
