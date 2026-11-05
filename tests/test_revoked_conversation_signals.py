"""Deactivating a conversation must stop the signals as well as the content.

`is_active = False` is the only lever a superadmin has to cut a tenant off from a
cross-org group: archiving or deleting one leaves every ConversationParticipant row
in place (cross_org.py delete_group sets deleted_at and is_active and nothing
else), so the flag IS the revocation. Every content path enforces it —
_require_send_access, _member_attachment, the message lists, search, the media
drawer, the contact list.

Three signal paths did not, and they run on membership alone:
  * typing indicators (hub._is_participant),
  * presence fan-out (hub._conversation_partner_ids),
  * read receipts (messaging.mark_read).

So a tenant cut off from a group kept broadcasting into it and kept receiving its
members' online status, indefinitely, while their messages and media 404'd.
"""

import contextlib
import datetime as dt
import uuid

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.db.models import (
    Call,
    CallParticipant,
    Conversation,
    ConversationParticipant,
    ConversationType,
    Message,
    MessageReaction,
    User,
)
from app.db.session import SessionLocal
from app.main import app
from app.realtime import hub
from app.services import calls as calls_service
from app.services import conversations as conv_service
from app.services import enrich, messaging
from tests.conftest import CSRF, login, make_org, make_user


async def _deactivate_conversation(conv_id: uuid.UUID) -> None:
    """What archiving or deleting a group does: the flag, and nothing else."""
    async with SessionLocal() as db:
        conv = await db.get(Conversation, conv_id)
        conv.is_active = False
        await db.commit()


@contextlib.contextmanager
def _patched(module, name, value):
    """Swap an attribute back on exit — these tests patch around an await."""
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def _fresh_client() -> AsyncClient:
    """A client of its own, so a test can hold a logged-in session across calls."""
    c = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    c.headers.update(CSRF)
    return c


async def _direct_conversation(email_a: str, other: User) -> uuid.UUID:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c.headers.update(CSRF)
        await login(c, email_a)
        resp = await c.post("/api/conversations/direct", json={"participant_id": str(other.id)})
        assert resp.status_code == 200, resp.text
        return uuid.UUID(resp.json()["_id"])


def _capture(monkeypatch) -> list:
    sent: list = []

    async def _publish(user_ids, event):
        sent.append(([str(u) for u in user_ids], event))

    monkeypatch.setattr(hub, "publish_to_users", _publish)
    return sent


async def test_typing_is_not_broadcast_into_a_deactivated_conversation(client, monkeypatch):
    org = await make_org("Signals Typing Co")
    a = await make_user("a@sigtype.com", org_id=org.id)
    b = await make_user("b@sigtype.com", org_id=org.id)
    conv_id = await _direct_conversation("a@sigtype.com", b)

    sent = _capture(monkeypatch)
    await hub._handle_inbound(a, {"type": "typing_start", "conversation_id": str(conv_id)})
    assert [e["type"] for _, e in sent] == ["typing"], "control: it broadcasts while active"

    await _deactivate_conversation(conv_id)
    sent.clear()
    await hub._handle_inbound(a, {"type": "typing_start", "conversation_id": str(conv_id)})
    assert sent == [], "a revoked conversation must not carry typing indicators"


async def test_presence_does_not_reach_partners_of_a_deactivated_conversation(client, monkeypatch):
    org = await make_org("Signals Presence Co")
    a = await make_user("a@sigpres.com", org_id=org.id)
    b = await make_user("b@sigpres.com", org_id=org.id)
    conv_id = await _direct_conversation("a@sigpres.com", b)

    sent = _capture(monkeypatch)
    await hub._broadcast_presence(a, "online")
    assert [str(b.id)] in [uids for uids, _ in sent], "control: b hears a while active"

    await _deactivate_conversation(conv_id)
    sent.clear()
    await hub._broadcast_presence(a, "online")
    recipients = [uid for uids, _ in sent for uid in uids]
    assert str(b.id) not in recipients, (
        "online status kept reaching someone whose only shared conversation was revoked"
    )


async def test_read_receipts_are_refused_on_a_deactivated_conversation(client):
    org = await make_org("Signals Read Co")
    a = await make_user("a@sigread.com", org_id=org.id)
    b = await make_user("b@sigread.com", org_id=org.id)
    conv_id = await _direct_conversation("a@sigread.com", b)
    await _deactivate_conversation(conv_id)

    async with SessionLocal() as db:
        reader = await db.get(User, a.id)
        try:
            await messaging.mark_read(db, conversation_id=conv_id, reader=reader)
        except messaging.SendError as exc:
            assert exc.status_code == 404
        else:
            raise AssertionError("a read receipt was accepted on a revoked conversation")


# ---------------------------------------------------------------------------
# The rule itself, once. Three call sites apply it; this is what they apply.
# ---------------------------------------------------------------------------


def _conv(*, kind=ConversationType.group, org_id=None, is_active=True) -> Conversation:
    return Conversation(type=kind, org_id=org_id, is_active=is_active)


def _user(*, org_id=None, is_active=True) -> User:
    return User(email="x@y.com", display_name="X", password_hash="x", org_id=org_id, is_active=is_active)


def test_the_shared_gate_allows_an_ordinary_member():
    org = uuid.uuid4()
    messaging.assert_conversation_access(_conv(org_id=org), _user(org_id=org), is_member=True)


def test_the_shared_gate_refuses_a_deactivated_account_before_anything_else():
    """403 and this wording, because a revoked account is not a missing conversation."""
    org = uuid.uuid4()
    try:
        messaging.assert_conversation_access(
            _conv(org_id=org), _user(org_id=org, is_active=False), is_member=True
        )
    except messaging.SendError as exc:
        assert exc.status_code == 403
        assert "no longer active" in exc.detail
    else:
        raise AssertionError("a deactivated account passed the gate")


def test_the_shared_gate_refuses_a_deactivated_conversation():
    org = uuid.uuid4()
    try:
        messaging.assert_conversation_access(
            _conv(org_id=org, is_active=False), _user(org_id=org), is_member=True
        )
    except messaging.SendError as exc:
        assert exc.status_code == 404, "revocation must be indistinguishable from absence"
    else:
        raise AssertionError("a revoked conversation passed the gate")


def test_the_shared_gate_refuses_a_non_member():
    org = uuid.uuid4()
    try:
        messaging.assert_conversation_access(_conv(org_id=org), _user(org_id=org), is_member=False)
    except messaging.SendError as exc:
        assert exc.status_code == 404
    else:
        raise AssertionError("a non-member passed the gate")


def test_the_shared_gate_refuses_a_member_filed_under_another_org():
    """Representable, and only this predicate excludes it.

    A participant row carries no org, so nothing in the schema stops one pointing
    at a conversation in a different tenant. CodeRabbit made the same point about
    users.dept_id in Batch 46: a plain FK with no org-consistency constraint means
    the mismatched row exists as far as the database is concerned.
    """
    try:
        messaging.assert_conversation_access(
            _conv(org_id=uuid.uuid4()), _user(org_id=uuid.uuid4()), is_member=True
        )
    except messaging.SendError as exc:
        assert exc.status_code == 403
        assert exc.detail == "Access denied"
    else:
        raise AssertionError("a cross-tenant member passed the gate")


def test_the_shared_gate_lets_a_cross_org_conversation_span_orgs():
    """The converse of the test above, and the reason the org rule is not simply
    `org_id == user.org_id`: a cross_org group is supposed to span tenants."""
    messaging.assert_conversation_access(
        _conv(kind=ConversationType.cross_org, org_id=uuid.uuid4()),
        _user(org_id=uuid.uuid4()),
        is_member=True,
    )


# ---------------------------------------------------------------------------
# The signal paths, through the gate
# ---------------------------------------------------------------------------


async def test_a_deactivated_account_cannot_broadcast_typing(client, monkeypatch):
    """The other half of the gate, on a conversation that is perfectly live.

    Without reloading the user row this would pass anyway: the User handed to
    _handle_inbound was loaded when the socket opened, so its is_active is as stale
    as the socket.
    """
    org = await make_org("Signals Deact Co")
    a = await make_user("a@sigdeact.com", org_id=org.id)
    b = await make_user("b@sigdeact.com", org_id=org.id)
    conv_id = await _direct_conversation("a@sigdeact.com", b)

    sent = _capture(monkeypatch)
    async with SessionLocal() as db:
        row = await db.get(User, a.id)
        row.is_active = False
        await db.commit()

    # `a` is deliberately the stale object the socket would still be holding.
    await hub._handle_inbound(a, {"type": "typing_start", "conversation_id": str(conv_id)})
    assert sent == [], "a deactivated account broadcast typing on a live conversation"


async def test_presence_still_reaches_partners_of_live_conversations(client, monkeypatch):
    """The filter must be per conversation, not all-or-nothing.

    Without this, dropping presence entirely once any shared conversation was
    revoked would satisfy the revocation test — and would silence a working chat.
    """
    org = await make_org("Signals Mixed Co")
    a = await make_user("a@sigmix.com", org_id=org.id)
    revoked_partner = await make_user("b@sigmix.com", org_id=org.id)
    live_partner = await make_user("c@sigmix.com", org_id=org.id)

    dead_conv = await _direct_conversation("a@sigmix.com", revoked_partner)
    await _direct_conversation("a@sigmix.com", live_partner)
    await _deactivate_conversation(dead_conv)

    sent = _capture(monkeypatch)
    await hub._broadcast_presence(a, "online")
    recipients = [uid for uids, _ in sent for uid in uids]
    assert str(live_partner.id) in recipients, "the live conversation's partner must still hear it"
    assert str(revoked_partner.id) not in recipients


async def test_read_receipts_still_work_on_a_live_conversation(client):
    """Negative control for the mark_read gate."""
    org = await make_org("Signals Read OK Co")
    a = await make_user("a@sigreadok.com", org_id=org.id)
    b = await make_user("b@sigreadok.com", org_id=org.id)
    conv_id = await _direct_conversation("a@sigreadok.com", b)

    async with SessionLocal() as db:
        reader = await db.get(User, a.id)
        await messaging.mark_read(db, conversation_id=conv_id, reader=reader)
        me = await db.get(ConversationParticipant, (conv_id, a.id))
        assert me.last_read_at is not None


# ---------------------------------------------------------------------------
# The audience, not just the actor: a participant row carries no org
# ---------------------------------------------------------------------------


async def _intrude(conv_id: uuid.UUID, user_id: uuid.UUID) -> None:
    """Persist a foreign-org membership row in an ordinary conversation.

    Nothing in the schema forbids it: conversation_participants has no org column
    and no constraint tying it to the conversation's tenant. This is the row the
    recipient predicate exists for, and it is the same shape of representable
    inconsistency CodeRabbit identified for users.dept_id in Batch 46.
    """
    async with SessionLocal() as db:
        db.add(ConversationParticipant(conversation_id=conv_id, user_id=user_id))
        await db.commit()


async def test_no_signal_reaches_a_foreign_org_participant_row(client, monkeypatch):
    """Typing, presence and read receipts must all exclude it.

    Gating who may ACT and not who may be TOLD would leave the tenant boundary
    open in the direction that actually discloses data.
    """
    org_a = await make_org("Tenant A Signals")
    org_b = await make_org("Tenant B Signals")
    a = await make_user("a@tenanta.com", org_id=org_a.id)
    b = await make_user("b@tenanta.com", org_id=org_a.id)
    outsider = await make_user("x@tenantb.com", org_id=org_b.id)

    conv_id = await _direct_conversation("a@tenanta.com", b)
    await _intrude(conv_id, outsider.id)

    sent = _capture(monkeypatch)
    await hub._handle_inbound(a, {"type": "typing_start", "conversation_id": str(conv_id)})
    typing_recipients = [uid for uids, _ in sent for uid in uids]
    assert str(b.id) in typing_recipients, "control: the legitimate member is still told"
    assert str(outsider.id) not in typing_recipients, "typing leaked across tenants"

    sent.clear()
    await hub._broadcast_presence(a, "online")
    presence_recipients = [uid for uids, _ in sent for uid in uids]
    assert str(b.id) in presence_recipients, "control: the legitimate member still hears presence"
    assert str(outsider.id) not in presence_recipients, "presence leaked across tenants"

    read_sent: list = []

    async def _capture_read(user_ids, event):
        read_sent.append(([str(u) for u in user_ids], event))

    monkeypatch.setattr(messaging, "publish_to_users", _capture_read)
    async with SessionLocal() as db:
        reader = await db.get(User, a.id)
        await messaging.mark_read(db, conversation_id=conv_id, reader=reader)
    read_recipients = [uid for uids, _ in read_sent for uid in uids]
    assert str(b.id) in read_recipients, "control: the legitimate member gets the receipt"
    assert str(outsider.id) not in read_recipients, "read receipts leaked across tenants"


async def test_a_foreign_org_participant_row_is_not_sent_the_message_body(client, monkeypatch):
    """The severe case, and the one that was reached by the same unfiltered query.

    `send_message` fanned out to every membership row, so the foreign-org row
    received `new_message` with the serialized body — not merely the fact that
    someone was typing.
    """
    org_a = await make_org("Tenant A Body")
    org_b = await make_org("Tenant B Body")
    a = await make_user("a@bodya.com", org_id=org_a.id)
    b = await make_user("b@bodya.com", org_id=org_a.id)
    outsider = await make_user("x@bodyb.com", org_id=org_b.id)

    conv_id = await _direct_conversation("a@bodya.com", b)
    await _intrude(conv_id, outsider.id)

    sent: list = []

    async def _capture_send(user_ids, event):
        sent.append(([str(u) for u in user_ids], event))

    monkeypatch.setattr(messaging, "publish_to_users", _capture_send)
    async with SessionLocal() as db:
        sender = await db.get(User, a.id)
        await messaging.send_message(db, conversation_id=conv_id, sender=sender, content="tenant A internal")

    delivered = [uid for uids, event in sent if event.get("type") == "new_message" for uid in uids]
    assert str(b.id) in delivered, "control: the legitimate member receives the message"
    assert str(outsider.id) not in delivered, "the message body was delivered across tenants"


async def test_a_cross_org_group_still_reaches_every_tenant(client, monkeypatch):
    """The converse, and the reason the predicate is not a plain org equality.

    A cross_org conversation is supposed to span tenants, so the same rule that
    excludes the row above must let these through — otherwise the fix would
    silently break the feature it is protecting.
    """
    org_a = await make_org("Cross A")
    org_b = await make_org("Cross B")
    a = await make_user("a@crossa.com", org_id=org_a.id)
    partner = await make_user("p@crossb.com", org_id=org_b.id)

    async with SessionLocal() as db:
        conv = Conversation(type=ConversationType.cross_org, org_id=org_a.id, is_active=True)
        db.add(conv)
        await db.flush()
        db.add_all(
            [
                ConversationParticipant(conversation_id=conv.id, user_id=a.id),
                ConversationParticipant(conversation_id=conv.id, user_id=partner.id),
            ]
        )
        await db.commit()
        conv_id = conv.id

    async with SessionLocal() as db:
        recipients = await messaging.conversation_recipients(db, conv_id, exclude=a.id)
    assert [str(u) for u in recipients] == [str(partner.id)], (
        "a cross-org group must still deliver to the other tenant"
    )

    sent = _capture(monkeypatch)
    await hub._handle_inbound(a, {"type": "typing_start", "conversation_id": str(conv_id)})
    assert str(partner.id) in [uid for uids, _ in sent for uid in uids]


# ---------------------------------------------------------------------------
# Mutating a message, not just receiving one
# ---------------------------------------------------------------------------


async def _send_one(conv_id: uuid.UUID, sender_id: uuid.UUID) -> uuid.UUID:
    async with SessionLocal() as db:
        sender = await db.get(User, sender_id)
        doc = await messaging.send_message(db, conversation_id=conv_id, sender=sender, content="hi")
        return uuid.UUID(doc["_id"])


async def _expect_refused(coro, *, status: int) -> None:
    try:
        await coro
    except messaging.SendError as exc:
        assert exc.status_code == status, f"expected {status}, got {exc.status_code}: {exc.detail}"
    else:
        raise AssertionError("the mutation was allowed")


async def test_a_revoked_member_cannot_mutate_a_message(client):
    """Filtering the audience stops the disclosure; it does not stop the write.

    Reactions, edits, stars and pins all checked the participant row and nothing
    else, so a tenant cut off from a group could still change message state in it —
    and the resulting broadcast, now correctly filtered, would simply hide that it
    had happened.
    """
    org = await make_org("Mutate Revoked Co")
    a = await make_user("a@mutrev.com", org_id=org.id)
    b = await make_user("b@mutrev.com", org_id=org.id)
    conv_id = await _direct_conversation("a@mutrev.com", b)
    msg_id = await _send_one(conv_id, a.id)
    await _deactivate_conversation(conv_id)

    async with SessionLocal() as db:
        actor = await db.get(User, a.id)
        # 404 throughout: a revoked conversation must be indistinguishable from a
        # message that is not there.
        await _expect_refused(
            messaging.toggle_reaction(db, message_id=msg_id, actor=actor, emoji="👍"), status=404
        )
        await _expect_refused(
            messaging.edit_message(db, message_id=msg_id, actor=actor, content="edited"), status=404
        )
        await _expect_refused(messaging.toggle_star(db, message_id=msg_id, actor=actor), status=404)
        await _expect_refused(messaging.toggle_pin(db, message_id=msg_id, actor=actor), status=404)


async def test_a_foreign_org_participant_row_cannot_mutate_a_message(client):
    """The other representable state, on the write side."""
    org_a = await make_org("Mutate Tenant A")
    org_b = await make_org("Mutate Tenant B")
    a = await make_user("a@muta.com", org_id=org_a.id)
    b = await make_user("b@muta.com", org_id=org_a.id)
    outsider = await make_user("x@mutb.com", org_id=org_b.id)

    conv_id = await _direct_conversation("a@muta.com", b)
    msg_id = await _send_one(conv_id, a.id)
    await _intrude(conv_id, outsider.id)

    async with SessionLocal() as db:
        actor = await db.get(User, outsider.id)
        await _expect_refused(
            messaging.toggle_reaction(db, message_id=msg_id, actor=actor, emoji="👍"), status=404
        )
        await _expect_refused(messaging.toggle_star(db, message_id=msg_id, actor=actor), status=404)


async def test_a_deactivated_account_gets_403_not_404_on_a_mutation(client):
    """A dead account is not information about the conversation.

    The conversation-scoped refusals collapse to 404 to keep a foreign message
    indistinguishable from a missing one; the account case keeps the 403 the send
    path gives, which is why it has its own exception type rather than being told
    apart by restating the condition.
    """
    org = await make_org("Mutate Deact Co")
    a = await make_user("a@mutdeact.com", org_id=org.id)
    b = await make_user("b@mutdeact.com", org_id=org.id)
    conv_id = await _direct_conversation("a@mutdeact.com", b)
    msg_id = await _send_one(conv_id, a.id)

    async with SessionLocal() as db:
        actor = await db.get(User, a.id)
        actor.is_active = False
        await db.commit()

    async with SessionLocal() as db:
        actor = await db.get(User, a.id)
        try:
            await messaging.toggle_reaction(db, message_id=msg_id, actor=actor, emoji="👍")
        except messaging.AccountInactive as exc:
            assert exc.status_code == 403
            assert "no longer active" in exc.detail
        else:
            raise AssertionError("a deactivated account mutated a message")


async def test_an_ordinary_member_can_still_mutate(client):
    """Negative control. Without it, a gate that refused everything would pass."""
    org = await make_org("Mutate OK Co")
    a = await make_user("a@mutok.com", org_id=org.id)
    b = await make_user("b@mutok.com", org_id=org.id)
    conv_id = await _direct_conversation("a@mutok.com", b)
    msg_id = await _send_one(conv_id, a.id)

    async with SessionLocal() as db:
        actor = await db.get(User, a.id)
        assert await messaging.toggle_star(db, message_id=msg_id, actor=actor) is True
        reactions = await messaging.toggle_reaction(db, message_id=msg_id, actor=actor, emoji="👍")
        assert reactions, "the reaction should have been recorded"
        edited = await messaging.edit_message(db, message_id=msg_id, actor=actor, content="edited fine")
        assert edited["content"] == "edited fine"


# ---------------------------------------------------------------------------
# Forwarding: the source, the target, and the receipts
# ---------------------------------------------------------------------------


async def test_forwarding_out_of_a_revoked_conversation_is_refused(client):
    """Otherwise revocation is a formality: copy the message into a live group.

    The source conversation keeps its participant rows, so membership alone let a
    former member launder content past the revocation into somewhere they still
    belong.
    """
    org = await make_org("Forward Source Co")
    a = await make_user("a@fwdsrc.com", org_id=org.id)
    b = await make_user("b@fwdsrc.com", org_id=org.id)
    c = await make_user("c@fwdsrc.com", org_id=org.id)

    source = await _direct_conversation("a@fwdsrc.com", b)
    target = await _direct_conversation("a@fwdsrc.com", c)
    msg_id = await _send_one(source, a.id)
    await _deactivate_conversation(source)

    async with SessionLocal() as db:
        actor = await db.get(User, a.id)
        await _expect_refused(
            messaging.forward_message(
                db,
                actor=actor,
                message_id=msg_id,
                conversation_ids=[str(target)],
                contact_ids=[],
            ),
            status=404,
        )


async def test_a_forwarded_message_is_not_delivered_to_a_foreign_org_row(client, monkeypatch):
    """The forward path built its own fan-out and missed the tenant filter."""
    org_a = await make_org("Forward Tenant A")
    org_b = await make_org("Forward Tenant B")
    a = await make_user("a@fwda.com", org_id=org_a.id)
    b = await make_user("b@fwda.com", org_id=org_a.id)
    c = await make_user("c@fwda.com", org_id=org_a.id)
    outsider = await make_user("x@fwdb.com", org_id=org_b.id)

    source = await _direct_conversation("a@fwda.com", b)
    target = await _direct_conversation("a@fwda.com", c)
    msg_id = await _send_one(source, a.id)
    await _intrude(target, outsider.id)

    sent: list = []

    async def _capture_fwd(user_ids, event):
        sent.append(([str(u) for u in user_ids], event))

    monkeypatch.setattr(messaging, "publish_to_users", _capture_fwd)
    async with SessionLocal() as db:
        actor = await db.get(User, a.id)
        await messaging.forward_message(
            db, actor=actor, message_id=msg_id, conversation_ids=[str(target)], contact_ids=[]
        )

    delivered = [uid for uids, e in sent if e.get("type") == "new_message" for uid in uids]
    assert str(c.id) in delivered, "control: the legitimate target member receives it"
    assert str(outsider.id) not in delivered, "a forwarded message crossed tenants"


async def test_receipts_in_a_serialized_message_exclude_a_foreign_org_row(client):
    """The subtler half: conv_participants becomes read_by/delivered_to.

    Filtering delivery is not enough — the document every legitimate recipient
    receives is derived from the participant list, so an unfiltered list discloses
    the foreign row's user id to all of them even though nothing is delivered to it.
    """
    org_a = await make_org("Receipts Tenant A")
    org_b = await make_org("Receipts Tenant B")
    a = await make_user("a@rcpta.com", org_id=org_a.id)
    b = await make_user("b@rcpta.com", org_id=org_a.id)
    outsider = await make_user("x@rcptb.com", org_id=org_b.id)

    conv_id = await _direct_conversation("a@rcpta.com", b)
    msg_id = await _send_one(conv_id, a.id)
    await _intrude(conv_id, outsider.id)

    # Give both the legitimate member and the intruder a read timestamp after the
    # message, which is what puts a participant into read_by at all.
    async with SessionLocal() as db:
        later = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=1)
        for uid in (b.id, outsider.id):
            row = await db.get(ConversationParticipant, (conv_id, uid))
            row.last_read_at = later
        await db.commit()

    async with SessionLocal() as db:
        loaded = await enrich.load_message(db, msg_id)
        # conv_participants omitted on purpose: this exercises the serializer's own
        # fallback query, which was unfiltered too.
        doc = await enrich.serialize_message(db, loaded)

    seen = {r["user_id"] for r in doc["read_by"]} | {r["user_id"] for r in doc["delivered_to"]}
    assert str(b.id) in seen, "control: the legitimate member's receipt is still reported"
    assert str(outsider.id) not in seen, "a foreign-org participant was disclosed in the receipts"


# ---------------------------------------------------------------------------
# The HTTP read paths, and stored rows that outlive the gate
# ---------------------------------------------------------------------------


async def _mark_participant_read(conv_id: uuid.UUID, user_id: uuid.UUID) -> None:
    async with SessionLocal() as db:
        row = await db.get(ConversationParticipant, (conv_id, user_id))
        row.last_read_at = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=1)
        await db.commit()


async def test_the_message_page_does_not_disclose_a_foreign_org_row(client, two_orgs_with_users):
    """`_participants_of` fed the serializer directly, bypassing its safe fallback.

    This is the busiest path of all — every message load — and it is the one place
    the filtered fallback could not save, because the caller supplies the list.
    """
    users = two_orgs_with_users
    async with _fresh_client() as c:
        await login(c, "alice@a.com")
        conv_id = uuid.UUID(
            (await c.post("/api/conversations/direct", json={"participant_id": str(users["bob"].id)})).json()[
                "_id"
            ]
        )
        await c.post(f"/api/conversations/{conv_id}/messages", json={"content": "tenant only"})

        outsider = users["carol"]
        await _intrude(conv_id, outsider.id)
        await _mark_participant_read(conv_id, users["bob"].id)
        await _mark_participant_read(conv_id, outsider.id)

        body = (await c.get(f"/api/conversations/{conv_id}/messages")).json()

    seen = {
        r["user_id"]
        for m in body["messages"]
        for r in (m.get("read_by") or []) + (m.get("delivered_to") or [])
    }
    assert str(users["bob"].id) in seen, "control: the legitimate member's receipt is reported"
    assert str(outsider.id) not in seen, "the message page disclosed a foreign-org participant"


async def test_a_stale_foreign_reaction_is_not_serialized(client, two_orgs_with_users):
    """Gating new reactions does not remove the old ones.

    A reaction row created before the mutation gate landed outlives it, and
    serializing it hands every legitimate member that user's id and display name.
    """
    users = two_orgs_with_users
    async with _fresh_client() as c:
        await login(c, "alice@a.com")
        conv_id = uuid.UUID(
            (await c.post("/api/conversations/direct", json={"participant_id": str(users["bob"].id)})).json()[
                "_id"
            ]
        )
        msg_id = uuid.UUID(
            (await c.post(f"/api/conversations/{conv_id}/messages", json={"content": "hi"})).json()["_id"]
        )

        outsider = users["carol"]
        await _intrude(conv_id, outsider.id)
        # Written directly: the endpoint now refuses it, which is the point — this
        # is the row that predates the gate.
        async with SessionLocal() as db:
            db.add(MessageReaction(message_id=msg_id, user_id=outsider.id, emoji="👀"))
            db.add(MessageReaction(message_id=msg_id, user_id=users["bob"].id, emoji="👍"))
            await db.commit()

        body = (await c.get(f"/api/conversations/{conv_id}/messages")).json()

    reactors = {r["user_id"] for m in body["messages"] for r in (m.get("reactions") or [])}
    assert str(users["bob"].id) in reactors, "control: the legitimate reaction still renders"
    assert str(outsider.id) not in reactors, "a stale foreign reaction was disclosed"


async def test_a_cross_org_group_keeps_every_reaction_author(client, two_orgs_with_users):
    """The converse: filtering on the author's org must not strip a cross-org group.

    Without this, a filter that dropped every out-of-org author would satisfy the
    test above and quietly break reactions in exactly the conversations that are
    supposed to span tenants.
    """
    users = two_orgs_with_users
    async with SessionLocal() as db:
        conv = Conversation(type=ConversationType.cross_org, org_id=users["org_a"].id, is_active=True)
        db.add(conv)
        await db.flush()
        db.add_all(
            [
                ConversationParticipant(conversation_id=conv.id, user_id=users["alice"].id),
                ConversationParticipant(conversation_id=conv.id, user_id=users["carol"].id),
            ]
        )
        await db.commit()
        conv_id = conv.id

    msg_id = await _send_one(conv_id, users["alice"].id)
    async with SessionLocal() as db:
        db.add(MessageReaction(message_id=msg_id, user_id=users["carol"].id, emoji="🎉"))
        await db.commit()

    async with SessionLocal() as db:
        loaded = await enrich.load_message(db, msg_id)
        doc = await enrich.serialize_message(db, loaded)

    assert [r["user_id"] for r in doc["reactions"]] == [str(users["carol"].id)], (
        "a cross-org group must keep reaction authors from the other tenant"
    )


async def test_message_info_does_not_disclose_a_foreign_org_participant(client, two_orgs_with_users):
    """The "who has read this" route serializes each participant by name.

    The caller's own org is checked; the participant rows were not. So the sender
    asking who had read their message was shown the foreign-org row's display name
    and read state.
    """
    users = two_orgs_with_users
    async with _fresh_client() as c:
        await login(c, "alice@a.com")
        conv_id = uuid.UUID(
            (await c.post("/api/conversations/direct", json={"participant_id": str(users["bob"].id)})).json()[
                "_id"
            ]
        )
        msg_id = (
            await c.post(f"/api/conversations/{conv_id}/messages", json={"content": "who read this"})
        ).json()["_id"]

        outsider = users["carol"]
        await _intrude(conv_id, outsider.id)
        await _mark_participant_read(conv_id, users["bob"].id)
        await _mark_participant_read(conv_id, outsider.id)

        # Router prefix is /api/conversations, so the route is nested under it.
        info = await c.get(f"/api/conversations/messages/{msg_id}/info")
        assert info.status_code == 200, info.text
        body = info.json()

    listed = {
        # This route reports participants by display NAME, which is the disclosure:
        # the sender is told who read their message.
        entry["user_name"]
        for key in ("read_by", "delivered_to", "pending")
        for entry in (body.get(key) or [])
    }
    assert users["bob"].display_name in listed, "control: the legitimate member is still listed"
    assert outsider.display_name not in listed, "message info disclosed a foreign-org participant by name"


# ---------------------------------------------------------------------------
# System messages: the callers with no actor in front of them
# ---------------------------------------------------------------------------


async def test_a_system_message_is_not_broadcast_into_a_revoked_conversation(client, monkeypatch):
    """`send_system_message` takes a conversation id and broadcasts. No actor.

    Group events and call records go through it, so a call ending on a group that
    was archived mid-call — or an admin action on an archived one — pushed
    `new_message` to every former participant of a conversation they had been cut
    off from. Every other path into `conversation_recipients` has an actor gate in
    front of it; these two do not.
    """
    org = await make_org("System Msg Co")
    await make_user("a@sysmsg.com", org_id=org.id)
    b = await make_user("b@sysmsg.com", org_id=org.id)
    conv_id = await _direct_conversation("a@sysmsg.com", b)

    sent: list = []

    async def _capture(user_ids, event):
        sent.append(([str(u) for u in user_ids], event))

    monkeypatch.setattr(messaging, "publish_to_users", _capture)

    async with SessionLocal() as db:
        await messaging.send_system_message(db, conv_id, "Alice added Bob")
    assert [uid for uids, _ in sent for uid in uids], "control: it broadcasts while active"

    await _deactivate_conversation(conv_id)
    sent.clear()
    async with SessionLocal() as db:
        await messaging.send_system_message(db, conv_id, "📞 Voice call · 0:42")
    # The publish is still called, with nobody in it — that is a no-op delivery,
    # and the recipient list is the property that matters.
    assert [uid for uids, _ in sent for uid in uids] == [], (
        "a system message was pushed into a revoked conversation"
    )


async def test_a_batched_system_message_is_not_broadcast_into_a_revoked_conversation(client, monkeypatch):
    """The batch variant shares the helper, so it must share the refusal."""
    org = await make_org("System Batch Co")
    await make_user("a@sysbatch.com", org_id=org.id)
    b = await make_user("b@sysbatch.com", org_id=org.id)
    conv_id = await _direct_conversation("a@sysbatch.com", b)
    await _deactivate_conversation(conv_id)

    sent: list = []

    async def _capture(user_ids, event):
        sent.append(([str(u) for u in user_ids], event))

    monkeypatch.setattr(messaging, "publish_to_users", _capture)
    async with SessionLocal() as db:
        await messaging.send_system_messages(db, conv_id, ["one left", "two left"])
    assert [uid for uids, _ in sent for uid in uids] == []


async def test_the_system_message_row_is_still_written_when_revoked(client):
    """Not delivered is not the same as not recorded.

    The row is history and no read path will serve it, so refusing the broadcast
    must not also refuse the write — a call record that vanished because the group
    was archived mid-call would be a different bug.
    """
    org = await make_org("System Row Co")
    await make_user("a@sysrow.com", org_id=org.id)
    b = await make_user("b@sysrow.com", org_id=org.id)
    conv_id = await _direct_conversation("a@sysrow.com", b)
    await _deactivate_conversation(conv_id)

    async with SessionLocal() as db:
        doc = await messaging.send_system_message(db, conv_id, "📞 Voice call · 1:03")
    assert doc["content"] == "📞 Voice call · 1:03"

    async with SessionLocal() as db:
        stored = await db.get(Message, uuid.UUID(doc["_id"]))
        assert stored is not None, "the system message row must still be recorded"
        assert stored.conversation_id == conv_id


async def test_receipts_still_render_for_a_revoked_conversation(client):
    """Why `is_active` is on conversation_recipients and not on tenant_participants.

    They answer different questions: one is "who should be told", the other is
    "whose rows may be shown". Putting the revocation check on the second would
    blank out the receipts of already-stored history, which is a display change
    nobody asked for rather than a leak being closed.
    """
    org = await make_org("System Receipts Co")
    a = await make_user("a@sysrcpt.com", org_id=org.id)
    b = await make_user("b@sysrcpt.com", org_id=org.id)
    conv_id = await _direct_conversation("a@sysrcpt.com", b)
    msg_id = await _send_one(conv_id, a.id)
    await _mark_participant_read(conv_id, b.id)
    await _deactivate_conversation(conv_id)

    async with SessionLocal() as db:
        rows = await enrich.tenant_participants(db, conv_id)
        assert {p.user_id for p in rows} == {a.id, b.id}
        loaded = await enrich.load_message(db, msg_id)
        doc = await enrich.serialize_message(db, loaded)
        recipients = await messaging.conversation_recipients(db, conv_id)

    assert [r["user_id"] for r in doc["read_by"]] == [str(b.id)], "history keeps its receipts"
    assert recipients == [], "but nothing is delivered"


# ---------------------------------------------------------------------------
# Group calls: the roster is a seat, not just an audience
# ---------------------------------------------------------------------------


async def test_a_group_call_roster_excludes_a_foreign_org_participant(client, two_orgs_with_users):
    """Worse than a signal leak: the roster grants a seat.

    initiate_group_call created a CallParticipant row for every membership row, so
    a foreign-org row in an ordinary conversation was handed the caller's identity,
    the conversation id and the group name — and a seat it could use to join the
    audio, because POST /{id}/token only requires a CallParticipant row.
    """
    users = two_orgs_with_users
    async with SessionLocal() as db:
        conv = Conversation(type=ConversationType.group, org_id=users["org_a"].id, is_active=True)
        db.add(conv)
        await db.flush()
        db.add_all(
            [
                ConversationParticipant(conversation_id=conv.id, user_id=users["alice"].id),
                ConversationParticipant(conversation_id=conv.id, user_id=users["bob"].id),
            ]
        )
        await db.commit()
        conv_id = conv.id

    outsider = users["carol"]
    await _intrude(conv_id, outsider.id)

    sent: list = []

    async def _capture(user_ids, event):
        sent.append(([str(u) for u in user_ids], event))

    with _patched(calls_service, "publish_to_users", _capture):
        async with SessionLocal() as db:
            caller = await db.get(User, users["alice"].id)
        await calls_service.initiate_group_call(caller, conv_id, "voice")

    told = {
        uid for uids, e in sent if e.get("type") in ("call:incoming", "call:group_active") for uid in uids
    }
    assert str(users["bob"].id) in told, "control: the legitimate member is rung"
    assert str(outsider.id) not in told, "a foreign-org row was rung into a group call"

    async with SessionLocal() as db:
        call = (await db.execute(select(Call).where(Call.conversation_id == conv_id))).scalar_one()
        seats = {
            p.user_id
            for p in (
                await db.execute(select(CallParticipant).where(CallParticipant.call_id == call.id))
            ).scalars()
        }
    assert users["bob"].id in seats, "control: the legitimate member has a seat"
    assert outsider.id not in seats, "a foreign-org row was given a seat in the call"


async def test_a_cross_org_group_call_still_rings_the_other_tenant(client, two_orgs_with_users):
    """The converse. A cross_org group call is meant to span tenants."""
    users = two_orgs_with_users
    async with SessionLocal() as db:
        conv = Conversation(type=ConversationType.cross_org, org_id=users["org_a"].id, is_active=True)
        db.add(conv)
        await db.flush()
        db.add_all(
            [
                ConversationParticipant(conversation_id=conv.id, user_id=users["alice"].id),
                ConversationParticipant(conversation_id=conv.id, user_id=users["carol"].id),
            ]
        )
        await db.commit()
        conv_id = conv.id

    sent: list = []

    async def _capture(user_ids, event):
        sent.append(([str(u) for u in user_ids], event))

    with _patched(calls_service, "publish_to_users", _capture):
        async with SessionLocal() as db:
            caller = await db.get(User, users["alice"].id)
        await calls_service.initiate_group_call(caller, conv_id, "voice")

    told = {
        uid for uids, e in sent if e.get("type") in ("call:incoming", "call:group_active") for uid in uids
    }
    assert str(users["carol"].id) in told, "a cross-org group call must ring the other tenant"


# ---------------------------------------------------------------------------
# Conversation-level fan-out
# ---------------------------------------------------------------------------


async def _ordinary_group(users, *, members: list) -> uuid.UUID:
    async with SessionLocal() as db:
        conv = Conversation(
            type=ConversationType.group,
            org_id=users["org_a"].id,
            is_active=True,
            name="Tenant A Group",
        )
        db.add(conv)
        await db.flush()
        db.add_all([ConversationParticipant(conversation_id=conv.id, user_id=m.id) for m in members])
        await db.commit()
        return conv.id


async def test_conversation_events_skip_a_foreign_org_participant(client, two_orgs_with_users):
    """Icon changes, renames, removals and leaves all went through one helper.

    Every caller of broadcast_conversation_event is in api/groups.py — ordinary
    group conversations — and it read the raw membership rows.
    """
    users = two_orgs_with_users
    conv_id = await _ordinary_group(users, members=[users["alice"], users["bob"]])
    outsider = users["carol"]
    await _intrude(conv_id, outsider.id)

    sent: list = []

    async def _capture(user_ids, event):
        sent.append(([str(u) for u in user_ids], event))

    with _patched(conv_service, "publish_to_users", _capture):
        async with SessionLocal() as db:
            await conv_service.broadcast_conversation_event(
                db, conv_id, {"type": "group_updated", "name": "Tenant A Group"}
            )

    told = [uid for uids, _ in sent for uid in uids]
    assert str(users["bob"].id) in told, "control: the legitimate member is told"
    assert str(outsider.id) not in told, "a group event reached a foreign-org row"


async def test_conversation_created_is_not_sent_to_a_foreign_org_participant(client, two_orgs_with_users):
    """The largest single disclosure of the set: a whole serialized conversation.

    notify_conversation_created targets `loaded.participants` — the raw
    relationship — and sends each recipient the conversation's name, participant
    list and last message.
    """
    users = two_orgs_with_users
    conv_id = await _ordinary_group(users, members=[users["alice"], users["bob"]])
    outsider = users["carol"]
    await _intrude(conv_id, outsider.id)

    sent: list = []

    async def _capture(user_ids, event):
        sent.append(([str(u) for u in user_ids], event))

    with _patched(conv_service, "publish_to_users", _capture):
        async with SessionLocal() as db:
            await conv_service.notify_conversation_created(db, conv_id)

    told = [uid for uids, _ in sent for uid in uids]
    assert str(users["bob"].id) in told, "control: the legitimate member is notified"
    assert str(outsider.id) not in told, "a conversation document reached a foreign-org row"


async def test_participant_ids_stays_unfiltered_on_purpose(client, two_orgs_with_users):
    """It answers "who is a member", which is not the same question.

    cross_org.py uses it to build dedup and emptiness sets, where a foreign-org row
    must still count as present or an add would collide with the primary key.
    Filtering it would look tidier and would break that.
    """
    users = two_orgs_with_users
    conv_id = await _ordinary_group(users, members=[users["alice"], users["bob"]])
    outsider = users["carol"]
    await _intrude(conv_id, outsider.id)

    async with SessionLocal() as db:
        raw = set(await conv_service.participant_ids(db, conv_id))
        filtered = {p.user_id for p in await enrich.tenant_participants(db, conv_id)}

    assert outsider.id in raw, "membership is membership, however the row got there"
    assert outsider.id not in filtered
