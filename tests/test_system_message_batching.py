"""Batched system messages: N rows, one commit.

Four call sites emitted these one per member in a loop, and send_system_message
commits, re-loads and serializes on every call. Measured against the real schema
before the change: 7 queries per message, so 256 members cost 1,792 queries and
576ms of request time — and every one of those loops discarded the document it
paid for. After: 3 queries and 12ms, independent of the member count.
"""

import itertools

from sqlalchemy import event, select

import app.services.messaging as messaging_mod
from app.db.models import Conversation, ConversationType, Message, MessageType
from app.db.session import SessionLocal, engine
from app.services.messaging import send_system_messages
from app.utils import now_utc
from tests.conftest import make_org

_orgs = itertools.count()


async def _conversation():
    # Distinct slug per call: make_org slugifies the name, and two conversations in
    # one test would otherwise collide on organizations_slug_key.
    org = await make_org(f"Batch Co {next(_orgs)}")
    async with SessionLocal() as db:
        conv = Conversation(
            org_id=org.id,
            type=ConversationType.group,
            name="Batch",
            created_at=now_utc(),
            last_message_at=now_utc(),
        )
        db.add(conv)
        await db.commit()
        return conv.id


async def _system_contents(conversation_id) -> list[str]:
    async with SessionLocal() as db:
        return (
            (
                await db.execute(
                    select(Message.content)
                    .where(
                        Message.conversation_id == conversation_id,
                        Message.type == MessageType.system,
                    )
                    .order_by(Message.created_at)
                )
            )
            .scalars()
            .all()
        )


def _count_queries():
    """Context-manager-ish counter over the shared engine."""
    counter = {"n": 0}

    def _on_execute(conn, cursor, statement, params, context, executemany):
        counter["n"] += 1

    event.listen(engine.sync_engine, "before_cursor_execute", _on_execute)
    counter["stop"] = lambda: event.remove(engine.sync_engine, "before_cursor_execute", _on_execute)
    return counter


async def test_a_batch_costs_the_same_whatever_its_size():
    """The property that matters, asserted as a shape rather than a magic number.

    Per-message this was 7 queries each and scaled linearly — 1,792 of them for a
    256-member group, on the request path. A fixed cost is the whole point, so the
    test compares a small batch against a large one instead of pinning an exact
    count that would break on any unrelated query change.
    """
    small_conv = await _conversation()
    large_conv = await _conversation()

    counter = _count_queries()
    try:
        async with SessionLocal() as db:
            counter["n"] = 0
            await send_system_messages(db, small_conv, [f"m{i}" for i in range(5)], broadcast=False)
            small = counter["n"]
        async with SessionLocal() as db:
            counter["n"] = 0
            await send_system_messages(db, large_conv, [f"m{i}" for i in range(50)], broadcast=False)
            large = counter["n"]
    finally:
        counter["stop"]()

    assert small == large, f"cost scaled with the batch: {small} queries for 5, {large} for 50"
    # Ten times the messages for the same price, and both batches really landed.
    assert len(await _system_contents(small_conv)) == 5
    assert len(await _system_contents(large_conv)) == 50


async def test_the_batch_keeps_its_order():
    """History is ordered by created_at.

    A batch sharing one timestamp would be free to come back in any order —
    "Admin added Bob" before "Admin added Alice" — so each row is separated by a
    microsecond.
    """
    conv_id = await _conversation()
    wanted = [f"Admin added Member {i}" for i in range(25)]

    async with SessionLocal() as db:
        await send_system_messages(db, conv_id, wanted, broadcast=False)

    assert await _system_contents(conv_id) == wanted

    async with SessionLocal() as db:
        stamps = (
            (
                await db.execute(
                    select(Message.created_at)
                    .where(Message.conversation_id == conv_id)
                    .order_by(Message.created_at)
                )
            )
            .scalars()
            .all()
        )
    assert len(set(stamps)) == len(stamps), "two system messages share a timestamp"


async def test_broadcast_still_sends_one_frame_per_message(monkeypatch):
    """The batching is in the commit, not in what subscribers receive.

    api/groups.py add_members broadcasts these, so collapsing them into a single
    frame would change what a client gets. This costs less; it must not do less.
    """
    conv_id = await _conversation()
    frames: list[dict] = []

    async def capture(user_ids, payload):
        frames.append(payload)

    monkeypatch.setattr(messaging_mod, "publish_to_users", capture)

    async with SessionLocal() as db:
        await send_system_messages(db, conv_id, ["one", "two", "three"], broadcast=True)

    assert len(frames) == 3, f"expected one frame per message, got {len(frames)}"
    assert [f["message"]["content"] for f in frames] == ["one", "two", "three"]
    assert all(f["type"] == "new_message" for f in frames)


async def test_no_broadcast_publishes_nothing(monkeypatch):
    conv_id = await _conversation()
    frames: list[dict] = []

    async def capture(user_ids, payload):
        frames.append(payload)

    monkeypatch.setattr(messaging_mod, "publish_to_users", capture)

    async with SessionLocal() as db:
        await send_system_messages(db, conv_id, ["quiet"], broadcast=False)

    assert frames == []


async def test_an_empty_batch_writes_nothing_and_does_not_touch_the_conversation():
    conv_id = await _conversation()
    async with SessionLocal() as db:
        before = (await db.get(Conversation, conv_id)).last_message_at
        await send_system_messages(db, conv_id, [], broadcast=True)
    assert await _system_contents(conv_id) == []
    async with SessionLocal() as db:
        assert (await db.get(Conversation, conv_id)).last_message_at == before


async def test_the_conversation_is_bumped_to_the_last_message():
    """last_message_at drives sidebar ordering, so it must follow the batch."""
    conv_id = await _conversation()
    async with SessionLocal() as db:
        await send_system_messages(db, conv_id, ["a", "b", "c"], broadcast=False)

    async with SessionLocal() as db:
        conv = await db.get(Conversation, conv_id)
        newest = (
            await db.execute(
                select(Message.created_at)
                .where(Message.conversation_id == conv_id)
                .order_by(Message.created_at.desc())
                .limit(1)
            )
        ).scalar_one()
    assert conv.last_message_at == newest
