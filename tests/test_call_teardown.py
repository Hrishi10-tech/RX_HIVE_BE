"""The server's authority over a call that has ended.

Ending a call is three separate facts: the row says so, Redis says so, and the
media room is gone. These cover the seams between them, because a client only
ever sees the last one — a room it can still publish into is a live call, whatever
the database says.
"""

import asyncio
import uuid

from sqlalchemy import select

from app.api import calls as api_calls
from app.db.models import Call, CallParticipant, CallStatus
from app.db.session import SessionLocal
from app.services import calls as calls_service
from app.utils import now_utc
from tests.test_calls_and_media import _client_for, _seed_connected_call


async def test_a_call_ended_concurrently_does_not_still_mint_a_token(client, two_orgs_with_users):
    """The `Call is not active` guard has to survive a concurrent teardown.

    The endpoint decides on `call.status`, then writes `joined_at`/`left_at` and
    mints a six-hour token. Read the status without locking the row and those are
    two separate transactions: a call ended in between is still handed a token, and
    the participant it just marked present keeps `left_at IS NULL` — which is the
    exact predicate the roster and last-one-out checks use.

    Driven with a real second transaction rather than a monkeypatch, because the
    thing under test is whether the read takes a row lock at all.
    """
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)

    holding = asyncio.Event()
    release = asyncio.Event()

    async def ender() -> None:
        """Hold the call row the way `_finalize`'s UPDATE does, uncommitted."""
        async with SessionLocal() as db:
            call = (
                await db.execute(select(Call).where(Call.id == uuid.UUID(call_id)).with_for_update())
            ).scalar_one()
            call.status = CallStatus.answered
            call.ended_at = now_utc()
            await db.flush()
            holding.set()
            await release.wait()
            await db.commit()

    task = asyncio.create_task(ender())
    await asyncio.wait_for(holding.wait(), timeout=5)

    async with _client_for("bob@a.com") as bob:
        request = asyncio.create_task(bob.post(f"/api/calls/{call_id}/token"))
        # Long enough that an unlocked read would have finished: it reads the row as
        # it was before the teardown, passes the guard and commits, all while the
        # ender still holds the lock.
        await asyncio.sleep(0.5)
        blocked = not request.done()
        release.set()
        await task
        resp = await request

    assert blocked, (
        "the token request completed while a concurrent teardown held the call row, "
        "so its status check ran against a row that was already being ended"
    )
    assert resp.status_code == 400, resp.text

    async with SessionLocal() as db:
        member = await db.get(CallParticipant, (uuid.UUID(call_id), users["bob"].id))
        assert member.joined_at is None, "a refused join must not mark the participant present"
        assert member.left_at is None


class _Participant:
    def __init__(self, identity: str) -> None:
        self.identity = identity


class _SfuRecorder:
    """What the server asked of the SFU, and whether it cleaned up after itself."""

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.removed: list[str] = []
        self.closed: list[bool] = []
        self.timeouts: list = []


class _StubRoomService:
    def __init__(
        self, recorder: _SfuRecorder, *, fail_delete: bool, present: list[str], unremovable: set[str]
    ) -> None:
        self._recorder = recorder
        self._fail_delete = fail_delete
        self._present = present
        self._unremovable = unremovable

    async def delete_room(self, request) -> None:
        self._recorder.deleted.append(request.room)
        if self._fail_delete:
            raise RuntimeError("SFU unreachable")

    async def list_participants(self, request):
        class _Roster:
            participants = [_Participant(i) for i in self._present]

        return _Roster()

    async def remove_participant(self, request) -> None:
        self._recorder.removed.append(request.identity)
        if request.identity in self._unremovable:
            raise RuntimeError("participant is wedged")


def _stub_sfu(
    monkeypatch,
    *,
    fail: bool = False,
    present: list[str] | None = None,
    unremovable: set[str] | None = None,
) -> _SfuRecorder:
    """Swap the LiveKit client out, leaving the real service functions under test."""
    recorder = _SfuRecorder()

    class _StubAPI:
        def __init__(self, *_args, **kwargs) -> None:
            recorder.timeouts.append(kwargs.get("timeout"))
            self.room = _StubRoomService(
                recorder,
                fail_delete=fail,
                present=present or [],
                unremovable=unremovable or set(),
            )

        async def aclose(self) -> None:
            recorder.closed.append(True)

    monkeypatch.setattr(calls_service.lk_api, "LiveKitAPI", _StubAPI)
    monkeypatch.setattr(calls_service, "publish_to_users", _noop_publish)
    return recorder


async def _noop_publish(user_ids, event) -> None:
    return None


async def test_ending_a_call_deletes_the_media_room(client, two_orgs_with_users, monkeypatch):
    """The room is the only part of a call a client can still act on.

    Every participant of a call that has just ended holds a six-hour join token
    that nothing can revoke, so a room left running is a call that can be rejoined
    after it ended — with no ring, no roster and no history.
    """
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)
    async with SessionLocal() as db:
        room_name = (await db.get(Call, uuid.UUID(call_id))).room_name

    sfu = _stub_sfu(monkeypatch)
    await calls_service._end(users["alice"], uuid.UUID(call_id))

    assert sfu.deleted == [room_name]
    assert sfu.closed == [True], "the SFU client must be closed even on the happy path"

    async with SessionLocal() as db:
        assert (await db.get(Call, uuid.UUID(call_id))).status == CallStatus.answered


async def test_a_room_that_cannot_be_deleted_still_ends_the_call(
    client, two_orgs_with_users, monkeypatch, caplog
):
    """Best-effort, like eviction: an unreachable SFU must not strand a live call."""
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)

    sfu = _stub_sfu(monkeypatch, fail=True)
    with caplog.at_level("WARNING", logger="app.services.calls"):
        await calls_service._end(users["alice"], uuid.UUID(call_id))

    assert len(sfu.deleted) == 1, "it must have tried"
    assert sfu.closed == [True], "the client must be closed on the failure path too"
    async with SessionLocal() as db:
        assert (await db.get(Call, uuid.UUID(call_id))).status == CallStatus.answered

    # WARNING, not DEBUG: a room outliving its call is invisible to every client
    # until one of them rejoins, so production has to be able to see it.
    assert any(
        "call.room_delete_failed" in record.message and record.levelname == "WARNING"
        for record in caplog.records
    ), [r.message for r in caplog.records]


async def test_a_call_with_no_room_name_asks_the_sfu_for_nothing(client, two_orgs_with_users, monkeypatch):
    """Guards the guard: `_delete_room` must not send an empty room name.

    Deleting room "" is not a no-op at the SFU, it is a malformed request, and a
    call can hold an empty `room_name` between INSERT and the flush that fills it.
    """
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)
    sfu = _stub_sfu(monkeypatch)

    async with SessionLocal() as db:
        call = await db.get(Call, uuid.UUID(call_id))
        call.room_name = ""
        assert await calls_service._delete_room(call) is False

    assert sfu.deleted == []


async def test_one_wedged_leg_does_not_strand_the_others(client, two_orgs_with_users, monkeypatch):
    """One person must be exactly one audio leg, and the loop has to finish.

    A stale leg that refuses to go used to abort the whole eviction, so a user with
    three legs kept two — each one a live microphone the other side hears as their
    own voice repeating.
    """
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)
    bob = users["bob"].id
    keep = calls_service.identity_for(bob, "laptop")
    stale_a = calls_service.identity_for(bob, "phone")
    stale_b = calls_service.identity_for(bob, "tablet")

    sfu = _stub_sfu(
        monkeypatch,
        present=[keep, stale_a, stale_b, calls_service.identity_for(users["alice"].id, "desk")],
        unremovable={stale_a},
    )
    async with SessionLocal() as db:
        call = await db.get(Call, uuid.UUID(call_id))
        removed = await calls_service.evict_other_devices(call, bob, keep)

    assert sfu.removed == [stale_a, stale_b], "both stale legs must be attempted, in order"
    assert removed == 1, "only the leg that actually went may be counted"
    assert sfu.closed == [True]


async def test_eviction_leaves_the_kept_leg_and_other_users_alone(client, two_orgs_with_users, monkeypatch):
    """The converse. Without it the test above would pass on a function that evicts everyone."""
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)
    bob = users["bob"].id
    keep = calls_service.identity_for(bob, "laptop")
    alice_leg = calls_service.identity_for(users["alice"].id, "desk")

    sfu = _stub_sfu(monkeypatch, present=[keep, alice_leg])
    async with SessionLocal() as db:
        call = await db.get(Call, uuid.UUID(call_id))
        removed = await calls_service.evict_other_devices(call, bob, keep)

    assert sfu.removed == []
    assert removed == 0


async def test_a_failed_eviction_is_logged_loudly(client, two_orgs_with_users, monkeypatch, caplog):
    """A wedged leg is not the benign first-join case and must not share its DEBUG line."""
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)
    bob = users["bob"].id
    keep = calls_service.identity_for(bob, "laptop")
    stale = calls_service.identity_for(bob, "phone")

    _stub_sfu(monkeypatch, present=[keep, stale], unremovable={stale})
    with caplog.at_level("WARNING", logger="app.services.calls"):
        async with SessionLocal() as db:
            call = await db.get(Call, uuid.UUID(call_id))
            await calls_service.evict_other_devices(call, bob, keep)

    assert any(
        "call.evict_failed" in record.message and record.levelname == "WARNING" for record in caplog.records
    ), [r.message for r in caplog.records]


async def test_an_unreachable_sfu_is_still_quiet(client, two_orgs_with_users, monkeypatch, caplog):
    """The first join of every call raises here, so this path must stay at DEBUG."""
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)

    class _Unreachable:
        def __init__(self, *_a, **_k) -> None:
            raise RuntimeError("no such room")

    monkeypatch.setattr(calls_service.lk_api, "LiveKitAPI", _Unreachable)
    with caplog.at_level("WARNING", logger="app.services.calls"):
        async with SessionLocal() as db:
            call = await db.get(Call, uuid.UUID(call_id))
            assert await calls_service.evict_other_devices(call, users["bob"].id, "x") == 0

    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


async def test_both_sfu_calls_use_a_short_timeout(client, two_orgs_with_users, monkeypatch):
    """The LiveKit client defaults to a 60-second total timeout.

    On these two paths that is a stall rather than a timeout: an SFU that accepts a
    connection and then stops answering would hold `POST /{id}/token` for a minute,
    and delay every participant's `call:ended` by the same — clients sitting in a
    call the server had already finished ending. Both callers are best-effort, so
    giving up fast is strictly better than waiting.
    """
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)
    sfu = _stub_sfu(monkeypatch, present=[])

    async with SessionLocal() as db:
        call = await db.get(Call, uuid.UUID(call_id))
        await calls_service._delete_room(call)
        await calls_service.evict_other_devices(call, users["bob"].id, "keep")

    assert len(sfu.timeouts) == 2, "both the teardown and the eviction build a client"
    for timeout in sfu.timeouts:
        assert timeout is not None, "the client's 60-second default must not be inherited"
        assert timeout.total is not None and timeout.total <= 10, timeout


class _StubEvent:
    """The three fields `_handle_webhook_event` reads off a verified LiveKit event."""

    def __init__(self, event: str, room: str, identity: str) -> None:
        self.event = event
        self.room = type("_Room", (), {"name": room})()
        self.participant = type("_P", (), {"identity": identity})()


async def test_a_stale_token_rejoining_an_ended_call_is_thrown_out(
    client, two_orgs_with_users, monkeypatch, caplog
):
    """Admission, enforced where it can be enforced.

    A join token is a JWT with a six-hour TTL that nothing can revoke, and deleting
    the room does not close that: LiveKit creates a room on join, so a holder can
    walk back into a deleted room. Two holders could then hold a conversation with
    no call record, no history, and nobody able to see them leave.

    The token cannot carry the check, but LiveKit reports the join — so the room is
    torn down again, and the participant must NOT be put back on the call.
    """
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)
    bob = users["bob"].id

    sfu = _stub_sfu(monkeypatch)
    await calls_service._end(users["alice"], uuid.UUID(call_id))
    sfu.deleted.clear()

    async with SessionLocal() as db:
        call = await db.get(Call, uuid.UUID(call_id))
        assert call.status == CallStatus.answered
        room_name = call.room_name
        # `_end` only stamps `left_at` on the person who ended it, so bob's row is
        # exactly what the handler would corrupt: joined_at NULL, left_at NULL.
        member = await db.get(CallParticipant, (uuid.UUID(call_id), bob))
        before = (member.joined_at, member.left_at)

    event = _StubEvent("participant_joined", room_name, calls_service.identity_for(bob, "phone"))
    with caplog.at_level("WARNING", logger="app.api.calls"):
        async with SessionLocal() as db:
            await api_calls._handle_webhook_event(db, event)

    assert sfu.deleted == [room_name], "the recreated room must be torn down again"
    async with SessionLocal() as db:
        member = await db.get(CallParticipant, (uuid.UUID(call_id), bob))
        assert (member.joined_at, member.left_at) == before, (
            "a join after the end must leave the participant row alone; the unguarded "
            "handler stamps joined_at and clears left_at, putting a live participant "
            "back onto a call that had already ended"
        )
        assert member.joined_at is None
    assert any("call.join_after_end" in r.message for r in caplog.records), [
        r.message for r in caplog.records
    ]


async def test_a_join_on_a_live_call_is_still_recorded(client, two_orgs_with_users, monkeypatch):
    """The converse. Without it the guard above could refuse every join and pass."""
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)
    bob = users["bob"].id
    sfu = _stub_sfu(monkeypatch)

    async with SessionLocal() as db:
        room_name = (await db.get(Call, uuid.UUID(call_id))).room_name

    event = _StubEvent("participant_joined", room_name, calls_service.identity_for(bob, "phone"))
    async with SessionLocal() as db:
        await api_calls._handle_webhook_event(db, event)

    assert sfu.deleted == [], "a live call's room must not be torn down"
    async with SessionLocal() as db:
        member = await db.get(CallParticipant, (uuid.UUID(call_id), bob))
        assert member.joined_at is not None
        assert member.left_at is None


async def test_a_webhook_join_racing_a_finalize_does_not_mark_the_participant_present(
    client, two_orgs_with_users, monkeypatch
):
    """The webhook's status check has to be under the same lock as its write.

    Under READ COMMITTED an unlocked read leaves this interleaving: the handler
    reads an active call, `_finalize` commits a terminal status, and the handler
    then commits `joined_at` / `left_at = None` — a participant marked present on a
    call that has ended, having passed the terminal branch that would have torn the
    room down. That is the same defect the token endpoint had, by a second route.

    Driven with a real second transaction, like the token test: what is under test
    is whether the read takes the lock at all.
    """
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)
    bob = users["bob"].id
    _stub_sfu(monkeypatch)

    async with SessionLocal() as db:
        room_name = (await db.get(Call, uuid.UUID(call_id))).room_name

    holding = asyncio.Event()
    release = asyncio.Event()

    async def finalizer() -> None:
        async with SessionLocal() as db:
            call = (
                await db.execute(select(Call).where(Call.id == uuid.UUID(call_id)).with_for_update())
            ).scalar_one()
            call.status = CallStatus.answered
            call.ended_at = now_utc()
            await db.flush()
            holding.set()
            await release.wait()
            await db.commit()

    async def webhook() -> None:
        async with SessionLocal() as db:
            await api_calls._handle_webhook_event(
                db, _StubEvent("participant_joined", room_name, calls_service.identity_for(bob, "phone"))
            )

    task = asyncio.create_task(finalizer())
    await asyncio.wait_for(holding.wait(), timeout=5)
    handler = asyncio.create_task(webhook())
    await asyncio.sleep(0.5)
    blocked = not handler.done()
    release.set()
    await task
    await handler

    assert blocked, (
        "the webhook handler ran to completion while a finalize held the call row, "
        "so its status check saw a call that was already being ended"
    )
    async with SessionLocal() as db:
        member = await db.get(CallParticipant, (uuid.UUID(call_id), bob))
        assert member.joined_at is None, "a join racing the teardown must not mark the participant present"
        assert member.left_at is None
