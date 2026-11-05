"""Calls surviving bad networks: ring delivery, reconnect grace, durable timers.

Every test here corresponds to a specific way a call used to fail that no client
could work around, because the decision was the server's:

  * a callee whose socket happened to be absent was refused outright, so the phone
    never rang (`test_ring_is_not_refused_...`, `test_replay_...`);
  * a socket drop ended the call instantly, including the routine 4001 every client
    takes when its 15-minute access cookie lapses (`test_link_down_does_not_end...`);
  * ring timeouts lived in a per-process dict, so a worker restart left a call
    ringing forever (`test_ring_timeout_fires_from_the_sweeper`).
"""

import asyncio

import pytest

from app.db.models import Call, CallParticipant, CallStatus, CallType
from app.db.session import SessionLocal
from app.realtime.redis_bus import get_redis
from app.services import call_deadlines
from app.services import calls as calls_service
from app.utils import now_utc


@pytest.fixture
def captured_events(monkeypatch):
    """Collect every publish_to_users call instead of sending it to Redis.

    Patched at both the bus and the two importers, because `services/calls.py` and
    `api/calls.py` bind the name at import time.
    """
    sent: list[tuple[list, dict]] = []

    async def _capture(user_ids, event):
        sent.append(([str(u) for u in user_ids], event))

    monkeypatch.setattr(calls_service, "publish_to_users", _capture)
    return sent


def events_of(sent, type_: str) -> list[dict]:
    return [event for _, event in sent if event.get("type") == type_]


def recipients_of(sent, type_: str) -> list[str]:
    return [uid for uids, event in sent if event.get("type") == type_ for uid in uids]


async def _seed_call(users, *, status: CallStatus, is_group: bool = False) -> Call:
    async with SessionLocal() as db:
        call = Call(
            org_id=users["org_a"].id,
            initiated_by=users["alice"].id,
            type=CallType.voice,
            status=status,
            is_group=is_group,
            room_name="",
            answered_at=now_utc() if status == CallStatus.connected else None,
        )
        db.add(call)
        await db.flush()
        call.room_name = calls_service.room_name_for(call.id)
        joined = now_utc() if status == CallStatus.connected else None
        db.add_all(
            [
                CallParticipant(call_id=call.id, user_id=users["alice"].id, joined_at=joined),
                CallParticipant(call_id=call.id, user_id=users["bob"].id, joined_at=joined),
            ]
        )
        await db.commit()
        await db.refresh(call)
        return call


async def _status_of(call_id) -> CallStatus:
    async with SessionLocal() as db:
        return (await db.get(Call, call_id)).status


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


async def test_ring_is_not_refused_when_the_callee_has_no_live_socket(two_orgs_with_users, captured_events):
    """An absent socket is not an absent person.

    `presence.is_online` is false for an iOS app that just backgrounded, a client
    refreshing a lapsed access cookie, a Wi-Fi handover, and a Redis blip. Refusing
    the call in those windows is what made "the call is not received" the normal
    case rather than the exception.
    """
    users = two_orgs_with_users
    await calls_service.initiate_direct_call(users["alice"], users["bob"].id, "voice", None)

    assert events_of(captured_events, "call:unavailable") == []
    incoming = events_of(captured_events, "call:incoming")
    assert len(incoming) == 1
    assert incoming[0]["caller"]["id"] == str(users["alice"].id)
    assert recipients_of(captured_events, "call:incoming") == [str(users["bob"].id)]

    # The caller is told the callee looked offline, as a hint rather than a refusal.
    ringing = events_of(captured_events, "call:ringing_started")[0]
    assert ringing["callee_online"] is False
    assert ringing["ring_timeout"] == calls_service.RING_TIMEOUT_SECONDS

    # And the call really is ringing, with a durable deadline behind it.
    call_id = incoming[0]["call_id"]
    assert await _status_of(call_id) == CallStatus.ringing
    assert await call_deadlines.remaining(call_deadlines.RING, call_id) is not None


async def test_replay_pending_ring_redelivers_a_ring_missed_while_offline(
    two_orgs_with_users, captured_events
):
    """The frame that used to be lost forever.

    `call:incoming` is a fire-and-forget publish. Sent while the callee's socket was
    reconnecting it went to a channel with no subscriber and evaporated — the server
    rang on for the rest of the window while the phone showed nothing.
    """
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.ringing)
    captured_events.clear()

    await calls_service.resume_calls_for(users["bob"].id)

    replayed = events_of(captured_events, "call:incoming")
    assert len(replayed) == 1
    assert replayed[0]["call_id"] == str(call.id)
    assert replayed[0]["replayed"] is True
    assert replayed[0]["caller"]["id"] == str(users["alice"].id)

    # The caller is not re-rung; they get the resumable state instead.
    captured_events.clear()
    await calls_service.resume_calls_for(users["alice"].id)
    assert events_of(captured_events, "call:incoming") == []
    resume = events_of(captured_events, "call:resume")
    assert len(resume) == 1
    assert resume[0]["call"]["is_initiator"] is True


async def test_replay_pending_ring_is_silent_with_no_live_call(two_orgs_with_users, captured_events):
    await calls_service.resume_calls_for(two_orgs_with_users["bob"].id)
    assert captured_events == []


async def test_active_call_state_survives_a_redis_flush(two_orgs_with_users):
    """Read from Postgres, not from the `call:user:*` marker: the call row is the
    only authority on whether a call is live, and it is what a failover keeps."""
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.connected)
    await get_redis().flushdb()

    async with SessionLocal() as db:
        state = await calls_service.active_call_state(db, users["bob"].id)
    assert state is not None
    assert state["call_id"] == str(call.id)
    assert state["room"] == f"call_{call.id}"
    assert state["self_joined"] is True
    assert sorted(state["peer_links"].values()) == [calls_service.LINK_UP] * 2


# ---------------------------------------------------------------------------
# Survivability
# ---------------------------------------------------------------------------


async def test_link_down_does_not_end_a_connected_call(two_orgs_with_users, captured_events):
    """The single most damaging behaviour this replaces.

    The old handler ended the call the instant the last socket dropped. Because the
    server closes every socket 4001 when the access cookie lapses, that killed every
    call that outlived the remaining cookie lifetime, mid-sentence.
    """
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.connected)
    await calls_service._set_in_call([users["alice"].id, users["bob"].id], call.id)
    captured_events.clear()

    await calls_service.handle_user_link_down(users["bob"])

    assert await _status_of(call.id) == CallStatus.connected
    assert events_of(captured_events, "call:ended") == []

    # The peer is told, so both ends can show "Connecting…" for the same event.
    peer = events_of(captured_events, "call:peer_state")
    assert len(peer) == 1
    assert peer[0]["state"] == "reconnecting"
    assert peer[0]["user_id"] == str(users["bob"].id)
    assert recipients_of(captured_events, "call:peer_state") == [str(users["alice"].id)]

    # ...and a grace deadline is armed rather than the call being resolved now.
    member = calls_service._grace_member(call.id, users["bob"].id)
    assert await call_deadlines.remaining(call_deadlines.GRACE, member) is not None


async def test_link_up_inside_the_grace_window_resumes_the_call(two_orgs_with_users, captured_events):
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.connected)
    await calls_service._set_in_call([users["alice"].id, users["bob"].id], call.id)

    await calls_service.handle_user_link_down(users["bob"])
    captured_events.clear()
    await calls_service.handle_user_link_up(users["bob"].id)

    member = calls_service._grace_member(call.id, users["bob"].id)
    assert await call_deadlines.remaining(call_deadlines.GRACE, member) is None
    assert await _status_of(call.id) == CallStatus.connected

    peer = events_of(captured_events, "call:peer_state")[0]
    assert peer["state"] == "connected"

    # The window has been closed, so firing the expiry late is a no-op.
    await calls_service._grace_expired(call.id, users["bob"].id)
    assert await _status_of(call.id) == CallStatus.connected


async def test_grace_expiry_ends_a_call_the_user_never_came_back_to(two_orgs_with_users, captured_events):
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.connected)
    await calls_service._set_in_call([users["alice"].id, users["bob"].id], call.id)

    await calls_service.handle_user_link_down(users["bob"])
    captured_events.clear()
    await calls_service._grace_expired(call.id, users["bob"].id)

    assert await _status_of(call.id) == CallStatus.answered
    ended = events_of(captured_events, "call:ended")
    assert len(ended) == 1
    assert ended[0]["reason"] == "disconnected"
    assert recipients_of(captured_events, "call:ended") == [str(users["alice"].id)]


async def test_a_ringing_callee_who_drops_is_not_marked_missed_early(two_orgs_with_users, captured_events):
    """The ring window owns the miss, not the grace window.

    Cutting the ring short here would report a miss the callee could still have
    answered a second later, which is exactly what the old immediate-finalise did.
    """
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.ringing)
    await calls_service._set_in_call([users["alice"].id, users["bob"].id], call.id)

    await calls_service.handle_user_link_down(users["bob"])
    await calls_service._grace_expired(call.id, users["bob"].id)

    assert await _status_of(call.id) == CallStatus.ringing


async def test_a_caller_who_vanishes_while_ringing_cancels_after_grace(two_orgs_with_users, captured_events):
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.ringing)
    await calls_service._set_in_call([users["alice"].id, users["bob"].id], call.id)

    await calls_service.handle_user_link_down(users["alice"])
    assert await _status_of(call.id) == CallStatus.ringing  # still ringing during grace

    captured_events.clear()
    await calls_service._grace_expired(call.id, users["alice"].id)

    assert await _status_of(call.id) == CallStatus.cancelled
    assert recipients_of(captured_events, "call:cancelled") == [str(users["bob"].id)]


# ---------------------------------------------------------------------------
# Durable timers
# ---------------------------------------------------------------------------


async def test_ring_timeout_fires_from_the_sweeper(two_orgs_with_users, captured_events):
    """A deadline scheduled by a worker that has since died still fires.

    Ring timeouts used to be asyncio tasks in a module-level dict, so a deploy
    mid-ring lost them: the call stayed `ringing` forever, the caller heard ringback
    with no timeout, and the row never reached call history.
    """
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.ringing)
    # Due immediately, as if scheduled 45s ago by a process that is now gone.
    await call_deadlines.schedule(call_deadlines.RING, str(call.id), -1)
    captured_events.clear()

    assert await calls_service.sweep_due_deadlines() == 1

    assert await _status_of(call.id) == CallStatus.missed
    missed = events_of(captured_events, "call:missed")
    assert len(missed) == 2  # caller (bare) and callee (with the caller brief)
    assert {m.get("caller") is None for m in missed} == {True, False}

    # Claimed, so a second sweep — from this or any other worker — is a no-op.
    assert await calls_service.sweep_due_deadlines() == 0


async def test_sweeper_fires_a_due_grace_window(two_orgs_with_users, captured_events):
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.connected)
    await calls_service._set_in_call([users["alice"].id, users["bob"].id], call.id)
    await calls_service.handle_user_link_down(users["bob"])
    await call_deadlines.schedule(
        call_deadlines.GRACE, calls_service._grace_member(call.id, users["bob"].id), -1
    )
    captured_events.clear()

    assert await calls_service.sweep_due_deadlines() == 1
    assert await _status_of(call.id) == CallStatus.answered


async def test_claiming_a_deadline_is_exclusive(two_orgs_with_users):
    """Two workers sweeping at once must not both fire the same deadline."""
    await call_deadlines.schedule(call_deadlines.RING, "member-a", -1)
    await call_deadlines.schedule(call_deadlines.RING, "member-b", -1)

    first, second = await asyncio.gather(
        call_deadlines.claim_due(call_deadlines.RING),
        call_deadlines.claim_due(call_deadlines.RING),
    )
    assert sorted(first + second) == ["member-a", "member-b"]
    assert not set(first) & set(second)


async def test_a_deadline_in_the_future_is_not_claimed():
    await call_deadlines.schedule(call_deadlines.RING, "later", 300)
    assert await call_deadlines.claim_due(call_deadlines.RING) == []
    assert await call_deadlines.remaining(call_deadlines.RING, "later") > 290
    await call_deadlines.cancel(call_deadlines.RING, "later")
    assert await call_deadlines.remaining(call_deadlines.RING, "later") is None


async def test_a_failed_ring_timeout_is_retried_not_lost(two_orgs_with_users, monkeypatch):
    """A handler that raises must not take the deadline down with it.

    claim_due ZREMs in the same script that reads, so an exception after the claim
    leaves the deadline existing nowhere — the same "nothing to sweep" state a
    lost asyncio timer used to produce, and the call stays ringing forever.
    """
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.ringing)
    await call_deadlines.schedule(call_deadlines.RING, str(call.id), -1)

    calls = {"n": 0}
    real = calls_service._ring_timeout

    async def _flaky(call_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database blip")
        return await real(call_id)

    monkeypatch.setattr(calls_service, "_ring_timeout", _flaky)

    swept = await calls_service.sweep_due_deadlines()

    # The deadline survived its handler: back on the set, due again shortly.
    left = await call_deadlines.remaining(call_deadlines.RING, str(call.id))
    assert left is not None, "the deadline was lost when its handler raised"
    assert 0 < left <= calls_service.DEADLINE_RETRY_SECONDS
    # Nothing fired, and the call is untouched, still waiting to be resolved.
    assert swept == 0
    assert await _status_of(call.id) == CallStatus.ringing

    # Once it is due again the retry lands, and the call finally resolves.
    await call_deadlines.schedule(call_deadlines.RING, str(call.id), -1)
    assert await calls_service.sweep_due_deadlines() == 1
    assert calls["n"] == 2
    assert await _status_of(call.id) == CallStatus.missed
    assert await call_deadlines.remaining(call_deadlines.RING, str(call.id)) is None


async def test_a_failed_grace_expiry_is_retried_not_lost(two_orgs_with_users, monkeypatch):
    """Same guarantee on the other deadline kind.

    This one has a live way to fail that the ring path does not: _grace_expired
    reads the Redis link key outside any degrade guard, deliberately, because
    neither answer is safe to assume.
    """
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.connected)
    await calls_service._set_in_call([users["alice"].id, users["bob"].id], call.id)
    await calls_service.handle_user_link_down(users["bob"])
    member = calls_service._grace_member(call.id, users["bob"].id)
    await call_deadlines.schedule(call_deadlines.GRACE, member, -1)

    calls = {"n": 0}
    real = calls_service._grace_expired

    async def _flaky(call_id, user_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("redis refused the link read")
        return await real(call_id, user_id)

    monkeypatch.setattr(calls_service, "_grace_expired", _flaky)

    swept = await calls_service.sweep_due_deadlines()

    # The deadline survived its handler, and nothing about the call moved.
    left = await call_deadlines.remaining(call_deadlines.GRACE, member)
    assert left is not None, "the grace deadline was lost when its handler raised"
    assert 0 < left <= calls_service.DEADLINE_RETRY_SECONDS
    assert swept == 0
    assert await _status_of(call.id) == CallStatus.connected

    # Once due again the retry lands, resolving the call without the absent
    # participant and consuming the deadline for good.
    await call_deadlines.schedule(call_deadlines.GRACE, member, -1)
    assert await calls_service.sweep_due_deadlines() == 1
    assert calls["n"] == 2
    assert await _status_of(call.id) == CallStatus.answered
    assert await call_deadlines.remaining(call_deadlines.GRACE, member) is None


# ---------------------------------------------------------------------------
# Accept / join, and the errors that used to be silence
# ---------------------------------------------------------------------------


async def test_accepting_an_already_connected_call_replays_the_acceptance(
    two_orgs_with_users, captured_events
):
    """A retried Accept after a socket blip must connect, not refuse.

    The first `call:accept` can be swallowed by a socket that closed under it. The
    user taps Answer again; refusing the second one (as `no_longer_ringing` did)
    left them on "Connecting…" for a call that was live and joinable.
    """
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.connected)
    captured_events.clear()

    await calls_service._accept(users["bob"], call.id)

    accepted = events_of(captured_events, "call:accepted")
    assert len(accepted) == 1
    assert accepted[0]["accepter_id"] == str(users["bob"].id)
    assert events_of(captured_events, "call:error") == []


async def test_accepting_a_dead_call_answers_with_an_error(two_orgs_with_users, captured_events):
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.ringing)
    async with SessionLocal() as db:
        row = await db.get(Call, call.id)
        row.status = CallStatus.missed
        await db.commit()
    captured_events.clear()

    await calls_service._accept(users["bob"], call.id)

    error = events_of(captured_events, "call:error")[0]
    assert error["reason"] == "no_longer_ringing"
    assert error["status"] == "missed"


async def test_joining_a_finished_group_call_answers_with_an_error(two_orgs_with_users, captured_events):
    """`_join` used to return silently on every refusal, leaving the joiner's UI on
    "Connecting…" with nothing to act on."""
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.connected, is_group=True)
    async with SessionLocal() as db:
        row = await db.get(Call, call.id)
        row.status = CallStatus.answered
        await db.commit()
    captured_events.clear()

    await calls_service._join(users["bob"], call.id)

    error = events_of(captured_events, "call:error")[0]
    assert error["reason"] == "not_joinable"


async def test_peer_state_relay_reaches_only_the_other_participants(two_orgs_with_users, captured_events):
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.connected)
    captured_events.clear()

    await calls_service._relay_peer_state(users["bob"], call.id, "reconnecting", "poor")

    peer = events_of(captured_events, "call:peer_state")[0]
    assert peer["state"] == "reconnecting"
    assert peer["quality"] == "poor"
    assert peer["user_id"] == str(users["bob"].id)
    assert recipients_of(captured_events, "call:peer_state") == [str(users["alice"].id)]


async def test_peer_state_rejects_values_outside_the_vocabulary(two_orgs_with_users, captured_events):
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.connected)
    captured_events.clear()

    await calls_service._relay_peer_state(users["bob"], call.id, "on-fire", "terrible")

    assert events_of(captured_events, "call:peer_state") == []


async def test_media_toggle_is_ignored_once_the_call_is_over(two_orgs_with_users, captured_events):
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.connected)
    async with SessionLocal() as db:
        row = await db.get(Call, call.id)
        row.status = CallStatus.answered
        await db.commit()
    captured_events.clear()

    await calls_service._toggle_media(users["bob"], call.id, "audio", False)

    assert events_of(captured_events, "call:media_toggle") == []


def test_identity_round_trips_and_tolerates_a_bare_user_id():
    identity = calls_service.identity_for("11111111-1111-1111-1111-111111111111", "phone")
    assert identity == "11111111-1111-1111-1111-111111111111#phone"
    assert calls_service.user_id_from_identity(identity) == "11111111-1111-1111-1111-111111111111"
    # A webhook queued by an older build carries a bare id.
    assert calls_service.user_id_from_identity("abc") == "abc"
    assert calls_service.user_id_from_identity(None) == ""
    # Omitting the device still yields something unique per call.
    assert calls_service.identity_for("u", None) != calls_service.identity_for("u", None)


# ---------------------------------------------------------------------------
# Group calls: inviting people into one already in progress
# ---------------------------------------------------------------------------


async def _seed_group_call(users, *, joined: list[str]) -> Call:
    """A connected group call whose roster is exactly `joined`, all of them joined."""
    async with SessionLocal() as db:
        call = Call(
            org_id=users["org_a"].id,
            initiated_by=users["alice"].id,
            type=CallType.video,
            status=CallStatus.connected,
            is_group=True,
            room_name="",
            answered_at=now_utc(),
        )
        db.add(call)
        await db.flush()
        call.room_name = calls_service.room_name_for(call.id)
        for name in joined:
            db.add(CallParticipant(call_id=call.id, user_id=users[name].id, joined_at=now_utc()))
        await db.commit()
        await db.refresh(call)
        return call


async def test_invite_adds_participants_and_rings_them(two_orgs_with_users, captured_events):
    """The missing half of group calling: a call used to be a closed set fixed at the
    moment it started, because `_join` only admits users who already hold a roster row
    and those rows were created once from the conversation's membership."""
    users = two_orgs_with_users
    call = await _seed_group_call(users, joined=["alice"])
    captured_events.clear()

    result = await calls_service.invite_to_call(users["alice"], call.id, [users["bob"].id])

    assert result["invited"] == [str(users["bob"].id)]
    assert result["outcome"][str(users["bob"].id)] == "invited"

    # Bob is rung, and told the call is joinable.
    ring = events_of(captured_events, "call:incoming")
    assert len(ring) == 1
    assert ring[0]["is_group"] is True
    assert ring[0]["invited"] is True
    assert ring[0]["caller"]["id"] == str(users["alice"].id)
    assert recipients_of(captured_events, "call:incoming") == [str(users["bob"].id)]
    assert events_of(captured_events, "call:group_active")

    # Alice — already in the call — learns who is on the way, so the grid can show a
    # placeholder rather than someone appearing from nowhere.
    told = events_of(captured_events, "call:participants_invited")
    assert len(told) == 1
    assert told[0]["participants"][0]["id"] == str(users["bob"].id)
    assert recipients_of(captured_events, "call:participants_invited") == [str(users["alice"].id)]

    # And the roster row exists, so `_join` will now accept him through the ordinary path.
    async with SessionLocal() as db:
        assert await db.get(CallParticipant, (call.id, users["bob"].id)) is not None


async def test_invite_then_join_puts_them_in_the_call(two_orgs_with_users, captured_events):
    users = two_orgs_with_users
    call = await _seed_group_call(users, joined=["alice"])
    await calls_service.invite_to_call(users["alice"], call.id, [users["bob"].id])
    captured_events.clear()

    await calls_service._join(users["bob"], call.id)

    assert events_of(captured_events, "call:error") == []
    # Bob gets the roster; Alice is told he arrived.
    assert events_of(captured_events, "call:group_participants")
    joined = events_of(captured_events, "call:participant_joined")
    assert len(joined) == 1
    assert joined[0]["participant"]["id"] == str(users["bob"].id)


async def test_only_someone_in_the_call_may_invite(two_orgs_with_users, captured_events):
    """Otherwise knowing a call id is enough to make anyone's phone ring."""
    users = two_orgs_with_users
    call = await _seed_group_call(users, joined=["alice"])
    captured_events.clear()

    result = await calls_service.invite_to_call(users["bob"], call.id, [users["carol"].id])

    assert result == {"error": "not_a_participant"}
    assert captured_events == []


async def test_invite_refuses_a_direct_call(two_orgs_with_users, captured_events):
    """Adding a third party to a 1:1 would silently change what both people agreed to."""
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.connected)
    captured_events.clear()

    result = await calls_service.invite_to_call(users["alice"], call.id, [users["carol"].id])

    assert result == {"error": "not_a_group_call"}
    assert captured_events == []


async def test_invite_will_not_cross_orgs(two_orgs_with_users, captured_events):
    """A call must not become a reachability channel that skips tenant isolation."""
    users = two_orgs_with_users
    call = await _seed_group_call(users, joined=["alice"])
    captured_events.clear()

    result = await calls_service.invite_to_call(users["alice"], call.id, [users["carol"].id])

    assert result["invited"] == []
    assert result["outcome"][str(users["carol"].id)] == "different_org"
    assert events_of(captured_events, "call:incoming") == []
    async with SessionLocal() as db:
        assert await db.get(CallParticipant, (call.id, users["carol"].id)) is None


async def test_invite_reports_each_invitee_separately(two_orgs_with_users, captured_events):
    """A partial result must not read as a flat success — the UI has to be able to say
    which ones it could not reach."""
    users = two_orgs_with_users
    call = await _seed_group_call(users, joined=["alice"])
    captured_events.clear()

    result = await calls_service.invite_to_call(
        users["alice"], call.id, [users["bob"].id, users["carol"].id, users["alice"].id]
    )

    assert result["outcome"][str(users["bob"].id)] == "invited"
    assert result["outcome"][str(users["carol"].id)] == "different_org"
    assert result["outcome"][str(users["alice"].id)] == "already_invited"
    assert result["invited"] == [str(users["bob"].id)]


async def test_inviting_someone_who_left_rings_them_again(two_orgs_with_users, captured_events):
    """Getting somebody back into a call they dropped out of.

    Holding a roster row used to be enough to be reported `already_invited` and rung by
    nobody, which made a departed participant unreachable: the only way back in was the
    Join chip in the conversation, and someone who was invited from outside that
    conversation has no such chip."""
    users = two_orgs_with_users
    call = await _seed_group_call(users, joined=["alice", "bob"])
    await calls_service._leave(users["bob"], call.id)
    captured_events.clear()

    result = await calls_service.invite_to_call(users["alice"], call.id, [users["bob"].id])

    assert result["invited"] == [str(users["bob"].id)]
    assert result["outcome"][str(users["bob"].id)] == "invited"
    assert recipients_of(captured_events, "call:incoming") == [str(users["bob"].id)]
    # Not counted as gone any more, but not in the room either until he answers — that
    # is `_join`'s job, and the distinction is what stops the last-one-out check from
    # tearing the call down under someone who has been re-invited but not yet arrived.
    async with SessionLocal() as db:
        row = await db.get(CallParticipant, (call.id, users["bob"].id))
        assert row.left_at is None

    # And the re-invited person is not also told they invited themselves.
    assert recipients_of(captured_events, "call:participants_invited") == [str(users["alice"].id)]


async def test_declining_a_group_call_tells_the_others_and_does_not_end_it(
    two_orgs_with_users, captured_events
):
    """Declining a group call used to be a complete no-op.

    A group call is created `connected` (`initiate_group_call`), never `ringing`, so a
    decline fell straight through `_decline`'s status check: nobody was told, and the
    decliner kept a roster row marked as somebody still expected to arrive — which is
    what a "Ringing…" placeholder reads from, so it would have sat there for the rest of
    the call. And one person saying no must not end a call the others are in."""
    users = two_orgs_with_users
    call = await _seed_group_call(users, joined=["alice"])
    await calls_service.invite_to_call(users["alice"], call.id, [users["bob"].id])
    captured_events.clear()

    await calls_service._decline(users["bob"], call.id)

    told = events_of(captured_events, "call:participant_declined")
    assert len(told) == 1
    assert told[0]["participant_id"] == str(users["bob"].id)
    assert told[0]["participant"]["id"] == str(users["bob"].id)
    assert recipients_of(captured_events, "call:participant_declined") == [str(users["alice"].id)]
    # NOT a finalise: no `call:ended`, and the call row is still connected.
    assert events_of(captured_events, "call:ended") == []
    assert events_of(captured_events, "call:declined") == []
    async with SessionLocal() as db:
        assert (await db.get(Call, call.id)).status == CallStatus.connected
        # Marked gone, so nothing counts them as still on the way.
        assert (await db.get(CallParticipant, (call.id, users["bob"].id))).left_at is not None


async def test_someone_already_in_a_group_call_cannot_decline_it(two_orgs_with_users, captured_events):
    """That is a leave, and hanging up sends one. Treating it as a decline would mark a
    live participant as gone while their media was still in the room."""
    users = two_orgs_with_users
    call = await _seed_group_call(users, joined=["alice", "bob"])
    captured_events.clear()

    await calls_service._decline(users["bob"], call.id)

    assert captured_events == []
    async with SessionLocal() as db:
        assert (await db.get(CallParticipant, (call.id, users["bob"].id))).left_at is None


async def test_invite_refuses_once_the_call_is_over(two_orgs_with_users, captured_events):
    users = two_orgs_with_users
    call = await _seed_group_call(users, joined=["alice"])
    async with SessionLocal() as db:
        row = await db.get(Call, call.id)
        row.status = CallStatus.answered
        await db.commit()
    captured_events.clear()

    result = await calls_service.invite_to_call(users["alice"], call.id, [users["bob"].id])

    assert result == {"error": "not_joinable"}
    assert captured_events == []
