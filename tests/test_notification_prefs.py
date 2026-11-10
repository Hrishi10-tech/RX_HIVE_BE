"""Timed mutes, per-user notification preferences, and the push payloads they shape.

Push is sent by the server, so every rule here is one only the server can apply:
a client toggle that the send path never reads silences nothing (the original
is_muted shipped exactly that way). Each test drives the real send/call path and
captures what reaches dispatch_push_to_users.
"""

import contextlib
import datetime as dt

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, update

from app.db.models import Call, CallStatus, ConversationParticipant, User
from app.db.session import SessionLocal
from app.main import app
from app.services import calls as calls_service
from app.services import messaging as messaging_mod
from app.utils import now_utc
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


async def _group(client, name, member_ids) -> str:
    resp = await client.post(
        "/api/conversations/group", json={"name": name, "member_ids": [str(m) for m in member_ids]}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["_id"]


async def _send(client, conv_id, content) -> dict:
    resp = await client.post(
        f"/api/conversations/{conv_id}/messages", json={"content": content, "type": "text"}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _listed(client, conv_id) -> dict:
    resp = await client.get("/api/conversations")
    assert resp.status_code == 200, resp.text
    return next(c for c in resp.json()["data"] if c["_id"] == conv_id)


def _parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


async def _set_prefs(user_id, **prefs) -> None:
    async with SessionLocal() as db:
        await db.execute(update(User).where(User.id == user_id).values(notification_prefs=prefs))
        await db.commit()


async def _backdate_mute(conv_id, user_id) -> None:
    async with SessionLocal() as db:
        await db.execute(
            update(ConversationParticipant)
            .where(
                ConversationParticipant.conversation_id == conv_id,
                ConversationParticipant.user_id == user_id,
            )
            .values(muted_until=now_utc() - dt.timedelta(minutes=1))
        )
        await db.commit()


@pytest.fixture
def pushes(monkeypatch):
    """Every dispatch_push_to_users call as (sorted user ids, payload).

    Patched on its own module: both send paths import it lazily, so the name is
    looked up there at call time. Everyone also reads as OFFLINE, or nobody is a
    message-push candidate in the first place.
    """
    import app.services.push as push_mod

    sent: list[tuple[list[str], dict]] = []

    def _capture(user_ids, payload):
        sent.append((sorted(str(u) for u in user_ids), payload))

    async def _offline(user_ids):
        return {str(u): "offline" for u in user_ids}

    monkeypatch.setattr(push_mod, "dispatch_push_to_users", _capture)
    monkeypatch.setattr(messaging_mod.presence, "get_statuses", _offline)
    return sent


def _to(pushes, user) -> list[dict]:
    return [payload for uids, payload in pushes if str(user.id) in uids]


# ---------------------------------------------------------------------------
# Mute with a duration
# ---------------------------------------------------------------------------


async def test_mute_durations_set_and_serve_muted_until(client, two_orgs_with_users):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    listed = await _listed(client, conv)
    assert listed["is_muted"] is False and listed["muted_until"] is None

    for duration, span in (("8h", dt.timedelta(hours=8)), ("1w", dt.timedelta(weeks=1))):
        before = now_utc()
        resp = await client.put(f"/api/conversations/{conv}/mute", json={"duration": duration})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["is_muted"] is True
        until = _parse(body["muted_until"])
        assert before + span - dt.timedelta(seconds=5) <= until <= now_utc() + span
        # The conversation list carries the same pair.
        listed = await _listed(client, conv)
        assert listed["is_muted"] is True and listed["muted_until"] == body["muted_until"]

    resp = await client.put(f"/api/conversations/{conv}/mute", json={"duration": "always"})
    assert resp.json() == {"is_muted": True, "muted_until": None}
    listed = await _listed(client, conv)
    assert listed["is_muted"] is True and listed["muted_until"] is None

    resp = await client.put(f"/api/conversations/{conv}/mute", json={"duration": "off"})
    assert resp.json() == {"is_muted": False, "muted_until": None}
    assert (await _listed(client, conv))["is_muted"] is False

    # Unknown durations are refused, not guessed at.
    for bad in ("2d", "", 8, "ALWAYS"):
        resp = await client.put(f"/api/conversations/{conv}/mute", json={"duration": bad})
        assert resp.status_code == 422, (bad, resp.text)


async def test_no_body_toggle_is_unchanged_and_reads_the_effective_state(client, two_orgs_with_users):
    """The iOS and Android apps send no body; that must still be a plain toggle.

    And the toggle must read the EFFECTIVE mute: from a live timed mute it unmutes,
    from a lapsed one it mutes (always) — flipping the stored flag would "unmute" a
    chat the user already sees as unmuted.
    """
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    # An empty body with a JSON content type is still "no body".
    resp = await client.put(
        f"/api/conversations/{conv}/mute", content=b"", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"is_muted": True, "muted_until": None}
    assert (await client.put(f"/api/conversations/{conv}/mute")).json() == {
        "is_muted": False,
        "muted_until": None,
    }

    # Live timed mute -> toggle unmutes.
    await client.put(f"/api/conversations/{conv}/mute", json={"duration": "8h"})
    assert (await client.put(f"/api/conversations/{conv}/mute")).json() == {
        "is_muted": False,
        "muted_until": None,
    }

    # Lapsed timed mute -> served unmuted, and the toggle mutes (always).
    await client.put(f"/api/conversations/{conv}/mute", json={"duration": "8h"})
    await _backdate_mute(conv, users["alice"].id)
    listed = await _listed(client, conv)
    assert listed["is_muted"] is False and listed["muted_until"] is None
    assert (await client.put(f"/api/conversations/{conv}/mute")).json() == {
        "is_muted": True,
        "muted_until": None,
    }


async def test_timed_mute_silences_push_until_it_lapses(client, two_orgs_with_users, pushes):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    async with _client_for("bob@a.com") as bob:
        resp = await bob.put(f"/api/conversations/{conv}/mute", json={"duration": "8h"})
        assert resp.json()["is_muted"] is True

    await _send(client, conv, "while muted")
    assert _to(pushes, users["bob"]) == []

    # Lapsed: the flag is still set in the row, but it no longer counts.
    await _backdate_mute(conv, users["bob"].id)
    await _send(client, conv, "after it lapsed")
    assert [p["body"] for p in _to(pushes, users["bob"])] == ["after it lapsed"]


# ---------------------------------------------------------------------------
# Preferences API
# ---------------------------------------------------------------------------


async def test_preferences_default_and_partial_update(client, two_orgs_with_users):
    await login(client, "alice@a.com")
    everything_on = {"show_preview": True, "messages": True, "groups": True, "calls": True}

    resp = await client.get("/api/notifications/preferences")
    assert resp.status_code == 200, resp.text
    assert resp.json() == everything_on

    resp = await client.put("/api/notifications/preferences", json={"show_preview": False})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {**everything_on, "show_preview": False}

    # Partial: a second update keeps the first.
    resp = await client.put("/api/notifications/preferences", json={"calls": False, "groups": False})
    expected = {"show_preview": False, "messages": True, "groups": False, "calls": False}
    assert resp.json() == expected
    assert (await client.get("/api/notifications/preferences")).json() == expected

    # An empty update changes nothing and still answers with the full object.
    assert (await client.put("/api/notifications/preferences", json={})).json() == expected

    # Per user: Bob is untouched.
    async with _client_for("bob@a.com") as bob:
        assert (await bob.get("/api/notifications/preferences")).json() == everything_on


async def test_preferences_are_validated(client, two_orgs_with_users):
    await login(client, "alice@a.com")
    for bad in ({"messages": "false"}, {"messages": 0}, {"messages": None}, {"sms": True}, ["calls"]):
        resp = await client.put("/api/notifications/preferences", json=bad)
        assert resp.status_code == 422, (bad, resp.text)
    resp = await client.get("/api/notifications/preferences")
    assert all(resp.json().values()), "a refused update must not have written anything"


async def test_preferences_require_authentication(client):
    assert (await client.get("/api/notifications/preferences")).status_code == 401
    resp = await client.put("/api/notifications/preferences", json={"calls": False})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Message push payloads
# ---------------------------------------------------------------------------


async def test_direct_message_push_keeps_sender_title(client, two_orgs_with_users, pushes):
    users = two_orgs_with_users
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)

    await _send(client, conv, "bed 4 needs review")
    [payload] = _to(pushes, users["bob"])
    assert payload == {
        "title": "Alice",
        "body": "bed 4 needs review",
        "tag": conv,
        "kind": "message",
        "conversation_id": conv,
        "url": f"/chat?c={conv}",
    }


async def test_group_push_is_titled_with_the_group_and_names_the_sender(client, two_orgs_with_users, pushes):
    users = two_orgs_with_users
    erin = await make_user("erin@a.com", org_id=users["org_a"].id, display_name="Erin")
    await login(client, "alice@a.com")
    group = await _group(client, "Ward 7", [users["bob"].id, erin.id])

    pushes.clear()
    await _send(client, group, "handover at 6")
    assert len(pushes) == 1
    uids, payload = pushes[0]
    assert uids == sorted([str(users["bob"].id), str(erin.id)])
    assert payload["title"] == "Ward 7"
    assert payload["body"] == "Alice: handover at 6"
    assert payload["kind"] == "message"
    assert payload["tag"] == group and payload["conversation_id"] == group
    assert payload["url"] == f"/chat?c={group}"


async def test_show_preview_off_hides_the_text_only_for_that_user(client, two_orgs_with_users, pushes):
    users = two_orgs_with_users
    erin = await make_user("erin@a.com", org_id=users["org_a"].id, display_name="Erin")
    await _set_prefs(users["bob"].id, show_preview=False)
    await login(client, "alice@a.com")
    group = await _group(client, "Ward 7", [users["bob"].id, erin.id])
    direct = await _direct(client, users["bob"].id)

    pushes.clear()
    await _send(client, group, "patient name and bed")
    [bob_payload] = _to(pushes, users["bob"])
    [erin_payload] = _to(pushes, erin)
    # Group: title stays the group, body is just "New message" — not even the sender.
    assert bob_payload["title"] == "Ward 7" and bob_payload["body"] == "New message"
    assert erin_payload["body"] == "Alice: patient name and bed"
    assert bob_payload["tag"] == group and bob_payload["kind"] == "message"

    pushes.clear()
    await _send(client, direct, "private detail")
    [payload] = _to(pushes, users["bob"])
    assert payload["title"] == "Alice" and payload["body"] == "New message"
    assert "private detail" not in str(payload)


async def test_messages_and_groups_preferences_gate_their_own_chat_kind(client, two_orgs_with_users, pushes):
    users = two_orgs_with_users
    erin = await make_user("erin@a.com", org_id=users["org_a"].id, display_name="Erin")
    await login(client, "alice@a.com")
    group = await _group(client, "Ward 7", [users["bob"].id, erin.id])
    direct = await _direct(client, users["bob"].id)

    # messages=false: direct chats silent, groups still push.
    await _set_prefs(users["bob"].id, messages=False)
    pushes.clear()
    await _send(client, direct, "direct one")
    assert _to(pushes, users["bob"]) == []
    await _send(client, group, "group one")
    assert [p["body"] for p in _to(pushes, users["bob"])] == ["Alice: group one"]

    # groups=false: the reverse. Erin, with defaults, is unaffected throughout.
    await _set_prefs(users["bob"].id, groups=False)
    pushes.clear()
    await _send(client, group, "group two")
    assert _to(pushes, users["bob"]) == []
    assert [p["body"] for p in _to(pushes, erin)] == ["Alice: group two"]
    await _send(client, direct, "direct two")
    assert [p["body"] for p in _to(pushes, users["bob"])] == ["direct two"]


async def test_a_failing_preference_lookup_does_not_fail_the_send(client, two_orgs_with_users, monkeypatch):
    """The message is committed and broadcast before push; push must not undo the 200."""
    users = two_orgs_with_users

    async def _boom(*_a, **_k):
        raise RuntimeError("push exploded")

    async def _offline(user_ids):
        return {str(u): "offline" for u in user_ids}

    monkeypatch.setattr(messaging_mod, "_dispatch_message_push", _boom)
    monkeypatch.setattr(messaging_mod.presence, "get_statuses", _offline)
    await login(client, "alice@a.com")
    conv = await _direct(client, users["bob"].id)
    await _send(client, conv, "still delivered")


# ---------------------------------------------------------------------------
# Call pushes
# ---------------------------------------------------------------------------


@pytest.fixture
def call_events(monkeypatch):
    sent: list[tuple[list[str], dict]] = []

    async def _capture(user_ids, event):
        sent.append(([str(u) for u in user_ids], event))

    monkeypatch.setattr(calls_service, "publish_to_users", _capture)
    return sent


async def _ring(users, call_type="voice") -> str:
    """Alice rings Bob (who has no socket, so the ring push is due). Returns call id."""
    await calls_service.initiate_direct_call(users["alice"], users["bob"].id, call_type, None)
    async with SessionLocal() as db:
        call = (
            await db.execute(
                select(Call)
                .where(Call.initiated_by == users["alice"].id)
                .order_by(Call.started_at.desc())
                .limit(1)
            )
        ).scalar_one()
        return str(call.id)


async def _call_status(call_id) -> CallStatus:
    async with SessionLocal() as db:
        return (await db.get(Call, call_id)).status


async def test_incoming_call_push_respects_the_calls_preference(two_orgs_with_users, pushes, call_events):
    users = two_orgs_with_users
    call_id = await _ring(users)
    [payload] = _to(pushes, users["bob"])
    assert payload["kind"] == "call" and payload["tag"] == f"call:{call_id}"

    # Off: no push, but the call still rings (frame + ringing status).
    await calls_service._cancel(users["alice"], call_id)
    await _set_prefs(users["bob"].id, calls=False)
    pushes.clear()
    call_events.clear()
    call_id = await _ring(users)
    assert _to(pushes, users["bob"]) == []
    assert any(e.get("type") == "call:incoming" for _, e in call_events)
    assert await _call_status(call_id) == CallStatus.ringing


@pytest.mark.parametrize(
    ("call_type", "body"), [("voice", "Missed voice call"), ("video", "Missed video call")]
)
async def test_ring_timeout_pushes_a_missed_call_under_the_ring_tag(
    two_orgs_with_users, pushes, call_events, call_type, body
):
    users = two_orgs_with_users
    call_id = await _ring(users, call_type)
    pushes.clear()

    await calls_service._ring_timeout(call_id)
    assert await _call_status(call_id) == CallStatus.missed

    [payload] = _to(pushes, users["bob"])
    conv_id = payload["conversation_id"]
    assert conv_id
    assert payload == {
        "title": "Alice",
        "body": body,
        "tag": f"call:{call_id}",  # same tag as the ring: replaces it on the device
        "kind": "missed_call",
        "call_id": call_id,
        "conversation_id": conv_id,
        "url": f"/chat?c={conv_id}",
    }
    assert _to(pushes, users["alice"]) == [], "the caller is never told they missed their own call"


async def test_caller_cancel_before_answer_is_a_missed_call(two_orgs_with_users, pushes, call_events):
    users = two_orgs_with_users
    call_id = await _ring(users)
    pushes.clear()

    await calls_service._cancel(users["alice"], call_id)
    assert await _call_status(call_id) == CallStatus.cancelled
    [payload] = _to(pushes, users["bob"])
    assert payload["kind"] == "missed_call" and payload["tag"] == f"call:{call_id}"
    assert payload["body"] == "Missed voice call"


async def test_declined_call_is_not_a_missed_call(two_orgs_with_users, pushes, call_events):
    users = two_orgs_with_users
    call_id = await _ring(users)
    pushes.clear()

    await calls_service._decline(users["bob"], call_id)
    assert await _call_status(call_id) == CallStatus.declined
    assert pushes == []


async def test_missed_call_push_respects_the_calls_preference(two_orgs_with_users, pushes, call_events):
    users = two_orgs_with_users
    await _set_prefs(users["bob"].id, calls=False)
    call_id = await _ring(users)
    await calls_service._ring_timeout(call_id)
    assert await _call_status(call_id) == CallStatus.missed
    assert pushes == []


async def test_missed_call_push_failure_does_not_break_the_call(
    two_orgs_with_users, call_events, monkeypatch
):
    import app.services.push as push_mod

    users = two_orgs_with_users
    call_id = await _ring(users)

    def _boom(*_a, **_k):
        raise RuntimeError("push service down")

    monkeypatch.setattr(push_mod, "dispatch_push_to_users", _boom)
    await calls_service._ring_timeout(call_id)  # must not raise
    assert await _call_status(call_id) == CallStatus.missed
    missed = [e for _, e in call_events if e.get("type") == "call:missed"]
    assert len(missed) == 2, "both sides are still told over the socket"
