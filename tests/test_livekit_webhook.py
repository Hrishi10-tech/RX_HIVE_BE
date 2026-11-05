"""The LiveKit webhook: registration, signature verification, and the reconciliation
it drives.

The bug these guard: `webhook_router` was declared in `api/calls.py` but never
passed to `app.include_router`, so `POST /api/livekit/webhook` 404'd. Nothing
failed loudly — calls connected and messaging was untouched — but every
SFU-driven path was dead. A client killed mid-call left `call_history.status`
'connected' forever, because `room_finished` is the only thing that finalizes a
call nobody explicitly ended.

`test_webhook_route_is_registered` is the regression test for exactly that: it
asserts the route is reachable at all, independently of whether the payload is
any good.
"""

import base64
import datetime as dt
import hashlib
import json
import uuid

import jwt
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Call, CallParticipant, CallStatus, CallType
from app.db.session import SessionLocal
from app.services.calls import room_name_for
from app.utils import now_utc


def _sign(body: str, *, api_key: str | None = None, api_secret: str | None = None) -> str:
    """Produce the `Authorization` value LiveKit sends with a webhook.

    Mirrors `livekit.api.WebhookReceiver.receive`, which requires an HS256 JWT
    whose `iss` is the API key and whose `sha256` claim is the **base64** digest of
    the raw body. Built here with PyJWT rather than the LiveKit SDK because the SDK
    exposes only the receiving half — there is no sender to borrow.
    """
    settings = get_settings()
    digest = hashlib.sha256(body.encode()).digest()
    now = dt.datetime.now(dt.UTC)
    return jwt.encode(
        {
            "iss": api_key or settings.livekit_api_key,
            "sha256": base64.b64encode(digest).decode(),
            "nbf": now,
            "exp": now + dt.timedelta(minutes=5),
        },
        api_secret or settings.livekit_api_secret,
        algorithm="HS256",
    )


async def _post_webhook(client, payload: dict, **sign_kwargs):
    """Send a webhook exactly as the SFU would: raw JSON body + signed token.

    `json=` cannot be used — the signature covers the exact bytes, and letting
    httpx re-serialize would change them and trip the hash check.
    """
    body = json.dumps(payload)
    return await client.post(
        "/api/livekit/webhook",
        content=body,
        headers={
            "Authorization": _sign(body, **sign_kwargs),
            "Content-Type": "application/json",
        },
    )


async def _seed_call(users, status: CallStatus = CallStatus.connected) -> Call:
    async with SessionLocal() as db:
        call = Call(
            org_id=users["org_a"].id,
            initiated_by=users["alice"].id,
            type=CallType.video,
            status=status,
            room_name="",
            answered_at=now_utc() if status == CallStatus.connected else None,
        )
        db.add(call)
        await db.flush()
        call.room_name = room_name_for(call.id)
        db.add_all(
            [
                CallParticipant(call_id=call.id, user_id=users["alice"].id),
                CallParticipant(call_id=call.id, user_id=users["bob"].id),
            ]
        )
        await db.commit()
        await db.refresh(call)
        return call


async def _reload(call_id) -> Call:
    async with SessionLocal() as db:
        return (await db.execute(select(Call).where(Call.id == call_id))).scalar_one()


# ---------------------------------------------------------------------------
# Registration — the actual bug
# ---------------------------------------------------------------------------


async def test_webhook_route_is_registered(client):
    """The route must exist. Asserted via the router table AND a live request, so
    this fails if `include_router(calls.webhook_router)` is ever dropped again."""
    from app.main import app

    paths = {route.path for route in app.routes}
    assert "/api/livekit/webhook" in paths

    # And it is genuinely reachable — not shadowed, not 404, not 405.
    resp = await client.post("/api/livekit/webhook", content="{}", headers={"Authorization": "junk"})
    assert resp.status_code == 200


async def test_webhook_does_not_require_a_session(client):
    """The SFU has no cookie and cannot send the CSRF header. A logged-out client
    must still be able to reach it, or the mount is useless in production."""
    # `client` has the CSRF header from conftest; strip it to simulate the SFU.
    resp = await client.post(
        "/api/livekit/webhook",
        content="{}",
        headers={"Authorization": "junk", "X-Requested-With": ""},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"message": "ignored"}


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------


async def test_unsigned_and_missigned_payloads_are_ignored(client, two_orgs_with_users):
    users = two_orgs_with_users
    call = await _seed_call(users)
    payload = {"event": "room_finished", "room": {"name": call.room_name}}

    # No token at all.
    resp = await client.post("/api/livekit/webhook", content=json.dumps(payload))
    assert resp.status_code == 200
    assert resp.json() == {"message": "ignored"}

    # Signed with the wrong secret.
    resp = await _post_webhook(client, payload, api_secret="not-the-real-secret-at-all-32chars")
    assert resp.status_code == 200
    assert resp.json() == {"message": "ignored"}

    # Signed with the wrong key (issuer mismatch — the api_key drift footgun).
    resp = await _post_webhook(client, payload, api_key="some-other-key")
    assert resp.status_code == 200
    assert resp.json() == {"message": "ignored"}

    # None of those may have touched the call.
    assert (await _reload(call.id)).status == CallStatus.connected


async def test_body_tampering_is_rejected(client, two_orgs_with_users):
    """The token's sha256 claim covers the body, so swapping the body after signing
    must fail even though the signature itself is valid."""
    users = two_orgs_with_users
    call = await _seed_call(users)

    honest = json.dumps({"event": "room_started", "room": {"name": call.room_name}})
    tampered = json.dumps({"event": "room_finished", "room": {"name": call.room_name}})

    resp = await client.post(
        "/api/livekit/webhook",
        content=tampered,
        headers={"Authorization": _sign(honest), "Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"message": "ignored"}
    assert (await _reload(call.id)).status == CallStatus.connected


# ---------------------------------------------------------------------------
# room_finished — the path that was most costly to lose
# ---------------------------------------------------------------------------


async def test_room_finished_finalizes_a_connected_call(client, two_orgs_with_users):
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.connected)

    resp = await _post_webhook(client, {"event": "room_finished", "room": {"name": call.room_name}})
    assert resp.status_code == 200
    assert resp.json() == {"message": "ok"}

    finished = await _reload(call.id)
    assert finished.status == CallStatus.answered
    assert finished.ended_at is not None


async def test_room_finished_leaves_a_ringing_call_alone(client, two_orgs_with_users):
    """Only a *connected* call is finalized as answered. A room that ends while the
    call is still ringing was never answered, and the ring-timeout path owns it —
    marking it 'answered' here would invent a conversation that did not happen."""
    users = two_orgs_with_users
    call = await _seed_call(users, status=CallStatus.ringing)

    resp = await _post_webhook(client, {"event": "room_finished", "room": {"name": call.room_name}})
    assert resp.status_code == 200
    assert (await _reload(call.id)).status == CallStatus.ringing


# ---------------------------------------------------------------------------
# Participant tracking
# ---------------------------------------------------------------------------


async def test_participant_joined_stamps_joined_at(client, two_orgs_with_users):
    users = two_orgs_with_users
    call = await _seed_call(users)

    resp = await _post_webhook(
        client,
        {
            "event": "participant_joined",
            "room": {"name": call.room_name},
            "participant": {"identity": str(users["bob"].id)},
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"message": "ok"}

    async with SessionLocal() as db:
        member = await db.get(CallParticipant, (call.id, users["bob"].id))
        assert member.joined_at is not None
        assert member.left_at is None


async def test_participant_left_stamps_left_at(client, two_orgs_with_users):
    users = two_orgs_with_users
    call = await _seed_call(users)

    await _post_webhook(
        client,
        {
            "event": "participant_left",
            "room": {"name": call.room_name},
            "participant": {"identity": str(users["bob"].id)},
        },
    )

    async with SessionLocal() as db:
        member = await db.get(CallParticipant, (call.id, users["bob"].id))
        assert member.left_at is not None


async def test_participant_events_for_a_non_member_are_ignored(client, two_orgs_with_users):
    """carol is in another org and not on this call. A forged identity must not
    create a participant row or otherwise mutate the call."""
    users = two_orgs_with_users
    call = await _seed_call(users)

    resp = await _post_webhook(
        client,
        {
            "event": "participant_joined",
            "room": {"name": call.room_name},
            "participant": {"identity": str(users["carol"].id)},
        },
    )
    assert resp.status_code == 200

    async with SessionLocal() as db:
        assert await db.get(CallParticipant, (call.id, users["carol"].id)) is None
        rows = (
            (await db.execute(select(CallParticipant).where(CallParticipant.call_id == call.id)))
            .scalars()
            .all()
        )
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Payloads that must not raise
# ---------------------------------------------------------------------------


async def test_unrelated_rooms_and_unknown_events_are_harmless(client, two_orgs_with_users):
    """Everything here is well-signed and must be shrugged off with a 200. A 500
    would make LiveKit retry, and a webhook that 500-loops is worse than one that
    does nothing."""
    users = two_orgs_with_users
    call = await _seed_call(users)

    cases = [
        # A room that is not one of ours (LiveKit may host others).
        {"event": "room_finished", "room": {"name": "some-other-app-room"}},
        # Our prefix but not a uuid.
        {"event": "room_finished", "room": {"name": "call_not-a-uuid"}},
        # A well-formed uuid for a call that does not exist.
        {"event": "room_finished", "room": {"name": f"call_{uuid.uuid4()}"}},
        # No room at all — proto3 defaults this to an empty name.
        {"event": "room_finished"},
        # An event type this handler does not care about.
        {"event": "track_published", "room": {"name": call.room_name}},
        # A participant event with no participant.
        {"event": "participant_joined", "room": {"name": call.room_name}},
        # Empty object.
        {},
    ]
    for payload in cases:
        resp = await _post_webhook(client, payload)
        assert resp.status_code == 200, f"{payload} -> {resp.status_code} {resp.text}"

    # And the real call is untouched by all of it.
    assert (await _reload(call.id)).status == CallStatus.connected
