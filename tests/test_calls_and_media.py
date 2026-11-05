"""Calls (LiveKit token minting, history, isolation) and media auth."""

import contextlib
import io
import uuid

import jwt
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.db.models import Call, CallParticipant, CallStatus, CallType, Conversation, Message
from app.db.session import SessionLocal
from app.main import app
from app.services.calls import room_name_for, user_id_from_identity
from app.utils import now_utc
from tests.conftest import CSRF, login


def _png_bytes() -> bytes:
    import struct
    import zlib

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes([16, 185, 129] * 8) for _ in range(8))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


@contextlib.asynccontextmanager
async def _client_for(email):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c.headers.update(CSRF)
        await login(c, email)
        yield c


async def _seed_connected_call(users) -> str:
    """Create a connected 1:1 call between alice and bob directly in the DB."""

    async with SessionLocal() as db:
        call = Call(
            org_id=users["org_a"].id,
            initiated_by=users["alice"].id,
            type=CallType.video,
            status=CallStatus.connected,
            room_name="",
            answered_at=now_utc(),
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
        return str(call.id)


async def test_call_token_is_scoped_and_only_for_participants(client, two_orgs_with_users):
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)

    async with _client_for("alice@a.com") as alice:
        resp = await alice.post(f"/api/calls/{call_id}/token")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["room"] == f"call_{call_id}"
        settings = get_settings()
        claims = jwt.decode(body["token"], settings.livekit_api_secret, algorithms=["HS256"])
        # token grants join to exactly this room
        assert claims["video"]["room"] == f"call_{call_id}"
        assert claims["video"]["roomJoin"] is True
        # The identity is `{user_id}#{device}`, not the bare user id: a LiveKit
        # identity must be unique per connection or the SFU evicts the earlier
        # client as a duplicate. It still resolves back to the user.
        assert user_id_from_identity(claims["sub"]) == str(users["alice"].id)
        assert claims["sub"] == body["identity"]

    # a foreign-org user cannot mint a token for someone else's call
    async with _client_for("carol@b.com") as carol:
        resp = await carol.post(f"/api/calls/{call_id}/token")
        assert resp.status_code == 404


async def test_two_devices_of_one_user_get_distinct_room_identities(client, two_orgs_with_users):
    """The regression that made "the call connects then goes silent" reproducible.

    Both devices of a user used to be issued the bare user id as their LiveKit
    identity, so whichever connected second evicted the first as a duplicate — with
    no error surfaced anywhere. Distinct identities make a second client additive.
    """
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)

    async with _client_for("alice@a.com") as alice:
        phone = await alice.post(f"/api/calls/{call_id}/token", json={"device_id": "phone"})
        laptop = await alice.post(f"/api/calls/{call_id}/token", json={"device_id": "laptop"})

    assert phone.status_code == 200 and laptop.status_code == 200
    a, b = phone.json()["identity"], laptop.json()["identity"]
    assert a != b
    assert user_id_from_identity(a) == user_id_from_identity(b) == str(users["alice"].id)


async def test_ringing_call_is_not_joinable_by_the_callee_before_accepting(client, two_orgs_with_users):
    """A token while merely ringing would publish the callee into the room without
    them ever answering — the caller would hear a call they believed was ringing."""
    users = two_orgs_with_users
    async with SessionLocal() as db:
        call = Call(
            org_id=users["org_a"].id,
            initiated_by=users["alice"].id,
            type=CallType.voice,
            status=CallStatus.ringing,
            room_name="",
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
        call_id = str(call.id)

    async with _client_for("bob@a.com") as bob:
        assert (await bob.post(f"/api/calls/{call_id}/token")).status_code == 400
    # The caller may pre-join their own ringing call, so ringback and the first
    # audio packet do not wait on a second round trip after the answer.
    async with _client_for("alice@a.com") as alice:
        assert (await alice.post(f"/api/calls/{call_id}/token")).status_code == 200


async def test_active_call_lets_a_reconnecting_client_recover_its_call(client, two_orgs_with_users):
    """The recovery path for every frame lost while a socket was down."""
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)

    async with _client_for("bob@a.com") as bob:
        body = (await bob.get("/api/calls/active")).json()
        assert body["call"]["call_id"] == call_id
        assert body["call"]["status"] == "connected"
        assert body["call"]["is_initiator"] is False
        # Not joined yet, so a resuming client puts itself back on the ringer /
        # join affordance rather than straight into a live room.
        assert body["call"]["self_joined"] is False

    # Someone with no live call gets an explicit null, not a 404.
    async with _client_for("carol@b.com") as carol:
        assert (await carol.get("/api/calls/active")).json() == {"call": None}


async def test_active_call_reports_self_joined_after_taking_a_token(client, two_orgs_with_users):
    users = two_orgs_with_users
    call_id = await _seed_connected_call(users)

    async with _client_for("bob@a.com") as bob:
        await bob.post(f"/api/calls/{call_id}/token", json={"device_id": "phone"})
        call = (await bob.get("/api/calls/active")).json()["call"]
        assert call["self_joined"] is True
        assert str(users["bob"].id) in call["joined"]


async def test_call_history_only_shows_own_calls(client, two_orgs_with_users):
    users = two_orgs_with_users
    await _seed_connected_call(users)

    async with _client_for("alice@a.com") as alice:
        resp = await alice.get("/api/calls/history")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    async with _client_for("carol@b.com") as carol:
        resp = await carol.get("/api/calls/history")
        assert resp.json()["total"] == 0


async def test_media_upload_and_authenticated_serving(client, two_orgs_with_users):
    users = two_orgs_with_users
    async with _client_for("alice@a.com") as alice:
        # upload an image
        files = {"file": ("pic.png", io.BytesIO(_png_bytes()), "image/png")}
        resp = await alice.post("/api/upload", files=files)
        assert resp.status_code == 200, resp.text
        up = resp.json()
        assert up["file_type"] == "image" and up["file_url"].startswith("/api/media/up/")

        # create a direct conversation + send the image
        conv = (
            await alice.post("/api/conversations/direct", json={"participant_id": str(users["bob"].id)})
        ).json()["_id"]
        msg = await alice.post(
            f"/api/conversations/{conv}/messages",
            json={"content": "", "type": "image", "media_url": up["file_url"], "temp_id": "m1"},
        )
        assert msg.status_code == 200, msg.text
        media_url = msg.json()["media_url"]
        assert media_url.startswith("/api/media/")

        # owner/participant gets a redirect to a presigned URL (not the bytes inline)
        resp = await alice.get(media_url, follow_redirects=False)
        assert resp.status_code == 307
        assert "X-Amz-Signature" in resp.headers["location"]

    # a foreign-org user cannot fetch the attachment
    async with _client_for("carol@b.com") as carol:
        resp = await carol.get(media_url, follow_redirects=False)
        assert resp.status_code == 404

    # anonymous fetch is rejected
    anon = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    anon.headers.update(CSRF)
    async with anon:
        resp = await anon.get(media_url, follow_redirects=False)
        assert resp.status_code == 401


async def test_upload_accepts_any_extension(client, two_orgs_with_users):
    """There is no allow-list any more: every format uploads.

    Inverted from what it asserted before, which was that `.exe` 400s. Kept
    rather than deleted because the replacement contract is worth pinning: an
    unrecognised extension is a DOCUMENT, not a rejection, and that is what puts
    it in the generic file card instead of nowhere.
    """
    async with _client_for("alice@a.com") as alice:
        for name in ("tool.exe", "design.psd", "plan.dwg", "archive.rar", "scan.dcm"):
            resp = await alice.post("/api/upload", files={"file": (name, io.BytesIO(b"x" * 32))})
            assert resp.status_code == 200, f"{name}: {resp.text}"
            assert resp.json()["file_type"] == "document", name


async def test_unknown_types_are_stored_as_octet_stream(client, two_orgs_with_users):
    """The safety property that replaced the allow-list.

    Media is served from a SAME-ORIGIN path with Content-Disposition: inline, so
    a stored content type the browser renders is stored XSS on the app's own
    origin — an uploaded .html or .svg would execute as us. Everything outside
    MIME_BY_EXT must stay application/octet-stream so the browser downloads it.

    This is the test that fails if someone "improves" the content-type lookup
    into mimetypes.guess_type, which maps .html to text/html.
    """
    async with _client_for("alice@a.com") as alice:
        for name in ("page.html", "vector.svg", "script.js"):
            resp = await alice.post("/api/upload", files={"file": (name, io.BytesIO(b"<h1>hi</h1>"))})
            assert resp.status_code == 200, f"{name}: {resp.text}"
            assert resp.json()["mime_type"] == "application/octet-stream", name


async def test_upload_still_refuses_past_the_ceiling(client, two_orgs_with_users, monkeypatch):
    """One limit for every type now, but there is still a limit.

    This only ever asserted that the constant equalled 2 GB, so deleting the size
    check from the upload route entirely would not have failed it. The ceiling is
    lowered for the duration instead, which exercises the real rejection branch
    without a test moving two gigabytes.
    """
    from app.services import storage

    # The shipped ceiling is the deliberate one, and it comes from the setting
    # rather than a literal, so RXHIVE_MAX_UPLOAD_BYTES actually moves it.
    assert storage.MAX_UPLOAD_BYTES == get_settings().max_upload_bytes
    assert get_settings().max_upload_bytes == 2 * 1024 * 1024 * 1024
    assert storage.THUMBNAIL_SOURCE_LIMIT < storage.MAX_UPLOAD_BYTES

    # Lowered for the duration so the real rejection branch runs without a test
    # moving two gigabytes.
    monkeypatch.setattr(storage, "MAX_UPLOAD_BYTES", 32)
    async with _client_for("alice@a.com") as alice:
        resp = await alice.post("/api/upload", files={"file": ("big.bin", io.BytesIO(b"x" * 64))})
        assert resp.status_code == 400
        assert "large" in resp.json()["detail"].lower()
        # Just under the ceiling still succeeds — the branch discriminates.
        ok = await alice.post("/api/upload", files={"file": ("small.bin", io.BytesIO(b"x" * 16))})
        assert ok.status_code == 200, ok.text


async def test_voice_note_duration_round_trip(client, two_orgs_with_users):
    """A voice note's length is stored and served, and .weba classifies as audio.

    `.webm` is deliberately VIDEO in storage.py, so the recorder names its Opus
    fallback `.weba`. If that ever stops classifying as audio, voice notes silently
    render in the video bubble.
    """
    from app.services import storage

    assert storage.classify(".m4a") == "audio"
    assert storage.classify(".weba") == "audio", ".weba must be audio, not video"
    assert storage.classify(".webm") == "video", ".webm stays video by design"
    assert storage.MIME_BY_EXT[".weba"] == "audio/webm"

    users = two_orgs_with_users
    await login(client, "alice@a.com")
    resp = await client.post("/api/conversations/direct", json={"participant_id": str(users["bob"].id)})
    assert resp.status_code == 200, resp.text
    conv = resp.json()["_id"]

    resp = await client.post(
        "/api/upload",
        files={"file": ("voice-123.m4a", io.BytesIO(b"\x00\x00\x00\x20ftypM4A " + b"x" * 64), "audio/mp4")},
    )
    assert resp.status_code == 200, resp.text
    up = resp.json()
    assert up["file_type"] == "audio", up

    resp = await client.post(
        f"/api/conversations/{conv}/messages",
        json={
            "content": "",
            "type": "audio",
            "media_url": up["file_url"],
            "duration": 12.4,
        },
    )
    assert resp.status_code == 200, resp.text
    sent = resp.json()
    assert sent["type"] == "audio"
    assert sent["duration"] == 12.4, sent
    assert sent["attachments"][0]["duration"] == 12.4

    # And it survives a reload, which is the point of storing it at all.
    resp = await client.get(f"/api/conversations/{conv}/messages")
    loaded = [m for m in resp.json()["messages"] if m["_id"] == sent["_id"]][0]
    assert loaded["duration"] == 12.4


async def _seed_ringing_call(users) -> str:
    """An inbound call to bob that is still ringing — the badge's live case."""
    async with SessionLocal() as db:
        call = Call(
            org_id=users["org_a"].id,
            initiated_by=users["alice"].id,
            type=CallType.voice,
            status=CallStatus.ringing,
            room_name="",
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
        return str(call.id)


async def test_mark_seen_leaves_calls_that_are_still_ringing_unseen(client, two_orgs_with_users):
    """Opening the Calls tab mid-ring must not pre-read the call that follows.

    CallParticipant rows exist from initiation, so mark-seen used to stamp seen_at
    on a call that was still ringing. When the ring then timed out into `missed`,
    missed-count still skipped it — the one call the user most needed flagged was
    the one silently marked read.
    """
    users = two_orgs_with_users
    call_id = await _seed_ringing_call(users)

    async with _client_for("bob@a.com") as bob:
        assert (await bob.post("/api/calls/mark-seen")).status_code == 200

        # The ring times out into a missed call, exactly as _ring_timeout does.
        async with SessionLocal() as db:
            call = await db.get(Call, uuid.UUID(call_id))
            call.status = CallStatus.missed
            await db.commit()

        resp = await bob.get("/api/calls/missed-count")
        assert resp.status_code == 200, resp.text
        assert resp.json()["count"] == 1, "mark-seen swallowed a call that had not happened yet"


async def test_mark_seen_still_clears_calls_that_have_finished(client, two_orgs_with_users):
    """The other half of the contract: settled calls are cleared as before."""
    users = two_orgs_with_users
    call_id = await _seed_ringing_call(users)
    async with SessionLocal() as db:
        call = await db.get(Call, uuid.UUID(call_id))
        call.status = CallStatus.missed
        await db.commit()

    async with _client_for("bob@a.com") as bob:
        assert (await bob.get("/api/calls/missed-count")).json()["count"] == 1
        assert (await bob.post("/api/calls/mark-seen")).status_code == 200
        assert (await bob.get("/api/calls/missed-count")).json()["count"] == 0


async def test_every_per_asset_route_404s_for_a_non_member_and_a_tombstoned_message(
    client, two_orgs_with_users
):
    """The attachment authorisation gate, on all four routes that use it.

    _member_attachment resolves attachment -> message -> membership. It used to do
    that as three sequential db.get calls and is now a single join, so the three 404
    conditions it encodes are worth pinning rather than trusting: the existing media
    test only covered a non-member against the bytes route, leaving /thumb, /meta and
    /page/{n} — and the tombstone condition entirely — unasserted.

    All three failures must stay indistinguishable from one another: 404 everywhere,
    never a 403, so the endpoint cannot be used to prove an attachment exists.
    """
    users = two_orgs_with_users
    async with _client_for("alice@a.com") as alice:
        files = {"file": ("pic.png", io.BytesIO(_png_bytes()), "image/png")}
        up = (await alice.post("/api/upload", files=files)).json()
        conv = (
            await alice.post("/api/conversations/direct", json={"participant_id": str(users["bob"].id)})
        ).json()["_id"]
        msg = (
            await alice.post(
                f"/api/conversations/{conv}/messages",
                json={"content": "", "type": "image", "media_url": up["file_url"], "temp_id": "auth1"},
            )
        ).json()
        attachment_id = msg["media_url"].rsplit("/", 1)[-1]

        routes = [
            f"/api/media/{attachment_id}",
            f"/api/media/{attachment_id}/thumb",
            f"/api/media/{attachment_id}/meta",
            f"/api/media/{attachment_id}/page/1",
        ]
        # A participant is served (or 404s only because a PNG has no PDF pages).
        assert (await alice.get(routes[0], follow_redirects=False)).status_code == 307
        assert (await alice.get(routes[2])).status_code == 200

    # Foreign org: every route, same answer.
    async with _client_for("carol@b.com") as carol:
        for route in routes:
            resp = await carol.get(route, follow_redirects=False)
            assert resp.status_code == 404, f"{route} -> {resp.status_code}"

    # Deactivating the conversation must revoke the BYTES, not only the listings.
    #
    # Deactivation is the only lever a superadmin has to cut a tenant off from a
    # cross-org group: it leaves the participant rows in place, so membership alone
    # outlives the revocation. /messages and the media drawer already honoured the
    # flag and this path did not — measured before the fix as /messages 404 and
    # /media 404 while the attachment served 307 and /meta served 200, so anyone
    # holding an id kept reading files out of a group they had been removed from.
    async with SessionLocal() as db:
        row = await db.get(Conversation, uuid.UUID(conv))
        row.is_active = False
        await db.commit()

    async with _client_for("alice@a.com") as alice:
        for route in routes:
            resp = await alice.get(route, follow_redirects=False)
            assert resp.status_code == 404, f"deactivated {route} -> {resp.status_code}"

    async with SessionLocal() as db:
        row = await db.get(Conversation, uuid.UUID(conv))
        row.is_active = True
        await db.commit()

    # Tombstone the message: the owner now 404s too, on every route.
    async with SessionLocal() as db:
        row = await db.get(Message, uuid.UUID(msg["_id"]))
        row.deleted_at = now_utc()
        await db.commit()

    async with _client_for("alice@a.com") as alice:
        for route in routes:
            resp = await alice.get(route, follow_redirects=False)
            assert resp.status_code == 404, f"tombstoned {route} -> {resp.status_code}"
