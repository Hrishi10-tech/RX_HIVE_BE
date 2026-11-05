"""What reading an attachment back costs, versus what uploading it was allowed to.

The upload path refuses to hold a preview source in memory above
THUMBNAIL_SOURCE_LIMIT (64 MB) — api/media.py says so outright: "Previews need the
bytes in memory, so they are the one thing still bounded by size."

The paths that read the same object back applied no bound at all. Worse, the
self-healing preview is aimed exactly at the files upload declined: its guard runs
when a PDF has no thumbnail_key AND no page_count, which is precisely the state a
>64 MB upload is left in. So the file upload refused to buffer was the file a
member's first chat open pulled into the worker whole, up to max_upload_bytes (2 GB
by default).

storage._PDF_LIMITER is CapacityLimiter(1), which makes it worse rather than
better: it serialises the render, so concurrent requests queue — each already
holding its own full copy of the file.
"""

import uuid

import pytest
from sqlalchemy import select

from app.db.models import (
    Conversation,
    ConversationParticipant,
    ConversationType,
    Message,
    MessageAttachment,
    MessageType,
)
from app.db.session import SessionLocal
from app.services import storage
from tests.conftest import make_org, make_user
from tests.test_calls_and_media import _client_for

OVERSIZE = storage.THUMBNAIL_SOURCE_LIMIT + 1


async def _pdf_attachment(*, file_size: int) -> tuple[uuid.UUID, str]:
    """A claimed PDF attachment in the state upload leaves an unpreviewed one."""
    org = await make_org(f"Preview Co {uuid.uuid4().hex[:6]}")
    alice = await make_user(f"a{uuid.uuid4().hex[:6]}@prev.com", org_id=org.id)
    bob = await make_user(f"b{uuid.uuid4().hex[:6]}@prev.com", org_id=org.id)
    key = f"{org.id}/{uuid.uuid4()}.pdf"
    await storage.put_object(key, b"%PDF-1.4 not a real document", "application/pdf")

    async with SessionLocal() as db:
        conv = Conversation(type=ConversationType.direct, org_id=org.id, is_active=True)
        db.add(conv)
        await db.flush()
        db.add_all(
            [
                ConversationParticipant(conversation_id=conv.id, user_id=alice.id),
                ConversationParticipant(conversation_id=conv.id, user_id=bob.id),
            ]
        )
        msg = Message(conversation_id=conv.id, sender_id=alice.id, type=MessageType.file, content="")
        db.add(msg)
        await db.flush()
        att = MessageAttachment(
            message_id=msg.id,
            storage_key=key,
            thumbnail_key=None,
            filename="report.pdf",
            mime_type="application/pdf",
            file_size=file_size,
            page_count=None,
        )
        db.add(att)
        await db.commit()
        return att.id, alice.email


@pytest.fixture
def watch_fetches(monkeypatch):
    """Record every object body pulled into memory, and stub the renderer.

    The renderer is stubbed because the gate is what is under test: whether the
    bytes are fetched at all, not whether a fake PDF parses.
    """
    fetched: list[str] = []
    real_get = storage.get_object

    async def _get(key: str) -> bytes:
        fetched.append(key)
        return await real_get(key)

    async def _preview(data: bytes):
        return b"jpegbytes", 3

    async def _window(data: bytes, start: int):
        return {start: b"jpegbytes"}

    for module in (storage,):
        monkeypatch.setattr(module, "get_object", _get)
        monkeypatch.setattr(module, "make_pdf_preview", _preview)
        monkeypatch.setattr(module, "render_pdf_window", _window)
    return fetched


async def test_an_oversized_pdf_is_not_pulled_into_memory_for_a_thumbnail(client, watch_fetches):
    att_id, email = await _pdf_attachment(file_size=OVERSIZE)

    async with _client_for(email) as c:
        resp = await c.get(f"/api/media/{att_id}/thumb")

    assert watch_fetches == [], (
        f"the whole object was read to build a thumbnail for a {OVERSIZE}-byte file, "
        "which the upload path refused to buffer at that size"
    )
    assert resp.status_code == 404, resp.text


async def test_an_oversized_pdf_is_marked_so_it_is_not_retried(client, watch_fetches):
    """page_count = 0 is this module's "tried and could not render" marker.

    Without it every view of the file re-enters the path, so the refusal would
    have to be made again on every request forever.
    """
    att_id, email = await _pdf_attachment(file_size=OVERSIZE)

    async with _client_for(email) as c:
        await c.get(f"/api/media/{att_id}/thumb")

    async with SessionLocal() as db:
        att = (await db.execute(select(MessageAttachment).where(MessageAttachment.id == att_id))).scalar_one()
    assert att.page_count == 0
    assert att.thumbnail_key is None


async def test_an_oversized_pdf_page_is_not_rendered_on_demand(client, watch_fetches):
    """The page reader reads the same object, once per window miss."""
    att_id, email = await _pdf_attachment(file_size=OVERSIZE)

    async with _client_for(email) as c:
        resp = await c.get(f"/api/media/{att_id}/page/1")

    assert watch_fetches == [], "the page reader pulled the whole object into memory"
    assert resp.status_code == 404, resp.text


async def test_a_pdf_within_the_limit_still_previews(client, watch_fetches):
    """The converse. A bound that refused everything would satisfy the tests above."""
    att_id, email = await _pdf_attachment(file_size=storage.THUMBNAIL_SOURCE_LIMIT)

    async with _client_for(email) as c:
        resp = await c.get(f"/api/media/{att_id}/thumb")

    assert watch_fetches, "a file inside the limit must still be previewed"
    assert resp.status_code in (200, 307), resp.text

    async with SessionLocal() as db:
        att = (await db.execute(select(MessageAttachment).where(MessageAttachment.id == att_id))).scalar_one()
    assert att.thumbnail_key is not None
    assert att.page_count == 3


async def test_a_window_miss_on_an_already_counted_oversized_pdf_is_refused(client, watch_fetches):
    """A page_count already set does not prove the file is small.

    app/tools/backfill_pdf_previews.py and regenerate_previews.py set page_count
    with no size bound of their own, and rows predate this limit. Either way
    _ensure_pdf_preview returns early on such a row — page_count is not None — so
    the page reader needs its own check or a window miss fetches the whole object.
    """
    att_id, email = await _pdf_attachment(file_size=OVERSIZE)
    async with SessionLocal() as db:
        att = (await db.execute(select(MessageAttachment).where(MessageAttachment.id == att_id))).scalar_one()
        att.page_count = 40  # as a tool would have left it
        await db.commit()

    async with _client_for(email) as c:
        resp = await c.get(f"/api/media/{att_id}/page/7")

    assert watch_fetches == [], "a window miss fetched the whole object anyway"
    assert resp.status_code == 404, resp.text


async def test_a_page_already_rendered_is_still_served_for_an_oversized_pdf(client, watch_fetches):
    """The refusal is about rendering, not about serving what already exists.

    Otherwise a large PDF whose pages were rendered before this limit would stop
    working — the fix would break files it was meant to leave alone.
    """
    att_id, email = await _pdf_attachment(file_size=OVERSIZE)
    async with SessionLocal() as db:
        att = (await db.execute(select(MessageAttachment).where(MessageAttachment.id == att_id))).scalar_one()
        att.page_count = 40
        key = att.storage_key
        await db.commit()
    await storage.put_object(f"{key}_p7.jpg", b"already rendered", "image/jpeg")

    async with _client_for(email) as c:
        resp = await c.get(f"/api/media/{att_id}/page/7")

    assert resp.status_code == 307, resp.text
    assert watch_fetches == [], "serving a cached page must not re-read the source"
