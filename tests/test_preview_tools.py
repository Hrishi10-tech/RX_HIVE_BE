"""The maintenance tools, against the constraints the request paths enforce.

Batch 54 bounded the two request paths that render a preview: they decide from
file_size on the row and never fetch an object bigger than an upload is allowed to
buffer. app/tools/backfill_pdf_previews.py and app/tools/regenerate_previews.py
were left out of that, and api/media.py's own comment records the consequence —

    a page_count already set does not prove the file is small. The backfill and
    regenerate tools in app/tools set it with no size bound of their own

— so the tools can produce exactly the row state the request path then has to keep
a second guard against. They also read every matching row into memory at once when
no --limit is given, which for `regenerate --kind all` is every image and PDF
attachment in the database.

Driven by calling the tools' own functions rather than their argparse main(), so
the assertions are about what they do, not about how they are invoked.
"""

import datetime as dt
import uuid

import pytest
from sqlalchemy import select, text

from app.db.models import (
    Conversation,
    ConversationParticipant,
    ConversationType,
    Message,
    MessageAttachment,
    MessageType,
    Upload,
)
from app.db.session import SessionLocal
from app.services import storage
from app.tools import backfill_pdf_previews, regenerate_previews
from app.tools.paging import iter_keyset
from tests.conftest import make_org, make_user

OVERSIZE = storage.THUMBNAIL_SOURCE_LIMIT + 1


async def _attachment(*, mime: str, file_size: int, page_count=None, thumb=False):
    org = await make_org(f"Tools Co {uuid.uuid4().hex[:6]}")
    alice = await make_user(f"a{uuid.uuid4().hex[:6]}@tools.com", org_id=org.id)
    key = f"{org.id}/{uuid.uuid4()}.pdf"
    await storage.put_object(key, b"%PDF-1.4 tiny", mime)
    thumb_key = None
    if thumb:
        thumb_key = f"{key}_thumb.jpg"
        await storage.put_object(thumb_key, b"jpeg", "image/jpeg")

    async with SessionLocal() as db:
        conv = Conversation(type=ConversationType.direct, org_id=org.id, is_active=True)
        db.add(conv)
        await db.flush()
        db.add(ConversationParticipant(conversation_id=conv.id, user_id=alice.id))
        msg = Message(conversation_id=conv.id, sender_id=alice.id, type=MessageType.file, content="")
        db.add(msg)
        await db.flush()
        att = MessageAttachment(
            message_id=msg.id,
            storage_key=key,
            thumbnail_key=thumb_key,
            filename="report.pdf",
            mime_type=mime,
            file_size=file_size,
            page_count=page_count,
        )
        db.add(att)
        await db.commit()
        return att.id


@pytest.fixture
def watch_tool_reads(monkeypatch):
    """Record every object body a tool pulls into memory; stub the renderers."""
    fetched: list[str] = []
    real_get = storage.get_object

    async def _get(key: str) -> bytes:
        fetched.append(key)
        return await real_get(key)

    async def _pdf(data: bytes):
        return b"jpeg", 4

    async def _thumb(data: bytes):
        return b"jpeg"

    monkeypatch.setattr(storage, "get_object", _get)
    monkeypatch.setattr(storage, "make_pdf_preview", _pdf)
    monkeypatch.setattr(storage, "make_thumbnail", _thumb)
    return fetched


async def test_backfill_does_not_read_an_oversized_pdf(client, watch_tool_reads):
    """The tool must decline the same files the request paths decline.

    Otherwise it renders one and writes a positive page_count, which is precisely
    the row state api/media.py keeps its second guard for.
    """
    att_id = await _attachment(mime="application/pdf", file_size=OVERSIZE)

    await backfill_pdf_previews._backfill_attachments(dry_run=False, limit=None)

    assert watch_tool_reads == [], (
        f"the backfill read a {OVERSIZE}-byte object whole; the request paths refuse the same file"
    )
    async with SessionLocal() as db:
        att = (await db.execute(select(MessageAttachment).where(MessageAttachment.id == att_id))).scalar_one()
    assert att.page_count == 0, "it must be marked as tried, not left to be retried forever"
    assert att.thumbnail_key is None


async def test_regenerate_does_not_read_an_oversized_attachment(client, watch_tool_reads):
    att_id = await _attachment(mime="application/pdf", file_size=OVERSIZE, page_count=9)

    await regenerate_previews._regenerate(kind="all", dry_run=False, limit=None, only_missing=False)

    assert watch_tool_reads == [], "regenerate read an oversized object whole"
    async with SessionLocal() as db:
        att = (await db.execute(select(MessageAttachment).where(MessageAttachment.id == att_id))).scalar_one()
    assert att.page_count == 9, "an existing count must not be rewritten by a skip"


async def test_backfill_still_processes_a_file_within_the_limit(client, watch_tool_reads):
    """The converse. A bound that declined everything would satisfy the tests above."""
    att_id = await _attachment(mime="application/pdf", file_size=storage.THUMBNAIL_SOURCE_LIMIT)

    await backfill_pdf_previews._backfill_attachments(dry_run=False, limit=None)

    assert watch_tool_reads, "a file inside the limit must still be backfilled"
    async with SessionLocal() as db:
        att = (await db.execute(select(MessageAttachment).where(MessageAttachment.id == att_id))).scalar_one()
    assert att.thumbnail_key is not None
    assert att.page_count == 4


async def test_the_uploads_pass_honours_a_limit(client, watch_tool_reads):
    """`_backfill_uploads` took no limit at all, so --limit bounded half a run."""
    org = await make_org("Tools Uploads Co")
    uploader = await make_user("u@toolsup.com", org_id=org.id)
    async with SessionLocal() as db:
        for _ in range(3):
            key = f"{org.id}/{uuid.uuid4()}.pdf"
            await storage.put_object(key, b"%PDF-1.4 tiny", "application/pdf")
            db.add(
                Upload(
                    uploader_id=uploader.id,
                    org_id=org.id,
                    storage_key=key,
                    filename="x.pdf",
                    mime_type="application/pdf",
                    file_type="document",
                    file_size=10,
                    claimed=False,
                    page_count=None,
                )
            )
        await db.commit()

    done = await backfill_pdf_previews._backfill_uploads(dry_run=False, limit=1)
    assert done == 1, "the uploads pass ignored the limit"
    assert len(watch_tool_reads) == 1


# ---------------------------------------------------------------------------
# The paginator itself
# ---------------------------------------------------------------------------


async def _uploads_sharing_a_timestamp(count: int, *, same_time: bool) -> list[uuid.UUID]:
    """`count` unclaimed PDF uploads, optionally all with one created_at."""
    org = await make_org(f"Paging Co {uuid.uuid4().hex[:6]}")
    uploader = await make_user(f"p{uuid.uuid4().hex[:6]}@paging.com", org_id=org.id)
    stamp = dt.datetime.now(dt.UTC)
    ids = []
    async with SessionLocal() as db:
        for i in range(count):
            row = Upload(
                uploader_id=uploader.id,
                org_id=org.id,
                storage_key=f"{org.id}/{uuid.uuid4()}.pdf",
                filename=f"{i}.pdf",
                mime_type="application/pdf",
                file_type="document",
                file_size=10,
                claimed=False,
                page_count=None,
                created_at=stamp if same_time else stamp + dt.timedelta(seconds=i),
            )
            db.add(row)
            await db.flush()
            ids.append(row.id)
        await db.commit()
    return ids


def _selection():
    return select(Upload).where(Upload.page_count.is_(None), Upload.claimed.is_(False))


async def test_the_paginator_walks_past_the_page_boundary(client):
    ids = await _uploads_sharing_a_timestamp(5, same_time=False)
    async with SessionLocal() as db:
        seen = [row.id async for row in iter_keyset(db, _selection(), Upload.created_at, Upload.id, page=2)]
    assert seen == ids, "rows must come back once each, in created_at order"


async def test_the_paginator_honours_a_limit_across_pages(client):
    ids = await _uploads_sharing_a_timestamp(5, same_time=False)
    async with SessionLocal() as db:
        seen = [
            row.id
            async for row in iter_keyset(db, _selection(), Upload.created_at, Upload.id, limit=3, page=2)
        ]
    assert seen == ids[:3], "a limit that falls mid-page must still stop at the limit"


async def test_the_paginator_does_not_skip_or_repeat_rows_sharing_a_timestamp(client):
    """Why the id is in the key at all.

    created_at has no uniqueness and rows written in one transaction routinely
    share it, so a keyset on created_at alone would either skip the rest of a tied
    group or hand it back forever. Page size 1 puts a boundary between every pair.
    """
    ids = await _uploads_sharing_a_timestamp(4, same_time=True)
    async with SessionLocal() as db:
        seen = [row.id async for row in iter_keyset(db, _selection(), Upload.created_at, Upload.id, page=1)]
    assert sorted(str(i) for i in seen) == sorted(str(i) for i in ids)
    assert len(seen) == len(set(seen)), "a tied timestamp must not repeat a row"


async def test_the_paginator_survives_a_commit_mid_walk(client):
    """The reason this is keyset paging and not a server-side cursor.

    stream_scalars/yield_per would have its cursor invalidated by the first commit,
    and both tools commit once per row.
    """
    ids = await _uploads_sharing_a_timestamp(4, same_time=False)
    seen = []
    async with SessionLocal() as db:
        async for row in iter_keyset(db, _selection(), Upload.created_at, Upload.id, page=2):
            seen.append(row.id)
            row.filename = f"touched-{row.filename}"
            await db.commit()
    assert seen == ids, "a commit per row must not truncate the walk"


async def test_a_selection_that_never_shrinks_terminates(client):
    """`regenerate` without --only-missing re-renders in place.

    So the walk has to advance on its own. "Re-query the first N" would hand back
    the same page forever here, which is what the keyset avoids.
    """
    ids = await _uploads_sharing_a_timestamp(3, same_time=False)
    async with SessionLocal() as db:
        seen = [
            row.id
            # No write at all inside the loop: nothing leaves the selection.
            async for row in iter_keyset(db, _selection(), Upload.created_at, Upload.id, page=1)
        ]
    assert seen == ids


# ---------------------------------------------------------------------------
# The indexes the paged walks depend on
# ---------------------------------------------------------------------------

# Each entry is (label, index name, the query the tool actually issues). The
# predicates must stay identical to the tools' selections and to the migration's
# WHERE clauses: Postgres silently stops using a partial index when they drift,
# and the symptom is a slow tool rather than a failure. Migration a4f81c6b2e07
# records the same trade the links-tab index does.
_KEYSET_PLANS = [
    (
        "regenerate --kind all",
        "ix_message_attachments_created_keyset",
        """SELECT * FROM message_attachments
           WHERE (mime_type LIKE 'image/%' OR mime_type = 'application/pdf')
             AND (created_at, id) > (now(), '00000000-0000-0000-0000-000000000000'::uuid)
           ORDER BY created_at, id LIMIT 200""",
    ),
    (
        "backfill attachments",
        "ix_message_attachments_pdf_unpreviewed",
        """SELECT * FROM message_attachments
           WHERE mime_type = 'application/pdf' AND page_count IS NULL
             AND (created_at, id) > (now(), '00000000-0000-0000-0000-000000000000'::uuid)
           ORDER BY created_at, id LIMIT 200""",
    ),
    (
        "backfill uploads",
        "ix_uploads_pdf_unpreviewed",
        """SELECT * FROM uploads
           WHERE mime_type = 'application/pdf' AND page_count IS NULL AND claimed = false
             AND (created_at, id) > (now(), '00000000-0000-0000-0000-000000000000'::uuid)
           ORDER BY created_at, id LIMIT 200""",
    ),
]


@pytest.mark.parametrize(("label", "index", "query"), _KEYSET_PLANS, ids=[p[0] for p in _KEYSET_PLANS])
async def test_the_keyset_walk_is_index_backed(client, label, index, query):
    """Paging without a supporting index is the worse half of the trade.

    Before migration a4f81c6b2e07 neither table had ANY index on created_at, so
    every continuation query filtered and sorted the remaining matching rows —
    bounding Python memory by growing database work instead.

    WHAT THIS ASSERTS, AND WHAT IT DELIBERATELY DOES NOT

    That the index is USABLE for the query. That is the property which can break
    silently: a partial index stops being considered the moment its predicate
    drifts from the tool's selection, and the symptom is a slow tool rather than
    a failure.

    It does NOT assert the absence of a Sort node, even though an ordered index
    scan is the point. Whether the planner takes an ordered Index Scan or a
    Bitmap Index Scan plus a Sort depends on row-count statistics rather than on
    whether the index matches: on these near-empty test tables it picks the
    bitmap, and against a populated database it picks the ordered scan. Verified
    by hand there, on all three queries, with no Sort node. Asserting it here
    would be asserting the planner's cost model on an empty table.

    enable_seqscan is forced off for the same reason: on a tiny table a
    sequential scan wins on cost and says nothing about the index.
    """
    async with SessionLocal() as db:
        await db.execute(text("SET LOCAL enable_seqscan = off"))
        plan = "\n".join(row[0] for row in (await db.execute(text("EXPLAIN " + query))).all())

    assert index in plan, (
        f"{label} did not use {index} — its predicate has probably drifted from the index's:\n{plan}"
    )
