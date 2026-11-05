"""Generate previews for PDFs uploaded before the feature existed.

    python -m app.tools.backfill_pdf_previews [--dry-run] [--limit N]

PDFs sent before page-1 previews shipped have thumbnail_key and page_count NULL,
so their bubbles show the plain icon and the reader has no pages to fetch. This
walks those rows, renders page 1, stores the thumbnail alongside the original and
records the page count.

Strictly speaking this is optional: /media/{id}/thumb renders lazily on first
view, so a PDF nobody opens costs nothing and one that is opened heals itself.
The batch job exists because the lazy path cannot populate `page_count` in a
payload the client has ALREADY fetched — without it the bubble subtitle omits
"N pages" and the reader opens empty until the conversation is reloaded.

Idempotent: rows that already have a page_count are skipped, so re-running is
free. An un-renderable PDF (encrypted, corrupt) is marked page_count=0, which is
the "tried and failed" marker that stops it being re-parsed forever.

Reads and writes one row at a time rather than loading every PDF into memory,
and rendering is serialised by storage's PDFium limiter regardless.
"""

import argparse
import asyncio

from sqlalchemy import select

from app.db.models import MessageAttachment, Upload
from app.db.session import SessionLocal, engine
from app.services import storage
from app.tools.paging import iter_keyset

PDF_MIME = "application/pdf"


async def _backfill_attachments(dry_run: bool, limit: int | None) -> tuple[int, int, int]:
    done = failed = skipped = 0
    async with SessionLocal() as db:
        stmt = select(MessageAttachment).where(
            MessageAttachment.mime_type == PDF_MIME,
            MessageAttachment.page_count.is_(None),
        )
        # Paged for the same reason as regenerate. This pass does remove rows from
        # its own selection as it fills in page_count, so "re-query the first N"
        # would have advanced here — but not for a row it SKIPS without marking,
        # such as one whose object is missing from storage, which would then be
        # handed back forever.
        print("message_attachments: walking PDFs without a preview")

        async for a in iter_keyset(db, stmt, MessageAttachment.created_at, MessageAttachment.id, limit=limit):
            label = f"  {a.filename[:52]:52}"
            if dry_run:
                print(f"{label} would render")
                skipped += 1
                continue
            # The same ceiling the request paths apply, from the same function.
            # Without it this tool renders a file api/media.py refuses and writes a
            # positive page_count onto it — which is exactly the row state that
            # forces serve_pdf_page to keep a second guard. Marked as tried so the
            # lazy path does not come back to it either.
            if storage.too_large_to_render(a.file_size):
                print(f"{label} SKIP — too large to render ({a.file_size} bytes)")
                a.page_count = 0
                await db.commit()
                skipped += 1
                continue
            try:
                data = await storage.get_object(a.storage_key)
            except Exception as exc:  # object missing from storage
                print(f"{label} SKIP — cannot read object ({type(exc).__name__})")
                skipped += 1
                continue

            thumb, pages = await storage.make_pdf_preview(data)
            if thumb is None:
                # Mark as tried so neither this job nor the lazy path retries it.
                a.page_count = 0
                await db.commit()
                print(f"{label} UNRENDERABLE — marked, will keep the icon")
                failed += 1
                continue

            key = f"{a.storage_key}_thumb.jpg"
            await storage.put_object(key, thumb, "image/jpeg")
            a.thumbnail_key = key
            a.page_count = pages
            await db.commit()
            print(f"{label} ok — {pages} page(s), {len(thumb)} byte thumbnail")
            done += 1
    return done, failed, skipped


async def _backfill_uploads(dry_run: bool, limit: int | None = None) -> int:
    """Staging rows. Only matters for uploads not yet claimed by a message —
    a claimed one already copied its (then NULL) page_count onto the attachment,
    which the pass above fixes directly."""
    done = 0
    async with SessionLocal() as db:
        stmt = select(Upload).where(
            Upload.mime_type == PDF_MIME,
            Upload.page_count.is_(None),
            Upload.claimed.is_(False),
        )
        # --limit bounded the attachments pass and not this one, so a run capped at
        # ten attachments still walked every unclaimed upload in the table.
        print("uploads (unclaimed): walking PDFs without a page count")
        async for u in iter_keyset(db, stmt, Upload.created_at, Upload.id, limit=limit):
            if dry_run:
                continue
            if storage.too_large_to_render(u.file_size):
                u.page_count = 0
                await db.commit()
                continue
            try:
                data = await storage.get_object(u.storage_key)
            except Exception:
                continue
            thumb, pages = await storage.make_pdf_preview(data)
            if thumb is None:
                u.page_count = 0
                await db.commit()
                continue
            key = f"{u.storage_key}_thumb.jpg"
            await storage.put_object(key, thumb, "image/jpeg")
            u.thumbnail_key = key
            u.page_count = pages
            await db.commit()
            done += 1
    return done


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="list what would change, touch nothing")
    ap.add_argument("--limit", type=int, default=None, help="cap how many rows each pass processes")
    args = ap.parse_args()

    if args.dry_run:
        print("DRY RUN — nothing will be written\n")

    done, failed, skipped = await _backfill_attachments(args.dry_run, args.limit)
    up = await _backfill_uploads(args.dry_run, args.limit)

    print(
        f"\ndone: {done} rendered, {failed} unrenderable, {skipped} skipped, {up} unclaimed upload(s) updated"
    )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
