"""Re-render existing preview thumbnails at the current quality settings.

    python -m app.tools.regenerate_previews [--kind image|pdf|all] [--dry-run]
                                            [--limit N] [--only-missing]

Raising THUMB_PX / PDF_THUMB_PX only affects NEW uploads: a thumbnail already in
object storage stays exactly as it was rendered. Every image and PDF already in a
conversation therefore keeps its old, softer preview until it is re-rendered,
which is what this does.

Distinct from backfill_pdf_previews, which fills in previews that were never
generated and deliberately skips anything already done. This REPLACES existing
ones, so by default it re-renders everything of the chosen kind. Pass
--only-missing to get the other behaviour.

Overwrites in place under the same key, so nothing needs to change in the
database for an unchanged thumbnail_key — the URL is stable and clients pick the
new bytes up on their next fetch. Note that means a client holding the old image
in its HTTP cache keeps showing it until that entry expires; presigned URLs are
short-lived, so in practice this is minutes.

Safe to interrupt and re-run: each attachment is independent and committed as it
goes. An un-renderable file is reported and skipped, never left half-written.
"""

import argparse
import asyncio

from sqlalchemy import or_, select

from app.db.models import MessageAttachment
from app.db.session import SessionLocal, engine
from app.services import storage
from app.tools.paging import iter_keyset

PDF_MIME = "application/pdf"


async def _regenerate(kind: str, dry_run: bool, limit: int | None, only_missing: bool) -> None:
    done = failed = skipped = 0
    bytes_before = bytes_after = 0

    async with SessionLocal() as db:
        conds = []
        if kind in ("image", "all"):
            conds.append(MessageAttachment.mime_type.like("image/%"))
        if kind in ("pdf", "all"):
            conds.append(MessageAttachment.mime_type == PDF_MIME)

        stmt = select(MessageAttachment).where(or_(*conds))
        if only_missing:
            stmt = stmt.where(MessageAttachment.thumbnail_key.is_(None))

        # Paged, not `.all()`. Without --limit this selection is every image and
        # every PDF attachment in the database, and reading it whole put the entire
        # table in the operator's process before the first file was touched. The
        # docstring's promise — "committed as it goes" — was true of the writes and
        # not of the read. No count printed up front any more: knowing it would mean
        # the COUNT(*) this exists to avoid.
        print(f"walking attachments (kind={kind})\n")

        async for a in iter_keyset(
            db,
            stmt,
            MessageAttachment.created_at,
            MessageAttachment.id,
            limit=limit,
        ):
            label = f"  {a.filename[:44]:44}"
            is_pdf = a.mime_type == PDF_MIME

            # page_count 0 is the "tried and could not render" marker. Re-running
            # a known-bad file every time this tool is invoked buys nothing.
            if is_pdf and a.page_count == 0:
                print(f"{label} SKIP — previously un-renderable")
                skipped += 1
                continue

            if dry_run:
                print(f"{label} would re-render ({'pdf' if is_pdf else 'image'})")
                skipped += 1
                continue

            # The same ceiling the request paths apply, from the same function.
            # This tool exists to RE-render, so it reaches files that already have a
            # preview — including ones rendered before that ceiling existed. Without
            # this it reads them whole, and a successful re-render would write a
            # positive page_count that api/media.py then has to defend against.
            #
            # page_count is left exactly as it was. Unlike the backfill, which is
            # visiting never-tried rows, an existing count here is a fact about a
            # preview that already exists, and a skip is not new information about
            # the document.
            if storage.too_large_to_render(a.file_size):
                print(f"{label} SKIP — too large to re-render ({a.file_size} bytes)")
                skipped += 1
                continue

            try:
                data = await storage.get_object(a.storage_key)
            except Exception as exc:
                print(f"{label} SKIP — cannot read original ({type(exc).__name__})")
                skipped += 1
                continue

            if is_pdf:
                thumb, pages = await storage.make_pdf_preview(data)
                if thumb is None:
                    a.page_count = 0
                    await db.commit()
                    print(f"{label} UNRENDERABLE — marked, keeps its icon")
                    failed += 1
                    continue
                a.page_count = pages
            else:
                thumb = await storage.make_thumbnail(data)
                if thumb is None:
                    print(f"{label} UNRENDERABLE — image could not be decoded")
                    failed += 1
                    continue

            key = a.thumbnail_key or f"{a.storage_key}_thumb.jpg"
            # Measure the outgoing one so the run reports what it actually cost.
            if a.thumbnail_key:
                try:
                    bytes_before += len(await storage.get_object(a.thumbnail_key))
                except Exception:
                    pass
            await storage.put_object(key, thumb, "image/jpeg")
            a.thumbnail_key = key
            bytes_after += len(thumb)
            await db.commit()
            print(f"{label} ok — {len(thumb) // 1024} KB")
            done += 1

    print(
        f"\nre-rendered {done}, unrenderable {failed}, skipped {skipped}"
        f"\nthumbnail bytes: {bytes_before // 1024} KB -> {bytes_after // 1024} KB"
    )
    await engine.dispose()


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=("image", "pdf", "all"), default="all")
    ap.add_argument("--dry-run", action="store_true", help="list what would change, write nothing")
    ap.add_argument("--limit", type=int, default=None, help="cap how many to process")
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="only attachments with no thumbnail at all (backfill, not re-render)",
    )
    args = ap.parse_args()

    print(
        f"settings in force: THUMB_PX={storage.THUMB_PX} q{storage.THUMB_QUALITY}, "
        f"PDF_THUMB_PX={storage.PDF_THUMB_PX} q{storage.PDF_QUALITY}"
    )
    if args.dry_run:
        print("DRY RUN — nothing will be written\n")

    await _regenerate(args.kind, args.dry_run, args.limit, args.only_missing)


if __name__ == "__main__":
    asyncio.run(main())
