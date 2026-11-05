"""Keyset paging for the maintenance tools.

Both tools read a selection and then commit once per row. That combination rules
out the two obvious ways to bound the read:

  * A server-side cursor — `stream_scalars`, or `yield_per` — has its cursor
    invalidated by the first commit, so the walk dies one row in.
  * "Re-query the first N" only advances when processing removes a row from the
    selection. The backfill does remove rows (it fills in the NULL page_count it
    selects on), but `regenerate` without --only-missing does not: it re-renders in
    place, so the same page would be handed back forever.

So: keyset on (created_at, id), one page at a time. Memory scales with the page
rather than with the table, which is the property both tools already claimed —
"Safe to interrupt and re-run: each attachment is independent and committed as it
goes" was true of the writes and not of the read.

The id is in the key for stability, not for ordering: created_at has no uniqueness
and two rows written in the same transaction routinely share it, so a keyset on
created_at alone would either skip or repeat them.
"""

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import tuple_
from sqlalchemy.ext.asyncio import AsyncSession

PAGE = 200


async def iter_keyset(
    db: AsyncSession,
    stmt,
    created_col,
    id_col,
    *,
    limit: int | None = None,
    page: int = PAGE,
) -> AsyncIterator[Any]:
    """Yield rows matching `stmt`, a page at a time, honouring `limit`.

    `stmt` must NOT carry its own ORDER BY or LIMIT — both belong to the paging and
    are applied here.

    THE WALK IS BEST-EFFORT, NOT A SNAPSHOT

    A short page proves only that the selection was exhausted when THAT query ran.
    A matching row committed afterwards — by ordinary traffic, while the sweep is
    running — is not processed in this run. Nor is one whose created_at sorts before
    the cursor, since the keyset has already moved past it.

    That is the right trade for these tools and worth stating rather than leaving to
    be discovered: both are re-runnable sweeps whose next run picks up whatever this
    one missed, and neither promises to process an exact start-of-run set. A tool
    that did need that guarantee would want a high-water mark taken before the walk
    begins and carried as an upper bound, not this.
    """
    ordered = stmt.order_by(created_col, id_col)
    seen = 0
    cursor: tuple[Any, Any] | None = None
    while True:
        take = page if limit is None else min(page, limit - seen)
        if take <= 0:
            return
        query = ordered
        if cursor is not None:
            query = query.where(tuple_(created_col, id_col) > cursor)
        rows = list((await db.execute(query.limit(take))).scalars().all())
        if not rows:
            return
        for row in rows:
            yield row
            seen += 1
        # A short page means the selection is exhausted. Checked before advancing
        # the cursor so a final partial page does not cost an extra round trip.
        if len(rows) < take:
            return
        last = rows[-1]
        cursor = (getattr(last, created_col.key), getattr(last, id_col.key))
