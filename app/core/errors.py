"""Error responses that carry a machine-readable code beside the prose.

`HTTPException` serialises to `{"detail": ...}` and nothing else, so a client that
has to *branch* on which refusal it got — not merely show it — is left matching
English. That is what the two mobile 403s forced: they are separated only by their
wording, and one contains the other's giveaway phrase (`MOBILE_NOT_APPROVED` ends
"Ask your super admin to approve mobile sign-in"), so a copy edit on this side
silently reroutes users to the wrong denial screen.

`CodedHTTPException` adds a sibling `code` to the envelope and leaves `detail`
byte-for-byte as it was. Nothing that reads `detail` as a string — the web
frontend, the existing tests, any other consumer — sees a change.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

# PostgreSQL unique_violation. The one integrity failure these call sites mean.
UNIQUE_VIOLATION = "23505"


class CodedHTTPException(HTTPException):
    """An `HTTPException` whose response also carries a stable `code`.

    The code, not the sentence, is the contract with the client: `detail` stays
    free to be reworded for the person reading it.
    """

    def __init__(
        self,
        status_code: int,
        detail: str,
        code: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.code = code


async def coded_http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Serialise `{"detail": ..., "code": ...}`.

    Registered for `CodedHTTPException` only. Starlette resolves handlers by
    walking the exception's MRO, so this wins for the subclass while every plain
    `HTTPException` keeps FastAPI's own handler and its exact response shape.
    """
    assert isinstance(exc, CodedHTTPException)  # registered for this type alone
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": exc.code},
        headers=exc.headers,
    )


@contextlib.asynccontextmanager
async def conflict_as_400(db: AsyncSession, detail: str) -> AsyncIterator[None]:
    """Turn a unique-constraint violation into the 400 the caller expected.

    Every create and rename in the admin surfaces checks for a clash and then
    writes. That check cannot be the whole answer — it is check-then-act, and two
    requests can both pass it — so the database constraint is what actually
    enforces uniqueness, and the loser of that race arrives here.

    Wraps whichever statement raises, which is NOT the same statement everywhere.
    A create with an explicit `await db.flush()` raises there; a rename that only
    mutates the instance and commits raises from `commit()`, because that is when
    the UPDATE is emitted. Batch 42 fixed a bug that was exactly this distinction
    misread — a guard sitting on `commit()` while the INSERT had already been
    flushed forty lines earlier, so the branch could never run.

    Used as:

        async with conflict_as_400(db, "Email already in use"):
            await db.flush()

    The rollback matters: without it the session is left in a failed transaction
    and the next statement on it fails too, which turns one refused request into a
    confusing second error.

    Only a UNIQUE violation becomes the 400. `IntegrityError` also covers foreign
    key, not-null and check failures — verified against this schema, a duplicate
    department carries SQLSTATE 23505 and one with a nonexistent org_id carries
    23503 — and converting those too would answer a broken reference with
    "already exists".
    """
    try:
        yield
    except IntegrityError as exc:
        # Roll back for EVERY integrity failure — the session is unusable either
        # way, and the handler's audit-log write is the next statement on it.
        await db.rollback()
        sqlstate = getattr(getattr(exc, "orig", None), "sqlstate", None)
        if sqlstate != UNIQUE_VIOLATION:
            # Not ours. Department creation can fail on its org_id foreign key
            # (23503), and answering that with "already exists" sends whoever is
            # debugging it in precisely the wrong direction. Re-raised, so it
            # surfaces as the unexpected condition it is.
            raise
        raise HTTPException(status_code=400, detail=detail) from exc
