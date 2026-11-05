"""Concurrent creates and renames must refuse with 400, not 500.

Every create and rename in the two admin surfaces checks for a clash and then
writes. That check is check-then-act: two requests can both pass it, and the
database constraint is what actually enforces uniqueness — so the loser of the
race has to become the same 400 a sequential duplicate gets.

Three of nine sites did that. api/admin.py guarded its creates and not its
renames; api/org_admin.py guarded nothing. An org admin creating a user whose
email was taken concurrently got a 500 where a superadmin doing the identical
thing got a clean 400.

WHAT THESE TESTS DO AND DO NOT PROVE

A genuine two-request race is not forced here. The pre-check answers first in any
sequential test, so driving one over HTTP exercises the pre-check and not the
guard — which is how a test like that passes while saying it covers the race.
Instead: the helper is tested directly against a real IntegrityError, the
constraints are proven to fire, and the nine call sites are pinned by a wiring
assertion. Said plainly rather than dressed up.
"""

import ast
import pathlib
import uuid

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.errors import conflict_as_400
from app.db.models import Department, Organization, UserRole
from app.db.session import SessionLocal
from app.main import app
from app.utils import slugify
from tests.conftest import CSRF, login, make_org, make_user

APP = pathlib.Path(__file__).resolve().parents[1] / "app"


async def test_the_helper_turns_a_constraint_violation_into_a_400():
    org = await make_org("Helper Co")
    async with SessionLocal() as db:
        db.add(Department(org_id=org.id, name="Once"))
        await db.commit()

    async with SessionLocal() as db:
        db.add(Department(org_id=org.id, name="Once"))
        with pytest.raises(HTTPException) as caught:
            async with conflict_as_400(db, "Department already exists"):
                await db.flush()

    assert caught.value.status_code == 400
    assert caught.value.detail == "Department already exists"
    # The original cause is kept, so the traceback still names the constraint.
    assert isinstance(caught.value.__cause__, IntegrityError)


async def test_the_helper_rolls_back_so_the_session_stays_usable():
    """Load-bearing: every one of these handlers writes an audit row after the
    write that failed. Without the rollback the session is in a failed
    transaction and that next statement fails too, turning one refused request
    into a second, confusing error."""
    org = await make_org("Rollback Co")
    async with SessionLocal() as db:
        db.add(Department(org_id=org.id, name="Twice"))
        await db.commit()

    async with SessionLocal() as db:
        db.add(Department(org_id=org.id, name="Twice"))
        with pytest.raises(HTTPException):
            async with conflict_as_400(db, "Department already exists"):
                await db.flush()

        # The session works immediately afterwards.
        rows = (await db.execute(select(Department.name).where(Department.org_id == org.id))).scalars().all()
        assert rows == ["Twice"]


async def test_a_foreign_key_violation_is_not_reported_as_a_duplicate():
    """IntegrityError covers more than uniqueness, and the messages are not
    interchangeable.

    A department whose org_id does not exist fails with SQLSTATE 23503, not
    23505. Answering that with "already exists" sends whoever is debugging it in
    exactly the wrong direction, so it is re-raised as the unexpected condition it
    is. Verified against this schema: duplicate is 23505, orphan FK is 23503.
    """
    async with SessionLocal() as db:
        db.add(Department(org_id=uuid.uuid4(), name="Orphan"))
        with pytest.raises(IntegrityError) as caught:
            async with conflict_as_400(db, "Department already exists"):
                await db.flush()

    assert caught.value.orig.sqlstate == "23503"


async def test_the_session_is_rolled_back_even_for_an_error_it_re_raises():
    """The rollback is unconditional on purpose: the session is unusable after any
    integrity failure, and the handler's audit write is the next statement."""
    org = await make_org("Rollback Any Co")
    async with SessionLocal() as db:
        db.add(Department(org_id=uuid.uuid4(), name="Orphan"))
        with pytest.raises(IntegrityError):
            async with conflict_as_400(db, "Department already exists"):
                await db.flush()

        # Usable immediately afterwards.
        db.add(Department(org_id=org.id, name="Fine"))
        await db.commit()

    async with SessionLocal() as db:
        names = (await db.execute(select(Department.name).where(Department.org_id == org.id))).scalars().all()
    assert names == ["Fine"]


async def test_the_helper_passes_other_errors_through():
    """It must not swallow anything that is not a uniqueness conflict."""
    async with SessionLocal() as db:
        with pytest.raises(ValueError, match="not a constraint problem"):
            async with conflict_as_400(db, "Department already exists"):
                raise ValueError("not a constraint problem")


async def test_the_constraints_are_what_actually_enforce_uniqueness():
    """If these stop raising, every guard above is unreachable and the tests that
    exercise them measure nothing."""
    org = await make_org("Constraint Co")
    async with SessionLocal() as db:
        db.add(Department(org_id=org.id, name="Dup"))
        await db.commit()
    with pytest.raises(IntegrityError):
        async with SessionLocal() as db:
            db.add(Department(org_id=org.id, name="Dup"))
            await db.commit()

    await make_user("dup-email@a.com")
    with pytest.raises(IntegrityError):
        await make_user("dup-email@a.com")


async def test_names_that_differ_can_still_collide_on_the_slug():
    """Why the RENAME paths need the guard and not just the name comparison:
    "Acme Inc" and "acme inc!" are different names and the same slug."""
    assert slugify("Acme Inc") == slugify("acme inc!")
    async with SessionLocal() as db:
        db.add(Organization(name="Acme Inc", slug=slugify("Acme Inc"), is_active=True))
        await db.commit()
    with pytest.raises(IntegrityError):
        async with SessionLocal() as db:
            db.add(Organization(name="acme inc!", slug=slugify("acme inc!"), is_active=True))
            await db.commit()


def test_every_uniqueness_write_is_guarded():
    """The wiring, asserted as a shape rather than a list of nine names.

    This is the assertion that would have caught the original gap: the guard
    existed, in three places, and nothing said the other six needed it.
    """
    unguarded = []
    for name in ("admin.py", "org_admin.py"):
        path = APP / "api" / name
        src = path.read_text()
        tree = ast.parse(src, filename=str(path))
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]:
            body = ast.get_source_segment(src, fn) or ""
            writes = any(k in body for k in ("db.add(", ".name =", ".email =", ".slug ="))
            unique_field = any(k in body.lower() for k in ("email", "slug", "name"))
            persists = "await db.flush()" in body or "await db.commit()" in body
            if writes and unique_field and persists and "conflict_as_400" not in body:
                unguarded.append(f"{name}:{fn.name}")
    assert not unguarded, f"uniqueness writes with no conflict guard: {unguarded}"


async def test_a_sequential_duplicate_is_refused_by_the_pre_check():
    """The common case, pinned so the distinction stays visible: this never
    reaches the constraint, which is exactly why the guard needed its own test."""
    org = await make_org("Sequential Co")
    await make_user("seq-admin@a.com", role=UserRole.org_admin, org_id=org.id)
    async with SessionLocal() as db:
        dept = Department(org_id=org.id, name="Ward")
        db.add(dept)
        await db.commit()
        dept_id = str(dept.id)
    await make_user("seq-taken@a.com", org_id=org.id)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c.headers.update(CSRF)
        await login(c, "seq-admin@a.com")
        resp = await c.post(
            "/api/org-admin/users",
            json={
                "email": "seq-taken@a.com",
                "display_name": "Dup",
                "password": "TestPass1234",
                "dept_id": dept_id,
            },
        )
    assert resp.status_code == 400, resp.text
    assert "Email already in use" in resp.text
