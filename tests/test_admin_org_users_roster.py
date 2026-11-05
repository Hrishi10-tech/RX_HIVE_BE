"""GET /api/admin/organizations/{org_id}/users — the cross-org member picker's roster.

This endpoint had no tests. It is also the only roster endpoint in the API with no
ceiling: org_admin.list_users and admin's own user list both paginate at
`limit <= 100`, and api/contacts.list_contacts caps at 500 after measuring the
unbounded version at 24,999 rows and 5.5 MB on a 25,000-user tenant. This one
still returns every active member of an organization, and its one caller
(pages/admin/CrossOrgGroups.jsx) is a member picker.

What is fixed here is the waste that needed no contract change: it fetched rows it
then discarded, and materialized every column of the User entity — password_hash
included — to serialize five fields.
"""

import contextlib
import itertools

from httpx import ASGITransport, AsyncClient

from app.core.security import hash_password
from app.db.models import Department, User, UserRole
from app.db.session import SessionLocal
from app.main import app
from tests.conftest import CSRF, login, make_org, make_user


@contextlib.asynccontextmanager
async def _superadmin():
    await make_user("root@x.com", role=UserRole.superadmin)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c.headers.update(CSRF)
        await login(c, "root@x.com")
        yield c


_seeds = itertools.count()


async def _seed(org, *, with_dept: int, without_dept: int, inactive: int = 0, depts: int = 2):
    # Emails are globally unique, so each seeded organization needs its own
    # namespace — two orgs in one test collided on users_email_key otherwise.
    # Names are namespaced as well as emails. Namespacing only the emails made the
    # tenancy assertion below vacuous: both organizations produced "In 0000",
    # "In 0001", ... so a query that returned the WRONG organization's three users
    # satisfied the test. Caught in review.
    tag = next(_seeds)
    pw = await hash_password("TestPass1234")
    created = {"with_dept": [], "without_dept": [], "inactive": [], "depts": []}
    async with SessionLocal() as db:
        rows = [Department(org_id=org.id, name=f"Dept {tag}-{i}") for i in range(depts)]
        db.add_all(rows)
        await db.flush()
        dept_ids = [d.id for d in rows]
        created["depts"] = [d.name for d in rows]
        users = []
        for i in range(with_dept):
            users.append(
                User(
                    email=f"in{i}-{tag}@r.com",
                    display_name=f"In {tag}-{i:04d}",
                    password_hash=pw,
                    role=UserRole.member,
                    org_id=org.id,
                    dept_id=dept_ids[i % depts],
                    is_active=True,
                )
            )
            created["with_dept"].append(f"In {tag}-{i:04d}")
        for i in range(without_dept):
            users.append(
                User(
                    email=f"nodept{i}-{tag}@r.com",
                    display_name=f"NoDept {tag}-{i:04d}",
                    password_hash=pw,
                    role=UserRole.member,
                    org_id=org.id,
                    dept_id=None,
                    is_active=True,
                )
            )
            created["without_dept"].append(f"NoDept {tag}-{i:04d}")
        for i in range(inactive):
            users.append(
                User(
                    email=f"gone{i}-{tag}@r.com",
                    display_name=f"Gone {tag}-{i:04d}",
                    password_hash=pw,
                    role=UserRole.member,
                    org_id=org.id,
                    dept_id=dept_ids[0],
                    is_active=False,
                )
            )
            created["inactive"].append(f"Gone {tag}-{i:04d}")
        db.add_all(users)
        await db.commit()
    created["dept_ids"] = dept_ids
    return created


async def test_departmentless_users_are_excluded_by_the_query_not_by_the_loop():
    """They were never in the response; they were just paid for first.

    The endpoint groups by department, so a user with no department has nowhere to
    appear — but it loaded every one of them and dropped them in Python. On a
    seeded 2,000-user organization with a tenth departmentless that was 200 rows
    fetched and discarded per request.

    Asserted through the response, which is the only thing a caller can see: the
    departmentless users are absent, and everyone else is present.
    """
    org = await make_org("Roster Co")
    seeded = await _seed(org, with_dept=6, without_dept=4)

    async with _superadmin() as c:
        resp = await c.get(f"/api/admin/organizations/{org.id}/users")

    assert resp.status_code == 200, resp.text
    served = [u["display_name"] for d in resp.json() for u in d["users"]]
    assert sorted(served) == sorted(seeded["with_dept"])
    assert not set(served) & set(seeded["without_dept"])


async def test_the_response_carries_no_credential_material():
    """The query used to be `select(User)`, which reads all fifteen mapped columns
    including password_hash for every active member. Nothing leaked — the hash is
    never serialized — but it was read and held in memory on a directory render.

    This asserts the observable half: no field of any row carries a hash, and the
    row shape is exactly the five documented keys.
    """
    org = await make_org("Hash Co")
    await _seed(org, with_dept=4, without_dept=0)

    async with _superadmin() as c:
        resp = await c.get(f"/api/admin/organizations/{org.id}/users")

    rows = [u for d in resp.json() for u in d["users"]]
    assert rows, "expected a populated roster"
    for row in rows:
        assert set(row) == {"id", "display_name", "email", "avatar_url", "role"}
        assert not any("$2b$" in str(v) for v in row.values()), "a bcrypt hash reached the wire"


async def test_inactive_members_and_other_orgs_stay_out():
    org = await make_org("Scoped Co")
    other = await make_org("Other Co")
    target = await _seed(org, with_dept=3, without_dept=0, inactive=2)
    outsider = await _seed(other, with_dept=3, without_dept=0)

    async with _superadmin() as c:
        resp = await c.get(f"/api/admin/organizations/{org.id}/users")

    served = [u["display_name"] for d in resp.json() for u in d["users"]]
    # Identity, not just a count. A count of three would also be satisfied by the
    # WRONG organization's three users, which is what made the earlier version of
    # this assertion prove nothing about the org_id predicate.
    assert sorted(served) == sorted(target["with_dept"]), served
    assert not set(served) & set(outsider["with_dept"]), "another organization's users were served"
    assert not set(served) & set(target["inactive"]), "an inactive member was served"

    # The DEPARTMENTS too, which is the half a user-only assertion cannot see.
    #
    # Remove Department.org_id == oid and the foreign USERS are still filtered out
    # by the user query, but every other organization's DEPARTMENT NAMES appear in
    # the response. Verified by removing it: a user-identity assertion alone passes
    # right through that leak, which is what the first version of this test did.
    #
    # The other predicate, User.org_id == oid, is NOT redundant, though this
    # fixture cannot show it — see
    # test_a_foreign_user_pointed_at_our_department_is_still_excluded.
    names = [d["name"] for d in resp.json()]
    assert sorted(names) == sorted(target["depts"]), names
    assert not set(names) & set(outsider["depts"]), "another organization's departments were served"


async def test_departments_with_no_members_are_still_listed():
    """The picker renders the tree, so an empty department must still appear —
    the SQL-side dept filter must not turn into an inner join by accident."""
    org = await make_org("Empty Dept Co")
    await _seed(org, with_dept=2, without_dept=0, depts=4)

    async with _superadmin() as c:
        resp = await c.get(f"/api/admin/organizations/{org.id}/users")

    body = resp.json()
    assert len(body) == 4, f"expected all four departments, got {len(body)}"
    assert sum(1 for d in body if d["users"] == []) == 2


async def test_an_organization_with_no_users_returns_its_departments():
    org = await make_org("Nobody Co")
    await _seed(org, with_dept=0, without_dept=0, depts=2)

    async with _superadmin() as c:
        resp = await c.get(f"/api/admin/organizations/{org.id}/users")

    assert resp.status_code == 200
    assert [d["users"] for d in resp.json()] == [[], []]


async def test_a_bad_org_id_is_rejected():
    async with _superadmin() as c:
        resp = await c.get("/api/admin/organizations/not-a-uuid/users")
    assert resp.status_code == 400, resp.text


async def test_the_roster_query_filters_in_sql_and_never_reads_the_password_hash():
    """The one test here that fails against the previous implementation.

    Every other test in this file passes both before and after, because the
    response is deliberately unchanged — the fix removes work, not behaviour. So
    the property has to be asserted where the change actually is: in the statement
    sent to Postgres.

    White-box on purpose, and narrow: it looks only at the SELECT against `users`
    raised by this endpoint, and asserts the two things that moved — the
    department filter is a WHERE clause rather than a Python `continue`, and
    password_hash is not among the columns read.
    """
    from sqlalchemy import event

    from app.db.session import engine

    org = await make_org("SQL Co")
    await _seed(org, with_dept=3, without_dept=3)

    statements: list[str] = []

    def _capture(conn, cursor, statement, params, context, executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", _capture)
    try:
        async with _superadmin() as c:
            resp = await c.get(f"/api/admin/organizations/{org.id}/users")
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _capture)

    assert resp.status_code == 200, resp.text
    roster = [
        s
        for s in statements
        if "FROM users" in s and "users.display_name" in s and "ORDER BY users.display_name" in s
    ]
    assert roster, f"did not observe the roster query among {len(statements)} statements"
    sql = roster[-1]
    assert "dept_id IS NOT NULL" in sql, f"department filter is not in SQL:\n{sql}"
    assert "password_hash" not in sql, f"the roster query still reads the password hash:\n{sql}"


async def test_a_foreign_user_pointed_at_our_department_is_still_excluded():
    """The case that makes `User.org_id == oid` load-bearing rather than decorative.

    I first read that predicate as redundant: the reply is assembled by iterating
    THIS organization's departments, so deleting it changed nothing I could observe
    and every other test in this file still passed. That reasoning was wrong, and
    the schema is why. users.dept_id is a plain foreign key to departments.id —
    verified against the live schema, there is no composite key and no CHECK tying
    users.org_id to the department's org_id — so a user in one organization holding
    a dept_id that points at ANOTHER organization's department is a representable
    state, and admin paths that move a user between organizations can produce it.

    Such a user IS reachable through the department grouping, because their dept_id
    matches a department this endpoint returns. Only User.org_id == oid keeps them
    out. Verified: removing that predicate fails this test and nothing else.
    """
    org = await make_org("Ours Co")
    other = await make_org("Theirs Co")
    ours = await _seed(org, with_dept=2, without_dept=0)
    await _seed(other, with_dept=1, without_dept=0)

    pw = await hash_password("TestPass1234")
    async with SessionLocal() as db:
        # In `other`, but filed under one of OUR departments.
        db.add(
            User(
                email="misfiled@theirs.com",
                display_name="Misfiled Outsider",
                password_hash=pw,
                role=UserRole.member,
                org_id=other.id,
                dept_id=ours["dept_ids"][0],
                is_active=True,
            )
        )
        await db.commit()

    async with _superadmin() as c:
        resp = await c.get(f"/api/admin/organizations/{org.id}/users")

    assert resp.status_code == 200, resp.text
    served = [u["display_name"] for d in resp.json() for u in d["users"]]
    assert "Misfiled Outsider" not in served, (
        "a user from another organization was served because their dept_id "
        "pointed at one of this organization's departments"
    )
    assert sorted(served) == sorted(ours["with_dept"]), served


async def test_the_roster_is_capped():
    """The last roster in the API without a ceiling now has one.

    org_admin.list_users and the admin user list paginate at limit <= 100;
    contacts.list_contacts caps at 500 after measuring the unbounded version at
    24,999 rows and 5.5 MB on a 25,000-user tenant. This one returned every active
    member of an organization.
    """
    org = await make_org("Capped Co")
    await _seed(org, with_dept=12, without_dept=0)

    async with _superadmin() as c:
        resp = await c.get(f"/api/admin/organizations/{org.id}/users", params={"limit": 5})

    served = [u for d in resp.json() for u in d["users"]]
    assert len(served) == 5, len(served)


async def test_the_cap_has_a_ceiling_of_its_own():
    """So a caller cannot opt back into the old behaviour with a large number —
    the same reasoning contacts.list_contacts documents for its own le=500."""
    org = await make_org("Ceiling Co")
    await _seed(org, with_dept=1, without_dept=0)

    async with _superadmin() as c:
        assert (
            await c.get(f"/api/admin/organizations/{org.id}/users", params={"limit": 501})
        ).status_code == 422
        assert (
            await c.get(f"/api/admin/organizations/{org.id}/users", params={"limit": 0})
        ).status_code == 422


async def test_search_narrows_by_name_and_by_email():
    """The cap is only safe because this exists.

    Truncating the response while the caller filtered in the browser would have
    made every member past the ceiling silently unpickable. contacts.list_contacts
    states the precondition — "Every picker in the app already narrows with
    `search`" — and this picker did not until now.
    """
    org = await make_org("Search Co")
    seeded = await _seed(org, with_dept=6, without_dept=0)
    wanted = seeded["with_dept"][3]  # "In <tag>-0003"

    async with _superadmin() as c:
        by_name = await c.get(f"/api/admin/organizations/{org.id}/users", params={"search": wanted})
        by_email = await c.get(f"/api/admin/organizations/{org.id}/users", params={"search": "in3-"})

    assert [u["display_name"] for d in by_name.json() for u in d["users"]] == [wanted]
    emails = [u["email"] for d in by_email.json() for u in d["users"]]
    assert emails and all("in3-" in e for e in emails), emails


async def test_search_wildcards_are_matched_literally():
    """A raw LIKE pattern would make `%` select the whole roster back."""
    org = await make_org("Wildcard Co")
    await _seed(org, with_dept=5, without_dept=0)

    async with _superadmin() as c:
        pct = await c.get(f"/api/admin/organizations/{org.id}/users", params={"search": "%"})
        underscore = await c.get(f"/api/admin/organizations/{org.id}/users", params={"search": "_"})

    assert [u for d in pct.json() for u in d["users"]] == [], "% behaved as a wildcard"
    assert [u for d in underscore.json() for u in d["users"]] == [], "_ behaved as a wildcard"


async def test_every_department_is_returned_even_when_the_search_empties_it():
    """The picker renders the tree, so a department whose members are all filtered
    out still has to appear — otherwise an admin cannot tell an empty result from a
    department that has gone missing."""
    org = await make_org("Tree Co")
    seeded = await _seed(org, with_dept=4, without_dept=0, depts=4)

    async with _superadmin() as c:
        resp = await c.get(
            f"/api/admin/organizations/{org.id}/users",
            params={"search": seeded["with_dept"][0]},
        )

    body = resp.json()
    assert len(body) == 4, "departments were filtered by the search term"
    assert sum(len(d["users"]) for d in body) == 1
