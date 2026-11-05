"""Superadmin cross-org group management: invariants, validation, audit trail.

The create path is the one with teeth — it enforces >=2 organizations, >=2
members, >=1 admin and >=1 member per organization. These tests cover the places
the sibling paths did not hold the same line.
"""

import contextlib

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError

from app.db.models import AuditLog, Conversation, UserRole
from app.db.session import SessionLocal
from app.main import app
from tests.conftest import CSRF, login, make_user


@contextlib.asynccontextmanager
async def _superadmin():
    await make_user("root@x.com", role=UserRole.superadmin)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c.headers.update(CSRF)
        await login(c, "root@x.com")
        yield c


def _payload(users, org_ids):
    return {
        "name": "Joint Programme",
        "org_ids": org_ids,
        "members": [
            {"user_id": str(users["alice"].id), "role": "admin"},
            {"user_id": str(users["carol"].id), "role": "member"},
        ],
    }


async def test_duplicate_org_ids_cannot_create_a_single_org_cross_org_group(client, two_orgs_with_users):
    """The >=2-organizations check ran on the raw request, and the de-duplication
    ran after it, so passing one organization twice satisfied the check and then
    collapsed to a group spanning a single org — a cross-org group that crosses
    nothing. This is the same defect the members check was fixed for: count what
    survives filtering, not what was sent."""
    users = two_orgs_with_users
    org_a = str(users["org_a"].id)
    async with _superadmin() as root:
        resp = await root.post(
            "/api/admin/cross-org-groups",
            json={
                "name": "Not Actually Cross Org",
                "org_ids": [org_a, org_a],
                "members": [
                    {"user_id": str(users["alice"].id), "role": "admin"},
                    {"user_id": str(users["bob"].id), "role": "member"},
                ],
            },
        )
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == "At least 2 organizations required"

        # and nothing was persisted
        async with SessionLocal() as db:
            assert (await db.execute(select(Conversation))).scalars().all() == []

        # The genuine two-org case still works, including when one is sent twice.
        resp = await root.post(
            "/api/admin/cross-org-groups",
            json=_payload(users, [org_a, org_a, str(users["org_b"].id)]),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["allowed_org_ids"] == [org_a, str(users["org_b"].id)]


async def test_group_name_cannot_be_emptied_by_an_update(client, two_orgs_with_users):
    """create_group rejects a blank name; the update path applied one. A group
    whose name sanitizes away renders as a nameless row in every member's
    sidebar, so both paths hold the same line."""
    users = two_orgs_with_users
    async with _superadmin() as root:
        resp = await root.post(
            "/api/admin/cross-org-groups",
            json=_payload(users, [str(users["org_a"].id), str(users["org_b"].id)]),
        )
        assert resp.status_code == 200, resp.text
        group = resp.json()["_id"]

        # Empty, whitespace-only, and markup that sanitizes to nothing.
        for name in ("", "   ", "<b></b>"):
            resp = await root.put(f"/api/admin/cross-org-groups/{group}", json={"name": name})
            assert resp.status_code == 400, f"{name!r} was accepted: {resp.text}"
            assert resp.json()["detail"] == "Group name required"

        assert (await root.get(f"/api/admin/cross-org-groups/{group}")).json()["name"] == "Joint Programme"

        # A real rename is untouched.
        resp = await root.put(f"/api/admin/cross-org-groups/{group}", json={"name": "Renamed"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["name"] == "Renamed"


async def test_archiving_a_group_is_audited(client, two_orgs_with_users):
    """Archiving drops the group out of every member's list — the same state
    change PUT /{id} makes and audits. Going through /archive left no record of
    who did it."""
    users = two_orgs_with_users
    async with _superadmin() as root:
        resp = await root.post(
            "/api/admin/cross-org-groups",
            json=_payload(users, [str(users["org_a"].id), str(users["org_b"].id)]),
        )
        assert resp.status_code == 200, resp.text
        group = resp.json()["_id"]

        assert (await root.post(f"/api/admin/cross-org-groups/{group}/archive")).json()["is_active"] is False
        assert (await root.post(f"/api/admin/cross-org-groups/{group}/archive")).json()["is_active"] is True

    async with SessionLocal() as db:
        actions = (
            (
                await db.execute(
                    select(AuditLog.action).where(AuditLog.target == group).order_by(AuditLog.created_at)
                )
            )
            .scalars()
            .all()
        )
    assert "cross_org_group_archived" in actions
    assert "cross_org_group_unarchived" in actions


async def test_a_deleted_group_cannot_be_resurrected_by_an_archive_toggle(client, two_orgs_with_users):
    """Delete and archive both set is_active = False, and that was the only state
    either wrote — so /archive read a deleted group as merely archived and
    toggled it back to active, re-broadcasting it to every member. Deletion is
    now a distinct durable state and the admin paths refuse to load it."""
    users = two_orgs_with_users
    async with _superadmin() as root:
        resp = await root.post(
            "/api/admin/cross-org-groups",
            json=_payload(users, [str(users["org_a"].id), str(users["org_b"].id)]),
        )
        assert resp.status_code == 200, resp.text
        group = resp.json()["_id"]

        assert (await root.delete(f"/api/admin/cross-org-groups/{group}")).status_code == 200

        # Every path that mutates the group, not just the toggle that exposed it.
        assert (await root.post(f"/api/admin/cross-org-groups/{group}/archive")).status_code == 404
        assert (
            await root.put(f"/api/admin/cross-org-groups/{group}", json={"is_active": True})
        ).status_code == 404
        assert (await root.get(f"/api/admin/cross-org-groups/{group}")).status_code == 404
        assert (await root.delete(f"/api/admin/cross-org-groups/{group}")).status_code == 404

        # And it is listed under neither status — "archived" is where it used to
        # surface with an unarchive button next to it.
        for status in ("active", "archived"):
            listing = (await root.get("/api/admin/cross-org-groups", params={"status": status})).json()
            assert [g["_id"] for g in listing["data"]] == []
            assert listing["total"] == 0

    async with SessionLocal() as db:
        conv = (await db.execute(select(Conversation))).scalar_one()
        assert conv.deleted_at is not None
        assert conv.is_active is False


async def test_archiving_still_round_trips_for_a_group_that_was_never_deleted(client, two_orgs_with_users):
    """The tombstone check must not catch ordinary archived groups: an archived
    group stays listable under status=archived and unarchivable."""
    users = two_orgs_with_users
    async with _superadmin() as root:
        resp = await root.post(
            "/api/admin/cross-org-groups",
            json=_payload(users, [str(users["org_a"].id), str(users["org_b"].id)]),
        )
        group = resp.json()["_id"]

        assert (await root.post(f"/api/admin/cross-org-groups/{group}/archive")).json()["is_active"] is False
        listing = (await root.get("/api/admin/cross-org-groups", params={"status": "archived"})).json()
        assert [g["_id"] for g in listing["data"]] == [group]

        assert (await root.post(f"/api/admin/cross-org-groups/{group}/archive")).json()["is_active"] is True
        listing = (await root.get("/api/admin/cross-org-groups", params={"status": "active"})).json()
        assert [g["_id"] for g in listing["data"]] == [group]


async def test_a_cross_org_group_cannot_be_created_above_the_member_cap(
    client, two_orgs_with_users, monkeypatch
):
    """Cross-org creation had a floor and no ceiling.

    Every size rule in this file is a minimum — two orgs, two members, one admin,
    one member per org — while a same-org group has been capped at
    MAX_GROUP_MEMBERS all along. That asymmetry is named in two other modules as
    the reason their own tails were reachable: services/messaging.py measured a
    per-recipient set rebuild at 703ms of event-loop time for 5,000 recipients and
    says regular groups stayed in single-digit milliseconds only because of the
    cap, and services/push.py notes the fan-out duration scales with a
    subscription count nothing bounds.

    The cap is patched down rather than creating 257 users, which would make this
    test slower than the rest of the file put together.
    """
    from app.api import cross_org as cross_org_mod

    monkeypatch.setattr(cross_org_mod, "MAX_GROUP_MEMBERS", 2)

    users = two_orgs_with_users
    org_ids = [str(users["org_a"].id), str(users["org_b"].id)]
    payload = _payload(users, org_ids)
    payload["members"] = [
        {"user_id": str(users["alice"].id), "role": "admin"},
        {"user_id": str(users["bob"].id), "role": "member"},
        {"user_id": str(users["carol"].id), "role": "member"},
    ]

    async with _superadmin() as c:
        resp = await c.post("/api/admin/cross-org-groups", json=payload)

    assert resp.status_code == 400, resp.text
    assert "Maximum 2 members" in resp.text

    # And nothing was written on the way to refusing.
    async with SessionLocal() as db:
        rows = (
            (await db.execute(select(Conversation).where(Conversation.name == "Joint Programme")))
            .scalars()
            .all()
        )
    assert rows == []


async def test_members_cannot_be_added_past_the_cap(client, two_orgs_with_users, monkeypatch):
    """The add path had no ceiling either, so a capped group could be grown past it.

    Counted against the CURRENT membership, not against the request alone: two
    members already in the group plus one more is over a cap of two.
    """
    from app.api import cross_org as cross_org_mod

    users = two_orgs_with_users
    org_ids = [str(users["org_a"].id), str(users["org_b"].id)]

    async with _superadmin() as c:
        created = await c.post("/api/admin/cross-org-groups", json=_payload(users, org_ids))
        assert created.status_code in (200, 201), created.text
        group_id = created.json()["_id"]

        # The cap only comes down AFTER a legitimate two-member group exists, which
        # is also the state a group predating this check would be in.
        monkeypatch.setattr(cross_org_mod, "MAX_GROUP_MEMBERS", 2)

        resp = await c.post(
            f"/api/admin/cross-org-groups/{group_id}/members",
            json={"members": [{"user_id": str(users["bob"].id), "role": "member"}]},
        )

    assert resp.status_code == 400, resp.text
    assert "Maximum 2 members" in resp.text


async def test_an_add_that_resolves_to_nobody_is_still_allowed_at_the_cap(
    client, two_orgs_with_users, monkeypatch
):
    """The cap must not turn a no-op into an error.

    Unknown, inactive and already-present ids are dropped before the count, so a
    request that adds nobody stays a no-op even when the group is full. Checking
    the raw request instead would have refused it.
    """
    from app.api import cross_org as cross_org_mod

    users = two_orgs_with_users
    org_ids = [str(users["org_a"].id), str(users["org_b"].id)]

    async with _superadmin() as c:
        created = await c.post("/api/admin/cross-org-groups", json=_payload(users, org_ids))
        group_id = created.json()["_id"]
        monkeypatch.setattr(cross_org_mod, "MAX_GROUP_MEMBERS", 2)

        # Alice is already a member — she resolves away, so nothing is added.
        resp = await c.post(
            f"/api/admin/cross-org-groups/{group_id}/members",
            json={"members": [{"user_id": str(users["alice"].id), "role": "member"}]},
        )

    assert resp.status_code == 200, resp.text


async def test_add_members_holds_a_row_lock_while_it_reads_the_membership(
    client, two_orgs_with_users, monkeypatch
):
    """The cap is a read-check-insert, so it has to be serialized to be a cap.

    Under PostgreSQL's default READ COMMITTED two concurrent adds each see the
    same pre-insert count. With one slot free both observe it, both pass the
    check, and the group ends up over the limit. add_members takes FOR UPDATE on
    the conversation row before reading the membership.

    RESTORED IN BATCH 42. This test was written in Batch 41 and withheld: it, and
    every other design that actually proved the lock, deterministically broke
    test_concurrent_identical_sends_store_one_message in the same run. That was
    not this test's fault — the idempotency backstop was guarding db.commit()
    while the unique index fires at db.flush(), so the loser of any forced race
    got an IntegrityError out of the endpoint. Anything scheduled before that test
    exposed it. Fixed in this batch, so the coverage can come back.

    Asserted by probing for the lock from a SEPARATE transaction at the moment the
    membership is read, rather than by racing two requests: SELECT ... FOR UPDATE
    NOWAIT raises instead of blocking when the row is already locked, which makes
    the observation deterministic. A racing version was tried first and passed with
    the lock REMOVED, because the event loop ran the first request to completion
    before the second began.
    """
    from app.api import cross_org as cross_org_mod
    from app.db.models import Conversation

    users = two_orgs_with_users
    org_ids = [str(users["org_a"].id), str(users["org_b"].id)]
    observed: dict = {}
    original_participant_ids = cross_org_mod.participant_ids

    async def probing_participant_ids(db, conv_id):
        # A separate session is a separate transaction, which is the only way to
        # see somebody else's row lock.
        async with SessionLocal() as probe:
            try:
                await probe.execute(
                    select(Conversation.id).where(Conversation.id == conv_id).with_for_update(nowait=True)
                )
                observed["locked"] = False
            except DBAPIError as exc:
                # ONLY 55P03 (lock_not_available) counts as evidence. A bare
                # `except Exception` here would let a typo in the query, a dropped
                # connection or a missing table read as "the row was locked", so the
                # test would pass without ever observing contention — which is the
                # same wrong-reason-green this whole batch is about.
                if getattr(exc.orig, "sqlstate", None) != "55P03":
                    raise
                observed["locked"] = True
            finally:
                await probe.rollback()
        return await original_participant_ids(db, conv_id)

    async with _superadmin() as c:
        created = await c.post("/api/admin/cross-org-groups", json=_payload(users, org_ids))
        group_id = created.json()["_id"]

        monkeypatch.setattr(cross_org_mod, "participant_ids", probing_participant_ids)
        resp = await c.post(
            f"/api/admin/cross-org-groups/{group_id}/members",
            json={"members": [{"user_id": str(users["bob"].id), "role": "member"}]},
        )

    assert resp.status_code == 200, resp.text
    assert observed.get("locked") is True, (
        "the membership was read without holding the conversation row lock, so two "
        "concurrent adds can both take the last slot"
    )
