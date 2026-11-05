"""The background Web Push fan-out, and what it holds while it runs.

services/push.py had no test coverage of any kind. These cover the property that
made the fan-out worth moving off the request path in the first place — that it
must not hold a pooled database connection across N remote HTTPS POSTs — plus the
pruning of endpoints the push service has rejected.
"""

import pytest
from sqlalchemy import select

import app.services.push as push_mod
from app.db.models import PushSubscription
from app.db.session import SessionLocal, engine
from tests.conftest import make_org, make_user


async def _add_subscription(user_id, endpoint: str) -> None:
    async with SessionLocal() as db:
        db.add(PushSubscription(user_id=user_id, endpoint=endpoint, keys={"p256dh": "k", "auth": "a"}))
        await db.commit()


@pytest.fixture
def push_configured(monkeypatch):
    """Make the module believe Web Push is available and configured.

    Both gates are real: dispatch and _push_in_background return immediately
    without pywebpush or a VAPID private key, so every assertion below would pass
    vacuously against a no-op.
    """
    monkeypatch.setattr(push_mod, "_HAS_WEBPUSH", True)
    settings = push_mod.get_settings()
    monkeypatch.setattr(settings, "vapid_private_key", "test-private-key", raising=False)
    monkeypatch.setattr(settings, "vapid_subject", "mailto:test@rhythmrx.ai", raising=False)


async def test_fan_out_holds_no_pooled_connection(push_configured, monkeypatch):
    """The regression this file exists for.

    dispatch_push_to_users moved the fan-out off the request because awaiting it
    inline pinned the request's session for its whole duration. Giving the
    background task its own session MOVED that pinning rather than removing it:
    one pooled connection sat checked out, and idle, for every remote POST.

    Sends are bounded to 10 at a time and each may spend 5s on a black-holed
    endpoint, so the wall time scales with the subscription count — and nothing
    caps that, since cross-org groups bound only their minimum size. A pool of 10
    is exhausted by a handful of concurrent fan-outs, and then no route can get a
    session, which is every route.
    """
    org = await make_org("Push Co")
    user = await make_user("push-pool@a.com", org_id=org.id)
    await _add_subscription(user.id, "https://push.example.test/f/pool")

    observed: dict = {}

    async def spy_fan_out(targets, payload):
        # Measured INSIDE the network phase, which is the only moment that matters.
        observed["checked_out"] = engine.pool.checkedout()
        observed["targets"] = list(targets)
        return []

    monkeypatch.setattr(push_mod, "_fan_out", spy_fan_out)

    baseline = engine.pool.checkedout()
    await push_mod._push_in_background([user.id], {"title": "Dr Okafor", "body": "bed 4"})

    assert observed["checked_out"] == baseline, (
        "the fan-out ran with a database connection still checked out: "
        f"{observed['checked_out']} vs a baseline of {baseline}"
    )
    # And it really did have work to do, so the assertion above is not vacuous.
    assert [t.endpoint for t in observed["targets"]] == ["https://push.example.test/f/pool"]


async def test_targets_carry_no_orm_state(push_configured, monkeypatch):
    """What reaches the worker threads is detached from the session.

    _deliver hands its target to a thread via anyio.to_thread. An ORM instance
    there is a latent MissingGreenlet — an attribute that has to go back to the
    database raises, from a thread, and after the change above there is
    deliberately no session left to go back to.
    """
    org = await make_org("Detached Co")
    user = await make_user("push-detached@a.com", org_id=org.id)
    await _add_subscription(user.id, "https://push.example.test/f/detached")

    seen: list = []

    async def spy_fan_out(targets, payload):
        seen.extend(targets)
        return []

    monkeypatch.setattr(push_mod, "_fan_out", spy_fan_out)
    await push_mod._push_in_background([user.id], {"title": "x"})

    assert len(seen) == 1
    target = seen[0]
    assert not isinstance(target, PushSubscription)
    # Plain values, readable with no session anywhere in sight.
    assert target.endpoint == "https://push.example.test/f/detached"
    assert target.keys == {"p256dh": "k", "auth": "a"}


async def test_rejected_endpoints_are_pruned(push_configured, monkeypatch):
    """A 404/410 from the push service means the subscription is gone for good.

    The prune now runs in its own short session after the network, so this also
    proves the third phase still happens once the fan-out no longer owns one.
    """
    org = await make_org("Prune Co")
    user = await make_user("push-prune@a.com", org_id=org.id)
    await _add_subscription(user.id, "https://push.example.test/f/dead")
    await _add_subscription(user.id, "https://push.example.test/f/live")

    async def spy_fan_out(targets, payload):
        # Rejections are reported as the loaded targets, so the prune can match on
        # identity and owner rather than on an endpoint that may have changed hands.
        return [t for t in targets if t.endpoint.endswith("/dead")]

    monkeypatch.setattr(push_mod, "_fan_out", spy_fan_out)
    await push_mod._push_in_background([user.id], {"title": "x"})

    async with SessionLocal() as db:
        remaining = (
            (await db.execute(select(PushSubscription.endpoint).where(PushSubscription.user_id == user.id)))
            .scalars()
            .all()
        )
    assert sorted(remaining) == ["https://push.example.test/f/live"]


async def test_no_subscriptions_touches_nothing(push_configured, monkeypatch):
    """No subscriptions means no fan-out and no second session."""
    org = await make_org("Empty Co")
    user = await make_user("push-empty@a.com", org_id=org.id)

    called = []

    async def spy_fan_out(targets, payload):
        called.append(list(targets))
        return []

    monkeypatch.setattr(push_mod, "_fan_out", spy_fan_out)
    await push_mod._push_in_background([user.id], {"title": "x"})

    assert called == []


async def test_background_push_never_raises(push_configured, monkeypatch):
    """Best-effort by contract: the caller has already returned and there is
    nobody left to surface an exception to."""
    org = await make_org("Boom Co")
    user = await make_user("push-boom@a.com", org_id=org.id)
    await _add_subscription(user.id, "https://push.example.test/f/boom")

    async def exploding_fan_out(targets, payload):
        raise RuntimeError("the push service fell over")

    monkeypatch.setattr(push_mod, "_fan_out", exploding_fan_out)

    # No assertion needed beyond this returning at all.
    await push_mod._push_in_background([user.id], {"title": "x"})


async def test_prune_leaves_a_rebound_subscription_alone(push_configured, monkeypatch):
    """A stale rejection must not delete a subscription that has changed hands.

    Endpoints are globally unique and api/notifications.py re-binds one to whoever
    presents it, so a row can move to another user — in another organisation —
    between _load_targets and a delayed 404/410 arriving.

    Reachable on a shared workstation: if a sign-out teardown does not complete the
    browser keeps its subscription, the next person to sign in on that profile gets
    the SAME endpoint back from pushManager.subscribe(), and the route moves the row
    to them. Pruning by endpoint alone then deleted a live subscription belonging to
    somebody else, who silently stopped receiving push with no way to know why.
    """
    endpoint = "https://push.example.test/f/shared-device"
    org_a = await make_org("Rebind Org A")
    org_b = await make_org("Rebind Org B")
    alice = await make_user("rebind-alice@a.com", org_id=org_a.id)
    bob = await make_user("rebind-bob@b.com", org_id=org_b.id)
    await _add_subscription(alice.id, endpoint)

    async def rebinding_fan_out(targets, payload):
        # Mid-fan-out, the endpoint is re-bound to Bob — exactly what
        # POST /api/notifications/subscribe does when he presents it.
        async with SessionLocal() as db:
            row = (
                await db.execute(select(PushSubscription).where(PushSubscription.endpoint == endpoint))
            ).scalar_one()
            row.user_id = bob.id
            await db.commit()
        # Alice's send then comes back 410, reported against the target as LOADED.
        return list(targets)

    monkeypatch.setattr(push_mod, "_fan_out", rebinding_fan_out)
    await push_mod._push_in_background([alice.id], {"title": "x"})

    async with SessionLocal() as db:
        row = (
            await db.execute(select(PushSubscription).where(PushSubscription.endpoint == endpoint))
        ).scalar_one_or_none()

    assert row is not None, "a stale rejection deleted a subscription owned by another org's user"
    assert row.user_id == bob.id, "the subscription should still belong to Bob"
