"""Durable, worker-agnostic deadlines for the call lifecycle.

Two things in a call must happen at a wall-clock moment even when nobody is
looking: a ringing call has to time out, and a participant whose link dropped has
to be given a grace window and then resolved one way or the other.

Both used to be `asyncio` tasks in a module-level dict (`calls._ring_timers`).
That has three failure modes, all of which present to the user as "the call never
connects" or "the call rang forever":

  * **Invisible to other workers.** The accept arrives on whichever worker holds
    the callee's socket, so `_ring_timers.pop()` there finds nothing and the timer
    on the *caller's* worker keeps running.
  * **Lost on restart.** A deploy, a crash, or a worker recycle drops every
    pending timer. The row stays `ringing` forever: the caller hears ringback with
    no timeout and the row is never written to history.
  * **Nothing to sweep.** With no record of the deadline anywhere durable, a call
    left mid-state can only be fixed by hand.

Deadlines now live in Redis sorted sets scored by their firing time, so any
worker can fire any deadline, a worker dying loses nothing, and the in-process
`asyncio` task is kept purely as a latency optimisation rather than as the
mechanism. Firing twice is harmless by construction: every handler re-reads the
row and is a no-op unless the state it acts on is still true.
"""

from __future__ import annotations

import logging
import time

from app.realtime.redis_bus import degrade_on_outage, get_redis

logger = logging.getLogger(__name__)

# One sorted set per deadline kind. Members are opaque strings owned by the
# caller; scores are absolute unix seconds.
RING = "call:deadline:ring"
GRACE = "call:deadline:grace"

# Claiming has to be atomic across workers, or two sweepers both fire the same
# deadline — harmless for our idempotent handlers, but it doubles the log noise
# and the DB reads for no benefit. ZRANGEBYSCORE+ZREM in one script gives us
# exactly-one-claimant without a distributed lock.
_CLAIM_DUE = """
local due = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, ARGV[2])
if #due > 0 then
  redis.call('ZREM', KEYS[1], unpack(due))
end
return due
"""


def _now() -> float:
    return time.time()


async def schedule(kind: str, member: str, seconds: float) -> None:
    """Record that `member` is due in `seconds`. Re-scheduling moves the deadline."""
    async with degrade_on_outage(f"deadline.schedule({kind})"):
        await get_redis().zadd(kind, {member: _now() + seconds})


async def cancel(kind: str, member: str) -> None:
    """Drop a deadline. Safe when it was never scheduled or has already fired."""
    async with degrade_on_outage(f"deadline.cancel({kind})"):
        await get_redis().zrem(kind, member)


async def remaining(kind: str, member: str) -> float | None:
    """Seconds until `member` fires, or None when it isn't scheduled.

    Used to tell a reconnecting client how much of its ring window is left, so a
    resumed ringing screen counts down against the server's clock rather than
    restarting a 45-second timer of its own.
    """
    score = None
    async with degrade_on_outage(f"deadline.remaining({kind})"):
        score = await get_redis().zscore(kind, member)
    if score is None:
        return None
    return max(0.0, float(score) - _now())


async def claim_due(kind: str, limit: int = 100) -> list[str]:
    """Atomically take every deadline of `kind` that is due, up to `limit`."""
    members: list[str] = []
    async with degrade_on_outage(f"deadline.claim_due({kind})"):
        raw = await get_redis().eval(_CLAIM_DUE, 1, kind, str(_now()), str(limit))
        members = [m if isinstance(m, str) else m.decode() for m in (raw or [])]
    return members
