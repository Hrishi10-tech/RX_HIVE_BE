"""Redis-backed fixed-window rate limiting.

The Mongo build shipped a limiter attached to a route path that never matched —
dead code. This one is a plain dependency wired directly onto the routes it
protects, with a test proving it fires.

It also fails OPEN now. The limiter runs BEFORE the work a request came to do, so
an unhandled Redis error escaped it as a 500 on /api/auth/login and
/api/auth/refresh, and both clients read a failed refresh as a dead session: one
broker restart signed the whole estate out and made everyone retype their
password. Rate limiting is a best-effort control, so an unenforced window during
an outage is the cheaper failure by a wide margin.
"""

from fastapi import HTTPException, Request

from app.core.config import get_settings
from app.realtime.redis_bus import degrade_on_outage, get_redis


def _client_ip(request: Request) -> str:
    """Real client IP. Behind our own reverse proxy (Caddy) the peer is the proxy,
    so the address has to come out of X-Forwarded-For; otherwise use the peer.
    Without this, every request shares the proxy's IP and the limiter degrades
    into a single global bucket.

    THE LAST HOP, NOT THE FIRST. A proxy APPENDS to X-Forwarded-For, so the
    header a request arrives with is `<whatever the client sent>, <peer Caddy
    saw>`. The leftmost entry is therefore supplied by the caller and means
    nothing; reading it made every limiter in this module opt-in, because one
    header the attacker chooses per request mints a fresh counter per request.
    The rightmost entry is the one Caddy wrote from the TCP peer, and is the only
    value in the header the caller cannot forge.

    This is correct for exactly one trusted proxy in front of the app, which is
    what trust_proxy already documents as its precondition (Caddy is the sole
    ingress — see infra/Caddyfile.prod and the security group in
    infra/terraform/network.tf). Adding a second proxy layer means taking the
    (n+1)th hop from the right, and this function would have to learn the count.

    Deliberately parsed here rather than taken from request.client.host, which
    looks like the safe option and is not: production runs uvicorn with
    --forwarded-allow-ips='*' (the peer is Caddy on a container IP that compose
    assigns, so it cannot be enumerated), and in that mode uvicorn's
    _TrustedHosts.get_trusted_client_host returns hosts[0] — the same forgeable
    leftmost entry. Its reverse walk only runs when the trust list is explicit.
    """
    if get_settings().trust_proxy:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            hops = [hop.strip() for hop in xff.split(",") if hop.strip()]
            if hops:
                return hops[-1]
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Fixed-window limiter. `times` is resolved per-request from settings so
    limits are tunable per deployment (and can be relaxed for E2E runs, where
    every request legitimately shares one source IP) without touching code."""

    def __init__(self, scope: str, default_times: int, seconds: int = 60):
        self.scope = scope
        self.default_times = default_times
        self.seconds = seconds

    def _limit(self) -> int:
        return getattr(get_settings(), f"rate_limit_{self.scope}", self.default_times)

    async def _hits(self, key: str) -> int | None:
        """Hits in the current window, or None when the counter is unreachable.

        None means "allow it through": the counter is an approximation of load, not
        the authorization decision, and the endpoints behind it — sign-in and
        session refresh — are the ones users cannot work around. Refusing every
        authentication because a best-effort counter is down trades a throttling
        gap for a total outage, so the gap wins.
        """
        hits: int | None = None
        async with degrade_on_outage(f"rate limit ({self.scope})"):
            # INCR and EXPIRE must land together. As two round-trips, a connection
            # lost between them left the counter with no TTL: it never rolled over,
            # so once it passed the limit that IP was 429ed out of sign-in
            # permanently, with Retry-After telling it to try again in a second,
            # forever. MULTI/EXEC makes the pair atomic, and EXPIRE NX sets the TTL
            # only when the key has none — so a fixed window is never extended by a
            # later hit, and a counter that did lose its TTL is repaired on the next
            # request rather than staying immortal.
            pipe = get_redis().pipeline()
            pipe.incr(key)
            pipe.expire(key, self.seconds, nx=True)
            hits = (await pipe.execute())[0]
        return hits

    async def _retry_after(self, key: str) -> int:
        """Seconds until the window rolls over, falling back to the whole window if
        Redis goes away between the increment and this read."""
        ttl = self.seconds
        async with degrade_on_outage(f"rate limit TTL ({self.scope})"):
            ttl = await get_redis().ttl(key)
        return max(ttl, 1)

    async def __call__(self, request: Request) -> None:
        limit = self._limit()
        if limit <= 0:
            return  # 0 disables the limiter for this scope
        key = f"ratelimit:{self.scope}:{_client_ip(request)}"
        hits = await self._hits(key)
        if hits is None:
            return
        if hits > limit:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Try again later.",
                headers={"Retry-After": str(await self._retry_after(key))},
            )


login_limiter = RateLimiter("login", default_times=10)
refresh_limiter = RateLimiter("refresh", default_times=30)
password_limiter = RateLimiter("password", default_times=5)
upload_limiter = RateLimiter("upload", default_times=30)
# POST /api/notifications/subscribe hands a caller-chosen hostname to the
# blocking libc resolver on a worker thread. One call is bounded by a deadline;
# the number of calls in flight was bounded by nothing, and each one that is
# still resolving holds a thread. Low by design — a browser subscribes once per
# subscription change, not once per page.
push_subscribe_limiter = RateLimiter("push_subscribe", default_times=10)
