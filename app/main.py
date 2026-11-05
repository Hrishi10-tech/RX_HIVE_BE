import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import get_settings
from app.core.deps import require_superadmin
from app.core.errors import CodedHTTPException, coded_http_exception_handler
from app.core.observability import AccessLogMiddleware, install_json_logging, snapshot
from app.db.models import User
from app.db.session import engine
from app.realtime import hub
from app.realtime.redis_bus import close_redis, get_redis
from app.utils import iso_z, now_utc

logger = logging.getLogger("rxhive")

# JSON on stdout, and the write off the event loop. core/observability's access
# log has always assumed both and had neither: `basicConfig(level=INFO)` alone
# installs the default "%(levelname)s:%(name)s:%(message)s" format, and every
# field this app records goes through `extra=` — which that format string does not
# emit. So each request logged the single word "access", and errors logged a
# traceback with no request id and no path to attribute it to.
#
# The listener is returned so lifespan shutdown can drain the queue; see
# install_json_logging for why the write must not happen on the loop.
_log_listener = install_json_logging()

MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
# Cookie-authed browsers must present the custom header (CSRF defense: cross-site
# forms can't set custom headers without passing a CORS preflight we control).
CSRF_EXEMPT_PATHS = {"/api/auth/login", "/api/auth/refresh", "/api/livekit/webhook"}


# How often each worker looks for call deadlines that have come due. Every worker
# sweeps; claiming is atomic, so N workers cost N cheap ZRANGEBYSCOREs per tick and
# firing is never duplicated. Two seconds is well inside the precision a ring
# timeout or a reconnect grace window needs.
CALL_SWEEP_INTERVAL_SECONDS = 2


async def _call_deadline_sweeper() -> None:
    """Fire ring timeouts and reconnect-grace expiries whoever scheduled them.

    The safety net that makes call timers survive a worker restart. Previously they
    were asyncio tasks in a per-process dict: a deploy mid-ring left the call
    `ringing` forever, so the caller heard ringback with no timeout and the row
    never reached history. See app/services/call_deadlines.py.
    """
    from app.services.calls import sweep_due_deadlines

    while True:
        try:
            await asyncio.sleep(CALL_SWEEP_INTERVAL_SECONDS)
            await sweep_due_deadlines()
        except asyncio.CancelledError:
            raise
        except Exception:
            # Never let one bad tick end the sweeper — that would silently take
            # every future call timeout with it.
            logger.exception("call deadline sweep failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Production secret validation happens in Settings (config.py) and raises at
    # instantiation, so an unsafe production config never reaches here.
    with contextlib.suppress(Exception):
        from app.services.storage import ensure_bucket

        ensure_bucket()
    await hub.registry.start()
    sweeper = asyncio.create_task(_call_deadline_sweeper())
    yield
    sweeper.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await sweeper
    await hub.registry.stop()
    await close_redis()
    await engine.dispose()
    # Last, and after everything above has had its say: stopping the listener is
    # what drains the queue, so anything the shutdown path logged is still
    # buffered until this runs.
    _log_listener.stop()


settings = get_settings()
# Interactive API docs are dev-only — no schema/enumeration surface in prod.
app = FastAPI(
    title="RX HIVE API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)

# Only refusals that a client has to *branch* on raise this; everything else keeps
# FastAPI's own `{"detail": ...}` handler untouched.
app.add_exception_handler(CodedHTTPException, coded_http_exception_handler)

if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    )
# No wildcard fallback: with no configured origins, cross-origin browsers are
# simply refused (same-origin deployments behind Caddy need no CORS at all).

app.add_middleware(AccessLogMiddleware)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    # CSRF: cookie-authenticated mutating requests must carry the custom header.
    if (
        request.method in MUTATING_METHODS
        and request.url.path.startswith("/api")
        and request.url.path not in CSRF_EXEMPT_PATHS
        and "authorization" not in request.headers
        and request.cookies
        and request.headers.get("x-requested-with") != "XMLHttpRequest"
    ):
        return JSONResponse(status_code=403, content={"detail": "Missing CSRF header"})

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if settings.cookies_secure:
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


from app.api import (  # noqa: E402
    admin,
    auth,
    calls,
    contacts,
    conversations,
    cross_org,
    groups,
    media,
    messages,
    notifications,
    org_admin,
    search,
    validation,
)

app.include_router(auth.router)
app.include_router(admin.router)
# Registered after admin.router: both live under /api/admin, and this one's
# paths are all /api/admin/access/*, which cannot collide with admin's.
app.include_router(validation.router)
app.include_router(conversations.router)
app.include_router(messages.router)
app.include_router(contacts.router)
app.include_router(groups.router)
app.include_router(media.router)
app.include_router(search.router)
app.include_router(cross_org.router)
app.include_router(org_admin.router)
app.include_router(calls.router)
# The LiveKit webhook lives on its own router because it is the one route in the
# app that is NOT under /api/calls and NOT user-authenticated: the SFU calls it
# server-to-server and proves itself with a signed JWT instead of a session
# cookie. Registering it was missed when the router was split out, which left
# every SFU-driven reconciliation path dead — room_finished never marked a call
# answered, so a client that was killed mid-call left the row "connected"
# forever, and the webhook's participant_joined/left fan-out never fired.
# It must also be listed in CSRF_EXEMPT_PATHS above (it is): the SFU is not a
# browser and cannot send X-Requested-With.
app.include_router(calls.webhook_router)
app.include_router(notifications.router)
app.include_router(hub.router)


LIVEKIT_PROBE_TIMEOUT = 2.0


async def livekit_status() -> str:
    """Liveness of the calling SFU: connected | unreachable | not_configured.

    Deliberately separate from the healthy/unhealthy verdict — messaging works
    fine without LiveKit, only calls break. Without this field a stopped
    livekit-server shows up as nothing but a generic "could not connect the
    call" toast in the browser, which is what made call outages so hard to spot.
    """
    target = settings.livekit_probe_url
    if not target:
        return "not_configured"
    try:
        async with httpx.AsyncClient(timeout=LIVEKIT_PROBE_TIMEOUT) as http:
            resp = await http.get(target)
    except Exception as exc:
        logger.warning("LiveKit probe failed for %s: %s", target, exc)
        return "unreachable"
    # LiveKit answers `/` with 200 OK; any non-5xx means the process is serving.
    return "connected" if resp.status_code < 500 else "unreachable"


@app.get("/api/health")
async def health():
    db_status = "connected"
    redis_status = "connected"
    healthy = True
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        db_status, healthy = "disconnected", False
    try:
        await get_redis().ping()
    except Exception:
        redis_status, healthy = "disconnected", False
    livekit = await livekit_status()
    body = {
        "status": "healthy" if healthy else "unhealthy",
        "version": "1.0.0",
        "service": "RxHive API",
        "timestamp": iso_z(now_utc()),
        "database": db_status,
        "redis": redis_status,
        # Calls need the SFU; messaging does not. Reported, never fatal.
        "livekit": livekit,
        "calls_available": livekit == "connected",
    }
    return JSONResponse(status_code=200 if healthy else 503, content=body)


@app.get("/api/metrics")
async def metrics(_admin: User = Depends(require_superadmin)):
    """In-process request counters + average latencies for scraping/dashboards.
    Cheap and dependency-free; swap for Prometheus if/when scale demands.

    Superadmin-only: per-route counts and latencies leak the private API surface
    and traffic shape to anyone who can reach the port. /api/health stays public
    because load balancers probe it unauthenticated."""
    data = snapshot()
    data["ws_local_sockets"] = hub.registry.local_socket_count()
    return data
