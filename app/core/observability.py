"""Lightweight observability: request-id + structured access logs, an in-process
metrics counter, and a /metrics endpoint. No external APM dependency for v1;
the log format is JSON so it drops straight into any log aggregator.

The JSON is produced by JsonLogFormatter below, which app/main.py installs on the
root logger. That is load-bearing rather than decorative: every field this module
records is passed through logging's `extra=`, and `extra` only reaches the output
if the FORMATTER emits it. Under the default format string
("%(levelname)s:%(name)s:%(message)s") every access log line in production read

    INFO:rxhive.access:access

with no method, no path, no status, no latency and no request id — and the error
branch logged a traceback with nothing to attribute it to. The structure was all
being computed and then discarded at the last step.
"""

import json
import logging
import logging.handlers
import queue
import re
import time
import uuid
from collections import defaultdict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("rxhive.access")

# Attributes logging puts on every record itself. Anything NOT in here arrived
# through `extra=` and is what this formatter exists to emit. Derived from a real
# record rather than hardcoded, so a new stdlib attribute (taskName arrived in
# 3.12) does not start showing up as if it were application data.
_STANDARD_RECORD_ATTRS = frozenset(
    vars(logging.LogRecord(name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None))
) | {"message", "asctime", "taskName"}


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line: the standard fields, plus everything from `extra`.

    default=str on the dump because `extra` carries whatever a caller passed and a
    log line must never be the thing that raises. Exceptions are rendered into the
    same object rather than trailing after it, so a traceback cannot be split from
    its request id by line-based log collection.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str)


# Records buffered between the event loop and the sink. Bounded on purpose: a
# stderr that has stopped draining should cost log records, not unbounded memory
# in the process serving requests. 10k is minutes of traffic at this app's rate.
LOG_QUEUE_MAXSIZE = 10_000


def install_json_logging(level: int = logging.INFO) -> logging.handlers.QueueListener:
    """Put JSON logging on the root logger, with the WRITE off the event loop.

    logging.StreamHandler formats and writes SYNCHRONOUSLY. Called from
    AccessLogMiddleware.dispatch — which is on the event loop, once per request —
    that makes the log sink an availability dependency: a stderr that blocks
    stalls the worker, and every request behind it, for as long as the write takes.
    In a container stderr is a pipe to the log driver, so "blocks" means the
    collector is wedged rather than anything exotic.

    That was survivable while the access log emitted the single word "access".
    This batch makes it emit a few hundred bytes per request instead, which is
    the point of the batch and also a tenfold increase in what the event loop is
    waiting on, so the two changes belong together.

    QueueHandler only enqueues; a QueueListener thread does the formatting and the
    write. Returns the listener so lifespan shutdown can stop it — that is what
    drains the queue, and skipping it loses whatever is still buffered, including
    the log lines explaining why the process is going down.

    force=True because basicConfig is a no-op when the root logger already has a
    handler, and the point here is to be certain WHICH handler is installed rather
    than to depend on import order.
    """
    sink = logging.StreamHandler()
    sink.setFormatter(JsonLogFormatter())
    record_queue: queue.Queue = queue.Queue(maxsize=LOG_QUEUE_MAXSIZE)
    logging.basicConfig(level=level, handlers=[logging.handlers.QueueHandler(record_queue)], force=True)
    listener = logging.handlers.QueueListener(record_queue, sink, respect_handler_level=True)
    listener.start()
    return listener


# An inbound request id is honoured so a proxy or client can correlate across a
# hop, but only when it is short and boring. It is caller-controlled and lands in
# two places — the response header and every log line for that request — so
# without a bound this hands anyone who can reach the port a way to write ~16 KB
# (uvicorn's per-header ceiling) of chosen text into the logs on every request.
# That was dormant only because the formatter above was dropping the field
# entirely; making the logs work is what makes the bound necessary.
_REQUEST_ID_PATTERN = re.compile(r"\A[A-Za-z0-9._:-]{1,64}\Z")


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id", "")
    if _REQUEST_ID_PATTERN.match(supplied):
        return supplied
    return uuid.uuid4().hex[:12]


_counters: dict[str, int] = defaultdict(int)
_latency_ms_total: dict[str, float] = defaultdict(float)

# Counters live for the process lifetime and are never evicted, so a per-request
# key must come from a bounded set. Requests that matched no route (404s to
# arbitrary URLs) have no route template, and keying them by raw path let anyone
# who can reach the port mint unbounded keys — a memory leak, unauthenticated.
# They all share this one label; the raw path still reaches the access log.
UNMATCHED_ROUTE_LABEL = "<unmatched>"


def incr(name: str, by: int = 1) -> None:
    _counters[name] += by


def snapshot() -> dict:
    return {
        "counters": dict(_counters),
        "latency_ms_avg": {
            k: round(_latency_ms_total[k] / _counters[k], 2) for k in _latency_ms_total if _counters.get(k)
        },
    }


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = _request_id(request)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            incr("http_requests_error")
            logger.exception("request failed", extra={"request_id": request_id, "path": request.url.path})
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000

        route = request.scope.get("route")
        route_path = getattr(route, "path", None)
        key = f"{request.method} {route_path or UNMATCHED_ROUTE_LABEL}"
        incr("http_requests_total")
        incr(f"status_{status // 100}xx")
        _counters[key] += 1
        _latency_ms_total[key] += elapsed_ms

        response.headers["X-Request-ID"] = request_id
        # Structured single-line access log (path template, not raw path, so
        # ids don't explode cardinality; never logs query strings or bodies).
        logger.info(
            "access",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": route_path or request.url.path,
                "status": status,
                "ms": round(elapsed_ms, 1),
            },
        )
        return response
