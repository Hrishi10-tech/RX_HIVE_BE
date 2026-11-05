"""Structure that is computed and then discarded is not observability.

core/observability builds a request id, method, path template, status and latency
for every request and passes them through logging's `extra=`. `extra` only
reaches the output if the FORMATTER emits it, and the app installed none — so
under basicConfig's default "%(levelname)s:%(name)s:%(message)s" every access log
line in production read exactly

    INFO:rxhive.access:access

and the error branch logged a traceback with no request id and no path to
attribute it to. These tests pin the formatter, the wiring that installs it, and
the one field in that payload a caller controls.
"""

import io
import json
import logging
import logging.handlers
import queue
import threading
import time

from app.core.observability import (
    LOG_QUEUE_MAXSIZE,
    JsonLogFormatter,
    _request_id,
    install_json_logging,
)


class _Req:
    def __init__(self, headers: dict):
        self.headers = headers


def _emit(record_factory) -> dict:
    """Format one record through the real formatter and parse it back."""
    return json.loads(JsonLogFormatter().format(record_factory()))


def _record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="rxhive.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="access",
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_the_access_fields_actually_reach_the_output():
    """The finding. Every one of these was being dropped."""
    payload = _emit(
        lambda: _record(request_id="abc123def456", method="POST", path="/api/auth/login", status=429, ms=12.3)
    )
    assert payload["request_id"] == "abc123def456"
    assert payload["method"] == "POST"
    assert payload["path"] == "/api/auth/login"
    assert payload["status"] == 429
    assert payload["ms"] == 12.3
    # And the standard fields an aggregator needs to sort and filter on.
    assert payload["level"] == "INFO"
    assert payload["logger"] == "rxhive.access"
    assert payload["msg"] == "access"
    assert payload["ts"]


def test_logging_internals_do_not_masquerade_as_application_fields():
    """`extra` is the signal; LogRecord's own attributes are not.

    Emitting record.__dict__ wholesale would put pathname, lineno, threadName and
    the rest in every line, which is how a JSON log stops being readable. The
    reserved set is derived from a real record so a new stdlib attribute cannot
    quietly start appearing as if the application had passed it.
    """
    payload = _emit(lambda: _record(request_id="r1"))
    for leaked in ("pathname", "lineno", "args", "levelno", "threadName", "processName", "msecs"):
        assert leaked not in payload, f"{leaked} leaked into the log payload"
    assert set(payload) == {"ts", "level", "logger", "msg", "request_id"}


def test_a_traceback_travels_in_the_same_object_as_its_request_id():
    """Line-based collection must not be able to separate the two.

    This is the branch that mattered most and said least: the middleware's
    `except` logged "request failed" with the request id and path in `extra`, and
    the operator got a bare traceback with neither.
    """
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    def factory():
        record = _record(request_id="abc123def456", path="/api/messages")
        record.levelname, record.levelno = "ERROR", logging.ERROR
        record.msg = "request failed"
        record.exc_info = exc_info
        return record

    payload = _emit(factory)
    assert payload["request_id"] == "abc123def456"
    assert payload["path"] == "/api/messages"
    assert "ValueError: boom" in payload["exc"]
    assert "Traceback" in payload["exc"]


def test_an_unserializable_extra_does_not_take_the_log_line_down():
    """A log line must never be the thing that raises. `extra` carries whatever a
    caller passed, so the dump falls back to str rather than propagating."""
    payload = _emit(lambda: _record(request_id="r1", weird=object(), when=logging))
    assert isinstance(payload["weird"], str)
    assert payload["request_id"] == "r1"


def test_the_root_logger_enqueues_rather_than_writing_on_the_caller():
    """The wiring, not just the class.

    Worth its own test because the bug being fixed was exactly this shape: the
    structure existed and nothing was configured to emit it. A JsonLogFormatter
    that no handler uses would reproduce the finding with more code.

    What is asserted is a QueueHandler, because that is the whole point of the
    arrangement: AccessLogMiddleware logs from the event loop once per request,
    and a StreamHandler there formats and writes to stderr synchronously. The
    JsonLogFormatter belongs on the LISTENER's sink, off the loop.
    """
    import app.main  # noqa: F401 - imported for its logging side effect

    handlers = logging.getLogger().handlers
    queue_handlers = [h for h in handlers if isinstance(h, logging.handlers.QueueHandler)]
    assert queue_handlers, f"root handlers are {[type(h).__name__ for h in handlers]}"
    assert queue_handlers[0].queue.maxsize == LOG_QUEUE_MAXSIZE, (
        "an unbounded queue trades a stalled sink for unbounded memory"
    )


def test_a_record_logged_from_the_event_loop_reaches_the_sink_as_json():
    """End to end through the real arrangement: enqueue here, format and write there.

    Built on a private root logger rather than the app's, so the assertion does
    not depend on what else the session has configured, and torn down so it does
    not leave a listener thread behind.
    """
    stream = io.StringIO()
    sink = logging.StreamHandler(stream)
    sink.setFormatter(JsonLogFormatter())
    record_queue: queue.Queue = queue.Queue(maxsize=LOG_QUEUE_MAXSIZE)
    listener = logging.handlers.QueueListener(record_queue, sink, respect_handler_level=True)
    listener.start()

    logger = logging.getLogger("rxhive.test.queue")
    logger.handlers = [logging.handlers.QueueHandler(record_queue)]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        logger.info("access", extra={"request_id": "q1", "path": "/api/health", "status": 200})
        listener.stop()  # drains the queue, which is what makes this deterministic
    finally:
        logger.handlers = []

    payload = json.loads(stream.getvalue().strip())
    assert payload["request_id"] == "q1"
    assert payload["path"] == "/api/health"
    assert payload["status"] == 200


def test_a_blocked_sink_does_not_block_the_caller():
    """The finding: the log sink was an availability dependency of every request.

    The sink here holds its write for far longer than any request should wait. The
    caller must return immediately anyway, because it only enqueues — under the
    previous arrangement the same call would have waited out the whole write.
    """
    released = threading.Event()

    class _WedgedStream(io.StringIO):
        def write(self, value):  # noqa: D102
            released.wait(5)
            return super().write(value)

    sink = logging.StreamHandler(_WedgedStream())
    sink.setFormatter(JsonLogFormatter())
    record_queue: queue.Queue = queue.Queue(maxsize=LOG_QUEUE_MAXSIZE)
    listener = logging.handlers.QueueListener(record_queue, sink, respect_handler_level=True)
    listener.start()

    logger = logging.getLogger("rxhive.test.wedged")
    logger.handlers = [logging.handlers.QueueHandler(record_queue)]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        started = time.monotonic()
        for _ in range(20):
            logger.info("access", extra={"request_id": "wedged"})
        elapsed = time.monotonic() - started
        assert elapsed < 0.5, f"logging blocked the caller for {elapsed:.2f}s"
    finally:
        released.set()
        listener.stop()
        logger.handlers = []


def test_stopping_the_listener_flushes_what_is_still_queued():
    """Why lifespan shutdown stops it.

    Without the stop, the queue is dropped on the floor along with whatever the
    shutdown path itself logged — which is exactly the part an operator reads to
    find out why the process went down.
    """
    stream = io.StringIO()
    sink = logging.StreamHandler(stream)
    sink.setFormatter(JsonLogFormatter())
    record_queue: queue.Queue = queue.Queue(maxsize=LOG_QUEUE_MAXSIZE)
    listener = logging.handlers.QueueListener(record_queue, sink, respect_handler_level=True)
    listener.start()

    logger = logging.getLogger("rxhive.test.flush")
    logger.handlers = [logging.handlers.QueueHandler(record_queue)]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        for i in range(50):
            logger.info("shutting down", extra={"seq": i})
        listener.stop()
    finally:
        logger.handlers = []

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert len(lines) == 50, f"only {len(lines)} of 50 records survived the stop"
    assert json.loads(lines[-1])["seq"] == 49


def test_install_json_logging_is_idempotent_enough_to_call_once_per_process():
    """It uses force=True, so a second call must not leave two sinks stacked."""
    listener = install_json_logging()
    try:
        handlers = logging.getLogger().handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.handlers.QueueHandler)
    finally:
        listener.stop()
        # Put the session's own arrangement back for the tests that follow.
        install_json_logging()


def test_a_sane_inbound_request_id_is_honoured():
    """A proxy or client gets to correlate across the hop."""
    assert _request_id(_Req({"x-request-id": "req-0a1b2c3d"})) == "req-0a1b2c3d"
    assert _request_id(_Req({"x-request-id": "A.b_1:2-3"})) == "A.b_1:2-3"


def test_a_hostile_or_oversized_request_id_is_replaced():
    """The field is caller-controlled and now genuinely reaches the logs.

    Before the formatter was fixed this was dormant — the value was computed and
    dropped. Making the logs work is what turns an unbounded, arbitrary,
    caller-chosen string into ~16 KB of chosen text written to the log on every
    request, so the bound lands in the same change.
    """
    for hostile in (
        "x" * 65,
        "x" * 16_000,
        "",
        "has spaces",
        'quote"and\\backslash',
        '{"json": "injection"}',
        "semi;colon",
        "sl/ash",
    ):
        generated = _request_id(_Req({"x-request-id": hostile}))
        assert generated != hostile, f"accepted {hostile[:32]!r}"
        assert len(generated) == 12 and generated.isalnum()

    # No header at all still yields one.
    assert len(_request_id(_Req({}))) == 12


async def test_the_response_carries_a_request_id_end_to_end(client):
    """It is the value the client can use to find its own request in the logs."""
    resp = await client.get("/api/health")
    assert resp.headers.get("X-Request-ID")

    echoed = await client.get("/api/health", headers={"x-request-id": "req-trace-01"})
    assert echoed.headers["X-Request-ID"] == "req-trace-01"

    rejected = await client.get("/api/health", headers={"x-request-id": "y" * 500})
    assert rejected.headers["X-Request-ID"] != "y" * 500
    assert len(rejected.headers["X-Request-ID"]) == 12
