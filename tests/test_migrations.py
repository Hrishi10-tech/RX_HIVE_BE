"""The migration chain, checked against the models and against the way it boots.

Three things had nothing watching them.

The models and the migrations are two descriptions of one database, and nothing
compared them. conftest migrates to head, so a column that exists only in the
models is invisible here and surfaces as a failing query in production; and the
reverse — schema the migrations own outright — reads to
`alembic revision --autogenerate` as something to DROP, which is how a generated
migration once arrived carrying the removal of messages.search_tsv
(bdbf829f823f caught it by hand).

alembic/env.py is the module every migration runs through, and production runs
it before uvicorn: `alembic upgrade head && python -m app.seed && uvicorn ...`.
It had never been exercised with a DSN of the shape the bootstrap actually
generates.

And the unique index behind retry safety is built CONCURRENTLY, which means the
migration can leave an INVALID index behind and a retry has to clear it.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from app.db.models import (
    MIGRATION_OWNED_COLUMNS,
    MIGRATION_OWNED_INDEXES,
    Base,
    include_in_autogenerate,
)
from app.db.session import engine

BACKEND_DIR = Path(__file__).resolve().parent.parent
_IDEMPOTENCY_MIGRATION = (
    BACKEND_DIR / "alembic" / "versions" / "c5d81e37a204_message_client_idempotency_key.py"
)


def _diff(sync_connection, *, filtered: bool):
    opts = {"include_object": include_in_autogenerate} if filtered else {}
    return compare_metadata(MigrationContext.configure(sync_connection, opts=opts), Base.metadata)


async def _compare(*, filtered: bool):
    async with engine.connect() as conn:
        return await conn.run_sync(_diff, filtered=filtered)


async def test_the_models_and_the_migrations_describe_the_same_database():
    """No drift in either direction, once the migration-owned objects are excluded.

    This is the assertion that catches the dangerous case: a column or index
    added to app/db/models.py and never given a migration. Tests pass either way
    today — conftest builds the schema from the migrations, so the model-only
    column simply is not there — and the first thing to notice is a 500 from the
    query that selects it.
    """
    diffs = await _compare(filtered=True)
    assert diffs == [], f"models and migrations have drifted: {diffs}"


async def test_the_autogenerate_filter_is_exactly_as_wide_as_reality():
    """Unfiltered, autogenerate wants to drop precisely the declared set.

    Both halves of this matter. If it proposed something NOT in the set, the next
    `--autogenerate` would carry a drop nobody declared. If the set is wider than
    what it proposes, the extra entries are hiding schema that really is missing
    — an over-broad filter turns the test above into a rubber stamp.
    """
    diffs = await _compare(filtered=False)

    assert {d[0] for d in diffs} <= {"remove_index", "remove_column"}, (
        f"autogenerate proposes more than dropping migration-owned objects: {diffs}"
    )
    assert {d[1].name for d in diffs if d[0] == "remove_index"} == set(MIGRATION_OWNED_INDEXES)
    assert {(d[2], d[3].name) for d in diffs if d[0] == "remove_column"} == set(MIGRATION_OWNED_COLUMNS)


def _percent_encode_password(url: str) -> str:
    """The same DSN, with the password's first character written as its %XX escape.

    Reproduces what infra/terraform/user_data.sh.tftpl produces for any password
    containing a URL-reserved character: it urlencodes the password on purpose, so
    `%` in RXHIVE_DATABASE_URL is the normal case rather than a corrupt value. The
    DSN still resolves to the same credentials — SQLAlchemy unquotes it — so a
    passing run proves the URL survived config parsing, not that it was ignored.

    Decoded before it is re-encoded, because urlsplit hands back the password
    still escaped. A configured DSN that already carried one — which is precisely
    the deployment this test speaks for — would otherwise come back with `%40`
    turned into `%2540`, i.e. different credentials, and the subprocess would fail
    on authentication rather than on the interpolation this is here to catch.
    """
    parts = urlsplit(url)
    password = unquote(parts.password or "")
    if not password or not password.isascii():
        pytest.skip("configured DSN has no ASCII password to percent-encode")
    encoded = f"%{ord(password[0]):02X}{quote(password[1:], safe='-._~')}"
    netloc = f"{parts.username}:{encoded}@{parts.hostname}"
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def test_alembic_boots_with_a_percent_encoded_database_password():
    """env.py must survive the DSN the bootstrap writes.

    Run as a subprocess because that is the only way to exercise env.py: it runs
    migrations on import, and the failure being guarded against happens at import,
    on the line that hands the URL to alembic's configparser-backed Config.

    Before the escaping this raised ValueError("invalid interpolation syntax")
    before any migration ran, and because the container's command is
    `alembic upgrade head && python -m app.seed && uvicorn ...`, uvicorn was never
    reached — a password rotation took the whole API down.
    """
    env = {
        **os.environ,
        "RXHIVE_DATABASE_URL": _percent_encode_password(os.environ["RXHIVE_DATABASE_URL"]),
    }
    proc = subprocess.run(
        [sys.executable, "-m", "alembic", "current"],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "interpolation" not in proc.stderr.lower(), proc.stderr
    assert proc.returncode == 0, proc.stderr


async def _idempotency_index_state():
    """(indisvalid, indisunique) for the retry-safety index, or None if it is absent."""
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    """SELECT i.indisvalid, i.indisunique
                       FROM pg_class c
                       JOIN pg_index i ON i.indexrelid = c.oid
                       WHERE c.relname = 'uq_messages_client_msg_id'
                         AND c.relnamespace = current_schema()::regnamespace"""
                )
            )
        ).first()
    return None if row is None else (row.indisvalid, row.indisunique)


def _idempotency_upgrade_statements() -> list[str]:
    """The raw SQL c5d81e37a204's upgrade() runs, read out of the migration itself.

    Parsed rather than copied so the tests below cannot quietly drift from the
    statements production actually executes — a copy would keep passing while the
    migration changed underneath it, which is the one failure mode that matters
    for a migration nobody re-reads.
    """
    tree = ast.parse(_IDEMPOTENCY_MIGRATION.read_text())
    upgrade = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "upgrade")
    return [
        node.args[0].value
        for node in ast.walk(upgrade)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]


async def test_the_idempotency_index_is_valid_at_head():
    """indisvalid, not merely present.

    A cancelled or lock-timed-out CREATE INDEX CONCURRENTLY leaves an INVALID
    index behind, and an INVALID unique index enforces nothing while appearing —
    in every listing an operator would actually check — exactly like a finished
    one. So the guarantee behind retry safety is this flag, not the index's
    existence.
    """
    assert await _idempotency_index_state() == (True, True)


async def test_the_idempotency_upgrade_can_be_rerun_and_still_lands_a_valid_index():
    """The retry path, executed for real against the migrated schema.

    upgrade() cannot be atomic — CONCURRENTLY has to leave the transaction — so
    alembic_version is only stamped after every statement has committed, and a
    build interrupted anywhere in between leaves the revision unstamped and the
    work half-done. The retry therefore re-enters upgrade() on a database that is
    already partway through it, which is only safe if all three statements are
    re-runnable. Running them here proves that, rather than reasoning about it.
    """
    async with engine.connect() as conn:
        # CONCURRENTLY cannot run inside a transaction block, which is also what
        # the next test uses to pin down the drop's lock level.
        autocommit = await conn.execution_options(isolation_level="AUTOCOMMIT")
        for statement in _idempotency_upgrade_statements():
            await autocommit.execute(text(statement))

    assert await _idempotency_index_state() == (True, True)


async def test_the_pre_drop_does_not_take_an_exclusive_lock_on_messages():
    """The drop is the CONCURRENTLY form, proven by the one thing SQL can observe.

    A plain DROP INDEX takes ACCESS EXCLUSIVE on messages — every read and every
    write queued behind it — and no query can report the lock a statement WOULD
    have taken. What is observable is that Postgres refuses the concurrent form
    inside a transaction block and accepts the plain one, so the refusal below is
    the assertion.

    It matters because the drop is easy to read as the harmless half: it is a
    no-op in the ordinary case. The case it exists for is the retry after a failed
    concurrent build, which is exactly when the index is there, is INVALID, and
    the table is live — and since production boots with
    `alembic upgrade head && ... && uvicorn`, blocking there does not stall sends,
    it stops the API from starting at all.
    """
    drop = next(s for s in _idempotency_upgrade_statements() if s.lstrip().upper().startswith("DROP INDEX"))

    async with engine.connect() as conn:
        transaction = await conn.begin()
        try:
            # Never committed: were this the plain form, it would succeed here and
            # take the index out from under every other test in the session.
            with pytest.raises(DBAPIError) as excinfo:
                await conn.execute(text(drop))
        finally:
            await transaction.rollback()

    assert "cannot run inside a transaction block" in str(excinfo.value)
    assert await _idempotency_index_state() == (True, True)
