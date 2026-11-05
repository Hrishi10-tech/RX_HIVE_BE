"""An index built on a table that already has rows must not block writes.

Three migrations say this, at length. b7e21c4a9d33 calls it not optional:

    A plain CREATE INDEX takes SHARE for the length of a full heap scan, which
    conflicts with the ROW EXCLUSIVE that INSERT/UPDATE/DELETE take: it blocks
    writes and leaves reads alone.

and a4f81c6b2e07 says what that costs in this deployment specifically:

    Production boots with `alembic upgrade head && python -m app.seed &&
    uvicorn ...` (infra/docker-compose.prod.yml), so blocking here does not
    merely queue writes — the API never finishes starting.

Nothing enforced it. Two migrations built indexes on tables that already
existed, plainly: 4471a6d661d3 on `messages`, a row per message sent, and
c3f1a7b92d04 on `conversation_participants`. The first is fixed — it creates an
index and nothing else, so leaving the transaction costs it no atomicity. The
second is exempt below, for a stated reason and only for the index it names.

An index on a table the same migration creates is a different case and is
allowed: the table is empty, nothing else can hold a lock on it, and
CONCURRENTLY cannot run in the transaction that created it anyway.

Everything here reads statements per function and in order. A file-wide search
cannot answer any of these questions honestly: an autocommit block in
downgrade() would vouch for a bare CONCURRENTLY in upgrade(), and a drop in
downgrade() would vouch for a build in upgrade() that never clears its own
failed attempt.
"""

import ast
import pathlib
import re
from dataclasses import dataclass

import pytest

VERSIONS = pathlib.Path(__file__).resolve().parents[1] / "alembic" / "versions"

# (index, table) pairs allowed to be built on an existing table without
# CONCURRENTLY, by revision, and why. Each entry is a decision, not a backlog:
# adding one means arguing that the lock is worth what going concurrent costs.
# Scoped to the pair, so the exemption does not quietly cover the next index
# somebody adds to the same revision.
EXEMPT: dict[str, dict[tuple[str, str], str]] = {
    "c3f1a7b92d04": {
        ("ix_participants_pin_order", "conversation_participants"): (
            "The index shares a transaction with the add_column it indexes. Splitting "
            "it out would trade a lock for a migration that cannot be retried: the "
            "revision is stamped only when upgrade() returns, so a failure after an "
            "autocommit block leaves the column added, the revision unstamped, and "
            "the retry failing on a column that already exists."
        ),
    },
}

_CREATE_INDEX = re.compile(
    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?"
    r"([\w.\"]+)\s+ON\s+([\w.\"]+)",
    re.IGNORECASE,
)
_DROP_INDEX = re.compile(r"DROP\s+INDEX\s+(CONCURRENTLY\s+)?(?:IF\s+EXISTS\s+)?([\w.\"]+)", re.IGNORECASE)
_CREATE_TABLE = re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.\"]+)", re.IGNORECASE)


@dataclass(frozen=True)
class Op:
    """One index statement, with where it sits."""

    kind: str  # "create" | "drop"
    index: str
    table: str | None
    concurrent: bool
    in_block: bool
    conditional: bool  # sits inside an if/for/while/try, so it may not run


def _migrations() -> list[pathlib.Path]:
    """Every revision file, oldest first by name so failures read stably."""
    return sorted(p for p in VERSIONS.glob("*.py") if p.name != "__init__.py")


def _revision_id(path: pathlib.Path) -> str:
    """The revision this file declares, which is what EXEMPT is keyed on."""
    match = re.search(r'^revision(?::\s*str)?\s*=\s*["\']([^"\']+)', path.read_text(), re.M)
    assert match, f"{path.name} declares no revision"
    return match.group(1)


def _bare(name: str) -> str:
    """Strip the quoting an identifier may carry in raw SQL."""
    return name.strip('"').split(".")[-1]


def _is_autocommit_block(node: ast.With) -> bool:
    """Whether this `with` is op.get_context().autocommit_block()."""
    return any(
        isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Attribute)
        and item.context_expr.func.attr == "autocommit_block"
        for item in node.items
    )


def _ops_from_call(call: ast.Call, in_block: bool, conditional: bool = False) -> list[Op]:
    """Index statements expressed by one call — raw SQL or the alembic helper."""
    if not isinstance(call.func, ast.Attribute):
        return []
    name = call.func.attr
    if name == "execute" and call.args and isinstance(call.args[0], ast.Constant):
        sql = call.args[0].value
        if not isinstance(sql, str):
            return []
        found = [
            Op("create", _bare(m.group(2)), _bare(m.group(3)), bool(m.group(1)), in_block, conditional)
            for m in _CREATE_INDEX.finditer(sql)
        ]
        found += [
            Op("drop", _bare(m.group(2)), None, bool(m.group(1)), in_block, conditional)
            for m in _DROP_INDEX.finditer(sql)
        ]
        return found
    if (
        name == "create_index"
        and len(call.args) >= 2
        and all(isinstance(a, ast.Constant) for a in call.args[:2])
    ):
        concurrent = any(
            kw.arg == "postgresql_concurrently" and getattr(kw.value, "value", False) is True
            for kw in call.keywords
        )
        return [
            Op("create", str(call.args[0].value), str(call.args[1].value), concurrent, in_block, conditional)
        ]
    if name == "drop_index" and call.args and isinstance(call.args[0], ast.Constant):
        concurrent = any(
            kw.arg == "postgresql_concurrently" and getattr(kw.value, "value", False) is True
            for kw in call.keywords
        )
        return [Op("drop", str(call.args[0].value), None, concurrent, in_block, conditional)]
    return []


def _ops_in(function: ast.FunctionDef) -> list[Op]:
    """Every index statement in one function, in source order.

    Order is the point: a build has to clear its own failed attempt BEFORE it
    runs, and a set of names cannot tell you that.
    """
    ops: list[Op] = []

    def walk(body: list[ast.stmt], in_block: bool, conditional: bool) -> None:
        for node in body:
            if isinstance(node, ast.With):
                walk(node.body, in_block or _is_autocommit_block(node), conditional)
                continue
            if isinstance(node, ast.If | ast.For | ast.While):
                walk(node.body, in_block, True)
                walk(node.orelse, in_block, True)
                continue
            if isinstance(node, ast.Try | ast.TryStar):
                # TryStar as well as Try: `except*` is a different node, and
                # falling through to the generic walk below would record its body
                # as unconditional — the one thing it is not.
                #
                # The body too, not just the handlers: a drop that raises and is
                # caught leaves execution running on past it, so the statement
                # having completed is exactly what cannot be assumed.
                for part in (node.body, node.orelse, node.finalbody):
                    walk(part, in_block, True)
                for handler in node.handlers:
                    walk(handler.body, in_block, True)
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    ops.extend(_ops_from_call(sub, in_block, conditional))

    walk(function.body, False, False)
    return ops


def _functions(path: pathlib.Path) -> dict[str, ast.FunctionDef]:
    """upgrade / downgrade, by name."""
    tree = ast.parse(path.read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _tables_created(path: pathlib.Path) -> set[str]:
    """Tables this migration creates itself, by either route."""
    tree = ast.parse(path.read_text())
    created = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_table"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            created.add(_bare(str(node.args[0].value)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            created.update(_bare(m.group(1)) for m in _CREATE_TABLE.finditer(node.value))
    return created


@pytest.mark.parametrize("path", _migrations(), ids=lambda p: p.name)
def test_an_index_on_an_existing_table_is_built_concurrently(path: pathlib.Path):
    """The rule three migrations state and nothing checked."""
    created = _tables_created(path)
    allowed = EXEMPT.get(_revision_id(path), {})
    offenders = [
        (op.index, op.table)
        for function in _functions(path).values()
        for op in _ops_in(function)
        if op.kind == "create" and op.table not in created and not op.concurrent
    ]
    unexcused = [pair for pair in offenders if pair not in allowed]
    assert not unexcused, (
        f"{path.name} builds {', '.join(f'{i} on {t}' for i, t in unexcused)} without "
        "CONCURRENTLY, on a table it did not create in this migration. A plain "
        "CREATE INDEX holds SHARE for a full heap scan and blocks every write to "
        "that table while it runs, and production runs migrations before uvicorn "
        "starts. See b7e21c4a9d33 for the pattern, or add an argued entry to EXEMPT."
    )
    stale = set(allowed) - set(offenders)
    assert not stale, (
        f"{path.name} is exempt for {sorted(stale)} but no longer builds them that way. "
        "Remove the entry — an exemption nothing uses reads as though the rule does "
        "not apply here."
    )


@pytest.mark.parametrize("path", _migrations(), ids=lambda p: p.name)
def test_concurrent_statements_leave_the_transaction(path: pathlib.Path):
    """CONCURRENTLY outside an autocommit block fails at runtime, not here.

    Alembic wraps a migration in a transaction, and Postgres refuses both CREATE
    and DROP INDEX CONCURRENTLY inside one. Per function, because a block in
    downgrade() says nothing about a statement in upgrade().
    """
    loose = [
        (name, op.kind, op.index)
        for name, function in _functions(path).items()
        for op in _ops_in(function)
        if op.concurrent and not op.in_block
    ]
    if not loose and not any(op.concurrent for f in _functions(path).values() for op in _ops_in(f)):
        pytest.skip("no concurrent DDL")
    assert not loose, (
        f"{path.name} runs {', '.join(f'{k} {i} in {fn}()' for fn, k, i in loose)} "
        "concurrently outside op.get_context().autocommit_block(); Postgres refuses "
        "that inside a transaction, so this raises on the boot that runs it."
    )


@pytest.mark.parametrize("path", _migrations(), ids=lambda p: p.name)
def test_a_concurrent_build_clears_an_invalid_index_first(path: pathlib.Path):
    """A cancelled concurrent build leaves an INVALID index of the same name.

    The revision is stamped only once upgrade() returns, so the retry re-enters
    and has to clear it. CREATE ... IF NOT EXISTS would take the invalid one for
    the finished article; a DROP CONCURRENTLY IF EXISTS *earlier in the same
    function* does not. A drop in downgrade() never runs on that retry.

    Both statements have to be unconditional, too. A drop inside a branch — or
    inside a `try` or `try`/`except*` whose handler swallows the failure — may
    not run, or may not finish, on the path the build takes, and the pairing is the whole point —
    while `IF EXISTS` already makes the drop a no-op when there is nothing to
    clear, so the branch buys nothing.

    The drop has to be concurrent as well. A plain DROP INDEX takes ACCESS
    EXCLUSIVE — stronger than the SHARE the CREATE was made concurrent to avoid
    — and it is easy to read as the harmless half because it is usually a no-op.
    The run it is not a no-op on is the retry, which is the run where the table
    is live and the index is there.
    """
    functions = _functions(path)
    checked = False
    for name, function in functions.items():
        ops = _ops_in(function)
        if not any(op.kind == "create" and op.concurrent for op in ops):
            continue
        checked = True
        dropped: set[str] = set()
        plain_dropped: set[str] = set()
        for op in ops:
            if op.kind == "drop":
                if op.conditional:
                    # A drop that may not run cannot be the one the retry needs.
                    continue
                (dropped if op.concurrent else plain_dropped).add(op.index)
                continue
            if not op.concurrent:
                continue
            if op.conditional:
                raise AssertionError(
                    f"{path.name}: {name}() builds {op.index} concurrently inside a "
                    "branch. Whether its pre-drop runs on the same path cannot be read "
                    "from here, and the pairing is the whole point. DROP INDEX "
                    "CONCURRENTLY IF EXISTS is already conditional in SQL, so the "
                    "branch buys nothing — take both statements out of it."
                )
            if op.index in dropped:
                continue
            if op.index in plain_dropped:
                raise AssertionError(
                    f"{path.name}: {name}() clears {op.index} with a plain DROP INDEX "
                    "before building it concurrently. That drop takes ACCESS EXCLUSIVE "
                    "on the table — a stronger lock than the CREATE was made concurrent "
                    "to avoid, and taken on exactly the run that needs it: the retry "
                    "after a cancelled build, when the index is there and the table is "
                    "live. Drop it CONCURRENTLY too, as a4f81c6b2e07 does."
                )
            raise AssertionError(
                f"{path.name}: {name}() builds {op.index} concurrently without "
                "dropping it first, in this function and before the build. A retry "
                "after a cancelled build would find an INVALID index and keep it."
            )
    if not checked:
        pytest.skip("no concurrent index build")


def test_the_guard_sees_the_migrations():
    """Every assertion above passes vacuously on an empty list."""
    paths = _migrations()
    assert len(paths) >= 10, [p.name for p in paths]
    builds = sum(1 for p in paths for f in _functions(p).values() for op in _ops_in(f) if op.kind == "create")
    assert builds >= 15, f"only found {builds} index builds across {len(paths)} migrations"


# ── The guard's own rules ────────────────────────────────────────────────────
#
# Everything above reads the real migrations, which all pass — so nothing there
# exercises a rule's failure path, and a rule that stopped firing would look
# exactly the same. These run each rule against synthetic migrations instead,
# in both directions. Four of them were live blind spots found in review.

_HEAD = 'from alembic import op\n\nrevision = "zz1"\n\n\n'
_BLOCK = "    with op.get_context().autocommit_block():\n"


def _synthetic(tmp_path: pathlib.Path, body: str, revision: str = "zz1") -> pathlib.Path:
    """A migration file on disk, which is what every rule above reads."""
    path = tmp_path / "x.py"
    path.write_text(f'from alembic import op\n\nrevision = "{revision}"\n\n\n{body}')
    return path


def _fires(rule, path: pathlib.Path) -> bool:
    """Whether a rule rejects this migration. A skip is not a rejection."""
    try:
        rule(path)
    except AssertionError:
        return True
    except BaseException as exc:  # pytest's Skipped is not an Exception
        if type(exc).__name__ != "Skipped":
            raise
    return False


_EXEMPT_REVISION = next(iter(EXEMPT))
_EXEMPT_INDEX, _EXEMPT_TABLE = next(iter(EXEMPT[_EXEMPT_REVISION]))


@pytest.mark.parametrize(
    ("case", "body", "revision", "rejected"),
    [
        (
            "plain index on an existing table",
            'def upgrade():\n    op.create_index("i", "messages", ["a"])\n',
            "zz1",
            True,
        ),
        (
            "raw plain CREATE INDEX",
            'def upgrade():\n    op.execute("CREATE INDEX i ON messages (a)")\n',
            "zz1",
            True,
        ),
        (
            "index on a table this migration creates",
            'def upgrade():\n    op.create_table("t")\n    op.create_index("i", "t", ["a"])\n',
            "zz1",
            False,
        ),
        (
            "postgresql_concurrently=True",
            'def upgrade():\n    op.create_index("i", "messages", ["a"], postgresql_concurrently=True)\n',
            "zz1",
            False,
        ),
        (
            "the exempt pair itself",
            f'def upgrade():\n    op.create_index("{_EXEMPT_INDEX}", "{_EXEMPT_TABLE}", ["a"])\n',
            _EXEMPT_REVISION,
            False,
        ),
        (
            "a different index in the exempt revision",
            f'def upgrade():\n    op.create_index("{_EXEMPT_INDEX}", "{_EXEMPT_TABLE}", ["a"])\n'
            '    op.create_index("i2", "messages", ["b"])\n',
            _EXEMPT_REVISION,
            True,
        ),
        (
            "an exemption that is no longer used",
            f'def upgrade():\n    op.create_index("{_EXEMPT_INDEX}", "{_EXEMPT_TABLE}", ["a"], '
            "postgresql_concurrently=True)\n",
            _EXEMPT_REVISION,
            True,
        ),
    ],
)
def test_the_existing_table_rule_fires_exactly_when_it_should(
    tmp_path: pathlib.Path, case: str, body: str, revision: str, rejected: bool
):
    """Both directions, including the exemption and the tables it does not cover."""
    path = _synthetic(tmp_path, body, revision)
    assert _fires(test_an_index_on_an_existing_table_is_built_concurrently, path) is rejected, case


@pytest.mark.parametrize(
    ("case", "body", "rejected"),
    [
        ("inside the block", _BLOCK + '        op.execute("CREATE INDEX CONCURRENTLY i ON t (a)")\n', False),
        ("outside the block", '    op.execute("CREATE INDEX CONCURRENTLY i ON t (a)")\n', True),
        (
            "concurrent DROP outside the block",
            '    op.execute("DROP INDEX CONCURRENTLY IF EXISTS i")\n',
            True,
        ),
        ("no concurrent DDL at all", '    op.create_index("i", "t", ["a"])\n', False),
    ],
)
def test_the_transaction_rule_fires_exactly_when_it_should(
    tmp_path: pathlib.Path, case: str, body: str, rejected: bool
):
    """A block in the other function is covered separately, below."""
    path = _synthetic(tmp_path, f"def upgrade():\n{body}")
    assert _fires(test_concurrent_statements_leave_the_transaction, path) is rejected, case


def test_a_block_in_one_function_does_not_vouch_for_another(tmp_path: pathlib.Path):
    """The reason statements are read per function rather than per file."""
    path = _synthetic(
        tmp_path,
        'def upgrade():\n    op.execute("CREATE INDEX CONCURRENTLY i ON t (a)")\n\n\n'
        "def downgrade():\n" + _BLOCK + "        pass\n",
    )
    assert _fires(test_concurrent_statements_leave_the_transaction, path)


@pytest.mark.parametrize(
    ("case", "body", "rejected"),
    [
        (
            "drop then create",
            _BLOCK
            + '        op.execute("DROP INDEX CONCURRENTLY IF EXISTS i")\n'
            + '        op.execute("CREATE INDEX CONCURRENTLY i ON t (a)")\n',
            False,
        ),
        (
            "create with no drop",
            _BLOCK + '        op.execute("CREATE INDEX CONCURRENTLY i ON t (a)")\n',
            True,
        ),
        (
            "drop after the create",
            _BLOCK
            + '        op.execute("CREATE INDEX CONCURRENTLY i ON t (a)")\n'
            + '        op.execute("DROP INDEX CONCURRENTLY IF EXISTS i")\n',
            True,
        ),
        (
            "drops a different index",
            _BLOCK
            + '        op.execute("DROP INDEX CONCURRENTLY IF EXISTS other")\n'
            + '        op.execute("CREATE INDEX CONCURRENTLY i ON t (a)")\n',
            True,
        ),
        (
            "a plain DROP as the pre-drop",
            _BLOCK
            + '        op.execute("DROP INDEX IF EXISTS i")\n'
            + '        op.execute("CREATE INDEX CONCURRENTLY i ON t (a)")\n',
            True,
        ),
        (
            "op.drop_index() as the pre-drop",
            _BLOCK
            + '        op.drop_index("i")\n        op.execute("CREATE INDEX CONCURRENTLY i ON t (a)")\n',
            True,
        ),
        (
            "a pre-drop whose failure is caught",
            _BLOCK
            + "        try:\n"
            + '            op.execute("DROP INDEX CONCURRENTLY IF EXISTS i")\n'
            + "        except Exception:\n"
            + "            pass\n"
            + '        op.execute("CREATE INDEX CONCURRENTLY i ON t (a)")\n',
            True,
        ),
        (
            "a pre-drop whose failure is caught by except*",
            _BLOCK
            + "        try:\n"
            + '            op.execute("DROP INDEX CONCURRENTLY IF EXISTS i")\n'
            + "        except* Exception:\n"
            + "            pass\n"
            + '        op.execute("CREATE INDEX CONCURRENTLY i ON t (a)")\n',
            True,
        ),
        (
            "a conditional pre-drop",
            _BLOCK
            + '        if op.get_bind().dialect.name == "postgresql":\n'
            + '            op.execute("DROP INDEX CONCURRENTLY IF EXISTS i")\n'
            + '        op.execute("CREATE INDEX CONCURRENTLY i ON t (a)")\n',
            True,
        ),
    ],
)
def test_the_retry_rule_fires_exactly_when_it_should(
    tmp_path: pathlib.Path, case: str, body: str, rejected: bool
):
    """The pre-drop has to run, be concurrent, and come first."""
    path = _synthetic(tmp_path, f"def upgrade():\n{body}")
    assert _fires(test_a_concurrent_build_clears_an_invalid_index_first, path) is rejected, case


def test_a_drop_in_downgrade_does_not_vouch_for_a_build_in_upgrade(tmp_path: pathlib.Path):
    """A drop that runs on the way down is not the drop a retry needs on the way up."""
    path = _synthetic(
        tmp_path,
        "def upgrade():\n" + _BLOCK + '        op.execute("CREATE INDEX CONCURRENTLY i ON t (a)")\n\n\n'
        "def downgrade():\n" + _BLOCK + '        op.execute("DROP INDEX CONCURRENTLY IF EXISTS i")\n',
    )
    assert _fires(test_a_concurrent_build_clears_an_invalid_index_first, path)
