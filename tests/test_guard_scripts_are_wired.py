"""A guard nothing runs is not a guard.

scripts/check_dependency_groups.py was written in batch 48 against a measured
failure — four of the twenty-one Dependabot PRs open on 2026-09-03 could not be
resolved by pip, because pins that constrain each other were split across
separate update groups. The script reads the real dependabot.yml and
requirements.txt and fails when a coupled cluster is split again.

ci.yml never invoked it. Its only two mentions anywhere under .github were
prose, in comments inside dependabot.yml.

The reason that survived review is the interesting part. The script has a test,
tests/test_dependency_groups_check.py, and that test is thorough — it builds
synthetic manifests and asserts the script rejects each way a group can go
wrong. So the suite was green, the guard was covered, and the invariant it
exists for was still unchecked against this repository. Coverage of the checker
is not coverage of the thing checked.

This asserts the wiring itself, so the next guard cannot be born dead.
"""

import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
WORKFLOWS = ROOT / ".github" / "workflows"


def _guard_scripts() -> list[pathlib.Path]:
    """The scripts that assert an invariant about this repository.

    Named by convention: `check_*.py` is a guard and must run in CI. Anything
    else under scripts/ is a tool a human reaches for — scripts/inspect-call.py
    is one — and carries no such claim.
    """
    return sorted(SCRIPTS.glob("check_*.py"))


def _ci_commands() -> list[str]:
    """Every shell command CI actually runs, across all workflows.

    Taken from the parsed YAML rather than the file text, so a script named only
    in a comment does not read as wired — which is exactly how this was missed:
    dependabot.yml discussed check_dependency_groups.py twice while nothing ran
    it.
    """
    commands: list[str] = []
    for path in sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml")):
        workflow = yaml.safe_load(path.read_text()) or {}
        for job in (workflow.get("jobs") or {}).values():
            for step in (job or {}).get("steps") or []:
                run = (step or {}).get("run")
                if run:
                    commands.append(run)
    return commands


def test_there_are_guard_scripts_to_check():
    """The parametrized test below passes vacuously on an empty list."""
    assert len(_guard_scripts()) >= 4, [p.name for p in _guard_scripts()]


# `&&`, `;` and friends separate commands; `(` and `)` group them. shlex hands
# these back as their own tokens only with punctuation_chars set.
# Characters that end one command and begin the next. `&&` and `||` are runs of
# these, consumed as one.
_SEPARATOR_CHARS = frozenset({";", "|", "&", "(", ")"})

# `python`, `python3`, `python3.12`, `/usr/bin/python` — an interpreter whose
# first argument is the script it runs.
_PYTHON = re.compile(r"^(?:\S*/)?python[\d.]*$")

# Inside double quotes a backslash is literal except before one of these.
_DOUBLE_QUOTE_ESCAPES = frozenset({"\\", '"', "$", "`"})


def _commands(text: str) -> list[list[str]]:
    """Split a whole `run:` value into commands, each a list of unquoted words.

    Hand-written rather than shlex, after three review findings in a row on this
    helper, each one shlex disagreeing with the shell in a different place: a
    regex split that read a quoted `;` as a separator, commenters that cut at a
    `#` anywhere, and non-posix mode ignoring a backslash escape.

    It takes the whole value, not a line at a time. Splitting first loses the
    two things that cross a line boundary: a trailing backslash continues the
    command onto the next line, so

        echo continued \\
        python scripts/check_x.py

    is one `echo` and not an invocation; and a quote opened on one line runs on,
    newline included, until it closes.

    A value whose quote never closes is not a command the shell would run, so it
    yields nothing rather than its words. Reporting a wired guard as unwired is
    the safe direction; reporting an unwired one as fine is not.

    No expansion is attempted: `$VAR`, globs and substitutions stay literal.
    They cannot turn something that is not an invocation into one, since the
    path has to be written out to be matched.
    """
    commands: list[list[str]] = []
    words: list[str] = []
    word: list[str] = []
    started = False  # a word is open, possibly empty — `""` is a word
    quote: str | None = None
    index, length = 0, len(text)

    def end_word() -> None:
        nonlocal word, started
        if started:
            words.append("".join(word))
        word, started = [], False

    def end_command() -> None:
        nonlocal words
        end_word()
        if words:
            commands.append(words)
        words = []

    while index < length:
        char = text[index]
        following = text[index + 1 : index + 2]
        if quote == "'":
            # Single quotes protect everything, backslash and newline included.
            if char == "'":
                quote = None
            else:
                word.append(char)
            index += 1
        elif quote == '"':
            if char == "\\" and following == "\n":
                index += 2  # line continuation, removed
                continue
            if char == "\\" and following in _DOUBLE_QUOTE_ESCAPES:
                word.append(following)
                index += 2
                continue
            if char == '"':
                quote = None
            else:
                word.append(char)
            index += 1
        elif char == "\\" and following == "\n":
            index += 2  # line continuation: both characters vanish
        elif char == "\\" and index + 1 < length:
            word.append(following)
            started = True
            index += 2
        elif char in ("'", '"'):
            quote = char
            started = True
            index += 1
        elif char == "#" and not started:
            # A shell opens a comment at `#` only where a word begins, and it
            # runs to the end of THIS line rather than of the whole value.
            newline = text.find("\n", index)
            if newline < 0:
                break
            index = newline
        elif char == "\n":
            end_command()
            index += 1
        elif char.isspace():
            end_word()
            index += 1
        elif char in _SEPARATOR_CHARS:
            end_command()
            while index < length and text[index] in _SEPARATOR_CHARS:
                index += 1
        else:
            word.append(char)
            started = True
            index += 1

    if quote is not None:
        return []
    end_command()
    return commands


def _is_invocation(words: list[str], script: str) -> bool:
    """Whether one command — already split from its neighbours — runs the script."""
    if not words:
        return False
    target = f"scripts/{script}"
    if words[0] in (target, f"./{target}"):
        return True
    return len(words) > 1 and bool(_PYTHON.match(words[0])) and words[1] == target


def _invokes(command: str, script: str) -> bool:
    """Whether a shell command actually runs `scripts/<script>`.

    Naming the path is not running it, and the ways to name it are not obvious —
    see _commands for the four that got through review. The path has to be the
    program of one of the commands in the value, or the first argument to a
    python interpreter.
    """
    return any(_is_invocation(words, script) for words in _commands(command))


@pytest.mark.parametrize(
    ("command", "runs"),
    [
        ("python scripts/check_contracts.py --check-mappings", True),
        ("python3 scripts/check_contracts.py", True),
        ("./scripts/check_contracts.py", True),
        ("cd backend && python scripts/check_contracts.py", True),
        ("  python scripts/check_contracts.py  ", True),
        # Mentions, not invocations — each of these would have satisfied a plain
        # substring match, which is what this guard was first written with.
        ("echo scripts/check_contracts.py", False),
        ("# python scripts/check_contracts.py", False),
        ("echo 'see scripts/check_contracts.py for why'", False),
        ("python scripts/check_contracts.py.bak", False),
        ("python other/scripts/check_contracts.py", False),
        # Running a different guard is not running this one.
        ("python scripts/check_dependency_groups.py", False),
        ("python scripts/check_pinned_imports.py --check-mappings", False),
        # A separator inside a quoted string is not a separator.
        ("echo 'note; python scripts/check_contracts.py --check-mappings'", False),
        ('echo "a && python scripts/check_contracts.py"', False),
        # An inline comment ends the line, wherever it starts.
        ("true # && python scripts/check_contracts.py", False),
        ("true  #; python scripts/check_contracts.py", False),
        # ...but the same separators, unquoted, do separate.
        ("true && python scripts/check_contracts.py", True),
        ("python scripts/check_contracts.py; true", True),
        ("false || python scripts/check_contracts.py", True),
        ("python scripts/check_contracts.py | tee log", True),
        ("bare scripts/check_contracts.py", False),
        ("scripts/check_contracts.py", True),
        # shlex cuts at `#` anywhere; a shell only does so where a word begins.
        ("python scripts/check_contracts.py#note", False),
        ("python scripts/check_contracts.py #note", True),
        ("python scripts/check_contracts.py --flag#x", True),
        # ...and a quoted `#` opens nothing.
        ("echo '#' && python scripts/check_contracts.py", True),
        ('python "scripts/check_contracts.py"', True),
        # A backslash escape inside double quotes does not end the string, so
        # the `#` after it is still quoted and the command after `&&` still runs.
        ('echo "note\\"#still quoted" && python scripts/check_contracts.py', True),
        # An escaped `#` outside quotes is a literal, not a comment.
        ("echo \\# && python scripts/check_contracts.py", True),
        ("echo \\#not-a-comment && python scripts/check_contracts.py", True),
        # ...but the escape only covers the character after it.
        ("echo \\x # && python scripts/check_contracts.py", False),
        # Single quotes protect a backslash, so this one stays open to the `&&`.
        ("echo 'a\\' && python scripts/check_contracts.py", True),
        # An escaped `&&` does not separate, so the invocation never starts —
        # without escape handling this parses as two commands and looks wired.
        ("echo x\\&\\& python scripts/check_contracts.py", False),
        # ...and an escape inside the path resolves, so this one does run.
        ("python scripts\\/check_contracts.py", True),
    ],
)
def test_only_a_real_invocation_counts(command: str, runs: bool):
    """The matcher itself, since the guard is only as good as it."""
    assert _invokes(command, "check_contracts.py") is runs


@pytest.mark.parametrize(
    ("name", "command", "runs"),
    [
        # A quote that never closes is not a command the shell runs at all.
        ("unterminated single quote", "python scripts/check_contracts.py '", False),
        ("unterminated double quote", 'python scripts/check_contracts.py "', False),
        # A trailing backslash joins the next line onto this command, so the
        # path below is an argument to echo and nothing is invoked.
        ("continuation joins the lines", "echo continued \\\npython scripts/check_contracts.py", False),
        ("continuation inside quotes", 'echo "a \\\npython scripts/check_contracts.py"', False),
        # A quote opened on one line swallows the newline and everything after.
        ("quote spanning lines", "echo 'a\npython scripts/check_contracts.py'", False),
        # ...but a plain newline separates, and a comment ends at its own line.
        ("two separate lines", "echo hi\npython scripts/check_contracts.py", True),
        ("comment ends at the line", "true # x\npython scripts/check_contracts.py", True),
        # A continuation that joins an invocation back together still runs it —
        # the branch has to remove the pair, not merely stop the newline ending
        # the word. Splitting a long command this way is ordinary in CI YAML.
        ("continued invocation", "python \\\nscripts/check_contracts.py", True),
        ("continuation inside a quote", 'python "scripts/check_\\\ncontracts.py"', True),
    ],
)
def test_state_carries_across_lines(name: str, command: str, runs: bool):
    """A `run:` block is one script, not a list of independent lines.

    Scanning it line by line loses the two things that cross a boundary — a
    trailing backslash continuing the command, and a quote that stays open — and
    both turn into false positives: a path that is an argument to echo, read as
    an invocation.
    """
    assert _invokes(command, "check_contracts.py") is runs


@pytest.mark.parametrize("script", _guard_scripts(), ids=lambda p: p.name)
def test_every_guard_script_is_run_by_ci(script: pathlib.Path):
    """A guard that CI never invokes cannot fail, so it protects nothing."""
    assert any(_invokes(cmd, script.name) for cmd in _ci_commands()), (
        f"scripts/{script.name} is never run by any workflow. It can only pass, whatever "
        "the repository does. Add a step for it, or drop the check_ prefix if it is a tool "
        "rather than a guard."
    )


def test_the_coupling_guard_runs_after_its_dependencies_are_installed():
    """check_dependency_groups.py is the one guard that cannot run stdlib-only.

    The other two that run before `pip install` do so deliberately: they are
    stdlib-only, and running them first means a manifest that lets the resolver
    choose an unpinned package is rejected before that package's build backend
    executes on the runner. This one reads couplings from importlib.metadata and
    parses YAML, so it has to come after the install — and placing it with the
    early pair would fail on the import rather than on the invariant.
    """
    steps = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())["jobs"]["contracts"]["steps"]
    runs = [(step or {}).get("run", "") for step in steps]
    install = next(i for i, r in enumerate(runs) if r.startswith("pip install"))
    guard = next(i for i, r in enumerate(runs) if "check_dependency_groups.py" in r)
    assert guard > install, (
        "check_dependency_groups.py runs before the pip install it depends on; it would "
        "fail on `import yaml` rather than on a split dependency group."
    )
    assert "requirements-dev" in runs[install], (
        f"the contracts job installs {runs[install]!r}, which does not carry PyYAML — "
        "check_dependency_groups.py imports yaml."
    )
