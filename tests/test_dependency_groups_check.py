"""Tests for scripts/check_dependency_groups.py.

The check exists because grouping pip updates by update type alone ignored which
pins constrain each other, and Dependabot then proposed manifests pip refuses to
resolve. Four of the twenty-one PRs open on 2026-09-03 failed at `pip install` for
that reason, none of them a broken dependency:

  #71  starlette past fastapi's starlette<0.47.0
  #67  pywebpush 2.5.0 needing cryptography>=47.0.0 against a pinned 44.0.0
  #70  pytest 9.1.1 against pytest-asyncio 0.25.3's pytest<9
  #68  pytest-asyncio 1.4.0 needing pytest>=8.4 against a pinned 8.3.4

A guard against that is only worth having if it cannot quietly pass, and the ways
this one could are specific: an update-types restriction on a cluster lets a major
escape into its own PR, a member left out of the patterns or removed by
exclude-patterns splits off, an earlier overlapping group takes one member, and
three separate things make the graph itself wrong rather than the grouping — a pin
whose metadata is not installed, a pin installed at a version other than the one
pinned, and a pin whose extras are dropped so the extra's requirements evaluate
False without raising. Each is asserted to fail here.
"""

import importlib.util
import pathlib
import textwrap

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "check_dependency_groups.py"
_RUNTIME = _ROOT / "backend" / "requirements.txt"
_DEV = _ROOT / "backend" / "requirements-dev.txt"
_MANIFESTS = [str(_RUNTIME), str(_DEV)]


def _load():
    spec = importlib.util.spec_from_file_location("check_dependency_groups", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load()


def _write_config(tmp_path: pathlib.Path, groups_yaml: str) -> pathlib.Path:
    config = tmp_path / "dependabot.yml"
    config.write_text(
        "version: 2\n"
        "updates:\n"
        "  - package-ecosystem: pip\n"
        "    directory: /backend\n"
        "    groups:\n" + textwrap.indent(textwrap.dedent(groups_yaml).strip() + "\n", "      ")
    )
    return config


# Mirrors the committed pip groups. Every assertion below mutates one thing in it.
REAL_CLUSTERS = """
python-asgi-stack:
  patterns: [fastapi, starlette, "pydantic*", anyio]
python-http-stack:
  patterns: [requests, urllib3, pywebpush, cryptography]
python-database:
  patterns: [alembic, sqlalchemy, asyncpg]
python-livekit:
  patterns: ["livekit*", pyjwt, aiohttp]
python-pytest:
  patterns: ["pytest*"]
python-uvicorn-standard:
  patterns: [uvicorn, pyyaml, websockets]
python-minor-and-patch:
  patterns: ["*"]
  update-types: [minor, patch]
"""


def test_the_committed_config_passes(checker, capsys):
    """The real file, unmodified. Every assertion below is relative to this."""
    assert checker.main(_MANIFESTS) == 0
    assert "coupled cluster" in capsys.readouterr().out


def test_the_couplings_are_actually_found(checker):
    """If this graph were empty the check would pass on any config at all."""
    pins = checker.read_pins([_RUNTIME, _DEV])
    edges, missing, mismatched = checker.couplings(pins)
    assert not missing
    assert not mismatched
    pairs = {(dependent, dependency) for dependent, dependency, _ in edges}
    # The four that produced red PRs, plus two more upper bounds already in force.
    assert ("fastapi", "starlette") in pairs
    assert ("pywebpush", "cryptography") in pairs
    assert ("pytest-asyncio", "pytest") in pairs
    assert ("requests", "urllib3") in pairs
    assert ("starlette", "anyio") in pairs
    # Only reachable because the extras on a pin are honoured.
    assert ("uvicorn", "websockets") in pairs
    assert sorted(len(c) for c in checker.components(pins, edges)) == [2, 2, 3, 3, 4, 5]


def test_update_types_on_a_cluster_lets_a_major_escape(checker, tmp_path, monkeypatch, capsys):
    """Exactly how #71 happened: the cluster grouped, the major bump not."""
    restricted = REAL_CLUSTERS.replace(
        'python-asgi-stack:\n  patterns: [fastapi, starlette, "pydantic*", anyio]',
        'python-asgi-stack:\n  patterns: [fastapi, starlette, "pydantic*", anyio]'
        "\n  update-types: [minor, patch]",
    )
    monkeypatch.setattr(checker, "CONFIG", _write_config(tmp_path, restricted))
    assert checker.main(_MANIFESTS) == 1
    err = capsys.readouterr().err
    assert "major bumps split" in err
    assert "its own PR" in err


def test_a_member_dropped_from_the_patterns_splits_the_cluster(checker, tmp_path, monkeypatch, capsys):
    """The #67 shape: cryptography left behind while pywebpush moves."""
    without_crypto = REAL_CLUSTERS.replace(
        "patterns: [requests, urllib3, pywebpush, cryptography]",
        "patterns: [requests, urllib3, pywebpush]",
    )
    monkeypatch.setattr(checker, "CONFIG", _write_config(tmp_path, without_crypto))
    assert checker.main(_MANIFESTS) == 1
    err = capsys.readouterr().err
    assert "cryptography" in err
    assert "pywebpush" in err


def test_the_pytest_pair_must_stay_together(checker, tmp_path, monkeypatch, capsys):
    """#70 and #68 are this cluster split, hit from opposite ends."""
    without_pytest = REAL_CLUSTERS.replace('python-pytest:\n  patterns: ["pytest*"]\n', "")
    monkeypatch.setattr(checker, "CONFIG", _write_config(tmp_path, without_pytest))
    assert checker.main(_MANIFESTS) == 1
    err = capsys.readouterr().err
    assert "pytest" in err
    assert "pytest-asyncio" in err


def test_exclude_patterns_are_honoured(checker, tmp_path, monkeypatch, capsys):
    """A cluster member excluded by name is still a split, not still a member."""
    excluded = REAL_CLUSTERS.replace(
        "patterns: [requests, urllib3, pywebpush, cryptography]",
        "patterns: [requests, urllib3, pywebpush, cryptography]\n  exclude-patterns: [cryptography]",
    )
    monkeypatch.setattr(checker, "CONFIG", _write_config(tmp_path, excluded))
    assert checker.main(_MANIFESTS) == 1
    assert "cryptography" in capsys.readouterr().err


def test_catch_all_first_is_still_whole_because_the_types_are_disjoint(checker, tmp_path, monkeypatch):
    """Ordering against the catch-all is a preference here, and the check says so.

    The catch-all takes only [minor, patch] and the cluster groups are unrestricted,
    so the two partitions never compete for the same bump: majors resolve to the
    cluster either way. This asserts the honest outcome rather than the tidier claim
    that the catch-all must come last — the earlier version of this test asserted the
    tidier claim and failed.
    """
    clusters, _, catch_all = REAL_CLUSTERS.partition("python-minor-and-patch")
    reordered = f"python-minor-and-patch{catch_all}\n{clusters.strip()}"
    monkeypatch.setattr(checker, "CONFIG", _write_config(tmp_path, reordered))
    assert checker.main(_MANIFESTS) == 0


def test_an_earlier_overlapping_group_splits_a_cluster(checker, tmp_path, monkeypatch, capsys):
    """Where ordering IS load-bearing: two cluster groups whose patterns overlap.

    A group matching sqlalchemy alone, listed above python-database, takes it out of
    the cluster and leaves alembic behind — alembic requires sqlalchemy, so that is a
    pair pip can be handed inconsistently.
    """
    overlapping = 'python-orm-only:\n  patterns: ["*alchemy*"]\n' + REAL_CLUSTERS
    monkeypatch.setattr(checker, "CONFIG", _write_config(tmp_path, overlapping))
    assert checker.main(_MANIFESTS) == 1
    err = capsys.readouterr().err
    assert "python-orm-only" in err
    assert "python-database" in err


def test_an_uninstalled_pin_fails_rather_than_being_skipped(checker, tmp_path, capsys):
    """A missing distribution removes edges. Passing on that is passing on nothing."""
    manifest = tmp_path / "requirements.txt"
    manifest.write_text("rxhive-not-a-real-package==1.0.0\n")
    assert checker.main([str(manifest)]) == 1
    err = capsys.readouterr().err
    assert "rxhive-not-a-real-package" in err
    assert "no installed metadata for" in err


def test_a_pin_installed_at_another_version_fails(checker, tmp_path, capsys):
    """requires() answers for the installed version, not the pinned one.

    A stale environment would otherwise have this validate a graph that is not the
    one being shipped, and bless a configuration that cannot install.
    """
    manifest = tmp_path / "requirements.txt"
    manifest.write_text("fastapi==0.1.0\n")
    assert checker.main([str(manifest)]) == 1
    err = capsys.readouterr().err
    assert "wrong version installed for fastapi" in err
    assert "pinned 0.1.0" in err


def test_extras_on_a_pin_are_honoured(checker, tmp_path):
    """An extra's requirements are gated behind `extra == "..."` markers.

    Those evaluate False with no extra in the environment — silently, without
    raising — so dropping the extras dropped every coupling the pin contributed.
    redis[ocsp] requires cryptography>=36.0.1 and requests>=2.31.0, both pinned.
    """
    manifest = tmp_path / "requirements.txt"
    manifest.write_text("redis[ocsp]==5.2.1\ncryptography==44.0.0\nrequests==2.34.2\n")
    pins = checker.read_pins([manifest])
    assert pins["redis"] == ("5.2.1", frozenset({"ocsp"}))
    edges, missing, mismatched = checker.couplings(pins)
    assert not missing and not mismatched
    pairs = {(dependent, dependency) for dependent, dependency, _ in edges}
    assert ("redis", "cryptography") in pairs
    assert ("redis", "requests") in pairs


def test_an_extra_that_is_not_requested_stays_gated(checker, tmp_path):
    """The converse, or the test above would pass on a marker check that always says yes."""
    manifest = tmp_path / "requirements.txt"
    manifest.write_text("redis==5.2.1\ncryptography==44.0.0\nrequests==2.34.2\n")
    pins = checker.read_pins([manifest])
    assert pins["redis"] == ("5.2.1", frozenset())
    edges, _, _ = checker.couplings(pins)
    assert not [e for e in edges if e[0] == "redis"]


@pytest.mark.parametrize(
    ("written", "normalized"),
    [
        ("PyJWT", "pyjwt"),
        ("livekit_api", "livekit-api"),
        ("Foo__bar", "foo-bar"),
        ("foo--bar", "foo-bar"),
        ("foo..bar", "foo-bar"),
        ("Foo_.bar", "foo-bar"),
    ],
)
def test_names_use_pep_503_normalization(checker, written, normalized):
    """Runs of -, _ and . collapse to a single -.

    Hand-rolled as two str.replace calls this turned Foo__bar into foo--bar, and
    since couplings() matches dependency names against the pins by exact membership,
    an equivalent spelling would simply not match and the edge would vanish.
    """
    assert checker.canonical(written) == normalized


def test_glob_patterns_survive_normalization(checker):
    """The patterns are globs, and normalizing them must not eat the wildcards."""
    assert [checker.canonical(p) for p in ("*", "pydantic*", "*alchemy*", "livekit*")] == [
        "*",
        "pydantic*",
        "*alchemy*",
        "livekit*",
    ]


def test_a_pin_gated_off_by_a_root_marker_is_skipped(checker, tmp_path, capsys):
    """pip skips it on this platform, so demanding its metadata would misfire.

    The guard insists every pin it governs has matching installed metadata. A root
    requirement whose environment marker is false was never installed, so requiring
    it would fail the check for a reason that has nothing to do with grouping.
    """
    manifest = tmp_path / "requirements.txt"
    manifest.write_text(
        'rxhive-not-a-real-package==1.0.0; sys_platform == "rxhive-no-such-platform"\nfastapi==0.115.12\n'
    )
    assert checker.read_pins([manifest]) == {"fastapi": ("0.115.12", frozenset())}
    assert checker.main([str(manifest)]) == 0
    assert "rxhive-not-a-real-package" not in capsys.readouterr().err


def test_a_pin_whose_root_marker_is_true_is_kept(checker, tmp_path):
    """The converse, or the test above would pass on a check that dropped every marker."""
    manifest = tmp_path / "requirements.txt"
    manifest.write_text('fastapi==0.115.12; python_version >= "3.8"\n')
    assert checker.read_pins([manifest]) == {"fastapi": ("0.115.12", frozenset())}


def test_unpinned_and_commented_lines_are_ignored(checker, tmp_path):
    manifest = tmp_path / "requirements.txt"
    manifest.write_text(
        "# fastapi==0.1.0\nrequests>=2.0\n-r requirements.txt\nfastapi==0.115.12  # trailing\n\n"
    )
    assert checker.read_pins([manifest]) == {"fastapi": ("0.115.12", frozenset())}
