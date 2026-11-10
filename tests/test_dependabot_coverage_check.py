"""Tests for scripts/check_dependabot_coverage.py.

The config it guards exists because ci.yml spent months telling readers
"(Dependabot handles pinned SHAs and rewrites the comment)" while no Dependabot
config existed — every pinned action three major versions behind, with a comment
reassuring anyone who looked.

A coverage check is only worth having if it cannot quietly pass, and this one
already failed that once: keying terraform on "main.tf" detected nothing, because
infra/terraform holds compute.tf and seven others. It passed while blind to the
whole ecosystem. These cover detection per ecosystem rather than the aggregate.
"""

import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "check_dependabot_coverage.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_dependabot_coverage", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load()


def test_the_real_repository_is_covered(checker):
    assert checker.main() == 0


@pytest.mark.parametrize(
    "ecosystem,directory",
    [
        ("pip", "/backend"),
        ("npm", "/frontend"),
        ("docker", "/backend"),
        ("docker", "/frontend"),
        ("terraform", "/infra/terraform"),
        ("docker-compose", "/infra"),
        ("github-actions", "/"),
    ],
)
def test_every_ecosystem_is_actually_detected(checker, ecosystem, directory):
    """Named individually, not counted.

    The aggregate assertion is what let the terraform blind spot through: five
    detected looked plausible, and nothing said which five.
    """
    present = checker.present_ecosystems()
    assert (ecosystem, directory) in present, sorted(present)


def test_terraform_is_found_without_a_main_tf(checker):
    """The specific bug. infra/terraform has no main.tf, so a filename match on it
    detected nothing and the check could not have caught a missing entry."""
    source = checker.present_ecosystems()[("terraform", "/infra/terraform")]
    assert source.suffix == ".tf"
    assert source.name != "main.tf"


def test_vendored_manifests_are_ignored(checker):
    """ios/build holds SPM checkouts with their own package manifests and even
    their own dependabot.yml. Treating those as ours would demand entries for
    somebody else's dependencies.

    They are excluded because they are UNTRACKED, not because a blocklist names
    them — which is also why this stays true for the next vendored tree nobody
    thought to list.
    """
    # Asserted on the PATHS, not against IGNORED_PARTS. Exclusion is by git
    # tracking now, so testing the blocklist tested a rule the code no longer
    # uses: a regression that started detecting ios/build/.../Package.resolved
    # would have passed, because that path's exclusion has nothing to do with
    # the blocklist any more.
    vendored_roots = ("ios/build", "ios/build-device", "node_modules", ".claude")
    for (ecosystem, directory), source in checker.present_ecosystems().items():
        text = source.as_posix()
        assert not any(text.startswith(root) for root in vendored_roots), (
            f"{ecosystem} at {directory} came from a vendored tree: {source}"
        )


def test_only_git_tracked_files_count(checker):
    """Dependabot reads the repository, so an untracked manifest is invisible to it.

    ios/RxHive.xcodeproj/.../Package.resolved is a real Swift manifest that
    .gitignore line 46 excludes. A filesystem walk would demand a `swift` entry for
    it, and that entry would be inert — Dependabot could never act on a file it
    cannot see. Review asked for SPM detection; this is the half that makes the
    detection correct rather than merely present.
    """
    tracked = set(checker.repository_files())
    assert pathlib.Path("backend/requirements.txt") in tracked
    assert (
        pathlib.Path("ios/RxHive.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved")
        not in tracked
    )
    # So no swift ecosystem is reported, and none is configured.
    assert not any(eco == "swift" for eco, _ in checker.present_ecosystems())


def test_a_tracked_swift_manifest_would_demand_an_entry(checker, monkeypatch):
    """The regression test review asked for: SPM is recognised the moment it is
    committed, rather than being a gap nobody notices."""
    monkeypatch.setattr(
        checker,
        "repository_files",
        lambda: [pathlib.Path("ios/Package.resolved")],
    )
    present = checker.present_ecosystems()
    assert ("swift", "/ios") in present, sorted(present)
    # And the real config has no swift entry, so coverage must fail.
    assert checker.main() == 1


def test_every_configured_entry_groups_minor_and_patch(checker):
    """A routine week has to be one PR per ecosystem, not five.

    Three entries were left ungrouped in the first version — the two Dockerfiles
    and terraform — which contradicted the rationale written directly above them.
    """
    # Parsed, not string-matched. Matching the exact inline spelling
    # `update-types: [minor, patch]` was wrong twice over: an equivalent
    # block-style sequence would have failed it, and a COMMENT containing that
    # text would have satisfied it. The first version also counted the file's
    # header comment as a seventh entry and asserted 7 == 6.
    import yaml

    config = yaml.safe_load(checker.CONFIG.read_text())
    assert config["version"] == 2
    entries = config["updates"]
    assert len(entries) == 7, [e["package-ecosystem"] for e in entries]

    ungrouped = []
    for entry in entries:
        groups = entry.get("groups") or {}
        grouped = any(set(group.get("update-types") or []) >= {"minor", "patch"} for group in groups.values())
        if not grouped:
            ungrouped.append(f"{entry['package-ecosystem']}@{entry['directory']}")
    assert not ungrouped, f"entries without a minor/patch group: {ungrouped}"


def test_a_missing_entry_fails(checker, tmp_path, monkeypatch):
    config = tmp_path / "dependabot.yml"
    config.write_text("version: 2\nupdates:\n  - package-ecosystem: pip\n    directory: /backend\n")
    monkeypatch.setattr(checker, "CONFIG", config)
    assert checker.main() == 1


def test_a_missing_config_fails(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "CONFIG", tmp_path / "nothing-here.yml")
    assert checker.main() == 1


def test_mismatched_keys_are_rejected(checker, tmp_path, monkeypatch):
    """An entry with an ecosystem and no directory would otherwise pair wrongly
    with the next entry's directory and report coverage that does not exist."""
    config = tmp_path / "dependabot.yml"
    config.write_text(
        "version: 2\nupdates:\n"
        "  - package-ecosystem: pip\n"
        "  - package-ecosystem: npm\n    directory: /frontend\n"
    )
    monkeypatch.setattr(checker, "CONFIG", config)
    with pytest.raises(SystemExit) as exit_info:
        checker.configured_ecosystems()
    assert exit_info.value.code == 1


def test_compose_files_are_detected_as_their_own_ecosystem(checker):
    """`docker-compose` is a distinct Dependabot ecosystem from `docker`.

    Both infra compose files are tracked and pin the images the production box
    actually runs — postgres, redis, caddy, minio, livekit-server — and none was
    covered while this script reported full coverage, because MANIFESTS had no
    entry for the filename.
    """
    present = checker.present_ecosystems()
    assert ("docker-compose", "/infra") in present, sorted(present)
    assert present[("docker-compose", "/infra")].name.startswith("docker-compose")


def test_github_actions_is_derived_from_tracked_paths(checker, monkeypatch):
    """It was an is_dir() check on the filesystem, so it would have counted an
    untracked .github/workflows and missed a tracked root action.yml — the two
    cases Dependabot itself treats as the same `/` entry."""
    monkeypatch.setattr(checker, "repository_files", lambda: [pathlib.Path("action.yml")])
    assert ("github-actions", "/") in checker.present_ecosystems()

    monkeypatch.setattr(checker, "repository_files", lambda: [pathlib.Path("README.md")])
    assert not any(eco == "github-actions" for eco, _ in checker.present_ecosystems())
