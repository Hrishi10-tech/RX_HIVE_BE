"""Postgres and Redis are pinned in four places. One of them is watched.

    engine      ci.yml      compose.yml   compose.prod.yml   terraform
    postgres    16-alpine   16-alpine     -- (managed RDS)   engine_version "16"
    redis       7-alpine    7-alpine      7-alpine           --

All four agree today. Nothing makes them.

Dependabot covers infra/ under the docker-compose ecosystem and moves those
files on its own. It does not cover the service images in ci.yml — the
github-actions ecosystem updates `uses:` references and nothing else — and its
terraform ecosystem tracks provider and module versions, not an `engine_version`
string on a resource. So a routine bump moves exactly one of the four.

Two such PRs are open right now: #57 takes compose from postgres:16-alpine to
18-alpine, and #58 takes redis from 7-alpine to 8-alpine. Merging #57 leaves the
development stack on Postgres 18, CI validating every migration and query
against 16, and production on an RDS instance created at 16. Two majors between
what developers run and what CI proves, in the direction where CI is the one
that is wrong — it would go green on SQL that production cannot run.

This does not decide which version is right. It asserts only that the four
declarations say the same thing, so moving one of them is a failure that names
the other three rather than a silent divergence discovered later.
"""

import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
COMPOSE_DEV = ROOT / "infra" / "docker-compose.yml"
COMPOSE_PROD = ROOT / "infra" / "docker-compose.prod.yml"
TERRAFORM_DATA = ROOT / "infra" / "terraform" / "data.tf"

# The application's RDS instance, by name. A DOTALL search for engine_version
# across the whole file can pair the `engine` of one resource with the
# `engine_version` of another — there is one instance today, and a read replica
# or a second database would silently become the thing this check reads.
_RDS_MAIN_BLOCK = re.compile(r'^resource\s+"aws_db_instance"\s+"main"\s*\{(.*?)^\}', re.DOTALL | re.MULTILINE)
_ENGINE = re.compile(r'^\s*engine\s*=\s*"([^"]+)"', re.MULTILINE)
_ENGINE_VERSION = re.compile(r'^\s*engine_version\s*=\s*"([^"]+)"', re.MULTILINE)


def _rds_engine_version(terraform: str) -> str | None:
    """The engine version of `aws_db_instance.main`, if it is a Postgres instance.

    Takes the text rather than reading the file, so the block scoping can be
    tested against a decoy resource without writing one into infra/.
    """
    block = _RDS_MAIN_BLOCK.search(terraform)
    if block is None:
        return None
    body = block.group(1)
    engine = _ENGINE.search(body)
    version = _ENGINE_VERSION.search(body)
    if engine is None or engine.group(1) != "postgres" or version is None:
        return None
    return version.group(1)


def _major(reference: str) -> str:
    """The major version in an image tag or a bare version string.

    `postgres:16-alpine` and `16` both give `16`. Comparing majors rather than
    whole tags is deliberate: RDS takes a major only, by design — AWS resolves it
    to the latest 16.x at create time — so a check on the full tag could never
    pass against `16-alpine`.
    """
    tag = reference.rpartition(":")[2] if ":" in reference else reference
    match = re.match(r"\d+", tag)
    assert match, f"no major version in {reference!r}"
    return match.group(0)


def _ci_images(service: str) -> dict[str, str]:
    """Every declaration of `service` among the workflow's service containers.

    Keyed by job, because the backend and e2e jobs declare them separately and
    could drift from each other as easily as from compose.
    """
    workflow = yaml.safe_load(WORKFLOW.read_text())
    found = {}
    for job, spec in (workflow.get("jobs") or {}).items():
        image = ((spec.get("services") or {}).get(service) or {}).get("image")
        if image:
            found[f"ci.yml ({job} job)"] = image
    return found


def _compose_image(path: pathlib.Path, service: str) -> dict[str, str]:
    """The image `service` is pinned at in one compose file, if it appears."""
    compose = yaml.safe_load(path.read_text()) or {}
    image = ((compose.get("services") or {}).get(service) or {}).get("image")
    return {f"infra/{path.name}": image} if image else {}


def _declared(service: str) -> dict[str, str]:
    """Every place this repository states a version for `service`."""
    found = dict(_ci_images(service))
    found.update(_compose_image(COMPOSE_DEV, service))
    found.update(_compose_image(COMPOSE_PROD, service))
    if service == "postgres":
        version = _rds_engine_version(TERRAFORM_DATA.read_text())
        if version:
            found["infra/terraform/data.tf (RDS engine_version)"] = version
    return found


# Named rather than counted. A count lets a declaration drop out unnoticed as
# long as enough others remain — renaming compose's `redis` service to
# `redis-cache` removed a source and left a count-based check green, which is
# the same silent-divergence failure this file is about.
EXPECTED_SOURCES = {
    "postgres": {
        "ci.yml (backend job)",
        "ci.yml (e2e job)",
        "infra/docker-compose.yml",
        "infra/terraform/data.tf (RDS engine_version)",
    },
    "redis": {
        "ci.yml (backend job)",
        "ci.yml (e2e job)",
        "infra/docker-compose.yml",
        "infra/docker-compose.prod.yml",
    },
}


@pytest.mark.parametrize("service", sorted(EXPECTED_SOURCES))
def test_every_declaration_is_still_found(service: str):
    """Guards the comparison below, which passes trivially on one source.

    A declaration that stops being visible here — renamed service, restructured
    file, a version moved into a variable — silently shrinks the check instead of
    failing it. A new declaration has to be added to this set too, which is the
    point: it then has to agree with the others.
    """
    assert set(_declared(service)) == EXPECTED_SOURCES[service]


@pytest.mark.parametrize("service", sorted(EXPECTED_SOURCES))
def test_the_same_major_version_everywhere(service: str):
    """CI has to run the engine that development and production run."""
    declared = _declared(service)
    majors = {source: _major(ref) for source, ref in declared.items()}
    distinct = set(majors.values())
    assert len(distinct) == 1, (
        f"{service} is pinned at {len(distinct)} different major versions: "
        + "; ".join(f"{source} -> {ref}" for source, ref in sorted(declared.items()))
        + ". Dependabot moves infra/ on its own but covers neither ci.yml's service "
        "images nor the RDS engine_version, so a bump lands in one file at a time. "
        "Move them together — and note that the RDS engine_version is a real major "
        "upgrade of the production database, not a tag change."
    )


_DECOY_TERRAFORM = """
resource "aws_db_instance" "analytics" {
  engine         = "postgres"
  engine_version = "14"
}

resource "aws_db_instance" "main" {
  identifier     = "rxhive-prod"
  engine         = "postgres"
  engine_version = "16"
}

resource "aws_db_instance" "replica" {
  engine         = "postgres"
  engine_version = "18"
}
"""


def test_the_rds_version_comes_from_the_main_instance_only():
    """A second Postgres resource must not be read in place of `main`.

    There is one instance today, so an unscoped search happens to be right. A
    read replica or an analytics database would make it wrong without anyone
    touching this file — and it would still pass, reading someone else's version.
    """
    assert _rds_engine_version(_DECOY_TERRAFORM) == "16"


def test_a_non_postgres_main_instance_is_not_read_as_postgres():
    """The version is only meaningful next to the engine it belongs to."""
    mysql = _DECOY_TERRAFORM.replace(
        '  engine         = "postgres"\n  engine_version = "16"',
        '  engine         = "mysql"\n  engine_version = "8"',
    )
    assert _rds_engine_version(mysql) is None


def test_the_real_terraform_still_parses():
    """The decoys prove the scoping; this proves it still finds the real instance."""
    assert _rds_engine_version(TERRAFORM_DATA.read_text()) == "16"
