"""CI's SFU is a pinned, checksum-verified LiveKit, not whatever is newest.

The calling specs run against a real LiveKit, installed in the e2e job. It came
from get.livekit.io, which installs the latest release at run time, so on
2026-10-07 the SFU under those specs moved from 1.13.7 to 1.13.8 with no commit
behind it. 1.13.8 cuts a 1:1 callee's data channels when the server deletes the
room at call end, calling.spec's console check caught it, and the build went red
on a branch that touched no call code (batch 73). Pinned, with the archive's
checksum checked, an upgrade is a commit that runs the calling specs against the
new SFU before anyone has to trust it.
"""

import pathlib
import re

import pytest

CI = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"


def test_livekit_is_pinned_to_a_version_and_its_checksum():
    """The e2e job installs a pinned, checksum-verified LiveKit instead of piping get.livekit.io to sh."""
    # The backend can be built and tested on its own (a container build, a checkout of
    # backend/ alone), where the workflow is not in the tree; CI always has it
    # (CodeRabbit, review of 3bdf4ee).
    if not CI.is_file():
        pytest.skip("the CI workflow is outside this backend tree")
    text = CI.read_text()
    # The command, not the name: the comment beside the pin explains why it left.
    piped = re.search(r"get\.livekit\.io\S*\s*\|\s*(sudo\s+)?(ba)?sh", text)
    assert not piped, "the e2e job installs whatever LiveKit is newest again"
    version = re.search(r"^\s*LIVEKIT_VERSION=\d+\.\d+\.\d+\s*$", text, re.M)
    digest = re.search(r"^\s*LIVEKIT_SHA256=[0-9a-f]{64}\s*$", text, re.M)
    assert version, "LiveKit is not pinned to a release"
    assert digest, "the LiveKit archive has no checksum"
    assert '"${LIVEKIT_SHA256}  /tmp/livekit.tar.gz" | sha256sum -c -' in text, (
        "the LiveKit archive is not checked against its checksum before it is installed"
    )
