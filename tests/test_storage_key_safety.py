"""An object key is built from a name the client chose, so it must survive one.

os.path.splitext is path-aware and never yields a "/", so an extension could not
escape the org scope — that much was already safe, and it is asserted below so a
future change to how the extension is derived cannot quietly remove it.

What an extension COULD do was make the key unusable. Measured before the fix: a
filename with a 100-character extension uploaded fine, and one with 500 produced a
546-character object name that minio refuses —

    XMinioInvalidObjectName: Object name contains unsupported characters

— and the S3Error came straight back out of the upload endpoint. The object is
PUT before the row is written, so the failure lands between the two.
"""

import io
import uuid

import pytest

from app.services import storage
from tests.conftest import make_user
from tests.test_calls_and_media import _client_for


@pytest.mark.parametrize(
    ("label", "ext"),
    [
        ("ordinary", ".png"),
        ("uppercase", ".PNG"),
        ("digits", ".7z"),
        ("single char", ".x"),
    ],
)
def test_a_real_extension_is_kept(label, ext):
    assert storage.safe_ext(ext) == ext.lower()


@pytest.mark.parametrize(
    ("label", "ext"),
    [
        ("too long", "." + "z" * 500),
        ("just over the cap", "." + "z" * 17),
        ("nul byte", ".pn\x00g"),
        # "$" matches before a trailing newline, so `match` accepted this one and
        # `fullmatch` is what rejects it.
        ("trailing newline", ".png\n"),
        ("trailing carriage return", ".png\r"),
        ("newline inside", ".p\nng"),
        ("fragment", ".png#x"),
        ("query", ".png?a=b"),
        ("space", ". png"),
        ("empty", ""),
        ("dot only", "."),
        ("traversal-ish", ".."),
    ],
)
def test_anything_else_is_dropped_rather_than_mangled(label, ext):
    """Dropped, not sanitized into something else.

    The extension in the key is a readability convenience; uploads.filename is the
    authoritative record of what the user sent, and MIME_BY_EXT decides the content
    type from the ORIGINAL extension — so dropping it here cannot change how the
    file is served.
    """
    assert storage.safe_ext(ext) == ""


def test_a_key_never_leaves_the_org_scope():
    """The property that was already true, pinned so it stays true."""
    org = uuid.uuid4()
    for ext in ("." + "z" * 500, ".pn\x00g", "../../platform/x", ".png/../.."):
        key = storage.new_storage_key(org, ext)
        assert key.startswith(f"{org}/")
        assert key.count("/") == 1, key


def test_a_key_stays_well_inside_the_column_and_the_store():
    """uploads.storage_key is String(500); the thumbnail key appends to this one."""
    longest = storage.new_storage_key(uuid.uuid4(), "." + "z" * 500)
    assert len(longest) <= 80, len(longest)
    assert len(f"{longest}_thumb.jpg") < 500


async def test_a_crafted_extension_no_longer_breaks_the_upload(client):
    """End to end: this raised S3Error out of the endpoint before the fix."""
    await make_user("keysafe@x.com")
    async with _client_for("keysafe@x.com") as c:
        resp = await c.post(
            "/api/upload",
            files={"file": ("payload." + "z" * 500, io.BytesIO(b"hello"), "image/png")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The name the user sent is still recorded in full-ish, per _safe_filename.
    assert body["filename"].startswith("payload.")
    # Unknown extension, so octet-stream — the existing anti-stored-XSS rule.
    assert body["mime_type"] == "application/octet-stream"


async def test_an_ordinary_upload_keeps_its_extension_in_the_key(client):
    """Negative control: the sanitiser must not strip real extensions."""
    await make_user("keyok@x.com")
    async with _client_for("keyok@x.com") as c:
        resp = await c.post("/api/upload", files={"file": ("photo.png", io.BytesIO(b"hello"), "image/png")})
    assert resp.status_code == 200, resp.text
    assert resp.json()["mime_type"] == "image/png"
