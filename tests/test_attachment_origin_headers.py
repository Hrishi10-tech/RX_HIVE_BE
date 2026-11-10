"""The attachment route's security headers, asserted against the Caddy config.

WHY THIS IS A TEST AND NOT JUST A CONFIG

api/media.py resolves an upload's Content-Type from `storage.MIME_BY_EXT` and
falls back to `application/octet-stream`, and its comment says why in as many
words: media is served from a SAME-ORIGIN path with `Content-Disposition:
inline`, so a stored type the browser renders is a stored-XSS primitive on the
app's own origin. It ends "Do not 'improve' this into mimetypes.guess_type."

That allow-list is a single-point mitigation for a documented XSS primitive, and
`X-Content-Type-Options: nosniff` is what stops the browser second-guessing the
type it picked. The bytes are served by Caddy straight from object storage —
reaching neither app/main.py's header middleware nor the SPA's nginx snippet — so
the header has to be declared in the Caddyfile, where nothing else in CI looks.

Deliberately a text assertion over the config rather than a live request: the
route needs Caddy, object storage and a signed URL to exercise, none of which the
suite has. This cannot prove Caddy applies the header; it proves the declaration
has not been dropped, which is the failure mode worth catching.
"""

import re
from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parents[2] / "infra"

# Both files, because dev is expected to mirror prod here — Caddyfile's own
# comments say "exactly as in Caddyfile.prod" about the neighbouring directive.
CADDYFILES = ["Caddyfile", "Caddyfile.prod"]

REQUIRED_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


def _s3_block(text: str) -> str:
    """The body of the `handle_path /s3/*` block, comments stripped.

    Comments are removed first so the prose above the directives — which names
    these very headers — cannot satisfy the assertions on its own.
    """
    start = re.search(r"^handle_path\s+/s3/\*\s*\{", text, re.MULTILINE)
    assert start, "no `handle_path /s3/*` block found — did the route move?"
    depth, i = 0, start.end() - 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                body = text[start.end() : i]
                return "\n".join(line for line in body.split("\n") if not line.strip().startswith("#"))
        i += 1
    raise AssertionError("unbalanced braces in the /s3 block")


@pytest.mark.parametrize("filename", CADDYFILES)
def test_attachment_route_declares_its_security_headers(filename):
    block = _s3_block((INFRA / filename).read_text())
    # Sanity: we are looking at directives, not at the comment block above them.
    assert "reverse_proxy" in block, f"{filename}: /s3 block did not parse as directives"

    for name, value in REQUIRED_HEADERS.items():
        assert re.search(rf"^\s*{re.escape(name)}\s+{re.escape(value)}\s*$", block, re.MULTILINE), (
            f"{filename}: the /s3 route no longer declares {name} {value}. "
            "These bytes are user-uploaded content on the app's own origin, and "
            "this route reaches neither main.py's middleware nor nginx's snippet."
        )


def test_the_mime_allow_list_still_excludes_renderable_types():
    """The other half of the same mitigation.

    nosniff stops the browser overriding the declared type; this stops a
    renderable type being declared in the first place. Either alone leaves the
    stored-XSS primitive api/media.py describes reachable, so both are pinned.
    """
    from app.services import storage

    renderable = {"text/html", "application/xhtml+xml", "image/svg+xml", "text/xml", "application/xml"}
    # Compare the media type alone. A value carrying parameters — the ordinary
    # shape for text types, "text/html; charset=utf-8" — is the same document to
    # the browser and would slide past an equality test against the bare name,
    # which is exactly the regression this exists to catch.
    offenders = {
        ext: mime
        for ext, mime in storage.MIME_BY_EXT.items()
        if mime.split(";", 1)[0].strip().lower() in renderable
    }
    assert not offenders, (
        f"MIME_BY_EXT maps {offenders} — a type the browser renders as a document. "
        "Attachments are served same-origin with Content-Disposition: inline."
    )
