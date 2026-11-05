"""Presigned-URL rewriting.

Guards two production failures that unit-less code hid:
  * minio signs VIRTUAL-HOST style against real AWS
    (<bucket>.s3.<region>.amazonaws.com), so the old prefix-replace against the
    configured endpoint silently no-opped and leaked a cross-origin URL.
  * CSP is enforced against redirect targets, so a cross-origin redirect out of
    /api/media is blocked by `img-src 'self'` and no attachment renders.
Keeping media same-origin via a path endpoint ("/s3") avoids both.
"""

import pytest

from app.services.storage import rewrite_to_public

AWS = "https://bkt.s3.ap-south-1.amazonaws.com/org/f.png?X-Amz-Signature=abc"


@pytest.mark.parametrize(
    "url,public,expected",
    [
        # Real AWS is virtual-host style; the bucket is in the HOST, not the path.
        (AWS, "/s3", "/s3/org/f.png?X-Amz-Signature=abc"),
        # MinIO (compose) is path-style; the bucket IS in the path and must survive.
        (
            "http://minio:9000/bkt/org/f.png?X-Amz-Signature=abc",
            "/s3",
            "/s3/bkt/org/f.png?X-Amz-Signature=abc",
        ),
        # Local dev: public endpoint already equals the signing endpoint.
        (
            "http://localhost:9000/bkt/k.png?sig=1",
            "http://localhost:9000",
            "http://localhost:9000/bkt/k.png?sig=1",
        ),
        # A CDN/alternate origin swaps host but keeps path + signature.
        (AWS, "https://cdn.example.com", "https://cdn.example.com/org/f.png?X-Amz-Signature=abc"),
        # Unset endpoint must not mangle the URL.
        (AWS, "", AWS),
    ],
)
def test_rewrite_to_public(url, public, expected):
    assert rewrite_to_public(url, public) == expected


def test_query_signature_is_never_dropped():
    """Losing the query string turns every attachment into a 403 from S3."""
    out = rewrite_to_public(AWS, "/s3")
    assert "X-Amz-Signature=abc" in out


def test_same_origin_endpoint_keeps_media_on_one_origin():
    """A path endpoint must produce a relative URL — anything absolute would be
    a second origin and get blocked by the app's img-src 'self' policy."""
    assert rewrite_to_public(AWS, "/s3").startswith("/s3/")
