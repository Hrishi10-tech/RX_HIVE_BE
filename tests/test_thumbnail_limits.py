"""What one upload can cost, and how many of them can cost it at once.

storage.make_thumbnail already refused a decompression bomb by its declared
dimensions. What nothing bounded was CONCURRENCY, and the two guards around it
measure different things: THUMBNAIL_SOURCE_LIMIT bounds the encoded bytes a
caller sends, while the memory goes on decoded pixels. A source just under the
pixel cap encodes to tens of kilobytes and decodes to hundreds of megabytes, so
the half of the guard chain that is cheap to satisfy is the half an attacker
controls.
"""

import io
import logging

import anyio
import anyio.to_thread
import pytest
from PIL import Image

from app.services import storage
from app.services.storage import THUMBNAIL_SOURCE_LIMIT, make_thumbnail

# 6300x6300 is 39.69 MP: the largest square the 40 MP cap admits, so it is the
# worst case make_thumbnail will actually agree to decode.
WORST_ALLOWED_EDGE = 6300


def _encode(edge: int, fmt: str = "WEBP") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (edge, edge), (120, 30, 200)).save(buf, fmt)
    return buf.getvalue()


def test_the_worst_case_source_is_cheap_to_send_and_expensive_to_decode():
    """The asymmetry the limiter exists for, stated as an assertion.

    If this ever stops holding — if the encoded size of a cap-sized image grew to
    approach THUMBNAIL_SOURCE_LIMIT — then the source limit would be doing the
    work on its own and the concurrency bound would matter less. It does not.
    """
    data = _encode(WORST_ALLOWED_EDGE)
    megapixels = (WORST_ALLOWED_EDGE**2) / 1e6

    assert megapixels < Image.MAX_IMAGE_PIXELS / 1e6, "the guard would reject this outright"
    assert len(data) < THUMBNAIL_SOURCE_LIMIT / 100, (
        f"a {megapixels:.1f} MP source encodes to {len(data) / 1e6:.2f} MB, "
        f"against a {THUMBNAIL_SOURCE_LIMIT / 1e6:.0f} MB source limit"
    )
    # Decoded, the same bytes are three orders of magnitude larger.
    assert (WORST_ALLOWED_EDGE**2) * 3 > 100 * len(data)


async def test_image_decoding_is_bounded_and_not_on_the_shared_pool():
    """A limiter of its own, smaller than the default, and not the PDF one.

    Sharing anyio's default 40 tokens would let image decoding starve every
    attachment put, presign and push send — the failure batch 32 established. And
    _PDF_LIMITER is 1 for a different reason (PDFium is not thread-safe), so
    reusing it here would silently serialise photo previews behind PDFs.
    """
    default_limiter = anyio.to_thread.current_default_thread_limiter()

    assert storage._IMAGE_LIMITER is not default_limiter
    assert storage._IMAGE_LIMITER is not storage._PDF_LIMITER
    assert storage._IMAGE_LIMITER.total_tokens < default_limiter.total_tokens
    assert storage._IMAGE_LIMITER.total_tokens >= 1


async def test_the_limiter_is_the_one_make_thumbnail_actually_uses():
    """Asserted by observation rather than by reading the call site.

    Borrowing every token first means a real make_thumbnail call cannot start
    until one is returned. If the function were still on the default pool this
    would sail straight through, so the timeout is the assertion.
    """
    limiter = storage._IMAGE_LIMITER
    total = int(limiter.total_tokens)
    data = _encode(64)  # tiny: any delay here is the limiter, not the decode

    holders = [object() for _ in range(total)]
    for holder in holders:
        await limiter.acquire_on_behalf_of(holder)
    try:
        with anyio.move_on_after(1.0) as scope:
            await make_thumbnail(data)
        assert scope.cancelled_caught, "make_thumbnail ran while its limiter was fully held"
    finally:
        for holder in holders:
            limiter.release_on_behalf_of(holder)

    # Released, it works again — so the bound throttles rather than breaks.
    assert await make_thumbnail(data) is not None


async def test_an_oversized_image_is_declined_and_says_so(caplog):
    """Still no preview, still not an error — but no longer silent."""
    edge = 8000  # 64 MP, past the cap
    assert edge**2 > Image.MAX_IMAGE_PIXELS
    with caplog.at_level(logging.INFO, logger="app.services.storage"):
        assert await make_thumbnail(_encode(edge)) is None
    assert any("exceeds" in record.getMessage() for record in caplog.records), caplog.text


async def test_undecodable_bytes_are_declined_and_logged(caplog):
    """The finding this file's MAXBLOCK note records: the upload must still
    succeed without a preview, but a failure that used to leave no trace at all
    now leaves one. A missing picture was the only symptom last time."""
    with caplog.at_level(logging.WARNING, logger="app.services.storage"):
        assert await make_thumbnail(b"this is not an image") is None
    assert any("thumbnail generation failed" in r.getMessage() for r in caplog.records), caplog.text


async def test_a_normal_photo_still_produces_a_thumbnail():
    """The bound must not change the ordinary outcome."""
    out = await make_thumbnail(_encode(1600, "JPEG"))
    assert out is not None
    thumb = Image.open(io.BytesIO(out))
    assert thumb.format == "JPEG"
    assert max(thumb.size) <= storage.THUMB_PX


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
async def test_every_supported_still_format_survives_the_bound(fmt):
    assert await make_thumbnail(_encode(800, fmt)) is not None
