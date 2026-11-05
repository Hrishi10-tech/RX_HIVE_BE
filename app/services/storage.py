"""Object storage (MinIO / any S3) — uploads, thumbnails, presigned reads.

Files never live on the app disk and are never served unauthenticated:
reads go through /api/media/* which checks membership then redirects to a
short-lived presigned URL.
"""

import datetime as dt
import io
import logging
import re
import uuid
from functools import lru_cache
from urllib.parse import urlsplit

import anyio
import pypdfium2 as pdfium
from minio import Minio
from minio.credentials import IamAwsProvider
from PIL import Image, ImageFile

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Decompression-bomb guard: cap decoded pixels well below a memory-exhaustion
# threshold. A crafted small file can otherwise claim huge dimensions.
Image.MAX_IMAGE_PIXELS = 40_000_000  # ~40 MP

# Required by the `optimize=True` JPEG saves below.
#
# With optimize (or progressive) Pillow encodes the whole scan into ONE buffer
# rather than streaming tiles, and raises "broken data stream when writing image
# file" if that buffer exceeds MAXBLOCK — 64 KB by default. A 720px 4:4:4
# thumbnail is several times that, so raising the thumbnail size turned this into
# a real failure: two of nineteen local images produced no thumbnail at all, and
# the exception was swallowed by make_thumbnail's except-return-None. Pillow's own
# documentation prescribes exactly this for the case.
#
# 32 MB comfortably covers both a THUMB_PX thumbnail and a PDF_PAGE_PX page, and
# is a buffer ceiling rather than a steady allocation.
ImageFile.MAXBLOCK = 32 * 1024 * 1024

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".m4v"}
# .weba is the audio-only WebM extension. It exists here because ".webm" is
# deliberately VIDEO above, and MediaRecorder's Chrome default is
# audio/webm;codecs=opus — uploaded as .webm a voice note would be classified as
# a VIDEO and render in the video bubble. The recorder names its Opus fallback
# .weba so the two can never collide.
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".aac", ".opus", ".weba"}
DOC_EXTS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".txt",
    ".csv",
    ".zip",
    ".md",
    ".json",
    ".rtf",
}

# One ceiling for every kind of file. There is no per-category limit any more and
# no allow-list of extensions: any format may be sent.
#
# Sourced from settings so RXHIVE_MAX_UPLOAD_BYTES actually does something: this
# was a hardcoded literal while config.py carried a `max_upload_bytes` nothing
# read, so an operator who set that variable — or read it to learn the limit —
# got a silent 2 GB instead of the 100 MB the setting advertised. The default
# lives in config.py and is unchanged at 2 GB; see the rationale there.
MAX_UPLOAD_BYTES = get_settings().max_upload_bytes

# Above this we skip preview generation rather than attempt it.
#
# Thumbnailing is the one step that genuinely needs the bytes in memory: Pillow
# and pypdfium both take a buffer. Without a bound, someone uploading a 2 GB TIFF
# named .jpg would have the API read all of it into RAM to make a 720px preview
# and take the process down. Past the bound the file still uploads and still
# sends; it just arrives without a preview image.
THUMBNAIL_SOURCE_LIMIT = 64 * 1024 * 1024


def too_large_to_render(file_size: int | None) -> bool:
    """Would rendering a preview for this hold more than an UPLOAD may buffer?

    The upload path declines to keep a preview source in memory above
    THUMBNAIL_SOURCE_LIMIT. Batch 54 applied the same ceiling to the two request
    paths that render on demand. This lives here, in the module that owns the
    limit, because the maintenance tools in app/tools need the identical answer and
    importing api/media.py into a CLI script would be the wrong direction — and a
    second copy of the comparison is how the tools came to disagree with the
    request paths in the first place.

    Answered from a stored size so no caller has to fetch an object to find out.
    """
    return (file_size or 0) > THUMBNAIL_SOURCE_LIMIT


MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".m4v": "video/x-m4v",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".opus": "audio/opus",
    ".weba": "audio/webm",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".zip": "application/zip",
    ".md": "text/markdown",
    ".json": "application/json",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".rtf": "application/rtf",
}


def classify(ext: str) -> str:
    """Which bubble renders this file. Never rejects — every extension maps.

    The four sets are no longer an allow-list, only a rendering hint: they decide
    whether something arrives as a photo, a video player, a voice note or a
    document card. Anything unrecognised is a document, which is the generic
    file card, so a .psd or a .dwg sends fine and simply shows as a file.

    Returning a str rather than str | None is the whole change. Callers used to
    treat None as "reject this upload"; there is nothing left to reject.
    """
    ext = ext.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return "document"


def _endpoint_parts(url: str) -> tuple[str, bool]:
    """Split an endpoint into (host[:port], secure), with the host LOWERCASED.

    The lowercasing is load-bearing, not tidiness. SigV4 signs the `Host` header
    verbatim, and **browsers always send the host lowercased** — while curl, and this
    process, send whatever case was configured. So a mixed-case endpoint (an mDNS
    name like `Carins-MacBook-Air.local`, or any hostname someone typed with capitals)
    produces presigned URLs that work from every tool you would reach for to test
    them, and fail in the one place that matters:

        curl  "…Carins-MacBook-Air.local:9000/…?X-Amz-Signature=…"  -> 200
        <img src=same-url>                                          -> 403

    MinIO answers `SignatureDoesNotMatch`, which reads as a credentials or clock
    problem and sends you looking at the keys. The visible symptom is simply that
    every image in the app is a broken-image icon.

    Hostnames are case-insensitive for resolution, so normalising here costs nothing
    and makes what we sign match what a browser will ask for. Only the host is
    touched — the bucket and key are case-SENSITIVE and never pass through here.
    """
    parts = urlsplit(url)
    secure = parts.scheme == "https"
    netloc = parts.netloc or parts.path
    return netloc.lower(), secure


@lru_cache
def _client() -> Minio:
    """S3 client. With no static keys configured, fall back to the AWS instance
    role (EC2/ECS metadata) — that is the deployed path, so no long-lived
    credentials ever sit on the box. MinIO locally still uses explicit keys."""
    settings = get_settings()
    host, secure = _endpoint_parts(settings.s3_endpoint)
    if settings.s3_access_key and settings.s3_secret_key:
        return Minio(
            host,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=secure,
            region=settings.s3_region,
        )
    return Minio(
        host,
        credentials=IamAwsProvider(region=settings.s3_region),
        secure=secure,
        region=settings.s3_region,
    )


def ensure_bucket() -> None:
    settings = get_settings()
    client = _client()
    if not client.bucket_exists(settings.s3_bucket):
        client.make_bucket(settings.s3_bucket)


async def put_object(key: str, data: bytes, content_type: str) -> None:
    settings = get_settings()

    def _put() -> None:
        _client().put_object(settings.s3_bucket, key, io.BytesIO(data), len(data), content_type=content_type)

    await anyio.to_thread.run_sync(_put)


async def put_stream(key: str, fileobj, length: int, content_type: str) -> None:
    """Upload from a file object without reading it into memory.

    put_object above takes `bytes`, which is fine for a thumbnail and fatal for a
    2 GB video: it would mean the whole file resident in the API process. This
    takes the SpooledTemporaryFile that Starlette already streamed the request
    body into, so the bytes go disk -> socket and the process holds only
    part_size at a time.

    part_size is the multipart chunk minio buffers per part. 64 MB over the 5 MB
    minimum keeps the part count for a 2 GB file at 32 rather than 400, which
    matters because each part is a separate round trip.
    """
    settings = get_settings()

    def _put() -> None:
        fileobj.seek(0)
        _client().put_object(
            settings.s3_bucket,
            key,
            fileobj,
            length,
            content_type=content_type,
            part_size=64 * 1024 * 1024,
        )

    await anyio.to_thread.run_sync(_put)


# Long edge for the in-bubble image preview.
#
# Sized from what it is DRAWN into, not picked round: ImageBubble renders a
# single image at up to 330x300 CSS px, which is 660x600 device px on a 2x
# display. The old 200px thumbnail was upscaled more than threefold there, which
# is why photos and any text inside them looked soft. 720 covers the 2x case with
# headroom and still costs a fraction of the original.
#
# Image.thumbnail() only ever shrinks, so a smaller original is left at its own
# size rather than being blown up.
THUMB_PX = 720
# 85 over 80, with DEFAULT (4:2:0) chroma.
#
# 4:4:4 was measured against 4:2:0 on real attachments before choosing: on a
# 3024x2268 camera photo it cost 449 KB against 244 KB, and on a screenshot 36 KB
# against 33 KB. Photographic content gains nothing perceptually from full chroma,
# so that is a doubling for nothing on exactly the largest files. Text-heavy
# images — screenshots — are small either way, and their sharpness comes from the
# resolution increase rather than the chroma. The PDF path keeps 4:4:4, where the
# content is entirely type and the cost is demonstrably low.
THUMB_QUALITY = 85

# Concurrent image decodes, bounded separately from anyio's shared pool.
#
# MAX_IMAGE_PIXELS bounds ONE thumbnail; nothing bounded how many run at once,
# and this is the memory-dominant step in the whole upload path. Measured with a
# source just under the 40 MP cap (6300x6300), which is the worst case the guard
# admits: one thumbnail grows RSS by about 290 MB, eight concurrently peaked at
# 3.0 GB. anyio's default limiter allows 40, and API_WORKERS is 2, so the box
# would admit 80 of them — on an m7i-flex.large with 8 GiB.
#
# The trigger costs nothing to send. 6300x6300 encodes to 70 KB as WebP, far
# under THUMBNAIL_SOURCE_LIMIT's 64 MB, because that limit bounds the ENCODED
# bytes and the cost here is in decoded pixels. So the cheap half of the guard
# chain is the half an attacker controls.
#
# 4 puts a worker's worst case near 1.2 GB and the box's near 2.4 GB, which
# leaves room for the rest of the process. Its own limiter rather than the
# default one for the reason batch 32 established: sharing the default 40 tokens
# would let this starve every attachment put, presign and push send.
#
# Distinct from _PDF_LIMITER below, which is 1 for a different reason entirely —
# PDFium is not thread-safe. This one is arithmetic, not correctness.
_IMAGE_LIMITER = anyio.CapacityLimiter(4)


async def make_thumbnail(data: bytes) -> bytes | None:
    """Bubble-sized JPEG thumbnail, transparent pixels flattened onto the app bg."""

    def _thumb() -> bytes | None:
        try:
            img = Image.open(io.BytesIO(data))
            # Reject decompression bombs by declared dimensions before decoding.
            if img.size[0] * img.size[1] > Image.MAX_IMAGE_PIXELS:
                logger.info(
                    "thumbnail skipped: %sx%s exceeds the %s pixel cap",
                    img.size[0],
                    img.size[1],
                    Image.MAX_IMAGE_PIXELS,
                )
                return None
            # LANCZOS explicitly: Pillow's default for thumbnail() is BICUBIC,
            # which is visibly softer on the fine detail this is meant to keep.
            img.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
            if img.mode in ("RGBA", "P", "LA"):
                background = Image.new("RGB", img.size, (26, 26, 26))
                img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=THUMB_QUALITY, optimize=True)
            return buf.getvalue()
        except Exception:
            # Returning None is correct — a file that will not decode simply has
            # no preview, and that must not fail the upload. Doing it SILENTLY was
            # not: this module's own MAXBLOCK note records a real bug that hid
            # here, where raising the thumbnail size broke two of nineteen local
            # images and the only symptom was a missing picture. Same outcome,
            # now with a trail.
            logger.warning("thumbnail generation failed (%d source bytes)", len(data), exc_info=True)
            return None

    return await anyio.to_thread.run_sync(_thumb, limiter=_IMAGE_LIMITER)


# --- PDF rasterisation -------------------------------------------------------
#
# Renders PDF pages to JPEG server-side, which is what lets a document bubble
# show page 1 and the viewer show the whole file. Rendering on the CLIENT was
# rejected: a virtualised scrollback of ten 20MB PDFs would download 200MB just
# to draw ten thumbnails.

# Page-1 preview inside the bubble. DocumentBubble draws it at 280x150 CSS px —
# 560x300 device px at 2x — and crops with object-cover, so the old 400px long
# edge (about 283px wide for portrait A4) was being upscaled roughly twofold.
# 1000 puts a portrait page at ~707px wide, which covers the 2x case.
PDF_THUMB_PX = 1000
PDF_PAGE_PX = 1600  # long edge for full-viewer pages
# Higher than the photo path and with 4:4:4 chroma. A PDF is text by definition,
# and subsampled chroma is what makes small type look smeared rather than merely
# small.
PDF_QUALITY = 88
# Hard clamp. The page count comes out of a parser fed untrusted input, so a
# hostile file reporting a huge count would otherwise turn the count itself into
# an attack on both the DOM and the per-page endpoint.
PDF_MAX_PAGES = 2000
PDF_WINDOW = 10  # pages rendered per on-demand pass

# PDFium is NOT thread-safe and pypdfium2 ships no lock of its own. anyio's
# default thread limiter allows 40, so without this two renders could overlap.
# Same idiom as services/push.py's send limiter.
_PDF_LIMITER = anyio.CapacityLimiter(1)


def _render_pdf_page(page, long_px: int) -> bytes:
    w_pt, h_pt = page.get_size()
    # Scale is derived from the page box, so the OUTPUT raster is bounded by
    # construction however large the page claims to be. This is the
    # enormous-page guard — Pillow's MAX_IMAGE_PIXELS does not apply here,
    # because nothing is being decoded from an image file.
    scale = min(long_px / max(w_pt, 1.0), long_px / max(h_pt, 1.0))
    # draw_annots=False is deliberate for a clinical product: comments, stamps
    # and redaction overlays stay out of the rendered page.
    bitmap = page.render(scale=scale, draw_annots=False, may_draw_forms=False)
    img = bitmap.to_pil().convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=PDF_QUALITY, optimize=True, subsampling=0)
    return buf.getvalue()


async def make_pdf_preview(data: bytes) -> tuple[bytes | None, int | None]:
    """(JPEG of page 1, page count). (None, None) on any parse failure —
    encrypted, truncated and plain-garbage files all land there."""

    def _go() -> tuple[bytes | None, int | None]:
        pdf = page = None
        try:
            pdf = pdfium.PdfDocument(data)
            n = min(len(pdf), PDF_MAX_PAGES)
            page = pdf[0]
            return _render_pdf_page(page, PDF_THUMB_PX), n
        except Exception:
            return None, None
        finally:
            if page is not None:
                page.close()
            if pdf is not None:
                pdf.close()

    return await anyio.to_thread.run_sync(_go, limiter=_PDF_LIMITER)


async def render_pdf_window(data: bytes, start: int) -> dict[int, bytes]:
    """1-indexed pages start..start+PDF_WINDOW-1 as JPEGs; {} on failure.

    A window rather than one page at a time: opening the document dominates the
    cost, so rendering ten pages per open is far cheaper than ten opens.
    """

    def _go() -> dict[int, bytes]:
        out: dict[int, bytes] = {}
        pdf = None
        try:
            pdf = pdfium.PdfDocument(data)
            n = min(len(pdf), PDF_MAX_PAGES)
            for i in range(start, min(start + PDF_WINDOW, n + 1)):
                page = pdf[i - 1]
                try:
                    out[i] = _render_pdf_page(page, PDF_PAGE_PX)
                finally:
                    page.close()
            return out
        except Exception:
            # Return whatever rendered before the failure rather than nothing.
            return out
        finally:
            if pdf is not None:
                pdf.close()

    return await anyio.to_thread.run_sync(_go, limiter=_PDF_LIMITER)


async def get_object(key: str) -> bytes:
    """Fetch an object's bytes. Used to re-read a PDF for page rendering."""

    def _get() -> bytes:
        resp = _client().get_object(get_settings().s3_bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    return await anyio.to_thread.run_sync(_get)


async def object_exists(key: str) -> bool:
    def _stat() -> bool:
        try:
            _client().stat_object(get_settings().s3_bucket, key)
            return True
        except Exception:
            return False

    return await anyio.to_thread.run_sync(_stat)


async def presign_get(key: str, *, filename: str | None = None, inline: bool = True) -> str:
    """Short-lived presigned GET URL, rewritten to the public endpoint the
    browser can reach (the bundled Caddy /s3 route, or a real S3/CDN origin)."""
    settings = get_settings()

    def _presign() -> str:
        headers = {}
        disposition = "inline" if inline else "attachment"
        if filename:
            safe = filename.replace('"', "").replace("\r", "").replace("\n", "")
            headers["response-content-disposition"] = f'{disposition}; filename="{safe}"'
        return _client().presigned_get_object(
            settings.s3_bucket,
            key,
            expires=dt.timedelta(seconds=settings.presign_expiry_seconds),
            response_headers=headers or None,
        )

    url = await anyio.to_thread.run_sync(_presign)
    return rewrite_to_public(url, settings.s3_public_endpoint)


def rewrite_to_public(url: str, public_endpoint: str) -> str:
    """Point a presigned URL at the endpoint the browser can actually reach.

    Rewrites by PARSING the URL rather than string-replacing the configured
    endpoint: against real AWS, minio signs virtual-host style
    (https://<bucket>.s3.<region>.amazonaws.com/<key>), which never contains the
    configured "https://s3.<region>.amazonaws.com" prefix — a prefix replace
    silently no-ops and leaks a cross-origin URL to the browser.

    A path-style public endpoint ("/s3") keeps media same-origin, which matters:
    CSP is enforced against redirect targets, so a cross-origin redirect out of
    /api/media is blocked by `img-src 'self'`. The signature stays valid because
    only the host is swapped — the proxy must forward the original Host.
    """
    public = (public_endpoint or "").rstrip("/")
    if not public:
        return url
    parts = urlsplit(url)
    query = f"?{parts.query}" if parts.query else ""
    if public.startswith(("http://", "https://")):
        pub = urlsplit(public)
        # Host lowercased for the same reason as in `_endpoint_parts`: a browser will
        # lowercase it before sending, and if that differs from what was signed the
        # signature fails. Matters most here, where a mixed-case *public* endpoint
        # would break the URL even when the signing endpoint was clean.
        netloc = pub.netloc.lower()
        if (pub.scheme, netloc) == (parts.scheme, parts.netloc):
            return url
        return f"{pub.scheme}://{netloc}{pub.path.rstrip('/')}{parts.path}{query}"
    # Same-origin path prefix (e.g. "/s3"), resolved by the browser.
    return f"{public}{parts.path}{query}"


# An extension is whatever followed the last dot in a name the client chose, so it
# is arbitrary text of arbitrary length. os.path.splitext is path-aware and never
# yields a "/", so it cannot escape the org scope — that much was already safe.
# What it could do was make the key unusable: a filename with a 500-character
# extension produced a 546-character object name, which minio refuses with
# XMinioInvalidObjectName, and the S3Error came straight back out of the upload
# endpoint as a 500. Measured before this: a 100-character extension uploaded
# fine, a 500-character one raised.
#
# 16 is well past every real extension (".jpeg", ".webm", ".xlsx") and leaves the
# key comfortably inside both the store's limit and uploads.storage_key's
# String(500).
_MAX_EXT = 16
_SAFE_EXT = re.compile(rf"^[a-z0-9]{{1,{_MAX_EXT}}}$")


def safe_ext(ext: str) -> str:
    """A client extension reduced to something that can be part of an object name.

    Anything else is dropped rather than mangled: the extension in the key is a
    readability convenience, and the authoritative record of what the user sent is
    uploads.filename. MIME_BY_EXT already decides the content type from the ORIGINAL
    extension, so dropping it here cannot change how a file is served.
    """
    candidate = (ext or "").lower().lstrip(".")
    # fullmatch, not match: "$" also matches just BEFORE a trailing newline, so
    # `match` accepted ".png\n" and put a control character in the object key. The
    # anchors in the pattern are kept anyway rather than relying on fullmatch alone,
    # so the intent survives someone changing one or the other.
    return f".{candidate}" if _SAFE_EXT.fullmatch(candidate) else ""


def new_storage_key(org_id, ext: str) -> str:
    scope = str(org_id) if org_id else "platform"
    return f"{scope}/{uuid.uuid4()}{safe_ext(ext)}"
