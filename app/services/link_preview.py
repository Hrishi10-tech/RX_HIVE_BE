"""Link previews: fetch a page's Open Graph data and a small preview image.

Security model — this module makes the SERVER fetch URLs that users paste:

* Every connection goes through push.py's public-only adapter, which checks the
  address the socket ACTUALLY connected to (not a DNS pre-check that a
  rebinding host can answer differently), for every redirect hop, for http and
  https. Private, loopback, link-local, CGNAT and multicast are refused.
* No environment proxies (they would bypass that adapter).
* Standard ports only, no credentials in URLs (link_preview_parse.normalize_url).
* Hard caps on time, redirects and bytes read; only HTML is parsed and only
  image/* is thumbnailed, re-encoded to JPEG by Pillow before anyone sees it.
* The browser never loads the remote image. The server stores a re-encoded
  thumbnail under the hash of the image URL and serves it from our own origin,
  so the endpoint cannot be used as an open image proxy: it only serves images
  this module itself fetched while building a preview.

Results (including "no preview") are cached in Redis, so a link shared in a busy
group is fetched once, not once per viewer.
"""

from __future__ import annotations

import base64
import json
import logging
import threading

import anyio
import requests

from app.realtime.redis_bus import get_redis
from app.services import storage
from app.services.link_preview_parse import cache_key, normalize_url, parse_preview
from app.services.push import _PublicOnlyAdapter

logger = logging.getLogger(__name__)

META_TTL_SECONDS = 24 * 3600
MISS_TTL_SECONDS = 3600
IMAGE_TTL_SECONDS = 24 * 3600
HTML_MAX_BYTES = 768 * 1024
IMAGE_MAX_BYTES = 5 * 1024 * 1024
CONNECT_TIMEOUT = 3
READ_TIMEOUT = 5
OVERALL_DEADLINE_SECONDS = 12
MAX_REDIRECTS = 3
USER_AGENT = "Mozilla/5.0 (compatible; RxHiveLinkPreview/1.0)"

# At most this many fetches in flight per worker: each holds a thread.
_fetch_limiter = anyio.CapacityLimiter(8)
_sessions = threading.local()


def _session() -> requests.Session:
    session = getattr(_sessions, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False  # proxies would route around the address guard
        session.max_redirects = MAX_REDIRECTS
        session.mount("https://", _PublicOnlyAdapter())
        session.mount("http://", _PublicOnlyAdapter())
        session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en"})
        _sessions.session = session
    return session


def _read_capped(resp: requests.Response, cap: int, stop_marker: bytes | None = None) -> bytes:
    buf = bytearray()
    for chunk in resp.iter_content(chunk_size=16384):
        buf.extend(chunk)
        if len(buf) >= cap:
            break
        # Open Graph lives in <head>; no need to download the rest of the page.
        if stop_marker and stop_marker in buf.lower():
            break
    return bytes(buf[:cap])


def _fetch_page(url: str) -> tuple[str, str, str] | None:
    """(final_url, content_type, text-or-empty) or None on any failure."""
    try:
        with _session().get(
            url,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            headers={"Accept": "text/html,application/xhtml+xml;q=0.9,image/*;q=0.8,*/*;q=0.5"},
        ) as resp:
            if resp.status_code >= 400:
                return None
            final_url = normalize_url(resp.url)
            if final_url is None:
                return None
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype not in ("text/html", "application/xhtml+xml"):
                return final_url, ctype, ""
            raw = _read_capped(resp, HTML_MAX_BYTES, stop_marker=b"</head>")
            encoding = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" else "utf-8"
            try:
                text = raw.decode(encoding, errors="replace")
            except LookupError:
                text = raw.decode("utf-8", errors="replace")
            return final_url, ctype, text
    except Exception as exc:  # network errors, blocked address, too many redirects
        logger.info("link preview fetch failed for %s: %s", url, type(exc).__name__)
        return None


def _fetch_image(url: str) -> bytes | None:
    try:
        with _session().get(
            url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), headers={"Accept": "image/*"}
        ) as resp:
            if resp.status_code >= 400:
                return None
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if not ctype.startswith("image/") or ctype == "image/svg+xml":
                return None
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > IMAGE_MAX_BYTES:
                return None
            data = _read_capped(resp, IMAGE_MAX_BYTES + 1)
            return None if len(data) > IMAGE_MAX_BYTES else data
    except Exception as exc:
        logger.info("link preview image fetch failed for %s: %s", url, type(exc).__name__)
        return None


async def _store_image(image_url: str) -> str | None:
    """Fetch, re-encode and cache the preview image; returns its key."""
    key = cache_key(image_url)
    redis = get_redis()
    if await redis.exists(f"lp:img:{key}"):
        return key
    data = await anyio.to_thread.run_sync(_fetch_image, image_url, limiter=_fetch_limiter, abandon_on_cancel=True)
    if not data:
        return None
    thumb = await storage.make_thumbnail(data)  # re-encoded JPEG, bomb-checked
    if not thumb:
        return None
    await redis.set(f"lp:img:{key}", base64.b64encode(thumb).decode("ascii"), ex=IMAGE_TTL_SECONDS)
    return key


async def get_preview(raw_url: str) -> dict | None:
    """The preview for a URL (cached), or None when there is nothing to show."""
    url = normalize_url(raw_url)
    if url is None:
        return None
    key = cache_key(url)
    redis = get_redis()
    cached = await redis.get(f"lp:meta:{key}")
    if cached is not None:
        return json.loads(cached) or None

    preview = None
    try:
        with anyio.fail_after(OVERALL_DEADLINE_SECONDS):
            page = await anyio.to_thread.run_sync(_fetch_page, url, limiter=_fetch_limiter, abandon_on_cancel=True)
            if page is not None:
                final_url, ctype, text = page
                if text:
                    parsed = parse_preview(text, final_url)
                    if parsed:
                        image_key = await _store_image(parsed["image_url"]) if parsed["image_url"] else None
                        preview = {
                            "url": url,
                            "title": parsed["title"],
                            "description": parsed["description"],
                            "site_name": parsed["site_name"],
                            "image": f"/api/link-preview/image/{image_key}" if image_key else None,
                        }
                elif ctype.startswith("image/") and ctype != "image/svg+xml":
                    # A direct link to a picture: the picture is the preview.
                    image_key = await _store_image(final_url)
                    if image_key:
                        preview = {
                            "url": url,
                            "title": final_url.rsplit("/", 1)[-1][:200] or final_url,
                            "description": None,
                            "site_name": None,
                            "image": f"/api/link-preview/image/{image_key}",
                        }
    except TimeoutError:
        logger.info("link preview timed out for %s", url)

    await redis.set(
        f"lp:meta:{key}",
        json.dumps(preview) if preview else "null",
        ex=META_TTL_SECONDS if preview else MISS_TTL_SECONDS,
    )
    return preview


async def get_cached_image(key: str) -> bytes | None:
    value = await get_redis().get(f"lp:img:{key}")
    if not value:
        return None
    try:
        return base64.b64decode(value)
    except ValueError:
        return None
