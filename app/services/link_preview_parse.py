"""Pure helpers for link previews: URL checks and Open Graph parsing.

Standard library only, on purpose — no network, no settings, no Redis — so the
rules that decide what the server will fetch and what it shows can be tested in
isolation. The fetching lives in services/link_preview.py.
"""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

MAX_URL_LENGTH = 2048
TITLE_MAX = 200
DESCRIPTION_MAX = 300
SITE_NAME_MAX = 80
# Only the standard web ports. A preview is a convenience, and refusing
# arbitrary ports stops the fetcher being used to knock on odd services of an
# otherwise public host (the address guard already refuses private hosts).
ALLOWED_PORTS = {None, 80, 443}


def normalize_url(raw: str) -> str | None:
    """The URL the server is willing to fetch, or None.

    http/https only, a hostname, no embedded credentials, a standard port, and a
    sane length. The fragment is dropped (never sent to a server, and it would
    otherwise split the cache for one page).
    """
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw or len(raw) > MAX_URL_LENGTH:
        return None
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https"):
        return None
    if not parts.hostname or parts.username or parts.password:
        return None
    if port not in ALLOWED_PORTS:
        return None
    netloc = parts.hostname.lower() + (f":{port}" if port else "")
    return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))


def cache_key(url: str) -> str:
    """Stable 64-hex key for a normalized URL (cache and image ids)."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _clean(value: str | None, limit: int) -> str | None:
    if not value:
        return None
    text = " ".join(value.split())
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class _HeadParser(HTMLParser):
    """Collects <title> and the meta tags a preview needs, then stops caring.

    Everything after </head> (or the first <body>) is ignored: Open Graph tags
    live in the head, and not parsing the body keeps a huge page cheap.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self._in_title = False
        self.done = False

    def handle_starttag(self, tag, attrs):
        if self.done:
            return
        if tag == "body":
            self.done = True
            return
        if tag == "title":
            self._in_title = True
            return
        if tag != "meta":
            return
        a = {k.lower(): (v or "") for k, v in attrs if k}
        key = (a.get("property") or a.get("name") or "").strip().lower()
        content = a.get("content")
        if key and content is not None and key not in self.meta:
            self.meta[key] = content

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "head":
            self.done = True

    def handle_data(self, data):
        if self._in_title and not self.done:
            self.title_parts.append(data)


def parse_preview(html: str, page_url: str) -> dict | None:
    """Title, description, site name and absolute image URL from a page's head.

    Returns None when the page offers nothing worth a card (no title at all).
    """
    parser = _HeadParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup: use whatever was collected
        pass
    m = parser.meta

    title = _clean(
        m.get("og:title") or m.get("twitter:title") or "".join(parser.title_parts),
        TITLE_MAX,
    )
    if not title:
        return None
    description = _clean(
        m.get("og:description") or m.get("twitter:description") or m.get("description"),
        DESCRIPTION_MAX,
    )
    site_name = _clean(m.get("og:site_name"), SITE_NAME_MAX) or urlsplit(page_url).hostname

    image = None
    raw_image = (m.get("og:image:secure_url") or m.get("og:image") or m.get("og:image:url")
                 or m.get("twitter:image") or m.get("twitter:image:src"))
    if raw_image:
        image = normalize_url(urljoin(page_url, raw_image.strip()))

    return {
        "title": title,
        "description": description,
        "site_name": site_name,
        "image_url": image,
    }
