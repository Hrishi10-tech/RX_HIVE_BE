"""Link previews: which URLs the server will fetch, and what it reads from a page.

Pure functions only (services/link_preview_parse.py) — no network, DB or Redis.
The address guard that refuses private hosts is push.py's, covered by
test_push_ssrf_guard.py.
"""

from app.services.link_preview_parse import cache_key, normalize_url, parse_preview


def test_normalize_accepts_plain_web_urls_and_drops_the_fragment():
    assert normalize_url("https://Example.com/a?b=1#top") == "https://example.com/a?b=1"
    assert normalize_url("http://example.com") == "http://example.com/"
    assert normalize_url("https://example.com:443/x") == "https://example.com:443/x"


def test_normalize_refuses_what_the_server_must_not_fetch():
    for bad in [
        "",
        "ftp://example.com/file",
        "javascript:alert(1)",
        "file:///etc/passwd",
        "https://user:pass@example.com/",  # embedded credentials
        "https://example.com:8080/",  # non-standard port
        "https://example.com:6379/",
        "https:///nohost",
        "https://" + "a" * 2100 + ".com/",
        None,
    ]:
        assert normalize_url(bad) is None, bad


def test_cache_key_is_stable_hex():
    k = cache_key("https://example.com/")
    assert k == cache_key("https://example.com/")
    assert len(k) == 64 and all(c in "0123456789abcdef" for c in k)


def test_open_graph_wins_and_relative_images_are_resolved():
    html = """<html><head>
      <title>Fallback title</title>
      <meta property="og:title" content="The &amp; Real Title">
      <meta property="og:description" content="  A   description  ">
      <meta property="og:site_name" content="Example">
      <meta property="og:image" content="/img/card.png">
    </head><body><meta property="og:title" content="ignored, in body"></body></html>"""
    p = parse_preview(html, "https://example.com/post/1")
    assert p == {
        "title": "The & Real Title",
        "description": "A description",
        "site_name": "Example",
        "image_url": "https://example.com/img/card.png",
    }


def test_falls_back_to_title_tag_and_host_name():
    p = parse_preview("<head><title> Just a page </title></head>", "https://news.example.org/x")
    assert p["title"] == "Just a page"
    assert p["site_name"] == "news.example.org"
    assert p["description"] is None and p["image_url"] is None


def test_no_title_means_no_preview():
    assert parse_preview("<html><head></head><body>hi</body></html>", "https://example.com/") is None


def test_image_on_a_bad_scheme_or_port_is_dropped():
    html = '<head><title>t</title><meta property="og:image" content="http://example.com:8080/x.png"></head>'
    assert parse_preview(html, "https://example.com/")["image_url"] is None


def test_long_fields_are_truncated():
    html = f'<head><meta property="og:title" content="{"x" * 500}"></head>'
    assert len(parse_preview(html, "https://example.com/")["title"]) == 200
