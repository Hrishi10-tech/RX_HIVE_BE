"""Small shared helpers: wire-format serialization, sanitization, slugs."""

import datetime as dt
import re
import secrets
import string
import uuid
from bisect import bisect_right


def _opens_tag(text: str, lt: int) -> bool:
    """Whether the `<` at `lt` opens an HTML tag.

    A browser's tokenizer treats `<` as markup only before an ASCII letter, `/`
    and an ASCII letter, or `!`/`?`. Everything else — `< 50`, `<=`, `<3`,
    `<120` — is text, which is the whole point: `<` and `>` are ordinary
    characters in clinical chat and must not be read as markup.
    """
    nxt = text[lt + 1 : lt + 2]
    if nxt in ("!", "?"):
        return True
    if nxt == "/":
        nxt = text[lt + 2 : lt + 3]
    return nxt.isascii() and nxt.isalpha()


# Element names `sanitize_text` treats as markup: every element in the WHATWG HTML
# standard, the obsolete ones browsers still parse, and the roots of the two foreign
# vocabularies (`svg`, `math`), whose children are inert outside them. Matched
# case-insensitively against the WHOLE tag-name token.
#
# A `<word ...>` whose word is not on this list is not HTML, and in clinical chat it
# usually is not markup either: "ALT <ULN, AST >3x ULN", "Plt <LLN, recheck if >2
# days", "Jane Doe <jane.doe@mercy.org>", "<https://portal/lab/123>". Treating every
# `<` + letter as a tag deleted from there to the next `>` — "ALT <ULN, AST >3x ULN"
# was stored as "ALT 3x ULN", which says the opposite of what was typed.
_HTML_ELEMENTS = frozenset(
    """
    a abbr acronym address applet area article aside audio b base basefont bdi bdo
    bgsound big blink blockquote body br button canvas caption center cite code col
    colgroup data datalist dd del details dfn dialog dir div dl dt em embed fieldset
    figcaption figure font footer form frame frameset h1 h2 h3 h4 h5 h6 head header
    hgroup hr html i iframe image img input ins isindex kbd keygen label legend li
    link listing main map mark marquee math menu menuitem meta meter multicol nav
    nextid nobr noembed noframes noscript object ol optgroup option output p param
    picture plaintext pre progress q rb rp rt rtc ruby s samp script search section
    select slot small source spacer span strike strong style sub summary sup svg
    table tbody td template textarea tfoot th thead time title tr track tt u ul var
    video wbr xmp
    """.split()
)

# An attribute that can run script whatever element carries it: an event handler,
# or a script/data URL as a value. `<x onclick=...>` is an unknown element to a
# browser and still fires the handler, so a tag with one of these is markup even
# though its name is not on the list above.
#
# The lookahead lists words that start with "on" and are clinical or English, not
# handlers — "onset = 2h ago", "ondansetron=4mg", "once=daily". No browser compiles
# them on any element, so excluding them can only keep text, never let a handler
# through. Never add a real handler name to it.
#
# Written to be linear. The script-URL half was `=\s*["']?\s*...`, whose two
# adjacent `\s*` could split a whitespace run every way: one `=` followed by 64k
# spaces cost 28s, and any `<` anywhere in a message was enough to reach it.
_ACTIVE_ATTRIBUTE = re.compile(
    r"""(?:^|[\s/"'])(?!on(?:sets?|ce|es?|ly|going|wards?|call|c|cology|cologist|dansetron)\s*=)"""
    r"""on[a-z]+\s*="""
    r"""|=\s*(?:["']\s*)?(?:javascript|vbscript|data)\s*:""",
    re.IGNORECASE,
)

# A quoted attribute value. A browser reads a `>` inside one as content and keeps
# going, so a tag that opens one may not end at the first `>` this scan stops at:
# in `<x a=">" onclick=alert(1)>` the handler is inside the tag. Such a span is
# never kept whole. It is stripped through the first `>`, as before batch 72; the
# leftover starts mid-value, with no `<` of its own, so it is inert text.
_QUOTED_VALUE = re.compile(r"""=\s*["']""")

# Text-level elements that do nothing at all without an attribute value. Named with
# words after it and no `=` anywhere — "<a week ago, fever >", "<MAP 65 or >",
# "<HR 55, call if HR >" — a span is shorthand, not markup: with no `=` there is no
# href, no style and no handler, and with no quoted value a browser ends the tag at
# the same `>` this scan does. Bare "<b>" and "<em>" are still markup, and nothing
# that can load, run or embed anything is on this list.
_INERT_ELEMENTS = frozenset(
    """
    a abbr b bdi bdo big br cite code data dd del dfn dl dt em font h1 h2 h3 h4 h5 h6
    hr i ins kbd label li map mark meter nobr ol output p pre q rb rp rt rtc ruby s
    samp small strike strong sub summary sup table tbody td tfoot th thead time tr tt
    u ul var wbr
    """.split()
)

# The tokenizer's whitespace. NBSP and VT do not end a tag name.
_HTML_SPACE = " \t\n\r\f"

_TAG_NAME = re.compile(r"[A-Za-z][A-Za-z0-9-]*")


def iso_z(value: dt.datetime | None) -> str | None:
    """ISO-8601 with trailing Z — the wire datetime format the frontend expects."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def sanitize_text(text: str | None) -> str:
    """Strip HTML markup from user text (message content, names).

    Defence in depth, not an HTML sanitizer: nothing in the product renders this
    text as HTML — the web client has no `dangerouslySetInnerHTML` and iOS draws
    native labels — so the job is to keep markup out of stored text without
    damaging text that merely contains `<` or `>`.

    This was `re.sub(r"<[^>]+>", "", text)`, which is neither. It deleted the
    span between any `<` and any later `>`, so "hold if HR < 50 and SBP > 90"
    was stored as "hold if HR  90" and "x <= 5 and y >= 3" as "x = 3" — the
    second states a different threshold rather than losing one, and the sender
    was never told. It was also quadratic: `[^>]` let every `<` in a run rescan
    to end-of-input, and 200k of them in one body cost 12.4s of CPU. This
    function is synchronous inside async handlers and `content` has no length
    cap, so that was the event loop, stalled by any authenticated caller.

    Hand-rolled rather than a regex because no regex is all three of linear,
    precise, and non-destructive here. `[^>]*` for the tag body is the accurate
    match — a quoted attribute and a `<!...>` bogus comment may both contain
    `<` — but trying it at every `<` is what makes it quadratic. Narrowing the
    body to `[^<>]*` is linear and silently stops matching those tags. Scanning
    once, and only looking for `>` after a `<` that actually opens a tag, is
    linear without giving anything up: 200k `<` now costs 0.008s.

    A `<` + letter is still not enough. "ALT <ULN, AST >3x ULN" opens a tag by the
    tokenizer's rule — `<` before a letter — and was stored as "ALT 3x ULN", which
    no longer says the ALT is below the upper limit of normal. No syntax rule tells
    that apart from `<b class>`; the element name does. So a span is markup only if
    it is a `<!...>` / `<?...>`, opens a quoted attribute value (`_QUOTED_VALUE`),
    names an HTML element (`_HTML_ELEMENTS` — except a text-level one used with
    words and no `=`, `_INERT_ELEMENTS`), or carries an attribute that can run
    script on any element (`_ACTIVE_ATTRIBUTE`). Any other `<word ...>` is text and
    is kept whole: in a product that renders nothing as HTML, deleting a
    clinician's words is the failure that actually happens.
    """
    if not text:
        return ""
    if "<" not in text:
        return text
    # Positions of every active attribute, found in one pass, so deciding whether a
    # span carries one is a bisect rather than a regex run per span.
    active = [m.start() for m in _ACTIVE_ATTRIBUTE.finditer(text)]
    out: list[str] = []
    kept = 0  # start of the run not yet copied to `out`
    scan = 0
    length = len(text)
    while scan < length:
        lt = text.find("<", scan)
        if lt < 0:
            break
        if not _opens_tag(text, lt):
            # Text, not markup. Costs O(1), which is what keeps this linear:
            # the old pattern read to end-of-input before reaching the same
            # conclusion, from every `<` in the string.
            scan = lt + 1
            continue
        gt = text.find(">", lt + 1)
        if gt < 0:
            break
        if not _is_markup(text, lt, gt, active):
            # A `<word ...>` that is not HTML — "<ULN, AST >", "<jane@x.org>" — kept
            # whole, `>` included, and scanning resumes after it. That is how a
            # browser reads it: one tag, an unknown element, with anything inside it
            # part of that tag rather than a tag of its own. Resuming inside instead
            # let a removal there splice the kept "<b" onto a later `>` and spell a
            # tag the input never held. Its active attributes were checked over the
            # whole span, so "<x <img onerror=...>" is still stripped above.
            scan = gt + 1
            continue
        # Deleting [lt, gt] would leave any `<` immediately before it touching
        # the text after, which can spell a tag the input never held: "<<b>b>"
        # would leave "<b>". Take that run with the tag. The `/` of a "</" goes
        # with it only when a `<` precedes it, so "a/<b>" keeps its slash.
        cut = lt
        probe = cut - 1
        if probe >= kept and text[probe] == "/":
            probe -= 1
        if probe >= kept and text[probe] == "<":
            cut = probe
            while cut > kept and text[cut - 1] == "<":
                cut -= 1
        out.append(text[kept:cut])
        kept = scan = gt + 1
    out.append(text[kept:])
    return "".join(out)


def _is_markup(text: str, lt: int, gt: int, active: list[int]) -> bool:
    """Whether the tag-shaped span `text[lt : gt + 1]` is HTML markup.

    It is if it is a comment, doctype or processing instruction (`<!`, `<?`); if it
    opens a quoted attribute value, after which a browser's tag may run past this
    `>`; if its tag name — the whole token, ended by whitespace, `/` or `>` — names
    an HTML element, unless that element is text-level and is followed by words
    with no `=`; or if it carries an attribute that runs script on any element.
    Anything else is a `<word ...>` that a browser would parse as an unknown, inert
    element and a clinician wrote as text. Every check is bounded by the span, and
    spans do not overlap, so the scan stays linear.
    """
    start = lt + 1
    if text[start] in "!?":
        return True
    if _QUOTED_VALUE.search(text, lt, gt):
        return True
    if text[start] == "/":
        start += 1
    name = _TAG_NAME.match(text, start, gt + 1)
    if name is not None:
        after = text[name.end()]
        tag = name.group().lower()
        if (after in _HTML_SPACE or after in "/>") and tag in _HTML_ELEMENTS:
            rest = text[name.end() : gt]
            if tag not in _INERT_ELEMENTS or "=" in rest or not rest.strip(_HTML_SPACE + "/"):
                return True
    # Any active attribute between this `<` and its `>`.
    i = bisect_right(active, lt)
    return i < len(active) and active[i] < gt


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def parse_uuid(value: str | None) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def generate_password(length: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%"
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(length))
        if re.search(r"[A-Za-z]", pw) and re.search(r"\d", pw):
            return pw


def generate_call_code() -> str:
    groups = ["".join(secrets.choice(string.ascii_lowercase) for _ in range(4)) for _ in range(3)]
    return "-".join(groups)
