"""sanitize_text runs over every message body and every name the product stores.

It was `<[^>]+>` — the span between any `<` and any later `>`, deleted. That is
not what a tag is, and `<` and `>` are ordinary characters in clinical chat, so
messages lost text nobody asked it to remove. Measured against the old pattern:

    "hold if HR < 50 and SBP > 90"   ->  "hold if HR  90"
    "x <= 5 and y >= 3"              ->  "x = 3"
    "BP <120>80"                     ->  "BP 80"
    "K+ <3.5 replace >4.0"           ->  "K+ 4.0"
    "give <5 units over >30 min"     ->  "give 30 min"

Silent in both directions: the sender is not told and the recipient receives the
mangled text as if it were typed. The second row is the one that matters most —
it does not lose a threshold, it states a different one.

The same pattern was also quadratic. `[^>]` let a failing start rescan to
end-of-input, so every `<` in a run paid O(n): 200k `<` characters in one message
body cost 12.4s. sanitize_text is synchronous and called straight from async
handlers, and `content` has no length cap, so that was the event loop stalled for
12 seconds by any authenticated caller. The same input now takes 0.008s.

Three properties are asserted here, because the fix trades against all three:
text with no markup in it must come back byte-identical, markup must still be
removed, and no output may contain markup — the last one because deleting a tag
can splice a leftover `<` onto what followed it and build one that was never in
the input ("<<b>b>" leaves "<b>").

Batch 72 narrowed "markup". Batch 58 treated every `<` + letter as a tag, which is
the browser tokenizer's rule, and so still deleted from there to the next `>`:

    "ALT <ULN, AST >3x ULN"                      ->  "ALT 3x ULN"
    "Plt <LLN, recheck if >2 days"               ->  "Plt 2 days"
    "Jane Doe <jane.doe@mercy.org> re: bed 12"   ->  "Jane Doe  re: bed 12"

The first now says the ALT is three times the upper limit of normal. No syntax rule
separates "<ULN AST >" from "<b class>"; the element name does. Markup is now an
HTML element by name, any tag carrying an attribute that runs script (`on...=`, a
script or data URL), or `<!...>` / `<?...>`. A `<word ...>` that is none of those —
which a browser would parse as an unknown, inert element — is text, and kept.
"""

import random
import re
import time

import pytest

from app.utils import sanitize_text

# Markup that must never survive, written independently of the implementation so it
# cannot agree with a mistake in it. A deliberately simple model of how a browser
# would read the OUTPUT: scanning from the left, a `<` before a letter, `/` + letter,
# `!` or `?` starts a tag, which runs to the first `>` that is not inside a quoted
# attribute value; everything else is text.
#
# A tag is markup if it is `<!...>` / `<?...>`; if its whole tag-name token (ended by
# whitespace, `/` or `>`) is one of these elements — a SUBSET of what the
# implementation strips, including every name the fuzz below can spell — unless it
# is a text-level one followed by words and no `=` at all; or if it carries an
# event handler or a script URL where the tokenizer starts an attribute: after
# whitespace, `/` or a closing quote.
_ORACLE_ELEMENTS = frozenset(
    "a b i p s u q br hr em script style img svg math iframe object embed div span".split()
    + "form input link meta base".split()
)
_ORACLE_TEXT_LEVEL = frozenset("a b i p s u q br hr em".split())
_ORACLE_ACTIVE = re.compile(
    r"""(?<=[\s/"'])on[a-z]+\s*=|=\s*['"]?\s*(?:javascript|vbscript|data)\s*:""", re.IGNORECASE
)
_ORACLE_QUOTE = re.compile(r"""=[ \t\n\r\f]*(["'])""")


def _markup_in(output: str) -> str | None:
    """The first tag in `output` that a browser would act on, or None."""
    i = 0
    while True:
        lt = output.find("<", i)
        if lt < 0 or lt + 1 >= len(output):
            return None
        nxt = output[lt + 1]
        opens = nxt in "!?" or (nxt.isascii() and nxt.isalpha())
        if nxt == "/" and lt + 2 < len(output):
            opens = output[lt + 2].isascii() and output[lt + 2].isalpha()
        if not opens:
            i = lt + 1
            continue
        gt = output.find(">", lt + 1)
        if gt < 0:
            return None
        # A `>` inside a quoted value is content: the tag goes on past it.
        j = lt + 1
        while (quote := _ORACLE_QUOTE.search(output, j, gt)) is not None:
            close = output.find(quote.group(1), quote.end())
            if close < 0:
                return None  # an unterminated value runs to the end; no tag completes
            j = close + 1
            if close > gt:
                gt = output.find(">", close + 1)
                if gt < 0:
                    return None
        tag = output[lt : gt + 1]
        head = re.match(r"</?([^\s/>]*)", tag)
        name = head.group(1).lower()
        rest = tag[head.end() : -1]
        text_level = name in _ORACLE_TEXT_LEVEL and rest.strip(" \t\n\r\f/") and "=" not in rest
        if nxt in "!?" or (name in _ORACLE_ELEMENTS and not text_level) or _ORACLE_ACTIVE.search(tag):
            return tag
        i = gt + 1


@pytest.mark.parametrize(
    "text",
    [
        "hold if HR < 50 and SBP > 90",
        "x <= 5 and y >= 3",
        "BP <120>80",
        "K+ <3.5 replace >4.0",
        "give <5 units over >30 min",
        "sat <90 on 2L, target >94",
        "temp <38.5>37 range",
        "INR <2 target >3",
        "a < b > c",
        "5 < 10",
        "see <3",
        "wean if FiO2 <40 and PEEP >5",
        # Batch 72: a `<` + letter that does not name an HTML element is text.
        # Lab shorthand, a contact's address, a pasted link — each was cut from
        # its `<` to the next `>`, and the first two read as different results.
        "ALT <ULN, AST >3x ULN",
        "ALT <ULN AST >3x ULN",
        "Plt <LLN, recheck if >2 days",
        "K <LLN -> replace",
        "Na <LLN and K >ULN",
        "dose <q6h> prn",
        "Forwarding from Jane Doe <jane.doe@mercy.org> re: bed 12",
        "results here <https://portal.mercy.org/lab/123> thanks",
        "on call: <Cardiology> fellow",
        "trend <Hb 7 -> 6.4> over 24h",
        # A text-level element name used as shorthand, with words and no `=`.
        "Call MD for <MAP 65 or >MAP 110",
        "hold metoprolol if <HR 55, call if HR >120",
        "Sx onset <a week ago, fever >38.5 x2",
        "UOP <a liter today, net I/O >2L positive",
        "Cr <b/l, UOP >0.5 mL/kg/h",
        "K <pre HD, recheck >post HD",
        # Clinical words that start with "on" are not event handlers.
        "Trop <ULN, onset = 2h ago, repeat if CP >30 min",
        "Plt <LLN, onset=day 5 of heparin, HIT score >4",
        "Plt <LLN, onsets=day 5, HIT >4",
        "K <LLN, ondansetron=4mg given, Cr >1.5",
        "Na <LLN, once=daily dosing ok, Cr >2",
        # A tag name must start with an ASCII letter, so these are text too.
        # Greek letters are ordinary notation in a clinical thread.
        "\u03b1 <\u03b2> \u03b3 blockade",
        "titrate <\u00e9lan> per protocol",
        "\u0434\u043e\u0437\u0430 <\u0432\u044b\u0441\u043e\u043a\u0430\u044f> today",
    ],
)
def test_text_without_a_tag_is_returned_verbatim(text: str):
    """No `<` here opens a tag, so nothing in any of these is markup."""
    assert sanitize_text(text) == text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Inner text is kept and the tags go — the contract the messaging and
        # cross-org-group tests already rely on.
        ("<script>alert(1)</script>hello", "alert(1)hello"),
        ("Launch <b>Team</b>", "Launch Team"),
        ("<b></b>", ""),
        ("<img src=x onerror=alert(1)>", ""),
        ("<svg/onload=alert(1)>", ""),
        ("<A HREF='x'>y</A>", "y"),
        ("<!-- comment -->", ""),
        ("<?php echo 1; ?>", ""),
        ("<div\nclass='x'>y</div>", "y"),
        # An unknown element still runs a handler if it is ever rendered, so an
        # active attribute makes any tag markup, whatever its name.
        ("<x onclick=alert(1)>hi", "hi"),
        ("<foo onmouseover='steal()'>y</foo>", "y</foo>"),
        ("<x href=javascript:alert(1)>y", "y"),
        ("<x src = 'data:text/html,<b>'>y", "'>y"),
        # Markup next to clinical text: only the markup goes.
        ("ALT <ULN, AST >3x ULN; see <b>trend</b>", "ALT <ULN, AST >3x ULN; see trend"),
        ("<SCRIPT>x</SCRIPT> K <LLN", "x K <LLN"),
        # A real tag hidden behind a stray `<word` is still caught: active
        # attributes are checked across the whole span a browser would read.
        ("a <x <img src=x onerror=alert(1)> b", "a  b"),
        ("a<x it's <b onclick=alert(1)>c", "ac"),
        # A quoted value can hide the tag's real end: the browser reads on past
        # the `>` inside it, so this span is never kept whole. It is cut at the
        # first `>` instead, and the leftover starts mid-value, with no `<`.
        ('<x foo=">" onclick=alert(1)>hi', '" onclick=alert(1)>hi'),
        ("<x title='>' onclick=alert(1)>hi", "' onclick=alert(1)>hi"),
        ("<a 1='>' href=https://evil>click", "' href=https://evil>click"),
        ("<b 1='>' onclick=alert(1)>x", "' onclick=alert(1)>x"),
        # An excluded "on" word cannot shield a real handler or script URL.
        ("<x onset=1 onerror=alert(1)>y", "y"),
        ("<x onset=javascript:alert(1)>y", "y"),
        ("<x onsetx=1>y", "y"),
        ("<x onend=alert(1)>y", "y"),
        # A text-level element with any `=` is markup: that is an attribute value,
        # which is what an href, a style or a handler needs.
        ("<a href=https://evil>y</a>", "y"),
        ("<b class=x>y</b>", "y"),
        # Script-capable elements are markup whatever words follow them.
        ("<script blue>x", "x"),
        ("<style 2>x", "x"),
    ],
)
def test_markup_is_stripped(raw: str, expected: str):
    assert sanitize_text(raw) == expected


@pytest.mark.parametrize(
    "raw", ["<<b>b>", "<<<b>b>b>", "a1</<?>a>b", "?=</<!1 >b>", "a<<b>c>", '<b<!">>=x->n<"a']
)
def test_a_removal_cannot_splice_a_new_tag(raw: str):
    """The reason for the fallback branch: these inputs leave a tag behind when
    tags are deleted in one left-to-right pass. The last one is batch 72's: with
    `<b` kept as text, removing the `<!">` inside it once spliced it onto the next
    `>` as "<b>"; a non-markup span is now kept whole, as a browser reads it."""
    assert _markup_in(sanitize_text(raw)) is None


def test_no_output_contains_a_tag():
    """Fuzz over the alphabet that can build one. Fixed seed so a failure is a
    reproducible case rather than a flake."""
    rng = random.Random(13)
    # Quotes, `-` and `=` are in here deliberately: they are what builds a
    # quoted attribute and a comment, the shapes where a `>` appears inside
    # what a browser would still be reading as one tag. `o`, `n` and `x` spell
    # handlers ("onx=") and unknown names ("x", "ax") that must not hide one.
    alphabet = "<>/!?-=abxon1 \"'"
    for _ in range(40000):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 20)))
        cleaned = sanitize_text(raw)
        assert _markup_in(cleaned) is None, f"{raw!r} -> {cleaned!r}"


@pytest.mark.parametrize("payload", ["<", "<a", "</", "<!", "<x ", "<uln, ", "<x onclick=1 "])
def test_unclosed_angle_brackets_do_not_stall_the_event_loop(payload: str):
    """The DoS half of the finding. Each of these repeated is a start that can
    never complete a tag; under `[^>]` each one rescanned to end-of-input.

    The budget is deliberately loose. Measured on the old pattern: 6.2-12.4s.
    Measured on the current one: under 0.01s. Anything between is a regression
    in the shape of the scan, and a CI runner two orders of magnitude slower
    than this laptop still passes.
    """
    text = payload * (200_000 // len(payload))
    started = time.perf_counter()
    sanitize_text(text)
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0, f"{payload!r} * n took {elapsed:.1f}s — the scan is quadratic again"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('<a title=">">x</a>', '">x'),
        ("<a title='>'>x</a>", "'>x"),
        ('<a title="><script>">x', '">x'),
        ("<!-- a > b -->", " b -->"),
    ],
)
def test_a_quoted_or_comment_contained_gt_ends_the_tag_early(raw: str, expected: str):
    """A known limitation, asserted so it stays known.

    The scan takes the first `>` after a tag opens. A browser would not: inside
    a quoted attribute value, or between `<!--` and `-->`, a `>` is content and
    the tag runs on. So markup of that shape is cut in the wrong place and part
    of it survives as text.

    Deliberately not fixed. It is exactly what `<[^>]+>` did before this change
    — the outputs above are byte-identical to the old pattern's, on every case
    in this table — so the behaviour is unchanged rather than introduced, and
    closing it means hand-rolling quote and comment states, i.e. an HTML
    tokenizer, for text that nothing in the product renders as HTML.

    What has to hold is the property below, and it does: the leftover is inert
    text, never a tag. The fuzz above covers this shape, and the oracle is not
    blind to it — `_markup_in` reads a tag to its first `>`, so it would flag a
    surviving `<a title=">">` readily. The oracle erring toward calling something a tag is
    the safe direction for an oracle.
    """
    assert sanitize_text(raw) == expected
    assert _markup_in(sanitize_text(raw)) is None


@pytest.mark.parametrize("falsy", [None, ""])
def test_falsy_input_becomes_empty_string(falsy):
    assert sanitize_text(falsy) == ""


@pytest.mark.parametrize("payload", ["<x ", "<uln, ", "<a1 "])
def test_a_run_of_text_openers_before_one_gt_stays_linear(payload: str):
    """Batch 72 made `<word ...>` that is not HTML text. Deciding that per opener by
    rescanning its span for an active attribute would be quadratic on exactly this
    shape — a long run of openers with a single `>` at the end — so the active
    attributes are found once, up front, and looked up.
    """
    text = payload * (200_000 // len(payload)) + ">"
    started = time.perf_counter()
    assert sanitize_text(text) == text
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0, f"{payload!r} * n + '>' took {elapsed:.1f}s — the scan is quadratic again"


# Pieces that build the attribute shapes a browser reads differently from a naive
# scan: quoted values holding `>`, handlers straight after a closing quote, an
# apostrophe in prose, and text-level names used as shorthand. The character
# alphabet above almost never spells these.
_TOKENS = [
    "<",
    ">",
    "</",
    "<x",
    "<b",
    "<a",
    "<x ",
    "<b ",
    "<script",
    "<!",
    "<?",
    "/",
    " ",
    ' a=">"',
    " a='>'",
    '"',
    "'",
    "=",
    " onx=1",
    " onclick=1",
    "onx=1",
    "it's",
    " href=javascript:1",
    " week ago",
    "b",
    "x",
    "1",
    "<img",
    " src=x",
    " onerror=1",
]


def test_no_output_contains_a_tag_built_from_attribute_pieces():
    rng = random.Random(72)
    for _ in range(30000):
        raw = "".join(rng.choice(_TOKENS) for _ in range(rng.randint(1, 10)))
        cleaned = sanitize_text(raw)
        assert _markup_in(cleaned) is None, f"{raw!r} -> {cleaned!r}"


@pytest.mark.parametrize(
    "text",
    [
        "5 < 10 =" + " " * 40_000 + "x",
        "a<=" + "\t" * 40_000 + "x",
        "a<=" + "\n" * 40_000 + "x",
        "<=" + " " * 20_000 + "'" + " " * 20_000 + "x",
        '<x a=">' * 20_000,
        "<x on" + " " * 40_000 + "x>",
    ],
)
def test_whitespace_after_an_attribute_sign_stays_linear(text: str):
    """The script-URL half of the active-attribute pattern had two adjacent `\\s*`
    around an optional quote, which can split a whitespace run every way: one `=`
    followed by 64k spaces cost 28s, and a single `<` anywhere let a message reach
    it. Caught by review before it shipped."""
    started = time.perf_counter()
    sanitize_text(text)
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0, f"{len(text)} chars took {elapsed:.1f}s — the pattern backtracks again"
