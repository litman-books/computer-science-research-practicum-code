#!/usr/bin/env python3
"""
verify_citations.py — sanity-check that the references in your .bib actually exist.

LLMs hallucinate citations (Walters & Wilder 2023: ~55% of GPT-3.5 and ~18% of GPT-4
citations were entirely fabricated). Before you submit OR review, every reference must
be findable in a real index. This script queries the free Semantic Scholar Graph API
by title and flags references it cannot match — those are the ones YOU verify by hand
in dblp or the publisher. A flag is not proof of fabrication; it's a "go look."

Usage:
    python verify_citations.py refs.bib

No API key is required, but keyless requests share one crowded pool and often get
HTTP 429 ("Too Many Requests"). Request a personal key with the form at
https://www.semanticscholar.org/product/api (it arrives by email), then
    export S2_API_KEY=your-key
The script sleeps between calls and retries politely on 429. Re-check the S2AG
endpoint/limits before a big run; rate limits drift.
"""
import email.utils, os, re, sys, time, json, urllib.error, urllib.parse, urllib.request

S2 = "https://api.semanticscholar.org/graph/v1/paper/search"
BACKOFF = (2, 4, 8)          # seconds to wait before each retry after an HTTP 429


def _read_value(body, i):
    """Read one field value starting at body[i]; return (value, next_index).

    Handles {braced {nested} values}, "quoted values", and bare values such as
    year = 2020 (a number) or month = jan (a macro).
    """
    if body[i] == "{":
        depth, j = 0, i
        while j < len(body):
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
                if depth == 0:
                    return body[i + 1:j], j + 1
            j += 1
        return body[i + 1:], len(body)
    if body[i] == '"':
        depth, j = 0, i + 1
        while j < len(body):
            c = body[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == '"' and depth == 0 and body[j - 1] != "\\":
                return body[i + 1:j], j + 1
            j += 1
        return body[i + 1:], len(body)
    m = re.match(r"[^,}\s]+", body[i:])
    return (m.group(0), i + m.end()) if m else ("", i)


def _clean(value):
    """{BERT}: Pre-training of ... -> BERT: Pre-training of ... (one line)."""
    return " ".join(value.replace("{", "").replace("}", "").split())


def _fields(body):
    """Parse 'name = value, name = value' pairs; field names are matched whole,
    so `booktitle` is never mistaken for `title`."""
    fields, i = {}, 0
    name_re = re.compile(r"\s*,?\s*([A-Za-z][\w:-]*)\s*=\s*", re.S)
    while True:
        m = name_re.match(body, i)
        if not m or m.end() >= len(body):
            return fields
        value, i = _read_value(body, m.end())
        fields[m.group(1).lower()] = _clean(value)


def parse_bib(text):
    """Return [(entry_type, cite_key, {field: value}), ...] for every @entry{...}."""
    entries, pos = [], 0
    entry_re = re.compile(r"@(\w+)\s*\{")
    while (m := entry_re.search(text, pos)):
        body, pos = _read_value(text, m.end() - 1)   # the whole balanced {...}
        etype = m.group(1).lower()
        if etype in ("comment", "preamble", "string"):
            continue
        key, _, rest = body.partition(",")
        entries.append((etype, key.strip(), _fields(rest)))
    return entries


def titles_from_bib(path):
    """The `title` field of every entry (never `booktitle`), braces removed."""
    with open(path, encoding="utf-8") as fh:
        return [f["title"] for _, _, f in parse_bib(fh.read()) if f.get("title")]

class RateLimited(Exception):
    """Semantic Scholar kept answering 429 after every retry."""

def _retry_after(value):
    """Seconds from a Retry-After header (a number or an HTTP date), capped at 60."""
    if not value:
        return None
    try:
        secs = float(value)
    except ValueError:
        try:
            secs = email.utils.parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError):
            return None
    return min(max(secs, 0.0), 60.0)

def s2_get(url):
    """GET a Semantic Scholar URL, sending S2_API_KEY if set, backing off on 429."""
    headers = {"User-Agent": "practicum-cite-check"}
    key = os.environ.get("S2_API_KEY")
    if key:
        headers["x-api-key"] = key
    for wait in (*BACKOFF, None):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            if wait is None:
                raise RateLimited() from e
            time.sleep(_retry_after(e.headers.get("Retry-After")) or wait)

def lookup(title):
    q = urllib.parse.urlencode({"query": title, "limit": 1, "fields": "title,year,externalIds"})
    try:
        data = s2_get(f"{S2}?{q}")
    except RateLimited:
        return None, "Semantic Scholar rate-limited this lookup (HTTP 429, after 3 retries)"
    except Exception as e:
        return None, f"query failed: {e}"
    hits = data.get("data") or []
    if not hits:
        return None, "no match in Semantic Scholar"
    return hits[0], None

def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python verify_citations.py refs.bib")
    titles = titles_from_bib(sys.argv[1])
    if not titles:
        sys.exit("no titles found — is this a BibTeX file?")
    flagged = 0
    for t in titles:
        hit, err = lookup(t)
        if err:
            print(f"[FLAG] {t[:70]!r}: {err}  -> verify by hand in dblp")
            flagged += 1
        else:
            print(f"[ ok ] {t[:70]!r} ~ {hit.get('year')} ({' '.join((hit.get('title') or '').split())[:50]})")
        time.sleep(1.5)  # be a good API citizen
    print(f"\n{flagged}/{len(titles)} references need manual verification.")
    print("A flag means 'go look' — confirm or remove. Never ship a citation you can't open.")

if __name__ == "__main__":
    main()
