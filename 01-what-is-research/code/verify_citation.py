#!/usr/bin/env python3
"""verify_citation.py — sanity-check that a citation actually exists.

LLMs fabricate citations: Walters & Wilder (2023) found 55% of GPT-3.5 and 18%
of GPT-4 citations entirely fabricated. This script does NOT prove a citation is
real — only you, against the publisher, can do that. It queries the free
Semantic Scholar Graph API for a title and shows you the closest matches so you
can VERIFY by eye. No API key is required, but keyless requests share one
crowded pool and often get HTTP 429 ("Too Many Requests"). If that happens,
request a personal API key with the form at https://www.semanticscholar.org/product/api
(it arrives by email) and export it before you run:  export S2_API_KEY=your-key
The script retries politely on 429; if it still fails, check the title by hand
on https://dblp.org instead.

Usage:
    python3 verify_citation.py "Attention Is All You Need"
    python3 verify_citation.py --title "Title here" --year 2017

The human owns the verdict. A near-miss title with the wrong year or wrong
authors is exactly the kind of plausible-wrong record an LLM invents.
"""
import argparse
import email.utils
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,year,authors,venue,externalIds"
BACKOFF = (2, 4, 8)          # seconds to wait before each retry after an HTTP 429


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


def s2_get(url: str) -> dict:
    """GET a Semantic Scholar URL, sending S2_API_KEY if set, backing off on 429."""
    headers = {"User-Agent": "rp-verify/1.0"}
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


def search(title: str, limit: int = 5) -> list:
    q = urllib.parse.urlencode({"query": title, "limit": limit, "fields": FIELDS})
    return s2_get(f"{API}?{q}").get("data", []) or []


def main() -> int:
    p = argparse.ArgumentParser(description="Verify a citation against Semantic Scholar.")
    p.add_argument("title", nargs="?", help="paper title to search")
    p.add_argument("--title", dest="title_opt")
    p.add_argument("--year", type=int, help="expected year (flagged if it disagrees)")
    a = p.parse_args()
    title = a.title_opt or a.title
    if not title:
        p.error("provide a title to search for")

    try:
        hits = search(title)
    except RateLimited:
        print("  Semantic Scholar rate-limited this lookup (HTTP 429, after 3 retries).",
              file=sys.stderr)
        print("  -> verify this citation BY HAND on https://dblp.org (or the publisher),")
        print("     or set S2_API_KEY and try again later.")
        return 3
    except Exception as e:  # network error, other HTTP error, etc.
        print(f"  could not reach Semantic Scholar: {e}", file=sys.stderr)
        print("  -> verify this citation MANUALLY in dblp or the publisher.")
        return 3

    if not hits:
        print(f'  NO MATCH for "{title}".')
        print("  A no-match is a RED FLAG — this may be a hallucinated citation.")
        return 1

    print(f'  Top matches for "{title}" (verify by eye — do not trust blindly):\n')
    for h in hits:
        authors = ", ".join(au["name"] for au in (h.get("authors") or [])[:3]) or "?"
        ext = h.get("externalIds") or {}
        ident = ext.get("DOI") or ext.get("ArXiv") or "no DOI/arXiv id"
        warn = ""
        if a.year and h.get("year") and h["year"] != a.year:
            warn = f"  <-- YEAR MISMATCH: index says {h['year']}, you said {a.year}"
        print(f"  - {h.get('title')} ({h.get('year')})")
        print(f"      {authors}  | {h.get('venue') or 'venue?'}  | {ident}{warn}")
    print("\n  A human still owns the verdict on every reference in your bibliography.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
