#!/usr/bin/env python3
"""verify_citations.py — Chapter 5, Computer Science Research Practicum.

LLMs fabricate citations (Walters & Wilder 2023: 55% of GPT-3.5 / 18% of GPT-4 citations
entirely fabricated). NEVER trust a model-generated reference. This script checks a title or
a query against the free Semantic Scholar Graph API (S2AG) so you can confirm a paper EXISTS
before it enters your bibliography. A hit here is necessary, not sufficient — still eyeball the
year/venue/authors against dblp or the publisher.

Usage:
    python verify_citations.py --query "Aurora reinforcement learning congestion control"
    python verify_citations.py --bib references.txt    # one title per line

No API key is required, but keyless requests share one crowded pool and often get HTTP 429
("Too Many Requests"). Request a personal key with the form at
https://www.semanticscholar.org/product/api (it arrives by email), then
    export S2_API_KEY=your-key
The script retries politely on 429. Exit status: 0 = every query matched, 1 = at least one
NO MATCH, 3 = at least one lookup failed (verify those by hand on https://dblp.org).
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

S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,year,venue,authors,externalIds"
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


def s2_get(url):
    """GET a Semantic Scholar URL, sending S2_API_KEY if set, backing off on 429."""
    headers = {"User-Agent": "practicum-citation-check/1.0"}
    key = os.environ.get("S2_API_KEY")
    if key:
        headers["x-api-key"] = key
    for wait in (*BACKOFF, None):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            if wait is None:
                raise RateLimited() from exc
            time.sleep(_retry_after(exc.headers.get("Retry-After")) or wait)


def search(query, limit=3):
    url = f"{S2_SEARCH}?query={urllib.parse.quote(query)}&limit={limit}&fields={FIELDS}"
    try:
        return s2_get(url).get("data") or []
    except RateLimited:
        print("  ! Semantic Scholar rate-limited this lookup (HTTP 429, after 3 retries); "
              "verify it by hand on https://dblp.org, or set S2_API_KEY.", file=sys.stderr)
        return None
    except Exception as exc:  # network error, other HTTP error, etc.
        print(f"  ! lookup failed ({exc}). Verify manually in dblp/Semantic Scholar.", file=sys.stderr)
        return None


def report(query, hits):
    print(f"\nQUERY: {query}")
    if hits is None:
        print("  UNVERIFIED — could not reach the index. Do NOT cite until checked by hand.")
        return
    if not hits:
        print("  NO MATCH — treat as FABRICATED until proven otherwise. Do NOT cite.")
        return
    for h in hits:
        authors = ", ".join(a.get("name", "?") for a in (h.get("authors") or [])[:3])
        ids = h.get("externalIds") or {}
        ref = ids.get("DOI") or ids.get("ArXiv") or ids.get("DBLP") or "no-external-id"
        print(f"  FOUND: {h.get('title')}")
        print(f"         {authors} — {h.get('venue') or '?'} ({h.get('year') or '?'}) [{ref}]")
    print("  -> Eyeball year/venue/authors. A title match is NOT proof your metadata is right.")


def main():
    ap = argparse.ArgumentParser(description="Verify a citation exists before you cite it.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--query", help="a single title or search string")
    g.add_argument("--bib", help="a text file with one title per line")
    args = ap.parse_args()

    queries = [args.query] if args.query else [
        ln.strip() for ln in open(args.bib, encoding="utf-8") if ln.strip()
    ]
    failed = unmatched = 0
    for i, q in enumerate(queries):
        if i:
            time.sleep(1)  # be kind to the free API (a key's starting limit is 1 request/second)
        hits = search(q)
        report(q, hits)
        failed += hits is None
        unmatched += hits == []
    if failed:
        print(f"\n{failed}/{len(queries)} lookups FAILED. Verify those by hand on https://dblp.org.")
        return 3
    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
