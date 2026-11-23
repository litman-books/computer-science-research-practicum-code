#!/usr/bin/env python3
"""citation_audit.py — turn your .bib into a per-reference verification ledger.

Reason this exists: LLMs fabricate citations. Walters & Wilder 2023 (Scientific
Reports) found ~55% of GPT-3.5 and ~18% of GPT-4 citations entirely fabricated;
among the real ones, 43% / 24% had substantive metadata errors. Before you
submit, EVERY reference must be verified against a real index (dblp, Semantic
Scholar, or the publisher). This script does not verify for you — it builds the
checklist so no reference slips through unverified.

Usage:
    python citation_audit.py references.bib
Writes citation_audit.csv: one row per @entry with columns to fill BY HAND
after you look each one up. Stdlib only; no install, no network, no AI.
"""
import csv
import re
import sys


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


def main(path):
    with open(path, encoding="utf-8") as fh:
        entries = parse_bib(fh.read())
    if not entries:
        print("No BibTeX entries found. Is the file an exported .bib?")
        return 1
    out = "citation_audit.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["cite_key", "type", "title", "year",
             "verified_in (dblp/S2/publisher)", "doi_confirmed (y/n)",
             "metadata_ok (y/n)", "ai_assisted_find (y/n)", "notes"]
        )
        for etype, key, f in entries:
            w.writerow(
                [key, etype, f.get("title", "")[:80], f.get("year", ""),
                 "", "", "", "", ""]
            )
    print(f"{len(entries)} references -> {out}")
    print("Now open it and verify EVERY row by hand. An unverified row is a")
    print("liability: a hallucinated citation can get a paper desk-rejected")
    print("without review (and arXiv can impose a one-year ban). You own every")
    print("reference, not the model.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
