"""Independently verify every entry in refs.bib against Crossref and OpenAlex.

Nothing here trusts the .bib file or whoever wrote it. Each entry is looked up
in a public bibliographic database and the returned record is compared field by
field against what the .bib claims. Anything that does not match, or that cannot
be found at all, is reported.

    python verify_refs.py [--bib ../refs.bib] [--mail you@example.com]

Exit status is 1 if any entry FAILs, so it can be wired into CI.

Status meanings
  OK        DOI resolved and title, first author and year all agree
  CHECK     found, but at least one field disagrees -- read the notes
  NO-DOI    no DOI in the .bib; matched by title search instead
  UNVERIFIED no DOI and no database record matched the title. The entry may be
            wrong, or simply not indexed -- conference proceedings and books
            often are not. Check these by hand.

A NOT FOUND is not proof an entry is fabricated, and an OK is not proof it is
appropriate to cite -- only that the record exists and the fields line up.
"""

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

CROSSREF = "https://api.crossref.org/works"
OPENALEX = "https://api.openalex.org/works"


def norm(s):
    """Casefold, strip accents and punctuation, collapse whitespace."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def similarity(a, b):
    """Token overlap (Jaccard) between two normalised strings."""
    ta, tb = set(norm(a).split()), set(norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def parse_bib(path):
    """Minimal BibTeX reader: enough for a flat, brace-delimited .bib."""
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"(?m)^\s*%.*$", "", text)
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.S):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*(?:,|$)", body, re.S):
            v = re.sub(r"\s+", " ", fm.group(2)).strip()
            fields[fm.group(1).lower()] = re.sub(r"[{}\\]", "", v)
        entries.append({"kind": kind, "key": key, **fields})
    return entries


def get(url, mail, tries=3):
    req = urllib.request.Request(url, headers={"User-Agent": f"verify-refs (mailto:{mail})"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            if attempt == tries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def first_surname(bib_author):
    """First author's family name from a BibTeX author field."""
    if not bib_author:
        return ""
    first = bib_author.split(" and ")[0].strip()
    return (first.split(",")[0] if "," in first else first.split()[-1]).strip()


def crossref_by_doi(doi, mail):
    d = get(f"{CROSSREF}/{urllib.parse.quote(doi)}?mailto={mail}", mail)
    return d.get("message") if d else None


def crossref_by_title(title, mail):
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 3, "mailto": mail})
    d = get(f"{CROSSREF}?{q}", mail)
    items = ((d or {}).get("message") or {}).get("items") or []
    return items[0] if items else None


def openalex_by_title(title, mail):
    q = urllib.parse.urlencode(
        {"filter": f"title.search:{title}", "per-page": 3, "mailto": mail}
    )
    d = get(f"{OPENALEX}?{q}", mail)
    res = (d or {}).get("results") or []
    return res[0] if res else None


def compare(entry, rec, source):
    """Compare a .bib entry against a fetched record. Returns (ok, notes)."""
    notes = []

    if source == "crossref":
        rec_title = (rec.get("title") or [""])[0]
        rec_year = ((rec.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        rec_authors = [a.get("family", "") for a in rec.get("author", []) if a.get("family")]
        rec_venue = (rec.get("container-title") or [""])[0]
    else:
        rec_title = rec.get("title") or ""
        rec_year = rec.get("publication_year")
        rec_authors = [
            a["author"]["display_name"].split()[-1] for a in rec.get("authorships", [])
        ]
        rec_venue = (
            ((rec.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
        )

    sim = similarity(entry.get("title", ""), rec_title)
    if sim < 0.6:
        notes.append(f"title differs (overlap {sim:.0%}): record says {rec_title!r}")

    want = norm(first_surname(entry.get("author", "")))
    got = [norm(a) for a in rec_authors]
    # substring match so "Kaium" still matches a record family name of
    # "Abdul Kaium", which is a parsing difference and not a mismatch
    if want and got and not any(want in g or g in want for g in got):
        notes.append(f"first author {want!r} not among {got[:4]}")

    if entry.get("year") and rec_year and str(rec_year) != str(entry["year"]):
        notes.append(f"year {entry['year']} vs record {rec_year}")

    venue = entry.get("journal") or entry.get("booktitle") or ""
    if venue and rec_venue and similarity(venue, rec_venue) < 0.34:
        notes.append(f"venue {venue!r} vs record {rec_venue!r}")

    return (not notes), notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", default="../refs.bib")
    ap.add_argument("--mail", default="mhmdkhalil.h@gmail.com")
    args = ap.parse_args()

    entries = parse_bib(args.bib)
    print(f"checking {len(entries)} entries in {args.bib}\n")

    failures = 0
    for e in entries:
        key, doi = e["key"], e.get("doi")
        rec, source, status = None, None, None

        if doi:
            rec, source = crossref_by_doi(doi, args.mail), "crossref"
            if rec is None:
                rec, source = openalex_by_title(e.get("title", ""), args.mail), "openalex"
                status = "CHECK" if rec else "NOT FOUND"
                if rec:
                    print(f"  ! {key}: DOI {doi} did not resolve at Crossref")
        else:
            # OpenAlex first: it indexes proceedings and preprints that Crossref
            # does not. A title search always returns its best guess, so the
            # candidate is only accepted if the titles genuinely agree --
            # otherwise the entry is reported unverified rather than mismatched.
            rec, source = openalex_by_title(e.get("title", ""), args.mail), "openalex"
            if rec is not None and similarity(e.get("title", ""), rec.get("title") or "") < 0.6:
                rec = None
            if rec is None:
                cand = crossref_by_title(e.get("title", ""), args.mail)
                if cand and similarity(e.get("title", ""), (cand.get("title") or [""])[0]) >= 0.6:
                    rec, source = cand, "crossref"
            status = "NO-DOI" if rec else "UNVERIFIED"

        if rec is None:
            label = status or "NOT FOUND"
            print(f"[{label:9s}] {key:16s} {e.get('title','')[:58]}")
            print("              - no DOI, and no database record matched the title")
            failures += 1
            time.sleep(0.3)
            continue

        ok, notes = compare(e, rec, source)
        if status is None:
            status = "OK" if ok else "CHECK"
        elif status == "NO-DOI" and not ok:
            status = "CHECK"

        print(f"[{status:9s}] {key:16s} {e.get('title','')[:58]}")
        for n in notes:
            print(f"              - {n}")
        if status in ("CHECK", "NOT FOUND"):
            failures += 1
        time.sleep(0.3)

    print(f"\n{len(entries) - failures}/{len(entries)} clean; {failures} need attention")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
