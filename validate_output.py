#!/usr/bin/env python3
"""Validate extracted_papers.json against the expected schema.

Usage: python validate_output.py [extracted_papers.json]
Exits with code 1 if any record has problems.
"""
import json
import re
import sys
from datetime import datetime

STR_FIELDS = ["arxiv_id", "title", "publication_date", "arxiv_pdf_url", "arxiv_page_url", "abstract"]
LIST_FIELDS = ["authors", "tasks", "methods", "models", "datasets", "sota_claims"]
CLAIM_KEYS = {"benchmark", "metric", "value", "comparison_target"}


def check(rec: dict) -> list[str]:
    errs = []
    for f in STR_FIELDS:
        if not isinstance(rec.get(f), str) or not rec[f].strip():
            errs.append(f"{f}: missing or empty")
    for f in LIST_FIELDS:
        if not isinstance(rec.get(f), list):
            errs.append(f"{f}: must be a list")
        elif not rec[f]:
            errs.append(f"{f}: empty")
    if not re.fullmatch(r"\d{4}\.\d{4,5}", str(rec.get("arxiv_id", ""))):
        errs.append("arxiv_id: bad format")
    try:
        datetime.strptime(str(rec.get("publication_date")), "%Y-%m-%d")
    except ValueError:
        errs.append("publication_date: must be YYYY-MM-DD")
    url = rec.get("github_url")
    if url is not None and not re.fullmatch(r"https://github\.com/[\w.\-]+/[\w.\-]+", url):
        errs.append("github_url: not a clean repo URL")
    if not isinstance(rec.get("is_official_code"), bool):
        errs.append("is_official_code: must be true/false")
    for i, c in enumerate(rec.get("sota_claims") or []):
        if not isinstance(c, dict) or set(c) != CLAIM_KEYS:
            errs.append(f"sota_claims[{i}]: keys must be {sorted(CLAIM_KEYS)}")
    return errs


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "extracted_papers.json"
    data = json.load(open(path, encoding="utf-8"))
    bad = 0
    for rec in data:
        errs = check(rec)
        print(("FAIL " if errs else "OK   ") + rec.get("arxiv_id", "?"), *errs, sep="\n   - " if errs else "")
        bad += bool(errs)
    print(f"\n{len(data) - bad}/{len(data)} records valid")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
