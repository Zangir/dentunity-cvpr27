"""Counts (never prints) panoramic file names that contain alphabetic name-like tokens.
Heuristic: the export stem (before '_jpg.rf.' / '.rf.') split on non-letters; a stem is
flagged when it holds >= 1 purely alphabetic token of >= 2 letters other than file-type
words. Also counts stems with a date pattern. Writes results/identifier_audit.json."""
from __future__ import annotations

import re
from pathlib import Path

from .common import DATA, RESULTS, read_json, write_json

SKIP = {"jpg", "jpeg", "png", "rf", "copy", "img", "image", "dcm", "bmp"}
DATE = re.compile(r"(19|20)\d\d[-_.]?\d\d[-_.]?\d\d|\d{1,2}[-_.]\d{1,2}[-_.](19|20)\d\d")


def main():
    px = read_json(DATA / "inv_px.json")
    flagged, dated = 0, 0
    fam = {}
    for r in px:
        stem = Path(r["path"]).name.split(".rf.")[0]
        toks = [t for t in re.split(r"[^A-Za-z]+", stem) if len(t) >= 2 and t.lower() not in SKIP]
        f = bool(toks)
        flagged += f
        dated += f and bool(DATE.search(stem))
        fam.setdefault(r["source_key"], f)
    write_json(RESULTS / "identifier_audit.json", {
        "provenance": "run", "files": len(px), "files_with_name_like_tokens": flagged,
        "files_with_name_like_tokens_and_date": dated,
        "share_files_with_name_like_tokens": flagged / len(px),
        "source_keys": len(fam), "source_keys_with_name_like_tokens": sum(fam.values())})
    print(read_json(RESULTS / "identifier_audit.json"))


if __name__ == "__main__":
    main()
