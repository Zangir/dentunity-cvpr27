"""Shared paths, hashing and union-find. All raw data are read in place.

Data roots are configurable through environment variables so the kit runs on any
machine that has downloaded the upstream releases (see release/BUILD.md).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DATA = REPO / "data"            # gitignored work area (symlink trees, caches)
RELEASE = REPO / "release"

_MR = Path(os.environ.get('DENTUNITY_ROOT', '.'))  # set the data roots below via env vars
ORAL_ROOT = Path(os.environ.get(
    "UNIDENTAL_ORAL_ROOT",
    _MR / "ideas/DentalMethod/Dataset_04_11_Intraoral_Expert"))
PANO_INVENTORY = Path(os.environ.get(
    "UNIDENTAL_PANO_INVENTORY",
    _MR / "projects/dental-datasets-detection/experiments/repo/private/complete_inventory.json"))
PANO_PAIRS = PANO_INVENTORY.parent / "geometry_audit" / "accepted_pairs.json"
# cached frozen features of the intraoral composite (read only)
FEAT_ROOT = Path(os.environ.get(
    "UNIDENTAL_FEAT_ROOT", _MR / "projects/dental-method/experiments/repo"))

IODET_DIR = ORAL_ROOT / ("Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/"
                         "Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/Data")


def sha256_file(p, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_seed(text: str) -> int:
    """Process-independent seed (never use Python hash())."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[max(ra, rb)] = min(ra, rb)
        return True

    def labels(self):
        roots = [self.find(i) for i in range(len(self.p))]
        idx = {r: k for k, r in enumerate(sorted(set(roots)))}
        return [idx[r] for r in roots]


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


def read_json(path):
    return json.loads(Path(path).read_text())
