"""Post-plan analysis A (external review request): types of label conflict among the exact-duplicate
groups of the IO-DET source, by DentUnity split of the kept copy. CPU only, counts only.

Each conflicting group (copies with byte-identical images but different box lists) gets ONE type,
checked in this order over all pairs of copies in the group:
  count  - the copies have different numbers of boxes
  class  - same number of boxes, different class multisets
  iou<0.5 - same classes; after optimal same-class matching (max IoU) some box has IoU < 0.5
  iou0.5-0.9 - every matched IoU >= 0.5 but the smallest is < 0.9
  iou>=0.9 - every matched IoU >= 0.9 (coordinate jitter only)
Non-exclusive counts (a group can show several kinds over its pairs) are reported as well.
Writes results/label_conflict_types.json and data/conflict_test_sha.json (sha256 list of the kept
test images in conflicting groups; hashes only).
  python3 scripts/label_conflict_types.py
"""
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

R = Path(__file__).resolve().parents[1]
inv = json.loads((R / "data" / "inv_io_det.json").read_text())
bench = json.loads((R / "data" / "bench.json").read_text())["io_det"]
split_of = {r["sha256"]: r["split"] for r in bench}


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def pair_kind(A, B):
    """A, B: lists of [cls, cx, cy, w, h]. Returns (kind, min matched IoU or None)."""
    if len(A) != len(B):
        return "count", None
    if sorted(int(a[0]) for a in A) != sorted(int(b[0]) for b in B):
        return "class", None
    if not A:
        return "same", None
    mins = []
    for c in {int(a[0]) for a in A}:
        a = [x[1:] for x in A if int(x[0]) == c]
        b = [x[1:] for x in B if int(x[0]) == c]
        M = np.array([[iou(p, q) for q in b] for p in a])
        r, k = linear_sum_assignment(-M)
        mins.append(M[r, k].min())
    m = float(min(mins))
    return ("iou<0.5" if m < 0.5 else "iou0.5-0.9" if m < 0.9 else "iou>=0.9"), m


key = lambda r: json.dumps(sorted((b[0], *(round(x, 4) for x in b[1:])) for b in r["boxes_cxcywh"]))
by = defaultdict(list)
for r in inv:
    by[r["sha256"]].append(r)
groups = {s: v for s, v in by.items() if len(v) > 1}
conflict = {s: v for s, v in groups.items() if len({key(r) for r in v}) > 1}
ORDER = ["count", "class", "iou<0.5", "iou0.5-0.9", "iou>=0.9"]
exclusive, nonexcl = {}, defaultdict(set)
min_iou = {}
for s, v in conflict.items():
    kinds = set()
    for a, b in combinations(v, 2):
        if key(a) == key(b):
            continue
        k, m = pair_kind(a["boxes_cxcywh"], b["boxes_cxcywh"])
        kinds.add(k)
        if m is not None:
            min_iou[s] = min(min_iou.get(s, 1.0), m)
    for k in kinds:
        nonexcl[k].add(s)
    exclusive[s] = next(k for k in ORDER if k in kinds)
splits = ["train", "val", "test"]
table = {k: {sp: sum(1 for s, t in exclusive.items() if t == k and split_of.get(s) == sp) for sp in splits}
         for k in ORDER}
for k in ORDER:
    table[k]["all"] = sum(table[k][sp] for sp in splits)
by_split_conf = Counter(split_of.get(s, "missing") for s in conflict)
by_split_all = Counter(split_of.get(s, "missing") for s in groups)
test_sha = sorted(s for s in conflict if split_of.get(s) == "test")
out = {"provenance": "run (scripts/label_conflict_types.py over data/inv_io_det.json + data/bench.json; "
                     "post-plan analysis A, external review request)",
       "io_det_files": len(inv), "exact_duplicate_groups": len(groups), "conflicting_groups": len(conflict),
       "duplicate_groups_by_split": {sp: by_split_all.get(sp, 0) for sp in splits},
       "conflicting_groups_by_split": {sp: by_split_conf.get(sp, 0) for sp in splits},
       "type_order": ORDER, "exclusive_type_by_split": table,
       "nonexclusive_type_counts": {k: len(nonexcl.get(k, ())) for k in ORDER},
       "copies_per_conflicting_group": dict(Counter(len(v) for v in conflict.values())),
       "kept_copy_rule": "max (number of boxes, path) per sha256 group (unidental/build.py)",
       "conflicting_test_images": len(test_sha)}
(R / "results" / "label_conflict_types.json").write_text(json.dumps(out, indent=1))
(R / "data" / "conflict_test_sha.json").write_text(json.dumps(test_sha))
print(json.dumps(out, indent=1))
