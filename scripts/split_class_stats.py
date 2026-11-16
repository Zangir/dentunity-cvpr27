"""Per-split, per-concept counts from the released manifests (CPU only, seconds).

Reads release/manifests/{io_cls,io_det,px}.csv (content-hash keyed; no file names) and
writes results/split_class_stats.json, which scripts/make_appendix_tables.py renders.
  python scripts/split_class_stats.py
"""
import csv, json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dentunity.taxonomy import UNIFIED  # noqa: E402

M = ROOT / "release" / "manifests"
NAME = {c["id"]: c["name"] for c in UNIFIED}
SPLITS = ("train", "val", "test")


def rows(name):
    with open(M / name, newline="") as f:
        return list(csv.DictReader(f))


def ids(s):
    return [int(t) for t in s.split()] if s.strip() else []


def size_stats(rs):
    w = [int(r["width"]) for r in rs]
    h = [int(r["height"]) for r in rs]
    return {"width_min": min(w), "width_median": statistics.median(w), "width_max": max(w),
            "height_min": min(h), "height_median": statistics.median(h), "height_max": max(h)}


def task_block(rs, col, count_instances):
    out = {"images": Counter(), "lineages": defaultdict(set), "per_concept": defaultdict(Counter),
           "per_concept_images": defaultdict(Counter), "annotations": Counter()}
    for r in rs:
        s = r["split"]
        out["images"][s] += 1
        out["lineages"][s].add(r["lineage"])
        cl = ids(r[col])
        for c in (cl if count_instances else set(cl)):
            out["per_concept"][NAME[c]][s] += 1
        for c in set(cl):
            out["per_concept_images"][NAME[c]][s] += 1
        out["annotations"][s] += len(cl)
    return {"images": {s: out["images"][s] for s in SPLITS},
            "lineages": {s: len(out["lineages"][s]) for s in SPLITS},
            "annotations": {s: out["annotations"][s] for s in SPLITS},
            "annotations_total": sum(out["annotations"][s] for s in SPLITS),
            "lineages_total": sum(len(out["lineages"][s]) for s in SPLITS),
            "per_concept": {k: {s: v[s] for s in SPLITS} for k, v in out["per_concept"].items()},
            "per_concept_images": {k: {s: v[s] for s in SPLITS} for k, v in out["per_concept_images"].items()},
            "image_size": size_stats(rs),
            "annotations_per_image": per_image(rs, col)}


def per_image(rs, col):
    k = sorted(len(ids(r[col])) for r in rs)
    q = lambda f: k[min(len(k) - 1, int(f * len(k)))]
    return {"min": k[0], "median": statistics.median(k), "p90": q(0.9), "max": k[-1],
            "mean": sum(k) / len(k), "empty": sum(v == 0 for v in k)}


def main():
    cls = rows("io_cls.csv")
    det = rows("io_det.csv")
    px = rows("px.csv")
    can = [r for r in px if r["role"] == "canonical"]
    seg = [r for r in can if r["in_seg"] == "1"]
    rex = [r for r in px if r["role"] == "re-export"]
    out = {"provenance": "run (scripts/split_class_stats.py over release/manifests/*.csv)",
           "io_cls": task_block(cls, "unified_labels", False),
           "io_det": task_block(det, "unified_classes", True),
           "px_det": task_block(can, "unified_classes", True),
           "px_seg_images": {s: sum(r["split"] == s for r in seg) for s in SPLITS},
           "px_reexports_by_split": {s: sum(r["split"] == s for r in rex) for s in SPLITS},
           "px_reexport_image_size": size_stats(rex)}
    (ROOT / "results" / "split_class_stats.json").write_text(json.dumps(out, indent=1))
    print("images", {k: sum(out[k]["images"].values()) for k in ("io_cls", "io_det", "px_det")})


if __name__ == "__main__":
    main()
