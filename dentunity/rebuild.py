"""Build kit: rebuild the benchmark from the user's own upstream downloads.

We release no images and no upstream annotation files. The user downloads the two
upstream releases (release/BUILD.md), then:

  python -m dentunity.rebuild --pano-root <Dental-Disease-Detection> \
      --oral-root <Dataset_04_11_Intraoral_Expert> --out dentunity_v1

Every file is matched to the sha256 manifests in release/manifests/ (never by name);
images are linked under hashed names; labels are re-derived from the upstream label
files and mapped to task-local ids (unified ids are in the manifests). Writes YOLO trees
for IO-DET / PX-DET / PX-SEG, a CSV for IO-CLS, and verify.json (found / missing counts).
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

from .common import RELEASE, sha256_file, write_json
from .taxonomy import IO_CLS_CLASSES, IO_DET_CLASSES, PX_CLASSES, to_unified

IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def index(root):
    """sha256 -> (image path, sibling YOLO label path or None)."""
    out = {}
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(IMG):
                p = Path(r) / f
                lab = Path(str(p.parent).replace("/images", "/labels")) / (p.stem + ".txt")
                out.setdefault(sha256_file(p), (p, lab if lab.exists() else None))
    return out


def _link(dst: Path, src: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src.resolve())


def _yaml(root: Path, names):
    root.joinpath("data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pano-root", required=True, nargs="+")
    ap.add_argument("--oral-root", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-reexports", action="store_true",
                    help="also add re-exported copies of TRAINING lineages to PX-DET train")
    a = ap.parse_args()
    out = Path(a.out)
    rep = {"found": defaultdict(int), "missing": defaultdict(int)}
    pano, oral = {}, {}
    for r in a.pano_root:
        pano.update({k: v for k, v in index(r).items() if k not in pano})
    for r in a.oral_root:
        oral.update({k: v for k, v in index(r).items() if k not in oral})
    m = RELEASE / "manifests"
    # IO-CLS
    out.mkdir(parents=True, exist_ok=True)
    uc = to_unified("io_cls")
    inv = {u: i for i, u in enumerate(uc)}
    with open(m / "io_cls.csv") as f, open(out / "io_cls.csv", "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["image", "split", "lineage"] + IO_CLS_CLASSES)
        for r in csv.DictReader(f):
            hit = oral.get(r["sha256"])
            rep["found" if hit else "missing"]["io_cls"] += 1
            if hit:
                dst = out / "io_cls" / r["split"] / f"{r['sha256'][:16]}{hit[0].suffix.lower()}"
                _link(dst, hit[0])
                labs = {inv[int(u)] for u in r["unified_labels"].split()}
                w.writerow([str(dst.relative_to(out)), r["split"], r["lineage"]] + [int(c in labs) for c in range(6)])
    # IO-DET (boxes are re-derived from the upstream YOLO label of the same file)
    for r in csv.DictReader(open(m / "io_det.csv")):
        hit = oral.get(r["sha256"])
        rep["found" if hit and hit[1] else "missing"]["io_det"] += 1
        if hit and hit[1]:
            name = f"{r['sha256'][:16]}{hit[0].suffix.lower()}"
            _link(out / "io_det" / "images" / r["split"] / name, hit[0])
            lines = [l.strip() for l in open(hit[1]) if len(l.split()) == 5]
            (out / "io_det" / "labels" / r["split"]).mkdir(parents=True, exist_ok=True)
            (out / "io_det" / "labels" / r["split"] / f"{r['sha256'][:16]}.txt").write_text("\n".join(lines) + "\n")
    _yaml(out / "io_det", IO_DET_CLASSES)
    # PX-DET / PX-SEG
    for r in csv.DictReader(open(m / "px.csv")):
        use_det = r["role"] == "canonical" or (a.with_reexports and r["split"] == "train")
        if not use_det:
            continue
        hit = pano.get(r["sha256"])
        rep["found" if hit and hit[1] else "missing"]["px"] += 1
        if not (hit and hit[1]):
            continue
        name = f"{r['sha256'][:16]}_{r['role'][:3]}{hit[0].suffix.lower()}"
        polys = [l.split() for l in open(hit[1]) if len(l.split()) >= 7]
        boxes = []
        for t in polys:
            xs, ys = [float(v) for v in t[1::2]], [float(v) for v in t[2::2]]
            x0, x1, y0, y1 = max(min(xs), 0), min(max(xs), 1), max(min(ys), 0), min(max(ys), 1)
            boxes.append(f"{t[0]} {(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}")
        _link(out / "px_det" / "images" / r["split"] / name, hit[0])
        (out / "px_det" / "labels" / r["split"]).mkdir(parents=True, exist_ok=True)
        (out / "px_det" / "labels" / r["split"] / (Path(name).stem + ".txt")).write_text("\n".join(boxes) + "\n")
        if r["in_seg"] == "1":
            _link(out / "px_seg" / "images" / r["split"] / name, hit[0])
            (out / "px_seg" / "labels" / r["split"]).mkdir(parents=True, exist_ok=True)
            (out / "px_seg" / "labels" / r["split"] / (Path(name).stem + ".txt")).write_text(
                "\n".join(" ".join(t) for t in polys) + "\n")
    _yaml(out / "px_det", PX_CLASSES)
    _yaml(out / "px_seg", PX_CLASSES)
    rep = {k: dict(v) for k, v in rep.items()}
    write_json(out / "verify.json", rep)
    print(rep)


if __name__ == "__main__":
    main()
