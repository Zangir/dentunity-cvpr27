"""Materialise YOLO training trees (symlinks under hashed names, never filenames).

  python -m dentunity.yolo_export   -> data/yolo/{pxdet,pxseg,iodet,pxdet_leaky_s<k>,pxdet_cleanctl_s<k>}/
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .common import DATA, read_json, stable_seed, write_json, RESULTS
from .taxonomy import IO_DET_CLASSES, PX_CLASSES

Y = DATA / "yolo"


def _link(root: Path, split: str, r: dict, label_lines: list[str], tag: str = ""):
    ext = Path(r["path"]).suffix.lower()
    import hashlib
    name = r["sha256"][:16] + "_" + hashlib.sha256(r["path"].encode()).hexdigest()[:6] + tag
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    dst = root / "images" / split / f"{name}{ext}"
    src = DATA / "px640" / f"{r['sha256']}.jpg"
    if not dst.exists():
        dst.symlink_to(src if src.exists() else r["path"])
    (root / "labels" / split / f"{name}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""))


def _yaml(root: Path, names, splits=("train", "val", "test")):
    lines = [f"path: {root}"] + [f"{s}: images/{s}" for s in splits]
    lines += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    (root / "data.yaml").write_text("\n".join(lines) + "\n")


def px_box_lines(r):
    out = []
    for x in r["instances"]:
        x0, y0, x1, y1 = [min(max(v, 0.0), 1.0) for v in x["box"]]
        out.append(f"{x['c']} {(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}")
    return out


def px_poly_lines(r):
    out = []
    for line in open(r["label_path"]):
        t = line.split()
        if len(t) >= 7:
            out.append(" ".join([t[0]] + [f"{min(max(float(v), 0.0), 1.0):.6f}" for v in t[1:]]))
    return out


def main():
    import sys
    only_io = "--only-io" in sys.argv
    b = read_json(DATA / "bench.json")
    px = [] if only_io else b["px"]
    can = [r for r in px if r["role"] == "canonical"]
    # D1 lineage-aware recipe: canonical train + re-exports of TRAIN lineages only
    for r in px:
        if r["split"] == "train":
            _link(Y / "pxdet_reexp", "train", r, px_box_lines(r))
    for r in can:
        if r["split"] in ("val", "test"):
            _link(Y / "pxdet_reexp", r["split"], r, px_box_lines(r))
    _yaml(Y / "pxdet_reexp", PX_CLASSES)
    for r in can:
        _link(Y / "pxdet", r["split"], r, px_box_lines(r))
        if r["in_seg"]:
            _link(Y / "pxseg", r["split"], r, px_poly_lines(r))
    _yaml(Y / "pxdet", PX_CLASSES)
    _yaml(Y / "pxseg", PX_CLASSES)
    for r in b["io_det"]:
        if r["split"] == "dropped":
            continue
        _link(Y / "iodet", r["split"], r,
              [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in r["boxes_cxcywh"]])
    _yaml(Y / "iodet", IO_DET_CLASSES)
    # L3: the shipped IO-DET split (train / val), exact duplicates removed as above
    for r in b["io_det"]:
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in r["boxes_cxcywh"]]
        sp = "train" if r["shipped_split"] == "train" else "val"
        _link(Y / "iodet_shipped", sp, r, lines)
        if sp == "val":
            _link(Y / "iodet_shipped", "test", r, lines)
    _yaml(Y / "iodet_shipped", IO_DET_CLASSES)
    if only_io:
        return
    # leak-inflation arms (equal-size training sets, same val/test as pxdet):
    # LEAKY swaps K random canonical training images for the K re-exported copies of
    # test images; CLEAN-CONTROL swaps the same K for re-exports of TRAIN images.
    rexp_test = [r for r in px if r["role"] == "re-export" and r["split"] == "test"]
    rexp_train = [r for r in px if r["role"] == "re-export" and r["split"] == "train"]
    train = [r for r in can if r["split"] == "train"]
    rng = np.random.default_rng(stable_seed("dentunity-leak"))
    K = min(len(rexp_test), len(rexp_train), len(train) // 2)
    drop = set(rng.choice(len(train), K, replace=False).tolist())
    keep = [r for i, r in enumerate(train) if i not in drop]
    add_t = [rexp_test[i] for i in rng.choice(len(rexp_test), K, replace=False)]
    add_c = [rexp_train[i] for i in rng.choice(len(rexp_train), K, replace=False)]
    for arm, add in [("pxdet_leaky", add_t), ("pxdet_cleanctl", add_c)]:
        root = Y / arm
        for r in keep + add:
            _link(root, "train", r, px_box_lines(r))
        for r in can:
            if r["split"] in ("val", "test"):
                _link(root, r["split"], r, px_box_lines(r))
        _yaml(root, PX_CLASSES)
    write_json(RESULTS / "leak_design_px.json", {
        "provenance": "run", "K_swapped": K, "train_size_each_arm": len(keep) + K,
        "reexports_of_test_available": len(rexp_test), "reexports_of_train_available": len(rexp_train),
        "test_lineages_with_copy_in_leaky_train": len({r["lineage"] for r in add_t}),
        "test_images": sum(r["split"] == "test" for r in can)})
    for d in sorted(Y.iterdir()):
        print(d.name, {s: len(os.listdir(d / "images" / s)) for s in os.listdir(d / "images")})


if __name__ == "__main__":
    main()
