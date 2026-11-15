"""Per-source inventories (sha256, size, labels, shipped split) for all four tasks.

Writes data/inv_{io_cls,io_aug,io_det,px}.json (local paths; gitignored) and the
path lists used by dentunity.embed. Nothing is copied; images are read in place.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .common import DATA, FEAT_ROOT, IODET_DIR, PANO_INVENTORY, sha256_file, write_json
from .taxonomy import PX_CLASSES


def folder_labels():
    """Multi-label targets = union of the source folders that hold a byte-identical file
    (no propagation across near-duplicates)."""
    from .common import ORAL_ROOT
    from .taxonomy import IO_CLS_CLASSES, IO_CLS_FOLDERS
    lab, n_files = {}, 0
    for c, name in enumerate(IO_CLS_CLASSES):
        for r, ds, fs in os.walk(ORAL_ROOT / IO_CLS_FOLDERS[name]):
            if "preview" in r.lower() or "augment" in r.lower():
                continue
            for f in fs:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    n_files += 1
                    lab.setdefault(sha256_file(os.path.join(r, f)), set()).add(c)
    return lab, n_files


def io_cls():
    s = json.loads((FEAT_ROOT / "results/splits_intraoral_lineage.json").read_text())
    lab, n_files = folder_labels()
    rows = []
    for split, rs in s["splits"].items():
        for r in rs:
            w, h = Image.open(r["path"]).size
            sha = sha256_file(r["path"])
            rows.append({"path": r["path"], "labels": sorted(lab[sha]), "labels_propagated_earlier": sorted(r["label"]),
                         "split": split,
                         "lineage_src": int(r["lineage"]), "sha256": sha,
                         "width": w, "height": h, "folder": r["path"].split(
                             "Dataset_04_11_Intraoral_Expert/")[1].split("/")[0]})
    return rows


def io_aug():
    d = np.load(FEAT_ROOT / "data/compact/intraoral_vits14.npz")
    kind = json.loads(str(d["extra"]))["kind"]
    rows = []
    for p, k in zip(d["paths"], kind):
        if k == "aug":
            rows.append({"path": str(p), "sha256": sha256_file(str(p)),
                         "folder": str(p).split("Dataset_04_11_Intraoral_Expert/")[1].split("/")[0]})
    return rows


def io_det():
    rows = []
    for split in ("train", "val"):
        idir, ldir = IODET_DIR / "images" / split, IODET_DIR / "labels" / split
        for f in sorted(os.listdir(idir)):
            p = idir / f
            lp = ldir / (Path(f).stem + ".txt")
            if not lp.exists():
                continue                     # 34 unlabelled images are excluded
            boxes = []
            for line in open(lp):
                t = line.split()
                if len(t) == 5:
                    boxes.append([int(t[0])] + [float(v) for v in t[1:]])
            im = Image.open(p)
            rows.append({"path": str(p), "label_path": str(lp), "shipped_split": split,
                         "sha256": sha256_file(p), "width": im.size[0], "height": im.size[1],
                         "boxes_cxcywh": boxes})
    return rows


def _poly_kind(pts):
    n = len(pts)
    if n in (4, 5):
        xs = np.unique(np.round(pts[:, 0], 4))
        ys = np.unique(np.round(pts[:, 1], 4))
        return "axis_box" if len(xs) <= 2 and len(ys) <= 2 else "quad"
    return "polygon"


def px():
    inv = json.loads(PANO_INVENTORY.read_text())
    root = PANO_INVENTORY.parents[1]
    rows = []
    for r in inv:
        p = r["path"] if r["path"].startswith("/") else str(root / r["path"])
        lp = r["label_path"] if r["label_path"].startswith("/") else str(root / r["label_path"])
        inst = []
        if "quarantine" not in r and os.path.exists(lp):
            for line in open(lp):
                t = line.split()
                if len(t) < 7:
                    continue
                pts = np.array([float(v) for v in t[1:]]).reshape(-1, 2)
                inst.append({"c": int(t[0]), "kind": _poly_kind(pts),
                             "box": [float(pts[:, 0].min()), float(pts[:, 1].min()),
                                     float(pts[:, 0].max()), float(pts[:, 1].max())]})
        rows.append({"path": p, "label_path": lp, "shipped_split": r["original_split"],
                     "source_key": r["source_key"], "sha256": r["sha256"],
                     "annotation_sha256": r.get("annotation_sha256"),
                     "width": r["width"], "height": r["height"],
                     "quarantine": r.get("quarantine"), "instances": inst})
    assert len(PX_CLASSES) == 31
    return rows


def main():
    DATA.mkdir(exist_ok=True)
    lab, n_files = folder_labels()
    write_json(DATA.parent / "results" / "io_cls_labels_audit.json", {
        "provenance": "run", "original_files": n_files, "unique_sha256": len(lab),
        "files_in_2plus_folders": sum(len(v) > 1 for v in lab.values()),
        "label_count_hist": {k: sum(len(v) == k for v in lab.values()) for k in range(1, 7)}})
    for name, fn in [("io_cls", io_cls), ("io_aug", io_aug), ("io_det", io_det), ("px", px)]:
        dst = DATA / f"inv_{name}.json"
        if not dst.exists():
            rows = fn()
            write_json(dst, rows)
        rows = json.loads(dst.read_text())
        (DATA / f"list_{name}.txt").write_text("\n".join(r["path"] for r in rows) + "\n")
        print(name, len(rows))


if __name__ == "__main__":
    main()
