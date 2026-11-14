#!/usr/bin/env python3
"""Rebuild DentUnity v1 from your own downloads of the two upstream releases.

This file is the whole build kit (Python 3.9+, standard library only). It reads no
network resource and never uses an upstream file name as an identifier: every image and
label file is matched to the manifests by the sha256 of its bytes, and output files are
named by content hash.

    python build_dentunity.py \
        --pano-root  <unzipped lokisilvres/dental-disease-panoramic-detection-dataset> \
        --oral-root  <unzipped salmansajid05/oral-diseases> \
        --out dentunity_v1 [--with-reexports] [--copy]

Outputs (under --out):
    io_cls/<split>/<sha16>.<ext> + io_cls.csv   multi-label classification (6 labels)
    io_det/   YOLO detection tree (4 classes)   intraoral lesion detection
    px_det/   YOLO detection tree (31 classes)  panoramic detection (boxes from polygons)
    px_seg/   YOLO segmentation tree            panoramic instance segmentation
    taxonomy.json                               36 unified concepts, 5 families
    verify.json                                 found / missing counts and content digests;
                                                compare with manifests/expected_digests.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFESTS = HERE.parent / "manifests"
IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ---------------------------------------------------------------- taxonomy (36 concepts)
FAMILIES = ["pathology", "periodontal / soft tissue", "tooth status / development", "anatomy", "restoration / device"]
CONCEPTS = [  # (unified name, family, {task: upstream label string})
    ("caries", "pathology", {"io_cls": "caries", "io_det": "Caries", "px": "Caries"}),
    ("calculus", "periodontal / soft tissue", {"io_cls": "calculus"}),
    ("gingivitis", "periodontal / soft tissue", {"io_cls": "gingivitis", "io_det": "Gingivitis"}),
    ("mouth ulcer", "periodontal / soft tissue", {"io_cls": "mouth ulcer", "io_det": "Ulcer"}),
    ("tooth discoloration", "pathology", {"io_cls": "tooth discoloration", "io_det": "Tooth Discoloration"}),
    ("hypodontia", "tooth status / development", {"io_cls": "hypodontia"}),
    ("periapical lesion", "pathology", {"px": "Periapical lesion"}),
    ("bone loss", "periodontal / soft tissue", {"px": "Bone Loss"}),
    ("bone defect", "pathology", {"px": "bone defect"}),
    ("cyst", "pathology", {"px": "Cyst"}),
    ("tooth fracture", "pathology", {"px": "Fracture teeth"}),
    ("attrition", "pathology", {"px": "attrition"}),
    ("root resorption", "pathology", {"px": "Root resorption"}),
    ("missing tooth", "tooth status / development", {"px": "Missing teeth"}),
    ("impacted tooth", "tooth status / development", {"px": "impacted tooth"}),
    ("malaligned tooth", "tooth status / development", {"px": "Malaligned"}),
    ("supra-eruption", "tooth status / development", {"px": "Supra Eruption"}),
    ("retained root", "tooth status / development", {"px": "Retained root"}),
    ("root piece", "tooth status / development", {"px": "Root Piece"}),
    ("permanent tooth", "tooth status / development", {"px": "Permanent Teeth"}),
    ("primary tooth", "tooth status / development", {"px": "Primary teeth"}),
    ("mandibular canal", "anatomy", {"px": "Mandibular Canal"}),
    ("maxillary sinus", "anatomy", {"px": "maxillary sinus"}),
    ("crown", "restoration / device", {"px": "Crown"}),
    ("filling", "restoration / device", {"px": "Filling"}),
    ("root canal treatment", "restoration / device", {"px": "Root Canal Treatment"}),
    ("post-core", "restoration / device", {"px": "post - core"}),
    ("implant", "restoration / device", {"px": "Implant"}),
    ("abutment", "restoration / device", {"px": "abutment"}),
    ("gingival former", "restoration / device", {"px": "gingival former"}),
    ("orthodontic bracket", "restoration / device", {"px": "orthodontic brackets"}),
    ("orthodontic wire", "restoration / device", {"px": "wire"}),
    ("metal band", "restoration / device", {"px": "metal band"}),
    ("permanent retainer", "restoration / device", {"px": "permanent retainer"}),
    ("temporary anchorage device", "restoration / device", {"px": "TAD"}),
    ("bone plating", "restoration / device", {"px": "plating"}),
]
# task-local class orders exactly as shipped upstream (YOLO ids)
IO_CLS_CLASSES = ["calculus", "caries", "gingivitis", "hypodontia", "tooth discoloration", "mouth ulcer"]
IO_DET_CLASSES = ["Caries", "Ulcer", "Tooth Discoloration", "Gingivitis"]
PX_CLASSES = ["Caries", "Crown", "Filling", "Implant", "Malaligned", "Mandibular Canal", "Missing teeth",
              "Periapical lesion", "Retained root", "Root Canal Treatment", "Root Piece", "impacted tooth",
              "maxillary sinus", "Bone Loss", "Fracture teeth", "Permanent Teeth", "Supra Eruption", "TAD",
              "abutment", "attrition", "bone defect", "gingival former", "metal band", "orthodontic brackets",
              "permanent retainer", "post - core", "plating", "wire", "Cyst", "Root resorption", "Primary teeth"]
# IO-CLS labels = membership of the six ORIGINAL condition folders (augmented folders excluded);
# a photograph filed in several folders gets several labels.
IO_CLS_FOLDERS = {"calculus": "Calculus/Calculus",
                  "caries": "Data caries/Data caries/caries orignal data set",
                  "gingivitis": "Gingivitis/Gingivitis",
                  "hypodontia": "hypodontia/hypodontia",
                  "tooth discoloration": "Tooth Discoloration/Tooth Discoloration/tooth discoloration original dataset",
                  "mouth ulcer": "Mouth Ulcer/Mouth Ulcer/ulcer original dataset"}


def unified_ids(task):
    lut = {s[task]: i for i, (_, _, s) in enumerate(CONCEPTS) if task in s}
    names = {"io_cls": IO_CLS_CLASSES, "io_det": IO_DET_CLASSES, "px": PX_CLASSES}[task]
    return [lut[n] for n in names]


# ---------------------------------------------------------------- helpers
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def walk(roots):
    """Index images and YOLO label files by the sha256 of their bytes.
    Returns (sha256 -> list of image paths, sha256 -> label path). Label files are found by
    content, so it does not matter in which folder of the download they sit."""
    imgs, labs = defaultdict(list), {}
    for root in roots:
        for r, _, fs in os.walk(root, followlinks=True):
            for f in sorted(fs):
                p = Path(r) / f
                if f.lower().endswith(IMG):
                    imgs[sha256(p)].append(p)
                elif f.endswith(".txt") and "labels" in p.parts:
                    labs.setdefault(sha256(p), p)
    return imgs, labs


def put(dst: Path, src: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_yaml(root: Path, names):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                                    + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))


def clip(v):
    return min(max(float(v), 0.0), 1.0)


def rows(name):
    with open(MANIFESTS / name) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- build
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pano-root", required=True, nargs="+")
    ap.add_argument("--oral-root", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-reexports", action="store_true",
                    help="add the re-exported copies of TRAINING lineages to px_det/train (never touches val/test)")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking them")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"found": defaultdict(int), "missing": defaultdict(int), "label_mismatch": defaultdict(int),
           "counts": defaultdict(int)}
    digest = defaultdict(hashlib.sha256)          # order-independent content digests per task
    lines_for_digest = defaultdict(list)

    print("hashing panoramic files ...", flush=True)
    pano, pano_labels = walk(a.pano_root)
    print("hashing intraoral files ...", flush=True)
    oral, oral_labels = walk(a.oral_root)

    # IO-CLS: labels from folder membership
    folder_labels = defaultdict(set)
    for root in a.oral_root:
        root = Path(root)
        for lab, rel in IO_CLS_FOLDERS.items():
            d = root / rel
            if not d.is_dir():
                hits = [p for p in root.rglob(Path(rel).name) if p.is_dir() and str(p).endswith(rel)]
                d = hits[0] if hits else d
            if d.is_dir():
                for p in sorted(d.rglob("*")):
                    if p.suffix.lower() in IMG:
                        folder_labels[sha256(p)].add(lab)
    uc = unified_ids("io_cls")
    with open(out / "io_cls.csv", "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["image", "split", "lineage"] + IO_CLS_CLASSES + ["unified_ids"])
        for r in rows("io_cls.csv"):
            ps, labs = oral.get(r["sha256"]), folder_labels.get(r["sha256"])
            if not ps or not labs:
                rep["missing"]["io_cls"] += 1
                continue
            rep["found"]["io_cls"] += 1
            dst = out / "io_cls" / r["split"] / f"{r['sha256'][:16]}{ps[0].suffix.lower()}"
            put(dst, ps[0], a.copy)
            y = [int(c in labs) for c in IO_CLS_CLASSES]
            w.writerow([str(dst.relative_to(out)), r["split"], r["lineage"]] + y +
                       [" ".join(str(uc[i]) for i, v in enumerate(y) if v)])
            rep["counts"]["io_cls_labels"] += sum(y)
            lines_for_digest["io_cls"].append(r["sha256"] + ":" + "".join(map(str, y)))

    # IO-DET: boxes from the upstream YOLO label whose bytes match label_sha256
    for r in rows("io_det.csv"):
        ps, lab = oral.get(r["sha256"]), oral_labels.get(r["label_sha256"])
        if not ps or lab is None:
            rep["missing" if not ps else "label_mismatch"]["io_det"] += 1
            continue
        rep["found"]["io_det"] += 1
        img = ps[0]
        name = f"{r['sha256'][:16]}{img.suffix.lower()}"
        put(out / "io_det" / "images" / r["split"] / name, img, a.copy)
        ls = [" ".join(t) for t in (l.split() for l in open(lab)) if len(t) == 5]
        (out / "io_det" / "labels" / r["split"]).mkdir(parents=True, exist_ok=True)
        (out / "io_det" / "labels" / r["split"] / f"{r['sha256'][:16]}.txt").write_text("\n".join(ls) + "\n")
        rep["counts"]["io_det_boxes"] += len(ls)
        lines_for_digest["io_det"] += [r["sha256"] + ":" + l for l in ls]
    write_yaml(out / "io_det", IO_DET_CLASSES)

    # PX-DET / PX-SEG: polygons from the upstream label whose bytes match label_sha256
    for r in rows("px.csv"):
        canonical = r["role"] == "canonical"
        if not (canonical or (a.with_reexports and r["split"] == "train")):
            continue
        ps, lab = pano.get(r["sha256"]), pano_labels.get(r["label_sha256"])
        key = "px" if canonical else "px_reexport"
        if not ps or lab is None:
            rep["missing" if not ps else "label_mismatch"][key] += 1
            continue
        rep["found"][key] += 1
        img = ps[0]
        polys = [t for t in (l.split() for l in open(lab)) if len(t) >= 7]
        boxes = []
        for t in polys:
            xs, ys = [float(v) for v in t[1::2]], [float(v) for v in t[2::2]]
            x0, y0, x1, y1 = clip(min(xs)), clip(min(ys)), clip(max(xs)), clip(max(ys))
            boxes.append(f"{t[0]} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
        name = f"{r['sha256'][:16]}{img.suffix.lower()}"
        put(out / "px_det" / "images" / r["split"] / name, img, a.copy)
        (out / "px_det" / "labels" / r["split"]).mkdir(parents=True, exist_ok=True)
        (out / "px_det" / "labels" / r["split"] / (Path(name).stem + ".txt")).write_text("\n".join(boxes) + "\n")
        if canonical:
            rep["counts"]["px_det_boxes"] += len(boxes)
            lines_for_digest["px_det"] += [r["sha256"] + ":" + b for b in boxes]
        if canonical and r["in_seg"] in ("1", "True", "true"):
            seg = [" ".join([t[0]] + [f"{clip(v):.6f}" for v in t[1:]]) for t in polys]
            put(out / "px_seg" / "images" / r["split"] / name, img, a.copy)
            (out / "px_seg" / "labels" / r["split"]).mkdir(parents=True, exist_ok=True)
            (out / "px_seg" / "labels" / r["split"] / (Path(name).stem + ".txt")).write_text("\n".join(seg) + "\n")
            rep["counts"]["px_seg_images"] += 1
            rep["counts"]["px_seg_instances"] += len(seg)
            lines_for_digest["px_seg"] += [r["sha256"] + ":" + s for s in seg]
    write_yaml(out / "px_det", PX_CLASSES)
    write_yaml(out / "px_seg", PX_CLASSES)

    (out / "taxonomy.json").write_text(json.dumps({
        "families": FAMILIES,
        "concepts": [{"id": i, "name": n, "family": f, "source_labels": s} for i, (n, f, s) in enumerate(CONCEPTS)],
        "task_to_unified": {t: unified_ids(t) for t in ("io_cls", "io_det", "px")}}, indent=1))
    for k, ls in lines_for_digest.items():
        for l in sorted(ls):
            digest[k].update((l + "\n").encode())
    res = {k: dict(v) for k, v in rep.items()}
    res["content_digests"] = {k: d.hexdigest() for k, d in digest.items()}
    exp = MANIFESTS / "expected_digests.json"
    if exp.exists():
        e = json.loads(exp.read_text())
        res["matches_expected"] = {k: res["content_digests"].get(k) == v for k, v in e["content_digests"].items()}
    (out / "verify.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
