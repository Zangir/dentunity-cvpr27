"""T1 data: DENTEX (fully annotated diagnosis subset) -> YOLO tree, plus the copy-lineage
check against every DentUnity panoramic file.

Inputs: data/dentex/raw/{train,val}/ from scripts/dentex_download.py. Train = 705 official
training images; evaluation = the 50 official validation images (4-class COCO labels).
Writes data/yolo/dentex/ and results/dentex_overlap.json. Needs MPS for the embedding.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import DATA, RESULTS, write_json
from .embed import embed_paths, load_model

RAW = DATA / "dentex" / "raw"
Y = DATA / "yolo" / "dentex"
NAMES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]   # DENTEX categories_3 ids 0..3


def coco_to_rows(js, img_dir):
    by = {im["id"]: im for im in js["images"]}
    ann = {}
    for a in js["annotations"]:
        ann.setdefault(a["image_id"], []).append(a)
    rows = []
    for iid, im in by.items():
        p = img_dir / im["file_name"]
        if not p.exists():
            continue
        W, H = im["width"], im["height"]
        boxes = [(a["category_id_3"], (a["bbox"][0] + a["bbox"][2] / 2) / W, (a["bbox"][1] + a["bbox"][3] / 2) / H,
                  a["bbox"][2] / W, a["bbox"][3] / H) for a in ann.get(iid, [])]
        rows.append({"path": p, "boxes": boxes})
    return rows


def main():
    tr = coco_to_rows(json.loads(next((RAW / "train").glob("*.json")).read_text()), RAW / "train")
    va = coco_to_rows(json.loads((RAW / "val" / "validation_triple.json").read_text()), RAW / "val")
    # copy-lineage check against all DentUnity panoramic files (same rule as the build)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(dev)
    E = embed_paths([str(r["path"]) for r in tr + va], model, dev)
    z = np.load(DATA / "linkemb_px.npz")
    Ep = z["E"].astype(np.float32)
    stats = json.loads((RESULTS / "dataset_stats.json").read_text())["audit"]
    tau = stats["px_tau_hi"]
    mx = (E @ Ep.T).max(1)
    flag = mx >= tau
    ntr = len(tr)
    keep_val = [r for r, f in zip(va, flag[ntr:]) if not f]
    res = {"provenance": "run", "tau": tau, "dentex_train_images": ntr, "dentex_eval_images": len(va),
           "train_images_with_copy_in_dentunity": int(flag[:ntr].sum()),
           "eval_images_with_copy_in_dentunity": int(flag[ntr:].sum()),
           "eval_images_kept": len(keep_val), "max_sim_quantiles": np.quantile(mx, [0.5, 0.9, 0.99, 1.0]).tolist(),
           "eval_boxes_kept": sum(len(r["boxes"]) for r in keep_val),
           "eval_class_boxes": dict(Counter(NAMES[b[0]] for r in keep_val for b in r["boxes"])),
           "train_boxes": sum(len(r["boxes"]) for r in tr)}
    for split, rows in (("train", tr), ("val", keep_val), ("test", keep_val)):
        (Y / "images" / split).mkdir(parents=True, exist_ok=True)
        (Y / "labels" / split).mkdir(parents=True, exist_ok=True)
        for r in rows:
            dst = Y / "images" / split / (r["path"].stem + ".jpg")
            if not dst.exists():
                im = Image.open(r["path"]).convert("RGB")
                s = 640 / max(im.size)
                im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BILINEAR).save(dst, quality=95)
            (Y / "labels" / split / (r["path"].stem + ".txt")).write_text(
                "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in r["boxes"]) + "\n")
    (Y / "data.yaml").write_text(f"path: {Y}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                                 + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    write_json(RESULTS / "dentex_overlap.json", res)
    print(res)


if __name__ == "__main__":
    main()
