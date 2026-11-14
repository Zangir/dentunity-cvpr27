"""L4: the YOLOv8x-seg checkpoint shipped with the panoramic release, evaluated on our
PX-DET and PX-SEG test splits. It was trained on the shipped training split, so we also
measure how many of our test lineages occur in that split (contamination by
construction). Writes results/shipped_checkpoint.json.
"""
from __future__ import annotations

from collections import defaultdict

from .common import DATA, RESULTS, read_json, write_json
from .yolo_train import _metrics


def main():
    from ultralytics import YOLO
    out = RESULTS / "shipped_checkpoint.json"
    px = read_json(DATA / "bench.json")["px"]
    shipped_train_lin = {r["lineage"] for r in px if r["shipped_split"] == "train"}
    res = {"provenance": "run", "checkpoint": "release best.pt (YOLOv8x-seg, 31 classes, trained on the shipped train split, imgsz 640)"}
    for task in ("pxdet", "pxseg"):
        test = [r for r in px if r["role"] == "canonical" and r["split"] == "test" and (task == "pxdet" or r["in_seg"])]
        res[f"{task}_test_images"] = len(test)
        res[f"{task}_test_share_lineage_in_shipped_train"] = sum(r["lineage"] in shipped_train_lin for r in test) / len(test)
    # a segment validator needs polygons: score PX-DET boxes through rectangle polygons
    # (box metrics only; the mask numbers of this tree are meaningless and not reported)
    import shutil
    src, dst = DATA / "yolo" / "pxdet", DATA / "yolo" / "pxdet_rectseg"
    if not (dst / "data.yaml").exists():
        for split in ("val", "test"):
            (dst / "images" / split).mkdir(parents=True, exist_ok=True)
            (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
            for f in (src / "images" / split).iterdir():
                link = dst / "images" / split / f.name
                if not link.exists():
                    link.symlink_to(f.resolve())
            for f in (src / "labels" / split).iterdir():
                rows = []
                for line in f.read_text().split("\n"):
                    t = line.split()
                    if len(t) == 5:
                        c, x, y, w, h = t[0], *map(float, t[1:])
                        x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
                        rows.append(f"{c} {x0:.6f} {y0:.6f} {x1:.6f} {y0:.6f} {x1:.6f} {y1:.6f} {x0:.6f} {y1:.6f}")
                (dst / "labels" / split / f.name).write_text("\n".join(rows) + "\n")
        y = (src / "data.yaml").read_text().replace(str(src), str(dst)).replace("train: images/train", "train: images/val")
        (dst / "data.yaml").write_text(y)
    m = YOLO(str(DATA / "weights" / "shipped_yolov8x_seg.pt"))
    names = m.names
    for task in ("pxseg", "pxdet"):
        key = f"{task}_test"
        if key in res and "error" not in res[key]:
            continue
        try:
            data = DATA / "yolo" / ("pxseg" if task == "pxseg" else "pxdet_rectseg") / "data.yaml"
            v = m.val(data=str(data), split="test", imgsz=640, batch=4,
                      device="mps", workers=2, plots=False, verbose=False, task="segment",
                      project=str(DATA / "runs"), name=f"shipped_{task}", exist_ok=True)
            res[key] = _metrics(v, names)
            if task == "pxdet":
                res[key].pop("mask", None)
        except Exception as e:  # box-only labels cannot be scored by a segment validator
            res[key] = {"error": str(e)[:300]}
        write_json(out, res)
    print({k: (v if not isinstance(v, dict) else {kk: vv.get("map50_95") if isinstance(vv, dict) else vv for kk, vv in v.items()}) for k, v in res.items()})


if __name__ == "__main__":
    main()
