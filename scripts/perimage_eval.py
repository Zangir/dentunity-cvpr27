"""Per-image detection matches for paired image-bootstrap intervals (post-plan analyses A-C).

Runs the standard Ultralytics validator (same matching and AP as the reported numbers) and records,
for every image, the per-prediction TP matrix over the 10 IoU thresholds, confidences, predicted
and target classes. Image keys are the (hashed / public) YOLO-tree file stems; nothing is printed
per image.
  python3 scripts/perimage_eval.py --weights W.pt --data data/yolo/iodet/data.yaml --split test \
      --out data/preds/pi_iodet_y11n_s0.pkl [--device cpu] [--batch 8]
Library: load(path) -> list of dicts; mapk(stats, keep=None) -> (mAP50:95, mAP50).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


_X = np.linspace(0, 1, 101)


def _ap(recall, precision):
    """Ultralytics compute_ap (101-point interpolated, COCO), vectorised over IoU thresholds."""
    out = np.empty(recall.shape[1])
    for j in range(recall.shape[1]):
        r, p = recall[:, j], precision[:, j]
        mrec = np.concatenate(([0.0], r, [r[-1] if len(r) else 1.0], [1.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], p, [0.0], [0.0])))))
        out[j] = np.trapezoid(np.interp(_X, mrec, mpre), _X)
    return out


def mapk(stats, idx=None):
    """mAP50:95 and mAP50 over the images in idx (indices into stats, repeats allowed). Same
    computation as ultralytics.utils.metrics.ap_per_class (checked equal in main())."""
    sel = stats if idx is None else [stats[i] for i in idx]
    tp = np.concatenate([s["tp"] for s in sel], 0)
    conf = np.concatenate([s["conf"] for s in sel])
    pc = np.concatenate([s["pred_cls"] for s in sel])
    tc = np.concatenate([s["target_cls"] for s in sel])
    if len(tc) == 0:
        return float("nan"), float("nan")
    o = np.argsort(-conf, kind="stable")
    tp, pc = tp[o].astype(float), pc[o]
    cls, nt = np.unique(tc, return_counts=True)
    ap = np.zeros((len(cls), tp.shape[1]))
    for ci, (c, n_l) in enumerate(zip(cls, nt)):
        i = pc == c
        if i.sum() == 0:
            continue
        tpc = tp[i].cumsum(0)
        fpc = (1 - tp[i]).cumsum(0)
        ap[ci] = _ap(tpc / (n_l + 1e-16), tpc / (tpc + fpc))
    return float(ap.mean()), float(ap[:, 0].mean())


def mapk_ultralytics(stats, idx=None):
    from ultralytics.utils.metrics import ap_per_class
    sel = stats if idx is None else [stats[i] for i in idx]
    cat = lambda k: np.concatenate([s[k] for s in sel], 0)
    ap = ap_per_class(cat("tp"), cat("conf"), cat("pred_cls"), cat("target_cls"))[5]
    return float(ap.mean()), float(ap[:, 0].mean())


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    from ultralytics import RTDETR, YOLO
    from ultralytics.utils import metrics as M
    rec = []
    orig = M.DetMetrics.update_stats

    def update_stats(self, stat):
        rec.append({"im": stat["im_name"].rsplit(".", 1)[0], "tp": np.asarray(stat["tp"], bool),
                    "conf": np.asarray(stat["conf"], np.float32), "pred_cls": np.asarray(stat["pred_cls"]),
                    "target_cls": np.asarray(stat["target_cls"])})
        return orig(self, stat)
    M.DetMetrics.update_stats = update_stats
    Model = RTDETR if "rtdetr" in Path(a.weights).name.lower() or "rtdetr" in a.weights else YOLO
    m = Model(a.weights).val(data=a.data, split=a.split, imgsz=a.imgsz, batch=a.batch, device=a.device,
                             workers=0, plots=False, verbose=False, save_json=False)
    out = {"weights": a.weights, "split": a.split, "n_images": len(rec),
           "ultralytics_map50_95": float(m.box.map), "ultralytics_map50": float(m.box.map50), "stats": rec}
    re = mapk(rec)
    out["recomputed_map50_95"], out["recomputed_map50"] = re
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    print(f"{Path(a.out).name}: n={len(rec)} ultralytics={m.box.map:.4f}/{m.box.map50:.4f} recomputed={re[0]:.4f}/{re[1]:.4f}")


if __name__ == "__main__":
    main()
