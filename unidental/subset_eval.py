"""Post-hoc, descriptive (plan amendment 2026-11-14 18:00): score models on the PX test
images that have NO copy (no shared lineage) in the SHIPPED training split, i.e. images
the shipped checkpoint cannot have memorised, with image-bootstrap 95% CIs and paired
differences. Per-image matching statistics are captured from the Ultralytics validator,
so the full-test numbers reproduce the standard metric exactly. This analysis never
selects a model and does not change the state-of-the-art rule.
Writes results/subset_eval.json (incremental per model).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .common import DATA, RESULTS, read_json, write_json

N_BOOT = 1000


def _subset_names(task):
    px = read_json(DATA / "bench.json")["px"]
    ship_lin = {r["lineage"] for r in px if r["shipped_split"] == "train"}
    out = {}
    for r in px:
        if r["role"] == "canonical" and r["split"] == "test" and (task == "pxdet" or r["in_seg"]):
            name = r["sha256"][:16] + "_" + hashlib.sha256(r["path"].encode()).hexdigest()[:6] + ".jpg"
            out[name] = r["lineage"] not in ship_lin
    return out


def _collect(weights, data, task):
    from ultralytics import YOLO
    m = YOLO(str(weights))
    rec = []
    import ultralytics.utils.metrics as UM
    origs = {C: C.__dict__["update_stats"] for C in (UM.DetMetrics, UM.SegmentMetrics)}

    flag = {"in": False}

    def make(orig):
        def wrap(self, stat):
            outer = not flag["in"]
            if outer:
                rec.append({k: (np.asarray(v).copy() if k != "im_name" else v) for k, v in stat.items()})
            flag["in"] = True
            try:
                return orig(self, stat)
            finally:
                if outer:
                    flag["in"] = False
        return wrap
    for C, o in origs.items():
        setattr(C, "update_stats", make(o))
    try:
        m.val(data=str(data), split="test", imgsz=640, batch=4, device="mps", workers=2, plots=False,
              verbose=False, task=task, project=str(DATA / "runs"), name="subset_eval", exist_ok=True)
    finally:
        for C, o in origs.items():
            setattr(C, "update_stats", o)
    return rec


def _map(stats, key):
    from ultralytics.utils.metrics import ap_per_class
    tp = np.concatenate([s[key] for s in stats]) if stats else np.zeros((0, 10))
    conf = np.concatenate([s["conf"] for s in stats])
    pc = np.concatenate([s["pred_cls"] for s in stats])
    tc = np.concatenate([s["target_cls"] for s in stats])
    if len(tc) == 0:
        return float("nan")
    ap = ap_per_class(tp, conf, pc, tc)[5]
    return float(ap.mean()) if len(ap) else 0.0


def main():
    out_p = RESULTS / "subset_eval.json"
    out = read_json(out_p) if out_p.exists() else {"provenance": "run (post-hoc descriptive)", "models": {}}
    W = DATA / "runs"
    cands = [("shipped_yolov8x_seg|pxdet", DATA / "weights/shipped_yolov8x_seg.pt", "pxdet_rectseg", "segment", "tp"),
             ("shipped_yolov8x_seg|pxseg", DATA / "weights/shipped_yolov8x_seg.pt", "pxseg", "segment", "tp_m")]
    for run in ("pxdet_y11s_s0", "pxdet_y11s_s1", "pxdet_y11m_s0", "pxdet_y11m_s1"):
        cands.append((f"{run}|pxdet", W / run / "weights/best.pt", "pxdet", "detect", "tp"))
    for run in ("pxseg_y11s_s0", "pxseg_y11s_s1", "pxseg_y11m_s0", "pxseg_y11m_s1"):
        cands.append((f"{run}|pxseg", W / run / "weights/best.pt", "pxseg", "segment", "tp_m"))
    for key, w, data, task, tpk in cands:
        run = key.split("|")[0]
        if key in out["models"] or not w.exists():
            continue
        if not key.startswith("shipped") and not (RESULTS / f"yolo_{run}.json").exists():
            continue
        stats = _collect(w, DATA / "yolo" / data / "data.yaml", task)
        sub = _subset_names("pxdet" if data != "pxseg" else "pxseg")
        S = [s for s in stats if sub.get(s["im_name"], False)]
        C = [s for s in stats if s["im_name"] in sub and not sub[s["im_name"]]]
        rng = np.random.default_rng(0)
        boots = [_map([S[i] for i in rng.integers(0, len(S), len(S))], tpk) for _ in range(N_BOOT)]
        out["models"][key] = {"metric": "mask mAP50-95" if tpk == "tp_m" else "box mAP50-95",
                              "full_test": _map(stats, tpk), "n_full": len(stats),
                              "no_copy_subset": _map(S, tpk), "n_subset": len(S),
                              "with_copy_subset": _map(C, tpk), "n_with_copy": len(C),
                              "no_copy_ci95": [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))],
                              "per_image_stats_file": f"data/subset_stats/{key.replace('|', '_')}.npy"}
        (DATA / "subset_stats").mkdir(exist_ok=True)
        np.save(DATA / "subset_stats" / f"{key.replace('|', '_')}.npy", np.array(stats, dtype=object), allow_pickle=True)
        write_json(out_p, out)
        print(key, {k: v for k, v in out["models"][key].items() if k != "per_image_stats_file"}, flush=True)
    # paired bootstrap on the no-copy subset: each of our models minus the shipped checkpoint
    pairs = {}
    for key in list(out["models"]):
        if key.startswith("shipped"):
            continue
        task = key.split("|")[1]
        sk = f"shipped_yolov8x_seg|{task}"
        f1, f2 = DATA / "subset_stats" / f"{key.replace('|', '_')}.npy", DATA / "subset_stats" / f"{sk.replace('|', '_')}.npy"
        if not (f1.exists() and f2.exists()):
            continue
        a = {s["im_name"]: s for s in np.load(f1, allow_pickle=True)}
        b = {s["im_name"]: s for s in np.load(f2, allow_pickle=True)}
        sub = _subset_names(task)
        names = sorted(n for n in a if sub.get(n, False) and n in b)
        tpk = "tp_m" if task == "pxseg" else "tp"
        rng = np.random.default_rng(1)
        d = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(names), len(names))
            d.append(_map([a[names[i]] for i in idx], tpk) - _map([b[names[i]] for i in idx], tpk))
        pairs[f"{key}-minus-shipped"] = {"diff": _map([a[n] for n in names], tpk) - _map([b[n] for n in names], tpk),
                                         "ci95": [float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))]}
    out["paired_no_copy_subset"] = pairs
    write_json(out_p, out)
    print(json.dumps(pairs, indent=1))


if __name__ == "__main__":
    main()
