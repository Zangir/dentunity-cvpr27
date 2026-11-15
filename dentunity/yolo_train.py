"""Chunked, resumable YOLO11 / RT-DETR training and one-shot test evaluation.

Each call trains until its wall-clock budget is nearly spent, exits right after the
epoch checkpoint is written (so the guarded runner never kills mid-epoch), and the
next call resumes from last.pt. Checkpoint selection = best validation fitness
(Ultralytics best.pt). The test split is evaluated once, after training, and the
numbers are written to results/yolo_<run>.json.

  python -m dentunity.yolo_train --run pxdet_y11s_640_s0 --data pxdet --model yolo11s.pt \
      --imgsz 640 --epochs 40 --seed 0 --budget 560
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .common import DATA, REPO, RESULTS, stable_seed, write_json

RUNS = DATA / "runs"


def _patch_mps_unique():
    """MPS workaround: torch.unique(return_counts=True) on MPS occasionally returns
    garbage counts, which crashed DENTEX training ('zeros: Dimension size must be
    non-negative'). Count targets per image with bincount on CPU instead; the loss is
    mathematically unchanged."""
    import torch
    from ultralytics.utils import loss as L
    from ultralytics.utils.ops import xywh2xyxy

    def preprocess(self, targets, batch_size, scale_tensor):
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)
        batch_idx = targets[:, 0].long()
        cmax = int(torch.bincount(batch_idx.cpu(), minlength=batch_size).max())
        out = torch.zeros(batch_size, cmax, ne - 1, device=self.device)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
        out[batch_idx, within_idx] = targets[:, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    L.v8DetectionLoss.preprocess = preprocess


_patch_mps_unique()
WEIGHTS = DATA / "weights"
T0 = time.time()


def _budget_callback(budget):
    state = {"last": T0, "dur": []}

    def cb(trainer):
        now = time.time()
        state["dur"].append(now - state["last"])
        state["last"] = now
        est = max(state["dur"][-2:]) if len(state["dur"]) > 1 else state["dur"][0]
        done = trainer.epoch + 1 >= trainer.epochs
        if not done and (now - T0) + 1.1 * est > budget:
            print(f"BUDGET: stopping after epoch {trainer.epoch + 1} (est {est:.0f}s)", flush=True)
            os._exit(3)          # last.pt already written; resume next chunk
    return cb


def _metrics(m, names):
    out = {}
    for key, obj in (("box", getattr(m, "box", None)), ("mask", getattr(m, "seg", None))):
        if obj is None or not hasattr(obj, "map") or obj.ap_class_index is None:
            continue
        out[key] = {"map50_95": float(obj.map), "map50": float(obj.map50), "map75": float(obj.map75),
                    "precision": float(obj.mp), "recall": float(obj.mr),
                    "per_class_ap50_95": {names[int(c)]: float(obj.ap[i].mean())
                                          for i, c in enumerate(obj.ap_class_index)},
                    "per_class_ap50": {names[int(c)]: float(obj.ap50[i])
                                       for i, c in enumerate(obj.ap_class_index)},
                    "n_classes_evaluated": int(len(obj.ap_class_index))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=560)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--final", choices=["best", "last"], default="best",
                    help="last = fixed schedule, no validation-based selection (T1)")
    ap.add_argument("--shards", type=int, default=1,
                    help="logged epochs per full pass (epochs below are FULL passes)")
    a = ap.parse_args()
    from ultralytics import RTDETR, YOLO
    Model = RTDETR if "rtdetr" in a.model else YOLO
    out = RESULTS / f"yolo_{a.run}.json"
    if out.exists():
        print("done", out)
        return
    data = DATA / "yolo" / a.data / "data.yaml"
    rdir = RUNS / a.run
    last, best = rdir / "weights" / "last.pt", rdir / "weights" / "best.pt"
    marker = rdir / "TRAIN_DONE"
    state = rdir / "shard_epoch.txt"
    trainer = None
    if a.shards > 1 and "rtdetr" not in a.model:
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.models.yolo.segment import SegmentationTrainer
        from .shards import make_trainer
        base = SegmentationTrainer if "seg" in a.model else DetectionTrainer
        trainer = make_trainer(base, a.shards, str(state), stable_seed(a.run))

    def on_epoch_start(tr):
        rdir.mkdir(parents=True, exist_ok=True)
        state.write_text(str(tr.epoch))
    if not marker.exists():
        if last.exists():
            m = Model(str(last))
            m.add_callback("on_fit_epoch_end", _budget_callback(a.budget))
            m.add_callback("on_train_epoch_start", on_epoch_start)
            m.train(resume=True, **({"trainer": trainer} if trainer else {}))
        else:
            w = WEIGHTS / a.model
            m = Model(str(w) if w.exists() else str(DATA / a.model) if (DATA / a.model).exists() else a.model)
            m.add_callback("on_fit_epoch_end", _budget_callback(a.budget))
            m.add_callback("on_train_epoch_start", on_epoch_start)
            kw = dict(data=str(data), epochs=a.epochs * a.shards, imgsz=a.imgsz, batch=a.batch, device="mps",
                      workers=2, cache=False, seed=a.seed, deterministic=True, project=str(RUNS),
                      name=a.run, exist_ok=True, plots=False, verbose=False, amp=False,
                      optimizer=a.optimizer, patience=100)
            if a.lr0 is not None:
                kw["lr0"] = a.lr0
            if a.final == "last":
                kw["val"] = False
            if trainer:
                kw["trainer"] = trainer
                kw["close_mosaic"] = 10 * a.shards // 2   # last ~5 full passes without mosaic
                kw["warmup_epochs"] = 3.0
            m.train(**kw)
        marker.write_text("ok")
        if time.time() - T0 > a.budget - 60:
            print("BUDGET: training finished; test evaluation next chunk", flush=True)
            return
    if a.final == "last":
        best = last
    names = Model(str(best)).names
    res = {"provenance": "run", "run": a.run, "data": a.data, "model": a.model, "imgsz": a.imgsz,
           "epochs_full_passes": a.epochs, "shards": a.shards, "batch": a.batch, "seed": a.seed,
           "checkpoint_selection": ("last epoch (fixed schedule)" if a.final == "last"
                                    else "best validation fitness (Ultralytics best.pt)")}
    for split in (("test",) if a.final == "last" else ("val", "test")):
        m = Model(str(best)).val(data=str(data), split=split, imgsz=a.imgsz, batch=a.batch,
                                 device="mps", workers=2, plots=False, verbose=False,
                                 project=str(RUNS), name=f"{a.run}_{split}", exist_ok=True)
        res[split] = _metrics(m, names)
        res[split]["speed_ms"] = {k: float(v) for k, v in m.speed.items()}
    csv = rdir / "results.csv"
    if csv.exists():
        res["logged_epochs_completed"] = sum(1 for _ in open(csv)) - 1
    write_json(out, res)
    print(json.dumps({s: {k: v.get("map50_95") for k, v in res[s].items() if isinstance(v, dict) and "map50_95" in v}
                      for s in ("val", "test") if s in res}))


if __name__ == "__main__":
    main()
