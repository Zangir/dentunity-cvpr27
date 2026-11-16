"""PX-SEG box-prompted SAM reference (plan amendment 2026-11-16 13:15).

The validation-selected PX-SEG family (YOLO11s-seg, seeds 0 and 1) is evaluated with the
standard Ultralytics SegmentationValidator, except that every predicted mask is replaced by
the mask SAM ViT-B returns when prompted with the predicted box (single-mask output). Boxes,
classes and confidences are unchanged, so the box mAP must equal the recorded one (pipeline
check); the mask mAP is scored by the same code as every other PX-SEG model. Weights come
from the local Hugging Face cache (facebook/sam-vit-base); no image leaves the machine.

  run_heavy.sh du-sam --timeout 7000 --mem-gb 5 -- python3 -u -m dentunity.sam_pxseg
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

from .common import DATA, RESULTS, read_json, write_json
from .yolo_train import RUNS, _metrics

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SAM_ID = "facebook/sam-vit-base"
OUT = RESULTS / "sam_pxseg.json"
_SAM = {}
# hard stop: the coordinator's cut-off for new runs is 2026-11-16 22:00 local
DEADLINE = time.mktime(time.strptime("2026-11-16 21:58", "%Y-%m-%d %H:%M"))


def _sam():
    if not _SAM:
        from transformers import SamModel, SamProcessor
        _SAM["proc"] = SamProcessor.from_pretrained(SAM_ID)
        _SAM["model"] = SamModel.from_pretrained(SAM_ID).to(DEV).eval()
    return _SAM["proc"], _SAM["model"]


@torch.no_grad()
def sam_masks(img, boxes, out_hw, chunk=12):
    """img: HxWx3 uint8 RGB (letterboxed network input); boxes: (n,4) xyxy in img pixels.
    Returns (n, *out_hw) uint8 masks. The 256x256 low-res logits cover the 1024 padded input, so the
    valid region is cropped at low resolution and resized straight to the validator's mask size
    (no 1024x1024 intermediate, which exceeded the MPS memory cap)."""
    proc, model = _sam()
    inp = proc(images=img, return_tensors="pt")
    emb = model.get_image_embeddings(inp["pixel_values"].to(DEV))
    rh, rw = [int(v) for v in inp["reshaped_input_sizes"][0]]
    h, w = img.shape[:2]
    s = rh / h                                         # SAM resizes the long side to 1024
    ch, cw = max(1, round(rh / 4)), max(1, round(rw / 4))
    out = []
    for c in range(0, len(boxes), chunk):
        b = torch.as_tensor(boxes[c:c + chunk], dtype=torch.float32) * s
        o = model(image_embeddings=emb, input_boxes=b[None].to(DEV), multimask_output=False)
        lo = o.pred_masks[0, :, 0][:, :ch, :cw]        # (k, ch, cw) logits of the valid region
        m = F.interpolate(lo[None], out_hw, mode="bilinear", align_corners=False)[0] > 0
        out.append(m.to(torch.uint8).cpu())
        del o, lo, m
    del emb
    if DEV == "mps":
        torch.mps.empty_cache()
    return torch.cat(out) if out else torch.zeros((0, *out_hw), dtype=torch.uint8)


def _validator_cls():
    from ultralytics.models.yolo.segment import SegmentationValidator

    class SAMSegValidator(SegmentationValidator):
        def update_metrics(self, preds, batch):
            if time.time() > DEADLINE:
                raise SystemExit("cut-off reached")
            for si, pred in enumerate(preds):
                n = int(pred["bboxes"].shape[0])
                if n:
                    img = (batch["img"][si].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
                    hw = tuple(pred["masks"].shape[1:])
                    m = sam_masks(img, pred["bboxes"].float().cpu().numpy(), hw)
                    pred["masks"] = m.to(pred["masks"].device, dtype=pred["masks"].dtype)
            super().update_metrics(preds, batch)
    return SAMSegValidator


def main():
    from ultralytics import YOLO
    out = read_json(OUT) if OUT.exists() else {
        "provenance": "run", "amendment": "2026-11-16T13:15+04:00", "sam": SAM_ID,
        "prompt": "boxes of the YOLO11s-seg run (conf/cls unchanged), single-mask output",
        "metric": "Ultralytics SegmentationValidator mask mAP (same as every PX-SEG model)", "runs": {}}
    data = DATA / "yolo" / "pxseg" / "data.yaml"
    V = _validator_cls()
    for seed, split in ((0, "val"), (0, "test"), (1, "test"), (1, "val")):
        run = f"pxseg_y11s_s{seed}"
        ref = read_json(RESULTS / f"yolo_{run}.json")
        best = RUNS / run / "weights" / "best.pt"
        key = f"{run}/{split}"
        if key in out["runs"]:
            continue
        t0 = time.time()
        model = YOLO(str(best))
        m = model.val(data=str(data), split=split, imgsz=640, batch=int(ref["batch"]), device=DEV,
                      workers=2, plots=False, verbose=False, validator=V,
                      project=str(RUNS), name=f"sam_{run}_{split}", exist_ok=True)
        r = _metrics(m, model.names)
        r["box_map50_95_recorded"] = ref[split]["box"]["map50_95"]
        r["yolo_mask_map50_95_recorded"] = ref[split]["mask"]["map50_95"]
        r["runtime_s"] = round(time.time() - t0, 1)
        out["runs"][key] = r
        write_json(OUT, out)
        print(key, json.dumps({"box": r["box"]["map50_95"], "box_rec": r["box_map50_95_recorded"],
                               "sam_mask": r["mask"]["map50_95"],
                               "yolo_mask": r["yolo_mask_map50_95_recorded"],
                               "s": r["runtime_s"]}), flush=True)



def summarize():
    """Seed means of the test scores (SAM masks and the YOLO11s masks of the same runs), written back
    into results/sam_pxseg.json so that every number in the paper is a stored value."""
    out = read_json(OUT)
    te = {k: r for k, r in out["runs"].items() if k.endswith("/test")}
    out["summary"] = {"test_seeds": len(te),
                      "test_mask_map50_95_mean": float(np.mean([r["mask"]["map50_95"] for r in te.values()])),
                      "test_mask_map50_mean": float(np.mean([r["mask"]["map50"] for r in te.values()])),
                      "yolo11s_test_mask_map50_95_mean_same_seeds":
                          float(np.mean([r["yolo_mask_map50_95_recorded"] for r in te.values()]))}
    write_json(OUT, out)
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    import sys
    summarize() if sys.argv[1:] == ["summarize"] else main()
