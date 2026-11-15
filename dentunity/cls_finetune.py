"""Fine-tuned classifiers on IO-CLS (ConvNeXt-T, DINOv2 ViT-S/14), chunked + resumable.

Recipe (fixed before any test evaluation): ImageNet/DINOv2 weights, AdamW, weight decay
0.05, cosine schedule, 15 epochs, batch 32 (2 x 16 accumulation), 224 px,
RandomResizedCrop(0.65-1)+flip+colour jitter, BCE multi-label loss; epoch chosen by
validation macro-AP. Writes data/preds/cls_ft_<model>_s<seed>.npz and
results/cls_ft_<model>_s<seed>.json.

  python -m dentunity.cls_finetune --model convnext_t --seed 0 --budget 540
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .cls_data import Y, rows
from .common import DATA, RESULTS, write_json
from .metrics import macro_ap, macro_auroc, per_class_ap

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
HUB = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
LR = {"convnext_t": 2e-4, "vits14": 5e-5}


class DS(torch.utils.data.Dataset):
    def __init__(self, recs, train, size=224):
        import torchvision.transforms as T
        self.recs, self.y = recs, Y(recs)
        norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
        self.tf = T.Compose(([T.RandomResizedCrop(size, scale=(0.65, 1.0)), T.RandomHorizontalFlip(),
                              T.ColorJitter(0.2, 0.2, 0.1)] if train else
                             [T.Resize(int(size * 1.15)), T.CenterCrop(size)]) + norm)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        return self.tf(Image.open(self.recs[i]["path"]).convert("RGB")), self.y[i]


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.hub.load(str(HUB), "dinov2_vits14", source="local")
        self.head = nn.Linear(384, 6)

    def forward(self, x):
        return self.head(self.enc(x))


def build(name):
    import torchvision
    if name == "convnext_t":
        m = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1")
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 6)
        return m
    return ViT()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convnext_t")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--budget", type=float, default=540)
    ap.add_argument("--split", default="split")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    run = f"cls{a.tag}_ft_{a.model}_s{a.seed}"
    res_p = RESULTS / f"{run}.json"
    if res_p.exists():
        print("done", res_p)
        return
    ck = DATA / "ckpt" / f"{run}.pt"
    ck.parent.mkdir(exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    R = rows(a.split)
    model = build(a.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR[a.model], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    g = torch.Generator().manual_seed(a.seed)
    tr = torch.utils.data.DataLoader(DS(R["train"], True), batch_size=16, shuffle=True, generator=g,
                                     num_workers=2, persistent_workers=False)
    yv, yt = Y(R["val"]), Y(R["test"])

    def predict(rs):
        model.eval()
        dl = torch.utils.data.DataLoader(DS(rs, False), batch_size=32, num_workers=2)
        out = []
        with torch.no_grad():
            for xb, _ in dl:
                out.append(torch.sigmoid(model(xb.to(dev))).float().cpu().numpy())
        model.train()
        return np.concatenate(out)
    ep0, best, best_state, hist = 0, -1.0, None, []
    if ck.exists():
        s = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        ep0, best, best_state, hist = s["epoch"], s["best"], s["best_state"], s["hist"]
        g.set_state(s["gen"]); torch.set_rng_state(s["rng"])
        print(f"resumed at epoch {ep0}", flush=True)
    times = []
    for ep in range(ep0, a.epochs):
        if times and time.time() - t0 + 1.2 * max(times) > a.budget:
            print("BUDGET stop before epoch", ep, flush=True)
            return
        te = time.time()
        model.train()
        for k, (xb, yb) in enumerate(tr):
            loss = F.binary_cross_entropy_with_logits(model(xb.to(dev)), yb.to(dev)) / 2
            loss.backward()
            if k % 2 == 1:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad(); sched.step()
        s = macro_ap(yv, predict(R["val"]))
        hist.append(s)
        if s > best:
            best = s
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1, "best": best, "best_state": best_state, "hist": hist,
                    "gen": g.get_state(), "rng": torch.get_rng_state()}, ck)
        times.append(time.time() - te)
        print(f"epoch {ep + 1}: val macro-AP {s:.4f} ({times[-1]:.0f}s)", flush=True)
    if time.time() - t0 > a.budget - 60:
        print("BUDGET: test next chunk", flush=True)
        return
    model.load_state_dict(best_state)
    pv, pt = predict(R["val"]), predict(R["test"])
    np.savez(DATA / "preds" / f"{run}.npz", val=pv, test=pt)
    write_json(res_p, {"provenance": "run", "model": a.model, "seed": a.seed, "epochs": a.epochs,
                       "best_epoch": int(np.argmax(hist) + 1), "val_history": hist,
                       "val_macro_ap": float(macro_ap(yv, pv)), "test_macro_ap": macro_ap(yt, pt),
                       "test_macro_auroc": macro_auroc(yt, pt), "test_per_class_ap": per_class_ap(yt, pt)})
    ck.unlink()
    print("test macro-AP", macro_ap(yt, pt), flush=True)


if __name__ == "__main__":
    main()
