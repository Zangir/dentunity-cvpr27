"""IO-CLS zero-shot vision-language reference (plan amendment 2026-11-16 13:15).

Two local CLIP-family models x four prompt-template sets. Label score = cosine similarity
between the image embedding and the label's text embedding (mean over the set's templates).
The candidate is selected on validation macro-AP only; the test split is scored once, for
the selected candidate only, with the same lineage-cluster bootstrap as the other IO-CLS
models. Weights come from the local Hugging Face cache; no image leaves the machine.

  nice -n 10 python3 -m dentunity.cls_zeroshot
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

from .cls_data import Y, rows
from .common import DATA, RESULTS, write_json
from .metrics import cluster_boot, macro_ap, macro_auroc, per_class_ap

# unified concept names in label-id order (0..5), as in the taxonomy
LABELS = ["caries", "calculus", "gingivitis", "mouth ulcer", "tooth discoloration", "hypodontia"]
TEMPLATES = {
    "T1": ["a photo of {}."],
    "T2": ["an intraoral photograph showing {}."],
    "T3": ["a clinical photograph of the mouth with {}."],
}
TEMPLATES["T4"] = TEMPLATES["T1"] + TEMPLATES["T2"] + TEMPLATES["T3"]
MODELS = {
    "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "clip_vitb16": ("ViT-B-16-quickgelu", "openai"),
}


def _load(name):
    import open_clip
    spec = MODELS[name]
    if isinstance(spec, str):
        model, _, pre = open_clip.create_model_and_transforms(spec)
        tok = open_clip.get_tokenizer(spec)
    else:
        model, _, pre = open_clip.create_model_and_transforms(spec[0], pretrained=spec[1])
        tok = open_clip.get_tokenizer(spec[0])
    return model.eval(), pre, tok


@torch.no_grad()
def _img_emb(model, pre, paths, bs=32):
    out = []
    for s in range(0, len(paths), bs):
        x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[s:s + bs]])
        e = model.encode_image(x).float()
        out.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.cat(out)


@torch.no_grad()
def _txt_emb(model, tok, templates):
    t = []
    for lab in LABELS:
        e = model.encode_text(tok([tp.format(lab) for tp in templates])).float()
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        t.append(torch.nn.functional.normalize(e, dim=0))
    return torch.stack(t)


def main():
    torch.set_num_threads(4)
    t0 = time.time()
    R = rows()
    yv, yt = Y(R["val"]), Y(R["test"])
    out = {"provenance": "run", "amendment": "2026-11-16T13:15+04:00", "labels": LABELS,
           "templates": TEMPLATES, "models": {k: str(v) for k, v in MODELS.items()},
           "selection": "validation macro-AP only; test scored once for the selected candidate",
           "candidates_val_macro_ap": {}}
    cache = {}
    for m in MODELS:
        model, pre, tok = _load(m)
        iv = _img_emb(model, pre, [r["path"] for r in R["val"]])
        for tn, tps in TEMPLATES.items():
            pv = (iv @ _txt_emb(model, tok, tps).T).numpy()
            out["candidates_val_macro_ap"][f"{m}/{tn}"] = macro_ap(yv, pv)
            print(m, tn, round(out["candidates_val_macro_ap"][f"{m}/{tn}"], 4), flush=True)
        cache[m] = (model, pre, tok)
    sel = max(out["candidates_val_macro_ap"], key=out["candidates_val_macro_ap"].get)
    m, tn = sel.split("/")
    model, pre, tok = cache[m]
    it = _img_emb(model, pre, [r["path"] for r in R["test"]])
    tx = _txt_emb(model, tok, TEMPLATES[tn])
    pt = (it @ tx.T).numpy()
    pv = (_img_emb(model, pre, [r["path"] for r in R["val"]]) @ tx.T).numpy()
    clusters = np.array([r["lineage"] for r in R["test"]])
    ci, _ = cluster_boot(yt, clusters, {"zs": pt}, n=1000)
    out.update({"selected": sel, "val_macro_ap": out["candidates_val_macro_ap"][sel],
                "test_macro_ap": macro_ap(yt, pt), "test_macro_auroc": macro_auroc(yt, pt),
                "test_per_class_ap": per_class_ap(yt, pt), "test_ci95": ci["zs"],
                "test_chance_macro_ap": float(yt.mean(0).mean()),  # constant score = prevalence
                "n_test": int(len(yt)), "runtime_s": round(time.time() - t0, 1)})
    (DATA / "preds").mkdir(exist_ok=True)
    np.savez(DATA / "preds" / "cls_zeroshot.npz", val=pv, test=pt)
    write_json(RESULTS / "cls_zeroshot.json", out)
    print("selected", sel, "test macro-AP", round(out["test_macro_ap"], 4), ci["zs"], flush=True)


if __name__ == "__main__":
    main()
