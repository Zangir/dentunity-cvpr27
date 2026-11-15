"""IO-CLS task data: split rows, label matrix, cached frozen-feature rows."""
from __future__ import annotations

import json

import numpy as np

from .common import DATA, FEAT_ROOT, read_json

ARCHS = ("vits14", "vitb14", "convnext_t")


def rows(split_key="split"):
    b = read_json(DATA / "bench.json")["io_cls"]
    if split_key == "split_hash":   # earlier md5 + dHash split of the same composite
        old = read_json(FEAT_ROOT / "results/splits_cls.json")
        osp = {r["path"]: k for k, v in old["splits"].items() for r in v}
        b = [dict(r, split_hash=osp[r["path"]]) for r in b]
    out = {s: [r for r in b if r[split_key] == s] for s in ("train", "val", "test")}
    return out


def Y(rs):
    y = np.zeros((len(rs), 6), np.float32)
    for i, r in enumerate(rs):
        y[i, r["labels"]] = 1
    return y


def load_compact(arch):
    z = np.load(FEAT_ROOT / "data/compact" / f"intraoral_{arch}.npz")
    return z["C"], {str(p): i for i, p in enumerate(z["paths"])}


def feats(C, idx, kind, chunk=512):
    """Descriptor families on the compact lattice cache (n, 2 views, L, [cls, mean], D).
    last_cls: final-layer class token (CNN: final global mean); dinov2_lin: last-4 class
    tokens + final mean patch (DINOv2 linear-eval protocol); lattice: every layer's
    [cls; mean] block, each L2-normalised, concatenated and re-normalised."""
    out = []
    cnn = C.shape[2] == 18
    for s in range(0, len(idx), chunk):
        x = C[np.sort(idx[s:s + chunk])][:, 0].astype(np.float32)
        order = np.argsort(np.argsort(idx[s:s + chunk]))
        x = x[order]
        if kind == "last_cls":
            z = x[:, -1, 1] if cnn else x[:, -1, 0]
        elif kind == "dinov2_lin":
            z = (x[:, -4:, 1].reshape(len(x), -1) if cnn else
                 np.concatenate([x[:, -4:, 0].reshape(len(x), -1), x[:, -1, 1]], 1))
        elif kind == "lattice":
            b = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
            z = b.reshape(len(x), -1)
            z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        else:
            raise ValueError(kind)
        out.append(z)
    return np.concatenate(out)
