"""Frozen-encoder read-outs on IO-CLS (CPU only, from cached features).

For each encoder and descriptor: standardise, PCA to 1024 dims if larger, one
logistic regression per label, C chosen on validation macro-AP only. Writes
results/cls_frozen.json and data/preds/cls_<name>.npz (val/test probabilities).

  python -m dentunity.cls_frozen [--split split|split_src_hash]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .cls_data import ARCHS, Y, feats, load_compact, rows
from .common import DATA, RESULTS, read_json, write_json
from .metrics import macro_ap, macro_auroc, per_class_ap

CS = (0.001, 0.003, 0.01, 0.03, 0.1, 1.0)
KINDS = ("last_cls", "dinov2_lin", "lattice")


def readout(X, Yt, Xv, yv, Xt):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    X, Xv, Xt = sc.transform(X), sc.transform(Xv), sc.transform(Xt)
    if X.shape[1] > 1024:
        p = PCA(1024, svd_solver="randomized", random_state=0).fit(X)
        X, Xv, Xt = p.transform(X), p.transform(Xv), p.transform(Xt)
    X, Xv, Xt = (np.ascontiguousarray(a, dtype=np.float64) for a in (X, Xv, Xt))
    best = None
    for C in CS:
        ms = [LogisticRegression(C=C, max_iter=3000).fit(X, Yt[:, c]) for c in range(Yt.shape[1])]
        pv = np.stack([m.predict_proba(Xv)[:, 1] for m in ms], 1)
        s = macro_ap(yv, pv)
        if best is None or s > best[0]:
            pt = np.stack([m.predict_proba(Xt)[:, 1] for m in ms], 1)
            best = (s, C, pv, pt)
    return best


def _pca_block(X, dim=1024):
    """Standardise + PCA(1024) one encoder's lattice (fit on train) before concatenation,
    so the multi-encoder read-out stays within the 5 GB memory cap."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X["train"])
    p = PCA(dim, svd_solver="randomized", random_state=0).fit(sc.transform(X["train"]))
    return {s: p.transform(sc.transform(v)).astype(np.float32) for s, v in X.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="split")
    ap.add_argument("--tag", default="")
    ap.add_argument("--archs", default=",".join(ARCHS))
    ap.add_argument("--kinds", default=",".join(KINDS))
    a = ap.parse_args()
    out_p = RESULTS / f"cls_frozen{a.tag}.json"
    out = read_json(out_p) if out_p.exists() else {"provenance": "run", "split": a.split, "models": {}}
    R = rows(a.split)
    ys = {s: Y(R[s]) for s in R}
    (DATA / "preds").mkdir(exist_ok=True)
    cache = {}
    for arch in a.archs.split(","):
        C, pos = load_compact(arch)
        idx = {s: np.array([pos[r["path"]] for r in R[s]]) for s in R}
        for kind in a.kinds.split(","):
            name = f"{arch}_{kind}"
            if name in out["models"]:
                continue
            t0 = time.time()
            X = {s: feats(C, idx[s], kind) for s in R}
            if kind == "lattice":
                cache[arch] = _pca_block(X)
            s, Cb, pv, pt = readout(X["train"], ys["train"], X["val"], ys["val"], X["test"])
            out["models"][name] = {"encoder": arch, "descriptor": kind, "C": Cb, "dims": int(X["train"].shape[1]),
                                   "val_macro_ap": s, "test_macro_ap": macro_ap(ys["test"], pt),
                                   "test_macro_auroc": macro_auroc(ys["test"], pt),
                                   "test_per_class_ap": per_class_ap(ys["test"], pt),
                                   "runtime_s": round(time.time() - t0, 1)}
            np.savez(DATA / "preds" / f"cls{a.tag}_{name}.npz", val=pv, test=pt)
            write_json(out_p, out)
            print(name, json.dumps({k: v for k, v in out["models"][name].items()
                                    if k in ("val_macro_ap", "test_macro_ap")}), flush=True)
        del C
    # multi-encoder lattice (concatenate the three normalised lattices)
    name = "multi_lattice"
    if name not in out["models"] and len(a.archs.split(",")) == 3 and "lattice" in a.kinds:
        for arch in ARCHS:
            if arch not in cache:
                C, pos = load_compact(arch)
                cache[arch] = _pca_block({s: feats(C, np.array([pos[r["path"]] for r in R[s]]), "lattice")
                                          for s in R})
                del C
        X = {s: np.concatenate([cache[ar][s] for ar in ARCHS], 1) for s in R}
        s, Cb, pv, pt = readout(X["train"], ys["train"], X["val"], ys["val"], X["test"])
        out["models"][name] = {"encoder": "+".join(ARCHS), "descriptor": "lattice", "C": Cb,
                               "dims": int(X["train"].shape[1]), "val_macro_ap": s,
                               "test_macro_ap": macro_ap(ys["test"], pt),
                               "test_macro_auroc": macro_auroc(ys["test"], pt),
                               "test_per_class_ap": per_class_ap(ys["test"], pt)}
        np.savez(DATA / "preds" / f"cls{a.tag}_{name}.npz", val=pv, test=pt)
        write_json(out_p, out)
        print(name, out["models"][name]["val_macro_ap"], out["models"][name]["test_macro_ap"])


if __name__ == "__main__":
    main()
