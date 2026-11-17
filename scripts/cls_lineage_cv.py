"""Repeated lineage-level 5-fold CV of the IO-CLS frozen read-outs (plan amendment 2026-11-17 11:07).

All IO-CLS images are pooled. The largest lineage stays in training (as in the released split);
the other lineages are assigned to 5 folds at random, 3 repeats. Each read-out keeps the C chosen
on the original validation split (results/cls_frozen.json); standardisation and the 1,024-d PCA are
fitted on the training folds only. CPU only, no new model selection.

  nice -n 10 python3 scripts/cls_lineage_cv.py   ->  results/cls_lineage_cv.json
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from dentunity.cls_data import ARCHS, Y, feats, load_compact, rows  # noqa: E402
from dentunity.common import RESULTS, read_json, write_json  # noqa: E402
from dentunity.metrics import macro_ap  # noqa: E402

KINDS = ("last_cls", "dinov2_lin", "lattice")
K, REPEATS = 5, 3


def prep(Xtr, Xte, dim=1024):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if Xtr.shape[1] > dim:
        p = PCA(dim, svd_solver="randomized", random_state=0).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
    return Xtr.astype(np.float64), Xte.astype(np.float64)


def fit_score(Xtr, ytr, Xte, yte, C):
    from sklearn.linear_model import LogisticRegression
    p = np.stack([LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr[:, c]).predict_proba(Xte)[:, 1]
                  for c in range(ytr.shape[1])], 1)
    return macro_ap(yte, p)


def main():
    t0 = time.time()
    Rr = rows()
    allr = Rr["train"] + Rr["val"] + Rr["test"]
    y = Y(allr)
    lin = np.array([r["lineage"] for r in allr])
    big = Counter(lin.tolist()).most_common(1)[0][0]
    others = sorted(set(lin.tolist()) - {big})
    folds = []
    for rep in range(REPEATS):
        rng = np.random.RandomState(rep)
        perm = rng.permutation(others)
        fold_of = {g: i % K for i, g in enumerate(perm)}
        for k in range(K):
            te = np.array([fold_of.get(g, -1) == k for g in lin])
            folds.append((rep, k, ~te, te))
    CF = read_json(RESULTS / "cls_frozen.json")["models"]
    scores = {}
    blocks = {}
    for arch in ARCHS:
        C, pos = load_compact(arch)
        idx = np.array([pos[r["path"]] for r in allr])
        for kind in KINDS:
            name = f"{arch}_{kind}"
            X = feats(C, idx, kind)
            s = []
            for rep, k, tr, te in folds:
                Xtr, Xte = prep(X[tr], X[te])
                s.append(fit_score(Xtr, y[tr], Xte, y[te], CF[name]["C"]))
                if kind == "lattice":
                    blocks.setdefault((rep, k), {})[arch] = (Xtr.astype(np.float16), Xte.astype(np.float16))
            scores[name] = s
            print(name, round(float(np.mean(s)), 4), round(time.time() - t0), "s", flush=True)
        del C
    s = []                                   # C1: concatenation of the three per-fold PCA'd lattices
    for rep, k, tr, te in folds:
        b = blocks[(rep, k)]
        Xtr = np.concatenate([b[a][0] for a in ARCHS], 1).astype(np.float64)
        Xte = np.concatenate([b[a][1] for a in ARCHS], 1).astype(np.float64)
        s.append(fit_score(Xtr, y[tr], Xte, y[te], CF["multi_lattice"]["C"]))
    scores["multi_lattice"] = s
    print("multi_lattice", round(float(np.mean(s)), 4), flush=True)
    best = max(scores, key=lambda m: np.mean(scores[m]))
    n = len(folds)
    tier = {}
    for m, v in scores.items():
        d = np.array(scores[best]) - np.array(v)
        half = 1.96 * d.std(ddof=1) / np.sqrt(n) if m != best else 0.0
        tier[m] = {"mean_diff_to_best": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)],
                   "tied_with_best": bool(m == best or d.mean() - half <= 0)}
    out = {"provenance": "run (scripts/cls_lineage_cv.py; plan amendment 2026-11-17 11:07)",
           "folds": K, "repeats": REPEATS, "n_images": int(len(allr)), "largest_lineage_kept_in_train": int((lin == big).sum()),
           "lineages_folded": len(others),
           "models": {m: {"fold_macro_ap": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                          "min": float(np.min(v)), "max": float(np.max(v))} for m, v in scores.items()},
           "best": best, "tiers": tier, "runtime_s": round(time.time() - t0, 1),
           "note": "fold-paired 95% interval (normal approximation over folds; folds of repeats overlap, so it is descriptive)"}
    write_json(RESULTS / "cls_lineage_cv.json", out)
    print("best", best, json.dumps({m: round(t["mean_diff_to_best"], 4) for m, t in tier.items()}))


if __name__ == "__main__":
    main()
