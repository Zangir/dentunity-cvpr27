"""Classification metrics and lineage-cluster bootstrap."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def per_class_ap(y, p):
    return [float(average_precision_score(y[:, c], p[:, c])) if 0 < y[:, c].sum() < len(y) else float("nan")
            for c in range(y.shape[1])]


def macro_ap(y, p):
    return float(np.nanmean(per_class_ap(y, p)))


def macro_auroc(y, p):
    return float(np.nanmean([roc_auc_score(y[:, c], p[:, c]) if 0 < y[:, c].sum() < len(y) else np.nan
                             for c in range(y.shape[1])]))


def cluster_boot(y, clusters, preds: dict, score=macro_ap, n=1000, seed=0):
    """Lineage-cluster bootstrap: resample whole test lineages. Returns per-model 95% CIs
    and the bootstrap replicate scores (for paired differences)."""
    rng = np.random.RandomState(seed)
    uc = np.unique(clusters)
    members = {c: np.nonzero(clusters == c)[0] for c in uc}
    reps = {k: [] for k in preds}
    while len(next(iter(reps.values()))) < n:
        idx = np.concatenate([members[c] for c in rng.choice(uc, len(uc))])
        yb = y[idx]
        if ((yb.sum(0) == 0) | (yb.sum(0) == len(yb))).any():
            continue
        for k, p in preds.items():
            reps[k].append(score(yb, p[idx]))
    reps = {k: np.array(v) for k, v in reps.items()}
    ci = {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in reps.items()}
    return ci, reps


def paired_ci(reps, a, b):
    d = reps[a] - reps[b]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
