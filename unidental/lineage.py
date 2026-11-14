"""Copy-lineage graphs. A lineage is a connected component of 'is a copy of' edges:
exact hash, export-tool source key, registered geometric copies, and calibrated
embedding edges (mutual-kNN, size-gated recall edges plus precision-first edges at
the 99.9th percentile of random-pair similarity). Splits are made over lineages,
so no copy of a test image can be in training.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .common import UnionFind


def null_quantile(E, q=0.999, n=40000, seed=0):
    rs = np.random.RandomState(seed)
    a, b = rs.randint(0, len(E), n), rs.randint(0, len(E), n)
    keep = a != b
    return float(np.quantile((E[a[keep]] * E[b[keep]]).sum(1), q))


def mutual_knn_edges(E, tau, k, block=2048, idx=None):
    """Edges (i, j) with sim >= tau where i and j are in each other's top-k."""
    idx = np.arange(len(E)) if idx is None else np.asarray(idx)
    X = E[idx]
    n = len(X)
    if n < 2:
        return []
    k = min(k, n - 1)
    nbr = np.zeros((n, k), np.int64)
    sims = np.zeros((n, k), np.float32)
    for s in range(0, n, block):
        S = X[s:s + block] @ X.T
        r = np.arange(s, min(s + block, n))
        S[r - s, r] = -2
        nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
        nbr[s:s + block] = nn
        sims[s:s + block] = np.take_along_axis(S, nn, 1)
    sets = [set(r.tolist()) for r in nbr]
    out = []
    for i in range(n):
        for j, sv in zip(nbr[i], sims[i]):
            j = int(j)
            if sv >= tau and i < j and i in sets[j]:
                out.append((int(idx[i]), int(idx[j]), float(sv)))
    return out


def size_gated_edges(E, sizes, tau, k=10):
    blocks = defaultdict(list)
    for i, s in enumerate(sizes):
        blocks[tuple(s)].append(i)
    out = []
    for ids in blocks.values():
        out += mutual_knn_edges(E, tau, k, idx=ids)
    return out


def components(n, edge_lists):
    uf = UnionFind(n)
    for edges in edge_lists:
        for e in edges:
            uf.union(int(e[0]), int(e[1]))
    return np.array(uf.labels())


def key_edges(keys):
    """Chain edges between all items sharing the same key value."""
    first, out = {}, []
    for i, k in enumerate(keys):
        if k is None:
            continue
        if k in first:
            out.append((first[k], i))
        else:
            first[k] = i
    return out
