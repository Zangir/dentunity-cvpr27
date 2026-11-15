"""Lineage-level stratified 70/10/20 splits and contamination audits."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .common import stable_seed

FRACS = {"train": 0.7, "val": 0.1, "test": 0.2}


def stratified_group_split(groups, group_labels, seed_text, fracs=FRACS, force_train=()):
    """Greedy iterative stratification over whole groups (lineages).

    group_labels: {gid: (n_images, {label: n_images_with_label})}. Groups that hold the
    rarest labels are placed first, each into the split with the largest remaining
    deficit for that label (ties: overall image deficit). force_train: groups that must
    go to training (very large lineages), placed before the greedy pass.
    """
    rng = np.random.default_rng(stable_seed(seed_text))
    gids = list(groups)
    rng.shuffle(gids)
    lab_tot, freq = Counter(), Counter()
    for g in gids:
        for c, k in group_labels[g][1].items():
            lab_tot[c] += k
            freq[c] += 1
    total = sum(group_labels[g][0] for g in gids)
    have = {s: Counter() for s in fracs}
    size = Counter()
    assign = {}

    def put(g, s):
        assign[g] = s
        size[s] += group_labels[g][0]
        for c, k in group_labels[g][1].items():
            have[s][c] += k
    for g in gids:
        if g in force_train:
            put(g, "train")

    def rarest(g):
        return min((freq[c] for c in group_labels[g][1]), default=10 ** 9)
    for g in sorted((g for g in gids if g not in assign), key=rarest):
        labs = group_labels[g][1]
        if labs:
            c = min(labs, key=lambda x: freq[x])
            deficit = {s: fracs[s] * lab_tot[c] - have[s][c] for s in fracs}
        else:
            deficit = {s: fracs[s] * total - size[s] for s in fracs}
        put(g, max(fracs, key=lambda s: (round(deficit[s], 6), fracs[s] * total - size[s])))
    return assign


def contamination(split_of_item, lineage_of_item, test_names=("test",), train_names=("train",)):
    """Share of test items whose lineage also has a member in train."""
    tr = {lineage_of_item[i] for i, s in enumerate(split_of_item) if s in train_names}
    te = [i for i, s in enumerate(split_of_item) if s in test_names]
    if not te:
        return None
    return float(np.mean([lineage_of_item[i] in tr for i in te]))
