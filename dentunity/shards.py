"""Sub-epoch sharding so every training chunk fits the 10-minute guarded slot.

Ultralytics checkpoints only at epoch ends. With --shards K, each logged epoch covers
one fixed 1/K shard of the training set (shards alternate in order), so K logged
epochs equal one full pass; the learning-rate schedule and warm-up run over the
logged epochs. The shard index is read from a small state file written by the main
process at each epoch start, so persistent dataloader workers follow it.
"""
from __future__ import annotations

import numpy as np
import torch


class ShardedDataset(torch.utils.data.Dataset):
    def __init__(self, ds, k: int, state_file: str, seed: int = 0):
        self.ds, self.k, self.state_file = ds, k, str(state_file)
        perm = np.random.default_rng(seed).permutation(len(ds.labels))
        self.shards = [np.sort(perm[i::k]) for i in range(k)]
        self.size = max(len(s) for s in self.shards)

    def __len__(self):
        return self.size

    def _part(self):
        try:
            with open(self.state_file) as f:
                return int(f.read().strip() or 0) % self.k
        except (OSError, ValueError):
            return 0

    def __getitem__(self, i):
        sh = self.shards[self._part()]
        return self.ds[int(sh[i % len(sh)])]

    def __getattr__(self, name):
        if name in ("ds", "k", "state_file", "shards", "size") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.ds, name)


def make_trainer(base_cls, k: int, state_file: str, seed: int):
    class ShardTrainer(base_cls):
        def build_dataset(self, img_path, mode="train", batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            return ShardedDataset(ds, k, state_file, seed) if (mode == "train" and k > 1) else ds

    return ShardTrainer
