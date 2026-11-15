#!/usr/bin/env python3
"""DentUnity entry point.

  python run_experiment.py --smoke            # 1-minute self-test on synthetic images (no data needed)
  python run_experiment.py build              # inventory -> lineages -> splits -> manifests (needs upstream data)
  python run_experiment.py queue              # one guarded chunk of the frozen experiment plan
  python run_experiment.py summarize          # aggregate results/*.json -> results/summary_*.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))


def smoke():
    """Synthetic end-to-end check of the copy-lineage split logic and the metrics:
    40 random 'source' images, each stored 1-3 times as resized / flipped / re-encoded
    copies; lineages from exact hash + a pixel-thumbnail embedding; stratified lineage
    split; the audit must report 0 contamination while a random file-level split leaks."""
    from PIL import Image
    from dentunity.common import write_json
    from dentunity.lineage import components, key_edges, mutual_knn_edges, null_quantile
    from dentunity.metrics import macro_ap
    from dentunity.splits import contamination, stratified_group_split
    rng = np.random.default_rng(0)
    out_dir = R / "sample_data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    files, src, labels = [], [], []
    for i in range(40):
        base = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
        base = np.kron(base, np.ones((4, 4, 1), np.uint8))           # 192x256 blocky image
        lab = int(rng.integers(0, 3))
        for k in range(int(rng.integers(1, 4))):
            im = Image.fromarray(base)
            if k == 1:
                im = im.resize((128, 96))
            if k == 2:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            p = out_dir / f"img{i:02d}_{k}.png"
            im.save(p)
            files.append(p); src.append(i); labels.append(lab)

    def emb(p):
        a = np.asarray(Image.open(p).convert("L").resize((16, 12)), np.float32).ravel()
        b = np.asarray(Image.open(p).convert("L").resize((16, 12)).transpose(Image.FLIP_LEFT_RIGHT), np.float32).ravel()
        e = a + b
        e -= e.mean()
        return e / (np.linalg.norm(e) + 1e-8)
    E = np.stack([emb(p) for p in files])
    tau = null_quantile(E, 0.95, n=4000)   # tiny set: random pairs often hit true copies
    lab = components(len(files), [mutual_knn_edges(E, tau, k=3), key_edges([p.read_bytes() for p in files])])
    groups = {}
    for i, g in enumerate(lab):
        groups.setdefault(int(g), []).append(i)
    gl = {g: (len(ms), {labels[ms[0]]: len(ms)}) for g, ms in groups.items()}
    assign = stratified_group_split(list(groups), gl, "smoke")
    split = [assign[int(g)] for g in lab]
    rand = list(np.random.default_rng(1).choice(["train", "val", "test"], len(files), p=[0.7, 0.1, 0.2]))
    true_lin = src
    y = np.eye(3)[labels]
    res = {"provenance": "run (synthetic smoke test)", "files": len(files), "sources": 40,
           "recovered_lineages": int(lab.max() + 1),
           "lineage_split_contamination_true_lineage": contamination(split, true_lin),
           "random_split_contamination_true_lineage": contamination(rand, true_lin),
           "macro_ap_perfect": macro_ap(y, y)}
    write_json(R / "results" / "smoke.json", res)
    print(json.dumps(res, indent=1))
    assert res["lineage_split_contamination_true_lineage"] == 0.0
    assert res["macro_ap_perfect"] == 1.0
    print("SMOKE OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default=None, choices=[None, "build", "queue", "summarize"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        return smoke()
    py = sys.executable
    if a.cmd == "build":
        for m in ("dentunity.inventory", "dentunity.build", "dentunity.identifier_audit"):
            subprocess.check_call([py, "-m", m], cwd=R)
    elif a.cmd == "queue":
        subprocess.check_call([py, "scripts/run_queue.py", "580"], cwd=R)
    elif a.cmd == "summarize":
        subprocess.check_call([py, "scripts/summarize.py"], cwd=R)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
