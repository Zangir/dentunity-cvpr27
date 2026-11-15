"""Aggregate every finished run into results/summary_*.json (the only files the paper
tables read). Safe to run at any time: missing runs are simply absent.
  python scripts/summarize.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentunity.cls_data import Y, rows
from dentunity.common import DATA, RESULTS, read_json, write_json
from dentunity.metrics import cluster_boot, macro_ap, macro_auroc, paired_ci, per_class_ap

J = lambda n: read_json(RESULTS / n) if (RESULTS / n).exists() else None


def det():
    out = {"provenance": "run (aggregated from results/yolo_*.json)", "runs": {}, "groups": {}}
    for p in sorted(RESULTS.glob("yolo_*.json")):
        r = read_json(p)
        out["runs"][r["run"]] = r
    groups = {}
    for name, r in out["runs"].items():
        g = name.rsplit("_s", 1)[0]
        groups.setdefault(g, []).append(r)
    members = {g: list(rs) for g, rs in groups.items()}
    for g, rs in members.items():
        e = {"seeds": [r["seed"] for r in rs], "n_seeds": len(rs)}
        for kind in ("box", "mask"):
            for split in ("val", "test"):
                vals = [r[split][kind] for r in rs if kind in r.get(split, {})]
                if not vals:
                    continue
                for m in ("map50_95", "map50"):
                    v = np.array([x[m] for x in vals])
                    e[f"{split}_{kind}_{m}_per_seed"] = v.tolist()
                    e[f"{split}_{kind}_{m}_mean"] = float(v.mean())
                    e[f"{split}_{kind}_{m}_min"] = float(v.min())
                    e[f"{split}_{kind}_{m}_max"] = float(v.max())
                    if len(v) > 1:
                        e[f"{split}_{kind}_{m}_std"] = float(v.std(ddof=1))
        groups[g] = e
    # long-tail split (PX tasks): class frequency bins from canonical TRAINING instances
    from collections import Counter
    from dentunity.taxonomy import PX_CLASSES
    b = read_json(DATA / "bench.json")["px"]
    tr = Counter(PX_CLASSES[x["c"]] for r in b if r["role"] == "canonical" and r["split"] == "train" for x in r["instances"])
    trs = Counter(PX_CLASSES[x["c"]] for r in b if r["role"] == "canonical" and r["split"] == "train" and r["in_seg"]
                  for x in r["instances"])
    BINS = {"head": (1000, 10 ** 9), "medium": (100, 1000), "tail": (0, 100)}
    out["px_train_instances"] = dict(tr)
    out["px_bins"] = {k: [c for c in PX_CLASSES if lo <= tr.get(c, 0) < hi] for k, (lo, hi) in BINS.items()}
    for g, rs in members.items():
        if not (g.startswith("pxdet") or g.startswith("pxseg") or g.startswith("pxleak") or g.startswith("pxdet_reexp")):
            continue
        kinds = ["mask", "box"] if g.startswith("pxseg") else ["box"]
        for kind in kinds:
            cnt = trs if g.startswith("pxseg") else tr
            for bn, (lo, hi) in BINS.items():
                per = []
                for r in rs:
                    pc = r["test"].get(kind, {}).get("per_class_ap50_95", {})
                    v = [a for c, a in pc.items() if lo <= cnt.get(c, 0) < hi]
                    if v:
                        per.append(float(np.mean(v)))
                if per:
                    groups[g][f"test_{kind}_{bn}_map50_95_mean"] = float(np.mean(per))
                    groups[g][f"test_{kind}_{bn}_n_classes"] = len([c for c in rs[0]["test"].get(kind, {}).get("per_class_ap50_95", {})
                                                                  if lo <= cnt.get(c, 0) < hi])
    out["groups"] = groups
    # L1 paired swap: leaky - clean-control per seed
    L = {}
    for s in (0, 1):
        a, b = out["runs"].get(f"pxleak_y11n_leaky_s{s}"), out["runs"].get(f"pxleak_y11n_cleanctl_s{s}")
        if a and b:
            for m in ("map50_95", "map50"):
                L.setdefault(m, []).append(a["test"]["box"][m] - b["test"]["box"][m])
                L.setdefault(f"{m}_leaky", []).append(a["test"]["box"][m])
                L.setdefault(f"{m}_clean", []).append(b["test"]["box"][m])
    if L:
        for m in ("map50_95", "map50"):
            L[f"{m}_diff_mean"] = float(np.mean(L[m]))
            L[f"{m}_rel_gain_mean"] = float(np.mean(np.array(L[m]) / np.array(L[f"{m}_clean"])))
        out["L1_swap"] = L
    write_json(RESULTS / "summary_det.json", out)
    return out


def cls():
    R = rows("split")
    yv, yt = Y(R["val"]), Y(R["test"])
    clusters = np.array([r["lineage"] for r in R["test"]])
    P, meta = {}, {}
    fr = J("cls_frozen.json")
    if fr:
        for name, m in fr["models"].items():
            z = np.load(DATA / "preds" / f"cls_{name}.npz")
            P[name] = (z["val"], z["test"])
            meta[name] = {"val_macro_ap": m["val_macro_ap"], "kind": "frozen"}
    for model in ("convnext_t", "vits14"):
        seeds = []
        for s in range(3):
            f = DATA / "preds" / f"cls_ft_{model}_s{s}.npz"
            if f.exists():
                z = np.load(f)
                P[f"ft_{model}_s{s}"] = (z["val"], z["test"])
                seeds.append(s)
        if seeds:
            P[f"ft_{model}_seedavg"] = tuple(np.mean([P[f"ft_{model}_s{s}"][k] for s in seeds], 0) for k in (0, 1))
            meta[f"ft_{model}_seedavg"] = {"seeds": seeds, "kind": "finetuned seed-average"}
    # C2: validation-weighted ensemble of FT ConvNeXt-T (seed avg) and the multi-encoder lattice
    if "ft_convnext_t_seedavg" in P and "multi_lattice" in P:
        a, b = P["ft_convnext_t_seedavg"], P["multi_lattice"]
        grid = [round(0.1 * i, 1) for i in range(11)]
        sc = {w: macro_ap(yv, w * a[0] + (1 - w) * b[0]) for w in grid}
        w = max(grid, key=lambda k: (sc[k], -abs(k - 0.5)))
        P["C2_ensemble"] = (w * a[0] + (1 - w) * b[0], w * a[1] + (1 - w) * b[1])
        meta["C2_ensemble"] = {"weight_ft_convnext": w, "val_grid": sc, "kind": "ensemble"}
    out = {"provenance": "run", "metric": "macro-AP", "n_test": len(yt), "n_test_lineages": int(len(set(clusters))),
           "models": {}}
    for k, (pv, pt) in P.items():
        out["models"][k] = dict(meta.get(k, {}), val_macro_ap=macro_ap(yv, pv), test_macro_ap=macro_ap(yt, pt),
                                test_macro_auroc=macro_auroc(yt, pt), test_per_class_ap=per_class_ap(yt, pt))
    for model in ("convnext_t", "vits14"):
        ks = [k for k in P if k.startswith(f"ft_{model}_s") and k[-1].isdigit()]
        if ks:
            v = np.array([out["models"][k]["test_macro_ap"] for k in ks])
            out[f"ft_{model}_test_macro_ap_mean"] = float(v.mean())
            out[f"ft_{model}_test_macro_ap_std"] = float(v.std(ddof=1)) if len(v) > 1 else None
            out[f"ft_{model}_test_macro_ap_per_seed"] = v.tolist()
    if P:
        keys = [k for k in P if not (k.startswith("ft_") and k[-1].isdigit())]
        ci, reps = cluster_boot(yt, clusters, {k: P[k][1] for k in keys}, n=1000)
        for k in keys:
            out["models"][k]["test_ci95"] = ci[k]
        best_val = max(keys, key=lambda k: out["models"][k]["val_macro_ap"])
        best_test = max(keys, key=lambda k: out["models"][k]["test_macro_ap"])
        out["best_by_val"], out["best_by_test"] = best_val, best_test
        out["lattice_minus_last"] = {a: {"diff": out["models"][f"{a}_lattice"]["test_macro_ap"] - out["models"][f"{a}_last_cls"]["test_macro_ap"],
                                         "ci95": paired_ci(reps, f"{a}_lattice", f"{a}_last_cls")}
                                     for a in ("vits14", "vitb14", "convnext_t") if f"{a}_lattice" in reps and f"{a}_last_cls" in reps}
        out["paired_ci"] = {f"{a}-minus-{b}": paired_ci(reps, a, b) for a in keys for b in keys
                            if a != b and a in (best_val, best_test, "C2_ensemble", "multi_lattice")}
    write_json(RESULTS / "summary_cls.json", out)
    return out


def leak_cls():
    """L2: models trained on the hash split vs the lineage split, scored on the test images
    both test sets share (identical evaluation images)."""
    fr_h, fr = J("cls_frozen_hash.json"), J("cls_frozen.json")
    if not (fr_h and fr):
        return None
    Rh, R = rows("split_hash"), rows("split")
    th = {r["sha256"]: i for i, r in enumerate(Rh["test"])}
    tl = {r["sha256"]: i for i, r in enumerate(R["test"])}
    common = sorted(set(th) & set(tl))
    ih, il = np.array([th[s] for s in common]), np.array([tl[s] for s in common])
    y = Y(R["test"])[il]
    out = {"provenance": "run", "n_common_test_images": len(common), "models": {}}
    for name in fr_h["models"]:
        if name not in fr["models"]:
            continue
        ph = np.load(DATA / "preds" / f"cls_hash_{name}.npz")["test"][ih]
        pl = np.load(DATA / "preds" / f"cls_{name}.npz")["test"][il]
        out["models"][name] = {"hash_split_full_test": fr_h["models"][name]["test_macro_ap"],
                               "lineage_split_full_test": fr["models"][name]["test_macro_ap"],
                               "common_test_trained_on_hash_split": macro_ap(y, ph),
                               "common_test_trained_on_lineage_split": macro_ap(y, pl)}
        m = out["models"][name]
        m["inflation_common_test"] = m["common_test_trained_on_hash_split"] - m["common_test_trained_on_lineage_split"]
    write_json(RESULTS / "summary_leak_cls.json", out)
    return out


if __name__ == "__main__":
    d = det()
    print({g: {k: round(v, 4) for k, v in e.items() if k.endswith("_mean") and "test" in k} for g, e in d["groups"].items()})
    c = cls()
    print({k: round(v["test_macro_ap"], 4) for k, v in c["models"].items()})
    l = leak_cls()
    if l:
        print({k: round(v["inflation_common_test"], 4) for k, v in l["models"].items()})
