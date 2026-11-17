"""Post-plan analyses requested in review (plan.json amendments of 2026-11-17 13:05): paired image
bootstrap intervals from per-image matches (data/preds/pi_*.pkl, written by scripts/perimage_eval.py).
Every part is computed only if its inputs exist; pending parts are recorded as pending.
  python3 scripts/postplan_analysis.py   -> results/postplan_review.json
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from perimage_eval import load, mapk  # noqa: E402

PRED = R / "data" / "preds"
PP = R / "results" / "postplan"
NB = 2000
out = {"provenance": "run (scripts/postplan_analysis.py; post-plan analyses, external review request)",
       "bootstrap_resamples": NB, "interval": "95% percentile, paired image bootstrap"}


def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def stats_by_im(path):
    d = load(path)
    return {s["im"]: s for s in d["stats"]}, d


# ---------------------------------------------------------------- A: IO-DET with / without conflict images
csha = set(json.loads((R / "data" / "conflict_test_sha.json").read_text()))
A = {}
for seed in (0, 1):
    p = PRED / f"pi_iodet_y11n_s{seed}.pkl"
    if not p.exists():
        continue
    st = load(p)["stats"]
    conf = np.array([s["im"][:16] in {h[:16] for h in csha} for s in st])
    keep = np.where(~conf)[0]
    full, excl = mapk(st), mapk(st, keep)
    rng = np.random.default_rng(1000 + seed)
    d95, d50 = [], []
    n = len(st)
    for _ in range(NB):
        idx = rng.integers(0, n, n)
        f = mapk(st, idx)
        e = mapk(st, idx[~conf[idx]])
        d95.append(f[0] - e[0])
        d50.append(f[1] - e[1])
    A[f"y11n_s{seed}"] = {"n_test_images": n, "n_conflict_images": int(conf.sum()),
                          "boxes_in_conflict_images": int(sum(len(st[i]["target_cls"]) for i in np.where(conf)[0])),
                          "all": {"map50_95": full[0], "map50": full[1]},
                          "without_conflict": {"map50_95": excl[0], "map50": excl[1]},
                          "diff_all_minus_without": {"map50_95": full[0] - excl[0], "map50": full[1] - excl[1],
                                                     "ci_map50_95": pct(d95), "ci_map50": pct(d50)}}
out["A_iodet_conflict"] = A


# ---------------------------------------------------------------- B(i)/(ii): DENTEX seeds, paired over the 50 images
def dentex_seeds(seeds, tag):
    pairs = []
    for s in seeds:
        c = R / "data/preds" / f"pi_{'' if s < 2 else 'pp_'}dentex_coco_y11s_s{s}.pkl"
        u = R / "data/preds" / f"pi_{'' if s < 2 else 'pp_'}dentex_du_y11s_s{s}.pkl"
        if c.exists() and u.exists():
            pairs.append((s, stats_by_im(c)[0], stats_by_im(u)[0]))
    if not pairs:
        return {"status": "pending"}
    ims = sorted(pairs[0][1])
    for _, c, u in pairs:
        assert sorted(c) == ims and sorted(u) == ims
    lst = [([c[i] for i in ims], [u[i] for i in ims]) for _, c, u in pairs]
    per = {}
    for (s, _, _), (c, u) in zip(pairs, lst):
        mc, mu = mapk(c), mapk(u)
        per[s] = {"coco": {"map50_95": mc[0], "map50": mc[1]}, "dentunity": {"map50_95": mu[0], "map50": mu[1]}}
    rng = np.random.default_rng(2000 + len(pairs))
    n = len(ims)
    bm, bm50, bs = [], [], {s: [] for s in per}
    for _ in range(NB):
        idx = rng.integers(0, n, n)
        ds, ds50 = [], []
        for (s, _, _), (c, u) in zip(pairs, lst):
            mc, mu = mapk(c, idx), mapk(u, idx)
            ds.append(mu[0] - mc[0])
            ds50.append(mu[1] - mc[1])
            bs[s].append(mu[0] - mc[0])
        bm.append(np.mean(ds))
        bm50.append(np.mean(ds50))
    for s in per:
        per[s]["diff_map50_95"] = per[s]["dentunity"]["map50_95"] - per[s]["coco"]["map50_95"]
        per[s]["ci_diff_map50_95"] = pct(bs[s])
    mean = lambda arm, k: float(np.mean([per[s][arm][k] for s in per]))
    return {"status": "done", "tag": tag, "seeds": sorted(per), "n_images": n, "per_seed": per,
            "mean": {"coco": {"map50_95": mean("coco", "map50_95"), "map50": mean("coco", "map50")},
                     "dentunity": {"map50_95": mean("dentunity", "map50_95"), "map50": mean("dentunity", "map50")}},
            "mean_diff_map50_95": mean("dentunity", "map50_95") - mean("coco", "map50_95"),
            "mean_diff_map50": mean("dentunity", "map50") - mean("coco", "map50"),
            "ci_mean_diff_map50_95": pct(bm), "ci_mean_diff_map50": pct(bm50),
            "seeds_with_dentunity_ahead": int(sum(per[s]["diff_map50_95"] > 0 for s in per))}


out["B_i_dentex_2seeds"] = dentex_seeds([0, 1], "existing seeds 0-1")
out["B_ii_dentex_5seeds"] = dentex_seeds([0, 1, 2, 3, 4], "seeds 0-4") if all(
    (PRED / f"pi_pp_dentex_{a}_y11s_s{s}.pkl").exists() for a in ("coco", "du") for s in (2, 3, 4)) else {
    "status": "pending", "finished_new_runs": sorted(p.stem for p in PRED.glob("pi_pp_dentex_*_y11s_s*.pkl"))}

# ---------------------------------------------------------------- B(iii): 5-fold CV over the 705 training images
cv = {}
for k in range(5):
    c, u = PRED / f"pi_pp_dentexcv{k}_coco_y11s_s0.pkl", PRED / f"pi_pp_dentexcv{k}_du_y11s_s0.pkl"
    if c.exists() and u.exists():
        sc, su = stats_by_im(c)[0], stats_by_im(u)[0]
        ims = sorted(sc)
        cv[k] = ([sc[i] for i in ims], [su[i] for i in ims])
if len(cv) == 5:
    per = {}
    for k, (c, u) in cv.items():
        mc, mu = mapk(c), mapk(u)
        per[k] = {"n_images": len(c), "coco": {"map50_95": mc[0], "map50": mc[1]},
                  "dentunity": {"map50_95": mu[0], "map50": mu[1]}, "diff_map50_95": mu[0] - mc[0],
                  "diff_map50": mu[1] - mc[1]}
    rng = np.random.default_rng(3000)
    bm = []
    for _ in range(NB):
        ds = []
        for k, (c, u) in cv.items():
            idx = rng.integers(0, len(c), len(c))
            ds.append(mapk(u, idx)[0] - mapk(c, idx)[0])
        bm.append(np.mean(ds))
    d = np.array([per[k]["diff_map50_95"] for k in per])
    from scipy import stats as SS
    h = SS.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    out["B_iii_dentex_cv5"] = {"status": "done", "per_fold": per,
                               "mean_coco_map50_95": float(np.mean([per[k]["coco"]["map50_95"] for k in per])),
                               "mean_dentunity_map50_95": float(np.mean([per[k]["dentunity"]["map50_95"] for k in per])),
                               "mean_diff_map50_95": float(d.mean()),
                               "mean_diff_map50": float(np.mean([per[k]["diff_map50"] for k in per])),
                               "ci_fold_stratified_bootstrap": pct(bm),
                               "ci_t_over_folds": [float(d.mean() - h), float(d.mean() + h)],
                               "folds_with_dentunity_ahead": int((d > 0).sum())}
else:
    out["B_iii_dentex_cv5"] = {"status": "pending", "folds_done": sorted(cv)}

# ---------------------------------------------------------------- C: RT-DETR-L on IO-DET
rj = PP / "yolo_pp_iodet_rtdetrl_s0.json"
st = PP / "rtdetr_status.json"
C = {"status": "pending"}
if rj.exists():
    r = json.loads(rj.read_text())
    C = {"status": "done", "val": r["val"]["box"], "test": r["test"]["box"], "batch": r["batch"],
         "epochs_full_passes": r["epochs_full_passes"], "logged_epochs_completed": r.get("logged_epochs_completed")}
    pr, py = PRED / "pi_pp_iodet_rtdetrl_s0.pkl", PRED / "pi_iodet_y11n_s0.pkl"
    if pr.exists():
        a, b = stats_by_im(pr)[0], stats_by_im(py)[0]
        ims = sorted(a)
        a, b = [a[i] for i in ims], [b[i] for i in ims]
        rng = np.random.default_rng(4000)
        dd = []
        for _ in range(NB):
            idx = rng.integers(0, len(ims), len(ims))
            dd.append(mapk(a, idx)[0] - mapk(b, idx)[0])
        C["diff_vs_y11n_s0_map50_95"] = mapk(a)[0] - mapk(b)[0]
        C["ci_diff_vs_y11n_s0_map50_95"] = pct(dd)
if st.exists():
    C["run_status"] = json.loads(st.read_text())
ml = R / "data" / "runs" / "pp_iodet_rtdetrl_s0" / "memlog.jsonl"
if ml.exists():
    rows = [json.loads(l) for l in ml.read_text().splitlines() if l.strip()]
    if rows:
        C["peak_mps_driver_gb"] = max(r.get("mps_driver_gb", 0) for r in rows)
        C["peak_rss_gb"] = max(r.get("maxrss_gb", 0) for r in rows)
        C["last_logged_epoch"] = max(r.get("epoch", 0) for r in rows)
out["C_rtdetr_iodet"] = C

# ---------------------------------------------------------------- A table + D are separate JSONs
out["A_types"] = json.loads((R / "results" / "label_conflict_types.json").read_text())
out["D_same_patient"] = json.loads((R / "results" / "px_same_patient_band.json").read_text())
(R / "results" / "postplan_review.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("A_types", "D_same_patient")}, indent=1)[:6000])
