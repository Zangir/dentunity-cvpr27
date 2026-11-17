"""Post-plan analysis D (external review request): same-patient CANDIDATE bound for the panoramic split.
Counts cross-split panoramic pairs whose link-embedding cosine falls just below the copy threshold.
CPU only; counts only (no file names are printed or written). Candidates are not confirmed patients.
  python3 scripts/same_patient_band.py   -> results/px_same_patient_band.json
"""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
z = np.load(R / "data" / "linkemb_px.npz")
pos = {p: i for i, p in enumerate(z["paths"].tolist())}
E = z["E"].astype(np.float32)
px = json.loads((R / "data" / "bench.json").read_text())["px"]
tau = json.loads((R / "results" / "dataset_stats.json").read_text())["audit"]["px_tau_hi"]
BANDS = {"[0.90,0.95)": (0.90, 0.95), "[0.95,tau)": (0.95, tau), ">=tau": (tau, 2.0)}


def idx(split, canonical_only):
    return np.array([pos[r["path"]] for r in px if r["split"] == split and r["path"] in pos
                     and (r["role"] == "canonical" or not canonical_only)])


out = {"provenance": "run (scripts/same_patient_band.py over data/linkemb_px.npz + data/bench.json; "
                     "post-plan analysis D, external review request)",
       "tau_copy": tau, "bands": {k: [lo, (None if hi > 1 else hi)] for k, (lo, hi) in BANDS.items()},
       "note": "candidates only: high descriptor similarity below the copy threshold; no identity check"}
for scope, can in (("canonical", True), ("all_files", False)):
    I = {s: idx(s, can) for s in ("train", "val", "test")}
    res = {"sizes": {s: int(len(v)) for s, v in I.items()}}
    for a, b in (("val", "train"), ("test", "train"), ("test", "val")):
        S = E[I[a]] @ E[I[b]].T
        mx = S.max(1)
        d = {}
        for k, (lo, hi) in BANDS.items():
            m = (S >= lo) & (S < hi)
            d[k] = {"pairs": int(m.sum()), f"{a}_images_with_partner": int(m.any(1).sum()),
                    f"{b}_images_with_partner": int(m.any(0).sum())}
        d["max_sim_quantiles_q50_q90_q99_max"] = [round(float(x), 4) for x in np.quantile(mx, [0.5, 0.9, 0.99, 1.0])]
        d[f"{a}_images_whose_nearest_{b}_is_in_[0.95,tau)"] = int(((mx >= 0.95) & (mx < tau)).sum())
        res[f"{a}-{b}"] = d
    out[scope] = res
# near-miss links: the build links two files only if they are MUTUAL top-k neighbours (k=3) with
# cosine >= tau (unidental/build.py). Absolute cosine alone is weak for radiographs (they all look
# alike to the descriptor), so the candidate set is: cross-split mutual top-k pairs whose cosine is
# in [0.95, tau), i.e. pairs that miss a link only because of the threshold.
keep = [r for r in px if r["path"] in pos]
J = np.array([pos[r["path"]] for r in keep])
Ek = E[J]
sp = np.array([r["split"] for r in keep])
can = np.array([r["role"] == "canonical" for r in keep])
fam = np.array([r["family"] for r in keep])
K = 3
top = np.zeros((len(J), K), int)
topv = np.zeros((len(J), K), np.float32)
for s0 in range(0, len(J), 2048):
    S = Ek[s0:s0 + 2048] @ Ek.T
    S[np.arange(S.shape[0]), np.arange(s0, s0 + S.shape[0])] = -2
    t = np.argpartition(-S, K, 1)[:, :K]
    top[s0:s0 + 2048] = t
    topv[s0:s0 + 2048] = np.take_along_axis(S, t, 1)
nb = [set(r) for r in top]
mut = {}
for i in range(len(J)):
    for j, v in zip(top[i], topv[i]):
        if i < j and i in nb[j]:
            mut[(i, int(j))] = float(v)
nm = {}
for lo, hi, name in ((0.95, tau, "[0.95,tau)"), (0.90, 0.95, "[0.90,0.95)"), (tau, 2.0, ">=tau")):
    d = {}
    for a, b in (("val", "train"), ("test", "train"), ("test", "val")):
        pr = [(i, j) for (i, j), v in mut.items() if lo <= v < hi and {sp[i], sp[j]} == {a, b}]
        ev = {i if sp[i] == a else j for i, j in pr}
        d[f"{a}-{b}"] = {"pairs_all_files": len(pr), f"{a}_files_involved": len(ev),
                         f"{a}_canonical_images_involved": int(sum(can[i] for i in ev)),
                         "pairs_both_canonical": int(sum(can[i] and can[j] for i, j in pr)),
                         f"{a}_copy_families_involved": len({int(fam[i]) for i in ev}),
                         f"{b}_copy_families_involved": len({int(fam[j if sp[i] == a else i]) for i, j in pr})}
    d["within_split_pairs_all_files"] = int(sum(lo <= v < hi and sp[i] == sp[j] for (i, j), v in mut.items()))
    nm[name] = d
out["mutual_top3_near_miss"] = {"k": K, "mutual_pairs_total": len(mut), "by_band": nm,
    "note": "cross-split pairs at >= tau that are mutual top-3 would be copy links; the build merged them "
            "into one lineage, so this count must be 0 by construction"}
(R / "results" / "px_same_patient_band.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
