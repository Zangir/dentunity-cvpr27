"""Writes the appended supplement section 'Post-plan analyses requested in review'
(paper/supp/s9_postplan.tex: prose + tables) from results/postplan_review.json and
results/postplan/*.json. Touches no table, figure or macro used by the main paper.
  python3 scripts/make_postplan_tables.py ../../paper
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
PAPER = Path(sys.argv[1]) if len(sys.argv) > 1 else R.parents[1] / "paper"
P = json.loads((R / "results" / "postplan_review.json").read_text())
FOLDS = R / "results" / "postplan" / "dentex_cv_folds.json"

pc = lambda x, d=1: f"{100 * x:.{d}f}"
sg = lambda x, d=1: f"{'+' if x >= 0 else '$-$'}{abs(100 * x):.{d}f}"
ci = lambda c, d=1: f"[{sg(c[0], d)}, {sg(c[1], d)}]"
num = lambda n: f"{n:,}".replace(",", "{,}")
L = []
w = L.append

# ------------------------------------------------------------------ intro
w(r"""\section{Post-plan analyses requested in review}
\label{sec:supp_postplan}
The analyses in this section were requested by reviewers after the plan was frozen and after all
registered results were reported. Each one was added to the released plan
(\texttt{plan.json}) as a dated amendment before it was run, and each is reported whatever it
showed. They change no number, table or figure above. Intervals are 95\% percentile intervals
from a paired image bootstrap with 2{,}000 resamples. Each per-image recomputation reproduces
the reported mAP of its model exactly.
""")

# ------------------------------------------------------------------ A
A = P["A_types"]
T = A["exclusive_type_by_split"]
lab = {"count": "box count differs", "class": "same count, class differs",
       "iou<0.5": "same classes, a box with IoU $<0.5$", "iou0.5-0.9": "same classes, min IoU 0.5--0.9",
       "iou>=0.9": "same classes, all IoU $\\geq 0.9$"}
ne = A["nonexclusive_type_counts"]
cs = A["conflicting_groups_by_split"]
w(r"""
\paragraph{Label conflicts among exact copies in IO-DET.}
Of the """ + str(A["exact_duplicate_groups"]) + r""" groups of byte-identical files in the IO-DET source, """ + str(A["conflicting_groups"]) + r""" carry
different box lists on their copies (\cref{tab:pp_conflict}). We give each conflicting group one
type, checked in a fixed order over all pairs of its copies: the number of boxes differs; else
the class lists differ; else, after optimal same-class matching, some box has IoU below 0.5;
else the smallest matched IoU lies between 0.5 and 0.9. Most groups differ in the number of
boxes (""" + str(T["count"]["all"]) + r"""). Counting every type a group shows over its pairs, """ + str(ne["count"]) + r""" groups differ in count,
""" + str(ne["iou<0.5"]) + r""" contain a box matched below IoU 0.5 and """ + str(ne["class"]) + r""" differ in class. \name keeps one copy per group
(the one with the most boxes) and does not adjudicate the conflict. The kept copies fall into
the train, validation and test splits as """ + f"{cs['train']}/{cs['val']}/{cs['test']}" + r""".

\begin{table}[!htbp]
\centering\small
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Conflict types among the """ + str(A["conflicting_groups"]) + r""" IO-DET exact-duplicate groups with differing
labels}, by the \name split of the kept copy. Each group has one type, assigned in the row order.}
\label{tab:pp_conflict}
\begin{tabular}{lrrrr}
\toprule
Conflict type & train & val & test & all \\
\midrule
""")
for k in A["type_order"]:
    w(f"{lab[k]} & {T[k]['train']} & {T[k]['val']} & {T[k]['test']} & {T[k]['all']} \\\\\n")
w(r"\midrule" + "\n" + f"all conflicting groups & {cs['train']} & {cs['val']} & {cs['test']} & {A['conflicting_groups']} \\\\\n")
w(r"""\bottomrule
\end{tabular}
\end{table}
""")
AA = P["A_iodet_conflict"]
a0 = AA["y11n_s0"]
w(r"""
To see whether the """ + str(a0["n_conflict_images"]) + r""" conflicting test images move the IO-DET result, we evaluated the
validation-selected YOLO11n models (seed 0 is the released weight) on all """ + str(a0["n_test_images"]) + r""" test images and
on the """ + str(a0["n_test_images"] - a0["n_conflict_images"]) + r""" others, with no retraining (\cref{tab:pp_conflict_map}). The kept copies of
these images hold """ + str(a0["boxes_in_conflict_images"]) + r""" of the test boxes. Without them, mAP rises by less than one point
for both seeds, and every interval of the difference includes zero.

\begin{table}[!htbp]
\centering\small
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{IO-DET test mAP (\%) with and without the """ + str(a0["n_conflict_images"]) + r""" test images from conflicting
duplicate groups}, YOLO11n (validation-selected family), existing weights. $\Delta$ = all minus
without, with its 95\% paired image-bootstrap interval.}
\label{tab:pp_conflict_map}
\begin{tabular}{llccc}
\toprule
Seed & Metric & all """ + str(a0["n_test_images"]) + r""" & without """ + str(a0["n_conflict_images"]) + r""" & $\Delta$ [95\% CI] \\
\midrule
""")
for s in (0, 1):
    a = AA[f"y11n_s{s}"]
    dd = a["diff_all_minus_without"]
    for m, mk, ck in (("mAP$_{50:95}$", "map50_95", "ci_map50_95"), ("mAP$_{50}$", "map50", "ci_map50")):
        w(f"{s if mk == 'map50_95' else ''} & {m} & {pc(a['all'][mk])} & {pc(a['without_conflict'][mk])} & "
          f"{sg(dd[mk])} {ci(dd[ck])} \\\\\n")
w(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------------ B
B1, B2, B3 = P["B_i_dentex_2seeds"], P["B_ii_dentex_5seeds"], P["B_iii_dentex_cv5"]
BB = B2 if B2.get("status") == "done" else B1
w(r"""
\paragraph{DENTEX transfer: intervals, more seeds and cross-validation.}
\Cref{tab:supp_dentex} reports two seeds per initialisation. We add three analyses with the
same recipe (YOLO11s, 640 pixels, 40 epochs, batch 8, last checkpoint): a paired image-bootstrap
interval for the difference on the 50 evaluation images, three more seeds per initialisation,
and a 5-fold cross-validation over the 705 DENTEX training images. The \name-initialised runs
of seeds 2--4 start from the PX-DET YOLO11s checkpoint of seed $s \bmod 2$, because only two
such checkpoints exist. The new DENTEX weights are not released.
""")
txt_b1 = (r"With the two existing seeds, \name initialisation is ahead by " + sg(B1["mean_diff_map50_95"]) +
          r" mAP$_{50:95}$ points " + ci(B1["ci_mean_diff_map50_95"]) + r", an interval that includes zero, and by "
          + sg(B1["mean_diff_map50"]) + r" mAP$_{50}$ points " + ci(B1["ci_mean_diff_map50"]) + ".")
per = B1["per_seed"]
txt_b1 += (r" Per seed, the mAP$_{50:95}$ difference is " + sg(per["0"]["diff_map50_95"]) + " " + ci(per["0"]["ci_diff_map50_95"])
           + " for seed 0 and " + sg(per["1"]["diff_map50_95"]) + " " + ci(per["1"]["ci_diff_map50_95"]) + " for seed 1.")
w(txt_b1 + "\n")
if B2.get("status") == "done":
    w(r"With five seeds (\cref{tab:pp_dentex}), the mean difference is " + sg(B2["mean_diff_map50_95"]) + r" mAP$_{50:95}$ points "
      + ci(B2["ci_mean_diff_map50_95"]) + r" and " + sg(B2["mean_diff_map50"]) + r" mAP$_{50}$ points " + ci(B2["ci_mean_diff_map50"])
      + r"; \name initialisation is ahead in " + str(B2["seeds_with_dentunity_ahead"]) + r" of 5 seeds." + "\n")
else:
    w(r"Seeds 2--4 had not finished when this version was written; the table lists the finished seeds." + "\n")
if B3.get("status") == "done":
    w(r"Over the five folds, the mean mAP$_{50:95}$ difference is " + sg(B3["mean_diff_map50_95"]) + r" points ("
      + r"fold-stratified bootstrap " + ci(B3["ci_fold_stratified_bootstrap"]) + r", $t$ interval over folds "
      + ci(B3["ci_t_over_folds"]) + r"), and \name initialisation is ahead in " + str(B3["folds_with_dentunity_ahead"])
      + r" of 5 folds (\cref{tab:pp_dentex_cv})." + "\n")
else:
    w(r"The 5-fold cross-validation had not finished when this version was written." + "\n")
seeds = BB["seeds"]
w(r"""
\begin{table}[!htbp]
\centering\small
\setlength{\tabcolsep}{3pt}
\caption{\textbf{DENTEX, """ + str(len(seeds)) + r""" seeds per initialisation}: box mAP$_{50:95}$ (\%) on the 50 evaluation images and the
paired difference \name minus COCO with its 95\% image-bootstrap interval. Seeds 0--1 are the
runs of \cref{tab:supp_dentex}.}
\label{tab:pp_dentex}
\begin{tabular}{lccc}
\toprule
Seed & COCO & \name & $\Delta$ [95\% CI] \\
\midrule
""")
for s in seeds:
    r = BB["per_seed"][str(s)]
    w(f"{s} & {pc(r['coco']['map50_95'])} & {pc(r['dentunity']['map50_95'])} & {sg(r['diff_map50_95'])} {ci(r['ci_diff_map50_95'])} \\\\\n")
w(r"\midrule" + "\n")
w(f"mean & {pc(BB['mean']['coco']['map50_95'])} & {pc(BB['mean']['dentunity']['map50_95'])} & "
  f"{sg(BB['mean_diff_map50_95'])} {ci(BB['ci_mean_diff_map50_95'])} \\\\\n")
w(f"mean mAP$_{{50}}$ & {pc(BB['mean']['coco']['map50'])} & {pc(BB['mean']['dentunity']['map50'])} & "
  f"{sg(BB['mean_diff_map50'])} {ci(BB['ci_mean_diff_map50'])} \\\\\n")
w(r"""\bottomrule
\end{tabular}
\end{table}
""")
if B3.get("status") == "done":
    F = json.loads(FOLDS.read_text())["folds"]
    w(r"""
\begin{table}[!htbp]
\centering\small
\setlength{\tabcolsep}{3pt}
\caption{\textbf{DENTEX 5-fold cross-validation} over the 705 training images (train on 564, evaluate
the held-out 141 with the last checkpoint; seed 0). Box mAP$_{50:95}$ (\%); $\Delta$ = \name minus
COCO with its 95\% paired image-bootstrap interval.}
\label{tab:pp_dentex_cv}
\begin{tabular}{lcccc}
\toprule
Fold & boxes & COCO & \name & $\Delta$ \\
\midrule
""")
    for k in range(5):
        r = B3["per_fold"][str(k)]
        w(f"{k + 1} & {F[str(k)]['held_out_boxes']} & {pc(r['coco']['map50_95'])} & {pc(r['dentunity']['map50_95'])} & {sg(r['diff_map50_95'])} \\\\\n")
    w(r"\midrule" + "\n" + f"mean & & {pc(B3['mean_coco_map50_95'])} & {pc(B3['mean_dentunity_map50_95'])} & "
      f"{sg(B3['mean_diff_map50_95'])} {ci(B3['ci_fold_stratified_bootstrap'])} \\\\\n")
    w(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------------ C
C = P["C_rtdetr_iodet"]
w(r"""
\paragraph{RT-DETR-l on IO-DET.}
RT-DETR-l was planned for PX-DET and dropped because it exceeded the memory cap at batch 4
(\cref{sec:training}). On request we trained it on IO-DET, the smallest detection task, with the
YOLO11 protocol where the hardware allows: the same splits, 640 pixels, 40 epochs, COCO initialisation, the checkpoint with
the best validation fitness and one test evaluation, but batch 2 instead of 8 and full precision,
because mixed precision is not available on our accelerator and larger batches exceed its
memory cap. One seed was run.""")
if C.get("status") == "done":
    t, v = C["test"], C["val"]
    w(r""" It reaches """ + pc(t["map50_95"]) + r""" test mAP$_{50:95}$ and """ + pc(t["map50"]) + r""" mAP$_{50}$ (validation """
      + pc(v["map50_95"]) + r""" and """ + pc(v["map50"]) + r""").""")
    if "diff_vs_y11n_s0_map50_95" in C:
        w(r""" Against the released YOLO11n (seed 0) on the same test images, the difference is """
          + sg(C["diff_vs_y11n_s0_map50_95"]) + r""" mAP$_{50:95}$ points """ + ci(C["ci_diff_vs_y11n_s0_map50_95"]) + r""".""")
    if "peak_mps_driver_gb" in C:
        w(r""" Peak accelerator memory was """ + f"{C['peak_mps_driver_gb']:.1f}" + r"""\,GB.""")
    w(" It used batch 2, so it is not a like-for-like comparison with the YOLO11 runs at batch 8.\n")
elif (C.get("run_status") or {}).get("state") == "failed":
    rs = C["run_status"]
    w(r""" The run did not finish: it stopped at epoch """ + str(C.get("last_logged_epoch", "?")) + r""" of 40 ("""
      + rs.get("error_type", "") + r""")""" + (f", with a peak accelerator memory of {C['peak_mps_driver_gb']:.1f}\\,GB" if "peak_mps_driver_gb" in C else "")
      + r""". We therefore report no RT-DETR-l number.""" + "\n")
else:
    w(r""" The run had not finished when this version was written.""" + "\n")

# ------------------------------------------------------------------ D
D = P["D_same_patient"]
can = D["canonical"]
nm = D["mutual_top3_near_miss"]["by_band"]
sz = can["sizes"]
tt, vt = can["test-train"], can["val-train"]
share = lambda c, a, b, k: c[f"{a}-{b}"][k]["pairs"] / (sz[a] * sz[b])
w(r"""
\paragraph{Same-patient candidates across the panoramic splits.}
Copy links cannot detect two different radiographs of the same patient. As a bound on how
often such pairs could cross the splits, we counted cross-split pairs of canonical panoramic
images whose linking similarity lies just below the copy threshold, in $[0.95, 0.981)$
(\cref{tab:pp_patient}). On its own this band says little: radiographs look alike to the
descriptor, and """ + pc(share(can, "test", "train", "[0.95,tau)")) + r"""\% of all test--train pairs fall in it. Even above the
threshold, """ + pc(share(can, "test", "train", ">=tau"), 2) + r"""\% of the test--train pairs reach 0.981, close to the random-pair rate that sets the
threshold, and none of these pairs is a link, since links also require mutual nearest neighbours.
We therefore apply the link rule except for the threshold: a pair is a candidate if the two
images are mutual top-3 nearest neighbours with similarity in $[0.95, 0.981)$. Under this rule,
""" + str(nm["[0.95,tau)"]["test-train"]["test_copy_families_involved"]) + r""" of the """ + num(sz["test"]) + r""" canonical test images and """
  + str(nm["[0.95,tau)"]["val-train"]["val_copy_families_involved"]) + r""" of the """ + num(sz["val"]) + r""" validation images have a candidate
partner in training. These are candidates only. They may be the same patient at another visit,
or different patients with similar anatomy; no identity information or clinical check was used,
and we did not inspect the images.

\begin{table}[!htbp]
\centering\small
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{Cross-split panoramic pairs near the copy threshold} (canonical images; train """ + num(sz["train"]) + r""",
validation """ + num(sz["val"]) + r""", test """ + num(sz["test"]) + r"""). Band = similarity in $[0.95, 0.981)$. Candidates =
band pairs, over all files including re-exports, that are mutual top-3 nearest neighbours;
images = evaluation images (copy families, i.e.\ a canonical image with its re-exports)
with at least one candidate. Counts only; not confirmed same-patient pairs.}
\label{tab:pp_patient}
\begin{tabular}{lrrrr}
\toprule
Splits & band pairs & $\geq$0.981 pairs & candidates & images \\
\midrule
""")
for a, b in (("val", "train"), ("test", "train"), ("test", "val")):
    m = nm["[0.95,tau)"][f"{a}-{b}"]
    w(f"{a}--{b} & {num(can[f'{a}-{b}']['[0.95,tau)']['pairs'])} & {num(can[f'{a}-{b}']['>=tau']['pairs'])} & "
      f"{m['pairs_all_files']} & {m[f'{a}_copy_families_involved']} \\\\\n")
w(r"""\bottomrule
\end{tabular}
\end{table}
\FloatBarrier
""")
(PAPER / "supp" / "s9_postplan.tex").write_text("".join(L))
print("wrote", PAPER / "supp" / "s9_postplan.tex")
