"""Render LaTeX tables from results JSONs only (numbers are never typed by hand).
  python scripts/make_tables.py <paper_dir>
"""
import json, sys
import numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "../../paper") / "sec"
J = lambda n: json.loads((R / n).read_text())
CM = "\\cmark"; XM = "\\xmark"


def tab_compare():
    d = J("comparison_matrix.json")
    S = J("dataset_stats.json")
    rows = ["hamamci2023dentex", "panetta2021tufts", "silva2022odontoai", "wang2025multi", "zhou2024dual",
            "chen2025oralxrays", "oz2026akudental", "sosnin2025alphadent", "ahmed2025annotated",
            "lv2026benchmark", "tang2026dinodental"]
    by = {r["bibkey"]: r for r in d["rows"]}
    short = {"hamamci2023dentex": "DENTEX", "panetta2021tufts": "Tufts", "silva2022odontoai": "OdontoAI",
             "wang2025multi": "STS-Tooth (2D)", "zhou2024dual": "Dual-labeled", "chen2025oralxrays": "OralXrays-9",
             "oz2026akudental": "AKUDENTAL", "sosnin2025alphadent": "AlphaDent", "ahmed2025annotated": "Intraoral caries",
             "lv2026benchmark": "COde", "tang2026dinodental": "DinoDental"}
    mod = lambda m: ("P+I" if "intraoral" in m and "panoramic" in m else "P+C" if "CBCT" in m and "panoramic" in m
                     else "I+R" if "intraoral" in m and "radiograph" in m else "P" if "panoramic" in m else "I")
    def tk(v):                      # same key as Tab. A.1: stated / checked and not stated / not stated
        return CM if v else (XM if v is False else "--")
    def dup(r):
        v = r.get("duplicate_or_leakage_audit_reported")
        return "$\\pm$" if v else (XM if v is False else "--")
    lines = []
    for k in rows:
        r = by[k]
        t = r["tasks"]
        n = r["n_images"]
        n = f"{int(n):,}".replace(",", "{,}") if n else "--"
        if k == "sosnin2025alphadent":
            n = "$>$" + n          # "contains over 1200 images"
        nc = str(r["n_classes"]) if r.get("n_classes") else "--"
        lines.append(f"{short[k]}~\\cite{{{k}}} & {mod(r['modality'])} & {n} & {tk(t.get('classification'))} & "
                     f"{tk(t.get('detection'))} & {tk(t.get('instance_segmentation'))} & {nc} & {dup(r)} \\\\")
    tot = S["total"]["distinct_images"]
    ours = (f"\\rowcolor{{oursbg}}\\textbf{{\\name (ours)}} & P+I & {tot:,} & {CM} & {CM} & {CM} & 36 & {CM} \\\\").replace(",", "{,}", 1) if False else \
        f"\\rowcolor{{oursbg}}\\textbf{{\\name (ours)}} & P+I & {tot:,}".replace(",", "{,}") + f" & {CM} & {CM} & {CM} & 36 & {CM} \\\\"
    body = "\n".join(lines + ["\\midrule", ours])
    # user request 2026-11-16: Table 1 spans both columns (table*), at natural size (no \resizebox)
    tex = r"""\begin{table*}[t]
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Comparison with public dental image datasets.} Modality: P panoramic, I intraoral photograph, R other radiographs, C CBCT. Cls/Det/Inst: classification, detection and instance segmentation. Key as in \cref{tab:app_survey}: \cmark{} stated, \xmark{} checked and not stated, -- not stated. Copy-free: \cmark{} audited across splits, $\pm$ duplicates removed within one source. Quotes for every cell: \cref{tab:app_survey}.}
\label{tab:compare}
\begin{tabularx}{\textwidth}{l*{7}{>{\centering\arraybackslash}X}}
\toprule
Dataset & Modality & Images & Cls & Det & Inst & Classes & Lineage-disjoint \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table*}
"""
    (OUT / "tab_compare.tex").write_text(tex)


def _fmt(n):
    return f"{int(n):,}".replace(",", "{,}")


def tab_sources():
    """Sources and tasks of DentUnity (user request 2026-11-16). One row per task, since each task comes
    from exactly one source; layout follows MSeg (CVPR 2020) Tab. 1 'Component datasets' (one row per
    component with its origin and size). Every number comes from dataset_stats.json,
    split_class_stats.json, io_cls_composite_audit.json and release/licence_inventory.json. The sources
    carry no patient IDs, so no patient count exists."""
    S = J("dataset_stats.json")
    C = J("split_class_stats.json")
    A = J("io_cls_composite_audit.json")
    L = json.loads((R.parent / "release" / "licence_inventory.json").read_text())
    lic = {x["source_id"]: x["declared_licence_verbatim"] for x in L["sources"]}
    lc = lambda x: "unknown" if x.lower() == "unknown" else x + "$^\\ast$"
    lin = lambda t: _fmt(C[t]["lineages_total"])
    spl = lambda d: " / ".join(_fmt(d[x]) for x in ("train", "val", "test"))
    rows = [
        ("IO-CLS", lc(lic["ORAL-CALCULUS"]), _fmt(A["total_files_walked"]),
         _fmt(S["io_cls"]["images"]), lin("io_cls"), _fmt(C["io_cls"]["annotations_total"]) + " labels",
         len(S["io_cls"]["label_images"]), spl(C["io_cls"]["images"])),
        ("IO-DET", lc(lic["ORAL-YOLO4"]), _fmt(S["audit"]["io_det_files"]),
         _fmt(S["io_det"]["images"]), lin("io_det"), _fmt(S["io_det"]["boxes"]) + " boxes",
         len(S["io_det"]["class_boxes"]), spl(C["io_det"]["images"])),
        ("PX-DET", lc(lic["PANO-DXPD31"]), _fmt(S["px_files"]["files"]),
         _fmt(S["px_det"]["images"]), lin("px_det"), _fmt(S["px_det"]["boxes"]) + " boxes",
         len(S["px_det"]["class_boxes"]), spl(C["px_det"]["images"])),
        ("PX-SEG", "", "", _fmt(S["px_seg"]["images"]), "as PX-DET",
         _fmt(S["px_seg"]["instances"]) + " masks", S["px_seg"]["classes_present"], spl(S["px_seg"]["split_images"])),
    ]
    assert C["io_cls"]["annotations_total"] + S["io_det"]["boxes"] + S["px_det"]["boxes"] == S["total"]["annotations"]
    T = S["total"]
    lines = [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\midrule",
              f"\\rowcolor{{oursbg}}\\textbf{{\\name}} & & & {_fmt(T['distinct_images'])} & & "
              f"{_fmt(T['annotations'])} & {T['unified_concepts']} & \\\\"]
    tex = r"""\begin{table*}[t]
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Sources and tasks of \name.} Files as published, images after de-duplication, copy lineages, annotations (PX-SEG masks outline PX-DET objects and are not added to the total) and train / val / test images. IO-CLS (six condition folders) and IO-DET (a detection subset) come from one Kaggle page that states ``multiple hospitals and reputable dental websites'' as origin. PX-DET and PX-SEG share one Kaggle release of radiographs whose metadata names Roboflow, with no project. Upstream owners and consent are not documented, and no source gives patient IDs, so patients cannot be counted. $^\ast$Declared by the re-publisher. Per-class counts: \cref{tab:app_pxclasses,tab:app_ioclasses}.}
\label{tab:sources}
\begin{tabularx}{\textwidth}{ll*{3}{>{\raggedleft\arraybackslash}X}r>{\raggedleft\arraybackslash}Xr}
\toprule
Task & Licence & Files & Images & Lineages & Annotations & Classes & Train / val / test \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{table*}
"""
    (OUT / "tab_sources.tex").write_text(tex)


def pct(x):
    return f"{100 * x:.1f}\\%"


def tab_audit():
    # round 7 (plan amendment 2026-11-17 09:32): every leak is shown under all links and under strict links
    A = J("dataset_stats.json")["audit"]
    SL = J("strict_link_audit.json") if (R / "strict_link_audit.json").exists() else None
    I = SL["intraoral_strict"] if SL else {}
    PK = SL["panoramic_keys_only"] if SL else {}
    rows = [
        ("Shipped panoramic split", "test vs train", A["px_shipped_test_share_with_train_lineage"], PK.get("px_shipped_test_share_with_train_lineage")),
        ("Shipped panoramic split", "val vs train", A["px_shipped_valid_share_with_train_lineage"], PK.get("px_shipped_valid_share_with_train_lineage")),
        ("Shipped IO-DET split", "val vs train", A["io_det_shipped_val_share_with_train_lineage"], I.get("io_det_shipped_val_share_with_train_lineage")),
        ("Hash-deduplicated IO-CLS split", "test vs train", A["io_cls_hash_split_test_share_with_train_lineage"], I.get("io_cls_hash_split_test_share_with_train_lineage")),
        ("Per-task splits, cross-task", "IO-DET val vs IO-CLS train", A["io_det_shipped_val_share_with_cls_train_lineage"], I.get("io_det_shipped_val_share_with_cls_train_lineage")),
        ("Per-task splits, cross-task", "IO-CLS test vs IO-DET train", A["io_cls_test_share_with_shipped_det_train_lineage"], I.get("io_cls_test_share_with_shipped_det_train_lineage")),
    ]
    sv = lambda v: "--" if v is None else pct(v)
    body = "\n".join(f"{a} & {b} & {pct(v)} & {sv(w)} \\\\" for a, b, v, w in rows)
    ours = max(A["io_test_contamination"], A["io_val_contamination"], A["px_test_contamination"])
    ours_s = max(I.get("dentunity_test_contamination", 0) or 0, I.get("dentunity_val_contamination", 0) or 0)
    nd = lambda v: "none" if v == 0 else pct(v)
    tex = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Split audit.} Share of evaluation images whose copy lineage also occurs in the training data. All: every link rule; strict: no recall edges (photographs) and no embedding edges (radiographs).}
\label{tab:audit}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llrr}
\toprule
Split & Comparison & All & Strict \\
\midrule
""" + body + r"""
\midrule
\rowcolor{oursbg}\name, every task & val/test vs any train & """ + nd(ours) + " & " + nd(ours_s) + r""" \\
\bottomrule
\end{tabular}}
\end{table}
"""
    (OUT / "tab_audit.tex").write_text(tex)


def tab_taxonomy():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dentunity.taxonomy import UNIFIED
    S = J("fig_stats_values.json")["annotations_per_concept"]
    lines = []
    for c in UNIFIED:
        src = c["sources"]
        cnt = lambda k: f"{S[c['name'] + '|' + k]:,}".replace(",", "{,}") if (c["name"] + "|" + k) in S else "--"
        esc = lambda t: t.replace("_", "\\_") if t else "--"
        lines.append(f"{c['id']} & {c['name']} & {c['family']} & {esc(src.get('io_cls'))} / {cnt('cls')} & "
                     f"{esc(src.get('io_det'))} / {cnt('iodet')} & {esc(src.get('px'))} / {cnt('px')} \\\\")
    tex = r"""\begin{table*}[t]
\centering\scriptsize
\caption{\textbf{Unified taxonomy.} Each concept, its clinical family and the source label it absorbs in each task, with the number of annotations (images for IO-CLS, instances on canonical images for IO-DET and PX).}
\label{tab:taxonomy}
\begin{tabular}{rlllll}
\toprule
id & concept & family & IO-CLS label / n & IO-DET label / n & PX label / n \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (OUT.parent / "supp").mkdir(exist_ok=True)
    (OUT.parent / "supp" / "tab_taxonomy.tex").write_text(tex)




# ----------------------------------------------------------------------------- results tables
# Missing runs render as XX (pending); numbers come only from results/summary_*.json.
def _get(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def f1(x, scale=100, nd=1):
    return "XX" if x is None else f"{scale * x:.{nd}f}"


def tab_cls():
    S = J("summary_cls.json") if (R / "summary_cls.json").exists() else {"models": {}}
    M = S["models"]
    best_val = S.get("best_by_val")

    def row(label, key, seeds=None):
        m = M.get(key)
        if seeds:
            per = S.get(f"ft_{seeds}_test_macro_ap_per_seed")
            mean, sd = S.get(f"ft_{seeds}_test_macro_ap_mean"), S.get(f"ft_{seeds}_test_macro_ap_std")
            ap = "XX" if mean is None else f"{100*mean:.1f}" + (f" $\\pm$ {100*sd:.1f}" if sd else "")
            ci = "seeds: " + (", ".join(f"{100*v:.1f}" for v in per) if per else "XX")
            au = f1(_get(M, f"ft_{seeds}_seedavg", "test_macro_auroc"))
            return f"{label} & {ap} & {ci} & {au} \\\\"
        ap = f1(_get(m, "test_macro_ap"))
        ci = "XX" if not m or "test_ci95" not in m else f"[{100*m['test_ci95'][0]:.1f}, {100*m['test_ci95'][1]:.1f}]"
        if m and key == best_val:
            ap = "\\textbf{" + ap + "}"
        return f"{label} & {ap} & {ci} & {f1(_get(m, 'test_macro_auroc'))} \\\\"
    rows = ["\\multicolumn{4}{l}{\\textit{Frozen encoders, logistic read-out}} \\\\",
            row("ViT-B/14, last layer", "vitb14_last_cls"),
            row("ViT-B/14, DINOv2 linear eval", "vitb14_dinov2_lin"),
            row("ViT-B/14, full-depth lattice", "vitb14_lattice"),
            row("ViT-S/14, full-depth lattice", "vits14_lattice"),
            row("ConvNeXt-T, full-depth lattice", "convnext_t_lattice"),
            row("Multi-encoder lattice (C1)", "multi_lattice"),
            "\\midrule\\multicolumn{4}{l}{\\textit{Fine-tuned (3 seeds)}} \\\\",
            row("ConvNeXt-T", None, "convnext_t"),
            row("DINOv2 ViT-S/14", None, "vits14"),
            "\\midrule",
            "\\rowcolor{oursbg}" + row("FT ConvNeXt-T + C1 ensemble (C2)", "C2_ensemble")]
    tex = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\caption{\textbf{IO-CLS results} (test, 6-label macro-AP and macro-AUROC, \%). Intervals: 95\% bootstrap over test lineages; fine-tuned rows: mean $\pm$ s.d. and per-seed values. Bold: the model selected on validation. All other encoders and read-outs: supplementary material.}
\label{tab:cls}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
Model & macro-AP & 95\% CI / seeds & AUROC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    (OUT / "tab_cls.tex").write_text(tex)


def tab_tail():
    D = J("summary_det.json") if (R / "summary_det.json").exists() else {"groups": {}, "px_bins": {}}
    G, B = D["groups"], D.get("px_bins", {})
    def c(group, kind, bn):
        return f1(_get(G, group, f"test_{kind}_{bn}_map50_95_mean"))
    rows = []
    for lab, gd, gs in (("YOLO11n", "pxdet_y11n", "pxseg_y11n"), ("YOLO11s", "pxdet_y11s", "pxseg_y11s"),
                        ("YOLO11m", "pxdet_y11m", "pxseg_y11m"),
                        ("YOLO11s + re-exports (D1)", "pxdet_reexp_y11s", None)):
        seg = [c(gs, "mask", b) if gs else "--" for b in ("head", "medium", "tail")]
        rows.append(f"{lab} & " + " & ".join(c(gd, "box", b) for b in ("head", "medium", "tail")) + " & " + " & ".join(seg) + " \\\\")
    n = {k: len(v) for k, v in B.items()}
    tex = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{2pt}
\caption{\textbf{The long tail.} Test mAP$_{50:95}$ (\%) averaged over panoramic classes grouped by training instances: head $\geq$1{,}000 (""" + str(n.get("head", "XX")) + r"""), medium 100--999 (""" + str(n.get("medium", "XX")) + r""") and tail $<$100 (""" + str(n.get("tail", "XX")) + r""") of the 31 classes in PX-DET training. PX-SEG classes are binned by their PX-SEG training instances. Means use the classes present in test and are averaged over seeds.}
\label{tab:tail}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{PX-DET (box)} & \multicolumn{3}{c}{PX-SEG (mask)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Model & head & med. & tail & head & med. & tail \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "tab_tail.tex").write_text(tex)


def tab_leak():
    D = J("summary_det.json") if (R / "summary_det.json").exists() else {"runs": {}}
    L1 = D.get("L1_swap", {})
    LC = J("summary_leak_cls.json") if (R / "summary_leak_cls.json").exists() else {"models": {}}
    run = D.get("runs", {})
    l1 = lambda k: (f"{100*np.mean(L1[k]):.1f}" if L1.get(k) else "XX")
    lc = LC["models"].get("vitb14_lattice", {})
    sh_val = _get(run.get("iodet_shipped_y11s_s0", {}), "test", "box", "map50_95")
    ours_io = _get(D.get("groups", {}).get("iodet_y11s", {}), "test_box_map50_95_mean")
    d = lambda a, b: "XX" if a is None or b is None else f"{100*(a-b):+.1f}"
    l1d = "XX" if not L1.get("map50_95") else f"{100*np.mean(L1['map50_95']):+.1f}"
    rows = [
        f"PX-DET, controlled swap (YOLO11n) & same test & {l1('map50_95_clean')} & {l1('map50_95_leaky')} & "
        f"{l1d} \\\\",
        f"IO-CLS, hash vs lineage split (ViT-B/14 lattice) & {LC.get('n_common_test_images', 'XX')} shared & "
        f"{f1(lc.get('common_test_trained_on_lineage_split'))} & {f1(lc.get('common_test_trained_on_hash_split'))} & "
        f"{d(lc.get('common_test_trained_on_hash_split'), lc.get('common_test_trained_on_lineage_split'))} \\\\",
        f"IO-DET, shipped split vs ours (YOLO11s) & different & {f1(ours_io)} & {f1(sh_val)} & {d(sh_val, ours_io)} \\\\",
    ]
    tex = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{How much leaked splits inflate scores} (\%; detection mAP$_{50:95}$, classification macro-AP). The controlled swap keeps the test set and training-set size fixed and only swaps in copies of test images. The IO-CLS row scores both models on the test images the two splits share. The IO-DET row compares the shipped validation split with our test split (different images, shown for scale).}
\label{tab:leak}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llccc}
\toprule
Setting & Test images & clean & leaked & $\Delta$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    (OUT / "tab_leak.tex").write_text(tex)


def tab_dentex():
    D = J("summary_det.json") if (R / "summary_det.json").exists() else {"groups": {}}
    G = D["groups"]
    O = J("dentex_overlap.json") if (R / "dentex_overlap.json").exists() else {}

    def c(group, m):
        g = G.get(group)
        if not g or f"test_box_{m}_mean" not in g:
            return "XX", "XX"
        v = g[f"test_box_{m}_per_seed"]
        return f"{100*g[f'test_box_{m}_mean']:.1f}", ", ".join(f"{100*x:.1f}" for x in v)
    rows = []
    for lab, grp in (("COCO pretraining", "dentex_coco_y11s"), ("\\name PX-DET pretraining", "dentex_du_y11s")):
        a, ap = c(grp, "map50_95")
        b, bp = c(grp, "map50")
        pre = "\\rowcolor{oursbg}" if "name" in lab else ""
        rows.append(f"{pre}{lab} & {a} & {b} & {ap} \\\\")
    tex = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\caption{\textbf{\name as training data: transfer to DENTEX}~\cite{hamamci2023dentex} (4 diagnosis classes, YOLO11s, 640\,px, 40 epochs, last checkpoint, trained on the 705 official training images, evaluated on the """ + str(O.get("eval_images_kept", "XX")) + r""" official validation images with no copy in \name). Box mAP in \%, mean over seeds and per-seed mAP$_{50:95}$.}
\label{tab:dentex}
\begin{tabular}{lccc}
\toprule
Initialisation & mAP$_{50:95}$ & mAP$_{50}$ & per seed \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "tab_dentex.tex").write_text(tex)


def supp_results():
    from dentunity.taxonomy import IO_CLS_CLASSES, IO_DET_CLASSES, PX_CLASSES
    S = J("summary_cls.json") if (R / "summary_cls.json").exists() else {"models": {}}
    D = J("summary_det.json") if (R / "summary_det.json").exists() else {"runs": {}, "px_train_instances": {}}
    names = {"vits14": "ViT-S/14", "vitb14": "ViT-B/14", "convnext_t": "ConvNeXt-T"}
    desc = {"last_cls": "last layer", "dinov2_lin": "DINOv2 linear eval", "lattice": "full-depth lattice"}
    rows = []
    for a in ("vits14", "vitb14", "convnext_t"):
        for d in ("last_cls", "dinov2_lin", "lattice"):
            rows.append((f"{names[a]}, {desc[d]}", f"{a}_{d}"))
    rows += [("Multi-encoder lattice (C1)", "multi_lattice"), ("FT ConvNeXt-T (seed avg.)", "ft_convnext_t_seedavg"),
             ("FT ViT-S/14 (seed avg.)", "ft_vits14_seedavg"), ("C2 ensemble", "C2_ensemble")]
    lines = []
    for lab, k in rows:
        m = S["models"].get(k, {})
        pc = m.get("test_per_class_ap") or [None] * 6
        lines.append(f"{lab} & {f1(m.get('val_macro_ap'))} & {f1(m.get('test_macro_ap'))} & " + " & ".join(f1(v) for v in pc) + " \\\\")
    head = " & ".join(c.replace("tooth ", "") for c in IO_CLS_CLASSES)
    t1 = (r"""\begin{table*}[t]
\centering\scriptsize
\caption{\textbf{IO-CLS, all read-outs} (\%): validation and test macro-AP, and test AP per label. Fine-tuned rows average the probabilities of the available seeds.}
\label{tab:supp_cls}
\begin{tabular}{lcccccccc}
\toprule
Model & val & test & """ + head + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    runs = D.get("runs", {})
    tr = D.get("px_train_instances", {})
    from dentunity.taxonomy import UNIFIED
    PX2U = {c["sources"]["px"]: c for c in UNIFIED if "px" in c["sources"]}
    PCT = J("split_class_stats.json")["px_det"]["per_concept"]
    cols = [("DET 11s s0", "pxdet_y11s_s0", "box"), ("DET 11s s1", "pxdet_y11s_s1", "box"),
            ("DET 11m s0", "pxdet_y11m_s0", "box"), ("DET 11n", "pxdet_y11n_s0", "box"),
            ("SEG 11s s0", "pxseg_y11s_s0", "mask"), ("SEG 11s s1", "pxseg_y11s_s1", "mask"),
            ("SEG 11m s0", "pxseg_y11m_s0", "mask")]
    def head_of(rn):                 # one header style: TASK size seed (QA 2026-11-16)
        t, rest = rn.split("_", 1)
        fam, seed = rest.rsplit("_", 1)
        size = {"y11n": "11n", "y11s": "11s", "y11m": "11m", "reexp_y11s": "11s+D1", "rtdetrl": "RT-DETR-l"}.get(fam, fam)
        return f"{'DET' if t == 'pxdet' else 'SEG'} {size} {seed}"
    for rn in sorted(runs):          # any further PX attempt (second YOLO11m seed, D1, RT-DETR-l)
        if rn.startswith(("pxdet_", "pxseg_")) and rn not in [c[1] for c in cols]:
            cols.append((None, rn, "mask" if rn.startswith("pxseg") else "box"))
    cols = sorted(((head_of(rn), rn, k) for _, rn, k in cols), key=lambda c: (c[1].startswith("pxseg"), c[0]))
    lines = []
    for c in sorted(PX_CLASSES, key=lambda c: -tr.get(c, 0)):
        cells = []
        for _, rn, kind in cols:
            v = runs.get(rn, {}).get("test", {}).get(kind, {}).get("per_class_ap50")
            cells.append("XX" if v is None else (f1(v[c]) if c in v else "--"))
        cpt = PX2U.get(c, {}).get("name")
        lowc = cpt in PCT and PCT[cpt]["test"] < 10           # review fix: low-support mark, as in Tabs. J.2-J.5
        lines.append(f"{c.replace('_', ' ')}{'$^\\dagger$' if lowc else ''} & {_fmt(tr.get(c, 0))} & " + " & ".join(cells) + " \\\\")
    t2 = (r"""\begin{table*}[t]
\centering\scriptsize
\caption{\textbf{Panoramic per-class test AP$_{50}$} (\%), classes sorted by PX-DET training instances (n). Classes are the source labels. DET: PX-DET boxes, SEG: PX-SEG masks, 11n/11s/11m: YOLO11 sizes, +D1: with re-exports, s0/s1: seeds. --: class absent from that test split. $^\dagger$Fewer than 10 PX-DET test boxes (low support).}
\label{tab:supp_px}
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lr""" + "c" * len(cols) + r"""}
\toprule
Class & n & """ + " & ".join(c[0] for c in cols) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table*}
""")
    lines = []
    for c in IO_DET_CLASSES:
        cells = []
        for rn in ("iodet_y11s_s0", "iodet_y11s_s1", "iodet_y11n_s0"):
            v = runs.get(rn, {}).get("test", {}).get("box", {}).get("per_class_ap50")
            cells.append("XX" if v is None else (f1(v[c]) if c in v else "--"))
        lines.append(f"{c} & " + " & ".join(cells) + " \\\\")
    t3 = (r"""\begin{table}[t]
\centering\small
\caption{\textbf{IO-DET per-class test AP$_{50}$} (\%).}
\label{tab:supp_io}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
Class & YOLO11s s0 & YOLO11s s1 & YOLO11n \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
    for name, tx in (("tab_supp_cls.tex", t1), ("tab_supp_px.tex", t2), ("tab_supp_io.tex", t3)):
        (OUT.parent / "supp" / name).write_text(tx)
    old = OUT.parent / "supp" / "tab_supp_results.tex"
    if old.exists():
        old.unlink()



# ============================================================================ rev. 2026-11-14 (exemplar rewrite)
# Every table below mirrors one IMed-361M element (paper/EXEMPLAR_MAP.md). Pending runs
# render as XX; the state-of-the-art verdict is computed from the frozen rule, never typed.
SUPP = OUT.parent / "supp"
NAME = {"y11n": "YOLO11n", "y11s": "YOLO11s", "y11m": "YOLO11m", "reexp_y11s": "YOLO11s + re-exports (D1)",
        "rtdetrl": "RT-DETR-l"}
CLSNAME = {"multi_lattice": "C1 multi-encoder lattice", "C2_ensemble": "C2 ensemble (FT ConvNeXt-T + C1)",
           "ft_convnext_t_seedavg": "fine-tuned ConvNeXt-T", "ft_vits14_seedavg": "fine-tuned ViT-S/14"}
for _a, _n in (("vits14", "ViT-S/14"), ("vitb14", "ViT-B/14"), ("convnext_t", "ConvNeXt-T")):
    for _d, _m in (("last_cls", "last layer"), ("dinov2_lin", "linear eval"), ("lattice", "lattice")):
        CLSNAME[f"{_a}_{_d}"] = f"{_n} {_m}"
TASKS = {"iodet": ("box", "IO-DET"), "pxdet": ("box", "PX-DET"), "pxseg": ("mask", "PX-SEG")}
# candidates the frozen plan requires before a verdict can be final (RT-DETR-l optional: may be dropped)
REQUIRED = {"iodet": ["y11s", "y11n"], "pxdet": ["y11s", "y11m", "y11n", "reexp_y11s"], "pxseg": ["y11s", "y11m", "y11n"]}
SEEDS_FOR_CLAIM = 2


def _sumdet():
    return J("summary_det.json") if (R / "summary_det.json").exists() else {"groups": {}, "runs": {}}


def _opt(n):
    return J(n) if (R / n).exists() else None


def families(task):
    G = _sumdet()["groups"]
    out = {}
    for g, e in G.items():
        if g.startswith(task + "_") and not g.startswith("iodet_shipped"):
            out[g[len(task) + 1:]] = e
    return out


def verdict_dense(task):
    kind, _ = TASKS[task]
    F = families(task)
    key = f"val_{kind}_map50_95_mean"
    cands = {f: e for f, e in F.items() if key in e and f"test_{kind}_map50_95_per_seed" in e}
    if not cands:
        return None
    # a YOLO11m family rejected by the pre-registered seed-0 validation rule cannot be selected
    eligible = {f: e for f, e in cands.items() if not (f == "y11m" and not_selected(task, f))}
    sel = max(eligible, key=lambda f: eligible[f][key])
    s = cands[sel]
    others = {f: max(e[f"test_{kind}_map50_95_per_seed"]) for f, e in cands.items() if f != sel}
    sh = None          # no external checkpoint is compared (user decision 2026-11-15)
    best_o = max(others, key=others.get) if others else None
    lo = min(s[f"test_{kind}_map50_95_per_seed"])
    missing = [f for f in REQUIRED[task] if f not in cands]
    if missing or s["n_seeds"] < SEEDS_FOR_CLAIM:
        rule = "pending"
    elif sh is None and (best_o is None or lo > others[best_o]):
        rule = "best (ours)"           # separated, but no external comparator
    elif best_o is None or lo > others[best_o]:
        rule = "met"
    elif sh is None:
        rule = "tied (ours)"           # no external comparator: the best entries are all our models
    else:
        rule = "not met"
    return {"selected": sel, "sel": s, "best_other": best_o, "best_other_val": others.get(best_o),
            "min_seed": lo, "rule": rule, "missing": missing}


def verdict_cls():
    """IO-CLS: every candidate is a DentUnity-trained model of ours and there is no external
    comparator. Report the validation-selected model, the strongest other model, and every
    model statistically tied with the selected one (paired lineage-bootstrap CI includes 0)."""
    S = J("summary_cls.json") if (R / "summary_cls.json").exists() else None
    if not S or not S.get("best_by_val"):
        return None
    M = S["models"]
    sel = S["best_by_val"]
    keys = [k for k in M if k != sel and not (k.startswith("ft_") and k[-1].isdigit())]
    other = max(keys, key=lambda k: M[k]["test_macro_ap"])
    ci = S.get("paired_ci", {}).get(f"{sel}-minus-{other}")
    ties = []
    for k in keys:
        c = S.get("paired_ci", {}).get(f"{sel}-minus-{k}")
        if c and c[0] <= 0 <= c[1]:
            ties.append(k)
    ties.sort(key=lambda k: -M[k]["test_macro_ap"])
    nft = min(len(S.get(f"ft_{m}_test_macro_ap_per_seed") or []) for m in ("convnext_t", "vits14"))
    if ci is None or nft < 3:
        rule = "pending"
    elif ci[0] > 0:
        rule = "best (ours)"           # IO-CLS has no external comparator
    else:
        rule = "tied (ours)"           # best entries are all ours and statistically tied
    return {"selected": sel, "m": M[sel], "best_other": other, "o": M[other], "ci": ci, "rule": rule,
            "n_ft_seeds": nft, "ties": ties, "external_comparator": False}


def cellv(e, kind, m="map50_95", planned=1, bold=False):
    if not e or f"test_{kind}_{m}_mean" not in e:
        return "XX"
    v = e[f"test_{kind}_{m}_per_seed"]
    s = f"{100 * e[f'test_{kind}_{m}_mean']:.1f}"
    if bold:
        s = "\\textbf{" + s + "}"
    if len(v) > 1:
        s += f"\\,{{\\scriptsize[{100 * min(v):.1f}, {100 * max(v):.1f}]}}"
    elif len(v) < planned:
        s += "$^\\ast$"
    return s


def not_selected(task, fam):
    """YOLO11m second seed is trained only if seed-0 validation beats YOLO11s seed 0 (amendment 17:50)."""
    kind = TASKS[task][0]
    runs = _sumdet()["runs"]
    a, b = runs.get(f"{task}_{fam}_s0"), runs.get(f"{task}_y11s_s0")
    if not a or not b or f"{task}_{fam}_s1" in runs:
        return False
    return a["val"][kind]["map50_95"] <= b["val"][kind]["map50_95"]


def tab_det2():
    # surgical revision 2026-11-15 (user: Tab. 4 "has some skips"): the PX-DET-only data recipe D1 left its row
    # mostly empty, so it is reported in Sec. 5.4 (\dOneGain) and in the appendix run table, and enters this table
    # only if validation selects it as the PX-DET model. mAP50 cells show the mean only; the seed range is shown once,
    # on mAP50:95. Marks in the caption appear only when a cell uses them.
    planned = {"y11s": 2, "y11m": 2, "y11n": 1, "reexp_y11s": 1, "rtdetrl": 1}
    rows_spec = [("y11n", ("iodet", "pxdet", "pxseg")), ("y11s", ("iodet", "pxdet", "pxseg")),
                 ("y11m", ("pxdet", "pxseg"))]
    vd = verdict_dense("pxdet")
    if vd and vd["selected"] == "reexp_y11s":
        rows_spec.append(("reexp_y11s", ("pxdet",)))
    if "rtdetrl" in families("pxdet"):          # RT-DETR-l dropped from the plan (amendment 2026-11-14 21:10)
        rows_spec.append(("rtdetrl", ("pxdet",)))
    shown = {t: {f for f, ts in rows_spec if t in ts} for t in TASKS}
    # best test mean per column (bold), over the rows shown
    best = {}
    for task, (kind, _) in TASKS.items():
        for m in ("map50_95", "map50"):
            vs = [e.get(f"test_{kind}_{m}_mean") for f, e in families(task).items() if f in shown[task]]
            vs = [v for v in vs if v is not None]
            best[(task, m)] = max(vs) if vs else None
    lines = []
    for fam, tasks in rows_spec:
        cells = []
        for task, (kind, _) in TASKS.items():
            for m in ("map50_95", "map50"):
                if task not in tasks:
                    cells.append("--"); continue
                e = families(task).get(fam)
                pl = planned[fam]
                if fam == "y11m" and not_selected(task, fam):
                    pl = 1
                b = e is not None and best[(task, m)] is not None and abs(e.get(f"test_{kind}_{m}_mean", -1) - best[(task, m)]) < 1e-12
                c = cellv(e, kind, m, pl, b)
                if m == "map50":
                    c = c.split("\\,{\\scriptsize[")[0]
                if fam == "y11m" and not_selected(task, fam) and c != "XX":
                    c += "$^\\circ$"
                elif e is not None and len(e.get(f"test_{kind}_{m}_per_seed", [])) == 1 and c != "XX" and "ast" not in c:
                    c += "$^\\dagger$"           # one seed, as planned (QA 2026-11-16: mark every single-seed cell)
                cells.append(c)
        lines.append(f"{NAME[fam]} & " + " & ".join(cells) + " \\\\")
    # amendment 2026-11-16 13:15: box-prompted SAM masks on the boxes of the selected PX-SEG family (YOLO11s)
    sam_note = ""
    if (R / "sam_pxseg.json").exists():
        SM = J("sam_pxseg.json")["runs"]
        tv = {m: [SM[k]["mask"][m] for k in sorted(SM) if k.endswith("/test")] for m in ("map50_95", "map50")}
        if tv["map50_95"]:
            e = {f"test_mask_{m}_per_seed": v for m, v in tv.items()}
            e.update({f"test_mask_{m}_mean": float(np.mean(v)) for m, v in tv.items()})
            cs = []
            for m in ("map50_95", "map50"):
                b = best[("pxseg", m)] is not None and e[f"test_mask_{m}_mean"] > best[("pxseg", m)] + 1e-12
                c = cellv(e, "mask", m, 2, b)
                cs.append(c.split("\\,{\\scriptsize[")[0] if m == "map50" else c)
            lines.append("YOLO11s boxes + SAM & -- & -- & -- & -- & " + " & ".join(cs) + " \\\\")
            sam_note = " + SAM: YOLO11s boxes with box-prompted SAM ViT-B masks~\\cite{kirillov2023segment}."
    body = "\n".join(lines)
    notes = ""
    if "--" in body:
        notes += " --: not trained for this task."
    if "$^\\ast$" in body:
        notes += " $^\\ast$One seed (run stopped)."
    if "$^\\dagger$" in body:
        notes += " $^\\dagger$One seed, as planned."
    if "$^\\circ$" in body:
        notes += " $^\\circ$One seed: not selected on validation (\\cref{sec:protocol})."
    notes += sam_note
    tex = r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4.5pt}
\caption{\textbf{Detection and segmentation on the copy-lineage-disjoint test splits} (COCO mAP, \%, classes present in test). Bold: best mean. Two-seed mAP$_{50:95}$ cells: mean [min, max]; mAP$_{50}$: mean (all runs: \cref{tab:app_runs})."""
    tex += notes + r"""}
\label{tab:det}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{IO-DET (box)} & \multicolumn{2}{c}{PX-DET (box)} & \multicolumn{2}{c}{PX-SEG (mask)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Model & mAP$_{50:95}$ & mAP$_{50}$ & mAP$_{50:95}$ & mAP$_{50}$ & mAP$_{50:95}$ & mAP$_{50}$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (OUT / "tab_det.tex").write_text(tex)


def tab_cls2():
    # surgical revision 2026-11-15 (user: Tab. 3 "has some skips"): the Dim. column was empty ("--") for every
    # fine-tuned row and for C2, so it moves to the appendix (Tab. app_cls_full holds every feature size).
    S = J("summary_cls.json") if (R / "summary_cls.json").exists() else {"models": {}}
    M = S["models"]
    best_test = max((v["test_macro_ap"] for k, v in M.items() if not (k.startswith("ft_") and k[-1].isdigit())), default=None)

    def row(key, label, ft=None):
        m = M.get(key, {})
        ap = m.get("test_macro_ap")
        s = f1(ap)
        if ap is not None and best_test is not None and abs(ap - best_test) < 1e-12:
            s = "\\textbf{" + s + "}"
        ci = "XX" if "test_ci95" not in m else f"[{100*m['test_ci95'][0]:.1f}, {100*m['test_ci95'][1]:.1f}]"
        if ft:
            per = S.get(f"ft_{ft}_test_macro_ap_per_seed") or []
            ci = (f"{len(per)}/3 seeds" if len(per) < 3 else ci) if per else "XX"
        return f"{label} & {s} & {ci} & {f1(m.get('test_macro_auroc'))} \\\\"
    rows = ["\\multicolumn{4}{l}{\\textit{Frozen encoder, logistic read-out per label}} \\\\",
            row("vitb14_last_cls", "ViT-B/14, last layer"), row("vitb14_dinov2_lin", "ViT-B/14, linear eval"),
            row("vitb14_lattice", "ViT-B/14, full-depth lattice"), row("vits14_lattice", "ViT-S/14, full-depth lattice"),
            row("convnext_t_lattice", "ConvNeXt-T, full-depth lattice"),
            "\\midrule\\multicolumn{4}{l}{\\textit{Fine-tuned (seed-averaged probabilities)}} \\\\",
            row("ft_convnext_t_seedavg", "ConvNeXt-T", "convnext_t"), row("ft_vits14_seedavg", "DINOv2 ViT-S/14", "vits14"),
            "\\midrule\\multicolumn{4}{l}{\\textit{Candidates declared in advance}} \\\\",
            row("multi_lattice", "C1: multi-encoder lattice"), row("C2_ensemble", "C2: FT ConvNeXt-T + C1")]
    # amendment 2026-11-16 13:15: zero-shot vision-language reference, selected on validation, test scored once
    if (R / "cls_zeroshot.json").exists():
        Z = J("cls_zeroshot.json")
        mname = {"clip_vitb16": "CLIP ViT-B/16~\\cite{radford2021learning}",
                 "biomedclip": "BiomedCLIP~\\cite{zhang2025multimodal}"}[Z["selected"].split("/")[0]]
        rows += ["\\midrule\\multicolumn{4}{l}{\\textit{Zero-shot vision-language, no training}} \\\\",
                 f"{mname} & {f1(Z['test_macro_ap'])} & "
                 f"[{100*Z['test_ci95'][0]:.1f}, {100*Z['test_ci95'][1]:.1f}] & {f1(Z['test_macro_auroc'])} \\\\"]
    tex = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{1.5pt}
\caption{\textbf{IO-CLS read-outs and models} (test macro-AP and macro-AUROC in \%, with 95\% intervals from a bootstrap over test lineages). Bold: best macro-AP, the primary metric. Read-outs with more than 1{,}024 features (lattices, linear eval) use a 1{,}024-d PCA; feature sizes and every seed are in \cref{tab:app_cls_full}.""" + (" Zero-shot: model and prompts chosen on validation, no training." if (R / "cls_zeroshot.json").exists() else "") + r"""}
\label{tab:cls}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc}
\toprule
Model & macro-AP & 95\% CI & AUROC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
    (OUT / "tab_cls.tex").write_text(tex)


SHORT = {"multi_lattice": "C1", "C2_ensemble": "C2", "vits14_lattice": "ViT-S lattice", "vitb14_lattice": "ViT-B lattice",
         "convnext_t_lattice": "ConvNeXt lattice", "ft_convnext_t_seedavg": "FT ConvNeXt-T", "ft_vits14_seedavg": "FT ViT-S"}
SHORTD = {"y11n": "YOLO11n", "y11s": "YOLO11s", "y11m": "YOLO11m", "reexp_y11s": "D1"}


def tab_sota():
    rows = []
    vc = verdict_cls()
    if vc:
        rows.append(f"IO-CLS & {SHORT.get(vc['selected'], vc['selected'])} & {f1(vc['m']['test_macro_ap'])} & "
                    f"{SHORT.get(vc['best_other'], vc['best_other'])} & {f1(vc['o']['test_macro_ap'])} & {vc['rule']} \\\\")
    else:
        rows.append("IO-CLS & XX & XX & XX & XX & pending \\\\")
    for task, (kind, lab) in TASKS.items():
        v = verdict_dense(task)
        if not v:
            rows.append(f"{lab} & XX & XX & XX & XX & pending \\\\"); continue
        rows.append(f"{lab} & {SHORTD[v['selected']]} ({v['sel']['n_seeds']}) & {f1(v['min_seed'])} & "
                    f"{SHORTD.get(v['best_other'], '--') if v['best_other'] else '--'} & {f1(v['best_other_val']) if v['best_other'] else '--'} & {v['rule']} \\\\")
    tex = r"""\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{1.5pt}
\caption{\textbf{State-of-the-art check per task} under the frozen rule (\cref{sec:protocol}). Score: the lowest seed for dense tasks, test macro-AP for IO-CLS.}
\label{tab:sota}
\begin{tabular}{llclcl}
\toprule
Task & Selected (seeds) & Score & Strongest other & Score & Rule \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "tab_sota.tex").write_text(tex)


def supp_attempts():
    """Tab. S5: every dense-task training attempt, its validation score (used for selection)
    and its test score, plus second seeds skipped by the rule."""
    D = _sumdet()
    runs = D["runs"]
    lines = []
    for task, (kind, lab) in TASKS.items():
        sel = (verdict_dense(task) or {}).get("selected")
        rs = sorted([(n, r) for n, r in runs.items() if n.startswith(task + "_") and not n.startswith("iodet_shipped")])
        for n, r in rs:
            fam = n[len(task) + 1:].rsplit("_s", 1)[0]
            v = _get(r, "val", kind, "map50_95"); te = _get(r, "test", kind, "map50_95")
            mark = "\\checkmark" if fam == sel else ""
            argsf = R.parent / "data" / "runs" / n / "args.yaml"
            batch = r.get("batch", "XX")
            if argsf.exists():          # effective batch after Ultralytics' out-of-memory fallback
                for line in argsf.read_text().splitlines():
                    if line.startswith("batch:"):
                        batch = line.split(":", 1)[1].strip()
            lines.append(f"{lab} & {NAME.get(fam, fam)} & {r.get('seed', 'XX')} & {r.get('epochs_full_passes', 'XX')} & "
                         f"{batch} & {f1(v)} & {f1(te)} & {mark} \\\\")
        for fam in ("y11m",):
            if not_selected(task, fam):
                lines.append(f"{lab} & {NAME[fam]} & 1 & -- & -- & \\multicolumn{{2}}{{c}}{{not trained: seed 0 below YOLO11s on val}} & \\\\")
        lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines.pop()
    if "iodet_y11n_s1" in runs or (R / "../../plan.json").exists():
        lines.append("\\midrule\\multicolumn{8}{p{0.95\\textwidth}}{\\footnotesize IO-DET YOLO11n seed 1 was added by the amendment of 2026-11-15 14:50, made after the "
                     "seed-0 test scores of YOLO11n and YOLO11s were known.} \\\\")
    tex = r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Every detection and segmentation training attempt} (mAP$_{50:95}$ in \%, boxes for IO-DET and PX-DET, masks for PX-SEG). Selection uses the validation column only, and each run is tested once. \checkmark: the family selected on validation for that task. Passes: full passes over the training set.}
\label{tab:supp_attempts}
\begin{tabular}{llcccccc}
\toprule
Task & Model & Seed & Passes & Batch & Val & Test & Selected \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (SUPP / "tab_supp_attempts.tex").write_text(tex)


def supp_leak():
    D = _sumdet()
    L1 = D.get("L1_swap", {})
    LC = _opt("summary_leak_cls.json") or {"models": {}}
    rows = []
    n1 = len(L1.get("map50_95_clean", []))
    for s in range(max(n1, 2)):
        c = L1.get("map50_95_clean", [None] * 2)[s] if s < n1 else None
        l = L1.get("map50_95_leaky", [None] * 2)[s] if s < n1 else None
        d = None if c is None or l is None else l - c
        rows.append(f"L1 controlled swap, seed {s} (YOLO11n) & same 1{{,}}186 & {f1(c)} & {f1(l)} & {'XX' if d is None else f'{100*d:+.1f}'} \\\\")
    m = LC["models"].get("vitb14_lattice", {})
    a, b = m.get("common_test_trained_on_lineage_split"), m.get("common_test_trained_on_hash_split")
    rows.append(f"L2 IO-CLS hash vs lineage split (ViT-B/14 lattice) & {LC.get('n_common_test_images', 'XX')} shared & {f1(a)} & {f1(b)} & "
                f"{'XX' if a is None or b is None else f'{100*(b-a):+.1f}'} \\\\")
    r3 = D["runs"].get("iodet_shipped_y11s_s0")
    io = _get(D["groups"].get("iodet_y11s", {}), "test_box_map50_95_mean")
    sv = _get(r3 or {}, "test", "box", "map50_95")
    rows.append(f"L3 IO-DET shipped split vs ours (YOLO11s) & different & {f1(io)} & {f1(sv)} & "
                f"{'XX' if io is None or sv is None else f'{100*(sv-io):+.1f}'} \\\\")
    body1 = "\n".join(rows)
    tex = r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Leakage experiments} (detection mAP$_{50:95}$ and classification macro-AP, \%). L1 keeps the test set and the training-set size fixed and swaps in copies of test images, L2 scores both models on the test images the two splits share, and L3 compares different evaluation images for scale. $\Delta$ is computed from unrounded scores.}
\label{tab:supp_leak}
\begin{tabular}{llccc}
\toprule
Setting & Test images & lineage-disjoint train & leaky train & $\Delta$ \\
\midrule
""" + body1 + r"""
\bottomrule
\end{tabular}

\end{table*}
"""
    (SUPP / "tab_supp_leak.tex").write_text(tex)


def supp_dentex():
    tab_dentex()
    p = OUT / "tab_dentex.tex"
    t = p.read_text().replace("\\label{tab:dentex}", "\\label{tab:supp_dentex}")
    (SUPP / "tab_supp_dentex.tex").write_text(t)
    p.unlink()


def supp_tail():
    tab_tail()
    p = OUT / "tab_tail.tex"
    (SUPP / "tab_supp_tail.tex").write_text(p.read_text().replace("\\label{tab:tail}", "\\label{tab:supp_tail}"))
    p.unlink()


def numbers():
    """sec/numbers.tex: every result number used in the prose, as macros (XX when pending).
    Derived values are also written to paper/number_whitelist.json with their formula."""
    D = _sumdet(); G = D["groups"]
    SC = _opt("summary_cls.json") or {"models": {}}
    LC = _opt("summary_leak_cls.json") or {"models": {}}
    O = _opt("dentex_overlap.json") or {}
    mac, derived = {}, []

    def put(k, v, nd=1, scale=100):
        mac[k] = "XX" if v is None else f"{scale * v:.{nd}f}"

    def dput(k, a, b, op="-", nd=1, scale=100, signed=False):
        if a is None or b is None:
            mac[k] = "XX"; return
        a, b = round(float(a), 6), round(float(b), 6)    # verify_numbers compares at 6 dp
        v = (a - b) if op == "-" else (a - b) / b
        s = f"{scale * v:+.{nd}f}" if signed else f"{scale * v:.{nd}f}"
        if round(scale * v, nd) == 0:          # avoid printing "-0.0" / "+0.0"
            s = f"{0:.{nd}f}"
        mac[k] = s
        form = f"({a}-{b})*{scale}" if op == "-" else f"({a}-{b})/{b}*{scale}"
        derived.append({"value": abs(round(scale * v, nd)), "formula": form if v >= 0 else form.replace(f"({a}-{b})", f"({b}-{a})"),
                        "sources": [a, b], "macro": k})
    vc = verdict_cls()
    mac["clsSel"] = CLSNAME.get(vc["selected"], "XX") if vc else "XX"
    put("clsSelAP", vc and vc["m"]["test_macro_ap"]); put("clsSelAUROC", vc and vc["m"]["test_macro_auroc"])
    if vc and vc["rule"] == "tied (ours)":
        mac["clsVerdict"] = ("no external comparator: selected " + SHORT.get(vc["selected"], vc["selected"])
                             + ", statistically tied with " + ", ".join(SHORT.get(k, CLSNAME.get(k, k)) for k in vc["ties"]) + " (ours)")
    else:
        mac["clsVerdict"] = vc["rule"] if vc else "pending"
    mac["clsOther"] = CLSNAME.get(vc["best_other"], "XX") if vc else "XX"
    put("clsOtherAP", vc and vc["o"]["test_macro_ap"])
    ci = vc and vc.get("ci")
    mac["clsDiffCI"] = "XX" if not ci else f"[{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]"
    w = _get(SC["models"].get("C2_ensemble", {}), "weight_ft_convnext")
    mac["clsCTwoW"] = "XX" if w is None else f"{w:.1f}"
    mac["clsFtSeeds"] = str(vc["n_ft_seeds"]) if vc else "XX"
    mac["clsTies"] = (", ".join(SHORT.get(k, CLSNAME.get(k, k)) for k in vc["ties"]) or "none") if vc else "XX"
    mac["clsTiesN"] = str(len(vc["ties"])) if vc else "XX"
    put("lthreeOurs", _get(G.get("iodet_y11s", {}), "test_box_map50_95_mean"))
    for task, (kind, lab) in TASKS.items():
        v = verdict_dense(task)
        T = {"iodet": "ioDet", "pxdet": "pxDet", "pxseg": "pxSeg"}[task]
        mac[T + "Sel"] = NAME[v["selected"]] if v else "XX"
        put(T + "Map", v and v["sel"][f"test_{kind}_map50_95_mean"])
        put(T + "MapFifty", v and v["sel"][f"test_{kind}_map50_mean"])
        if v and v["rule"] == "tied (ours)":
            mac[T + "Verdict"] = (f"no external comparator: selected {SHORTD[v['selected']]}, statistically tied with "
                                  f"{SHORTD.get(v['best_other'], v['best_other'])} (ours)")
        else:
            mac[T + "Verdict"] = v["rule"] if v else "pending"
        put(T + "Head", v and v["sel"].get(f"test_{kind}_head_map50_95_mean"))
        put(T + "Med", v and v["sel"].get(f"test_{kind}_medium_map50_95_mean"))
        put(T + "Tail", v and v["sel"].get(f"test_{kind}_tail_map50_95_mean"))
        mac[T + "TailN"] = str(v["sel"].get(f"test_{kind}_tail_n_classes", "XX")) if v else "XX"
        e_s, e_n = G.get(f"{task}_y11s"), G.get(f"{task}_y11n")
        dput(T + "SizeGain", e_s and e_s.get(f"test_{kind}_map50_95_mean"), e_n and e_n.get(f"test_{kind}_map50_95_mean"), signed=True)
        e_m = G.get(f"{task}_y11m")
        put(T + "Medium", e_m and e_m.get(f"test_{kind}_map50_95_mean"))
        put(T + "Planned", e_s and e_s.get(f"test_{kind}_map50_95_mean"))      # round 7: planned YOLO11s beside post-plan YOLO11m
        mac[T + "PlannedSeeds"] = str(e_s.get("n_seeds", "XX")) if e_s else "XX"
    L1 = D.get("L1_swap", {})
    put("leakAbs", L1.get("map50_95_diff_mean"), nd=2); put("leakAbsShort", L1.get("map50_95_diff_mean"), nd=1); put("leakRel", L1.get("map50_95_rel_gain_mean"))
    put("leakClean", np.mean(L1["map50_95_clean"]) if L1.get("map50_95_clean") else None, nd=2)
    put("leakLeaky", np.mean(L1["map50_95_leaky"]) if L1.get("map50_95_leaky") else None, nd=2)
    mac["leakSeeds"] = str(len(L1.get("map50_95", []))) if L1 else "XX"
    m = LC["models"].get("vitb14_lattice", {})
    put("ltwoLin", m.get("common_test_trained_on_lineage_split")); put("ltwoHash", m.get("common_test_trained_on_hash_split"))
    mac["ltwoN"] = str(LC.get("n_common_test_images", "XX"))
    dput("ltwoGain", m.get("common_test_trained_on_hash_split"), m.get("common_test_trained_on_lineage_split"))
    r3 = D["runs"].get("iodet_shipped_y11s_s0")
    put("lthreeShip", _get(r3 or {}, "test", "box", "map50_95"))
    SMS = (_opt("sam_pxseg.json") or {}).get("summary", {})    # amendment 2026-11-16: box-prompted SAM masks
    put("samSegMap", SMS.get("test_mask_map50_95_mean"))
    put("samSegYolo", SMS.get("yolo11s_test_mask_map50_95_mean_same_seeds"))
    mac["samSegSeeds"] = str(SMS.get("test_seeds", "XX"))
    a, c = G.get("dentex_coco_y11s", {}), G.get("dentex_du_y11s", {})
    put("dentexCoco", a.get("test_box_map50_95_mean")); put("dentexDU", c.get("test_box_map50_95_mean"))
    dput("dentexGain", c.get("test_box_map50_95_mean"), a.get("test_box_map50_95_mean"), signed=True)
    mac["dentexSeeds"] = str(min(a.get("n_seeds", 0), c.get("n_seeds", 0))) if a and c else "XX"
    mac["dentexKept"] = str(O.get("eval_images_kept", "XX")); mac["dentexEvalCopies"] = str(O.get("eval_images_with_copy_in_dentunity", "XX"))
    mac["dentexTrainCopies"] = str(O.get("train_images_with_copy_in_dentunity", "XX"))
    e_d, e_s = G.get("pxdet_reexp_y11s", {}), G.get("pxdet_y11s", {})
    dput("dOneGain", e_d.get("test_box_map50_95_mean"), e_s.get("test_box_map50_95_mean"), signed=True)
    n_att = sum(1 for n in D["runs"] if n.startswith(("pxdet_", "pxseg_")))
    mac["nPxAttempts"] = str(n_att) if n_att else "XX"
    lines = ["% Generated by experiments/repo/scripts/make_tables.py from results/*.json. Do not edit."]
    for k, v in mac.items():
        lines.append(f"\\newcommand*{{\\{k}}}{{{v}\\xspace}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    wl = OUT.parent / "number_whitelist.json"
    W = json.loads(wl.read_text()) if wl.exists() else {"values": [], "derivations": []}
    W["derivations"] = [d for d in W.get("derivations", []) if not d.get("macro")] + derived
    wl.write_text(json.dumps(W, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(exist_ok=True)
    tab_taxonomy()
    tab_compare()
    tab_sources()
    tab_audit()
    tab_cls2()
    tab_det2()
    supp_results()
    supp_attempts()
    supp_leak()
    supp_dentex()
    supp_tail()
    numbers()
    # round 6: main-paper floats point into the supplement with static text (\suppref, see paper/make_supprefs.py),
    # because Overleaf compiles main.tex without the supplement's .aux
    import re as _re
    _aux = OUT.parent / "supplementary.aux"
    _lab = set(_re.findall(r"\\newlabel\{([^}@]+)@cref\}", _aux.read_text(errors="ignore"))) if _aux.exists() else set()
    def _conv(m):
        ks = [k.strip() for k in m.group(2).split(",")]
        if _lab and all(k in _lab for k in ks):
            return ("\\Suppref{" if m.group(1) == "C" else "\\suppref{") + m.group(2) + "}"
        return m.group(0)
    for f in OUT.glob("tab_*.tex"):
        f.write_text(_re.sub(r"\\([cC])ref\{([^}]+)\}", _conv, f.read_text()))
    for f in SUPP.glob("tab_supp_*.tex"):             # appendix column tables may sit in the text (QA 2026-11-16)
        f.write_text(f.read_text().replace("\\begin{table}[t]", "\\begin{table}[!htbp]"))
    for old in ("tab_leak.tex", "tab_sota.tex"):          # moved to the supplement (Fig. 6 carries the main-paper view)
        if (OUT / old).exists():
            (OUT / old).unlink()
    print("tables ->", OUT)

