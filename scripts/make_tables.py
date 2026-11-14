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
    def tk(v):
        return CM if v else XM
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
    tex = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{2.6pt}
\caption{\textbf{Comparison with public dental image datasets.} Mod.: P panoramic, I intraoral photograph, R other radiographs, C CBCT. Cls/Det/Inst: tasks stated by each paper. Copy-free: \cmark{} audited across splits, $\pm$ duplicates removed within one source, \xmark{} not reported, -- not checked. Quotes for every cell: supplementary material.}
\label{tab:compare}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcrcccrc}
\toprule
Dataset & Mod. & Images & Cls & Det & Inst & Classes & Copy-free \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    (OUT / "tab_compare.tex").write_text(tex)


def pct(x):
    return f"{100 * x:.1f}\\%"


def tab_audit():
    A = J("dataset_stats.json")["audit"]
    rows = [
        ("Shipped panoramic split", "test vs train", A["px_shipped_test_share_with_train_lineage"]),
        ("Shipped panoramic split", "val vs train", A["px_shipped_valid_share_with_train_lineage"]),
        ("Shipped IO-DET split", "val vs train", A["io_det_shipped_val_share_with_train_lineage"]),
        ("Hash-deduplicated IO-CLS split", "test vs train", A["io_cls_hash_split_test_share_with_train_lineage"]),
        ("Per-task splits, cross-task", "IO-DET val vs IO-CLS train", A["io_det_shipped_val_share_with_cls_train_lineage"]),
        ("Per-task splits, cross-task", "IO-CLS test vs IO-DET train", A["io_cls_test_share_with_shipped_det_train_lineage"]),
    ]
    body = "\n".join(f"{a} & {b} & {pct(v)} \\\\" for a, b, v in rows)
    ours = max(A["io_test_contamination"], A["io_val_contamination"], A["px_test_contamination"])
    tex = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Split audit.} Share of evaluation images whose copy lineage also occurs in the training data, for the shipped and earlier splits and for \name.}
\label{tab:audit}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llr}
\toprule
Split & Comparison & Leaked \\
\midrule
""" + body + r"""
\midrule
\rowcolor{oursbg}\name, every task & val/test vs any train & """ + pct(ours) + r""" \\
\bottomrule
\end{tabular}}
\end{table}
"""
    (OUT / "tab_audit.tex").write_text(tex)


def tab_taxonomy():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from unidental.taxonomy import UNIFIED
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


def tab_det():
    D = J("summary_det.json") if (R / "summary_det.json").exists() else {"groups": {}, "runs": {}}
    G = D["groups"]
    SH = J("shipped_checkpoint.json") if (R / "shipped_checkpoint.json").exists() else {}

    def cell(group, kind="box", m="map50_95"):
        g = G.get(group)
        if not g or f"test_{kind}_{m}_mean" not in g:
            return "XX"
        v = g[f"test_{kind}_{m}_per_seed"]
        s = f"{100*g[f'test_{kind}_{m}_mean']:.1f}"
        planned = 2 if "y11s" in group else 1
        if len(v) < planned:
            return s + "$^\\ast$"
        return s if len(v) == 1 else s + f"\\,{{\\scriptsize[{100*min(v):.1f}, {100*max(v):.1f}]}}"
    sh = lambda task, kind, m: f1(_get(SH, f"{task}_test", kind, m))
    rows = [
        f"YOLO11n & {cell('iodet_y11n')} & {cell('iodet_y11n', m='map50')} & {cell('pxdet_y11n')} & {cell('pxdet_y11n', m='map50')} & {cell('pxseg_y11n', 'mask')} & {cell('pxseg_y11n', 'mask', 'map50')} \\\\",
        f"YOLO11s (2 seeds) & {cell('iodet_y11s')} & {cell('iodet_y11s', m='map50')} & {cell('pxdet_y11s')} & {cell('pxdet_y11s', m='map50')} & {cell('pxseg_y11s', 'mask')} & {cell('pxseg_y11s', 'mask', 'map50')} \\\\",
        f"RT-DETR-l & -- & -- & {cell('pxdet_rtdetrl')} & {cell('pxdet_rtdetrl', m='map50')} & -- & -- \\\\",
        f"\\rowcolor{{oursbg}}YOLO11s + train re-exports (D1) & -- & -- & {cell('pxdet_reexp_y11s')} & {cell('pxdet_reexp_y11s', m='map50')} & -- & -- \\\\",
        "\\midrule",
        f"Shipped YOLOv8x-seg$^\\dagger$ & -- & -- & {sh('pxdet', 'box', 'map50_95')} & {sh('pxdet', 'box', 'map50')} & {sh('pxseg', 'mask', 'map50_95')} & {sh('pxseg', 'mask', 'map50')} \\\\",
    ]
    tex = r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Detection and segmentation results} on the copy-free test splits (\%, COCO mAP over classes present in test; boxes for IO-DET and PX-DET, masks for PX-SEG). Two-seed rows show the mean and [min, max] over seeds. $^\dagger$Checkpoint shipped with the panoramic release, trained on the shipped training split, which holds copies of XX\% of our PX-DET test images: its scores are inflated by construction and shown only for reference. $^\ast$Interim: fewer seeds than planned.}
\label{tab:det}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{IO-DET} & \multicolumn{2}{c}{PX-DET} & \multicolumn{2}{c}{PX-SEG (mask)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Model & mAP$_{50:95}$ & mAP$_{50}$ & mAP$_{50:95}$ & mAP$_{50}$ & mAP$_{50:95}$ & mAP$_{50}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    share = _get(SH, "pxdet_test_share_lineage_in_shipped_train")
    tex = tex.replace("copies of XX\\%", f"copies of {f1(share)}\\%")
    (OUT / "tab_det.tex").write_text(tex)


def tab_tail():
    D = J("summary_det.json") if (R / "summary_det.json").exists() else {"groups": {}, "px_bins": {}}
    G, B = D["groups"], D.get("px_bins", {})
    def c(group, kind, bn):
        return f1(_get(G, group, f"test_{kind}_{bn}_map50_95_mean"))
    rows = []
    for lab, gd, gs in (("YOLO11n", "pxdet_y11n", "pxseg_y11n"), ("YOLO11s", "pxdet_y11s", "pxseg_y11s"),
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
    from unidental.taxonomy import IO_CLS_CLASSES, IO_DET_CLASSES, PX_CLASSES
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
    cols = [("DET 11s s0", "pxdet_y11s_s0", "box"), ("DET 11s s1", "pxdet_y11s_s1", "box"),
            ("DET 11m s0", "pxdet_y11m_s0", "box"), ("DET 11n", "pxdet_y11n_s0", "box"),
            ("SEG 11s s0", "pxseg_y11s_s0", "mask"), ("SEG 11s s1", "pxseg_y11s_s1", "mask"),
            ("SEG 11m s0", "pxseg_y11m_s0", "mask")]
    for rn in sorted(runs):          # any further PX attempt (second YOLO11m seed, D1, RT-DETR-l)
        if rn.startswith(("pxdet_", "pxseg_")) and rn not in [c[1] for c in cols]:
            cols.append((rn.replace("pxdet_", "DET ").replace("pxseg_", "SEG ").replace("_", " "), rn,
                         "mask" if rn.startswith("pxseg") else "box"))
    SHp = J("shipped_checkpoint.json") if (R / "shipped_checkpoint.json").exists() else {}
    for task, kind in (("pxdet", "box"), ("pxseg", "mask")):
        pc = _get(SHp, f"{task}_test", kind, "per_class_ap50")
        if pc is not None:
            runs = dict(runs, **{f"shipped_{task}": {"test": {kind: {"per_class_ap50": pc}}}})
            cols.append((f"{task[2:].upper()} shipped$^\\dagger$", f"shipped_{task}", kind))
    lines = []
    for c in sorted(PX_CLASSES, key=lambda c: -tr.get(c, 0)):
        cells = []
        for _, rn, kind in cols:
            v = runs.get(rn, {}).get("test", {}).get(kind, {}).get("per_class_ap50")
            cells.append("XX" if v is None else (f1(v[c]) if c in v else "--"))
        lines.append(f"{c.replace('_', ' ')} & {tr.get(c, 0)} & " + " & ".join(cells) + " \\\\")
    t2 = (r"""\begin{table*}[t]
\centering\scriptsize
\caption{\textbf{Panoramic per-class test AP$_{50}$} (\%), classes sorted by PX-DET training instances (n). DET: PX-DET boxes, SEG: PX-SEG masks, 11n/11s/11m: YOLO11 sizes, s0/s1: seeds, $^\dagger$shipped checkpoint (trained on copies of most test images). --: class absent from that test split.}
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
        "rtdetrl": "RT-DETR-l", "shipped": r"Shipped YOLOv8x-seg$^\dagger$"}
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
    SH = _opt("shipped_checkpoint.json") or {}
    key = f"val_{kind}_map50_95_mean"
    cands = {f: e for f, e in F.items() if key in e and f"test_{kind}_map50_95_per_seed" in e}
    if not cands:
        return None
    sel = max(cands, key=lambda f: cands[f][key])
    s = cands[sel]
    others = {f: max(e[f"test_{kind}_map50_95_per_seed"]) for f, e in cands.items() if f != sel}
    sh = _get(SH, f"{task}_test", kind, "map50_95")
    if sh is not None:
        others["shipped"] = sh
    best_o = max(others, key=others.get) if others else None
    lo = min(s[f"test_{kind}_map50_95_per_seed"])
    missing = [f for f in REQUIRED[task] if f not in cands]
    if missing or s["n_seeds"] < SEEDS_FOR_CLAIM:
        rule = "pending"
    else:
        rule = "met" if best_o is None or lo > others[best_o] else "not met"
    return {"selected": sel, "sel": s, "best_other": best_o, "best_other_val": others.get(best_o),
            "min_seed": lo, "rule": rule, "missing": missing}


def verdict_cls():
    S = J("summary_cls.json") if (R / "summary_cls.json").exists() else None
    if not S or not S.get("best_by_val"):
        return None
    M = S["models"]
    sel = S["best_by_val"]
    keys = [k for k in M if k != sel and not (k.startswith("ft_") and k[-1].isdigit())]
    other = max(keys, key=lambda k: M[k]["test_macro_ap"])
    ci = S.get("paired_ci", {}).get(f"{sel}-minus-{other}")
    nft = min(len(S.get(f"ft_{m}_test_macro_ap_per_seed") or []) for m in ("convnext_t", "vits14"))
    if ci is None or nft < 3:
        rule = "pending"
    else:
        rule = "met" if ci[0] > 0 else "not met"
    return {"selected": sel, "m": M[sel], "best_other": other, "o": M[other], "ci": ci, "rule": rule, "n_ft_seeds": nft}


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
    SH = _opt("shipped_checkpoint.json") or {}
    planned = {"y11s": 2, "y11m": 2, "y11n": 1, "reexp_y11s": 1, "rtdetrl": 1}
    rows_spec = [("y11n", ("iodet", "pxdet", "pxseg")), ("y11s", ("iodet", "pxdet", "pxseg")),
                 ("y11m", ("pxdet", "pxseg")), ("reexp_y11s", ("pxdet",)), ("rtdetrl", ("pxdet",))]
    # best test mean per column (bold), shipped included
    best = {}
    for task, (kind, _) in TASKS.items():
        for m in ("map50_95", "map50"):
            vs = [e.get(f"test_{kind}_{m}_mean") for e in families(task).values()]
            vs.append(_get(SH, f"{task}_test", kind, m))
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
                if fam == "y11m" and not_selected(task, fam) and c != "XX":
                    c += "$^\\circ$"
                cells.append(c)
        lines.append(f"{NAME[fam]} & " + " & ".join(cells) + " \\\\")
    sh = []
    for task, (kind, _) in TASKS.items():
        for m in ("map50_95", "map50"):
            v = _get(SH, f"{task}_test", kind, m)
            if task == "iodet" or v is None:
                sh.append("--" if task == "iodet" else "XX")
            else:
                s = f"{100 * v:.1f}"
                sh.append("\\textbf{" + s + "}" if best[(task, m)] is not None and abs(v - best[(task, m)]) < 1e-12 else s)
    share_d = f1(_get(SH, "pxdet_test_share_lineage_in_shipped_train"))
    share_s = f1(_get(SH, "pxseg_test_share_lineage_in_shipped_train"))
    tex = r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4.5pt}
\caption{\textbf{Detection and segmentation on the copy-free test splits} (COCO mAP, \%, over the classes present in test). Bold: best mean per column. Two-seed cells give the mean and [min, max]. $^\ast$Fewer seeds than planned so far. $^\circ$One seed: not selected on validation, so no second seed (\cref{sec:protocol}). $^\dagger$Trained on the shipped split, which holds copies of """ + share_d + r"""\% (PX-DET) and """ + share_s + r"""\% (PX-SEG) of these test images.}
\label{tab:det}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{IO-DET (box)} & \multicolumn{2}{c}{PX-DET (box)} & \multicolumn{2}{c}{PX-SEG (mask)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Model & mAP$_{50:95}$ & mAP$_{50}$ & mAP$_{50:95}$ & mAP$_{50}$ & mAP$_{50:95}$ & mAP$_{50}$ \\
\midrule
""" + "\n".join(lines) + "\n\\midrule\n" + f"{NAME['shipped']} & " + " & ".join(sh) + r""" \\
\bottomrule
\end{tabular}
\end{table*}
"""
    (OUT / "tab_det.tex").write_text(tex)


def tab_cls2():
    S = J("summary_cls.json") if (R / "summary_cls.json").exists() else {"models": {}}
    M = S["models"]
    dims = {k: v.get("dims") for k, v in (J("cls_frozen.json") if (R / "cls_frozen.json").exists() else {"models": {}})["models"].items()}
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
        d = dims.get(key)
        d = f"{d:,}".replace(",", "{,}") if d else "--"
        return f"{label} & {d} & {s} & {ci} & {f1(m.get('test_macro_auroc'))} \\\\"
    rows = ["\\multicolumn{5}{l}{\\textit{Frozen encoder, logistic read-out per label}} \\\\",
            row("vitb14_last_cls", "ViT-B/14, last layer"), row("vitb14_dinov2_lin", "ViT-B/14, linear eval"),
            row("vitb14_lattice", "ViT-B/14, full-depth lattice"), row("vits14_lattice", "ViT-S/14, full-depth lattice"),
            row("convnext_t_lattice", "ConvNeXt-T, full-depth lattice"),
            "\\midrule\\multicolumn{5}{l}{\\textit{Fine-tuned (seed-averaged probabilities)}} \\\\",
            row("ft_convnext_t_seedavg", "ConvNeXt-T", "convnext_t"), row("ft_vits14_seedavg", "DINOv2 ViT-S/14", "vits14"),
            "\\midrule\\multicolumn{5}{l}{\\textit{Candidates declared in advance}} \\\\",
            row("multi_lattice", "C1: multi-encoder lattice"), row("C2_ensemble", "C2: FT ConvNeXt-T + C1")]
    tex = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{2.6pt}
\caption{\textbf{IO-CLS read-outs and models} (test macro-AP and macro-AUROC in \%, with 95\% intervals from a bootstrap over test lineages). Dim.: read-out feature size. Bold: best test macro-AP. All nine read-outs: supplementary Tab.~S3.}
\label{tab:cls}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrccc}
\toprule
Model & Dim. & macro-AP & 95\% CI & AUROC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    (OUT / "tab_cls.tex").write_text(tex)


def tab_sota():
    rows = []
    vc = verdict_cls()
    if vc:
        rows.append(f"IO-CLS & {CLSNAME.get(vc['selected'], vc['selected'])} & {f1(vc['m']['test_macro_ap'])} & "
                    f"{CLSNAME.get(vc['best_other'], vc['best_other'])} & {f1(vc['o']['test_macro_ap'])} & {vc['rule']} \\\\")
    else:
        rows.append("IO-CLS & XX & XX & XX & XX & pending \\\\")
    for task, (kind, lab) in TASKS.items():
        v = verdict_dense(task)
        if not v:
            rows.append(f"{lab} & XX & XX & XX & XX & pending \\\\"); continue
        rows.append(f"{lab} & {NAME[v['selected']]} ({v['sel']['n_seeds']}) & {f1(v['min_seed'])} & "
                    f"{NAME.get(v['best_other'], '--') if v['best_other'] else '--'} & {f1(v['best_other_val'])} & {v['rule']} \\\\")
    tex = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{2.4pt}
\caption{\textbf{State-of-the-art check per task} under the frozen rule (\cref{sec:protocol}). Selected: chosen on validation (seeds). Its score: test macro-AP for IO-CLS, the lowest seed's test mAP$_{50:95}$ otherwise. Strongest other: the best other evaluated model on test (its best seed). Rule: met, not met, or pending runs.}
\label{tab:sota}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llclcc}
\toprule
Task & Selected (val) & Score & Strongest other & Score & Rule \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
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
            lines.append(f"{lab} & {NAME.get(fam, fam)} & {r.get('seed', 'XX')} & {r.get('epochs_full_passes', 'XX')} & "
                         f"{r.get('batch', 'XX')} & {f1(v)} & {f1(te)} & {mark} \\\\")
        for fam in ("y11m",):
            if not_selected(task, fam):
                lines.append(f"{lab} & {NAME[fam]} & 1 & -- & -- & \\multicolumn{{2}}{{c}}{{not trained: seed 0 below YOLO11s on val}} & \\\\")
        lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines.pop()
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
    SE = _opt("subset_eval.json") or {"models": {}, "paired_no_copy_subset": {}}
    SH = _opt("shipped_checkpoint.json") or {}
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
    rows = []
    for task, (kind, lab) in (("pxdet", ("box", "PX-DET")), ("pxseg", ("mask", "PX-SEG"))):
        share = f1(_get(SH, f"{task}_test_share_lineage_in_shipped_train"))
        keys = [k for k in SE["models"] if k.endswith("|" + task)]
        keys = sorted(keys, key=lambda k: (not k.startswith("shipped"), k))
        if not keys:
            rows.append(f"{lab} & shipped / ours & XX & XX & XX & XX & XX \\\\"); continue
        for k in keys:
            e = SE["models"][k]
            name = "Shipped YOLOv8x-seg$^\\dagger$" if k.startswith("shipped") else k.split("|")[0].replace("_", "\\_")
            pr = SE.get("paired_no_copy_subset", {}).get(f"{k}-minus-shipped")
            prs = "--" if not pr else f"{100*pr['diff']:+.1f} [{100*pr['ci95'][0]:+.1f}, {100*pr['ci95'][1]:+.1f}]"
            rows.append(f"{lab} & {name} & {f1(e['full_test'])} & {f1(e['with_copy_subset'])} & {f1(e['no_copy_subset'])} "
                        f"[{f1(e['no_copy_ci95'][0])}, {f1(e['no_copy_ci95'][1])}] & {e['n_subset']} & {prs} \\\\")
    body2 = "\n".join(rows)
    tex = r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Leakage experiments} (detection mAP$_{50:95}$ and classification macro-AP, \%). Top: L1 keeps the test set and the training-set size fixed and swaps in copies of test images, L2 scores both models on the test images the two splits share, and L3 compares different evaluation images for scale. Bottom (L4 and L4b, post hoc and descriptive): the shipped checkpoint and our models on all copy-free test images, on those with a copy lineage in the shipped training split, and on those without, with image-bootstrap 95\% intervals. Last column: ours minus shipped on the no-copy images, paired bootstrap.}
\label{tab:supp_leak}
\begin{tabular}{llccc}
\toprule
Setting & Test images & copy-free train & leaky train & $\Delta$ \\
\midrule
""" + body1 + r"""
\bottomrule
\end{tabular}

\vspace{6pt}
\begin{tabular}{llccccc}
\toprule
Task & Model & all test & with copy & no copy [95\% CI] & $n$ no copy & ours $-$ shipped [95\% CI] \\
\midrule
""" + body2 + r"""
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
    SH = _opt("shipped_checkpoint.json") or {}
    SC = _opt("summary_cls.json") or {"models": {}}
    LC = _opt("summary_leak_cls.json") or {"models": {}}
    SE = _opt("subset_eval.json") or {"models": {}, "paired_no_copy_subset": {}}
    O = _opt("dentex_overlap.json") or {}
    mac, derived = {}, []

    def put(k, v, nd=1, scale=100):
        mac[k] = "XX" if v is None else f"{scale * v:.{nd}f}"

    def dput(k, a, b, op="-", nd=1, scale=100, signed=False):
        if a is None or b is None:
            mac[k] = "XX"; return
        v = (a - b) if op == "-" else (a - b) / b
        s = f"{scale * v:+.{nd}f}" if signed else f"{scale * v:.{nd}f}"
        mac[k] = s
        form = f"({a}-{b})*{scale}" if op == "-" else f"({a}-{b})/{b}*{scale}"
        derived.append({"value": abs(round(scale * v, nd)), "formula": form if v >= 0 else form.replace(f"({a}-{b})", f"({b}-{a})"),
                        "sources": [a, b], "macro": k})
    vc = verdict_cls()
    mac["clsSel"] = CLSNAME.get(vc["selected"], "XX") if vc else "XX"
    put("clsSelAP", vc and vc["m"]["test_macro_ap"]); put("clsSelAUROC", vc and vc["m"]["test_macro_auroc"])
    mac["clsVerdict"] = vc["rule"] if vc else "pending"
    mac["clsOther"] = CLSNAME.get(vc["best_other"], "XX") if vc else "XX"
    put("clsOtherAP", vc and vc["o"]["test_macro_ap"])
    ci = vc and vc.get("ci")
    mac["clsDiffCI"] = "XX" if not ci else f"[{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]"
    w = _get(SC["models"].get("C2_ensemble", {}), "weight_ft_convnext")
    mac["clsCTwoW"] = "XX" if w is None else f"{w:.1f}"
    mac["clsFtSeeds"] = str(vc["n_ft_seeds"]) if vc else "XX"
    put("lthreeOurs", _get(G.get("iodet_y11s", {}), "test_box_map50_95_mean"))
    for task, (kind, lab) in TASKS.items():
        v = verdict_dense(task)
        T = {"iodet": "ioDet", "pxdet": "pxDet", "pxseg": "pxSeg"}[task]
        mac[T + "Sel"] = NAME[v["selected"]] if v else "XX"
        put(T + "Map", v and v["sel"][f"test_{kind}_map50_95_mean"])
        put(T + "MapFifty", v and v["sel"][f"test_{kind}_map50_mean"])
        mac[T + "Verdict"] = v["rule"] if v else "pending"
        put(T + "Head", v and v["sel"].get(f"test_{kind}_head_map50_95_mean"))
        put(T + "Med", v and v["sel"].get(f"test_{kind}_medium_map50_95_mean"))
        put(T + "Tail", v and v["sel"].get(f"test_{kind}_tail_map50_95_mean"))
        mac[T + "TailN"] = str(v["sel"].get(f"test_{kind}_tail_n_classes", "XX")) if v else "XX"
        e_s, e_n = G.get(f"{task}_y11s"), G.get(f"{task}_y11n")
        dput(T + "SizeGain", e_s and e_s.get(f"test_{kind}_map50_95_mean"), e_n and e_n.get(f"test_{kind}_map50_95_mean"), signed=True)
        e_m = G.get(f"{task}_y11m")
        put(T + "Medium", e_m and e_m.get(f"test_{kind}_map50_95_mean"))
    put("shipDet", _get(SH, "pxdet_test", "box", "map50_95")); put("shipSeg", _get(SH, "pxseg_test", "mask", "map50_95"))
    put("shipDetShare", _get(SH, "pxdet_test_share_lineage_in_shipped_train"))
    put("shipSegShare", _get(SH, "pxseg_test_share_lineage_in_shipped_train"))
    L1 = D.get("L1_swap", {})
    put("leakAbs", L1.get("map50_95_diff_mean")); put("leakRel", L1.get("map50_95_rel_gain_mean"))
    put("leakClean", np.mean(L1["map50_95_clean"]) if L1.get("map50_95_clean") else None)
    put("leakLeaky", np.mean(L1["map50_95_leaky"]) if L1.get("map50_95_leaky") else None)
    mac["leakSeeds"] = str(len(L1.get("map50_95", []))) if L1 else "XX"
    m = LC["models"].get("vitb14_lattice", {})
    put("ltwoLin", m.get("common_test_trained_on_lineage_split")); put("ltwoHash", m.get("common_test_trained_on_hash_split"))
    mac["ltwoN"] = str(LC.get("n_common_test_images", "XX"))
    dput("ltwoGain", m.get("common_test_trained_on_hash_split"), m.get("common_test_trained_on_lineage_split"))
    r3 = D["runs"].get("iodet_shipped_y11s_s0")
    put("lthreeShip", _get(r3 or {}, "test", "box", "map50_95"))
    for task, T in (("pxdet", "Det"), ("pxseg", "Seg")):
        s = SE["models"].get(f"shipped_yolov8x_seg|{task}", {})
        put(f"lfourShipNo{T}", s.get("no_copy_subset")); put(f"lfourShipWith{T}", s.get("with_copy_subset"))
        mac[f"lfourN{T}"] = str(s.get("n_subset", "XX"))
        sel = (verdict_dense(task) or {}).get("selected")
        ours = [k for k in SE["models"] if k.startswith(f"{task}_{sel}_s") and k.endswith("|" + task)] if sel else []
        put(f"lfourOursNo{T}", np.mean([SE["models"][k]["no_copy_subset"] for k in ours]) if ours else None)
        prs = [SE.get("paired_no_copy_subset", {}).get(f"{k}-minus-shipped") for k in ours]
        prs = [p for p in prs if p]
        if prs:
            mac[f"lfourDiff{T}"] = ", ".join(f"{100*p['diff']:+.1f} [{100*p['ci95'][0]:+.1f}, {100*p['ci95'][1]:+.1f}]" for p in prs)
        else:
            mac[f"lfourDiff{T}"] = "XX"
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
    tab_audit()
    tab_cls2()
    tab_det2()
    tab_sota()
    supp_results()
    supp_attempts()
    supp_leak()
    supp_dentex()
    supp_tail()
    numbers()
    for old in ("tab_leak.tex",):          # moved to the supplement (Fig. 6 carries the main-paper view)
        if (OUT / old).exists():
            (OUT / old).unlink()
    print("tables ->", OUT)

