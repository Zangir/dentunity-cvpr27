"""Render the appendix tables of the paper from result files only (no typed numbers).
  python scripts/make_appendix_tables.py <paper_dir>
Writes <paper_dir>/supp/tab_app_*.tex. Reads results/*.json, never release images.
Missing runs are skipped (pending runs appear once their result file exists).
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dentunity.taxonomy import UNIFIED, IO_CLS_CLASSES, IO_DET_CLASSES, PX_CLASSES  # noqa: E402

R = ROOT / "results"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "../../paper") / "supp"
OUT.mkdir(exist_ok=True)


def J(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else None


def f1(x, scale=100, nd=1):
    return "--" if x is None else f"{scale * x:.{nd}f}"


def n(x):
    return "--" if x is None else f"{int(x):,}".replace(",", "{,}")


def esc(t):
    return (str(t).replace("\\", "").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#").replace("'", "'"))


def write(name, tex):
    # no \resizebox: fonts keep their nominal size (legibility); tables are sized to fit instead
    tex = tex.replace("\\resizebox{\\columnwidth}{!}{%\n", "").replace("\\end{tabular}}", "\\end{tabular}")
    # column tables may sit in the text (QA 2026-11-16: fewer near-empty float pages in the appendix)
    tex = tex.replace("\\begin{table}[t]", "\\begin{table}[!htbp]")
    (OUT / name).write_text(tex)


PX2U = {c["sources"]["px"]: c for c in UNIFIED if "px" in c["sources"]}
FAMS = {"pathology": "path.", "periodontal / soft tissue": "perio.", "tooth status / development": "tooth",
        "anatomy": "anat.", "restoration / device": "restor."}
MODEL = {"yolo11n.pt": "YOLO11n", "yolo11s.pt": "YOLO11s", "yolo11m.pt": "YOLO11m",
         "yolo11n-seg.pt": "YOLO11n-seg", "yolo11s-seg.pt": "YOLO11s-seg", "yolo11m-seg.pt": "YOLO11m-seg"}


# ----------------------------------------------------------------------------- survey of public datasets
def survey():
    d = J("comparison_matrix.json")
    def tk(v):
        return "\\cmark" if v else ("\\xmark" if v is False else "--")
    lines = []
    for r in d["rows"]:
        t = r["tasks"]
        lic = r.get("licence")
        lic = esc(lic) if lic else "--"
        name = esc(r["name"]).replace("(STS-2D-Tooth / STS-3D-Tooth)", "")
        ni = n(r.get('n_images'))
        if r["bibkey"] == "sosnin2025alphadent" and ni != "--":
            ni = "$>$" + ni          # "contains over 1200 images", as in Tab. 1
        lines.append(f"{name}~\\cite{{{r['bibkey']}}} & {esc(r['modality'])} & {ni} & "
                     f"{n(r.get('n_annotations_or_null'))} & {tk(t.get('classification'))} & {tk(t.get('detection'))} & "
                     f"{tk(t.get('instance_segmentation'))} & {tk(t.get('semantic_segmentation'))} & "
                     f"{n(r.get('n_classes'))} & {lic} & {tk(r.get('duplicate_or_leakage_audit_reported'))} & "
                     f"{tk(r.get('official_split'))} \\\\")
    write("tab_app_survey.tex", r"""\begin{table*}[t]
\centering\scriptsize
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{Survey of public dental image datasets} (built 2026-11-11 from the papers and official pages, every non-empty field backed by a quoted source in \texttt{results/comparison\_matrix.json}). CLS: classification, DET: detection, ISEG / SSEG: instance / semantic segmentation, Dup.: the paper reports duplicate removal, a leak check or a patient-level split. \cmark: stated, \xmark: checked and not stated, --: not stated. Licence is the verbatim data licence, never the article licence.}
\label{tab:app_survey}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{3.6cm}>{\raggedright\arraybackslash}p{2.5cm}rrccccr>{\raggedright\arraybackslash}Xcc}
\toprule
Dataset & Modality & Images & Annot. & CLS & DET & ISEG & SSEG & Classes & Licence & Dup. & Split \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{table*}
""")


# ----------------------------------------------------------------------------- dataset statistics
def splits():
    S = J("split_class_stats.json")
    D = J("dataset_stats.json")
    if S is None:
        return
    rows = [("IO-CLS", S["io_cls"], "labels"), ("IO-DET", S["io_det"], "boxes"), ("PX-DET", S["px_det"], "boxes")]
    lines = []
    for lab, b, unit in rows:
        sz = b["image_size"]
        lines.append(f"{lab} & {unit} & " + " & ".join(n(b['images'][s]) for s in ("train", "val", "test")) + " & "
                     + " & ".join(n(b['lineages'][s]) for s in ("train", "val", "test")) + " & "
                     + " & ".join(n(b['annotations'][s]) for s in ("train", "val", "test"))
                     + f" & {n(sz['width_median'])}$\\times${n(sz['height_median'])} \\\\")
    ps = D["px_seg"]
    lines.append("PX-SEG & masks & " + " & ".join(n(S['px_seg_images'][s]) for s in ("train", "val", "test"))
                 + " & \\multicolumn{3}{c}{as PX-DET} & "
                 + " & ".join(n(ps['split_instances'][s]) for s in ("train", "val", "test"))
                 + f" & {n(S['px_det']['image_size']['width_median'])}$\\times${n(S['px_det']['image_size']['height_median'])} \\\\")
    rx = S["px_reexports_by_split"]
    lines.append("PX re-exports & optional & " + " & ".join(n(rx[s]) for s in ("train", "val", "test"))
                 + " & \\multicolumn{3}{c}{inherited} & \\multicolumn{3}{c}{--} & "
                 + f"{n(S['px_reexport_image_size']['width_median'])}$\\times${n(S['px_reexport_image_size']['height_median'])} \\\\")
    write("tab_app_splits.tex", r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Split statistics per task}, counted from the released manifests: images, copy lineages and annotations in the training, validation and test splits, and the median image size (width$\times$height in pixels). PX-SEG uses the radiographs of PX-DET that carry polygon outlines, so its lineages are those of PX-DET. Re-exported panoramic files inherit the split of their lineage and are optional training data; those that fall in validation or test lineages are never used.}
\label{tab:app_splits}
\begin{tabular}{llccccccccccc}
\toprule
 & & \multicolumn{3}{c}{images} & \multicolumn{3}{c}{lineages} & \multicolumn{3}{c}{annotations} & \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
Task & unit & train & val & test & train & val & test & train & val & test & median size \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def px_classes():
    S = J("split_class_stats.json")
    D = J("dataset_stats.json")
    if S is None:
        return
    pc = S["px_det"]["per_concept"]
    ci = D["px_det"]["class_images"]
    lines = []
    for src in sorted(PX_CLASSES, key=lambda c: -D["px_det"]["class_boxes"][c]):
        u = PX2U[src]
        v = pc.get(u["name"], {"train": 0, "val": 0, "test": 0})
        lines.append(f"{u['id']} & {u['name']} & {FAMS[u['family']]} & {n(D['px_det']['class_boxes'][src])} & "
                     f"{n(ci[src])} & {n(v['train'])} & {n(v['val'])} & {n(v['test'])} \\\\")
    write("tab_app_pxclasses.tex", r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{2pt}
\caption{\textbf{Panoramic concepts.} Unified id and name, family, boxes and images on the 5{,}929 canonical radiographs, and boxes per split, sorted by boxes. Source labels are in \cref{tab:taxonomy}.}
\label{tab:app_pxclasses}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{rllrrrrr}
\toprule
id & concept & fam. & boxes & images & train & val & test \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


def io_classes():
    S = J("split_class_stats.json")
    A = J("io_cls_composite_audit.json")
    L = J("io_cls_labels_audit.json")
    if S is None:
        return
    U = {c["sources"].get("io_cls"): c for c in UNIFIED}
    V = {c["sources"].get("io_det"): c for c in UNIFIED}
    lines = []
    for lab in IO_CLS_CLASSES:
        v = S["io_cls"]["per_concept"][U[lab]["name"]]
        lines.append(f"IO-CLS & {lab} & {U[lab]['id']} & {n(v['train'])} & {n(v['val'])} & {n(v['test'])} \\\\")
    lines.append("\\midrule")
    for lab in IO_DET_CLASSES:
        v = S["io_det"]["per_concept"][V[lab]["name"]]
        lines.append(f"IO-DET & {lab} & {V[lab]['id']} & {n(v['train'])} & {n(v['val'])} & {n(v['test'])} \\\\")
    pairs = sorted(A["pairwise_overlap_counts"].items(), key=lambda kv: -kv[1])
    pl = " ".join(f"{k.replace('&', ' \\& ')} ({n(c)})," for k, c in pairs).rstrip(",")
    hist = L["label_count_hist"]
    hl = ", ".join(f"{k} label{'s' if k != '1' else ''}: {n(hist[k])}" for k in ("1", "2", "3", "4"))
    write("tab_app_ioclasses.tex", r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Intraoral labels per split.} IO-CLS counts images that carry a label (an image can carry several), IO-DET counts boxes. Folder pairs (files): """ + pl + r""". Labels per IO-CLS image: """ + hl + r""".}
\label{tab:app_ioclasses}
\begin{tabular}{llrrrr}
\toprule
Task & source label & id & train & val & test \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ----------------------------------------------------------------------------- linking QC
def linkqc():
    Q = J("fig_linkqc_values.json")
    A = J("dataset_stats.json")["audit"]
    Dx = J("dentex_overlap.json")
    sm = J("smoke.json")
    rows = [
        ("Panoramic", "embedding threshold (99.9th pct. of random pairs)", f"{A['px_tau_hi']:.3f}"),
        ("Panoramic", "random pairs sampled", n(Q["px_random_pairs"])),
        ("Panoramic", "random pairs above threshold", f"{100 * Q['px_random_share_above_tau']:.2f}\\%"),
        ("Panoramic", "re-export pairs", n(Q["px_reexport_pairs"])),
        ("Panoramic", "re-export pairs above threshold", f"{100 * Q['px_reexport_share_above_tau']:.1f}\\%"),
        ("Panoramic", "median similarity of a re-export pair", f"{Q['px_reexport_median']:.4f}"),
        ("Panoramic", "embedding edges added", n(A["px_embedding_edges"])),
        ("Panoramic", "registered cross-stem copies", n(A["px_registered_links"])),
        ("Panoramic", "families from hash, source key and registration", n(A["px_lineages_source_hash_registered"])),
        ("Panoramic", "copy families with a usable file", n(A["px_copy_families_used"])),
        ("Panoramic", "lineages after embedding edges (all files)", n(A["px_lineages_final"])),
        ("Panoramic", "lineages with a usable radiograph (released)", n(J("split_class_stats.json")["px_det"]["lineages_total"])),
        ("Intraoral", "precision threshold", f"{A['io_tau_hi']:.3f}"),
        ("Intraoral", "recall threshold (5th pct. of augmented copies)", f"{A['io_tau_split']:.3f}"),
        ("Intraoral", "random pairs sampled", n(Q["io_random_pairs"])),
        ("Intraoral", "random pairs above precision threshold", f"{100 * Q['io_random_share_above_tau_hi']:.2f}\\%"),
        ("Intraoral", "random pairs above recall threshold", f"{100 * Q['io_random_share_above_tau_split']:.1f}\\%"),
        ("Intraoral", "augmented files", n(Q["io_aug_files"])),
        ("Intraoral", "augmented files above recall threshold", f"{100 * Q['io_aug_share_above_tau_split']:.1f}\\%"),
        ("Intraoral", "augmented files above precision threshold", f"{100 * Q['io_aug_share_above_tau_hi']:.1f}\\%"),
        ("Intraoral", "IO-DET images byte-identical to IO-CLS", n(A["io_det_bytes_identical_to_cls"])),
        ("Intraoral", "IO-DET images above precision threshold to IO-CLS", n(A["io_det_max_sim_to_cls_ge_tau_hi"])),
        ("Intraoral", "edges touching IO-DET images", n(A["io_det_edges"])),
        ("Intraoral", "cached IO-CLS lineages / joint lineages", f"{n(A['io_cls_lineages_cached'])} / {n(A['io_components'])}"),
        ("Intraoral", "largest lineage (assigned to training)", n(A["io_largest_component"])),
        ("Intraoral", "IO-DET exact-duplicate files removed", n(A["io_det_exact_duplicate_files_removed"])),
        ("Intraoral", "duplicate groups with conflicting boxes", n(A["io_det_duplicate_groups_with_conflicting_boxes"])),
        ("Check", "descriptor reproduction, min. cosine to cache", f"{A['link_embedding_reproduction_min_cos']:.6f}"),
        ("Check", "planted-copy set: files / sources / lineages", f"{sm['files']} / {sm['sources']} / {sm['recovered_lineages']}"),
        ("Check", "planted-copy test contamination, lineage / random split",
         f"{100 * sm['lineage_split_contamination_true_lineage']:.1f}\\% / {100 * sm['random_split_contamination_true_lineage']:.1f}\\%"),
        ("DENTEX", "similarity to the nearest \\name file: median / 90th / 99th pct.",
         " / ".join(f"{q:.3f}" for q in Dx["max_sim_quantiles"][:3])),
    ]
    lines, last = [], None
    for g, k, v in rows:
        if last and g != last:
            lines.append("\\midrule")
        lines.append(f"{g if g != last else ''} & {k} & {v} \\\\")
        last = g
    write("tab_app_linkqc.tex", r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Copy-linking quality control.} Thresholds, edge counts and the share of random and known-copy pairs on each side of the thresholds (40{,}000 sampled pairs per modality, seed 0). The descriptor check re-embeds the cached files and reports the lowest cosine to the cached vector.}
\label{tab:app_linkqc}
\begin{tabularx}{\columnwidth}{lXr}
\toprule
Group & quantity & value \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabularx}
\end{table}
""")


# ----------------------------------------------------------------------------- runs and per-class results
def _runs():
    out = {}
    for p in sorted(R.glob("yolo_*.json")):
        r = json.loads(p.read_text())
        if "shipped" in r.get("run", "") and r.get("data") not in ("iodet", "iodet_shipped"):
            continue  # only the L3 run trained with our recipe on the shipped IO-DET split is kept
        out[r["run"]] = r
    return out


def model_of(r):
    import re
    m = re.search(r"y11([nsm])", r["run"])
    base = f"YOLO11{m.group(1)}" if m else esc(r["model"])
    return base + ("-seg" if r["data"] == "pxseg" else "")


def eff_batch(r):
    """Effective batch after Ultralytics' out-of-memory fallback (training args), else the requested one."""
    f = ROOT / "data" / "runs" / r["run"] / "args.yaml"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.startswith("batch:"):
                return line.split(":", 1)[1].strip()
    return r.get("batch", "--")


TASKNAME = {"iodet": "IO-DET", "pxdet": "PX-DET", "pxseg": "PX-SEG", "pxleak": "L1 swap", "dentex": "DENTEX"}


def runs_table():
    runs = _runs()
    def role(rn):
        if rn.startswith("pxleak"):
            return "leaky" if "leaky" in rn else "clean"
        if rn.startswith("dentex"):
            return "COCO init" if "coco" in rn else "\\name init"
        if "shipped" in rn:
            return "shipped split"
        if "reexp" in rn:
            return "D1"
        return "main"
    order = ["iodet", "pxdet", "pxseg", "pxleak", "dentex"]
    lines = []
    for t in order:
        rs = [r for k, r in runs.items() if r["data"] == t or k.startswith(t + "_")]
        if not rs:
            continue
        if lines:
            lines.append("\\midrule")
        for r in sorted(rs, key=lambda r: (role(r["run"]), r["model"], r["seed"])):
            kind = "mask" if r["data"] == "pxseg" else "box"
            def m(split, k):
                return f1(((r.get(split) or {}).get(kind) or {}).get(k))
            lines.append(f"{TASKNAME[t]} & {model_of(r)} & {role(r['run'])} & {r['seed']} & "
                         f"{r['epochs_full_passes']} & {r['shards']} & {eff_batch(r)} & "
                         + " & ".join(m("val", k) for k in ("map50_95", "map50", "precision", "recall")) + " & "
                         + " & ".join(m("test", k) for k in ("map50_95", "map50", "map75", "precision", "recall")) + " \\\\")
    write("tab_app_runs.tex", r"""\begin{table*}[t]
\centering\scriptsize
\setlength{\tabcolsep}{3.2pt}
\caption{\textbf{Every dense training run} (\%, masks for PX-SEG, boxes otherwise). Passes: full passes over the training set; shards: equal parts of the training set, one per logged epoch, alternated; batch: effective batch. Validation is the split used for selection. DENTEX runs are not selected and keep the last checkpoint, so their test columns hold the score on the 50 DENTEX validation images. The L3 run (shipped split) has no copy-free evaluation, and both column groups hold its score on the shipped IO-DET validation split. P / R: precision and recall at the Ultralytics operating point.}
\label{tab:app_runs}
\begin{tabular}{lllcccc|cccc|ccccc}
\toprule
 & & & & & & & \multicolumn{4}{c|}{validation} & \multicolumn{5}{c}{test} \\
Task & Model & Arm & Seed & Passes & Shards & Batch & mAP$_{50:95}$ & mAP$_{50}$ & P & R & mAP$_{50:95}$ & mAP$_{50}$ & mAP$_{75}$ & P & R \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


def short(rn):
    r = rn.replace("pxdet_", "").replace("pxseg_", "").replace("iodet_", "")
    return r.replace("y11", "11").replace("_s", " s").replace("reexp", "D1").replace("_", " ")



LOW_N = 10       # review fix (2026-11-16): classes with fewer PX-DET evaluation boxes are marked as low support


def low(concept, split="test"):
    pc = J("split_class_stats.json")["px_det"]["per_concept"]
    return concept in pc and pc[concept]["val" if split != "test" else "test"] < LOW_N

def px_ap5095(split="test"):
    runs = _runs()
    tr = J("summary_det.json")["px_train_instances"]
    sp = "" if split == "test" else "_val"
    for task, kind, lab in (("pxdet", "box", "PX-DET box"), ("pxseg", "mask", "PX-SEG mask")):
        cols = sorted(k for k in runs if k.startswith(task + "_"))
        if not cols:
            continue
        lines = []
        for c in sorted(PX_CLASSES, key=lambda c: -tr.get(c, 0)):
            cells = []
            for rn in cols:
                v = runs[rn].get(split, {}).get(kind, {}).get("per_class_ap50_95") or {}
                cells.append(f1(v[c]) if c in v else "--")
            if all(x == "--" for x in cells):
                continue
            nm = PX2U[c]['name'] + ("$^\\dagger$" if low(PX2U[c]['name'], split) else "")
            lines.append(f"{nm} & {FAMS[PX2U[c]['family']]} & {n(tr.get(c, 0))} & " + " & ".join(cells) + " \\\\")
        word = "test" if split == "test" else "validation"
        env = "table*" if len(cols) > 4 else "table"     # more runs than fit one column -> full width
        write(f"tab_app_{task}{sp}_ap.tex", r"""\begin{""" + env + r"""}[t]
\centering\scriptsize
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{""" + lab + " " + word + r""" AP$_{50:95}$ per class} (\%) for every run (11n/11s/11m: YOLO11 size, s0/s1: seed), sorted by PX-DET training instances (n). Classes absent from the """ + word + r""" split are omitted. $^\dagger$Low support: fewer than """ + str(LOW_N) + " PX-DET " + word + r""" boxes, so the AP is unstable.}
\label{tab:app_""" + task + sp + r"""_ap}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llr""" + "c" * len(cols) + r"""}
\toprule
concept & fam. & n & """ + " & ".join(short(c) for c in cols) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{""" + env + r"""}
""")


def io_ap():
    runs = _runs()
    cols = sorted(k for k in runs if k.startswith("iodet_") and "shipped" not in k)
    lines = []
    for split in ("val", "test"):
        for c in IO_DET_CLASSES:
            cells = []
            for rn in cols:
                b = runs[rn].get(split, {}).get("box", {})
                cells.append(f"{f1(b.get('per_class_ap50_95', {}).get(c))} / {f1(b.get('per_class_ap50', {}).get(c))}")
            lines.append(f"{split} & {c} & " + " & ".join(cells) + " \\\\")
        if split == "val":
            lines.append("\\midrule")
    write("tab_app_io_ap.tex", r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{IO-DET per class}, AP$_{50:95}$ / AP$_{50}$ (\%) on the validation and test splits for every run.}
\label{tab:app_io_ap}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{ll""" + "c" * len(cols) + r"""}
\toprule
split & class & """ + " & ".join(short(c) for c in cols) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


# ----------------------------------------------------------------------------- classification
CLSNAME = {"multi_lattice": "C1 multi-encoder lattice", "C2_ensemble": "C2 (FT ConvNeXt-T + C1)"}
for _a, _n in (("vits14", "ViT-S/14"), ("vitb14", "ViT-B/14"), ("convnext_t", "ConvNeXt-T")):
    for _d, _m in (("last_cls", "last layer"), ("dinov2_lin", "linear eval"), ("lattice", "full-depth lattice")):
        CLSNAME[f"{_a}_{_d}"] = f"{_n}, {_m}"
for _a, _n in (("convnext_t", "FT ConvNeXt-T"), ("vits14", "FT ViT-S/14")):
    for _s in range(3):
        CLSNAME[f"ft_{_a}_s{_s}"] = f"{_n}, seed {_s}"
    CLSNAME[f"ft_{_a}_seedavg"] = f"{_n}, seed average"


def short_cls(k):
    return (CLSNAME[k].replace(", full-depth lattice", ", lattice").replace("C1 multi-encoder lattice", "C1 multi-enc. lattice")
            .replace("seed average", "seed avg.").replace("C2 (FT ConvNeXt-T + C1)", "C2 ensemble"))


def cls_full():
    S = J("summary_cls.json")
    F = J("cls_frozen.json")["models"]
    lines = []
    for k, m in S["models"].items():
        if k not in CLSNAME:
            continue
        ci = m.get("test_ci95")
        cis = f"[{f1(ci[0])}, {f1(ci[1])}]" if ci else "--"
        fz = F.get(k, {})
        c = fz.get("C")
        dims = n(fz["dims"]) if "dims" in fz else "--"
        lines.append(f"{short_cls(k)} & {dims} & {c if c is not None else '--'} & {f1(m.get('val_macro_ap'))} & "
                     f"{f1(m.get('test_macro_ap'))} & {cis} & {f1(m.get('test_macro_auroc'))} \\\\")
        if k in ("convnext_t_lattice", "multi_lattice", "ft_convnext_t_seedavg"):
            lines.append("\\midrule")
    write("tab_app_cls_full.tex", r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{1.8pt}
\caption{\textbf{IO-CLS, every model and seed} (\%). Lattice: full-depth lattice read-out. Dim.: read-out feature size before PCA; $C$: inverse regularisation chosen on validation; CI: 95\% interval from 1{,}000 bootstrap resamples of the 153 test lineages.}
\label{tab:app_cls_full}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrcccc}
\toprule
Model & Dim. & $C$ & val AP & test AP & 95\% CI & AUROC \\
\midrule
""" + "\n".join(lines[:-1] if lines and lines[-1] == "\\midrule" else lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


def cls_paired():
    S = J("summary_cls.json")
    P = S["paired_ci"]
    lines = []
    for ref, lab in (("C2_ensemble", "C2"), ("multi_lattice", "C1")):
        for k, ci in P.items():
            a, b = k.split("-minus-")
            if a != ref:
                continue
            sig = "" if ci[0] <= 0 <= ci[1] else "$^{*}$"
            other = CLSNAME.get(b, b)
            lines.append(f"{lab} $-$ {other} & [{f1(ci[0])}, {f1(ci[1])}]{sig} \\\\")
        lines.append("\\midrule")
    for enc, nm in (("vits14", "ViT-S/14"), ("vitb14", "ViT-B/14"), ("convnext_t", "ConvNeXt-T")):
        e = S["lattice_minus_last"][enc]
        lines.append(f"{nm}: lattice $-$ last layer ({f1(e['diff'])}) & [{f1(e['ci95'][0])}, {f1(e['ci95'][1])}]$^{{*}}$ \\\\")
    write("tab_app_cls_paired.tex", r"""\begin{table}[t]
\centering\scriptsize
\caption{\textbf{IO-CLS paired differences} in test macro-AP (points), 95\% intervals from a paired bootstrap over test lineages. $^{*}$: the interval excludes zero.}
\label{tab:app_cls_paired}
\begin{tabular}{lc}
\toprule
Difference & 95\% interval \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def cls_curves():
    runs = []
    for a, nm in (("convnext_t", "CNX-T"), ("vits14", "ViT-S")):
        for s in range(3):
            r = J(f"cls_ft_{a}_s{s}.json")
            if r:
                runs.append((f"{nm} s{s}", r))
    if not runs:
        return
    E = max(len(r["val_history"]) for _, r in runs)
    lines = []
    for e in range(E):
        cells = []
        for _, r in runs:
            h = r["val_history"]
            v = f1(h[e]) if e < len(h) else "--"
            cells.append(f"\\textbf{{{v}}}" if e < len(h) and e + 1 == r["best_epoch"] else v)
        lines.append(f"{e + 1} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("test AP & " + " & ".join(f1(r["test_macro_ap"]) for _, r in runs) + " \\\\")
    write("tab_app_cls_curves.tex", r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Fine-tuning curves.} Validation macro-AP (\%) after each epoch for every fine-tuning run (CNX-T: ConvNeXt-T, ViT-S: DINOv2 ViT-S/14, s: seed). Bold: the epoch kept; its test macro-AP is in the last row.}
\label{tab:app_cls_curves}
\begin{tabular}{c""" + "c" * len(runs) + r"""}
\toprule
epoch & """ + " & ".join(nm for nm, _ in runs) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ----------------------------------------------------------------------------- leakage and DENTEX
def leak_cls():
    S = J("summary_leak_cls.json")
    if S is None:
        return
    lines = []
    for k, m in S["models"].items():
        if k not in CLSNAME:
            continue
        lines.append(f"{CLSNAME[k]} & {f1(m['hash_split_full_test'])} & {f1(m['lineage_split_full_test'])} & "
                     f"{f1(m['common_test_trained_on_hash_split'])} & {f1(m['common_test_trained_on_lineage_split'])} & "
                     f"+{f1(m['inflation_common_test'])} \\\\")
    write("tab_app_leak_cls.tex", r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{L2 for every frozen read-out} (macro-AP, \%). Full test: each model on its own split's test set. Shared: both models on the """ + n(S["n_common_test_images"]) + r""" test images the two splits share; $\Delta$ is the inflation on these shared images, computed from unrounded scores.}
\label{tab:app_leak_cls}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
 & \multicolumn{2}{c}{full test} & \multicolumn{2}{c}{shared test} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Read-out & hash & lineage & hash & lineage & $\Delta$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


def leak_px_classes():
    runs = _runs()
    arms = [k for k in ("pxleak_y11n_cleanctl_s0", "pxleak_y11n_leaky_s0", "pxleak_y11n_cleanctl_s1",
                        "pxleak_y11n_leaky_s1") if k in runs]
    if not arms:
        return
    tr = J("summary_det.json")["px_train_instances"]
    lines = []
    for c in sorted(PX_CLASSES, key=lambda c: -tr.get(c, 0)):
        cells = []
        for rn in arms:
            v = runs[rn]["test"]["box"]["per_class_ap50_95"]
            cells.append(f1(v[c]) if c in v else "--")
        if all(x == "--" for x in cells):
            continue
        lines.append(f"{PX2U[c]['name']} & " + " & ".join(cells) + " \\\\")
    lab = {"pxleak_y11n_cleanctl_s0": "clean s0", "pxleak_y11n_leaky_s0": "leaky s0",
           "pxleak_y11n_cleanctl_s1": "clean s1", "pxleak_y11n_leaky_s1": "leaky s1"}
    write("tab_app_leak_px.tex", r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{\textbf{L1 per class.} PX-DET test AP$_{50:95}$ (\%) of the YOLO11n trained on the copy-free control and on the leaky training set, per seed, on the same 1{,}186 test radiographs.}
\label{tab:app_leak_px}
\begin{tabular}{l""" + "c" * len(arms) + r"""}
\toprule
concept & """ + " & ".join(lab[a] for a in arms) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def dentex():
    runs = _runs()
    D = J("dentex_overlap.json")
    arms = [k for k in ("dentex_coco_y11s_s0", "dentex_coco_y11s_s1", "dentex_du_y11s_s0", "dentex_du_y11s_s1") if k in runs]
    if not arms:
        return
    classes = list(D["eval_class_boxes"].keys())
    lines = []
    for c in classes:
        cells = [f"{f1(runs[a]['test']['box']['per_class_ap50_95'].get(c))} / {f1(runs[a]['test']['box']['per_class_ap50'].get(c))}" for a in arms]
        lines.append(f"{c} & {n(D['eval_class_boxes'][c])} & " + " & ".join(cells) + " \\\\")
    lab = {"dentex_coco_y11s_s0": "COCO s0", "dentex_coco_y11s_s1": "COCO s1",
           "dentex_du_y11s_s0": "\\name s0", "dentex_du_y11s_s1": "\\name s1"}
    write("tab_app_dentex.tex", r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{DENTEX per class}, AP$_{50:95}$ / AP$_{50}$ (\%) on the """ + n(D["eval_images_kept"]) + r""" validation images (""" + n(D["eval_boxes_kept"]) + r""" boxes), per initialisation and seed. Boxes: evaluation boxes of the class.}
\label{tab:app_dentex}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lr""" + "c" * len(arms) + r"""}
\toprule
class & boxes & """ + " & ".join(lab[a] for a in arms) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


def speed():
    S = J("speed_bench.json")
    lines = []
    for k, v in S.items():
        m, sz = k.split("@")
        m = m.replace(".pt", "")
        nm = m.replace("yolo11", "YOLO11").replace("rtdetr-l", "RT-DETR-l")
        if "error" in v:
            lines.append(f"{nm} & {sz} & \\multicolumn{{3}}{{c}}{{out of memory under the cap}} \\\\")
        else:
            lines.append(f"{nm} & {sz} & {v['n']} & {v['s']:.1f} & {v['img_per_s']:.2f} \\\\")
    write("tab_app_speed.tex", r"""\begin{table}[t]
\centering\scriptsize
\caption{\textbf{Training-speed probe} before the plan was frozen: images per second for a short training run on the GPU under the memory cap. The probe set the model sizes of the frozen plan.}
\label{tab:app_speed}
\begin{tabular}{lcrrr}
\toprule
Model & input & images & seconds & images/s \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def families():
    F = J("fig_analysis_values.json")
    fams = list(FAMS)
    lines = []
    for task, lab in (("pxdet", "PX-DET box"), ("pxseg", "PX-SEG mask")):
        for m, v in F.get(task, {}).items():
            pretty = {"reexp_y11s": "YOLO11s + D1", "rtdetrl": "RT-DETR-l"}.get(m)
            lines.append(f"{lab} & {pretty or MODEL.get('yolo' + m.replace('y', '') + '.pt', m.replace('_', ' '))} & "
                         + " & ".join(f1(v.get(f)) for f in fams) + " \\\\")
    write("tab_app_families.tex", r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Test AP$_{50:95}$ per clinical family} (\%, class-mean over the classes of the family present in test, per-class values averaged over the available seeds), the values behind panels (a, b) of \mainref{fig:analysis}{Fig.~4}. --: no class of the family in the PX-SEG test split.}
\label{tab:app_families}
\begin{tabular}{llccccc}
\toprule
Task & Model & """ + " & ".join(FAMS[f] for f in fams) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def cls_ft_labels():
    S = J("summary_cls.json")["models"]
    keys = [k for k in S if k.startswith("ft_")] + ["multi_lattice", "C2_ensemble"]
    lines = []
    for k in keys:
        m = S.get(k)
        if not m or k not in CLSNAME:
            continue
        pc = m.get("test_per_class_ap") or [None] * 6
        lines.append(f"{short_cls(k)} & " + " & ".join(f1(v) for v in pc) + " \\\\")
    ab = {"calculus": "calc.", "caries": "caries", "gingivitis": "ging.", "hypodontia": "hypo.",
          "tooth discoloration": "discol.", "mouth ulcer": "ulcer"}
    head = " & ".join(ab[c] for c in IO_CLS_CLASSES)
    write("tab_app_cls_labels.tex", r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{IO-CLS test AP per label} (\%) for every fine-tuning seed, the seed averages and the two declared candidates. calc.: calculus, ging.: gingivitis, hypo.: hypodontia, discol.: tooth discoloration.}
\label{tab:app_cls_labels}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
Model & """ + head + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


def per_image_table():
    S = J("split_class_stats.json")
    if S is None or "annotations_per_image" not in S["io_cls"]:
        return
    lines = []
    for t, lab, unit in (("io_cls", "IO-CLS", "labels"), ("io_det", "IO-DET", "boxes"), ("px_det", "PX-DET", "boxes")):
        a = S[t]["annotations_per_image"]
        lines.append(f"{lab} & {unit} & {a['min']} & {a['median']:g} & {a['p90']} & {a['max']} & {a['mean']:.2f} & {n(a['empty'])} \\\\")
    write("tab_app_perimage.tex", r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{\textbf{Annotations per image} on the released images of each task: minimum, median, 90th percentile, maximum, mean, and images with no annotation.}
\label{tab:app_perimage}
\begin{tabular}{llcccccc}
\toprule
Task & unit & min & median & p90 & max & mean & empty \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def trainfree():
    """Training-free references added by the 2026-11-16 amendment: every zero-shot IO-CLS candidate
    with its validation macro-AP (selection) and the single test score of the selected one, and the
    box-prompted SAM PX-SEG runs with the box-mAP pipeline check."""
    Z, SM = J("cls_zeroshot.json"), J("sam_pxseg.json")
    if Z is None and SM is None:
        return
    lines = []
    if Z is not None:
        mn = {"biomedclip": "BiomedCLIP~\\cite{zhang2025multimodal}", "clip_vitb16": "CLIP ViT-B/16~\\cite{radford2021learning}"}
        lines.append("\\multicolumn{5}{l}{\\textit{IO-CLS zero-shot (macro-AP, \\%; chance on test: "
                     f"{f1(Z['test_chance_macro_ap'])})}}}} \\\\")
        lines.append("Model & prompts & val & test & test AUROC \\\\")
        lines.append("\\midrule")
        for k, v in Z["candidates_val_macro_ap"].items():
            m, t = k.split("/")
            sel = k == Z["selected"]
            te = f1(Z["test_macro_ap"]) if sel else "--"
            au = f1(Z["test_macro_auroc"]) if sel else "--"
            vv = f1(v)
            if sel:
                vv = "\\textbf{" + vv + "}"
            lines.append(f"{mn[m]} & {t} & {vv} & {te} & {au} \\\\")
    if SM is not None and SM.get("runs"):
        lines += ["\\midrule", "\\multicolumn{5}{l}{\\textit{PX-SEG, YOLO11s boxes + SAM ViT-B~\\cite{kirillov2023segment} masks (mAP$_{50:95}$, \\%)}} \\\\",
                  "Run & split & SAM mask & YOLO11s mask & box (check) \\\\", "\\midrule"]
        for k in sorted(SM["runs"], key=lambda x: (x.split("/")[0], x.split("/")[1] != "val")):
            r = SM["runs"][k]
            run, split = k.split("/")
            lines.append(f"YOLO11s seed {run[-1]} & {split} & {f1(r['mask']['map50_95'])} & "
                         f"{f1(r['yolo_mask_map50_95_recorded'])} & {f1(r['box']['map50_95'])} / {f1(r['box_map50_95_recorded'])} \\\\")
    write("tab_app_trainfree.tex", r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{Training-free references} (amendment of 2026-11-16). Zero-shot prompt sets: T1 ``a photo of \{label\}.'', T2 ``an intraoral photograph showing \{label\}.'', T3 ``a clinical photograph of the mouth with \{label\}.'', T4 the mean of T1--T3. Bold: selected on validation. Only the selected candidate is tested. SAM rows: box mAP recomputed / as recorded for the YOLO11s run, which must agree because only the masks change. The SAM run was stopped by the authors after seed 0.}
\label{tab:app_trainfree}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llccc}
\toprule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table}
""")


def cls_cv():
    """Repeated lineage-level 5-fold CV of the IO-CLS frozen read-outs (plan amendment 2026-11-17 11:07)."""
    V = J("cls_lineage_cv.json")
    if V is None:
        return
    order = sorted(V["models"], key=lambda m: -V["models"][m]["mean"])
    name = lambda m: "C1 multi-encoder lattice" if m == "multi_lattice" else short_cls(m)
    lines = []
    for m in order:
        e, t = V["models"][m], V["tiers"][m]
        ci = "--" if m == V["best"] else f"[{100 * t['ci95'][0]:.1f}, {100 * t['ci95'][1]:.1f}]"
        lines.append(f"{name(m)} & {f1(e['mean'])} $\\pm$ {f1(e['sd'])} & [{f1(e['min'])}, {f1(e['max'])}] & {ci} & "
                     f"{'\\cmark' if t['tied_with_best'] else ''} \\\\")
    write("tab_app_cls_cv.tex", r"""\begin{table}[!htbp]
\centering\scriptsize
\setlength{\tabcolsep}{2.5pt}
\caption{\textbf{IO-CLS read-outs under repeated lineage-level cross-validation} (macro-AP, \%; """ + str(V["repeats"]) + " repeats of " + str(V["folds"]) + r"""-fold CV over """ + n(V["lineages_folded"]) + r""" lineages, the largest lineage kept in training). Mean $\pm$ s.d. and range over folds. Best $-$ read-out: fold-paired 95\% interval. \cmark: top tier (interval includes zero). C is fixed from the original validation split.}
\label{tab:app_cls_cv}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc}
\toprule
Read-out & macro-AP & range & best $-$ this & tier \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}
\end{table}
""")


if __name__ == "__main__":
    for fn in (survey, splits, px_classes, io_classes, linkqc, runs_table, px_ap5095, io_ap, cls_full, cls_paired,
               cls_curves, leak_cls, leak_px_classes, dentex, speed, families, cls_ft_labels, per_image_table, trainfree, cls_cv):
        fn()
    px_ap5095("val")
    print("appendix tables:", len(list(OUT.glob("tab_app_*.tex"))))
