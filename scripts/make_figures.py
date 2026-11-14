"""Paper figures from results/*.json and cached arrays only (no training, no GPU).
  python scripts/make_figures.py [paper_dir]
Writes figures/<name>.pdf/.png, copies the PDFs to <paper_dir>/figures/, and writes the
plotted values to results/fig_*_values.json. Missing (still running) results render as
"pending" panels, so re-running the script after each result fills the figures.

Privacy: file paths, source keys and label paths are used only in memory to align
arrays; they are never written to any figure, JSON or log.

Each figure mirrors one element of the IMed-361M paper (see paper/EXEMPLAR_MAP.md):
  fig_pipeline  <- IMIS Fig. 3 (boxed-stage schematic)
  fig_overview  <- IMIS Fig. 1 (a-e dataset overview)
  fig_linkqc    <- IMIS Fig. 2 (quality diagnostic with a shaded region)
  fig_analysis  <- IMIS Fig. 5 (radars + breakdown panels)
  fig_leak      <- IMIS Fig. 6 (data-design bars with printed values)
"""
import json, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
from paper_style import style, METHOD_COLORS
from unidental.common import DATA, RESULTS, REPO, write_json
from unidental.taxonomy import UNIFIED, PX_CLASSES, FAMILIES, to_unified, IO_CLS_CLASSES

FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)
PAPER = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (REPO / "../../paper").resolve()
FAMC = {"pathology": "#D55E00", "periodontal / soft tissue": "#CC79A7", "tooth status / development": "#009E73",
        "anatomy": "#999999", "restoration / device": "#0072B2"}
FAMS = {"pathology": "pathology", "periodontal / soft tissue": "periodontal", "tooth status / development": "tooth status",
        "anatomy": "anatomy", "restoration / device": "restoration"}
SPLITC = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}
TW, CW = 6.9, 3.3            # CVPR text width / column width in inches
J = lambda n: json.loads((RESULTS / n).read_text()) if (RESULTS / n).exists() else None
# fixed model colours across every result figure (IMIS keeps one colour per model)
MODEL_LABEL = {"y11n": "YOLO11n", "y11s": "YOLO11s", "y11m": "YOLO11m", "reexp_y11s": "YOLO11s+D1",
               "rtdetrl": "RT-DETR-l", "shipped": "shipped YOLOv8x-seg$^\\dagger$"}
for k in ("y11s", "y11m", "y11n", "reexp_y11s", "rtdetrl", "shipped"):
    METHOD_COLORS.assign(k)
COL = lambda k: "#555555" if k == "shipped" else METHOD_COLORS.assign(k)


def small():
    style()
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.titlepad": 3})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=250)
    plt.close(fig)
    (PAPER / "figures").mkdir(exist_ok=True)
    shutil.copy2(FIG / f"{name}.pdf", PAPER / "figures" / f"{name}.pdf")


def pending(ax, title, msg="pending"):
    ax.set_title(title, loc="left")
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#888888", fontsize=8,
            style="italic")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#cccccc"); s.set_linestyle(":")


def fmt(n):
    return f"{n:,}"


# ----------------------------------------------------------------------------- Fig. 2 pipeline
def card(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc="white", ec=color, lw=1.1))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.3), w, 0.3, boxstyle="round,pad=0.01,rounding_size=0.02", fc=color, ec=color, lw=1.1))
    ax.text(x + w / 2, y + h - 0.15, title, ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")
    for i, l in enumerate(lines):
        ax.text(x + 0.03, y + h - 0.45 - i * 0.21, l, ha="left", va="center", fontsize=4.9)


def fig_pipeline():
    S = J("dataset_stats.json"); A = S["audit"]
    small()
    fig, ax = plt.subplots(figsize=(TW, 1.05))
    ax.set_xlim(0, 5.02); ax.set_ylim(0, 1.0); ax.axis("off")
    cols = ["#999999", "#CC79A7", "#D55E00", "#0072B2", "#009E73"]
    cards = [("1 Inventory", [f"{fmt(S['px_files']['files'])} panoramic files", f"{fmt(S['io_cls']['images'])} + {fmt(S['io_det']['images'])} photos", "content hash, no file names"]),
             ("2 Harmonise", [f"{S['total']['unified_concepts']} unified concepts", "5 clinical families", f"{S['total']['tasks']} tasks, {S['total']['modalities']} modalities"]),
             ("3 Link copies", ["hash, export key", "registered copies", "embedding edges"]),
             ("4 Split lineages", ["whole lineages", "one split, all tasks", "stratified 70/10/20"]),
             ("5 Audit + release", ["no evaluation image", "has a training copy", "build kit, no images"])]
    for i, (t, ls) in enumerate(cards):
        card(ax, 0.02 + i * 1.0, 0.03, 0.9, 0.94, t, ls, cols[i])
        if i < 4:
            ax.annotate("", xy=(0.02 + (i + 1) * 1.0 - 0.004, 0.5), xytext=(0.02 + i * 1.0 + 0.904, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0))
    save(fig, "fig_pipeline")


# ----------------------------------------------------------------------------- Fig. 3 overview
def fig_overview():
    S = J("dataset_stats.json"); A = S["audit"]
    b = json.loads((DATA / "bench.json").read_text())
    small()
    fig = plt.figure(figsize=(TW, 3.9))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.78], hspace=0.95, wspace=0.95)
    vals = {}
    # (a) modality -> task donut
    ax = fig.add_subplot(gs[0, 0:2])
    tasks = [("IO-CLS", "io_cls", "#CC79A7"), ("IO-DET", "io_det", "#E69F00"), ("PX-DET", "px_det", "#0072B2"), ("PX-SEG", "px_seg", "#56B4E9")]
    ti = [S[k]["images"] for _, k, _ in tasks]
    mod = [S["io_cls"]["images"] + S["io_det"]["images"], S["px_det"]["images"] + S["px_seg"]["images"]]
    ax.pie(mod, radius=1.0, colors=["#F2D7E6", "#D6E6F4"], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.28, edgecolor="white"))
    w, _ = ax.pie(ti, radius=0.70, colors=[c for *_, c in tasks], startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.22, edgecolor="white"))
    ang0 = 90
    tot = sum(ti)
    for (lab, k, c), v in zip(tasks, ti):
        ang = ang0 - 360 * v / tot / 2; ang0 -= 360 * v / tot
        x, y = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        ax.annotate(f"{lab}\n{fmt(v)}", (0.59 * x, 0.59 * y), xytext=(1.30 * x + (0.38 if x > 0 else -0.38), 1.25 * y), ha="center", va="center",
                    fontsize=6, arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5))
    ax.text(0, 0.12, f"{fmt(S['total']['distinct_images'])}", ha="center", va="center", fontsize=6.5, fontweight="bold")
    ax.text(0, -0.12, "images", ha="center", va="center", fontsize=5.4)
    ax.set_title("(a) modalities and tasks", loc="left", x=-0.25)
    ax.text(0, -1.55, "outer ring: intraoral | panoramic", fontsize=5.0, color="#666666", ha="center")
    # (b) family small multiples of annotations per concept
    cnt = J("fig_stats_values.json")["annotations_per_concept"]
    sub = gs[0, 2:6].subgridspec(1, 5, wspace=0.12, width_ratios=[7, 5, 9, 2, 13])
    hatch = {"cls": "////", "iodet": "....", "px": ""}
    axs = []
    for j, fam in enumerate(FAMILIES):
        a = fig.add_subplot(sub[j], sharey=axs[0] if axs else None)
        axs.append(a)
        cs = sorted([c for c in UNIFIED if c["family"] == fam], key=lambda c: -sum(cnt.get(f"{c['name']}|{k}", 0) for k in hatch))
        x = 0
        ticks = []
        for c in cs:
            ks = [k for k in hatch if cnt.get(f"{c['name']}|{k}")]
            for i, k in enumerate(ks):
                a.bar(x + i * 0.8 / len(ks), cnt[f"{c['name']}|{k}"], 0.8 / len(ks), align="edge", color=FAMC[fam],
                      hatch=hatch[k], edgecolor="white", linewidth=0.2)
            ticks.append((x + 0.4, c["name"]))
            x += 1
        a.set_yscale("log"); a.set_ylim(0.7, 6e4)
        a.set_xticks([t for t, _ in ticks], [n.replace("temporary anchorage device", "TAD").replace("root canal treatment", "root canal") for _, n in ticks], rotation=90, fontsize=4.6)
        a.set_xlim(-0.3, x)
        a.set_title(FAMS[fam], fontsize=6.2, color=FAMC[fam])
        a.grid(axis="x", visible=False)
        if j:
            plt.setp(a.get_yticklabels(), visible=False)
        else:
            a.set_ylabel("annotations")
    axs[0].text(0, 1.2, "(b) annotations per unified concept, by clinical family", transform=axs[0].transAxes, fontsize=8)
    axs[-1].legend(handles=[Patch(facecolor="white", edgecolor="k", hatch=h, label=l, linewidth=0.4)
                            for l, h in (("IO-CLS", "////"), ("IO-DET", "...."), ("PX", ""))],
                   loc="upper right", fontsize=5.2, handlelength=1.2)
    # (c) image resolution per modality
    ax = fig.add_subplot(gs[1, 0:2])
    io = [max(r["width"], r["height"]) for r in b["io_cls"]] + [max(r["width"], r["height"]) for r in b["io_det"]
                                                                 if r["sha256"] not in {x["sha256"] for x in b["io_cls"]}]
    px = [max(r["width"], r["height"]) for r in b["px"] if r["role"] == "canonical"]
    edges = [0, 256, 512, 1024, 2048, 10 ** 6]
    labs = ["<256", "256+", "512+", "1024+", "2048+"]
    hi, _ = np.histogram(io, edges); hp, _ = np.histogram(px, edges)
    xs = np.arange(5)
    ax.bar(xs - 0.2, hi, 0.4, color="#CC79A7", label="intraoral")
    ax.bar(xs + 0.2, hp, 0.4, color="#0072B2", label="panoramic")
    ax.set_yscale("log"); ax.set_ylim(0.8, 3e4)
    ax.set_xticks(xs, labs, fontsize=5.6); ax.set_ylabel("images"); ax.set_xlabel("long side (px)", labelpad=1)
    ax.legend(loc="upper right", fontsize=5.6)
    ax.set_title("(c) image resolution", loc="left")
    vals["resolution_bins"] = labs
    vals["resolution_intraoral"] = hi.tolist(); vals["resolution_panoramic"] = hp.tolist()
    vals["panoramic_long_side_1615"] = int(sum(1 for v in px if v == 1615))
    # (d) box area share of the image
    ax = fig.add_subplot(gs[1, 2:4])
    ai = [x[3] * x[4] for r in b["io_det"] for x in r["boxes_cxcywh"]]
    ap = []
    for r in b["px"]:
        if r["role"] != "canonical":
            continue
        for x in r["instances"]:
            bx = x["box"]
            x0, y0, x1, y1 = [min(max(v, 0.0), 1.0) for v in bx]   # normalised corners
            ap.append((x1 - x0) * (y1 - y0))
    e2 = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 1.01]
    l2 = ["<.5", ".5-1", "1-2", "2-5", "5-10", "10+"]
    di, _ = np.histogram(ai, e2); dp, _ = np.histogram(ap, e2)
    ax.plot(range(6), di, "o-", color="#CC79A7", ms=3, lw=1, label="IO-DET boxes")
    ax.plot(range(6), dp, "s-", color="#0072B2", ms=3, lw=1, label="PX boxes")
    ax.set_yscale("log"); ax.set_xticks(range(6), l2, fontsize=5.6); ax.set_ylabel("boxes")
    ax.set_xlabel("box area / image area (%)", labelpad=1)
    ax.legend(loc="lower left", fontsize=5.6)
    ax.set_title("(d) object size", loc="left")
    vals["box_area_bins"] = l2; vals["box_area_iodet"] = di.tolist(); vals["box_area_px"] = dp.tolist()
    vals["px_box_share_below_2pct"] = float(np.mean(np.array(ap) < 0.02))
    vals["iodet_box_share_below_2pct"] = float(np.mean(np.array(ai) < 0.02))
    # (e) copy structure: raw files -> distinct -> lineages
    ax = fig.add_subplot(gs[1, 4:6])
    rows = [("PX files", S["px_files"]["files"], "#999999"), ("PX distinct", S["px_det"]["images"], "#0072B2"),
            ("PX lineages", A["px_lineages_final"], "#56B4E9"),
            ("IO files", S["io_cls"]["images"] and 5563 + A["io_det_files"], "#999999"),
            ("IO distinct", S["io_cls"]["images"] + S["io_det"]["images"] - A["io_det_bytes_identical_to_cls"], "#CC79A7"),
            ("IO lineages", A["io_components"], "#E8B9D4")]
    lab = J("io_cls_labels_audit.json")
    rows[3] = ("IO files", lab["original_files"] + A["io_det_files"], "#999999")
    for i, (l, v, c) in enumerate(rows):
        ax.barh(i, v, color=c, height=0.66)
        ax.text(v + 250, i, fmt(v), va="center", fontsize=5.6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=5.8); ax.invert_yaxis()
    ax.set_xlim(0, rows[0][1] * 1.32); ax.set_xticks([0, 5000, 10000], ["0", "5k", "10k"])
    ax.set_title("(e) copies collapse", loc="left")
    vals["copy_rows"] = {l: int(v) for l, v, _ in rows}
    save(fig, "fig_overview")
    write_json(RESULTS / "fig_overview_values.json", dict(provenance="run (scripts/make_figures.py)", **vals))


# ----------------------------------------------------------------------------- Fig. 4 link quality
def fig_linkqc():
    S = J("dataset_stats.json"); A = S["audit"]
    b = json.loads((DATA / "bench.json").read_text())
    rng = np.random.default_rng(0)
    z = np.load(DATA / "linkemb_px.npz", allow_pickle=True)
    E = z["E"].astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    idx = {p: i for i, p in enumerate(z["paths"])}
    px = [r for r in b["px"] if r["path"] in idx]
    canon = {r["family"]: idx[r["path"]] for r in px if r["role"] == "canonical"}
    lin = {r["family"]: r["lineage"] for r in px if r["role"] == "canonical"}
    cf = list(canon)
    n = 40000
    i1, i2 = rng.integers(0, len(cf), n), rng.integers(0, len(cf), n)
    keep = [(cf[a], cf[c]) for a, c in zip(i1, i2) if a != c and lin[cf[a]] != lin[cf[c]]]
    rand = np.array([float(E[canon[a]] @ E[canon[c]]) for a, c in keep])
    reex = np.array([float(E[idx[r["path"]]] @ E[canon[r["family"]]]) for r in px if r["role"] == "re-export" and r["family"] in canon])
    tau = A["px_tau_hi"]
    # intraoral: random IO-DET pairs vs shipped augmented files matched to their nearest original
    zi = np.load(DATA / "linkemb_io_det.npz", allow_pickle=True)
    Ei = zi["E"].astype(np.float32); Ei /= np.linalg.norm(Ei, axis=1, keepdims=True)
    j1, j2 = rng.integers(0, len(Ei), n), rng.integers(0, len(Ei), n)
    m = j1 != j2
    rand_io = np.einsum("ij,ij->i", Ei[j1[m]], Ei[j2[m]])
    aug = np.array([r["max_sim"] for r in b["io_aug"]])
    small()
    fig, axs = plt.subplots(1, 2, figsize=(CW, 1.4), gridspec_kw=dict(wspace=0.32))
    bins = np.linspace(0.3, 1.0, 71)
    ax = axs[0]
    ax.axvspan(tau, 1.0, color="#D55E00", alpha=0.12, lw=0)
    ax.hist(rand, bins, density=True, color="#999999", alpha=0.8, label="random pairs")
    ax.hist(reex, bins, density=True, color="#0072B2", alpha=0.75, label="re-export pairs")
    ax.axvline(tau, color="#D55E00", lw=0.8)
    ax.text(tau - 0.01, ax.get_ylim()[1] * 0.92, f"{tau:.3f}", ha="right", fontsize=5.6, color="#D55E00")
    ax.set_yscale("log"); ax.set_xlabel("cosine similarity", labelpad=1); ax.set_ylabel("density")
    ax.set_title("(a) panoramic", loc="left"); ax.legend(loc="upper left", fontsize=5.2)
    ax = axs[1]
    ax.axvspan(A["io_tau_split"], 1.0, color="#E69F00", alpha=0.10, lw=0)
    ax.axvspan(A["io_tau_hi"], 1.0, color="#D55E00", alpha=0.14, lw=0)
    ax.hist(rand_io, bins, density=True, color="#999999", alpha=0.8, label="random pairs")
    ax.hist(aug, bins, density=True, color="#CC79A7", alpha=0.75, label="augmented copies")
    for t in (A["io_tau_split"], A["io_tau_hi"]):
        ax.axvline(t, color="#D55E00", lw=0.8)
    ax.set_yscale("log"); ax.set_xlabel("cosine similarity", labelpad=1)
    ax.set_title("(b) intraoral", loc="left"); ax.legend(loc="upper left", fontsize=5.2)
    save(fig, "fig_linkqc")
    write_json(RESULTS / "fig_linkqc_values.json", {
        "provenance": "run (scripts/make_figures.py; 40,000 sampled pairs, seed 0)",
        "px_random_pairs": int(len(rand)), "px_random_q999": float(np.quantile(rand, 0.999)),
        "px_random_share_above_tau": float(np.mean(rand >= tau)),
        "px_reexport_pairs": int(len(reex)), "px_reexport_share_above_tau": float(np.mean(reex >= tau)),
        "px_reexport_median": float(np.median(reex)),
        "io_random_pairs": int(len(rand_io)), "io_random_share_above_tau_hi": float(np.mean(rand_io >= A["io_tau_hi"])),
        "io_random_share_above_tau_split": float(np.mean(rand_io >= A["io_tau_split"])),
        "io_aug_files": int(len(aug)), "io_aug_share_above_tau_split": float(np.mean(aug >= A["io_tau_split"])),
        "io_aug_share_above_tau_hi": float(np.mean(aug >= A["io_tau_hi"]))})


# ----------------------------------------------------------------------------- Fig. 5 analysis
def _family_of_px():
    up = to_unified("px")
    fam = {c["id"]: c["family"] for c in UNIFIED}
    return {PX_CLASSES[i]: fam[u] for i, u in enumerate(up)}


def _px_models(task):
    """Per-class test AP50:95 (seed-averaged) for every evaluated PX model family."""
    D = J("summary_det.json") or {"runs": {}}
    SH = J("shipped_checkpoint.json") or {}
    kind = "mask" if task == "pxseg" else "box"
    out = {}
    fams = defaultdict(list)
    for rn, r in D["runs"].items():
        if rn.startswith(task + "_") and not rn.startswith("pxdet_rectseg"):
            fams[rn[len(task) + 1:].rsplit("_s", 1)[0]].append(r)
    for f, rs in fams.items():
        pcs = [r["test"].get(kind, {}).get("per_class_ap50_95") for r in rs]
        pcs = [p for p in pcs if p]
        if pcs:
            out[f] = {c: float(np.mean([p[c] for p in pcs if c in p])) for c in pcs[0]}
    sh = SH.get(f"{task}_test", {}).get(kind, {}).get("per_class_ap50_95")
    if sh:
        out["shipped"] = sh
    return out


def fig_analysis():
    small()
    fam_of = _family_of_px()
    D = J("summary_det.json") or {}
    tr = D.get("px_train_instances", {})
    fig = plt.figure(figsize=(TW, 2.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.2, 1.3], wspace=0.55)
    vals = {}
    order = ["y11n", "y11s", "y11m", "reexp_y11s", "rtdetrl", "shipped"]
    for j, task in enumerate(("pxdet", "pxseg")):
        ax = fig.add_subplot(gs[j], projection="polar")
        M = _px_models(task)
        present = [f for f in FAMILIES if any(fam_of.get(c) == f for p in M.values() for c in p)]
        th = np.linspace(0, 2 * np.pi, len(present), endpoint=False)
        vals[task] = {}
        for k in [o for o in order if o in M]:
            pc = M[k]
            r = [np.mean([v for c, v in pc.items() if fam_of.get(c) == f]) for f in present]
            vals[task][k] = dict(zip(present, map(float, r)))
            rr = np.r_[r, r[0]]; tt = np.r_[th, th[0]]
            ls = "--" if k == "shipped" else "-"
            ax.plot(tt, 100 * rr, ls, color=COL(k), lw=1.0, label=MODEL_LABEL[k])
            if k != "shipped":
                ax.fill(tt, 100 * rr, color=COL(k), alpha=0.12)
        ax.set_xticks(th, [FAMS[f].replace(" ", "\n") for f in present], fontsize=5.0)
        ax.tick_params(axis="x", pad=-1)
        top = max([100 * max(v.values()) for v in vals[task].values()] + [10]) * 1.15
        tk = [5, 10, 20, 30, 40, 50]; tk = [x for x in tk if x < top][-2:]
        ax.set_ylim(0, top); ax.set_yticks(tk, [str(x) for x in tk], fontsize=4.6)
        ax.set_title(f"({'ab'[j]}) {'PX-DET box' if task == 'pxdet' else 'PX-SEG mask'} AP$_{{50:95}}$", loc="left",
                     fontsize=7, pad=10, x=-0.25)
        if j == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(-0.45, -0.05), ncol=1, fontsize=5.2, frameon=False)
    # (c) per-class AP vs training instances
    ax = fig.add_subplot(gs[2])
    M = _px_models("pxdet")
    vals["pxdet_per_class"] = {}
    for k in [o for o in order if o in M]:
        pc = M[k]
        xs = [max(tr.get(c, 0), 1) for c in pc]
        ax.scatter(xs, [100 * v for v in pc.values()], s=7 if k != "shipped" else 9, color=COL(k),
                   marker="o" if k != "shipped" else "x", lw=0.7, label=MODEL_LABEL[k], alpha=0.85, zorder=3)
        vals["pxdet_per_class"][k] = pc
    ax.set_xscale("log")
    ax.axvspan(0.7, 100, color="#D55E00", alpha=0.06, lw=0); ax.axvspan(1000, 3e4, color="#0072B2", alpha=0.06, lw=0)
    G = D.get("groups", {}).get("pxdet_y11s", {})
    for lo, hi, bn in ((1, 100, "tail"), (100, 1000, "medium"), (1000, 2e4, "head")):
        v = G.get(f"test_box_{bn}_map50_95_mean")
        if v is not None:
            ax.hlines(100 * v, lo, hi, color=COL("y11s"), lw=1.2, linestyles=":")
            ax.text(np.sqrt(lo * hi), 100 * v + 2.5, f"{100 * v:.1f}", ha="center", fontsize=5.6, color=COL("y11s"))
    ax.set_xlim(0.7, 3e4); ax.set_ylim(-3, 80)
    ax.set_xlabel("training instances (log)", labelpad=1); ax.set_ylabel("AP$_{50:95}$", labelpad=0)
    ax.text(3, 74, "tail", fontsize=5.6, color="#D55E00"); ax.text(2500, 74, "head", fontsize=5.6, color="#0072B2")
    ax.set_title("(c) PX-DET per class", loc="left")
    # (d) IO-CLS read-out depth
    ax = fig.add_subplot(gs[3])
    SC = J("summary_cls.json") or {"models": {}}
    enc = [("vits14", "ViT-S/14"), ("vitb14", "ViT-B/14"), ("convnext_t", "ConvNeXt-T")]
    des = [("last_cls", "last layer", "#bdbdbd"), ("dinov2_lin", "linear eval", "#56B4E9"), ("lattice", "full-depth lattice", "#0072B2")]
    vals["cls"] = {}
    for i, (e, el) in enumerate(enc):
        for jj, (d, dl, c) in enumerate(des):
            m = SC["models"].get(f"{e}_{d}")
            if not m:
                continue
            x = i + (jj - 1) * 0.26
            lo, hi = m["test_ci95"]
            ax.bar(x, 100 * m["test_macro_ap"], 0.24, color=c, label=dl if i == 0 else None)
            ax.errorbar(x, 100 * m["test_macro_ap"], yerr=[[100 * (m["test_macro_ap"] - lo)], [100 * (hi - m["test_macro_ap"])]],
                        color="#333333", lw=0.6, capsize=1.5)
            vals["cls"][f"{e}_{d}"] = m["test_macro_ap"]
    for i, (k, lab) in enumerate((("multi_lattice", "C1"), ("C2_ensemble", "C2"))):
        m = SC["models"].get(k)
        if m:
            x = 3.05 + 0.32 * i
            ax.bar(x, 100 * m["test_macro_ap"], 0.28, color="#E69F00" if k == "multi_lattice" else "#D55E00")
            lo, hi = m["test_ci95"]
            ax.errorbar(x, 100 * m["test_macro_ap"], yerr=[[100 * (m["test_macro_ap"] - lo)], [100 * (hi - m["test_macro_ap"])]],
                        color="#333333", lw=0.6, capsize=1.5)
            vals["cls"][k] = m["test_macro_ap"]
    ax.set_xticks([0, 1, 2, 3.2], ["ViT-S", "ViT-B", "CNX-T", "C1 C2"], fontsize=5.4)
    ax.set_ylim(40, 112); ax.set_yticks([40, 60, 80, 100]); ax.set_ylabel("IO-CLS macro-AP")
    ax.legend(loc="upper left", fontsize=4.8, ncol=3, handlelength=0.8, columnspacing=0.6)
    ax.set_title("(d) IO-CLS read-outs", loc="left")
    save(fig, "fig_analysis")
    write_json(RESULTS / "fig_analysis_values.json", dict(provenance="run (scripts/make_figures.py)", **vals))


# ----------------------------------------------------------------------------- Fig. 6 leak + transfer
def fig_leak():
    small()
    D = J("summary_det.json") or {"runs": {}, "groups": {}}
    fig, axs = plt.subplots(1, 4, figsize=(TW, 1.75), gridspec_kw=dict(wspace=0.45))
    vals = {}

    def pbars(ax, labels, a, b, la, lb, ca, cb, title, ylabel):
        xs = np.arange(len(labels))
        ax.bar(xs - 0.19, a, 0.36, color=ca, label=la)
        ax.bar(xs + 0.19, b, 0.36, color=cb, label=lb)
        for x, u, v in zip(xs, a, b):
            ax.text(x - 0.19, u + 0.4, f"{u:.1f}", ha="center", fontsize=5.4)
            ax.text(x + 0.19, v + 0.4, f"{v:.1f}", ha="center", fontsize=5.4)
        ax.set_xticks(xs, labels, fontsize=5.8)
        ax.set_ylim(0, max(max(a), max(b)) * 1.35 + 1)
        ax.set_title(title, loc="left"); ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", fontsize=5.2)
    # (a) L1 controlled swap
    L = D.get("L1_swap")
    if L and L.get("map50_95_clean"):
        c, l = [100 * v for v in L["map50_95_clean"]], [100 * v for v in L["map50_95_leaky"]]
        pbars(axs[0], [f"seed {i}" for i in range(len(c))], c, l, "clean control", "leaky", "#0072B2", "#D55E00",
              "(a) controlled swap", "PX-DET mAP$_{50:95}$")
        axs[0].text(0.98, 0.97, f"+{100 * L['map50_95_rel_gain_mean']:.1f}%", transform=axs[0].transAxes, ha="right", va="top",
                    fontsize=7, fontweight="bold", color="#D55E00")
        vals["L1"] = L
    else:
        pending(axs[0], "(a) controlled swap")
    # (b) shipped checkpoint vs ours on test images with / without a copy in the shipped training split
    SE = J("subset_eval.json")
    sk = "shipped_yolov8x_seg|pxdet"
    ours = sorted([k for k in (SE or {}).get("models", {}) if k.startswith("pxdet_") and k.endswith("|pxdet")])
    if SE and sk in SE["models"] and ours:
        s = SE["models"][sk]
        o = [SE["models"][k] for k in ours]
        a = [100 * s["with_copy_subset"], 100 * s["no_copy_subset"]]
        bb = [100 * np.mean([x["with_copy_subset"] for x in o]), 100 * np.mean([x["no_copy_subset"] for x in o])]
        pbars(axs[1], ["copy in\nshipped train", "no copy"], a, bb, "shipped$^\\dagger$", "ours", "#555555", "#0072B2",
              "(b) shipped vs. ours", "PX-DET mAP$_{50:95}$")
        vals["L4b"] = {"shipped": s, "ours_runs": ours}
    else:
        pending(axs[1], "(b) shipped vs. ours")
    # (c) L2 (IO-CLS) and L3 (IO-DET)
    LC = J("summary_leak_cls.json")
    r3 = D.get("runs", {}).get("iodet_shipped_y11s_s0")
    io = D.get("groups", {}).get("iodet_y11s", {}).get("test_box_map50_95_mean")
    if LC and LC["models"].get("vitb14_lattice") and r3:
        m = LC["models"]["vitb14_lattice"]
        pbars(axs[2], ["IO-CLS\nmacro-AP", "IO-DET\nmAP$_{50:95}$"],
              [100 * m["common_test_trained_on_lineage_split"], 100 * io],
              [100 * m["common_test_trained_on_hash_split"], 100 * r3["test"]["box"]["map50_95"]],
              "copy-free split", "leaky split", "#0072B2", "#D55E00", "(c) intraoral splits", "score")
        vals["L2_L3"] = True
    else:
        pending(axs[2], "(c) intraoral splits")
    # (d) DENTEX transfer
    G = D.get("groups", {})
    a, c = G.get("dentex_coco_y11s"), G.get("dentex_du_y11s")
    if a and c:
        ax = axs[3]
        for i, (g, lab, col) in enumerate(((a, "COCO", "#999999"), (c, "DentUnity", "#0072B2"))):
            v = [100 * x for x in g["test_box_map50_95_per_seed"]]
            ax.scatter([i] * len(v), v, color=col, s=12, zorder=3)
            ax.hlines(np.mean(v), i - 0.25, i + 0.25, color=col, lw=1.4)
            ax.text(i + 0.28, np.mean(v), f"{np.mean(v):.1f}", va="center", fontsize=5.6)
        ax.set_xticks([0, 1], ["COCO\ninit", "DentUnity\ninit"], fontsize=5.8); ax.set_xlim(-0.6, 1.8)
        ax.set_title("(d) DENTEX transfer", loc="left"); ax.set_ylabel("DENTEX mAP$_{50:95}$")
        vals["dentex"] = {"coco": a["test_box_map50_95_per_seed"], "dentunity": c["test_box_map50_95_per_seed"]}
    else:
        pending(axs[3], "(d) DENTEX transfer")
    save(fig, "fig_leak")
    write_json(RESULTS / "fig_leak_values.json", {"provenance": "run (scripts/make_figures.py)",
                                                    "panels_filled": sorted(vals)})


# ----------------------------------------------------------------------------- Fig. S1 lineage sizes
def fig_supp_lineage():
    """Supplementary Fig. S1 (mirrors IMIS Fig. 2 layout): how many files each lineage holds."""
    b = json.loads((DATA / "bench.json").read_text())
    small()
    px = Counter(r["lineage"] for r in b["px"])
    io = Counter([("c", r["lineage"]) for r in b["io_cls"]] + [("c", r["lineage"]) for r in b["io_det"]])
    fig, axs = plt.subplots(1, 2, figsize=(CW, 1.4), gridspec_kw=dict(wspace=0.35))
    vals = {}
    for ax, (lab, C, col) in zip(axs, (("(a) panoramic files per lineage", px, "#0072B2"),
                                       ("(b) intraoral images per lineage", io, "#CC79A7"))):
        sz = np.array(list(C.values()))
        edges = [1, 2, 3, 5, 10, 100, 10 ** 6]
        h, _ = np.histogram(sz, edges)
        ax.bar(range(len(h)), h, color=col)
        ax.set_xticks(range(len(h)), ["1", "2", "3-4", "5-9", "10-99", "100+"], fontsize=5.6)
        ax.set_yscale("log"); ax.set_title(lab, loc="left", fontsize=7); ax.set_ylabel("lineages")
        vals[lab[:3]] = {"hist": h.tolist(), "n_lineages": int(len(sz)), "max_size": int(sz.max())}
    save(fig, "fig_supp_lineage")
    write_json(RESULTS / "fig_supp_lineage_values.json", dict(provenance="run (scripts/make_figures.py)", **vals))


if __name__ == "__main__":
    which = sys.argv[2:] or ["pipeline", "overview", "linkqc", "analysis", "leak", "supp_lineage"]
    for w in which:
        globals()[f"fig_{w}"]()
        print("figure", w, "ok", flush=True)
