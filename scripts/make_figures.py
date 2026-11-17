"""Paper figures from results/*.json and cached arrays only (no training, no GPU).
  python scripts/make_figures.py [paper_dir]
Writes figures/<name>.pdf/.png, copies the PDFs to <paper_dir>/figures/, and writes the
plotted values to results/fig_*_values.json. Missing (still running) results render as
"pending" panels, so re-running the script after each result fills the figures.

Privacy: file paths, source keys and label paths are used only in memory to align
arrays; they are never written to any figure, JSON or log.

Each figure mirrors one element of the IMed-361M paper (see paper/EXEMPLAR_MAP.md):
  fig_pipeline  <- IMIS Fig. 3 (boxed stages) in the multi-panel "glimpse" layout of SiteShift Fig. 1
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
from dentunity.common import DATA, RESULTS, REPO, write_json
from dentunity.taxonomy import UNIFIED, PX_CLASSES, FAMILIES, to_unified, IO_CLS_CLASSES

FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)
PAPER = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (REPO / "../../paper").resolve()
FAMC = {"pathology": "#D55E00", "periodontal / soft tissue": "#CC79A7", "tooth status / development": "#009E73",
        "anatomy": "#F0E442", "restoration / device": "#56B4E9"}
FAMS = {"pathology": "pathology", "periodontal / soft tissue": "periodontal", "tooth status / development": "tooth status",
        "anatomy": "anatomy", "restoration / device": "restoration"}
SPLITC = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}
TW, CW = 6.9, 3.3            # CVPR text width / column width in inches
J = lambda n: json.loads((RESULTS / n).read_text()) if (RESULTS / n).exists() else None
# fixed model colours across every result figure (IMIS keeps one colour per model)
MODEL_LABEL = {"y11n": "YOLO11n", "y11s": "YOLO11s", "y11m": "YOLO11m", "reexp_y11s": "YOLO11s+D1",
               "rtdetrl": "RT-DETR-l"}
for k in ("y11s", "y11m", "y11n", "reexp_y11s", "rtdetrl"):
    METHOD_COLORS.assign(k)
COL = lambda k: METHOD_COLORS.assign(k)



def plainlog(ax, axis="y"):
    """Log axis with plain tick labels (1, 10, 100, 1k, 10k): superscript exponents would print below 7 pt."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    f = FuncFormatter(lambda v, _: (f"{v / 1000:g}k" if v >= 1000 else f"{v:g}") if v >= 1 else f"{v:g}")
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, numticks=12))
    a.set_major_formatter(f)
    a.set_minor_formatter(NullFormatter())

def small():
    style()
    # presentation (2026-11-14 review): sizes chosen so every label prints at >= ~6.5 pt
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 7.2,
                         "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "axes.titlepad": 3})


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
def card(fig, x, w, top, h, num, title, sub, color):
    """One stage of the glimpse: numbered heading + grey subtitle above a light panel (SiteShift Fig. 1 style).
    Coordinates in inches from the bottom-left of the figure. Returns the panel rectangle (x, y, w, h)."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none"); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(x, top, f"{num}  {title}", ha="left", va="top", fontsize=7.6, fontweight="bold", color=color)
    ax.text(x, top - 0.135, sub, ha="left", va="top", fontsize=7.2, color="#666666")
    y = 0.02
    ph = top - 0.32 - y
    ax.add_patch(FancyBboxPatch((x, y), w, ph, boxstyle="round,pad=0,rounding_size=0.04", fc="#F6F6F6", ec="#BBBBBB", lw=0.7))
    return ax, (x, y, w, ph)


def _inset(fig, r, l, b, w, h):
    """Axes at inches (l, b, w, h) relative to panel rectangle r."""
    W, H = fig.get_size_inches()
    return fig.add_axes([(r[0] + l) / W, (r[1] + b) / H, w / W, h / H])


def fig_pipeline():
    """Fig. 2: five construction stages as a multi-panel glimpse (IMIS Fig. 3 stages + SiteShift Fig. 1 layout),
    each with a small summary drawn from results/dataset_stats.json and the taxonomy; no images, no file names."""
    S = J("dataset_stats.json"); A = S["audit"]
    LA = J("io_cls_labels_audit.json") or {}
    small()
    H = 1.62
    fig = plt.figure(figsize=(TW, H))
    gap = 0.2
    w = (TW - 4 * gap) / 5
    top = H - 0.01
    cols = ["#555555", "#CC79A7", "#D55E00", "#0072B2", "#009E73"]
    xs = [i * (w + gap) for i in range(5)]
    tx = dict(fontsize=7.2, va="center")

    # 1 Inventory: the three source files sets
    ax, r = card(fig, xs[0], w, top, H, 1, "Inventory", "two community releases", cols[0])
    srcs = [("Panoramic release", f"{fmt(S['px_files']['files'])} files"),
            ("Intraoral photos", f"{fmt(LA.get('original_files', S['io_cls']['images']))} files"),
            ("Intraoral boxes", f"{fmt(A['io_det_files'])} files")]
    bh = (r[3] - 0.34) / 3
    for i, (a, b) in enumerate(srcs):
        y0 = r[1] + r[3] - 0.05 - (i + 1) * bh + 0.03
        ax.add_patch(FancyBboxPatch((r[0] + 0.05, y0), r[2] - 0.1, bh - 0.06, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc="white", ec="#56B4E9" if i == 0 else "#CC79A7", lw=0.8))
        ax.text(r[0] + 0.09, y0 + (bh - 0.06) * 0.68, a, fontweight="bold", ha="left", **tx)
        ax.text(r[0] + 0.09, y0 + (bh - 0.06) * 0.27, b, ha="left", color="#333333", **tx)
    ax.text(r[0] + r[2] / 2, r[1] + 0.2, f"dropped: {S['px_files']['files'] - S['px_files']['usable_files']} invalid,", ha="center", color="#333333", **tx)
    ax.text(r[0] + r[2] / 2, r[1] + 0.08, f"{A['io_det_exact_duplicate_files_removed']} exact duplicates", ha="center", color="#333333", **tx)

    # 2 Harmonise: unified concepts per clinical family
    n_src = len(PX_CLASSES) + len(IO_CLS_CLASSES) + len(S["io_det"]["class_boxes"])
    ax, r = card(fig, xs[1], w, top, H, 2, "Harmonise", f"{n_src} source labels mapped", cols[1])
    fam_n = Counter(u["family"] for u in UNIFIED)
    a2 = _inset(fig, r, 0.66, 0.1, r[2] - 0.74, r[3] - 0.33)
    v = [fam_n[f] for f in FAMILIES][::-1]
    a2.barh(range(5), v, color=[FAMC[f] for f in FAMILIES][::-1], height=0.66, ec="#444444", lw=0.4)
    for i, n in enumerate(v):
        a2.text(n + 0.4, i, str(n), ha="left", va="center", fontsize=7.2)
    a2.set_yticks(range(5), [FAMS[f] for f in FAMILIES][::-1], fontsize=7.2)
    a2.tick_params(axis="y", length=0, pad=2); a2.grid(False)
    a2.set_xticks([]); a2.set_xlim(0, max(v) * 1.22)
    for sp in ("bottom", "top", "right"):
        a2.spines[sp].set_visible(False)
    a2.patch.set_alpha(0)
    ax.text(r[0] + r[2] / 2, r[1] + r[3] - 0.13, f"{S['total']['unified_concepts']} concepts, 5 families", ha="center", **tx)

    # 3 Link copies: schematic copy graph with the four edge rules
    ax, r = card(fig, xs[2], w, top, H, 3, "Link copies", "lineage = component", cols[2])
    # two lineages drawn as small file nodes; edge styles = rules
    L = [(0.17, 0.90), (0.47, 1.00), (0.36, 0.71), (0.74, 0.92), (1.05, 0.98), (0.98, 0.70)]
    def P(p):
        return r[0] + p[0], r[1] + p[1] * 0 + p[1]
    E = [(0, 1, dict(color="#333333", lw=1.1, ls="-")), (0, 2, dict(color="#333333", lw=1.0, ls=(0, (2.5, 1.5)))),
         (1, 2, dict(color="#D55E00", lw=1.1, ls="-")), (3, 4, dict(color="#333333", lw=1.0, ls=(0, (0.8, 1.2)))),
         (4, 5, dict(color="#D55E00", lw=1.1, ls="-"))]
    for i, j, st in E:
        (x0, y0), (x1, y1) = P(L[i]), P(L[j])
        ax.plot([x0, x1], [y0, y1], solid_capstyle="round", zorder=2, **st)
    for k, p in enumerate(L):
        x0, y0 = P(p)
        ax.add_patch(FancyBboxPatch((x0 - 0.065, y0 - 0.045), 0.13, 0.09, boxstyle="round,pad=0,rounding_size=0.01",
                                    fc="white", ec="#0072B2" if k < 3 else "#009E73", lw=0.9, zorder=3))
    leg = [("-", "#333333", "same sha256"), ((0, (2.5, 1.5)), "#333333", "same export key"),
           ((0, (0.8, 1.2)), "#333333", f"registered ({A['px_registered_links']})"),
           ("-", "#D55E00", f"PX: cos \u2265 {A['px_tau_hi']:.3f}"),
           ("-", "#D55E00", f"IO: {A['io_tau_hi']:.3f} / {A['io_tau_split']:.3f}")]
    for i, (ls, c, lab) in enumerate(leg):
        y0 = r[1] + 0.555 - i * 0.12
        ax.plot([r[0] + 0.07, r[0] + 0.25], [y0, y0], color=c, ls=ls, lw=1.1)
        ax.text(r[0] + 0.3, y0, lab, ha="left", **tx)

    # 4 Split lineages: train / val / test image share per task
    ax, r = card(fig, xs[3], w, top, H, 4, "Split lineages", "whole lineages, all tasks", cols[3])
    a4 = _inset(fig, r, 0.43, 0.32, r[2] - 0.5, r[3] - 0.4)
    tasks = [("IO-CLS", S["io_cls"]["split_images"]), ("IO-DET", S["io_det"]["split_images"]),
             ("PX-DET", S["px_det"]["split_images"]), ("PX-SEG", S["px_seg"]["split_images"])]
    for i, (t, d) in enumerate(tasks[::-1]):
        tot = sum(d.values()); left = 0
        for sp in ("train", "val", "test"):
            f = d[sp] / tot
            a4.barh(i, f, left=left, color=SPLITC[sp], height=0.62, ec="white", lw=0.4)
            left += f
    a4.set_yticks(range(4), [t for t, _ in tasks[::-1]], fontsize=7.2)
    a4.set_xticks([]); a4.set_xlim(0, 1); a4.grid(False)
    for s in ("top", "right", "bottom", "left"):
        a4.spines[s].set_visible(False)
    a4.tick_params(axis="y", length=0, pad=2)
    a4.patch.set_alpha(0)
    for i, sp in enumerate(("train", "val", "test")):
        x0 = r[0] + 0.08 + i * 0.4
        ax.add_patch(plt.Rectangle((x0, r[1] + 0.1), 0.09, 0.09, fc=SPLITC[sp], ec="none"))
        ax.text(x0 + 0.12, r[1] + 0.145, sp, ha="left", **tx)

    # 5 Audit + release: leaked share of evaluation images
    ax, r = card(fig, xs[4], w, top, H, 5, "Audit + release", "eval. images with a copy", cols[4])
    # round 7: intraoral bars show the strict-link share (conservative); all links in brackets
    SL = J("strict_link_audit.json")["intraoral_strict"]
    rows = [("shipped PX test", A["px_shipped_test_share_with_train_lineage"], None),
            ("shipped IO-DET val", SL["io_det_shipped_val_share_with_train_lineage"], A["io_det_shipped_val_share_with_train_lineage"]),
            ("hash IO-CLS test", SL["io_cls_hash_split_test_share_with_train_lineage"], A["io_cls_hash_split_test_share_with_train_lineage"]),
            ("DentUnity", max(A["px_test_contamination"], A["io_test_contamination"]), None)]
    for i, (lab, v, vall) in enumerate(rows):
        y0 = r[1] + r[3] - 0.17 - i * 0.235
        ax.text(r[0] + 0.07, y0 + 0.07, lab, ha="left", fontweight="bold" if i == 3 else "normal", **tx)
        bw = (r[2] - 0.55) * v
        if i == 3:      # round 5: our splits are reported as copy-lineage-disjoint, not as a 0% figure
            assert v == 0.0
            ax.text(r[0] + 0.07, y0 - 0.03, "no detected copy", ha="left", color="#009E73", **tx)
            continue
        ax.add_patch(plt.Rectangle((r[0] + 0.07, y0 - 0.065), max(bw, 0.004), 0.07, fc="#999999", ec="none"))
        ax.text(r[0] + 0.1 + bw, y0 - 0.03, f"{100 * v:.1f}%" + (f" ({100 * vall:.0f}%)" if vall is not None else ""),
                ha="left", **tx)
    ax.text(r[0] + r[2] / 2, r[1] + 0.13, "build kit, no images", ha="center", color="#333333", **tx)

    # arrows between stages
    W, Hh = fig.get_size_inches()
    axa = fig.add_axes([0, 0, 1, 1], facecolor="none"); axa.set_xlim(0, W); axa.set_ylim(0, Hh); axa.axis("off")
    for i in range(4):
        x0 = xs[i] + w + 0.03
        axa.annotate("", xy=(x0 + gap - 0.06, 0.62), xytext=(x0, 0.62),
                     arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.9, mutation_scale=7))
    save(fig, "fig_pipeline")


# ----------------------------------------------------------------------------- Fig. 3 overview
def fig_overview():
    S = J("dataset_stats.json"); A = S["audit"]
    b = json.loads((DATA / "bench.json").read_text())
    small()
    fig = plt.figure(figsize=(TW, 4.0))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.7], hspace=1.55, wspace=1.05)
    vals = {}
    # (a) modality -> task donut
    ax = fig.add_subplot(gs[0, 0:2])
    tasks = [("IO-CLS", "io_cls", "#CC79A7"), ("IO-DET", "io_det", "#E69F00"), ("PX-DET", "px_det", "#0072B2"), ("PX-SEG", "px_seg", "#56B4E9")]
    ti = [S[k]["images"] for _, k, _ in tasks]
    mod = [S["io_cls"]["images"] + S["io_det"]["images"], S["px_det"]["images"] + S["px_seg"]["images"]]
    ax.pie(mod, radius=1.32, colors=["#F2D7E6", "#D6E6F4"], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.22, edgecolor="white"))
    w, _ = ax.pie(ti, radius=1.07, colors=[c for *_, c in tasks], startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.12, edgecolor="white"))
    ang0 = 90
    tot = sum(ti)
    for (lab, k, c), v in zip(tasks, ti):
        ang = ang0 - 360 * v / tot / 2; ang0 -= 360 * v / tot
        x, y = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        ax.annotate(f"{lab}\n{fmt(v)}", (1.0 * x, 1.0 * y), xytext=(1.55 * x + (0.3 if x > 0 else -0.3), 1.58 * y), ha="center", va="center",
                    fontsize=7.2, arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5), annotation_clip=False)
    ax.set_xlim(-1.95, 1.95); ax.set_ylim(-1.7, 1.7)
    ax.text(0, 0.24, f"{fmt(S['total']['distinct_images'])}", ha="center", va="center", fontsize=6.6, fontweight="bold")
    ax.text(0, 0.0, "images", ha="center", va="center", fontsize=6.6)     # donut centre keeps its size (user: do not touch Fig. 3a)
    ax.text(0, -0.24, f"{S['total']['unified_concepts']} concepts", ha="center", va="center", fontsize=6.6)
    ax.set_title("(a) modalities and tasks", loc="left", x=-0.25)

    # (b) family small multiples of annotations per concept
    cnt = J("fig_stats_values.json")["annotations_per_concept"]
    nfam = [sum(1 for c in UNIFIED if c["family"] == f) for f in FAMILIES]    # one tick slot per concept
    sub = gs[0, 2:6].subgridspec(1, 5, wspace=0.12, width_ratios=[n + 0.3 for n in nfam])
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
        a.set_yscale("log"); a.set_ylim(0.7, 6e4); plainlog(a)
        a.set_xticks([t for t, _ in ticks], [n.replace("temporary anchorage device", "TAD").replace("root canal treatment", "root canal") for _, n in ticks], rotation=90, fontsize=7)
        a.set_xlim(-0.3, x)
        a.set_title(FAMS[fam], fontsize=7.2, color="#B09A00" if fam == "anatomy" else FAMC[fam])
        a.grid(axis="x", visible=False)
        if j:
            plt.setp(a.get_yticklabels(), visible=False)
        else:
            a.set_ylabel("annotations", labelpad=1)
            a.tick_params(axis="y", labelsize=7.2, pad=1)
    axs[0].text(0, 1.2, "(b) annotations per concept and family", transform=axs[0].transAxes, fontsize=8)
    axs[-1].legend(handles=[Patch(facecolor="white", edgecolor="k", hatch=h, label=l, linewidth=0.4)
                            for l, h in (("IO-CLS", "////"), ("IO-DET", "...."), ("PX", ""))],
                   loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=7.2, handlelength=1.2,
                   frameon=False, borderaxespad=0.0)
    # (c) image resolution per modality
    ax = fig.add_subplot(gs[1, 0:2])
    io = [max(r["width"], r["height"]) for r in b["io_cls"]] + [max(r["width"], r["height"]) for r in b["io_det"]
                                                                 if r["sha256"] not in {x["sha256"] for x in b["io_cls"]}]
    px = [max(r["width"], r["height"]) for r in b["px"] if r["role"] == "canonical"]
    edges = [0, 256, 512, 1024, 2048, 10 ** 6]
    labs = ["<256", "256+", "512+", "1k+", "2k+"]
    hi, _ = np.histogram(io, edges); hp, _ = np.histogram(px, edges)
    xs = np.arange(5)
    ax.bar(xs - 0.2, hi, 0.4, color="#CC79A7", label="intraoral")
    ax.bar(xs + 0.2, hp, 0.4, color="#0072B2", label="panoramic")
    ax.set_yscale("log"); ax.set_ylim(0.8, 3e5); plainlog(ax)
    ax.set_xticks(xs, labs, fontsize=7.2, rotation=30, ha="right", rotation_mode="anchor"); ax.set_ylabel("images")
    ax.set_xlabel("long side (px)", labelpad=1)
    ax.legend(loc="upper left", ncol=2, fontsize=7.2, frameon=False, borderaxespad=0.1, handlelength=1.0, columnspacing=0.8)
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
    ax.plot(range(6), di, "o-", color="#CC79A7", ms=3, lw=1, label="IO-DET")
    ax.plot(range(6), dp, "s-", color="#0072B2", ms=3, lw=1, label="PX")
    ax.set_yscale("log"); ax.set_ylim(100, 4e5); plainlog(ax); ax.set_xticks(range(6), l2, fontsize=7.2, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("boxes")
    ax.set_xlabel("box area / image area (%)", labelpad=1)
    ax.legend(loc="upper right", ncol=2, fontsize=7.2, frameon=False, handlelength=1.4, columnspacing=0.8, borderaxespad=0.1)
    ax.set_title("(d) object size", loc="left")
    vals["box_area_bins"] = l2; vals["box_area_iodet"] = di.tolist(); vals["box_area_px"] = dp.tolist()
    vals["px_box_share_below_2pct"] = float(np.mean(np.array(ap) < 0.02))
    vals["iodet_box_share_below_2pct"] = float(np.mean(np.array(ai) < 0.02))
    # (e) copy structure: raw files -> distinct -> lineages
    ax = fig.add_subplot(gs[1, 4:6])
    rows = [("PX raw", S["px_files"]["files"], "#999999"), ("PX distinct", S["px_det"]["images"], "#0072B2"),
            # lineages that hold a usable radiograph (2 of the 5,255 graph lineages hold only dropped files)
            ("PX lin.", J("split_class_stats.json")["px_det"]["lineages_total"], "#56B4E9"),
            ("IO files", J("io_cls_composite_audit.json")["total_files_walked"] + A["io_det_files"], "#999999"),
            ("IO distinct", S["io_cls"]["images"] + S["io_det"]["images"] - A["io_det_bytes_identical_to_cls"], "#CC79A7"),
            ("IO lin.", A["io_components"], "#E8B9D4")]
    lab = J("io_cls_labels_audit.json")
    rows[3] = ("IO raw", lab["original_files"] + A["io_det_files"], "#999999")
    for i, (l, v, c) in enumerate(rows):
        ax.barh(i, v, color=c, height=0.66)
        ax.text(v + 250, i, fmt(v), va="center", fontsize=7.2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.2); ax.invert_yaxis()
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
    plt.rcParams.update({"font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.6,
                         "xtick.labelsize": 7.2, "ytick.labelsize": 7.2})
    fig, axs = plt.subplots(1, 2, figsize=(3.05, 1.45), gridspec_kw=dict(wspace=0.38))
    bins = np.linspace(0.3, 1.0, 71)
    ax = axs[0]
    ax.axvspan(tau, 1.0, color="#D55E00", alpha=0.25, lw=0)
    ax.hist(rand, bins, density=True, color="#999999", alpha=0.8, label="random pairs")
    ax.hist(reex, bins, density=True, color="#0072B2", alpha=0.75, label="re-export pairs")
    ax.axvline(tau, color="#D55E00", lw=0.8)
    ax.set_yscale("log"); plainlog(ax); ax.set_xlabel("cosine similarity", labelpad=1); ax.set_ylabel("density")
    yl = ax.get_ylim()
    # threshold label and legend sit in the empty low-similarity region (QA 2026-11-16: no overlap with bars)
    ax.annotate(f"{tau:.3f}", xy=(tau, yl[1] * 0.25), xytext=(0.62, yl[1] * 0.25), ha="center", va="center",
                fontsize=7.2, color="#D55E00", arrowprops=dict(arrowstyle="-", color="#D55E00", lw=0.6))
    ax.set_title("(a) panoramic", loc="left", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.98), fontsize=7.2, frameon=False, handlelength=0.9,
              borderaxespad=0.0, labelspacing=0.2)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax = axs[1]
    ax.axvspan(A["io_tau_split"], 1.0, color="#E69F00", alpha=0.10, lw=0)
    ax.axvspan(A["io_tau_hi"], 1.0, color="#D55E00", alpha=0.14, lw=0)
    ax.hist(rand_io, bins, density=True, color="#999999", alpha=0.8, label="random pairs")
    ax.hist(aug, bins, density=True, color="#CC79A7", alpha=0.75, label="augmented copies")
    for t in (A["io_tau_split"], A["io_tau_hi"]):
        ax.axvline(t, color="#D55E00", lw=0.8)
    ax.set_yscale("log"); plainlog(ax); ax.set_xlabel("cosine similarity", labelpad=1)
    yl = ax.get_ylim()
    for t in (A["io_tau_split"], A["io_tau_hi"]):
        ax.text(t - 0.012, yl[1] * 0.5, f"{t:.3f}", ha="right", va="top", fontsize=7.2, color="#D55E00", rotation=90,
                bbox=dict(fc="white", ec="none", pad=0.4))
    ax.set_title("(b) intraoral", loc="left", pad=19)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.08, 0.98), ncol=1, fontsize=7.2, frameon=False, borderaxespad=0.1,
              handlelength=0.9, labelspacing=0.2)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
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
    return out


def fig_analysis():
    small()
    fam_of = _family_of_px()
    D = J("summary_det.json") or {}
    tr = D.get("px_train_instances", {})
    fig = plt.figure(figsize=(TW, 2.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.95, 0.95, 1.2, 1.5], wspace=0.95)
    vals = {}
    order = ["y11n", "y11s", "y11m", "reexp_y11s", "rtdetrl"]
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
            ax.plot(tt, 100 * rr, "-", color=COL(k), lw=1.0, label=MODEL_LABEL[k])
            ax.fill(tt, 100 * rr, color=COL(k), alpha=0.12)
        ABR = {"pathology": "path.", "periodontal / soft tissue": "perio.", "tooth status / development": "tooth",
               "anatomy": "anat.", "restoration / device": "restor."}
        ax.set_xticks(th, [ABR[f] for f in present], fontsize=7.2)
        ax.tick_params(axis="x", pad=1)
        ax.set_rlabel_position(np.degrees(th[1] / 2) if len(th) > 1 else 45)   # between the first two spokes
        top = max([100 * max(v.values()) for v in vals[task].values()] + [10]) * 1.15
        tk = [5, 10, 20, 30, 40, 50, 60]; tk = [x for x in tk if x < top][-1:]
        ax.set_ylim(0, top); ax.set_yticks(tk, [str(x) for x in tk], fontsize=7.2, color="#555555")
        ax.set_title(f"({'ab'[j]}) {'PX-DET, box' if task == 'pxdet' else 'PX-SEG, mask'}", loc="left",
                     fontsize=7.6, pad=12, x=-0.25)
        if j == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(-0.45, -0.2), ncol=1, fontsize=7.2, frameon=False)
    # (c) per-class AP vs training instances
    ax = fig.add_subplot(gs[2])
    M = _px_models("pxdet")
    vals["pxdet_per_class"] = {}
    for k in [o for o in order if o in M]:
        pc = M[k]
        xs = [max(tr.get(c, 0), 1) for c in pc]
        ax.scatter(xs, [100 * v for v in pc.values()], s=7, color=COL(k),
                   marker="o", lw=0.7, label=MODEL_LABEL[k], alpha=0.85, zorder=3)
        vals["pxdet_per_class"][k] = pc
    ax.set_xscale("log"); plainlog(ax, "x")
    ax.axvspan(0.7, 100, color="#D55E00", alpha=0.06, lw=0); ax.axvspan(1000, 3e4, color="#0072B2", alpha=0.06, lw=0)
    # bin means of the validation-selected PX-DET family (same rule as the tables)
    import make_tables as _MT
    _v = _MT.verdict_dense("pxdet")
    selk = _v["selected"] if _v else "y11s"
    vals["pxdet_bin_means_family"] = selk
    G = D.get("groups", {}).get(f"pxdet_{selk}", {})
    for lo, hi, bn in ((1, 100, "tail"), (100, 1000, "medium"), (1000, 2e4, "head")):
        v = G.get(f"test_box_{bn}_map50_95_mean")
        if v is not None:
            ax.hlines(100 * v, lo, hi, color=COL(selk), lw=1.2, linestyles=":")
            ax.text(np.sqrt(lo * hi), 100 * v + 2.5, f"{100 * v:.1f}", ha="center", fontsize=7.2, color=COL(selk),
                    bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85), zorder=4)
    ax.set_xlim(0.7, 3e4); ax.set_ylim(-3, 80)
    ax.set_xlabel("training instances (log)", labelpad=1); ax.set_ylabel("AP$_{50:95}$", labelpad=2)
    ax.text(10, 74, "tail", fontsize=7.2, color="#D55E00", ha="center")
    ax.text(310, 74, "mid", fontsize=7.2, color="#555555", ha="center")
    ax.text(5000, 66, "head", fontsize=7.2, color="#0072B2", ha="center")      # staggered: no overlap with "mid"
    # model colours are keyed once, by the legend under panel (a) (QA 2026-11-16: this legend covered points)
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
            x = 3.0 + 0.62 * i
            ax.bar(x, 100 * m["test_macro_ap"], 0.4, color="#E69F00" if k == "multi_lattice" else "#D55E00")
            lo, hi = m["test_ci95"]
            ax.errorbar(x, 100 * m["test_macro_ap"], yerr=[[100 * (m["test_macro_ap"] - lo)], [100 * (hi - m["test_macro_ap"])]],
                        color="#333333", lw=0.6, capsize=1.5)
            vals["cls"][k] = m["test_macro_ap"]
    ax.set_xticks([0, 1, 2, 3.0, 3.62], ["ViT-S", "ViT-B", "CNX-T", "C1", "C2"], fontsize=7.2, rotation=35, ha="right",
                  rotation_mode="anchor")
    ax.set_ylim(40, 112); ax.set_yticks([40, 60, 80, 100]); ax.set_ylabel("IO-CLS macro-AP")
    ax.legend(loc="upper left", fontsize=7.2, ncol=2, handlelength=0.8, columnspacing=0.6, borderaxespad=0.2)
    ax.set_title("(d) IO-CLS read-outs", loc="left")
    save(fig, "fig_analysis")
    write_json(RESULTS / "fig_analysis_values.json", dict(provenance="run (scripts/make_figures.py)", **vals))


# ----------------------------------------------------------------------------- Fig. 6 leak + transfer
def fig_leak():
    small()
    D = J("summary_det.json") or {"runs": {}, "groups": {}}
    fig, axs = plt.subplots(1, 3, figsize=(TW, 1.75), gridspec_kw=dict(wspace=0.4))
    vals = {}

    def pbars(ax, labels, a, b, la, lb, ca, cb, title, ylabel, nd=1, legloc="upper left"):
        xs = np.arange(len(labels))
        ax.bar(xs - 0.19, a, 0.36, color=ca, label=la)
        ax.bar(xs + 0.19, b, 0.36, color=cb, label=lb)
        for x, u, v in zip(xs, a, b):
            ax.text(x - 0.19, u + 0.4, f"{u:.{nd}f}", ha="center", fontsize=7.2)
            ax.text(x + 0.19, v + 0.4, f"{v:.{nd}f}", ha="center", fontsize=7.2)
        ax.set_xticks(xs, labels, fontsize=7.2)
        ax.set_ylim(0, max(max(a), max(b)) * 1.6 + 1)
        ax.set_title(title, loc="left"); ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", ncol=2, fontsize=7.2, frameon=False, handlelength=0.9, columnspacing=0.8)
    # (a) L1 controlled swap
    L = D.get("L1_swap")
    if L and L.get("map50_95_clean"):
        c, l = [100 * v for v in L["map50_95_clean"]], [100 * v for v in L["map50_95_leaky"]]
        pbars(axs[0], [f"seed {i}" for i in range(len(c))], c, l, "clean control", "leaky", "#0072B2", "#D55E00",
              "(a) controlled swap", "PX-DET mAP$_{50:95}$", nd=2, legloc="upper left")
        for i, (u, v) in enumerate(zip(c, l)):
            axs[0].text(i + 0.19, v + 1.8, f"{100 * (v - u) / u:+.1f}%", ha="center", fontsize=7.2, fontweight="bold",
                        color="#D55E00")
        vals["L1"] = L
    else:
        pending(axs[0], "(a) controlled swap")
    # (c) L2 (IO-CLS) and L3 (IO-DET)
    LC = J("summary_leak_cls.json")
    r3 = D.get("runs", {}).get("iodet_shipped_y11s_s0")
    io = D.get("groups", {}).get("iodet_y11s", {}).get("test_box_map50_95_mean")
    if LC and LC["models"].get("vitb14_lattice") and r3:
        m = LC["models"]["vitb14_lattice"]
        pbars(axs[1], ["IO-CLS\nmacro-AP", "IO-DET\nmAP$_{50:95}$"],
              [100 * m["common_test_trained_on_lineage_split"], 100 * io],
              [100 * m["common_test_trained_on_hash_split"], 100 * r3["test"]["box"]["map50_95"]],
              "copy-lineage-disjoint", "leaky", "#0072B2", "#D55E00", "(b) intraoral splits", "score")
        vals["L2_L3"] = True
    else:
        pending(axs[1], "(b) intraoral splits")
    # (d) DENTEX transfer
    G = D.get("groups", {})
    a, c = G.get("dentex_coco_y11s"), G.get("dentex_du_y11s")
    if a and c:
        ax = axs[2]
        for i, (g, lab, col) in enumerate(((a, "COCO", "#999999"), (c, "DentUnity", "#0072B2"))):
            v = [100 * x for x in g["test_box_map50_95_per_seed"]]
            ax.scatter([i] * len(v), v, color=col, s=12, zorder=3)
            ax.hlines(np.mean(v), i - 0.25, i + 0.25, color=col, lw=1.4)
            ax.text(i + 0.28, np.mean(v), f"{np.mean(v):.1f}", va="center", fontsize=7.2)
        allv = [100 * x for g in (a, c) for x in g["test_box_map50_95_per_seed"]]
        ax.set_ylim(0, max(allv) * 1.5)
        dm = 100 * (np.mean(c["test_box_map50_95_per_seed"]) - np.mean(a["test_box_map50_95_per_seed"]))
        ns = min(len(a["test_box_map50_95_per_seed"]), len(c["test_box_map50_95_per_seed"]))
        ax.text(0.03, 0.97, f"mean change {dm:+.1f}\n(seeds: {ns}, 50 images)", transform=ax.transAxes, va="top",
                fontsize=7.2, bbox=dict(fc="white", ec="#cccccc", lw=0.4, pad=1.5))
        xt, xl = [0, 1], ["COCO\ninit", "DentUnity\ninit"]
        # UR 2026-11-16: the PX-DET D1-vs-base pair is no longer drawn on this DENTEX axis (different benchmark,
        # overlapping tick labels, not in the caption); D1 is reported in Sec. 5.4 and Tab. J.1.
        ax.set_xticks(xt, xl, fontsize=7.2); ax.set_xlim(-0.6, max(xt) + 0.8)
        ax.set_title("(c) DENTEX transfer", loc="left"); ax.set_ylabel("mAP$_{50:95}$")
        vals["dentex"] = {"coco": a["test_box_map50_95_per_seed"], "dentunity": c["test_box_map50_95_per_seed"]}
    else:
        pending(axs[2], "(c) DENTEX transfer")
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
    fig, axs = plt.subplots(1, 2, figsize=(CW, 1.5), gridspec_kw=dict(wspace=0.4))
    vals = {}
    for ax, (lab, C, col) in zip(axs, (("(a) panoramic", px, "#0072B2"),
                                       ("(b) intraoral", io, "#CC79A7"))):
        sz = np.array(list(C.values()))
        edges = [1, 2, 3, 5, 10, 100, 10 ** 6]
        h, _ = np.histogram(sz, edges)
        ax.bar(range(len(h)), h, color=col)
        ax.set_xticks(range(len(h)), ["1", "2", "3-4", "5-9", "10-99", "100+"], fontsize=7.2, rotation=45)
        ax.set_yscale("log"); plainlog(ax); ax.set_title(lab, loc="left", fontsize=7.4); ax.set_ylabel("lineages")
        vals[lab[:3]] = {"hist": h.tolist(), "n_lineages": int(len(sz)), "max_size": int(sz.max())}
    save(fig, "fig_supp_lineage")
    write_json(RESULTS / "fig_supp_lineage_values.json", dict(provenance="run (scripts/make_figures.py)", **vals))


if __name__ == "__main__":
    which = sys.argv[2:] or ["pipeline", "overview", "linkqc", "analysis", "leak", "supp_lineage"]
    for w in which:
        globals()[f"fig_{w}"]()
        print("figure", w, "ok", flush=True)
