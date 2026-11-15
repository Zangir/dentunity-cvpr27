"""Dataset figures (read only results/dataset_stats.json + data/bench.json):
fig_stats.pdf (per-task split counts, unified class distribution, file->family funnel)
fig_pipeline.pdf (construction pipeline stage cards)."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from paper_style import style
from dentunity.common import DATA, RESULTS, REPO, write_json
from dentunity.taxonomy import UNIFIED, IO_CLS_CLASSES, IO_DET_CLASSES, PX_CLASSES, to_unified

FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)
FAMC = {"pathology": "#D55E00", "periodontal / soft tissue": "#CC79A7", "tooth status / development": "#009E73",
        "anatomy": "#999999", "restoration / device": "#0072B2"}
SPLITC = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}


def stats_fig():
    S = json.loads((RESULTS / "dataset_stats.json").read_text())
    b = json.loads((DATA / "bench.json").read_text())
    style()
    fig = plt.figure(figsize=(14, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 3.2, 1.05], wspace=0.28)
    # (a) per-task split counts
    ax = fig.add_subplot(gs[0])
    tasks = [("IO-CLS", "io_cls"), ("IO-DET", "io_det"), ("PX-DET", "px_det"), ("PX-SEG", "px_seg")]
    left = np.zeros(4)
    for s in ("train", "val", "test"):
        v = np.array([S[k]["split_images"][s] for _, k in tasks])
        ax.barh(range(4), v, left=left, color=SPLITC[s], label=s, height=0.62)
        left += v
    for i, t in enumerate(left):
        ax.text(t + 80, i, f"{int(t):,}", va="center", fontsize=9)
    ax.set_yticks(range(4), [t for t, _ in tasks]); ax.invert_yaxis()
    ax.set_xlabel("images"); ax.set_xlim(0, left.max() * 1.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=3); ax.set_title("(a) images per task and split", fontsize=10)
    # (b) unified class distribution (annotations per concept, log scale)
    ax = fig.add_subplot(gs[1])
    cnt = Counter()
    uc, ud, up = to_unified("io_cls"), to_unified("io_det"), to_unified("px")
    for r in b["io_cls"]:
        for c in r["labels"]:
            cnt[(uc[c], "cls")] += 1
    for r in b["io_det"]:
        for x in r["boxes_cxcywh"]:
            cnt[(ud[x[0]], "iodet")] += 1
    for r in b["px"]:
        if r["role"] == "canonical":
            for x in r["instances"]:
                cnt[(up[x["c"]], "px")] += 1
    order = sorted(UNIFIED, key=lambda c: (list(FAMC).index(c["family"]),
                                          -sum(cnt[(c["id"], k)] for k in ("cls", "iodet", "px"))))
    xs = np.arange(len(order))
    w = 0.28
    for k, (key, lab, hatch) in enumerate([("cls", "IO-CLS labels", "//"), ("iodet", "IO-DET boxes", ".."),
                                           ("px", "PX boxes/masks", "")]):
        vals = [cnt[(c["id"], key)] for c in order]
        ax.bar(xs + (k - 1) * w, vals, w, color=[FAMC[c["family"]] for c in order], hatch=hatch,
               edgecolor="white", linewidth=0.3, label=lab)
    ax.set_yscale("log"); ax.set_ylim(0.8, 4e5)
    ax.set_xticks(xs, [c["name"] for c in order], rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("annotations (log)")
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=v, label=k) for k, v in FAMC.items()]
    hs += [Patch(facecolor="white", edgecolor="k", hatch=h, label=l)
           for l, h in [("IO-CLS", "//"), ("IO-DET", ".."), ("PX", "")]]
    ax.legend(handles=hs, ncol=4, fontsize=7.5, loc="upper right")
    ax.set_title("(b) one taxonomy: 36 unified concepts, 5 families", fontsize=10)
    # (c) panoramic files -> copy families -> lineages
    ax = fig.add_subplot(gs[2])
    A = S["audit"]
    vals = [S["px_files"]["files"], S["px_det"]["images"], A["px_lineages_final"]]
    labs = ["shipped\nfiles", "copy\nfamilies", "split\nlineages"]
    ax.bar(range(3), vals, color=["#999999", "#0072B2", "#56B4E9"], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 250, f"{v:,}", ha="center", fontsize=9)
    ax.set_xticks(range(3), labs, fontsize=8); ax.set_ylim(0, vals[0] * 1.18)
    ax.set_title("(c) panoramic release de-duplicated", fontsize=10)
    fig.savefig(FIG / "fig_stats.pdf"); fig.savefig(FIG / "fig_stats.png", dpi=200)
    write_json(RESULTS / "fig_stats_values.json", {"provenance": "run",
               "annotations_per_concept": {f"{c['name']}|{k}": cnt[(c['id'], k)] for c in UNIFIED
                                           for k in ("cls", "iodet", "px") if cnt[(c['id'], k)]}})


def card(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="white", ec=color, lw=1.4))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.16), w, 0.16, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=color, ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 0.08, title, ha="center", va="center", color="white", fontsize=9.5,
            fontweight="bold")
    for i, l in enumerate(lines):
        ax.text(x + 0.012, y + h - 0.25 - i * 0.135, l, ha="left", va="center", fontsize=7.6)


def pipeline_fig():
    S = json.loads((RESULTS / "dataset_stats.json").read_text())
    A = S["audit"]
    style()
    fig, ax = plt.subplots(figsize=(14, 2.5))
    ax.set_xlim(0, 5.05); ax.set_ylim(0, 1.02); ax.axis("off")
    cols = ["#999999", "#CC79A7", "#D55E00", "#0072B2", "#009E73"]
    cards = [
        ("1  Inventory", [f"{S['px_files']['files']:,} panoramic files", f"{S['io_cls']['images']:,} + {A['io_det_files']:,} intraoral files",
                          "sha256, size, shipped split", "no filenames kept"]),
        ("2  Harmonise", ["36 unified concepts", "5 clinical families", "4 tasks, 2 modalities",
                          "per-source metadata"]),
        ("3  Link copies", ["exact hash + export source", "registered copies", "calibrated embedding",
                            "edges (mutual k-NN)"]),
        ("4  Split lineages", ["whole lineages to one split", "joint over all tasks",
                               "stratified 70/10/20", "one canonical file/family"]),
        ("5  Audit + release", ["0 test images with a", "training copy (every task)", "sha256 manifests,",
                                "build kit, eval code"]),
    ]
    for i, (t, ls) in enumerate(cards):
        card(ax, 0.03 + i * 1.0, 0.04, 0.9, 0.94, t, ls, cols[i])
        if i < 4:
            ax.annotate("", xy=(0.03 + (i + 1) * 1.0 - 0.005, 0.5), xytext=(0.03 + i * 1.0 + 0.905, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4))
    fig.savefig(FIG / "fig_pipeline.pdf"); fig.savefig(FIG / "fig_pipeline.png", dpi=200)


if __name__ == "__main__":
    stats_fig()
    pipeline_fig()
