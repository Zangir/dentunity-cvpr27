"""Page-1 teaser: one example per task with its annotations, plus a leaked copy pair.
Privacy: images were chosen by eye to carry no burned-in text, crops trim the borders,
and no file name is ever rendered. Reads data/bench.json; writes figures/fig_teaser.pdf."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from PIL import Image
from paper_style import style
from unidental.common import DATA, REPO, RESULTS, write_json
from unidental.taxonomy import IO_CLS_CLASSES, IO_DET_CLASSES, PX_CLASSES, UNIFIED, to_unified

FAMC = {"pathology": "#D55E00", "periodontal / soft tissue": "#CC79A7", "tooth status / development": "#009E73",
        "anatomy": "#F0E442", "restoration / device": "#56B4E9"}
PICK = {"cls": ["ac31ffb4bdad55ae", "bfa0438091a88def"], "iodet": "b401b560",
        "pxdet": "4ec564156e981994", "pxseg": "7bc521906497b50b", "leak": "dfe875ab452eee9d"}
FAM = {c["id"]: c["family"] for c in UNIFIED}


def find(rows, pre):
    return next(r for r in rows if r["sha256"].startswith(pre))


def show(ax, path, crop=0.03):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im = im.crop((int(crop * w), int(crop * h), int((1 - crop) * w), int((1 - crop) * h)))
    ax.imshow(im, extent=(crop, 1 - crop, 1 - crop, crop))
    ax.set_xlim(crop, 1 - crop); ax.set_ylim(1 - crop, crop); ax.axis("off")
    return w, h


def ratio(path, crop):
    w, h = Image.open(path).size
    return (w * (1 - 2 * crop)) / (h * (1 - 2 * crop))


def main():
    b = json.loads((DATA / "bench.json").read_text())
    style()
    uc, ud, up = to_unified("io_cls"), to_unified("io_det"), to_unified("px")
    c1, c2 = [find(b["io_cls"], p) for p in PICK["cls"]]
    rd = find(b["io_det"], PICK["iodet"])
    rp, rs = find(b["px"], PICK["pxdet"]), find(b["px"], PICK["pxseg"])
    rl = find(b["px"], PICK["leak"])
    fam = [x for x in b["px"] if x["family"] == rl["family"]]
    tr = next(x for x in fam if x["shipped_split"] == "train")
    te = next(x for x in fam if x["shipped_split"] == "test")
    H, gap = 1.0, 0.06
    wa = max(ratio(c1["path"], 0), ratio(c2["path"], 0)) * (H - gap) / 2
    cols = [wa, ratio(rd["path"], 0) * H, ratio(rp["path"], .03) * H, ratio(rs["path"], .03) * H,
            max(ratio(tr["path"], .03), ratio(te["path"], .03)) * (H - gap) / 2]
    W = sum(cols) + gap * (len(cols) - 1)
    figw = 14.0
    fig = plt.figure(figsize=(figw, figw * (H + 0.32) / W))
    sx, sy = 1 / W, 1 / (H + 0.32)
    y0 = 0.16
    x = 0.0

    def axes(xx, yy, w, h):
        return fig.add_axes([xx * sx, (y0 + yy) * sy, w * sx, h * sy])

    def tag(ax, text):
        ax.text(0.03, 0.96, text, transform=ax.transAxes, fontsize=6.5, color="white", va="top",
                bbox=dict(fc="black", alpha=0.6, pad=1.2, lw=0))
    hh = (H - gap) / 2
    for k, r in enumerate((c1, c2)):
        ax = axes(x, (1 - k) * (hh + gap), ratio(r["path"], 0) * hh, hh); show(ax, r["path"], 0.0); ax.set_aspect("auto")
        tag(ax, " + ".join(IO_CLS_CLASSES[c] for c in r["labels"]))
    fig.text((x + cols[0] / 2) * sx, 0.02, "(a) IO-CLS: multi-label", ha="center", fontsize=9)
    x += cols[0] + gap
    ax = axes(x, 0, cols[1], H); show(ax, rd["path"], 0.0); ax.set_aspect("auto")
    for c, cx, cy, w, h in rd["boxes_cxcywh"]:
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec=FAMC[FAM[ud[c]]], lw=1.4))
    tag(ax, "caries boxes")
    fig.text((x + cols[1] / 2) * sx, 0.02, "(b) IO-DET: lesion boxes", ha="center", fontsize=9)
    x += cols[1] + gap
    ax = axes(x, 0, cols[2], H); show(ax, rp["path"]); ax.set_aspect("auto")
    for inst in rp["instances"]:
        a0, b0, a1, b1 = inst["box"]
        ax.add_patch(Rectangle((a0, b0), a1 - a0, b1 - b0, fill=False, ec=FAMC[FAM[up[inst['c']]]], lw=0.9))
    fig.text((x + cols[2] / 2) * sx, 0.02, "(c) PX-DET: 31 classes of findings and dental work", ha="center", fontsize=9)
    x += cols[2] + gap
    ax = axes(x, 0, cols[3], H); show(ax, rs["path"]); ax.set_aspect("auto")
    for line in open(rs["label_path"]):
        t = line.split()
        if len(t) >= 7:
            p = np.array([float(v) for v in t[1:]]).reshape(-1, 2)
            col = FAMC[FAM[up[int(t[0])]]]
            ax.add_patch(Polygon(p, closed=True, fc=col, ec=col, alpha=0.45, lw=0.5))
    fig.text((x + cols[3] / 2) * sx, 0.02, "(d) PX-SEG: instance masks", ha="center", fontsize=9)
    x += cols[3] + gap
    w1, w2 = ratio(tr["path"], .03) * hh, ratio(te["path"], .03) * hh
    ax = axes(x + (cols[4] - w1) / 2, hh + gap, w1, hh); show(ax, tr["path"]); ax.set_aspect("auto")
    tag(ax, "shipped train (original)")
    ax = axes(x + (cols[4] - w2) / 2, 0, w2, hh); show(ax, te["path"]); ax.set_aspect("auto")
    tag(ax, "shipped test (re-export)")
    A = json.loads((RESULTS / "dataset_stats.json").read_text())["audit"]
    fig.text((x + cols[4] / 2) * sx, 0.02,
             f"(e) leak: {100 * A['px_shipped_test_share_with_train_lineage']:.1f}% of shipped test are copies",
             ha="center", fontsize=9)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=v, label=k) for k, v in FAMC.items()], ncol=5, fontsize=7.5,
               loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    out = REPO / "figures"
    fig.savefig(out / "fig_teaser.pdf"); fig.savefig(out / "fig_teaser.png", dpi=220)


if __name__ == "__main__":
    main()
