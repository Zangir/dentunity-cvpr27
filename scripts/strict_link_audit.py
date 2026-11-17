"""Strict-link re-audit (plan amendment 2026-11-17 09:32), CPU only, no training.

Intraoral: rebuild the copy graph over IO-CLS + IO-DET images from the cached link embeddings
with strict links only, i.e. identical sha256 and mutual nearest neighbours above the 0.950
precision threshold. The 0.731 recall edges and the cached preliminary IO-CLS graph are not used.
Every intraoral leak figure of the audit is recomputed under this graph, next to the value under
all links (results/dataset_stats.json). Panoramic: the shipped-split leak is recomputed with
export keys, sha256 and registration only (no embedding edges).

  nice -n 10 python3 scripts/strict_link_audit.py   ->  results/strict_link_audit.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from dentunity.common import DATA, FEAT_ROOT, PANO_PAIRS, RESULTS, read_json, write_json  # noqa: E402
from dentunity.lineage import components, key_edges, mutual_knn_edges, size_gated_edges  # noqa: E402
from dentunity.splits import contamination  # noqa: E402


def emb(group):
    z = np.load(DATA / f"linkemb_{group}.npz")
    return z["E"].astype(np.float32), [str(p) for p in z["paths"]]


def intraoral(A):
    b = read_json(DATA / "bench.json")
    cls, det = b["io_cls"], b["io_det"]                    # released images, in build order
    lin = read_json(FEAT_ROOT / "results/lineage_intraoral.json")
    tau_hi = lin["tau_precision_first_null999"]
    z = np.load(FEAT_ROOT / "data/compact/intraoral_linkemb.npz")
    comp = np.load(FEAT_ROOT / "data/compact/intraoral_vits14.npz")
    kind = json.loads(str(comp["extra"]))["kind"]
    orig = [str(p) for p, k in zip(comp["paths"], kind) if k == "orig"]
    pos = {p: i for i, p in enumerate(orig)}
    Ec = z["E"][[pos[r["path"]] for r in cls]].astype(np.float32)
    Ed_all, pd_ = emb("io_det")
    Ed = Ed_all[[pd_.index(r["path"]) for r in det]]
    nC, nD = len(cls), len(det)
    E = np.concatenate([Ec, Ed])
    lab = components(nC + nD, [key_edges([r["sha256"] for r in cls + det]), mutual_knn_edges(E, tau_hi, k=3)])
    # reproduction check: the all-links graph of dentunity/build.py must give the published values
    sizes = [(r["width"], r["height"]) for r in cls + det]
    e_new = [e for e in mutual_knn_edges(E, tau_hi, k=3) + size_gated_edges(E, sizes, lin["tau_split"], k=10)
             if max(e[0], e[1]) >= nC]
    lab_all = components(nC + nD, [key_edges([r["lineage_src"] for r in cls]),
                                   key_edges([r["sha256"] for r in cls + det]), e_new])
    det_inv = {r["path"]: r for r in read_json(DATA / "inv_io_det.json")}
    shipped = [det_inv[r["path"]]["shipped_split"] for r in det]
    old = read_json(FEAT_ROOT / "results/splits_cls.json")
    osp = {r["path"]: k for k, v in old["splits"].items() for r in v}
    out = {"tau_precision": tau_hi, "edges": "sha256 + mutual 3-NN above the precision threshold",
           "lineages": int(len(set(lab.tolist()))),
           "largest_lineage": int(max(Counter(lab.tolist()).values())),
           "io_cls_hash_split_test_share_with_train_lineage": contamination([osp[r["path"]] for r in cls], lab[:nC].tolist()),
           "io_det_shipped_val_share_with_train_lineage": contamination(shipped, lab[nC:].tolist(), ("val",), ("train",))}
    indep = [osp_cached for osp_cached in [r.get("split_src") for r in cls]] + \
            ["train" if s == "train" else "test" for s in shipped]
    te_det = [i for i in range(nC, nC + nD) if indep[i] == "test"]
    tr_cls = {int(lab[i]) for i in range(nC) if indep[i] == "train"}
    out["io_det_shipped_val_share_with_cls_train_lineage"] = float(np.mean([int(lab[i]) in tr_cls for i in te_det]))
    te_cls = [i for i in range(nC) if indep[i] == "test"]
    tr_det = {int(lab[i]) for i in range(nC, nC + nD) if indep[i] == "train"}
    out["io_cls_test_share_with_shipped_det_train_lineage"] = float(np.mean([int(lab[i]) in tr_det for i in te_cls]))
    ours = [r["split"] for r in cls + det]
    out["dentunity_test_contamination"] = contamination(ours, lab.tolist())
    out["dentunity_val_contamination"] = contamination(ours, lab.tolist(), ("val",))
    out["all_links"] = {k: A[k] for k in ("io_cls_hash_split_test_share_with_train_lineage",
                                          "io_det_shipped_val_share_with_train_lineage",
                                          "io_det_shipped_val_share_with_cls_train_lineage",
                                          "io_cls_test_share_with_shipped_det_train_lineage",
                                          "io_test_contamination", "io_val_contamination")}
    out["all_links"]["largest_lineage"] = A["io_largest_component"]
    out["all_links"]["lineages"] = A["io_components"]
    out["reproduction_all_links_hash_test_leak"] = contamination([osp[r["path"]] for r in cls], lab_all[:nC].tolist())
    out["reproduction_all_links_largest"] = int(max(Counter(lab_all.tolist()).values()))
    return out


def panoramic(A):
    px = read_json(DATA / "inv_px.json")
    by_sha = {}
    for i, r in enumerate(px):
        by_sha.setdefault(r["sha256"], i)
    reg = [(by_sha[p["left"]], by_sha[p["right"]]) for p in read_json(PANO_PAIRS)
           if p["left"] in by_sha and p["right"] in by_sha]
    base = components(len(px), [key_edges([r["source_key"] for r in px]), key_edges([r["sha256"] for r in px]), reg])
    shipped = [r["shipped_split"] for r in px]
    return {"edges": "export key + sha256 + registration (no embedding edges)",
            "px_shipped_test_share_with_train_lineage": contamination(shipped, base.tolist(), ("test",), ("train",)),
            "px_shipped_valid_share_with_train_lineage": contamination(shipped, base.tolist(), ("valid",), ("train",)),
            "all_links": {"px_shipped_test_share_with_train_lineage": A["px_shipped_test_share_with_train_lineage"],
                          "px_shipped_valid_share_with_train_lineage": A["px_shipped_valid_share_with_train_lineage"]}}


def main():
    A = read_json(RESULTS / "dataset_stats.json")["audit"]
    out = {"provenance": "run (scripts/strict_link_audit.py; plan amendment 2026-11-17 09:32)",
           "intraoral_strict": intraoral(A), "panoramic_keys_only": panoramic(A)}
    write_json(RESULTS / "strict_link_audit.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
