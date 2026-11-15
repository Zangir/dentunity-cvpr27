"""Build the benchmark: lineages, cross-task-consistent splits, manifests, audits.

  python -m dentunity.build      -> data/bench.json (local paths, private)
                                    release/manifests/*.csv (sha256-keyed, public)
                                    results/build_audit.json, results/dataset_stats.json
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

from .common import DATA, FEAT_ROOT, PANO_PAIRS, RELEASE, RESULTS, read_json, write_json
from .lineage import components, key_edges, mutual_knn_edges, null_quantile, size_gated_edges
from .splits import contamination, stratified_group_split
from .taxonomy import IO_CLS_CLASSES, IO_DET_CLASSES, PX_CLASSES, to_unified

SEED = "dentunity-v1"


def _emb(group):
    z = np.load(DATA / f"linkemb_{group}.npz")
    return z["E"].astype(np.float32), [str(p) for p in z["paths"]]


def _hid(text):  # public lineage / image ids carry no filename information
    return hashlib.sha256(f"{SEED}:{text}".encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- intraoral
def build_intraoral(audit):
    """One copy-lineage graph over ALL intraoral images (classification + detection),
    then one joint lineage-level split, so no copy of any test image (of either task)
    is in either task's training set. Augmented-folder images are excluded from the
    benchmark and only attached afterwards (for the leak analysis)."""
    cls = read_json(DATA / "inv_io_cls.json")
    det_all = read_json(DATA / "inv_io_det.json")
    aug = read_json(DATA / "inv_io_aug.json")
    lin = read_json(FEAT_ROOT / "results/lineage_intraoral.json")
    # exact duplicates inside the detection subset: keep one file per sha256
    by = defaultdict(list)
    for r in det_all:
        by[r["sha256"]].append(r)
    det = [max(v, key=lambda r: (len(r["boxes_cxcywh"]), r["path"])) for v in by.values()]
    det.sort(key=lambda r: r["path"])
    keep_paths = {r["path"] for r in det}
    audit["io_det_files"] = len(det_all)
    audit["io_det_exact_duplicate_files_removed"] = len(det_all) - len(det)
    audit["io_det_duplicate_groups_with_conflicting_boxes"] = sum(
        len({json.dumps(r["boxes_cxcywh"]) for r in v}) > 1 for v in by.values() if len(v) > 1)
    tau, tau_hi = lin["tau_split"], lin["tau_precision_first_null999"]
    z = np.load(FEAT_ROOT / "data/compact/intraoral_linkemb.npz")
    comp = np.load(FEAT_ROOT / "data/compact/intraoral_vits14.npz")
    kind = json.loads(str(comp["extra"]))["kind"]
    orig_paths = [str(p) for p, k in zip(comp["paths"], kind) if k == "orig"]
    pos = {p: i for i, p in enumerate(orig_paths)}
    Ec = z["E"][[pos[r["path"]] for r in cls]].astype(np.float32)
    Ed_all, pd_ = _emb("io_det")
    assert pd_ == [r["path"] for r in det_all]
    Ed = Ed_all[[pd_.index(r["path"]) for r in det]]
    Ea, pa = _emb("io_aug")
    assert pa == [r["path"] for r in aug]
    E64, p64 = _emb("io_cls64")   # reproduction check of the embedding recipe
    audit["link_embedding_reproduction_min_cos"] = float((E64 * z["E"][[pos[p] for p in p64]]).sum(1).min())
    audit["io_tau_split"], audit["io_tau_hi"] = tau, tau_hi
    nC, nD = len(cls), len(det)
    E = np.concatenate([Ec, Ed])
    sizes = [(r["width"], r["height"]) for r in cls + det]
    sha_c = {r["sha256"] for r in cls}
    audit["io_det_bytes_identical_to_cls"] = sum(r["sha256"] in sha_c for r in det)
    audit["io_det_max_sim_to_cls_ge_tau_hi"] = int(((Ed @ Ec.T).max(1) >= tau_hi).sum())
    cls_lin = key_edges([r["lineage_src"] for r in cls])
    e_sha = key_edges([r["sha256"] for r in cls + det])
    e_new = [e for e in mutual_knn_edges(E, tau_hi, k=3) + size_gated_edges(E, sizes, tau, k=10)
             if max(e[0], e[1]) >= nC]          # cls-cls edges come from the cached graph
    lab = components(nC + nD, [cls_lin, e_sha, e_new])
    audit["io_det_edges"] = len(e_new)
    audit["io_cls_lineages_cached"] = len({r["lineage_src"] for r in cls})
    groups = defaultdict(list)
    for i in range(nC + nD):
        groups[int(lab[i])].append(i)
    glab = {}
    for g, ms in groups.items():
        labs = Counter()
        for i in ms:
            if i < nC:
                labs.update({f"c{c}" for c in cls[i]["labels"]})
            else:
                labs.update({f"d{b[0]}" for b in det[i - nC]["boxes_cxcywh"]})
        glab[g] = (len(ms), labs)
    big = {g for g, ms in groups.items() if len(ms) > 0.05 * (nC + nD)}
    assign = stratified_group_split(list(groups), glab, SEED + ":io", force_train=big)
    audit["io_components"] = len(groups)
    audit["io_largest_component"] = max(len(ms) for ms in groups.values())
    audit["io_forced_train_components"] = len(big)
    sp = [assign[int(lab[i])] for i in range(nC + nD)]
    audit["io_test_contamination"] = contamination(sp, lab.tolist())
    audit["io_val_contamination"] = contamination(sp, lab.tolist(), ("val",))
    # previously used splits, measured under the same graph
    shipped = [r["shipped_split"] for r in det]
    assert len(keep_paths) == len(det)
    audit["io_det_shipped_val_share_with_train_lineage"] = contamination(
        shipped, lab[nC:].tolist(), ("val",), ("train",))
    audit["io_det_shipped_sizes"] = dict(Counter(shipped))
    old = read_json(FEAT_ROOT / "results/splits_cls.json")
    osp = {r["path"]: k for k, v in old["splits"].items() for r in v}
    audit["io_cls_hash_split_test_share_with_train_lineage"] = contamination(
        [osp[r["path"]] for r in cls], lab[:nC].tolist())
    audit["io_cls_hash_split_sizes"] = dict(Counter(osp[r["path"]] for r in cls))
    # independent per-task splits (cached cls split + shipped det split): cross-task leak
    indep = [r["split"] for r in cls] + ["train" if s == "train" else "test" for s in shipped]
    te_det = [i for i in range(nC, nC + nD) if indep[i] == "test"]
    tr_cls = {int(lab[i]) for i in range(nC) if indep[i] == "train"}
    audit["io_det_shipped_val_share_with_cls_train_lineage"] = float(
        np.mean([int(lab[i]) in tr_cls for i in te_det]))
    te_cls = [i for i in range(nC) if indep[i] == "test"]
    tr_det = {int(lab[i]) for i in range(nC, nC + nD) if indep[i] == "train"}
    audit["io_cls_test_share_with_shipped_det_train_lineage"] = float(
        np.mean([int(lab[i]) in tr_det for i in te_cls]))
    # augmented-folder images: precision-first nearest original (for the leak analysis)
    S = Ea @ E.T
    nn, mx = S.argmax(1), S.max(1)
    aug_rows = [dict(r, copy_of=int(nn[k]) if mx[k] >= tau_hi else None,
                     copy_of_split=sp[int(nn[k])] if mx[k] >= tau_hi else None,
                     max_sim=float(mx[k])) for k, r in enumerate(aug)]
    audit["io_aug_precision_copies_by_split"] = dict(Counter(r["copy_of_split"] for r in aug_rows))
    return {"io_cls": [dict(r, split=sp[i], split_src=r["split"], lineage=int(lab[i]))
                       for i, r in enumerate(cls)],
            "io_det": [dict(r, split=sp[nC + j], lineage=int(lab[nC + j])) for j, r in enumerate(det)],
            "io_aug": aug_rows}


# ----------------------------------------------------------------------------- panoramic
def build_panoramic(audit):
    px = read_json(DATA / "inv_px.json")
    n = len(px)
    E, pp = _emb("px")
    assert pp == [r["path"] for r in px]
    by_sha = {}
    for i, r in enumerate(px):
        by_sha.setdefault(r["sha256"], i)
    reg = [(by_sha[p["left"]], by_sha[p["right"]]) for p in read_json(PANO_PAIRS)
           if p["left"] in by_sha and p["right"] in by_sha]
    tau_hi = null_quantile(E)
    e_emb = mutual_knn_edges(E, tau_hi, k=3)
    base = components(n, [key_edges([r["source_key"] for r in px]),
                          key_edges([r["sha256"] for r in px]), reg])
    lab = components(n, [key_edges(base.tolist()), e_emb])
    audit.update({"px_tau_hi": tau_hi, "px_registered_links": len(reg),
                  "px_embedding_edges": len(e_emb),
                  "px_lineages_source_hash_registered": int(base.max() + 1),
                  "px_lineages_final": int(lab.max() + 1)})
    members = defaultdict(list)        # copy families (source key, hash, registered)
    for i in range(n):
        if not px[i]["quarantine"]:
            members[int(base[i])].append(i)
    fam_lin = {f: int(lab[ms[0]]) for f, ms in members.items()}

    def rank(i):
        r = px[i]
        npoly = sum(x["kind"] == "polygon" for x in r["instances"])
        return (-(r["width"] == 1615 and len(r["instances"]) > 0), -npoly, -len(r["instances"]),
                {"test": 0, "valid": 1, "train": 2}[r["shipped_split"]], r["sha256"])
    canon = {f: min(ms, key=rank) for f, ms in members.items()}
    glabels = defaultdict(lambda: [0, Counter()])
    for f, c in canon.items():
        glabels[fam_lin[f]][0] += 1
        glabels[fam_lin[f]][1].update({x["c"] for x in px[c]["instances"]})
    glabels = {g: (v[0], v[1]) for g, v in glabels.items()}
    assign = stratified_group_split(list(glabels), glabels, SEED + ":px")
    rows = []
    for f, ms in members.items():
        c, g = canon[f], fam_lin[f]
        for i in ms:
            r = px[i]
            role = "canonical" if i == c else "re-export"
            seg = (i == c and r["width"] == 1615
                   and any(x["kind"] == "polygon" for x in r["instances"]))
            rows.append(dict(r, lineage=g, family=f, split=assign[g], role=role, in_seg=bool(seg)))
    # shipped split audit under final lineages
    shipped = [r["shipped_split"] for r in px]
    audit["px_shipped_test_share_with_train_lineage"] = contamination(shipped, lab.tolist(), ("test",), ("train",))
    audit["px_shipped_valid_share_with_train_lineage"] = contamination(shipped, lab.tolist(), ("valid",), ("train",))
    sp = [r["split"] if r["role"] == "canonical" else "re-export" for r in rows]
    li = [r["lineage"] for r in rows]
    audit["px_test_contamination"] = contamination(sp, li)
    audit["px_test_contamination_incl_reexports_in_train"] = contamination(
        [r["split"] for r in rows], li)  # re-exports inherit their lineage split: still 0
    return rows


# ----------------------------------------------------------------------------- outputs
def write_manifests(io, px):
    m = RELEASE / "manifests"
    m.mkdir(parents=True, exist_ok=True)
    uc, ud, up = to_unified("io_cls"), to_unified("io_det"), to_unified("px")
    with open(m / "io_cls.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "split", "lineage", "source", "width", "height", "unified_labels"])
        for r in io["io_cls"]:
            w.writerow([r["sha256"], r["split"], _hid(f"io{r['lineage']}"), "ORAL-" + r["folder"],
                        r["width"], r["height"], " ".join(str(uc[c]) for c in r["labels"])])
    with open(m / "io_det.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "split", "lineage", "source", "width", "height", "n_boxes", "unified_classes"])
        for r in io["io_det"]:     # box geometry is re-derived from the upstream label files
            w.writerow([r["sha256"], r["split"], _hid(f"io{r['lineage']}"), "ORAL-YOLO", r["width"], r["height"],
                        len(r["boxes_cxcywh"]), " ".join(str(ud[b[0]]) for b in r["boxes_cxcywh"])])
    with open(m / "px.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "split", "lineage", "role", "in_seg", "source", "width", "height",
                    "annotation_sha256", "n_instances", "unified_classes"])
        for r in sorted(px, key=lambda r: (r["split"], r["lineage"], r["role"], r["sha256"])):
            w.writerow([r["sha256"], r["split"], _hid(f"px{r['lineage']}"), r["role"], int(r["in_seg"]),
                        "PANO-DXPD31", r["width"], r["height"], r["annotation_sha256"],
                        len(r["instances"]), " ".join(str(up[x["c"]]) for x in r["instances"])])


def stats(io, px, audit):
    S = {"provenance": "run"}
    cls = io["io_cls"]
    S["io_cls"] = {"images": len(cls), "split_old_lineage_recipe": dict(Counter(r["split_src"] for r in cls)), "lineages": len({r["lineage"] for r in cls}),
                   "split_images": dict(Counter(r["split"] for r in cls)),
                   "split_lineages": {s: len({r["lineage"] for r in cls if r["split"] == s})
                                      for s in ("train", "val", "test")},
                   "label_images": {IO_CLS_CLASSES[c]: sum(c in r["labels"] for r in cls) for c in range(6)},
                   "multi_label_images": sum(len(r["labels"]) > 1 for r in cls),
                   "labels_per_image_mean": float(np.mean([len(r["labels"]) for r in cls]))}
    det = [r for r in io["io_det"] if r["split"] != "dropped"]
    S["io_det"] = {"images": len(det), "boxes": sum(len(r["boxes_cxcywh"]) for r in det),
                   "split_images": dict(Counter(r["split"] for r in det)),
                   "split_boxes": {s: sum(len(r["boxes_cxcywh"]) for r in det if r["split"] == s)
                                   for s in ("train", "val", "test")},
                   "class_boxes": {IO_DET_CLASSES[c]: sum(b[0] == c for r in det for b in r["boxes_cxcywh"])
                                   for c in range(4)},
                   "lineages": len({r["lineage"] for r in det})}
    can = [r for r in px if r["role"] == "canonical"]
    seg = [r for r in can if r["in_seg"]]
    S["px_files"] = {"files": len(read_json(DATA / "inv_px.json")), "usable_files": len(px),
                     "re_exports": sum(r["role"] == "re-export" for r in px),
                     "boxes_all_files": sum(len(r["instances"]) for r in px)}
    S["px_det"] = {"images": len(can), "boxes": sum(len(r["instances"]) for r in can),
                   "split_images": dict(Counter(r["split"] for r in can)),
                   "split_boxes": {s: sum(len(r["instances"]) for r in can if r["split"] == s)
                                   for s in ("train", "val", "test")},
                   "class_boxes": {PX_CLASSES[c]: sum(x["c"] == c for r in can for x in r["instances"])
                                   for c in range(31)},
                   "class_images": {PX_CLASSES[c]: sum(any(x["c"] == c for x in r["instances"]) for r in can)
                                    for c in range(31)},
                   "classes_in_test": sum(any(x["c"] == c for r in can if r["split"] == "test"
                                              for x in r["instances"]) for c in range(31)),
                   "width_1615": sum(r["width"] == 1615 for r in can),
                   "empty_images": sum(len(r["instances"]) == 0 for r in can)}
    S["px_seg"] = {"images": len(seg), "instances": sum(len(r["instances"]) for r in seg),
                   "polygon_instances": sum(x["kind"] == "polygon" for r in seg for x in r["instances"]),
                   "split_images": dict(Counter(r["split"] for r in seg)),
                   "split_instances": {s: sum(len(r["instances"]) for r in seg if r["split"] == s)
                                       for s in ("train", "val", "test")},
                   "classes_present": sum(any(x["c"] == c for r in seg for x in r["instances"]) for c in range(31))}
    S["px_seg"]["polygon_share"] = S["px_seg"]["polygon_instances"] / S["px_seg"]["instances"]
    S["total"] = {"images": S["io_cls"]["images"] + S["io_det"]["images"] + S["px_det"]["images"],
                  "annotations": (sum(len(r["labels"]) for r in cls) + S["io_det"]["boxes"]
                                  + S["px_det"]["boxes"]),
                  "tasks": 4, "modalities": 2, "unified_concepts": 36,
                  "distinct_images": len({r["sha256"] for r in cls} | {r["sha256"] for r in det}
                                         | {r["sha256"] for r in can}),
                  "task_image_instances": S["io_cls"]["images"] + S["io_det"]["images"]
                  + S["px_det"]["images"] + S["px_seg"]["images"]}
    S["audit"] = audit
    return S


def main():
    audit = {"provenance": "run"}
    io = build_intraoral(audit)
    px = build_panoramic(audit)
    write_json(DATA / "bench.json", {"io_cls": io["io_cls"], "io_det": io["io_det"],
                                     "io_aug": io["io_aug"], "px": px})
    audit["px_copy_families_used"] = len({r["family"] for r in px})
    write_manifests(io, px)
    S = stats(io, px, audit)
    write_json(RESULTS / "dataset_stats.json", S)
    print(json.dumps({k: v for k, v in S.items() if k not in ("px_det",)}, indent=1)[:4000])


if __name__ == "__main__":
    main()
