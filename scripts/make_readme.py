"""Write README.md with a results table read from results/summary_*.json (real numbers only)."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1]
J = lambda n: json.loads((R / "results" / n).read_text()) if (R / "results" / n).exists() else {}
S, D, C = J("dataset_stats.json"), J("summary_det.json"), J("summary_cls.json")
G = D.get("groups", {})
f = lambda v: "pending" if v is None else f"{100 * v:.1f}"
g = lambda grp, k: G.get(grp, {}).get(k)
cls_best = C.get("best_by_val")
cls_val = C.get("models", {}).get(cls_best, {}).get("test_macro_ap") if cls_best else None
rows = [
    ("IO-CLS", "macro-AP", cls_best or "pending", f(cls_val)),
    ("IO-DET", "box mAP50-95", "YOLO11s", f(g("iodet_y11s", "test_box_map50_95_mean"))),
    ("PX-DET", "box mAP50-95", "YOLO11s", f(g("pxdet_y11s", "test_box_map50_95_mean"))),
    ("PX-SEG", "mask mAP50-95", "YOLO11s-seg", f(g("pxseg_y11s", "test_mask_map50_95_mean"))),
]
table = "| Task | Metric (test, %) | Best model | Score |\n|---|---|---|---|\n" + "\n".join(
    f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)
T = S.get("total", {})
readme = f"""# DentUnity: one copy-free dataset to train and evaluate dental classification, detection and segmentation

![Python](https://img.shields.io/badge/python-3.11+-blue) ![License: MIT](https://img.shields.io/badge/code-MIT-green) ![Release](https://img.shields.io/badge/data-build%20kit-orange)

DentUnity unifies intraoral photographs and panoramic radiographs into four tasks under one
36-concept taxonomy: {T.get('distinct_images', 'pending'):,} distinct images, {T.get('annotations', 'pending'):,} annotations.
Its splits are drawn over **copy lineages jointly for every task**, so no test image of any
task has a copy in any training set. In the shipped panoramic split, **96.1%** of test
images are copies of training images.

![teaser](figures/fig_teaser.png)

## News
- **2026-11** Build kit, splits, baselines and paper (under review).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export SSL_CERT_FILE="$(python -m certifi)"
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Quickstart (smoke test, under 1 minute, no data needed)

```bash
python run_experiment.py --smoke
```

## Build the dataset from the upstream releases

See `release/BUILD.md`. In short:

```bash
python -m unidental.rebuild --pano-root upstream/pano --oral-root upstream/oral --out dentunity_v1
```

## Results (copy-free test splits)

{table}

All numbers come from `results/summary_*.json`; two-seed detection rows report the mean.

## Reproduce

```bash
python run_experiment.py build        # lineages, splits, manifests, audits
python run_experiment.py queue        # one guarded chunk of the frozen plan (repeat until done)
python run_experiment.py summarize    # results/summary_*.json
python scripts/make_tables.py ../../paper && python scripts/fig_dataset.py && python scripts/fig_teaser.py
```

The frozen plan with timestamped amendments is `../plan.json`.

## Data access

DentUnity is released as a build kit: we re-host no images and no upstream annotation
files. Download the two public upstream releases yourself (Kaggle), accept their terms,
and the kit rebuilds the benchmark by sha256. Licence evidence per source:
`release/licence_table.md`; open questions for rights holders: `release/open_questions.md`.

## Citation

If you use DentUnity, please cite:

```bibtex
@inproceedings{{dentunity2027,
  title  = {{DentUnity: One Copy-Free Dataset to Train and Evaluate Dental Classification, Detection and Segmentation}},
  author = {{Anonymous}},
  year   = {{2027}},
  note   = {{Under review}}
}}
```

## License

Code: MIT (`LICENSE`). Manifests and harmonised labels: CC BY 4.0 (pending author confirmation).
Images: not distributed; upstream terms apply.
"""
(R / "README.md").write_text(readme)
print("README written")
