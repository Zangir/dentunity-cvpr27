# DentUnity: one copy-free dataset to train and evaluate dental classification, detection and segmentation

![Python](https://img.shields.io/badge/python-3.11+-blue) ![License: MIT](https://img.shields.io/badge/code-MIT-green) ![Release](https://img.shields.io/badge/data-build%20kit-orange)

DentUnity unifies intraoral photographs and panoramic radiographs into four tasks under one
36-concept taxonomy: 10,456 distinct images, 64,226 annotations.
Its splits are drawn over **copy lineages jointly for every task**, so no test image of any
task has a copy in any training set. In the shipped panoramic split, **96.1%** of test
images are copies of training images.

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
python -m dentunity.rebuild --pano-root upstream/pano --oral-root upstream/oral --out dentunity_v1
```

## Results (copy-free test splits)

| Task | Metric (test, %) | Best model | Score |
|:-|:-|:-|:-|
| IO-CLS | macro-AP | C2_ensemble | 83.3 |
| IO-DET | box mAP50-95 | YOLO11s | 14.9 |
| PX-DET | box mAP50-95 | YOLO11s | 13.5 |
| PX-SEG | mask mAP50-95 | YOLO11s-seg | 8.7 |

All numbers come from `results/summary_*.json`; two-seed detection rows report the mean.

## Reproduce

```bash
python run_experiment.py build        # lineages, splits, manifests, audits
python run_experiment.py queue        # one guarded chunk of the frozen plan (repeat until done)
python run_experiment.py summarize    # results/summary_*.json
python scripts/make_tables.py ../../paper && python scripts/fig_dataset.py && python scripts/fig_teaser.py
```

The frozen plan with timestamped amendments is `plan.json`.

## Data access

DentUnity is released as a build kit: we re-host no images and no upstream annotation
files. Download the two public upstream releases yourself (Kaggle), accept their terms,
and the kit rebuilds the benchmark by sha256. The build kit
(manifests, labels, splits) is in [`dataset/`](dataset/); trained model weights are in `weights/`. Licence evidence per source:
`release/licence_table.md`.

## Citation

If you use DentUnity, please cite:

```bibtex
@inproceedings{dentunity2027,
  title  = {DentUnity: One Copy-Free Dataset to Train and Evaluate Dental Classification, Detection and Segmentation},
  author = {Anonymous},
  year   = {2027},
  note   = {Under review}
}
```

## License

Code: MIT (`LICENSE`). Build kit (manifests, harmonised labels, splits, lineage ids): CC BY-NC 4.0. Released model weights: AGPL-3.0.
Images: not distributed; upstream terms apply.
