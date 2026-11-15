# DentUnity: one copy-free dataset to train and evaluate dental classification, detection and segmentation

> This folder is the DentUnity build kit. It is hosted next to the code during the anonymous
> review period and moves to a dataset hub on publication.

> **Status: under review.** This repository accompanies an anonymous CVPR 2027 submission.
> During review the build kit is public under the terms below.

## Summary
DentUnity unifies two widely shared community releases of dental images into four tasks on
two modalities:
- multi-label classification of intraoral photographs (**IO-CLS**);
- intraoral lesion detection (**IO-DET**);
- 31-class panoramic detection (**PX-DET**);
- panoramic instance segmentation (**PX-SEG**).

The dataset has these properties:
- **Size:** 10,456 distinct images with 64,226 annotations, mapped to one taxonomy of 36 clinical concepts in five families.
- **Copy-free splits:** the source releases store many copies of the same image (re-exports, resized versions, augmentations). DentUnity links all copies into copy lineages and assigns whole lineages to train / validation / test **jointly for every task**. As a result, no validation or test image of any task has a copy in the training data of any task.
- **How bad the shipped splits are:** in the official panoramic split, 96.1% of the test images have a copy in training.

**This repository is a build kit. It contains no images and no upstream annotation files.**
You download the upstream releases yourself and run one script. The script identifies every
file by the sha256 of its bytes, never by its original file name, and writes the dataset in
standard formats.

## Contents

| Path | What it is |
|---|---|
| `manifests/io_cls.csv` | One row per IO-CLS image: `sha256, split, lineage, width, height` |
| `manifests/io_det.csv` | One row per IO-DET image: `sha256, split, lineage, width, height, label_sha256` |
| `manifests/px.csv` | One row per usable panoramic file: `sha256, split, lineage, role (canonical / re-export), in_seg, width, height, label_sha256` |
| `manifests/expected_digests.json` | Expected counts and content digests of the rebuilt annotations, used to verify a rebuild |
| `taxonomy.csv` | The 36 unified concepts, their clinical family and the upstream label each one absorbs per task |
| `build/build_dentunity.py` | The build script (Python 3.9+, standard library only) |
| `SOURCES.md` | Where to download each upstream release, and its declared licence |

The manifests define:
- the **splits**: train / val / test per task;
- the **copy lineages**: anonymous lineage ids, shared across tasks;
- for panoramic files, the **canonical file** of each copy family and its **re-exports**.

Labels are not stored here. The script re-derives them from your download:
- IO-CLS labels come from folder membership.
- IO-DET and panoramic labels come from the upstream label files, matched by content hash.

## How to build the data
1. Download both upstream releases from Kaggle (see `SOURCES.md`) and accept their terms:
   ```bash
   pip install kaggle
   kaggle datasets download -d lokisilvres/dental-disease-panoramic-detection-dataset -p upstream/pano --unzip
   kaggle datasets download -d salmansajid05/oral-diseases -p upstream/oral --unzip
   ```
2. Rebuild the dataset:
   ```bash
   python build/build_dentunity.py --pano-root upstream/pano --oral-root upstream/oral --out dentunity_v1
   ```
   Add `--with-reexports` to put the re-exported copies of *training* lineages into the PX-DET training set; validation and test are never touched. Add `--copy` to copy images instead of symlinking them.
3. Check `dentunity_v1/verify.json`. Every entry of `matches_expected` must be `true`. If some files are missing upstream, the script reports them in `missing`.

The output contains:
- `io_cls/` plus `io_cls.csv` (six binary labels and unified ids);
- `io_det/`, `px_det/` and `px_seg/` as Ultralytics-style YOLO trees, each with a `data.yaml`;
- `taxonomy.json`.

Output files are named by content hash.

## Splits

| Task | Images | Annotations | Train / val / test |
|---|---|---|---|
| IO-CLS | 3,899 | 4,307 image labels | 2,697 / 394 / 808 |
| IO-DET | 1,373 | 3,129 boxes | 963 / 137 / 273 |
| PX-DET | 5,929 | 56,790 boxes | 4,150 / 593 / 1,186 |
| PX-SEG | 5,297 | 50,754 instances | 3,710 / 532 / 1,055 |

Whole lineages were placed by greedy iterative stratification (70/10/20). IO-CLS and IO-DET
share one lineage graph and one split; PX-DET and PX-SEG share radiographs. The 7,997
panoramic re-exports inherit the split of their lineage.

Evaluation rule:
- Evaluate on the canonical files of the val/test splits only.
- Re-exports may be used for training only.

## Licence and terms
- **Build-kit files:** this repository (manifests, lineage and split lists, taxonomy, build script) is released under CC BY-NC 4.0.
- **Not included:** images and upstream annotations. They remain under their publishers' terms, which you accept when you download them.
- **Upstream licences** (details and verbatim quotes in `SOURCES.md`):
  - The panoramic re-publisher declares "Apache 2.0" and names Roboflow as its source, without naming the upstream projects.
  - The intraoral re-publisher declares the licence as "Unknown".
  - Neither documents consent, ethics approval or de-identification.
- **Why images are not re-hosted:** about one in five panoramic source file names contains an apparent personal name. For this reason DentUnity never uses or redistributes original file names or images.
- **Use:** research only; not for clinical use. The data come from unknown populations and devices.

## Citation
Anonymous CVPR 2027 submission. The citation will be added on publication.

## Contact
Contact details will be added after the anonymous review period.
