# Building DentUnity from the upstream releases

DentUnity is released as a build kit. We do not re-host any image or any upstream
annotation file. You download the two upstream community releases yourself, accept
their terms, and the kit rebuilds the benchmark by matching file contents (sha256)
against our manifests.

## 1. Download the upstream releases (Kaggle account required)

```bash
pip install kaggle
kaggle datasets download -d lokisilvres/dental-disease-panoramic-detection-dataset -p upstream/pano --unzip
kaggle datasets download -d salmansajid05/oral-diseases -p upstream/oral --unzip
```

Read and follow each page's terms. The panoramic re-publisher declares Apache 2.0 and
names Roboflow as its source; the intraoral re-publisher declares the licence as
"Unknown" (see `licence_table.md`). Treat both as research-only until the open questions
in `open_questions.md` are resolved.

## 2. Rebuild and verify

```bash
python -m dentunity.rebuild --pano-root upstream/pano --oral-root upstream/oral --out dentunity_v1
cat dentunity_v1/verify.json        # found / missing counts per task
```

Files are linked under content-hash names; original file names (some of which embed
personal names) are never used. Labels are re-derived from the upstream label files and
mapped to task-local ids; unified concept ids are in `manifests/*.csv`.

## 3. What the manifests contain

| file | rows | columns |
|---|---|---|
| `manifests/io_cls.csv` | one per IO-CLS image | sha256, split, lineage id, source, size, unified labels |
| `manifests/io_det.csv` | one per IO-DET image | sha256, split, lineage id, source, size, number of boxes, unified classes |
| `manifests/px.csv` | one per usable panoramic file | sha256, split, lineage id, role (canonical / re-export), in_seg, size, annotation sha256, unified classes |

Rule: evaluate on canonical files of the val/test splits only. Re-exports may be used
for training only when their lineage is in the training split (`--with-reexports`).
