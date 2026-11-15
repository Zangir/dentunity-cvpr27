# Upstream sources: download and terms

DentUnity is built from two public community releases on Kaggle. The publisher metadata was
accessed on 2026-11-11 through the public Kaggle metadata API, and no image was downloaded
for that check. You need a Kaggle account and must accept each dataset's terms before
downloading.

## 1. Panoramic radiographs: PX-DET, PX-SEG
- **Page:** https://www.kaggle.com/datasets/lokisilvres/dental-disease-panoramic-detection-dataset (version 6)
- **Download:** `kaggle datasets download -d lokisilvres/dental-disease-panoramic-detection-dataset -p upstream/pano --unzip`
- **Content used:** 13,932 radiograph files exported by an annotation tool, with polygon labels for 31 classes. DentUnity uses the YOLO label files. 6 files with out-of-bounds coordinates are excluded, which leaves 13,926.
- **Declared licence:**
  - The re-publisher declares "Apache 2.0" (Kaggle `licenseName`). The metadata names "Roboflow" as the source.
  - The original Roboflow projects, owners and licences are not identified, so the declaration is unverified upstream.
- **Identifiers:** 2,774 of the 13,932 file names contain apparent personal-name tokens. DentUnity never stores, prints or redistributes these names. Every file is matched by the sha256 of its bytes.
- **Consent / ethics:** no statement found.
- **DentUnity decision:** build kit only. No image, file name or upstream annotation is redistributed.

## 2. Intraoral photographs: IO-CLS, IO-DET
- **Page:** https://www.kaggle.com/datasets/salmansajid05/oral-diseases (version 3)
- **Download:** `kaggle datasets download -d salmansajid05/oral-diseases -p upstream/oral --unzip`
- **Content used:**
  - IO-CLS: the six original condition folders (5,563 files, 3,899 distinct images). The augmented folders are not used. The folders are:
    - `Calculus/Calculus`
    - `Data caries/Data caries/caries orignal data set`
    - `Gingivitis/Gingivitis`
    - `hypodontia/hypodontia`
    - `Tooth Discoloration/Tooth Discoloration/tooth discoloration original dataset`
    - `Mouth Ulcer/Mouth Ulcer/ulcer original dataset`
  - IO-DET: the 4-class YOLO detection subset (1,508 labelled files, 1,373 distinct images).
- **Declared licence:**
  - "Unknown" (Kaggle `licenseName`).
  - The description says the images were sourced from multiple hospitals and dental websites, without naming them.
  - The detection subset matches the dataset described in arXiv:2403.12044, which states no licence.
- **Consent / ethics:** no statement found.
- **DentUnity decision:** build kit only.

## What DentUnity adds, and why it can be shared
The files in this repository contain:
- content hashes and image sizes of upstream files;
- our own split and lineage assignments, with anonymous lineage ids;
- the canonical / re-export roles;
- a mapping of upstream label names to 36 concepts;
- a script.

They contain:
- no pixels;
- no original file names;
- no upstream annotation content. Label files are referenced only by the sha256 of their bytes, and labels are re-derived on your machine.
