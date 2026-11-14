# Source licences and provenance (supplement table)

Accessed 2026-11-11 through the Kaggle public metadata API (`/api/v1/datasets/view`, `/metadata`, `/list`). No images or archives were downloaded. Roboflow Universe pages returned HTTP 403 to automated fetching, so the table records no Roboflow licence. Full quotes and URLs are in `licence_inventory.json`.

| Source ID | Local subset (files) | Modality / task | Publisher (URL) | Declared licence (verbatim) | Upstream chain | Identifiers in filenames | Consent / ethics statement | Release decision |
|---|---|---|---|---|---|---|---|---|
| PANO-DXPD31 | Dental-Disease-Detection COCO+YOLO: 13,932 images declared (9,481/2,871/1,580); 8,956 unique on disk (partial train) | Panoramic radiograph; 31-class detection and polygon segmentation | Kaggle `lokisilvres/dental-disease-panoramic-detection-dataset` v6 | "Apache 2.0" (API `licenseName`); metadata `"licenses":[{"name":"apache-2.0"}]`, `"userSpecifiedSources":"Roboflow"` | Roboflow export (all filenames `*.rf.<hash>.jpg`; data.yaml `# path: ../datasets/roboflow`). Original Roboflow project(s), owner(s) and licence(s) are not identified. Candidate: Roboflow "pano" (14 classes incl. `croen`), unverified | **Yes.** 2,774 of 13,932 stems contain apparent personal-name tokens, and 2,392 of them also contain dates | None found | **build_kit_only** |
| ORAL-CALCULUS | Calculus (1,296) | Intraoral photo; classification | Kaggle `salmansajid05/oral-diseases` v3 | "Unknown" | Kaggle description: "sourced from multiple hospitals and reputable dental websites". No upstream named | No (numeric names) | None found | **unclear_build_kit_only** |
| ORAL-CARIES | Data caries (219 original + 2,382 augmented = 2,601) | Intraoral photo; classification | same | "Unknown" | as above | No | None found | **unclear_build_kit_only** |
| ORAL-GINGIVITIS | Gingivitis (2,349) | Intraoral photo; classification | same | "Unknown" | as above | No | None found | **unclear_build_kit_only** |
| ORAL-HYPODONTIA | hypodontia (1,251) | Intraoral photo; classification | same | "Unknown" | as above | No | None found | **unclear_build_kit_only** |
| ORAL-MOUTH-ULCER | Mouth Ulcer (265 original + 2,541 augmented = 2,806) | Intraoral photo; classification | same | "Unknown" | as above | No | None found | **unclear_build_kit_only** |
| ORAL-TOOTH-DISCOLORATION | Tooth Discoloration (183 original + 1,834 augmented = 2,017) | Intraoral photo; classification | same | "Unknown" | as above | No | None found | **unclear_build_kit_only** |
| ORAL-YOLO4 | Caries_Gingivitus_ToothDiscoloration_Ulcer YOLO (1,493 train + 49 val images) | Intraoral photo; 4-class detection | same | "Unknown" | as above, plus a likely description in arXiv:2403.12044 (Amrita Vishwa Vidyapeetham, 2023): "sourced from multiple hospitals and dental websites". The paper gives no licence | No | None found (also none in arXiv:2403.12044) | **unclear_build_kit_only** |

Notes.
- The local copy of `salmansajid05/oral-diseases` matches the Kaggle v3 file listing for all 15,373 files by path and byte size. Every local panoramic file that appears in the Kaggle listing (11,893 of 11,893) also matches.
- Two downstream Kaggle mirrors re-declare Apache 2.0: `ishantrajkashyap/dental-disease-detection` (panoramic) and `mohamedeidshehata6/oral-diseases-image-classification` (intraoral). They are re-uploads and do not establish rights.
- Build kit: fetch each dataset from its Kaggle page through the Kaggle API, check it against a sha256 manifest keyed by content hash (never by original filename), rename panoramic files to hash names locally, and apply our split and label files keyed by sha256. No images or bundled model weights are re-hosted.
