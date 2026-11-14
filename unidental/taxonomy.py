"""One label taxonomy for every task.

Each unified concept has a stable id, a canonical name, a clinical family and the
source label strings it absorbs. Task-local YOLO ids are kept for training; the
release manifests store unified ids.
"""
from __future__ import annotations

FAMILIES = ["pathology", "periodontal / soft tissue", "tooth status / development",
            "anatomy", "restoration / device"]

# (unified name, family, {task: source label string})
_CONCEPTS = [
    ("caries", "pathology", {"io_cls": "caries", "io_det": "Caries", "px": "Caries"}),
    ("calculus", "periodontal / soft tissue", {"io_cls": "calculus"}),
    ("gingivitis", "periodontal / soft tissue", {"io_cls": "gingivitis", "io_det": "Gingivitis"}),
    ("mouth ulcer", "periodontal / soft tissue", {"io_cls": "mouth ulcer", "io_det": "Ulcer"}),
    ("tooth discoloration", "pathology", {"io_cls": "tooth discoloration",
                                          "io_det": "Tooth Discoloration"}),
    ("hypodontia", "tooth status / development", {"io_cls": "hypodontia"}),
    ("periapical lesion", "pathology", {"px": "Periapical lesion"}),
    ("bone loss", "periodontal / soft tissue", {"px": "Bone Loss"}),
    ("bone defect", "pathology", {"px": "bone defect"}),
    ("cyst", "pathology", {"px": "Cyst"}),
    ("tooth fracture", "pathology", {"px": "Fracture teeth"}),
    ("attrition", "pathology", {"px": "attrition"}),
    ("root resorption", "pathology", {"px": "Root resorption"}),
    ("missing tooth", "tooth status / development", {"px": "Missing teeth"}),
    ("impacted tooth", "tooth status / development", {"px": "impacted tooth"}),
    ("malaligned tooth", "tooth status / development", {"px": "Malaligned"}),
    ("supra-eruption", "tooth status / development", {"px": "Supra Eruption"}),
    ("retained root", "tooth status / development", {"px": "Retained root"}),
    ("root piece", "tooth status / development", {"px": "Root Piece"}),
    ("permanent tooth", "tooth status / development", {"px": "Permanent Teeth"}),
    ("primary tooth", "tooth status / development", {"px": "Primary teeth"}),
    ("mandibular canal", "anatomy", {"px": "Mandibular Canal"}),
    ("maxillary sinus", "anatomy", {"px": "maxillary sinus"}),
    ("crown", "restoration / device", {"px": "Crown"}),
    ("filling", "restoration / device", {"px": "Filling"}),
    ("root canal treatment", "restoration / device", {"px": "Root Canal Treatment"}),
    ("post-core", "restoration / device", {"px": "post - core"}),
    ("implant", "restoration / device", {"px": "Implant"}),
    ("abutment", "restoration / device", {"px": "abutment"}),
    ("gingival former", "restoration / device", {"px": "gingival former"}),
    ("orthodontic bracket", "restoration / device", {"px": "orthodontic brackets"}),
    ("orthodontic wire", "restoration / device", {"px": "wire"}),
    ("metal band", "restoration / device", {"px": "metal band"}),
    ("permanent retainer", "restoration / device", {"px": "permanent retainer"}),
    ("temporary anchorage device", "restoration / device", {"px": "TAD"}),
    ("bone plating", "restoration / device", {"px": "plating"}),
]

UNIFIED = [{"id": i, "name": n, "family": f, "sources": s} for i, (n, f, s) in enumerate(_CONCEPTS)]

# task-local label orders exactly as shipped upstream
IO_CLS_CLASSES = ["calculus", "caries", "gingivitis", "hypodontia", "tooth discoloration",
                  "mouth ulcer"]
# source folders of the intraoral classification photographs (augmented folders excluded)
IO_CLS_FOLDERS = {"calculus": "Calculus/Calculus",
                  "caries": "Data caries/Data caries/caries orignal data set",
                  "gingivitis": "Gingivitis/Gingivitis",
                  "hypodontia": "hypodontia/hypodontia",
                  "tooth discoloration": "Tooth Discoloration/Tooth Discoloration/tooth discoloration original dataset",
                  "mouth ulcer": "Mouth Ulcer/Mouth Ulcer/ulcer original dataset"}
IO_DET_CLASSES = ["Caries", "Ulcer", "Tooth Discoloration", "Gingivitis"]
PX_CLASSES = ["Caries", "Crown", "Filling", "Implant", "Malaligned", "Mandibular Canal",
              "Missing teeth", "Periapical lesion", "Retained root", "Root Canal Treatment",
              "Root Piece", "impacted tooth", "maxillary sinus", "Bone Loss", "Fracture teeth",
              "Permanent Teeth", "Supra Eruption", "TAD", "abutment", "attrition", "bone defect",
              "gingival former", "metal band", "orthodontic brackets", "permanent retainer",
              "post - core", "plating", "wire", "Cyst", "Root resorption", "Primary teeth"]


def to_unified(task: str) -> list[int]:
    """Map a task-local class index to its unified id (task in io_cls, io_det, px)."""
    names = {"io_cls": IO_CLS_CLASSES, "io_det": IO_DET_CLASSES, "px": PX_CLASSES}[task]
    lut = {c["sources"][task]: c["id"] for c in UNIFIED if task in c["sources"]}
    return [lut[n] for n in names]


def check() -> dict:
    out = {}
    for t in ("io_cls", "io_det", "px"):
        m = to_unified(t)
        assert len(set(m)) == len(m), t
        out[t] = m
    shared = [c["name"] for c in UNIFIED if len(c["sources"]) > 1]
    return {"n_unified": len(UNIFIED), "maps": out, "shared_concepts": shared,
            "n_families": len(FAMILIES)}


if __name__ == "__main__":
    import json
    print(json.dumps(check(), indent=1))
