"""Post-plan analysis B(iii): fixed 5-fold split of the 705 DENTEX training images -> YOLO trees
data/yolo/dentex_cv<k>/ (train = 4 folds, val = test = held-out fold; images are symlinks).
Folds: seeded permutation of the sorted file list, fold = rank mod 5. CPU only.
  python3 scripts/make_dentex_cv.py   -> results/postplan/dentex_cv_folds.json
"""
import hashlib
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(R))
from unidental.common import stable_seed  # noqa: E402

SRC = R / "data" / "yolo" / "dentex"
NAMES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]
stems = sorted(p.stem for p in (SRC / "images" / "train").glob("*.jpg"))
assert len(stems) == 705, len(stems)
perm = np.random.default_rng(stable_seed("dentex-cv5")).permutation(len(stems))
fold = {stems[j]: r % 5 for r, j in enumerate(perm)}
info = {"provenance": "run (scripts/make_dentex_cv.py; post-plan analysis B(iii))", "n_images": len(stems),
        "assignment_sha256": hashlib.sha256(json.dumps(sorted(fold.items())).encode()).hexdigest(), "folds": {}}
for k in range(5):
    Y = R / "data" / "yolo" / f"dentex_cv{k}"
    for sp in ("train", "val", "test"):
        (Y / "images" / sp).mkdir(parents=True, exist_ok=True)
        (Y / "labels" / sp).mkdir(parents=True, exist_ok=True)
    nb = 0
    for s, f in fold.items():
        for sp in (("val", "test") if f == k else ("train",)):
            dst = Y / "images" / sp / f"{s}.jpg"
            if not dst.exists():
                dst.symlink_to((SRC / "images" / "train" / f"{s}.jpg").resolve())
            lab = (SRC / "labels" / "train" / f"{s}.txt").read_text()
            (Y / "labels" / sp / f"{s}.txt").write_text(lab)
            if sp == "test":
                nb += sum(1 for l in lab.splitlines() if l.strip())
    (Y / "data.yaml").write_text(f"path: {Y}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                                 + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    info["folds"][k] = {"held_out_images": sum(f == k for f in fold.values()), "held_out_boxes": nb,
                        "train_images": sum(f != k for f in fold.values())}
(R / "results" / "postplan").mkdir(exist_ok=True)
(R / "results" / "postplan" / "dentex_cv_folds.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info["folds"]))
