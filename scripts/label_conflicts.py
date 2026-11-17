"""Label conflicts among exact copies in the IO-DET source (a lower bound on label noise).
Reads the local inventory (data/inv_io_det.json) and writes results/label_conflicts.json.
  python3 scripts/label_conflicts.py
"""
import json
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[1]
inv = json.loads((R / "data" / "inv_io_det.json").read_text())
by = defaultdict(list)
for r in inv:
    by[r["sha256"]].append(r)
groups = [v for v in by.values() if len(v) > 1]
key = lambda r: sorted((b[0], *(round(x, 4) for x in b[1:])) for b in r["boxes_cxcywh"])
conflict = [g for g in groups if len({json.dumps(key(r)) for r in g}) > 1]
out = {"provenance": "run (scripts/label_conflicts.py over data/inv_io_det.json)",
       "io_det_files": len(inv), "io_det_exact_duplicate_groups": len(groups),
       "io_det_exact_duplicate_files_removed": sum(len(g) - 1 for g in groups),
       "io_det_duplicate_groups_with_conflicting_boxes": len(conflict),
       "conflicting_share_of_duplicate_groups": len(conflict) / len(groups) if groups else None}
(R / "results" / "label_conflicts.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
