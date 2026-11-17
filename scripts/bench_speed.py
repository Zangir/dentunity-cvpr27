"""Measure YOLO training throughput on this machine's GPU (img/s) to size the plan."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentunity.common import DATA, RESULTS, write_json
from ultralytics import YOLO, RTDETR
out = RESULTS / "speed_bench.json"
res = json.loads(out.read_text()) if out.exists() else {}
cfgs = [("yolo11n.pt", "pxdet", 640, 8), ("yolo11s.pt", "pxdet", 640, 8), ("yolo11m.pt", "pxdet", 640, 8),
        ("yolo11s.pt", "pxdet", 1024, 4), ("yolo11s-seg.pt", "pxseg", 640, 8), ("yolo11m-seg.pt", "pxseg", 640, 8),
        ("rtdetr-l.pt", "pxdet", 640, 4)]
for w, data, sz, b in cfgs:
    key = f"{w}@{sz}"
    if key in res:
        continue
    M = RTDETR if "rtdetr" in w else YOLO
    m = M(str(DATA / "weights" / w))
    t = time.time()
    try:
        m.train(data=str(DATA / "yolo" / data / "data.yaml"), epochs=1, imgsz=sz, batch=b, device="mps",
                workers=2, fraction=0.06, val=False, plots=False, verbose=False, amp=False,
                project=str(DATA / "bench"), name=key, exist_ok=True)
        n = int(0.06 * 4150) if data == "pxdet" else int(0.06 * 3710)
        res[key] = {"img_per_s": n / (time.time() - t), "n": n, "s": time.time() - t}
    except Exception as e:
        res[key] = {"error": str(e)[:300]}
    write_json(out, res)
    print(key, res[key], flush=True)
