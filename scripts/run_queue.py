#!/usr/bin/env python3
"""Run the next unfinished jobs of the frozen plan inside one guarded chunk.

  run_heavy.sh du-queue --timeout 600 --mem-gb 5 -- python3 scripts/queue.py 560
Each job is idempotent and resumable; a job is done when its result file exists.
"""
import os, subprocess, sys, time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
os.chdir(R)
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 560
T0 = time.time()
PY = sys.executable
res = lambda n: (R / "results" / n).exists()
Y = lambda run, data, model, ep, seed, shards=1, extra=(): (
    f"yolo_{run}.json",
    [PY, "-u", "-m", "dentunity.yolo_train", "--run", run, "--data", data, "--model", model,
     "--imgsz", "640", "--epochs", str(ep), "--seed", str(seed), "--shards", str(shards), *extra])
JOBS = [
    # order: amendment 2026-11-14 21:10 (RT-DETR-l dropped; cheap verdict inputs moved earlier)
    ("prep", "px640_done.flag", [PY, "-u", "-c",
        "import subprocess,sys,pathlib;subprocess.check_call([sys.executable,'-m','dentunity.resize']);"
        "import shutil;shutil.rmtree('data/yolo',ignore_errors=True);"
        "subprocess.check_call([sys.executable,'-m','dentunity.yolo_export']);"
        "pathlib.Path('results/px640_done.flag').write_text('ok')"]),
    ("pxseg_y11s_s0",) + Y("pxseg_y11s_s0", "pxseg", "yolo11s-seg.pt", 20, 0, 2),
    ("iodet_y11s_s0",) + Y("iodet_y11s_s0", "iodet", "yolo11s.pt", 40, 0),
    ("cls_frozen", "cls_frozen.json.done", [PY, "-u", "-m", "dentunity.cls_frozen"]),
    ("cls_ft_convnext_t_s0", "cls_ft_convnext_t_s0.json", [PY, "-u", "-m", "dentunity.cls_finetune", "--model", "convnext_t", "--seed", "0"]),
    ("cls_ft_vits14_s0", "cls_ft_vits14_s0.json", [PY, "-u", "-m", "dentunity.cls_finetune", "--model", "vits14", "--seed", "0"]),
    ("pxdet_y11s_s0",) + Y("pxdet_y11s_s0", "pxdet", "yolo11s.pt", 20, 0, 2),
    ("leak_leaky_s0",) + Y("pxleak_y11n_leaky_s0", "pxdet_leaky", "yolo11n.pt", 15, 0),
    ("leak_clean_s0",) + Y("pxleak_y11n_cleanctl_s0", "pxdet_cleanctl", "yolo11n.pt", 15, 0),
    ("dentex_prep", "dentex_overlap.json", [PY, "-u", "-c",
        "import pathlib,sys,subprocess;n=len(list(pathlib.Path('data/dentex/raw/train').glob('*.png')));"
        "sys.exit(9) if n<705 else subprocess.check_call([sys.executable,'-m','dentunity.dentex'])"]),
    ("dentex_coco_s0",) + Y("dentex_coco_y11s_s0", "dentex", "yolo11s.pt", 40, 0, 1, ("--final", "last")),
    ("dentex_du_s0",) + Y("dentex_du_y11s_s0", "dentex", "runs/pxdet_y11s_s0/weights/best.pt", 40, 0, 1, ("--final", "last")),
    ("pxseg_y11m_s0",) + Y("pxseg_y11m_s0", "pxseg", "yolo11m-seg.pt", 20, 0, 4, ("--batch", "4")),
    ("cls_frozen_hash", "cls_frozen_hash.json.done", [PY, "-u", "-m", "dentunity.cls_frozen", "--split", "split_hash", "--tag", "_hash"]),
    ("cls_ft_convnext_t_s1", "cls_ft_convnext_t_s1.json", [PY, "-u", "-m", "dentunity.cls_finetune", "--model", "convnext_t", "--seed", "1"]),
    ("cls_ft_vits14_s1", "cls_ft_vits14_s1.json", [PY, "-u", "-m", "dentunity.cls_finetune", "--model", "vits14", "--seed", "1"]),
    ("cls_ft_convnext_t_s2", "cls_ft_convnext_t_s2.json", [PY, "-u", "-m", "dentunity.cls_finetune", "--model", "convnext_t", "--seed", "2"]),
    ("cls_ft_vits14_s2", "cls_ft_vits14_s2.json", [PY, "-u", "-m", "dentunity.cls_finetune", "--model", "vits14", "--seed", "2"]),
    ("iodet_shipped",) + Y("iodet_shipped_y11s_s0", "iodet_shipped", "yolo11s.pt", 40, 0),
    ("pxseg_y11s_s1",) + Y("pxseg_y11s_s1", "pxseg", "yolo11s-seg.pt", 20, 1, 2, ("--batch", "4")),
    ("pxdet_y11s_s1",) + Y("pxdet_y11s_s1", "pxdet", "yolo11s.pt", 20, 1, 2),
    ("pxdet_y11m_s0",) + Y("pxdet_y11m_s0", "pxdet", "yolo11m.pt", 20, 0, 4, ("--batch", "4")),
    ("iodet_y11s_s1",) + Y("iodet_y11s_s1", "iodet", "yolo11s.pt", 40, 1),
    ("iodet_y11n_s0",) + Y("iodet_y11n_s0", "iodet", "yolo11n.pt", 40, 0),
    ("leak_leaky_s1",) + Y("pxleak_y11n_leaky_s1", "pxdet_leaky", "yolo11n.pt", 15, 1),
    ("leak_clean_s1",) + Y("pxleak_y11n_cleanctl_s1", "pxdet_cleanctl", "yolo11n.pt", 15, 1),
    # amendment 2026-11-15 14:50: YOLO11n is the validation-selected IO-DET family -> second seed
    ("iodet_y11n_s1",) + Y("iodet_y11n_s1", "iodet", "yolo11n.pt", 40, 1),
    ("dentex_coco_s1",) + Y("dentex_coco_y11s_s1", "dentex", "yolo11s.pt", 40, 1, 1, ("--final", "last")),
    ("dentex_du_s1",) + Y("dentex_du_y11s_s1", "dentex", "runs/pxdet_y11s_s1/weights/best.pt", 40, 1, 1, ("--final", "last")),
    ("pxseg_y11m_s1",) + Y("pxseg_y11m_s1", "pxseg", "yolo11m-seg.pt", 20, 1, 4, ("--batch", "4")),
    ("pxdet_y11m_s1",) + Y("pxdet_y11m_s1", "pxdet", "yolo11m.pt", 20, 1, 4, ("--batch", "4")),
    ("pxdet_y11n_s0",) + Y("pxdet_y11n_s0", "pxdet", "yolo11n.pt", 20, 0),
    ("pxseg_y11n_s0",) + Y("pxseg_y11n_s0", "pxseg", "yolo11n-seg.pt", 20, 0, 1, ("--batch", "4")),
    ("pxdet_reexp_s0",) + Y("pxdet_reexp_y11s_s0", "pxdet_reexp", "yolo11s.pt", 9, 0, 4),
]


def done(flag):
    if flag.endswith(".done"):
        return res(flag[:-5]) and (R / "results" / flag).exists()
    return res(flag)


for name, flag, cmd in JOBS:
    if done(flag):
        continue
    if name.startswith("dentex_") and name != "dentex_prep" and not res("dentex_overlap.json"):
        continue
    if name == "dentex_du_s1" and not (R / "data/runs/pxdet_y11s_s1/weights/best.pt").exists():
        continue
    if name in ("pxseg_y11m_s1", "pxdet_y11m_s1"):
        import json as _j
        task = "pxseg" if "seg" in name else "pxdet"
        kind = "mask" if task == "pxseg" else "box"
        fm, fs = R / f"results/yolo_{task}_y11m_s0.json", R / f"results/yolo_{task}_y11s_s0.json"
        if not fm.exists():
            continue
        vm = _j.loads(fm.read_text())["val"][kind]["map50_95"]
        vs = _j.loads(fs.read_text())["val"][kind]["map50_95"]
        if vm <= vs:
            print(f"[queue] {name}: skipped, YOLO11m does not beat YOLO11s on validation ({vm:.4f} <= {vs:.4f})", flush=True)
            continue
    left = BUDGET - (time.time() - T0)
    if left < 150:
        break
    if name.startswith(("pxdet", "pxseg", "iodet", "leak", "rtdetr", "dentex_coco", "dentex_du")):
        cmd = cmd + ["--budget", str(int(left))]
    elif name.startswith("cls_ft"):
        cmd = cmd + ["--budget", str(int(left))]
    print(f"[queue] {name} (left {left:.0f}s)", flush=True)
    rc = subprocess.call(cmd)
    if flag.endswith(".done") and rc == 0:
        (R / "results" / flag).write_text("ok")
    if rc == 9:          # prerequisite not ready (e.g. download running): skip for now
        print(f"[queue] {name} waiting for its inputs", flush=True)
        continue
    if rc not in (0, 3):
        print(f"[queue] {name} rc={rc}", flush=True)
        break
    if not done(flag):
        break   # job still in progress (budget) -> next chunk resumes it
if all(done(f) for _, f, _ in JOBS):
    print("[queue] ALLDONE", flush=True)
print("[queue] chunk end", round(time.time() - T0), "s", flush=True)
