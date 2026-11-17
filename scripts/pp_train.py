"""One post-plan training run (review request, plan.json amendments 2026-11-17): wraps
unidental.yolo_train with (a) results written to results/postplan/ so that the registered tables and
macros never pick them up, (b) a memory log (data/runs/<run>/memlog.jsonl), (c) a status file on
failure, and (d) the per-image test matches (data/preds/pi_<run>.pkl) once the run is done.
Same arguments as unidental.yolo_train. Exit codes: 0 done, 3 budget chunk (resume), 1 failed.
"""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[1]
os.chdir(R)
sys.path.insert(0, str(R))
import unidental.yolo_train as YT  # noqa: E402

PP = R / "results" / "postplan"
PP.mkdir(exist_ok=True)
YT.RESULTS = PP
argv = sys.argv[1:]
run = argv[argv.index("--run") + 1]
rdir = YT.RUNS / run
status = PP / f"status_{run}.json"


def memlog_cb():
    import torch
    st = {"n": 0}

    def rec(trainer, where):
        rdir.mkdir(parents=True, exist_ok=True)
        row = {"t": round(time.time()), "where": where, "epoch": int(getattr(trainer, "epoch", -1)) + 1,
               "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 3)}
        if torch.backends.mps.is_available():
            row["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 2 ** 30, 3)
            row["mps_current_gb"] = round(torch.mps.current_allocated_memory() / 2 ** 30, 3)
        with open(rdir / "memlog.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def on_batch(trainer):
        st["n"] += 1
        if st["n"] % 50 == 1:
            rec(trainer, "train_batch")

    return on_batch, lambda tr: rec(tr, "epoch_end")


from ultralytics.utils.callbacks.base import default_callbacks  # noqa: E402

ob, oe = memlog_cb()
default_callbacks["on_train_batch_end"].append(ob)
default_callbacks["on_fit_epoch_end"].append(oe)
sys.argv = ["unidental.yolo_train"] + argv
try:
    YT.main()
except Exception as e:  # record honestly where it failed, then fail
    last = None
    ml = rdir / "memlog.jsonl"
    if ml.exists():
        rows = [json.loads(l) for l in ml.read_text().splitlines() if l.strip()]
        last = rows[-1] if rows else None
    status.write_text(json.dumps({"run": run, "state": "failed", "error_type": type(e).__name__,
                                  "error": str(e).split("\n")[0][:300], "last_memlog": last,
                                  "at": time.strftime("%Y-%m-%dT%H:%M%z")}, indent=1))
    print("FAILED", type(e).__name__, str(e)[:300], flush=True)
    sys.exit(1)
res = PP / f"yolo_{run}.json"
if not res.exists():
    sys.exit(3)              # training chunk ended before the test evaluation
pkl = R / "data" / "preds" / f"pi_{run}.pkl"
if not pkl.exists():
    r = json.loads(res.read_text())
    w = rdir / "weights" / ("last.pt" if r["checkpoint_selection"].startswith("last") else "best.pt")
    subprocess.check_call([sys.executable, "-u", "scripts/perimage_eval.py", "--weights", str(w),
                           "--data", str(R / "data" / "yolo" / r["data"] / "data.yaml"), "--split", "test",
                           "--out", str(pkl), "--device", "mps", "--batch", str(r["batch"])])
status.write_text(json.dumps({"run": run, "state": "done", "at": time.strftime("%Y-%m-%dT%H:%M%z")}))
sys.exit(0)
