#!/bin/bash
# Post-plan training queue (review request; plan.json amendments 2026-11-17 13:05).
# One training run per run_heavy.sh call (labels du-*), resumable, stoppable:
#   touch .cache/cvpr27/STOP_du  -> run_heavy kills the running job / refuses new ones (exit 75); the
#   queue then exits cleanly. Finished runs (results/postplan/yolo_<run>.json + data/preds/pi_<run>.pkl)
#   are skipped on restart; unfinished runs resume from last.pt (Ultralytics writes it every epoch).
#   bash scripts/postplan_queue.sh [all|dentex|cv|rtdetr]
G="${HEAVY_DIR:-$(cd "$(dirname "$0")/.." && pwd)/.heavy}"   # directory holding run_heavy.sh and the STOP switch
R="$(cd "$(dirname "$0")/.." && pwd)"
cd $R || exit 1
LOG=$R/logs_local/postplan_queue.log; mkdir -p logs_local results/postplan
WHAT=${1:-all}
say(){ echo "$(date '+%F %T') $*" | tee -a $LOG; }
stopped(){ [ -e "$G/STOP" ] || [ -e "$G/STOP_du" ]; }
done_run(){ [ -s results/postplan/yolo_$1.json ] && [ -s data/preds/pi_$1.pkl ]; }
failed_run(){ grep -q '"failed"' results/postplan/status_$1.json 2>/dev/null; }
# run <label> <run> <timeout_s> <max_kills> <args...>
run(){
  local label=$1 rn=$2 to=$3 maxk=$4; shift 4
  local kills=0
  while ! done_run $rn; do
    failed_run $rn && { say "skip $rn (failed earlier; see results/postplan/status_$rn.json)"; return 0; }
    stopped && { say "STOP file present: queue exits before $rn"; exit 75; }
    say "start $label ($rn)"
    $G/run_heavy.sh $label --timeout $to --mem-gb 5 -- python3 -u scripts/pp_train.py --run $rn "$@" --budget $((to-240)) >> logs_local/pp_$rn.log 2>&1
    rc=$?
    say "end $label rc=$rc"
    case $rc in
      0|3) ;;
      75) say "refused (stop file): queue exits"; exit 75;;
      137) stopped && { say "killed by stop file: queue exits (partial; resumes from last.pt)"; exit 75; }
           kills=$((kills+1)); reason=$(grep "KILLED \[$label\]" $G/guard.log | tail -1 | cut -d' ' -f4-)
           say "killed by guard ($reason), kill $kills/$maxk"
           if [ $kills -ge $maxk ]; then
             python3 -c "import json,sys;json.dump({'run':sys.argv[1],'state':'failed','error_type':'guard kill','error':sys.argv[2]},open('results/postplan/status_'+sys.argv[1]+'.json','w'),indent=1)" $rn "$reason"
           fi
           sleep 30;;
      *) failed_run $rn || python3 -c "import json,sys;json.dump({'run':sys.argv[1],'state':'failed','error_type':'exit code','error':'rc='+sys.argv[2]},open('results/postplan/status_'+sys.argv[1]+'.json','w'),indent=1)" $rn $rc
         say "$rn failed rc=$rc";;
    esac
    sleep 20      # let other agents take the slot between chunks
  done
  python3 scripts/postplan_analysis.py > /dev/null 2>&1
  return 0
}
if [ "$WHAT" = all ] || [ "$WHAT" = dentex ]; then
  for s in 2 3 4; do
    run du-dentex-coco-s$s pp_dentex_coco_y11s_s$s 3600 2 --data dentex --model yolo11s.pt --imgsz 640 --epochs 40 --seed $s --final last
    run du-dentex-du-s$s pp_dentex_du_y11s_s$s 3600 2 --data dentex --model runs/pxdet_y11s_s$((s%2))/weights/best.pt --imgsz 640 --epochs 40 --seed $s --final last
  done
fi
if [ "$WHAT" = all ] || [ "$WHAT" = cv ]; then
  [ -s results/postplan/dentex_cv_folds.json ] || python3 scripts/make_dentex_cv.py >> $LOG 2>&1
  for k in 0 1 2 3 4; do
    run du-dentexcv$k-coco pp_dentexcv${k}_coco_y11s_s0 3600 2 --data dentex_cv$k --model yolo11s.pt --imgsz 640 --epochs 40 --seed 0 --final last
    run du-dentexcv$k-du pp_dentexcv${k}_du_y11s_s0 3600 2 --data dentex_cv$k --model runs/pxdet_y11s_s0/weights/best.pt --imgsz 640 --epochs 40 --seed 0 --final last
  done
fi
if [ "$WHAT" = all ] || [ "$WHAT" = rtdetr ]; then
  run du-rtdetr-iodet pp_iodet_rtdetrl_s0 7200 2 --data iodet --model rtdetr-l.pt --imgsz 640 --epochs 40 --seed 0 --batch 2
fi
say "queue finished ($WHAT)"
