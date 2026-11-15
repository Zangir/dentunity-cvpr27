#!/bin/bash
# Exit (and print why) when a new result file appears, the queue reports rc!=0,
# or the guard kills >=3 du chunks. Used to wake the agent; no compute.
cd "$(dirname "$0")/.." || exit 1
G=<path>
shopt -s nullglob
snap(){ ls results/yolo_*.json results/*.done results/cls_ft_*.json results/dentex_overlap.json 2>/dev/null | sort; }
s0=$(snap); k0=$(grep -c "KILLED \[du" $G); q0=$(grep -ac "rc=[1-9]" ../../logs/queue.log)
while true; do
  sleep 60
  s=$(snap); [ "$s" != "$s0" ] && { comm -13 <(echo "$s0") <(echo "$s") | sed 's/^/NEW /'; exit 0; }
  [ "$(grep -c "KILLED \[du" $G)" -gt "$((k0+2))" ] && { echo "GUARD KILLS"; grep "KILLED \[du" $G | tail -3; exit 0; }
  [ "$(grep -ac "rc=[1-9]" ../../logs/queue.log)" != "$q0" ] && { echo "QUEUE ERROR"; grep -a "rc=[1-9]" ../../logs/queue.log | tail -2; exit 0; }
  grep -q "ALLDONE" ../../logs/queue.log && { echo ALLDONE; exit 0; }
done
