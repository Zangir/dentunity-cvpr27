#!/bin/bash
# Sequential driver: one guarded <=600 s chunk at a time (the lock is released between
# chunks so other agents can interleave). Stop by touching data/STOP.
cd "$(dirname "$0")/.." || exit 1
export SSL_CERT_FILE="$(python3 -m certifi)" YOLO_OFFLINE=1
G=./run_heavy.sh
mkdir -p ../../logs
for i in $(seq 1 400); do
  [ -f data/STOP ] && { echo "STOP file"; exit 0; }
  $G du-queue --timeout 600 --mem-gb 5 -- python3 -u scripts/run_queue.py 580 >> ../../logs/queue.log 2>&1
  rc=$?
  grep -q "ALLDONE" ../../logs/queue.log && exit 0
  tail -3 ../../logs/queue.log | grep -q "rc=" && [ $rc -ne 0 ] && sleep 30
  sleep 20   # give other agents a chance to take the slot
done
