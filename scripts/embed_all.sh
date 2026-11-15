#!/bin/bash
# Link embeddings for all groups (resumable). Run through the guarded runner.
cd "$(dirname "$0")/.." || exit 1
export SSL_CERT_FILE="$(python3 -m certifi)"
for g in io_cls64 io_det io_aug px; do
  python3 -u -m dentunity.embed $g data/list_$g.txt || exit 1
done
