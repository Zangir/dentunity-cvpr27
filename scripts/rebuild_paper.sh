#!/bin/bash
# Regenerate GENERATED paper outputs only (tables, numbers.tex, figures), compile, export.
# Honours paper/.edit_lock shared with the prose reviser: waits while it exists, holds it while rebuilding.
R=<path>
L=$R/paper/.edit_lock
cd $R/experiments/repo || exit 1
for i in $(seq 1 360); do [ -e "$L" ] || break; sleep 10; done
[ -e "$L" ] && { echo "lock still held after 1 h: $(cat $L)"; exit 3; }
echo "experiments-agent rebuild $(date '+%F %T') pid $$" > "$L"
trap 'rm -f "$L"' EXIT
python3 run_experiment.py summarize >/dev/null 2>&1 || { echo "summarize failed"; exit 1; }
python3 scripts/make_tables.py ../../paper >/dev/null || { echo "make_tables failed"; exit 1; }
python3 scripts/split_class_stats.py >/dev/null && python3 scripts/make_appendix_tables.py ../../paper >/dev/null || { echo "make_appendix_tables failed"; exit 1; }
python3 scripts/make_figures.py ../../paper >/dev/null 2>&1 || { echo "make_figures failed"; exit 1; }
python3 scripts/make_readme.py >/dev/null
cd $R/paper
# main <-> supplement pointers are static text generated from the .aux files (make_supprefs.py), so both
# documents also compile alone on Overleaf: main -> pointers -> supplement -> pointers -> main
python3 ../../../tools/compile_paper.py --paper-dir . >/dev/null; python3 make_supprefs.py >/dev/null || exit 1
python3 ../../../tools/compile_paper.py --paper-dir . --main supplementary.tex >/dev/null; python3 make_supprefs.py || exit 1
python3 ../../../tools/compile_paper.py --paper-dir . | tail -1
echo "main undefined/overfull: $(grep -cE 'Overfull|undefined' main.log)"
for pg in 8 9; do echo "p$pg: $(pdftotext -f $pg -l $pg main.pdf - 2>/dev/null | grep -oE 'default for public dental data|LLM Usage Statement|^References' | tr '\n' '|')"; done
python3 ../../../tools/compile_paper.py --paper-dir . --main supplementary.tex | tail -1
echo "supp undefined/overfull: $(grep -cE 'Overfull|undefined' supplementary.log)"
./push_overleaf.sh 2>&1 | tail -2
