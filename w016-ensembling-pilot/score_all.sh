#!/usr/bin/env bash
# Score every preregistered arm (single runs + offline ensembles) for w016.
set -euo pipefail
PY=/mnt/media/vesuvius/.venv-vesuvius/bin/python; S=/mnt/media/vesuvius/study/w016; P=$S/preds
cd /mnt/media/vesuvius/study
$PY score_w016.py --labels "$S/labels/pherc0139-w016" --out "$S/results_w016.json" --bootstrap 1000 \
  "s42_centre=$P/s42_w0.tif" "s43_centre=$P/s43_w0.tif" \
  "s42_shift-2=$P/s42_w-2.tif" "s42_shift-1=$P/s42_w-1.tif" "s42_shift+1=$P/s42_w1.tif" "s42_shift+2=$P/s42_w2.tif" \
  "s43_shift-2=$P/s43_w-2.tif" "s43_shift-1=$P/s43_w-1.tif" "s43_shift+1=$P/s43_w1.tif" "s43_shift+2=$P/s43_w2.tif" \
  "s42_window_ens5=$P/s42_w-2.tif,$P/s42_w-1.tif,$P/s42_w0.tif,$P/s42_w1.tif,$P/s42_w2.tif" \
  "s43_window_ens5=$P/s43_w-2.tif,$P/s43_w-1.tif,$P/s43_w0.tif,$P/s43_w1.tif,$P/s43_w2.tif" \
  "seed_ens_centre=$P/s42_w0.tif,$P/s43_w0.tif" \
  "s42_tta=$P/s42_w0_tta.tif" "s43_tta=$P/s43_w0_tta.tif" \
  "s42_step50k_centre=$P/s42_step50k_w0.tif" "s43_step50k_centre=$P/s43_step50k_w0.tif" \
  "seed_ens_step50k=$P/s42_step50k_w0.tif,$P/s43_step50k_w0.tif" \
  "seeds_x_windows_ens10=$P/s42_w-2.tif,$P/s42_w-1.tif,$P/s42_w0.tif,$P/s42_w1.tif,$P/s42_w2.tif,$P/s43_w-2.tif,$P/s43_w-1.tif,$P/s43_w0.tif,$P/s43_w1.tif,$P/s43_w2.tif" \
  "everything_ens12=$P/s42_w-2.tif,$P/s42_w-1.tif,$P/s42_w0.tif,$P/s42_w1.tif,$P/s42_w2.tif,$P/s43_w-2.tif,$P/s43_w-1.tif,$P/s43_w0.tif,$P/s43_w1.tif,$P/s43_w2.tif,$P/s42_w0_tta.tif,$P/s43_w0_tta.tif" \
  2>&1 | tee "$S/results_w016.txt"
echo "== wall times"; for f in $P/*.log; do printf "%-22s %s\n" "$(basename $f .log)" "$(grep -E 's wall' $f | tail -1)"; done | tee "$S/walltimes.txt"
