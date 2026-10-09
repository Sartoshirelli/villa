#!/usr/bin/env bash
# The 14 September arms on one pooled PHerc0841 render, in BOTH depth directions (28 runs),
# on the study-combined worktree with September's flags. Inference only; scoring is separate
# (seed agreement r is computed and frozen before any AUC). Usage: run_pherc0841.sh <short>
set -euo pipefail
SEG=$1
VENV=/mnt/media/vesuvius/.venv-vesuvius; PY=$VENV/bin/python
WT=/mnt/media/vesuvius/villa-wt-study; export PYTHONPATH=$WT/vesuvius/src
DATA=/mnt/media/vesuvius/data/pherc0841/$SEG; CK=/mnt/media/vesuvius/data/checkpoints/ink_9um
IN="$DATA/ct.zarr"
test -f "$DATA/meta.json" || { echo "no pooled data for $SEG"; exit 2; }
AMP=$(git -C /mnt/media/vesuvius/villa rev-parse backup/infer-amp-default-fp32-dfb05ed 2>/dev/null || echo dfb05edca)
git -C "$WT" merge-base --is-ancestor dfb05edca HEAD || { echo "worktree lacks the fp32 fix"; exit 3; }
echo "== $SEG inference start $(date -Is)"
$PY -c "import vesuvius; assert vesuvius.__file__.startswith('$WT'), vesuvius.__file__; print('package:', vesuvius.__file__)"
echo "code: $(git -C $WT log --oneline -1)  dirty: $(git -C $WT status --porcelain | wc -l) files"
DEPTH=$($PY -c "import zarr; print(zarr.open('$IN', mode='r').shape)")
echo "input shape=$DEPTH (window [8,25) = 17 planes centred on pooled plane 16)"
run () { local dir=$1 name=$2 ck=$3 ls=$4 le=$5 tta=$6; local out="$DATA/preds/$dir"; mkdir -p "$out"; local tif="$out/$name.tif"
  if [ -f "$tif" ]; then echo "skip $dir/$name (exists)"; return; fi
  local extra=""; [ "$tta" = "tta" ] && extra="--tta-mirror"
  echo "== $dir/$name: ckpt=$ck window=[$ls,$le) $extra"
  /usr/bin/time -f "%e s wall, %M KB maxrss" $PY -m vesuvius.ink_detection.inference.infer "$IN" "$CK/$ck" "$tif.partial.tif" \
    --overlap 0.5 --blend-mode hann --batch-size 4 --no-compile --num-workers 4 --amp-dtype default \
    --layer-start "$ls" --layer-end "$le" --direction "$dir" $extra 2> "$out/$name.log" >/dev/null
  mv "$tif.partial.tif" "$tif"; tail -1 "$out/$name.log"; }
for dir in reverse forward; do
  for seed in 42 43; do ck="hybrid_3d2d-seed$seed/step-075000.pth"
    for sh in 0 -2 -1 1 2; do run $dir "s${seed}_w${sh}" "$ck" $((8+sh)) $((25+sh)) none; done
    run $dir "s${seed}_w0_tta" "$ck" 8 25 tta; done
  for seed in 42 43; do run $dir "s${seed}_step50k_w0" "hybrid_3d2d-seed$seed/step-050000.pth" 8 25 none; done
done
echo "== provenance of one output:"; $VENV/bin/tiffcomment "$DATA/preds/reverse/s43_w0.tif" | head -c 700; echo
for dir in reverse forward; do for f in $DATA/preds/$dir/*.log; do printf "%-8s %-18s %s\n" $dir "$(basename $f .log)" "$(grep -E 's wall' $f | tail -1)"; done; done | tee /mnt/media/vesuvius/study/pherc0841/walltimes_$SEG.txt
echo "== $SEG inference done $(date -Is)"
