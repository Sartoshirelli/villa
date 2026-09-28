#!/usr/bin/env bash
# The 14 preregistered arms + scoring for one validation segment, on the combined study worktree
# (main + #1890 rebased + #1891). Usage: run_seg_pipeline.sh <seg>
set -euo pipefail
SEG=$1
VENV=/mnt/media/vesuvius/.venv-vesuvius; PY=$VENV/bin/python
WT=/mnt/media/vesuvius/villa-wt-study; export PYTHONPATH=$WT/vesuvius/src
S=/mnt/media/vesuvius/study/$SEG; D=/mnt/media/vesuvius/data/checkpoints/ink_9um
IN="$S/${SEG}_pooled.zarr"; OUT="$S/preds"; mkdir -p "$OUT"
AMP=$(git -C /mnt/media/vesuvius/villa rev-parse infer-amp-default-fp32); PROV=$(git -C /mnt/media/vesuvius/villa rev-parse ink-tiff-provenance)
echo "== $SEG pipeline start $(date -Is)"
until grep -q "== $SEG done" /mnt/media/vesuvius/study/logs/prep_$SEG.log 2>/dev/null; do sleep 30; done
until [ ! -f "$WT/.git/MERGE_HEAD" ] && git -C "$WT" merge-base --is-ancestor "$AMP" HEAD && git -C "$WT" merge-base --is-ancestor "$PROV" HEAD; do echo "waiting for study worktree to contain #1890+#1891"; sleep 60; done
$PY -c "import vesuvius; assert vesuvius.__file__.startswith('$WT'), vesuvius.__file__; print('package:', vesuvius.__file__)"
echo "code: $(git -C $WT log --oneline -1)"
DEPTH=$($PY -c "import zarr; z=zarr.open('$IN', mode='r'); print(z.shape[0] if hasattr(z,'shape') else z['0'].shape[0])")
echo "input depth=$DEPTH (17 of 21 used per run)"
run () { local name=$1 ck=$2 ls=$3 le=$4 tta=$5; local tif="$OUT/$name.tif"
  if [ -f "$tif" ]; then echo "skip $name (exists)"; return; fi
  local extra=""; [ "$tta" = "tta" ] && extra="--tta-mirror"
  echo "== $name: ckpt=$ck window=[$ls,$le) $extra"
  /usr/bin/time -f "%e s wall, %M KB maxrss" $PY -m vesuvius.ink_detection.inference.infer "$IN" "$D/$ck" "$tif" \
    --overlap 0.5 --blend-mode hann --batch-size 4 --no-compile --num-workers 4 --amp-dtype default \
    --layer-start "$ls" --layer-end "$le" $extra 2> "$OUT/$name.log" >/dev/null
  tail -1 "$OUT/$name.log"; }
for seed in 42 43; do ck="hybrid_3d2d-seed$seed/step-075000.pth"
  for sh in -2 -1 0 1 2; do ls=$((2+sh)); le=$((19+sh)); run "s${seed}_w${sh}" "$ck" "$ls" "$le" none; done
  run "s${seed}_w0_tta" "$ck" 2 19 tta; done
for seed in 42 43; do run "s${seed}_step50k_w0" "hybrid_3d2d-seed$seed/step-050000.pth" 2 19 none; done
echo "== arms done $(date -Is); provenance of one output:"; $VENV/bin/tiffcomment "$OUT/s43_w0.tif" | head -c 600; echo
cd /mnt/media/vesuvius/study; P=$OUT
$PY score_w016.py --help | grep -q -- "--segment" || { echo "score_w016.py has no --segment flag"; exit 4; }
$PY score_w016.py --segment "$SEG" --labels "$S/labels/$SEG" --out "$S/results_$SEG.json" --bootstrap 1000 \
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
  2>&1 | tee "$S/results_$SEG.txt"
for f in $P/*.log; do printf "%-22s %s\n" "$(basename $f .log)" "$(grep -E 's wall' $f | tail -1)"; done | tee "$S/walltimes.txt"
echo "== $SEG pipeline done $(date -Is)"
