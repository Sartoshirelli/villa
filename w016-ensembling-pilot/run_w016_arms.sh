#!/usr/bin/env bash
# Inference arms for the w016 pilot (preregistered 2026-09-28). Runs on the
# infer-amp-default-fp32 branch so --amp-dtype default is true fp32.
set -euo pipefail
VENV=/mnt/media/vesuvius/.venv-vesuvius; PY=$VENV/bin/python
REPO=/mnt/media/vesuvius/villa; S=/mnt/media/vesuvius/study/w016; D=/mnt/media/vesuvius/data/checkpoints/ink_9um
IN="$S/w016_pooled.zarr"; OUT="$S/preds"; mkdir -p "$OUT"
git -C "$REPO" checkout -q infer-amp-default-fp32
DEPTH=$($PY -c "import zarr; z=zarr.open('$IN', mode='r'); print(z.shape[0] if hasattr(z,'shape') else z['0'].shape[0])")
echo "input depth=$DEPTH (17 of 21 used per run)"
run () { # name checkpoint layer_start layer_end tta
  local name=$1 ck=$2 ls=$3 le=$4 tta=$5; local tif="$OUT/$name.tif"
  if [ -f "$tif" ]; then echo "skip $name (exists)"; return; fi
  local extra=""; [ "$tta" = "tta" ] && extra="--tta-mirror"
  echo "== $name: ckpt=$ck window=[$ls,$le) $extra"
  /usr/bin/time -f "%e s wall, %M KB maxrss" $PY -m vesuvius.ink_detection.inference.infer "$IN" "$D/$ck" "$tif" \
    --overlap 0.5 --blend-mode hann --batch-size 4 --no-compile --num-workers 4 --amp-dtype default \
    --layer-start "$ls" --layer-end "$le" $extra 2> "$OUT/$name.log" >/dev/null
  tail -1 "$OUT/$name.log"
}
# centred 17-of-21 window is [2,19); shifts move it by -2..+2
for seed in 42 43; do
  ck="hybrid_3d2d-seed$seed/step-075000.pth"
  for sh in -2 -1 0 1 2; do
    ls=$((2+sh)); le=$((19+sh)); run "s${seed}_w${sh}" "$ck" "$ls" "$le" none
  done
  run "s${seed}_w0_tta" "$ck" 2 19 tta
done
for seed in 42 43; do run "s${seed}_step50k_w0" "hybrid_3d2d-seed$seed/step-050000.pth" 2 19 none; done
git -C "$REPO" checkout -q ink-tiff-provenance
echo "== all arms done $(date -Is)"
