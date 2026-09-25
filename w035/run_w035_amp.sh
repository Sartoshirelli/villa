#!/usr/bin/env bash
# Evidence for the --amp-dtype default fix: on the fixed branch, 'default' must
# now be true fp32 (a new pixel hash, differing from the fp16 run that main's
# 'default' produced), and a forced fp16 run must reproduce main's hash exactly.
set -euo pipefail
VENV=/mnt/media/vesuvius/.venv-vesuvius; PY=$VENV/bin/python
REPO=/mnt/media/vesuvius/villa; DATA=/mnt/media/vesuvius/data; OUT=/mnt/media/vesuvius/evidence/w035
CKPT="$DATA/checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth"
git -C "$REPO" checkout -q infer-amp-default-fp32
echo "== branch $(git -C "$REPO" log --oneline -1)"
run_one () {  # label, amp value
  local label=$1 amp=$2
  local tif="$OUT/w035_9um_${label}.tif"; rm -f "$tif" "${tif%.tif}_reverse.tif"
  echo "== [$label] --amp-dtype $amp"
  /usr/bin/time -v $PY -m vesuvius.ink_detection.inference.infer \
      "$DATA/w035/9um.zarr" "$CKPT" "$tif" \
      --overlap 0.5 --blend-mode hann --batch-size 4 --direction both --no-compile --num-workers 4 --amp-dtype "$amp" \
      2> "$OUT/${label}.log" > "$OUT/${label}.stdout"
  grep -E "Elapsed" "$OUT/${label}.log" || true
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 || true
}
run_one fix_default default
run_one fix_fp16 fp16
git -C "$REPO" checkout -q ink-tiff-provenance
echo "== comparison"
$PY - "$OUT" <<PYEOF | tee "$OUT/amp_comparison.txt"
import sys, hashlib, numpy as np, tifffile
from pathlib import Path
out = Path(sys.argv[1])
def load(label, suffix=""): return tifffile.imread(out / f"w035_9um_{label}{suffix}.tif")
def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
def diff(a, b):
    d = np.abs(a.astype(int) - b.astype(int)); nz = (a > 0) | (b > 0)
    return int((d > 0).sum()), int(d.max()), int(nz.sum())
for suffix, name in (("", "forward"), ("_reverse", "reverse")):
    old_default = load("before_default", suffix)   # main, --amp-dtype default (silently fp16)
    fix_default = load("fix_default", suffix)      # branch, --amp-dtype default (fp32)
    fix_fp16 = load("fix_fp16", suffix)            # branch, --amp-dtype fp16
    n, m, nz = diff(old_default, fix_default)
    print(f"[{name}] main 'default' (fp16) = {h(old_default)} | branch 'default' (fp32) = {h(fix_default)} -> differing pixels = {n} of {nz} nonzero ({100*n/nz:.2f}%), max|diff| = {m}")
    n2, m2, _ = diff(old_default, fix_fp16)
    print(f"[{name}] main 'default' (fp16) = {h(old_default)} | branch 'fp16'          = {h(fix_fp16)} -> identical = {np.array_equal(old_default, fix_fp16)} (differing {n2})")
PYEOF
echo "== done"
