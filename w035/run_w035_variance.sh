#!/usr/bin/env bash
# Second half of the evidence: the tutorial's default settings minus torch.compile
# (triton cannot build its driver stub here: no Python headers), i.e. autocast
# from the checkpoint config and 4 loader workers. main twice for run-to-run
# variance, the branch once for parity under the same flags, then the comparison.
set -euo pipefail
VENV=/mnt/media/vesuvius/.venv-vesuvius; PY=$VENV/bin/python
REPO=/mnt/media/vesuvius/villa; DATA=/mnt/media/vesuvius/data; OUT=/mnt/media/vesuvius/evidence/w035
CKPT="$DATA/checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth"
BATCH="${BATCH:-4}"
run_one () {
  local ref=$1 label=$2
  git -C "$REPO" checkout -q "$ref"
  echo "== [$label] $(git -C "$REPO" log --oneline -1)"
  local tif="$OUT/w035_9um_${label}.tif"
  rm -f "$tif" "${tif%.tif}_reverse.tif"
  /usr/bin/time -v $PY -m vesuvius.ink_detection.inference.infer \
      "$DATA/w035/9um.zarr" "$CKPT" "$tif" \
      --overlap 0.5 --blend-mode hann --batch-size "$BATCH" --direction both --no-compile \
      2> "$OUT/${label}.log" > "$OUT/${label}.stdout"
  grep -E "Maximum resident|Elapsed" "$OUT/${label}.log" || true
  for f in "$tif" "${tif%.tif}_reverse.tif"; do
    $PY - "$f" <<PYEOF | tee -a "$OUT/${label}.tiffinfo.txt"
import sys, hashlib, tifffile, numpy as np
p = sys.argv[1]
with tifffile.TiffFile(p) as t:
    tags = {tg.name: tg.value for tg in t.pages[0].tags.values()}
print(f"--- {p.split('/')[-1]}: XResolution={tags.get('XResolution')} ResolutionUnit={tags.get('ResolutionUnit')}")
arr = tifffile.imread(p)
print("  pixel sha256:", hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(), "shape", arr.shape)
PYEOF
  done
}
run_one main before_default
run_one main before_default2
run_one ink-tiff-provenance after_default
git -C "$REPO" checkout -q ink-tiff-provenance

echo "== pixel comparison"
$PY - "$OUT" <<PYEOF | tee "$OUT/pixel_comparison.txt"
import sys, hashlib, numpy as np, tifffile
from pathlib import Path
out = Path(sys.argv[1])
def load(label, suffix=""): return tifffile.imread(out / f"w035_9um_{label}{suffix}.tif")
def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
def diff(a, b):
    d = np.abs(a.astype(int) - b.astype(int)); return int((d > 0).sum()), int(d.max())
for suffix, name in (("", "forward"), ("_reverse", "reverse")):
    before, after = load("before", suffix), load("after", suffix)
    d1, d2, a1 = load("before_default", suffix), load("before_default2", suffix), load("after_default", suffix)
    print(f"[{name}] shape={before.shape}")
    n, m = diff(before, after);  print(f"  fp32, no compile, workers 0 : main={h(before)} branch={h(after)}  identical={np.array_equal(before, after)}  differing={n} max|diff|={m}")
    n, m = diff(d1, a1);         print(f"  autocast, workers 4          : main={h(d1)} branch={h(a1)}  identical={np.array_equal(d1, a1)}  differing={n} max|diff|={m}")
    n, m = diff(d1, d2);         print(f"  autocast, workers 4, main x2 : main={h(d1)} main-again={h(d2)}  identical={np.array_equal(d1, d2)}  differing={n} max|diff|={m}  (run-to-run variance)")
    n, m = diff(before, d1);     print(f"  fp32 vs autocast (main)      : differing={n} max|diff|={m}  (precision effect, for context)")
with tifffile.TiffFile(out / "w035_9um_after.tif") as t:
    xn, xd = t.pages[0].tags["XResolution"].value
    print(f"after XResolution = {xn}/{xd} = {xn/xd:.6f} px/inch -> {25400*xd/xn:.6f} um/px (expected 9.362000)")
    assert abs(25400*xd/xn - 9.362) < 1e-9
PYEOF
echo "== done"
