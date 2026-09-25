#!/usr/bin/env bash
# Real-data evidence for PR 1: run the tutorial's 9 um flat inference on the
# published w035 surface volume, once on main ("before") and once on the
# ink-tiff-provenance branch ("after"); capture tiffinfo, the provenance
# reader's output, and a pixel hash for both.
#
# GATE: the scroll data is licensed (dl.ash2txt.org/LICENSE.txt). This script
# refuses to download anything until the package's agreement file exists,
# i.e. until Tayte has run `vesuvius.accept_terms --yes` himself.
set -euo pipefail
VENV=/mnt/media/vesuvius/.venv-vesuvius
PY=$VENV/bin/python
REPO=/mnt/media/vesuvius/villa
DATA=/mnt/media/vesuvius/data
OUT=/mnt/media/vesuvius/evidence/w035
AGREEMENT=$REPO/vesuvius/src/vesuvius/install/agreement.txt
SEG="PHerc0139/segments/20260317000000-w035_2026031718"
SV="$SEG/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr"
CKPT_REL="hybrid_3d2d-seed42/step-075000.pth"
BATCH="${BATCH:-4}"

if [ ! -f "$AGREEMENT" ]; then
  echo "REFUSING: $AGREEMENT not found. The data licence has not been accepted on this machine." >&2
  echo "Tayte: read https://dl.ash2txt.org/LICENSE.txt then run: $VENV/bin/vesuvius.accept_terms --yes" >&2
  exit 3
fi
mkdir -p "$DATA/w035" "$DATA/checkpoints/ink_9um" "$OUT"

# 1. data (once): the published 9.362 um render (~1.0 GB, 2,223 objects) via anonymous S3
if [ ! -f "$DATA/w035/9um.zarr/.zattrs" ]; then
  echo "== downloading the published w035 9.362 um surface volume"
  $PY - <<PYEOF
import s3fs
fs = s3fs.S3FileSystem(anon=True)
fs.get("vesuvius-challenge-open-data/$SV", "$DATA/w035/9um.zarr", recursive=True)
print("downloaded")
PYEOF
fi
# 2. checkpoint (once): 138 MB from Hugging Face
if [ ! -f "$DATA/checkpoints/ink_9um/$CKPT_REL" ]; then
  echo "== downloading checkpoint $CKPT_REL"
  $PY - <<PYEOF
from huggingface_hub import hf_hub_download
hf_hub_download("scrollprize/ink_9um", "$CKPT_REL", local_dir="$DATA/checkpoints/ink_9um")
print("downloaded")
PYEOF
fi
CKPT="$DATA/checkpoints/ink_9um/$CKPT_REL"
sha256sum "$CKPT" | tee "$OUT/checkpoint.sha256"

run_one () {  # $1 = git ref, $2 = label
  local ref=$1 label=$2
  git -C "$REPO" checkout -q "$ref"
  echo "== [$label] $(git -C "$REPO" log --oneline -1)"
  local tif="$OUT/w035_9um_${label}.tif"
  rm -f "$tif" "${tif%.tif}_reverse.tif"
  /usr/bin/time -v $PY -m vesuvius.ink_detection.inference.infer \
      "$DATA/w035/9um.zarr" "$CKPT" "$tif" \
      --overlap 0.5 --blend-mode hann --batch-size "$BATCH" --direction both $EXTRA_FLAGS \
      2> "$OUT/${label}.log" | tee "$OUT/${label}.stdout"
  grep -E "Maximum resident|Elapsed" "$OUT/${label}.log" || true
  for f in "$tif" "${tif%.tif}_reverse.tif"; do
    echo "--- tiffinfo $(basename "$f")" | tee -a "$OUT/${label}.tiffinfo.txt"
    $PY - "$f" <<PYEOF | tee -a "$OUT/${label}.tiffinfo.txt"
import sys, hashlib, tifffile, numpy as np
p = sys.argv[1]
with tifffile.TiffFile(p) as t:
    page = t.pages[0]
    for tag in page.tags.values():
        v = tag.value
        if tag.name == "ImageDescription": v = v[:400] + ("..." if len(v) > 400 else "")
        if tag.name in ("TileOffsets","TileByteCounts","StripOffsets","StripByteCounts"): v = f"<{len(v)} entries>"
        print(f"  {tag.name}: {v}")
arr = tifffile.imread(p)
print("  pixel sha256:", hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(), "shape", arr.shape)
PYEOF
  done
  if [ "$label" = "after" ]; then
    # tiffcomment ships with tifffile and prints the ImageDescription verbatim
    for f in "$tif" "${tif%.tif}_reverse.tif"; do echo "### $(basename "$f")"; $VENV/bin/tiffcomment "$f"; echo; done | tee "$OUT/provenance_tiffcomment.txt"
  fi
}
echo "== input scale metadata (level 0 axes/unit/scale from the surface volume's .zattrs)"
$PY - "$DATA/w035/9um.zarr" <<PYEOF | tee "$OUT/input_scale.txt"
import json, sys
a = json.load(open(sys.argv[1] + "/.zattrs"))["multiscales"][0]
print("axes:", [(ax.get("name"), ax.get("unit")) for ax in a["axes"]])
print("level 0 scale:", a["datasets"][0]["coordinateTransformations"])
PYEOF
echo "== environment"; { nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; $PY -c "import torch, tifffile, zarr; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| tifffile', tifffile.__version__, '| zarr', zarr.__version__)"; git -C "$REPO" log --oneline -1 main; git -C "$REPO" log --oneline -1 ink-tiff-provenance; } | tee "$OUT/environment.txt"

# Primary comparison: settings that remove the known sources of GPU drift
# (no torch.compile, no autocast, no worker processes), main vs branch.
EXTRA_FLAGS="--no-compile --amp-dtype default --num-workers 0"
run_one main before
run_one ink-tiff-provenance after
# Secondary: the tutorial's defaults, main twice, to report run-to-run variance.
EXTRA_FLAGS=""
run_one main before_default
run_one main before_default2
git -C "$REPO" checkout -q ink-tiff-provenance

echo "== pixel comparison"
$PY - "$OUT" <<PYEOF | tee "$OUT/pixel_comparison.txt"
import sys, hashlib, numpy as np, tifffile
from pathlib import Path
out = Path(sys.argv[1])
def load(label, suffix=""):
    return tifffile.imread(out / f"w035_9um_{label}{suffix}.tif")
def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
for suffix, name in (("", "forward"), ("_reverse", "reverse")):
    before, after = load("before", suffix), load("after", suffix)
    d1, d2 = load("before_default", suffix), load("before_default2", suffix)
    d_ba = np.abs(before.astype(int) - after.astype(int)); d_dd = np.abs(d1.astype(int) - d2.astype(int))
    print(f"[{name}] shape={before.shape}")
    print(f"  no-compile/fp32  main={h(before)} branch={h(after)}  identical={bool(np.array_equal(before, after))} pixels differing={int((d_ba>0).sum())} max|diff|={int(d_ba.max())}")
    print(f"  tutorial defaults main={h(d1)} main-again={h(d2)}  identical={bool(np.array_equal(d1, d2))} pixels differing={int((d_dd>0).sum())} max|diff|={int(d_dd.max())}  (run-to-run variance)")
# the after-TIFF must carry 25400/9.362 px per inch
with tifffile.TiffFile(out / "w035_9um_after.tif") as t:
    xn, xd = t.pages[0].tags["XResolution"].value
    print(f"after XResolution = {xn}/{xd} = {xn/xd:.6f} px/inch -> {25400*xd/xn:.6f} um/px (expected 9.362000)")
    assert abs(25400*xd/xn - 9.362) < 1e-9
PYEOF
echo "== done; see $OUT/pixel_comparison.txt, $OUT/before.tiffinfo.txt, $OUT/after.tiffinfo.txt, $OUT/provenance.json"
