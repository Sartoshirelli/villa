#!/usr/bin/env bash
# Pool a public 2.399 um surface volume to the ~9.6 um isotropic ink_9um input (level 2, 4x z mean).
# Usage: prep_seg.sh <seg-name> <s3 zarr url>
set -euo pipefail
SEG=$1; VOL=$2
VENV=/mnt/media/vesuvius/.venv-vesuvius; PY=$VENV/bin/python
S=/mnt/media/vesuvius/study/$SEG
test -f /mnt/media/vesuvius/villa/vesuvius/src/vesuvius/install/agreement.txt || { echo "licence not accepted"; exit 3; }
mkdir -p "$S"
cd /mnt/media/vesuvius/villa/vesuvius
echo "== $SEG: pooling $VOL  $(date -Is)"
test -f "$S/${SEG}_pooled.zarr/.zattrs" || /usr/bin/time -v $PY -m vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input "$VOL" "$S/${SEG}_pooled.zarr" --workers 8 2>&1 | grep -E "Elapsed|Maximum resident|tiles|Error|error|Traceback" || true
$PY - <<PYEOF
import zarr, json
z = zarr.open("$S/${SEG}_pooled.zarr", mode="r")
print("pooled:", getattr(z, "shape", None) or {k: z[k].shape for k in list(z.keys())[:3]}, "attrs:", json.dumps(dict(z.attrs))[:300])
PYEOF
echo "== $SEG done $(date -Is)"
