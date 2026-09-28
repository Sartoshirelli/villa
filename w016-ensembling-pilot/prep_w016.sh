#!/usr/bin/env bash
# Data prep for the ensembling pilot on validation segment pherc0139-w016 (= public w029).
set -euo pipefail
VENV=/mnt/media/vesuvius/.venv-vesuvius; PY=$VENV/bin/python; HF=$VENV/bin/hf
S=/mnt/media/vesuvius/study/w016; D=/mnt/media/vesuvius/data
SEG="PHerc0139/segments/20250108000004-w029_2025010827"
VOL="s3://vesuvius-challenge-open-data/$SEG/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr"
test -f /mnt/media/vesuvius/villa/vesuvius/src/vesuvius/install/agreement.txt || { echo "licence not accepted"; exit 3; }
echo "== labels (ink_9um, w016 only)"
mkdir -p "$S/labels"
$HF buckets sync "hf://buckets/scrollprize/datasets/ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0139-w016" "$S/labels/pherc0139-w016" 2>&1 | tail -3 || echo "hf sync failed (see above)"
ls "$S/labels/pherc0139-w016" 2>/dev/null || true
echo "== second seed checkpoint"
$PY - <<PYEOF
from huggingface_hub import hf_hub_download
for f in ("hybrid_3d2d-seed43/step-075000.pth", "hybrid_3d2d-seed42/step-050000.pth", "hybrid_3d2d-seed43/step-050000.pth"):
    hf_hub_download("scrollprize/ink_9um", f, local_dir="$D/checkpoints/ink_9um"); print("ok", f)
PYEOF
echo "== pooled ~9.6 um input from the public 2.399 um volume (level 2, 4x z mean)"
cd /mnt/media/vesuvius/villa/vesuvius
test -f "$S/w016_pooled.zarr/.zattrs" || /usr/bin/time -v $PY -m vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input "$VOL" "$S/w016_pooled.zarr" --workers 8 2>&1 | grep -E "Elapsed|Maximum resident|tiles|Error|error|Traceback" || true
$PY - <<PYEOF
import zarr, json
z = zarr.open("$S/w016_pooled.zarr", mode="r")
print("pooled:", getattr(z, "shape", None) or {k: z[k].shape for k in list(z.keys())[:3]}, "attrs:", json.dumps(dict(z.attrs))[:300])
PYEOF
echo "== done $(date -Is)"
