#!/usr/bin/env bash
# Run inference on each PHerc0841 segment as soon as its pooled data exists.
set -uo pipefail
cd /mnt/media/vesuvius/study/pherc0841
for SEG in ag144 ag174 w00; do
  until [ -f /mnt/media/vesuvius/data/pherc0841/$SEG/meta.json ]; do sleep 30; done
  ./run_pherc0841.sh $SEG > logs/infer_$SEG.log 2>&1; echo "$SEG exit $? $(date -Is)"
done
echo "ALL INFERENCE DONE $(date -Is)"
