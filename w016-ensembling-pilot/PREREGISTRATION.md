# Preregistration — inference-time ensembling for ink_9um (depth windows, seeds, TTA)

**Written 2026-09-28, before any result exists. Fixed once posted; deviations will be reported as such.**

## Question
Tutorial 5 and the ink_9um model card recommend, without measurement, shifting the depth window
(`--layer-start/--layer-end`), "averaging the predictions over a few nearby windows as a simple
ensemble", and "running both seeds and a few different steps". Which of these actually improves ink
detection on a held-out validation segment, by how much, and at what cost on an 8 GB consumer GPU?

## Data (fixed)
- Pilot segment: `pherc0139-w016` = public segment `20250108000004-w029_2025010827` (PHerc. 0139),
  one of the three segments the released checkpoints report validation metrics on. Input: the public
  2.399 µm surface volume `2.399um-0.22m-78keV-volume-20260102150214.zarr`, pooled to ~9.6 µm with
  `vesuvius.ink_detection.preprocessing.prepare_9um_isotropic_input` (level 2, 4× z mean, 21 slices),
  exactly as the models were trained.
- Labels: `ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0139-w016/` — `_inklabels.zarr`
  (annotated at Z=10), `_validation_mask.zarr` (evaluation region), `_supervision_mask.zarr`.
- Follow-up (October): the other two validation segments (`pherc0814-46527`, `pherc1667-w029`) and
  the held-out PHerc0841 renders from #1867.

## Arms (fixed)
Checkpoints: `hybrid_3d2d-seed42/step-075000`, `hybrid_3d2d-seed43/step-075000` (final step of both
runs); a later arm adds `step-050000` of each. Direction: forward (the labels' orientation).
1. **Single centred window** per checkpoint (the default `select_layer_indices` window).
2. **Window shift** ±1, ±2 slices (`--layer-start/--layer-end` moved together), each as a single run.
3. **Window ensemble**: mean of the 5 predictions in {−2, −1, 0, +1, +2}, computed offline from the
   saved TIFFs (no extra inference).
4. **Seed ensemble**: mean of seed42 and seed43 at the centred window.
5. **TTA mirror** (`--tta-mirror`) on vs off at the centred window.
6. **Everything**: seeds × windows × TTA mean.
All runs `--overlap 0.5 --blend-mode hann --batch-size 4 --no-compile --amp-dtype default`
(true fp32 on the branch with #1890; fp16 arms reported separately if time allows).

## Metric (fixed)
Pixel ROC-AUC and average precision of the uint8 prediction against `inklabels[Z=10] > 0`, restricted to
`validation_mask > 0`, computed with scikit-learn on the full-resolution canvas. Uncertainty: block
bootstrap over 64×64 px blocks of the validation region, 1000 resamples, 95 % percentile interval.
Cost: wall time and peak GPU memory per run on an RTX 2070 SUPER (8 GB), torch 2.14, from the run logs.

## Decision rule (fixed)
An arm "helps" if its AUC exceeds the single-centred-window AUC of the better seed by more than the
bootstrap interval half-width. Otherwise the tutorial's advice is reported as not supported on this
segment, which is a publishable result (cf. Camillo's depth-window study, Aug 2026).

## What is not claimed
One validation segment is a pilot, not a verdict; the segment is in-distribution (PHerc. 0139); no
retraining; no claim about other resolutions.

## Provenance
Every prediction TIFF carries its own run record (villa #1891), so each arm's checkpoint sha256,
window and settings are read back from the file, not from a notebook.
