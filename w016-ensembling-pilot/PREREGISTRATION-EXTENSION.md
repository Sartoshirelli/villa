# Preregistration — extension to the remaining two validation segments

**Written 2026-09-28 11:23 EDT, before any result on these segments exists; committed and pushed to this branch before the first inference run (the commit time is the timestamp).**

Extends `PREREGISTRATION.md` (the w016 pilot) to the other two segments the ink_9um checkpoints report validation metrics on. Everything not listed here is unchanged: the same 14 inference arms, the same flags (`--overlap 0.5 --blend-mode hann --batch-size 4 --no-compile --num-workers 4 --amp-dtype default`, forward direction), the same offline ensembles, the same metric (pixel ROC-AUC / AP on `validation_mask > 0` against `inklabels[Z=10] > 0`), the same 64-px block bootstrap (1000 resamples, 95 % percentile interval) and the same decision rule (an arm helps only if it beats the better seed's single centred run by more than the bootstrap half-width).

## Segments

| name | public segment | input (pooled with `prepare_9um_isotropic_input`: level 2, 4× z mean, 21 slices) | level-2 canvas |
|---|---|---|---|
| pherc0814-46527 | `PHerc0814/segments/20260226000000-46527_2um_try2` | `2.399um-0.22m-78keV-volume-20260309142202.zarr` | 2130 × 3455 |
| pherc1667-w029 | `PHerc1667/segments/20251212185248-w029_20251212185248662_flatboi` | `2.399um-0.22m-78keV-volume-20251217075048.zarr` | 9500 × 7830 |

Labels: `ink_9um/labels/aligned-scrollprizeorg-21slices/<segment>/` (`_inklabels.zarr`, `_validation_mask.zarr`). Both are validation cases of the training run, so still in-distribution — but from two scrolls other than the pilot's PHerc. 0139.

## Questions fixed in advance

- **Q1, replication of the seed effect.** On each segment, does seed43's centred run beat seed42's by more than the bootstrap half-width? Recorded per segment; no prediction is made.
- **Q2, ensembles.** Does any ensemble arm (5-window mean, seed mean, TTA, seeds × windows, everything) beat the better seed's single centred run by more than the bootstrap half-width? Same rule as the pilot.
- **Q3, who benefits.** Do window shifts and TTA help the weaker seed more than the stronger one, as on w016? Reported as ΔAUC per seed.
- **Summary across the three segments:** for every ensemble arm, the per-segment ΔAUC against the best single centred run and their mean. The pilot's recommendation changes only if the same arm clears the rule on at least two of the three segments.

## Code

Local branch `study-combined` = villa `main` 795ca2ba (which now includes khj1222's eager fallback for torch.compile, #1703) + #1890 (rebased onto that main today) + #1891, so every prediction TIFF carries its run record. Pixel output does not depend on #1891 (identical decoded-pixel hashes on w035). `--no-compile` is kept for parity with the pilot.

## What is not claimed

Three segments, all validation cases of the same training run; no held-out scroll yet (the PHerc0841 renders remain the October item). No retraining, no other resolutions.

## Addendum — a label-free rule, written 2026-09-28 11:39 EDT: after w016 and pherc0814-46527 were scored, before pherc1667-w029's arms ran

Post-hoc observation on the two scored segments: the seeds *disagree* on w016 (Pearson r between the seed42 and seed43 centred predictions: 0.80 on the full canvas, 0.65 inside the validation region) and *agree* on 46527 (0.91 / 0.92). Where they disagreed, averaging them cost 0.023 AUC against the better seed; where they agreed, it gained 0.014. r needs no labels, so it could tell a user of an unread segment whether to pick one seed or to average them.

Prediction for pherc1667-w029, fixed before its arms ran: if r(full canvas) ≥ 0.9, the centred seed ensemble will be within the noise band of, or above, the better seed's centred run; if r(full canvas) ≤ 0.8, it will fall below it. Script: `seed_agreement.py`. This is one exploratory rule tested on one new segment, not part of the original preregistration.
