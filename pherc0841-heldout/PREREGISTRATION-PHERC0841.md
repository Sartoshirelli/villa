# Preregistration: held-out check on PHerc0841 of the ink_9um checkpoint advice

**Written 2026-10-09, before any PHerc0841 CT is downloaded and before any inference. The time and sha256 of this file are recorded in `PREREG.sha256` next to it.**

This extends the September study (`w016-ensembling-pilot/PREREGISTRATION.md` and `PREREGISTRATION-EXTENSION.md` on the evidence branch). That study used three validation segments of ink_9um's own training scrolls. It found that no inference-time ensemble beat the better seed's single centred run beyond the bootstrap half-width. Draft PR #1946 therefore recommends that Tutorial 5 use `hybrid_3d2d-seed43/step-075000`. This study asks whether that advice holds on a scroll the model never trained on.

## What is already public, and so not blind

Issue ScrollPrize/villa#1867 (AndreasHad04) already published AUCs for these exact segments, arrays and checkpoints, in both directions: the "ARM X" scores, `results/xs/armx_scores.json` in AndreasHad04/villa-apple-silicon. I read them before writing this file. Their reverse-direction AUCs are as follows.

| segment | seed42 s75k | seed43 s75k | seed43 s50k | finals averaged |
|---|---|---|---|---|
| w00 | 0.6931 | 0.7055 | 0.7094 | 0.7150 |
| ag144 | 0.6989 | 0.7491 | 0.7285 | 0.7465 |
| ag174 | 0.6743 | 0.6422 | 0.6434 | 0.6628 |

So Q1 and Q2 below are an **independent re-run** on different hardware and code: CUDA fp32 on an RTX 2070 SUPER, the September harness and flags, and the `study-combined` branch. They use the September decision rule, which #1867 did not apply. They are not a blind test. Q3 is new: #1867 published no seed-agreement statistic, and r will be computed and frozen before any of my predictions are scored.

The issue thread also shows that these 4.681 µm label-bucket renders are a harder array than the published 2.403 µm and 9.366 µm surface volumes of the same segments. On those volumes, seed42 s75k scores 0.83 and 0.76 (median). This study uses the renders only, as #1867's main table did. Its conclusions are about the renders, and the report will say so.

## Data, fixed now

The data is the organisers' labelled renders in `huggingface.co/buckets/scrollprize/datasets` (public, read anonymously):

| short | bucket dir | CT array (level 0) |
|---|---|---|
| w00 | `ink/841/w00` | `w00.zarr/0`, 65 × 15872 × 18944 |
| ag144 | `ink/841/auto_grown_20260220144552896` | 65 × 16763 × 18712 |
| ag174 | `ink/841/auto_grown_20260220174252405` | 65 × 14268 × 19413 |

The native spacing is 4.681 µm (`meta.json`: `4.681um_113keV_1.2m_binmean_2_PHerc_0841`). Preprocessing copies #1867's `bench/xs_fetch.py::fetch_render` exactly:

- **Crop:** the bounding box of the supervision mask, plus a 128 px margin, snapped outward to the 128 px chunk grid and trimmed to even size.
- **CT:** planes 0..63 (64 of 65), mean-pooled 2×2×2 with round-half-up (`(sum + 4) // 8`). That gives 32 planes at 9.362 µm. The labelled plane (32 of 65) becomes pooled plane 16.
- **Labels (primary, as #1867):** `<seg>_inklabels.tif`, pooled as ink when at least 2 of 4 pixels are ink.
- **Support (primary, as #1867):** `<seg>_supervision_mask.tif`, pooled as supported only when all 4 pixels are supervised. ink_9um never trained on PHerc0841, so every supported pixel is held out.
- **Secondary label set (sensitivity only; no decision uses it):** `_inklabels_v2.tif` inside `_supervision_mask_v2.tif`, pooled the same way. These files exist in the bucket now; #1867 used the unsuffixed ones.

Only the CT crops and the 2D label TIFFs are downloaded. Pooled data goes under `/mnt/media/vesuvius/data/pherc0841/` (budget 12 GB).

## Inference, fixed now

Everything runs on the September harness: `vesuvius.ink_detection.inference.infer` from worktree `villa-wt-study` (branch `study-combined`, with the fp32 `--amp-dtype default` fix and TIFF provenance). The flags are September's: `--overlap 0.5 --blend-mode hann --batch-size 4 --no-compile --num-workers 4 --amp-dtype default`. #1867 used infer's defaults, which are also overlap 0.5 and Hann.

- **Window:** 17 pooled planes centred on the labelled plane, `--layer-start 8 --layer-end 25`, which is planes 8..24 with plane 16 at index 8. This is identical to #1867's `[6, 27)` after infer's centre-crop to 17. For a shift `s`, the window is `[8+s, 25+s)`.
- **Direction:** every run is done twice, `--direction forward` and `--direction reverse`. infer crops the window first and then reverses it.

**Inference arms per segment** are the September 14, each in both directions, so 28 runs per segment:

| arm | checkpoint | window |
|---|---|---|
| s42_w0 / s43_w0 | seed42 / seed43 step-075000 | centred |
| s42_w±1, s42_w±2 / s43_w±1, s43_w±2 | step-075000 | shifted ±1, ±2 pooled planes |
| s42_w0_tta / s43_w0_tta | step-075000 | centred, `--tta-mirror` |
| s42_step50k_w0 / s43_step50k_w0 | step-050000 | centred |

**Offline ensembles**, the September set, are formed within one direction as the mean of the uint8 TIFFs:

- each seed's 5-window ensemble
- `seed_ens_centre` (s42_w0 + s43_w0)
- `seed_ens_step50k`
- seeds × windows (10)
- everything (12)

## Direction handling, fixed now

**Label-free keep rule (#1867's, primary):** for each segment and each arm, ensembles included, compute the separation inside the support mask, `mean(p | p > 0.5) − mean(p | p ≤ 0.5)`. Keep the direction with the larger separation; a tie goes to forward. No ink label is used. The primary tables use each arm's kept direction.

**Also reported in full:**
- every arm in the **reverse** direction, because #1867 found reverse better on all three segments;
- every arm in the **forward** direction;
- how often the keep rule picked the direction with the higher AUC.

## Metric, fixed now

The metric is September's:

- pixel ROC-AUC and average precision over support pixels, against the pooled ink labels;
- a 64 px block bootstrap with 1000 resamples and a 95 % percentile interval, using September's `score_w016.block_bootstrap` resampling scheme (blocks on the pooled canvas that touch support, resampled with replacement, `default_rng(0)`, resamples with only one class skipped).

To keep 1000 resamples affordable on 4 M-pixel supports, AUC is computed exactly from per-block integer histograms. Predictions are uint8 and ensembles are sums of uint8, so binning loses nothing. Before any PHerc0841 scoring, this implementation is checked against September's `score_w016.py` on saved September TIFFs and must match to 1e-6.

**Half-width (hw):** half the 95 % CI width of the reference arm, which is always the better-scoring arm in a comparison, as in September.

## Questions and decision rules, fixed now

**Q1. Does #1946's pick, seed43 step-075000 centred, hold up held out?**

On each segment, in the kept direction, compute Δ = AUC(s43_w0) − AUC(comparator) for two comparators:
- (a) s42_w0, the other final;
- (b) s43_step50k_w0, #1867's best checkpoint on these renders.

seed43 s75k is "clearly worse" than a comparator on a segment if Δ < −hw, where hw is the comparator's half-width.

- **Stands unqualified:** clearly worse than neither comparator on any of the 3 segments.
- **Stands with a qualification:** clearly worse on exactly 1 of 3 segments. The qualification names the segment and comparator, and says the held-out evidence does not separate the checkpoints.
- **Should change:** clearly worse than the same comparator on 2 or more of 3 segments.

Also reported, with no decision attached: seed42 s50k vs s42 s75k, because #1946 also says to run seed42 at step-050000.

**Q2. Does the two-seed average beat the better single seed?**

On each segment, Δ = AUC(seed_ens_centre) − max(AUC(s42_w0), AUC(s43_w0)), each arm in its own kept direction. Averaging "helps" on a segment if Δ > hw, where hw is the better seed's half-width. This is September's rule.

#1946's "do not average" advice changes only if averaging helps on 2 or more of 3 segments, as in September. The other ensemble arms are reported against the same rule, for description only.

**Q3. Does the label-free seed-agreement r predict whether averaging helps?**

**r** is the Pearson correlation between the s42_w0 and s43_w0 uint8 predictions over the full pooled crop canvas (September's `seed_agreement.py`, full canvas), computed in the seed ensemble's kept direction. r in the other direction and r inside the support mask are recorded as secondary.

r is computed after inference and **before any AUC on PHerc0841 exists**. It is written with its timestamp and sha256 to `seed_agreement_pherc0841.txt`, and Δ(Q2) is filled in only after that.

**Threshold, from the September data:** across the September segments, averaging gained (Δ = +0.014) where r(full) = 0.913. It lost (−0.023) at r = 0.802, and tied (−0.002) at r = 0.801.

- **Primary cut, r\* = 0.85**, the midpoint of 0.80 and 0.91, chosen now. The prediction is Δ > 0 if r ≥ 0.85, and Δ ≤ 0 if r < 0.85.
- **Secondary bands (the September addendum's):**
  - r ≥ 0.90 predicts Δ ≥ −hw (within the noise or above);
  - r ≤ 0.80 predicts Δ < 0;
  - in between, no prediction.

The score is the number of correct sign predictions out of 3. With n = 3 this cannot validate the rule. Three correct out of three is consistent with the rule; any miss is reported as a miss. A September r of about 0.8 to 0.9 gives no basis for a tighter claim.

## Comparison with #1867

For seed42 s75k, seed43 s75k, seed43 s50k and the finals averaged, our reverse-direction AUCs are tabulated beside theirs.

**Criterion:** "reproduced" if |ΔAUC| ≤ 0.01 for each single-checkpoint cell. Larger differences are reported and, where possible, explained. Possible causes are CUDA vs MPS, the villa commit, and their integer `(a + b) // 2` ensemble against our float mean.

The direction the keep rule picks is also compared with theirs.

## What is not claimed

- Three segments of one scroll, all from one render campaign.
- Two seeds; no retraining.
- The 4.681 µm renders only, not the published surface volumes.
- No prediction images are published (data licence); only numbers and figures of numbers.
