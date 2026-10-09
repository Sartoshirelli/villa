# ink_9um held out on PHerc0841: does the September checkpoint and ensembling advice transfer?

**Preregistration:** `PREREGISTRATION-PHERC0841.md`, sha256 `a8b181ef35a28c5d3587a51d5d1ce72ef4371c61ffd81a711709d1c796f131e3`. Recorded 2026-10-09 13:08:06 EDT (`PREREG.sha256`), before any CT download or inference.

**Seed agreement r:** computed without labels and frozen at 14:19:34 EDT (`seed_agreement_pherc0841.sha256`). This was after inference and before any PHerc0841 AUC existed. Scoring ran at 14:19:38.

Full tables are in `tables_pherc0841.md`. Figure: `fig_auc_by_arm_pherc0841.png`.

## Short answer

1. **Q1, seed43 step-075000.** By the preregistered rule, #1946's pick **stands**. It is not clearly worse than seed42 step-075000 or seed43 step-050000 on any of the three segments.
   - It is clearly better than seed42 on ag144 (+0.050 against a half-width of 0.036).
   - On ag174 it is lower than seed42 (−0.032, inside the 0.038 half-width).
   - Held out, the large in-distribution margin of seed43 over seed42 does not carry over. The ranking is 2 of 3, and the intervals overlap.
   - **The other half of #1946's checkpoint advice does not transfer:** "run seed42 at step-050000". Held out, seed42 step-050000 is *worse* than step-075000 on all three segments (−0.045, −0.004, −0.004). The w00 drop exceeds the 0.025 half-width.
2. **Q2, two-seed average.** It beats the better single seed on **0 of 3** segments (Δ +0.010, −0.003, −0.011; half-widths 0.024 to 0.038). No ensemble arm of the eight clears the rule on any segment. The "do not average" advice stands.
3. **Q3, seed agreement.** The label-free r is 0.63, 0.67 and 0.62 (full canvas, reverse direction). All three are below both the preregistered cut r\* = 0.85 and the 0.80 band, so the rule predicted "averaging does not help" on all three. That is correct on 2 of 3 (ag144, ag174). w00 is a miss: Δ = +0.0095, positive but within the noise. With n = 3 and every r on one side of the threshold, this neither validates nor rejects the rule.
4. **Reproduction of #1867.** Their reverse-direction AUCs are reproduced exactly: |Δ| ≤ 0.0001 on all 15 cells (5 arms × 3 segments). This holds despite different hardware (CUDA fp32 RTX 2070 SUPER vs MPS), a different villa commit and a different scorer. Support and ink pixel counts match theirs exactly.
5. **Direction.** The label-free rule kept *reverse* for every one of 60 (segment, arm) pairs, and reverse had the higher AUC in all 60. Forward (stored order) gives an AUC of 0.50 to 0.61; reverse beats it by 0.09 to 0.20.

## Data

| segment | bucket dir | crop (native y0,y1,x0,x1) | pooled canvas | support px | ink px (rate) |
|---|---|---|---|---|---|
| w00 | `ink/841/w00` | 2304, 13568, 9984, 16512 | 32 × 5632 × 3264 | 4,278,448 | 1,368,784 (0.320) |
| ag144 | `ink/841/auto_grown_20260220144552896` | 2176, 5120, 8576, 13824 | 32 × 1472 × 2624 | 1,918,836 | 638,254 (0.333) |
| ag174 | `ink/841/auto_grown_20260220174252405` | 4480, 7552, 7808, 13184 | 32 × 1536 × 2688 | 1,946,014 | 630,244 (0.324) |

- **Source:** `huggingface.co/buckets/scrollprize/datasets`, public and read anonymously, with no terms to accept.
- **Arrays:** the level-0 4.681 µm renders, uncompressed 65 × 128 × 128 uint8 chunks.
- **Read:** only the chunks under each crop, 6.79 GB in total (w00 4.78, ag144 1.00, ag174 1.01). Throughput was about 2.5 MB/s through PRIDE's VPN.
- **Stored:** 1.2 GB under `/mnt/media/vesuvius/data/pherc0841/` (pooled `ct.zarr`, label and support `.npy`, 84 prediction TIFFs), plus the five 2D label/mask TIFFs per segment in `_labels_raw/`.
- **Pooling:** exactly as in #1867's `xs_fetch.fetch_render`: planes 0..63 mean-pooled 2×2×2 with `(sum+4)//8`, giving 9.362 µm. Labels count as ink when at least 2 of 4 pixels are ink; support requires all 4 pixels to be supervised. The labelled plane becomes pooled plane 16.

## Runs

- **Code:** `vesuvius.ink_detection.inference.infer` on `villa-wt-study` (`study-combined` feb51b458, clean, with the fp32 fix and TIFF provenance).
- **Flags (September's):** `--overlap 0.5 --blend-mode hann --batch-size 4 --no-compile --num-workers 4 --amp-dtype default`.
- **Window:** `--layer-start 8 --layer-end 25`, 17 planes centred on pooled plane 16. Shifts move it by ±1 or ±2 planes. Both `--direction` settings were run.
- **Arms:** the 14 September arms × 2 directions × 3 segments = 84 runs.
- **GPU wall time:** w00 1398 s (about 41 s per run, 100 s with TTA); ag144 507 s; ag174 520 s; about 40 min in total.
- **Scoring:** `score_exact.py`, September's AUC, AP and 64 px block bootstrap computed exactly from integer histograms. It was checked against September's `score_w016.py` on saved pherc0814-46527 and w016 TIFFs: maximum difference 2e-16 (`logs/validate_scorer.log`). Scoring took 37 s.

## Q1: seed43 step-075000 vs (a) seed42 step-075000 and (b) seed43 step-050000

Kept direction is reverse throughout. hw is the comparator's CI half-width.

| segment | seed43 s75k | (a) seed42 s75k | Δa | hw(a) | (b) seed43 s50k | Δb | hw(b) |
|---|---|---|---|---|---|---|---|
| w00 | 0.7055 [0.680, 0.728] | 0.6931 [0.667, 0.717] | +0.0124 | 0.025 | 0.7094 [0.685, 0.732] | −0.0039 | 0.023 |
| ag144 | 0.7491 [0.715, 0.781] | 0.6989 [0.664, 0.736] | **+0.0502** | 0.036 | 0.7285 [0.693, 0.763] | +0.0206 | 0.035 |
| ag174 | 0.6422 [0.605, 0.675] | 0.6743 [0.634, 0.711] | −0.0321 | 0.038 | 0.6434 [0.605, 0.677] | −0.0012 | 0.036 |

**Verdict:** clearly worse than either comparator on 0 of 3 segments, so the pick **stands unqualified** under the preregistered rule. The answer is the same in the all-reverse view (identical, since every arm kept reverse) and in the all-forward view.

**Secondary: seed42 step-050000 minus seed42 step-075000.**

| | w00 | ag144 | ag174 |
|---|---|---|---|
| held out (this study) | **−0.045** | −0.004 | −0.004 |
| in distribution (September) | +0.037 | +0.020 | +0.020 |

The step-050000 advice for seed42 is a property of the training scrolls.

## Q2: two-seed average (step-075000, centred) vs the better single seed

| segment | better seed | its AUC | average | Δ | hw | helps? |
|---|---|---|---|---|---|---|
| w00 | 43 | 0.7055 | 0.7150 | +0.0095 | 0.024 | no |
| ag144 | 43 | 0.7491 | 0.7465 | −0.0026 | 0.033 | no |
| ag174 | 42 | 0.6743 | 0.6629 | −0.0114 | 0.038 | no |

Every ensemble arm, as Δ against the better seed's centred run:

| arm | w00 | ag144 | ag174 | mean | cleared |
|---|---|---|---|---|---|
| seed42 window ens (5) | −0.002 | −0.036 | +0.020 | −0.006 | 0/3 |
| seed43 window ens (5) | +0.006 | +0.015 | −0.017 | +0.001 | 0/3 |
| seed42 + TTA | +0.010 | +0.020 | −0.005 | +0.008 | 0/3 |
| seed43 + TTA | +0.009 | +0.014 | +0.005 | +0.009 | 0/3 |
| seeds 42+43 centred | +0.010 | −0.003 | −0.011 | −0.002 | 0/3 |
| seeds 42+43 step-050000 | −0.010 | −0.017 | −0.014 | −0.014 | 0/3 |
| seeds × windows (10) | +0.017 | +0.010 | +0.009 | +0.012 | 0/3 |
| everything (12) | +0.019 | +0.016 | +0.009 | +0.015 | 0/3 |

**Exploratory, not preregistered:** the large ensembles (10 and 12 members) and seed43 + TTA are positive on all three held-out segments. Their gains are +0.009 to +0.019, about half the half-width. In September these arms were mixed in sign. This hints at a small, consistent held-out gain that three segments cannot resolve. It is not grounds to change the advice.

**On #1867's statement.** Their claim was that averaging the finals "beat single seeds on w00 and ag144 but not ag174". The average beats seed42 step-075000 and seed43 step-050000 on w00 and ag144, which are the columns in their table. It does not beat seed43 step-075000 on ag144 (0.7465 vs 0.7491), and it beats it on w00 only within noise. Both readings are correct. September's rule is the stricter one.

## Q3: seed agreement

r is Pearson's correlation between the s42 and s43 centred maps, frozen before scoring.

| segment | r full canvas (reverse) | r inside support | prediction (r\* = 0.85; band ≤ 0.80) | observed Δ | correct? |
|---|---|---|---|---|---|
| w00 | 0.628 | 0.719 | Δ ≤ 0 | +0.0095 | **no** (small, within noise) |
| ag144 | 0.667 | 0.708 | Δ ≤ 0 | −0.0026 | yes |
| ag174 | 0.619 | 0.719 | Δ ≤ 0 | −0.0114 | yes |

- In the forward direction, r is lower still: 0.40 to 0.42.
- The seeds agree much less on these held-out renders than on the in-distribution segments, where r was 0.80 to 0.91. That fits the model being off-distribution.
- Scored 2 of 3. No segment fell above the threshold, so the rule's positive branch ("averaging helps when r ≥ 0.85") was not tested at all.

## Comparison with #1867 (reverse direction)

| segment | seed42 s75k | seed43 s75k | seed42 s50k | seed43 s50k | finals averaged |
|---|---|---|---|---|---|
| w00 (#1867 / this study) | 0.6931 / 0.6931 | 0.7055 / 0.7055 | 0.6479 / 0.6479 | 0.7094 / 0.7094 | 0.7150 / 0.7150 |
| ag144 | 0.6989 / 0.6989 | 0.7491 / 0.7491 | 0.6946 / 0.6946 | 0.7285 / 0.7285 | 0.7465 / 0.7465 |
| ag174 | 0.6743 / 0.6743 | 0.6422 / 0.6422 | 0.6699 / 0.6699 | 0.6434 / 0.6434 | 0.6628 / 0.6629 |

- **Reproduced:** all 15 cells are within 0.0001. The forward-direction values also agree; the maximum difference over both directions is 5e-5.
- **The one 0.0001 difference** is their integer `(a+b)//2` average against our float mean.
- **Direction choice:** identical; the rule picks reverse everywhere, as in #1867.

## Sensitivity: v2 labels and masks

This uses `_inklabels_v2` inside `_supervision_mask_v2`; no decision depends on it. The v2 support is larger on w00 (6.06 M px) and ag174 (2.86 M px), and identical on ag144.

| segment | seed42 s75k | seed43 s75k | seed42 s50k | seed43 s50k | average |
|---|---|---|---|---|---|
| w00 | 0.7065 | 0.7240 | 0.6627 | 0.7295 | 0.7326 |
| ag144 | 0.7181 | 0.7578 | 0.7055 | 0.7365 | 0.7594 |
| ag174 | 0.6929 | 0.6557 | 0.6823 | 0.6591 | 0.6796 |

The orderings are the same as with v1.

## What this means for #1946

**seed43 step-075000 can stay the tutorial's example checkpoint.** On held-out PHerc0841 renders it is never clearly beaten. It is clearly better than seed42 on one segment, and it is as good as #1867's best checkpoint (seed43 step-050000).

Suggested qualifications for the text:

1. Held out, the two final seeds are close: seed43 leads on 2 of 3 segments and trails within noise on the third. The in-distribution gap of 0.06 to 0.16 should not be read as a general seed effect.
2. Drop or narrow "run seed42 at step-050000". It reverses held out, by −0.045 on w00.
3. "Do not expect averaging to help" holds held out (0 of 3).
4. **Check the depth direction per array.** On these renders, the wrong direction costs 0.09 to 0.20 AUC across all 20 arms. #1867's separation rule picks the right direction every time here.

## Limits

- **Scope:** three segments of one scroll, all from the same 4.681 µm render campaign. #1867's thread shows these renders are harder than the published 2.403 µm and 9.366 µm surface volumes of the same segments. On those volumes, seed42 step-075000 scores about 0.83 and 0.76.
- **Model:** two seeds; no retraining.
- **Wide CIs:** half-widths of 0.023 to 0.038 cannot separate checkpoints that differ by 0.01 to 0.03.
- **Not blind:** Q1 and Q2 re-run on numbers #1867 had already published, and I read them before writing the preregistration (stated there).
- **Deviations from plan:**
  - The fetch was restarted once, at 13:10, from 8 to 24 HTTP threads. This does not affect the data.
  - `analyze_0841.py` got path overrides and a fix to its Q1 verdict counting (segments, not segment × comparator pairs, as the prereg states). Both were made during a synthetic-data dry run, before any real scoring.
- **Licence:** no prediction images are included; the figure shows AUC values only.

## Files

| what | files |
|---|---|
| scripts | `fetch_pool.py`, `run_pherc0841.sh`, `drive_inference.sh`, `seed_agreement_0841.py`, `score_exact.py`, `validate_scorer.py` (imports `/mnt/media/vesuvius/study/score_w016.py`), `analyze_0841.py`, `fig_0841.py` |
| results | `results_pherc0841_all.json`; `results_<seg>_{kept,reverse,forward}.json` in September's format; `tables_pherc0841.md` |
| reference | `reference_1867_armx_scores.json`, a copy of #1867's published scores |
| logs | `logs/`: fetch, inference drivers, the 84 per-run infer logs under `logs/runs/`, scorer validation, analysis; plus `walltimes_<seg>.txt` and `data_meta_<seg>.json` |
| wall time | about 1 h 15 min end to end (13:04 to 14:20): prereg at 13:08; download 13:09–13:55, the bottleneck; inference 13:17–14:19, overlapping the download; scoring 14:20 |
