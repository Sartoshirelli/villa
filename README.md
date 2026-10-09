# Evidence for three villa pull requests (September 2026)

Real-data runs on the published w035 (PHerc. 0139) 9.362 µm surface volume with
`scrollprize/ink_9um` `hybrid_3d2d-seed42/step-075000.pth`, on an RTX 2070 SUPER (8 GB),
torch 2.14.0+cu130, tifffile 2026.9.20, zarr 3.2.1. No scroll data is included here:
only tag dumps, pixel-hash comparisons, a downsampled text-free preview, logs and the scripts.

- `w035/screenshot_pr1_tags_before_after.png`, `w035/screenshot_pr1_pixel_comparison.png`,
  `w035/w035_9um_after_scalebar_proof.png` — ink inference: provenance + physical scale in the output TIFF.
- `w035/screenshot_pr3_amp_comparison.png`, `w035/amp_comparison.txt` — `--amp-dtype default` means full precision.
- `w035/run_w035_*.sh` — the exact commands.

- `w016-ensembling-pilot/` — preregistered pilot: inference-time ensembling (depth windows, seeds, TTA, steps) for `ink_9um` on the validation segment pherc0139-w016. Report, figure, scores, logs, scripts. No prediction images (licence).

## Licence

Code, scripts and written material on this branch are released under the MIT License (see `LICENSE`), the same licence as ScrollPrize/villa. Images and scores derived from Vesuvius Challenge scroll data remain subject to the Vesuvius Challenge data licence.

- `pherc0841-heldout/` — October: the same 14 arms × both depth directions on held-out PHerc0841 renders (w00, ag144, ag174; organisers' labels via #1867), preregistered before any download (`PREREGISTRATION-PHERC0841.md`, sha256 a8b181ef…). Start with `RESULTS.md`. Reproduces #1867's reverse-direction AUCs to within 0.0001. No prediction images.
