# Evidence for three villa pull requests (September 2026)

Real-data runs on the published w035 (PHerc. 0139) 9.362 µm surface volume with
`scrollprize/ink_9um` `hybrid_3d2d-seed42/step-075000.pth`, on an RTX 2070 SUPER (8 GB),
torch 2.14.0+cu130, tifffile 2026.9.20, zarr 3.2.1. No scroll data is included here:
only tag dumps, pixel-hash comparisons, a downsampled text-free preview, logs and the scripts.

- `w035/screenshot_pr1_tags_before_after.png`, `w035/screenshot_pr1_pixel_comparison.png`,
  `w035/w035_9um_after_scalebar_proof.png` — ink inference: provenance + physical scale in the output TIFF.
- `w035/screenshot_pr3_amp_comparison.png`, `w035/amp_comparison.txt` — `--amp-dtype default` means full precision.
- `w035/run_w035_*.sh` — the exact commands.
