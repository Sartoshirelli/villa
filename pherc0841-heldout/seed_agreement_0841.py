#!/usr/bin/env python3
"""Q3, label-free and run BEFORE any PHerc0841 AUC: Pearson r between the seed42 and seed43
step-075000 centred predictions (September's seed_agreement.py statistic, full canvas), per
direction, plus r inside the support mask and #1867's separation for each seed and for their
mean (which fixes the seed ensemble's kept direction). Ink labels are never opened.

usage: seed_agreement_0841.py OUT.json SEG=data_dir [...]
"""
import json, sys, time
from pathlib import Path
import numpy as np, tifffile

R_STAR = 0.85   # preregistered primary cut


def sep(p, m):
    v = p[m]; hi, lo = v[v > 0.5], v[v <= 0.5]
    return float(hi.mean() - lo.mean()) if hi.size and lo.size else float("nan")


def main():
    out = Path(sys.argv[1]); res = {"when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "r_star": R_STAR, "segments": {}}
    lines = []
    for spec in sys.argv[2:]:
        seg, d = spec.split("=", 1); d = Path(d)
        m = np.load(d / "support.npy")          # support region only; no ink label is loaded
        row = {}
        for direction in ("forward", "reverse"):
            a = tifffile.imread(d / "preds" / direction / "s42_w0.tif").astype(np.float32)
            b = tifffile.imread(d / "preds" / direction / "s43_w0.tif").astype(np.float32)
            ens = (a + b) / 510.0
            row[direction] = dict(r_full=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), r_support=float(np.corrcoef(a[m], b[m])[0, 1]),
                                  sep_s42=sep(a / 255.0, m), sep_s43=sep(b / 255.0, m), sep_ens=sep(ens, m))
        kept = "forward" if row["forward"]["sep_ens"] >= row["reverse"]["sep_ens"] else "reverse"
        r = row[kept]["r_full"]
        row.update(ens_kept_direction=kept, r_primary=r, prediction_primary=("averaging helps (delta > 0)" if r >= R_STAR else "averaging does not help (delta <= 0)"),
                   prediction_bands=("delta >= -hw" if r >= 0.90 else "delta < 0" if r <= 0.80 else "no prediction"))
        res["segments"][seg] = row
        lines.append(f"{seg:6s} ens kept={kept:7s} r_full(kept)={r:.3f} -> {row['prediction_primary']} | bands: {row['prediction_bands']} | "
                     + " ".join(f"{dd}: r_full={row[dd]['r_full']:.3f} r_sup={row[dd]['r_support']:.3f} sep42={row[dd]['sep_s42']:.3f} sep43={row[dd]['sep_s43']:.3f} sepEns={row[dd]['sep_ens']:.3f}" for dd in ("forward", "reverse")))
    out.write_text(json.dumps(res, indent=1)); txt = out.with_suffix(".txt"); txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
