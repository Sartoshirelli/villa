#!/usr/bin/env python3
"""AUC by arm, one panel per segment. usage: fig_arms.py --out fig.png results_a.json [results_b.json ...]"""
import argparse, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["s42_centre", "s42_shift-2", "s42_shift-1", "s42_shift+1", "s42_shift+2", "s42_window_ens5", "s42_tta", "s42_step50k_centre",
         "s43_centre", "s43_shift-2", "s43_shift-1", "s43_shift+1", "s43_shift+2", "s43_window_ens5", "s43_tta", "s43_step50k_centre",
         "seed_ens_centre", "seed_ens_step50k", "seeds_x_windows_ens10", "everything_ens12"]
LAB = {"s42_centre": "seed42 centred", "s42_shift-2": "seed42 shift −2", "s42_shift-1": "seed42 shift −1", "s42_shift+1": "seed42 shift +1",
       "s42_shift+2": "seed42 shift +2", "s42_window_ens5": "seed42 window ens (5)", "s42_tta": "seed42 + TTA", "s42_step50k_centre": "seed42 step-050000",
       "s43_centre": "seed43 centred", "s43_shift-2": "seed43 shift −2", "s43_shift-1": "seed43 shift −1", "s43_shift+1": "seed43 shift +1",
       "s43_shift+2": "seed43 shift +2", "s43_window_ens5": "seed43 window ens (5)", "s43_tta": "seed43 + TTA", "s43_step50k_centre": "seed43 step-050000",
       "seed_ens_centre": "seeds 42+43 centred", "seed_ens_step50k": "seeds 42+43 step-050000", "seeds_x_windows_ens10": "seeds × windows (10)",
       "everything_ens12": "everything (12)"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("results", nargs="+"); a = ap.parse_args()
    segs = [json.load(open(p)) for p in a.results]
    fig, axes = plt.subplots(1, len(segs), figsize=(4.6 * len(segs) + 1.6, 7.2), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], segs):
        arms = d["arms"]; names = [n for n in ORDER if n in arms]
        for i, n in enumerate(names):
            r = arms[n]; lo, hi = r["auc_ci95"]
            col = "#1f77b4" if n.startswith("s42") else "#d62728" if n.startswith("s43") else "#2ca02c"
            ax.errorbar(r["auc"], i, xerr=[[r["auc"] - lo], [hi - r["auc"]]], fmt="o", color=col, capsize=3, ms=5)
        s42, s43 = arms["s42_centre"], arms["s43_centre"]; best = s43 if s43["auc"] >= s42["auc"] else s42
        hw = (best["auc_ci95"][1] - best["auc_ci95"][0]) / 2
        ax.axvspan(best["auc"] - hw, best["auc"] + hw, color="grey", alpha=0.15); ax.axvline(best["auc"], color="grey", ls="--", lw=1)
        ax.set_yticks(range(len(names))); ax.set_yticklabels([LAB.get(n, n) for n in names]); ax.invert_yaxis()
        ax.set_title(f"{d['segment']}\n(validation px {d['validation_px']:,})", fontsize=10)
        ax.set_xlabel("pixel ROC-AUC, 95 % block-bootstrap CI"); ax.grid(axis="x", alpha=0.3)
    fig.suptitle("ink_9um inference-time ensembling — blue seed42, red seed43, green cross-seed; grey band = better seed's centred run ± CI half-width (the preregistered 'helps' threshold)", fontsize=9)
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)

if __name__ == "__main__":
    main()
