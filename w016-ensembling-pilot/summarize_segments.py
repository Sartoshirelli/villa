#!/usr/bin/env python3
"""Summarise the preregistered arms across validation segments (Q1 seed effect, Q2 ensembles, Q3 who benefits).

usage: summarize_segments.py results_a.json [results_b.json ...]
Rule (preregistered): an arm helps on a segment if its AUC beats the better seed's single centred run
by more than that run's bootstrap CI half-width.
"""
import json, sys

ENS = [("s42_window_ens5", "seed42 window ensemble (5)"), ("s43_window_ens5", "seed43 window ensemble (5)"),
       ("s42_tta", "seed42 + TTA mirror"), ("s43_tta", "seed43 + TTA mirror"),
       ("seed_ens_centre", "seed ensemble, centred (42+43)"), ("seed_ens_step50k", "seed ensemble, step-050000"),
       ("seeds_x_windows_ens10", "seeds x windows (10)"), ("everything_ens12", "everything (12)")]
ROWS = [("s42_centre", "seed42 centred"), ("s42_shift-2", "seed42 shift -2"), ("s42_shift-1", "seed42 shift -1"),
        ("s42_shift+1", "seed42 shift +1"), ("s42_shift+2", "seed42 shift +2"), ("s42_window_ens5", "seed42 window ensemble (5)"),
        ("s42_tta", "seed42 + TTA mirror"), ("s42_step50k_centre", "seed42 step-050000 centred"),
        ("s43_centre", "seed43 centred"), ("s43_shift-2", "seed43 shift -2"), ("s43_shift-1", "seed43 shift -1"),
        ("s43_shift+1", "seed43 shift +1"), ("s43_shift+2", "seed43 shift +2"), ("s43_window_ens5", "seed43 window ensemble (5)"),
        ("s43_tta", "seed43 + TTA mirror"), ("s43_step50k_centre", "seed43 step-050000 centred"),
        ("seed_ens_centre", "seed ensemble, centred (42+43)"), ("seed_ens_step50k", "seed ensemble, step-050000"),
        ("seeds_x_windows_ens10", "seeds x windows (10)"), ("everything_ens12", "everything (12: + both TTA runs)")]

def ctx(d):
    a = d["arms"]; s42, s43 = a["s42_centre"]["auc"], a["s43_centre"]["auc"]
    better = "s43_centre" if s43 >= s42 else "s42_centre"
    lo, hi = a[better]["auc_ci95"]
    return {"better": better, "best": a[better]["auc"], "hw": (hi - lo) / 2, "gap": s43 - s42}

def main(paths):
    segs = [json.load(open(p)) for p in paths]
    print("## Per-segment raw results (AUC [95 % CI], AP)\n")
    for d in segs:
        a = d["arms"]; c = ctx(d)
        print(f"### {d['segment']} — validation px {d['validation_px']:,}; better seed {c['better'][1:3]}; rule threshold = {c['best']:.3f} + {c['hw']:.3f}\n")
        print("| arm | AUC | 95 % CI | AP | Δ vs better seed centred |\n|---|---|---|---|---|")
        for k, lab in ROWS:
            if k not in a: continue
            r = a[k]; lo, hi = r["auc_ci95"]; delta = r["auc"] - c["best"]
            flag = " **✓ clears rule**" if delta > c["hw"] else ""
            print(f"| {lab} | {r['auc']:.3f} | {lo:.3f}–{hi:.3f} | {r['ap']:.3f} | {delta:+.3f}{flag} |")
        print()
    print("## Q1 — seed effect (seed43 centred − seed42 centred)\n")
    print("| segment | seed42 | seed43 | gap | half-width | clears? |\n|---|---|---|---|---|---|")
    for d in segs:
        a = d["arms"]; c = ctx(d)
        print(f"| {d['segment']} | {a['s42_centre']['auc']:.3f} | {a['s43_centre']['auc']:.3f} | {c['gap']:+.3f} | {c['hw']:.3f} | {'yes' if abs(c['gap']) > c['hw'] else 'no'} |")
    print("\n## Q2 — ensembles vs the better seed's single centred run (ΔAUC; ✓ = clears the rule)\n")
    print("| arm | " + " | ".join(d["segment"] for d in segs) + " | mean Δ | segments cleared |\n|---|" + "---|" * (len(segs) + 2))
    for k, lab in ENS:
        cells, deltas, n = [], [], 0
        for d in segs:
            a = d["arms"]; c = ctx(d)
            if k not in a: cells.append("—"); continue
            delta = a[k]["auc"] - c["best"]; deltas.append(delta); ok = delta > c["hw"]; n += ok
            cells.append(f"{delta:+.3f}{' ✓' if ok else ''}")
        print(f"| {lab} | " + " | ".join(cells) + f" | {sum(deltas)/len(deltas):+.3f} | {n}/{len(segs)} |")
    print("\n## Q3 — who benefits: best shift / window ensemble / TTA, each vs that seed's own centred run (ΔAUC)\n")
    print("| segment | seed | centred | best shift | window ens (5) | TTA | step-050000 |\n|---|---|---|---|---|---|---|")
    for d in segs:
        a = d["arms"]
        for s in ("42", "43"):
            base = a[f"s{s}_centre"]["auc"]
            shifts = [a[k]["auc"] for k in (f"s{s}_shift-2", f"s{s}_shift-1", f"s{s}_shift+1", f"s{s}_shift+2") if k in a]
            bs = max(shifts) - base if shifts else float("nan")
            we = a[f"s{s}_window_ens5"]["auc"] - base; tta = a[f"s{s}_tta"]["auc"] - base; st = a[f"s{s}_step50k_centre"]["auc"] - base
            print(f"| {d['segment']} | {s} | {base:.3f} | {bs:+.3f} | {we:+.3f} | {tta:+.3f} | {st:+.3f} |")

if __name__ == "__main__":
    main(sys.argv[1:])
