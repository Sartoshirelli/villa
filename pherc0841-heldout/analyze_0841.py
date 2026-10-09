#!/usr/bin/env python3
"""Score every PHerc0841 arm in both directions (September metric, exact), apply #1867's
label-free direction rule, and write the preregistered Q1/Q2/Q3 tables + #1867 comparison.
Refuses to run unless the seed-agreement file has been frozen (PREREG order).

usage: analyze_0841.py
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import score_exact as SE

DATA = Path(os.environ.get("P0841_DATA", "/mnt/media/vesuvius/data/pherc0841"))
OUT = Path(os.environ.get("P0841_OUT", str(Path(__file__).resolve().parent)))
SEGS = ["w00", "ag144", "ag174"]
FROZEN = OUT / "seed_agreement_pherc0841.json"; FROZEN_HASH = OUT / "seed_agreement_pherc0841.sha256"
W = lambda s: [f"s{s}_w{k}" for k in (-2, -1, 0, 1, 2)]
ARMS = {"s42_centre": ["s42_w0"], "s43_centre": ["s43_w0"]}
for s in ("42", "43"):
    for k in (-2, -1, 1, 2): ARMS[f"s{s}_shift{k:+d}"] = [f"s{s}_w{k}"]
    ARMS[f"s{s}_tta"] = [f"s{s}_w0_tta"]; ARMS[f"s{s}_step50k_centre"] = [f"s{s}_step50k_w0"]
ARMS.update({"s42_window_ens5": W("42"), "s43_window_ens5": W("43"), "seed_ens_centre": ["s42_w0", "s43_w0"],
             "seed_ens_step50k": ["s42_step50k_w0", "s43_step50k_w0"], "seeds_x_windows_ens10": W("42") + W("43"),
             "everything_ens12": W("42") + W("43") + ["s42_w0_tta", "s43_w0_tta"]})
ENS = ["s42_window_ens5", "s43_window_ens5", "s42_tta", "s43_tta", "seed_ens_centre", "seed_ens_step50k", "seeds_x_windows_ens10", "everything_ens12"]
KEY_V2 = ["s42_centre", "s43_centre", "s42_step50k_centre", "s43_step50k_centre", "seed_ens_centre"]
# #1867 ARM X, reverse direction, from AndreasHad04/villa-apple-silicon results/xs/armx_scores.json (read 2026-10-09)
ARMX_REV = {"w00": {"s42_centre": 0.6931, "s43_centre": 0.7055, "s42_step50k_centre": 0.6479, "s43_step50k_centre": 0.7094, "seed_ens_centre": 0.7150},
            "ag144": {"s42_centre": 0.6989, "s43_centre": 0.7491, "s42_step50k_centre": 0.6946, "s43_step50k_centre": 0.7285, "seed_ens_centre": 0.7465},
            "ag174": {"s42_centre": 0.6743, "s43_centre": 0.6422, "s42_step50k_centre": 0.6699, "s43_step50k_centre": 0.6434, "seed_ens_centre": 0.6628}}
ARMX_NSUP = {"w00": 4278448, "ag144": 1918836, "ag174": 1946014}


def hw(r): return (r["auc_ci95"][1] - r["auc_ci95"][0]) / 2


def main():
    if not (FROZEN.exists() and FROZEN_HASH.exists()):
        raise SystemExit("seed agreement not frozen yet: run seed_agreement_0841.py and record its sha256 first")
    frozen = json.load(open(FROZEN))
    allres = {}
    for seg in SEGS:
        d = DATA / seg; meta = json.load(open(d / "meta.json"))
        y, m = np.load(d / "label.npy"), np.load(d / "support.npy")
        y2, m2 = np.load(d / "label_v2.npy"), np.load(d / "support_v2.npy")
        out = {"segment": seg, "meta": meta, "validation_px": int(m.sum()), "ink_px": int((y & m).sum()), "dirs": {}, "kept": {}, "v2": {}}
        for dr in ("forward", "reverse"):
            out["dirs"][dr] = {}
            for name, files in ARMS.items():
                t = time.time()
                out["dirs"][dr][name] = SE.score_arm([d / "preds" / dr / f"{f}.tif" for f in files], y, m)
                print(f"{seg} {dr:7s} {name:22s} AUC {out['dirs'][dr][name]['auc']:.4f} sep {out['dirs'][dr][name]['sep']:.3f} ({time.time()-t:.0f}s)", flush=True)
        for name in ARMS:
            f, r = out["dirs"]["forward"][name], out["dirs"]["reverse"][name]
            kept = "forward" if f["sep"] >= r["sep"] else "reverse"
            out["kept"][name] = dict(out["dirs"][kept][name], direction=kept,
                                     rule_picked_better=bool(out["dirs"][kept][name]["auc"] >= out["dirs"]["reverse" if kept == "forward" else "forward"][name]["auc"]))
        for name in KEY_V2:
            dr = out["kept"][name]["direction"]
            out["v2"][name] = dict(SE.score_arm([d / "preds" / dr / f"{f}.tif" for f in ARMS[name]], y2, m2, n_boot=1000), direction=dr)
        out["v2_support_px"] = int(m2.sum()); out["v2_ink_px"] = int((y2 & m2).sum())
        allres[seg] = out
        for view in ("kept", "reverse", "forward"):   # September-format files for fig/summarize reuse
            arms = out["kept"] if view == "kept" else out["dirs"][view]
            (OUT / f"results_{seg}_{view}.json").write_text(json.dumps({"segment": f"PHerc0841 {seg} ({view})", "validation_px": out["validation_px"], "arms": arms}, indent=1))
    (OUT / "results_pherc0841_all.json").write_text(json.dumps(allres, indent=1))
    tables(allres, frozen)


def q1_q2(allres, view, L):
    L.append(f"\n### Q1 ({view}): seed43 step-075000 centred vs (a) seed42 step-075000 and (b) seed43 step-050000\n")
    L.append("| segment | seed43 s75k | (a) seed42 s75k | Δa | hw(a) | (b) seed43 s50k | Δb | hw(b) | clearly worse than |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    nworse = {"a": 0, "b": 0}; nseg = 0
    for seg, o in allres.items():
        A = o["kept"] if view == "kept" else o["dirs"][view]
        t, a, b = A["s43_centre"], A["s42_centre"], A["s43_step50k_centre"]
        da, db = t["auc"] - a["auc"], t["auc"] - b["auc"]; wa, wb = da < -hw(a), db < -hw(b)
        nworse["a"] += wa; nworse["b"] += wb; nseg += (wa or wb)
        f = lambda r: f"{r['auc']:.4f} [{r['auc_ci95'][0]:.3f}, {r['auc_ci95'][1]:.3f}]" + (f" ({r['direction'][:3]})" if "direction" in r else "")
        L.append(f"| {seg} | {f(t)} | {f(a)} | {da:+.4f} | {hw(a):.3f} | {f(b)} | {db:+.4f} | {hw(b):.3f} | {', '.join(x for x, w in (('a', wa), ('b', wb)) if w) or 'neither'} |")
    worst = max(nworse.values())
    verdict = ("STANDS UNQUALIFIED" if nseg == 0 else "SHOULD CHANGE" if worst >= 2 else "STANDS WITH A QUALIFICATION" if nseg == 1
               else "NOT COVERED BY THE RULE (clearly worse on 2+ segments, but against different comparators)")
    L.append(f"\nClearly worse than (a) on {nworse['a']}/3, than (b) on {nworse['b']}/3, than either on {nseg}/3 → **{verdict}** (preregistered rule).")
    L.append(f"\nSecondary (#1946 also says to run seed42 at step-050000), seed42 s50k − seed42 s75k: " + ", ".join(
        f"{seg} {((o['kept'] if view == 'kept' else o['dirs'][view])['s42_step50k_centre']['auc'] - (o['kept'] if view == 'kept' else o['dirs'][view])['s42_centre']['auc']):+.4f}" for seg, o in allres.items()))
    L.append(f"\n### Q2 ({view}): two-seed average (step-075000, centred) vs the better single seed\n")
    L.append("| segment | better seed | its AUC | seed average | Δ | hw | helps? |\n|---|---|---|---|---|---|---|")
    nh = 0; deltas = {}
    for seg, o in allres.items():
        A = o["kept"] if view == "kept" else o["dirs"][view]
        best = "s43_centre" if A["s43_centre"]["auc"] >= A["s42_centre"]["auc"] else "s42_centre"
        dlt = A["seed_ens_centre"]["auc"] - A[best]["auc"]; h = dlt > hw(A[best]); nh += h; deltas[seg] = (dlt, hw(A[best]))
        L.append(f"| {seg} | {best[1:3]} | {A[best]['auc']:.4f} | {A['seed_ens_centre']['auc']:.4f} | {dlt:+.4f} | {hw(A[best]):.3f} | {'yes' if h else 'no'} |")
    L.append(f"\nAveraging helps on {nh}/3 → #1946's 'do not average' advice {'CHANGES' if nh >= 2 else 'stands'} (rule: ≥2/3).")
    L.append(f"\nAll ensemble arms, ΔAUC vs the better seed's centred run ({view}; ✓ = clears hw):\n")
    L.append("| arm | " + " | ".join(allres) + " | mean Δ | cleared |\n|---|" + "---|" * (len(allres) + 2))
    for k in ENS:
        cells, ds, n = [], [], 0
        for seg, o in allres.items():
            A = o["kept"] if view == "kept" else o["dirs"][view]
            best = "s43_centre" if A["s43_centre"]["auc"] >= A["s42_centre"]["auc"] else "s42_centre"
            dd = A[k]["auc"] - A[best]["auc"]; ok = dd > hw(A[best]); n += ok; ds.append(dd)
            cells.append(f"{dd:+.4f}{' ✓' if ok else ''}")
        L.append(f"| {k} | " + " | ".join(cells) + f" | {np.mean(ds):+.4f} | {n}/3 |")
    return deltas


def tables(allres, frozen):
    L = ["# PHerc0841 tables (generated by analyze_0841.py " + time.strftime("%Y-%m-%d %H:%M %Z") + ")\n"]
    L.append("## Data\n\n| segment | pooled canvas | support px | ink px (rate) | v2 support px | v2 ink px | #1867 support px |\n|---|---|---|---|---|---|---|")
    for seg, o in allres.items():
        L.append(f"| {seg} | {o['meta']['ct_shape']} | {o['validation_px']:,} | {o['ink_px']:,} ({o['ink_px']/o['validation_px']:.3f}) | {o['v2_support_px']:,} | {o['v2_ink_px']:,} | {ARMX_NSUP[seg]:,} |")
    L.append("\n## Per-arm AUC, both directions (AUC [95 % CI], separation; ← = kept by the label-free rule)\n")
    for seg, o in allres.items():
        L.append(f"\n### {seg}\n\n| arm | forward AUC | fwd sep | reverse AUC | rev sep | kept | AP (kept) | rule picked better AUC? |\n|---|---|---|---|---|---|---|---|")
        for k in ARMS:
            f, r, kp = o["dirs"]["forward"][k], o["dirs"]["reverse"][k], o["kept"][k]
            L.append(f"| {k} | {f['auc']:.4f} [{f['auc_ci95'][0]:.3f}, {f['auc_ci95'][1]:.3f}] | {f['sep']:.3f} | {r['auc']:.4f} [{r['auc_ci95'][0]:.3f}, {r['auc_ci95'][1]:.3f}] | {r['sep']:.3f} | {kp['direction']} | {kp['ap']:.4f} | {'yes' if kp['rule_picked_better'] else '**no**'} |")
    nk = sum(o["kept"][k]["rule_picked_better"] for o in allres.values() for k in ARMS); nt = len(allres) * len(ARMS)
    nrev = sum(o["kept"][k]["direction"] == "reverse" for o in allres.values() for k in ARMS)
    nrevbetter = sum(o["dirs"]["reverse"][k]["auc"] > o["dirs"]["forward"][k]["auc"] for o in allres.values() for k in ARMS)
    L.append(f"\nDirection: the rule kept reverse in {nrev}/{nt} (segment, arm) cases; reverse had the higher AUC in {nrevbetter}/{nt}; the rule picked the higher-AUC direction in {nk}/{nt}.")
    L.append("\n## Preregistered answers — primary view: each arm in its label-free kept direction")
    dk = q1_q2(allres, "kept", L)
    L.append("\n## Same, every arm in the reverse direction (#1867 found reverse better on all three)")
    dr = q1_q2(allres, "reverse", L)
    L.append("\n## Same, every arm in the forward direction (stored order)")
    q1_q2(allres, "forward", L)
    L.append(f"\n## Q3: seed agreement r (frozen before scoring: {frozen['when']}) vs whether averaging helped\n")
    L.append("| segment | ens kept dir | r full (kept) | r support (kept) | primary prediction (r* = 0.85) | Δ (kept) | sign correct? | band prediction | band correct? |\n|---|---|---|---|---|---|---|---|---|")
    nc = 0; nb = 0; nbp = 0
    for seg, row in frozen["segments"].items():
        dlt, h = dk[seg]; kd = row["ens_kept_direction"]; r = row["r_primary"]
        pred_pos = r >= 0.85; ok = (dlt > 0) == pred_pos; nc += ok
        band = row["prediction_bands"]
        if band == "no prediction": bok = "—"
        else:
            nbp += 1; bo = (dlt >= -h) if band == "delta >= -hw" else (dlt < 0); nb += bo; bok = "yes" if bo else "**no**"
        L.append(f"| {seg} | {kd} | {r:.3f} | {row[kd]['r_support']:.3f} | {'Δ > 0' if pred_pos else 'Δ ≤ 0'} | {dlt:+.4f} | {'yes' if ok else '**no**'} | {band} | {bok} |")
    L.append(f"\nPrimary: {nc}/3 sign predictions correct. Bands: {nb}/{nbp} correct where a prediction was made.")
    L.append("\nr in the other direction (label-free, for the record): " + "; ".join(
        f"{seg} fwd r_full {row['forward']['r_full']:.3f} / rev {row['reverse']['r_full']:.3f}" for seg, row in frozen["segments"].items()))
    L.append("\n## Comparison with #1867 (reverse direction; theirs from armx_scores.json)\n")
    L.append("| segment | arm | #1867 | this study | Δ (ours − theirs) | within 0.01? |\n|---|---|---|---|---|---|")
    for seg, o in allres.items():
        for k, v in ARMX_REV[seg].items():
            ours = o["dirs"]["reverse"][k]["auc"]; dd = ours - v
            L.append(f"| {seg} | {k} | {v:.4f} | {ours:.4f} | {dd:+.4f} | {'yes' if abs(dd) <= 0.01 else '**no**'} |")
    L.append("\n## Sensitivity: v2 labels inside the v2 supervision mask (kept direction; no decision uses this)\n")
    L.append("| segment | " + " | ".join(KEY_V2) + " |\n|---|" + "---|" * len(KEY_V2))
    for seg, o in allres.items():
        L.append(f"| {seg} | " + " | ".join(f"{o['v2'][k]['auc']:.4f} [{o['v2'][k]['auc_ci95'][0]:.3f}, {o['v2'][k]['auc_ci95'][1]:.3f}]" for k in KEY_V2) + " |")
    (OUT / "tables_pherc0841.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
