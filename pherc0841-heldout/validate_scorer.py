#!/usr/bin/env python3
"""Check score_exact against September's score_w016 results JSON (same TIFFs, same labels)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/mnt/media/vesuvius/study"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_w016 import load_labels
import score_exact as SE
seg, res = sys.argv[1], json.load(open(sys.argv[2]))
y, m, _ = load_labels(Path(f"/mnt/media/vesuvius/study/{seg}/labels/{seg}"), seg)
worst = 0
for name in sys.argv[3:]:
    r0 = res["arms"][name]; t = time.time(); r = SE.score_arm(r0["files"], y, m)
    d = max(abs(r["auc"] - r0["auc"]), abs(r["ap"] - r0["ap"]), abs(r["auc_ci95"][0] - r0["auc_ci95"][0]), abs(r["auc_ci95"][1] - r0["auc_ci95"][1]))
    worst = max(worst, d)
    print(f"{name:22s} auc {r['auc']:.6f}/{r0['auc']:.6f} ap {r['ap']:.6f}/{r0['ap']:.6f} ci {r['auc_ci95'][0]:.6f},{r['auc_ci95'][1]:.6f} / {r0['auc_ci95'][0]:.6f},{r0['auc_ci95'][1]:.6f} blocks {r['blocks']}/{r0['blocks']} maxdiff {d:.2e} ({time.time()-t:.1f}s)")
print("WORST", f"{worst:.2e}", "PASS" if worst <= 1e-6 else "FAIL")
