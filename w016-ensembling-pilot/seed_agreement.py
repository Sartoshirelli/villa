#!/usr/bin/env python3
"""Post-hoc, label-free diagnostic: how much do the two seeds' centred predictions agree?
usage: seed_agreement.py SEG=preds_dir:labels_dir ...   (Pearson r on the full canvas and inside the validation region)"""
import sys, numpy as np, tifffile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_w016 import load_labels
def main():
    for spec in sys.argv[1:]:
        seg, rest = spec.split("=", 1); preds, labels = rest.split(":", 1)
        a = tifffile.imread(f"{preds}/s42_w0.tif").astype(np.float32); b = tifffile.imread(f"{preds}/s43_w0.tif").astype(np.float32)
        y, m, _ = load_labels(Path(labels), seg)
        r_all = np.corrcoef(a.ravel(), b.ravel())[0, 1]; r_val = np.corrcoef(a[m], b[m])[0, 1]
        print(f"{seg:18s} r(full)={r_all:.3f} r(validation)={r_val:.3f} | mean|val 42={a[m].mean():.1f} 43={b[m].mean():.1f} | mean|ink 42={a[y&m].mean():.1f} 43={b[y&m].mean():.1f} | mean|no-ink 42={a[m&~y].mean():.1f} 43={b[m&~y].mean():.1f}")
if __name__ == "__main__": main()
