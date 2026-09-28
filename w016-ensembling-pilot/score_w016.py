#!/usr/bin/env python3
"""Score ink prediction TIFFs for pherc0139-w016 against the ink_9um labels.

AUC / average precision over validation_mask > 0 pixels, inklabels at Z=10,
with a 64x64 block bootstrap (95% percentile interval). Ensembles are means of
the listed TIFFs' uint8 values, so no extra inference is needed.

usage: score_w016.py --labels DIR --out results.json  ARM=file.tif[,file2.tif,...] ...
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import tifffile
import zarr
from sklearn.metrics import average_precision_score, roc_auc_score


def load_labels(labels_dir: Path, seg: str):
    ink = zarr.open(str(labels_dir / f"{seg}_inklabels.zarr"), mode="r")
    val = zarr.open(str(labels_dir / f"{seg}_validation_mask.zarr"), mode="r")
    ink = ink["0"] if hasattr(ink, "keys") and "0" in ink else ink
    val = val["0"] if hasattr(val, "keys") and "0" in val else val
    z_ink = ink.shape[0] // 2 if ink.ndim == 3 else None
    y = np.asarray(ink[z_ink] if z_ink is not None else ink[:]) > 0
    m = np.asarray(val[val.shape[0] // 2] if val.ndim == 3 else val[:]) > 0
    return y, m, {"inklabels_shape": list(ink.shape), "validation_shape": list(val.shape), "z_used": z_ink}


def block_bootstrap(y, s, m, block=64, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    H, W = m.shape
    by, bx = H // block + 1, W // block + 1
    blocks = [(i, j) for i in range(by) for j in range(bx)
              if m[i * block:(i + 1) * block, j * block:(j + 1) * block].any()]
    yy = {b: y[b[0] * block:(b[0] + 1) * block, b[1] * block:(b[1] + 1) * block][m[b[0] * block:(b[0] + 1) * block, b[1] * block:(b[1] + 1) * block]] for b in blocks}
    ss = {b: s[b[0] * block:(b[0] + 1) * block, b[1] * block:(b[1] + 1) * block][m[b[0] * block:(b[0] + 1) * block, b[1] * block:(b[1] + 1) * block]] for b in blocks}
    aucs = []
    for _ in range(n):
        pick = rng.choice(len(blocks), size=len(blocks), replace=True)
        yb = np.concatenate([yy[blocks[k]] for k in pick]); sb = np.concatenate([ss[blocks[k]] for k in pick])
        if yb.min() == yb.max():
            continue
        aucs.append(roc_auc_score(yb, sb))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), len(blocks)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--segment", default="pherc0139-w016")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("arms", nargs="+", help="NAME=a.tif[,b.tif,...] (mean of the files)")
    a = ap.parse_args(argv)
    y, m, meta = load_labels(a.labels, a.segment)
    print("labels:", meta, "| validation px:", int(m.sum()), "| ink px in region:", int((y & m).sum()))
    results = {"segment": a.segment, "labels": meta, "validation_px": int(m.sum()), "arms": {}}
    for arm in a.arms:
        name, files = arm.split("=", 1)
        parts = [tifffile.imread(f).astype(np.float32) for f in files.split(",")]
        for p in parts:
            if p.shape != y.shape:
                raise SystemExit(f"{name}: prediction {p.shape} vs labels {y.shape}: canvas mismatch — check the pooled input matches the labels")
        s = np.mean(parts, axis=0) / 255.0
        t = time.time()
        auc = roc_auc_score(y[m], s[m]); ap_ = average_precision_score(y[m], s[m])
        lo, hi, nb = block_bootstrap(y, s, m, n=a.bootstrap)
        results["arms"][name] = {"files": files.split(","), "auc": auc, "ap": ap_, "auc_ci95": [lo, hi], "blocks": nb}
        print(f"{name:28s} AUC={auc:.4f} [{lo:.4f}, {hi:.4f}]  AP={ap_:.4f}  ({len(parts)} file(s), {time.time()-t:.0f}s)")
    a.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
