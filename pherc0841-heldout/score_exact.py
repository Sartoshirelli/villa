#!/usr/bin/env python3
"""September's metric (score_w016.py), computed exactly from integer histograms so a
1000-resample 64 px block bootstrap is affordable on multi-million-pixel supports.

Predictions are uint8 TIFFs; an ensemble's score is the SUM of its members (same ranks
as the mean, so the same AUC/AP), giving integer scores in [0, 255*k]. AUC = P(s+ > s-)
+ 0.5 P(tie), exact. AP = sklearn's average_precision (step sum over distinct
thresholds), exact. The bootstrap replicates score_w016.block_bootstrap: blocks of the
canvas that touch the support, row-major, rng = default_rng(0), rng.choice(nb, nb)
per resample, single-class resamples skipped, 2.5/97.5 percentiles.

Also: separation (#1867's label-free direction statistic) = mean(p|p>0.5) - mean(p|p<=0.5)
inside the support, p = score / (255*k).
"""
from __future__ import annotations
import numpy as np, tifffile

BLOCK = 64


def load_sum(files):
    parts = [tifffile.imread(f) for f in files]
    s = np.zeros(parts[0].shape, np.int32)
    for p in parts:
        if p.shape != s.shape: raise SystemExit(f"shape mismatch {p.shape} vs {s.shape}: {files}")
        s += p.astype(np.int32)
    return s, len(parts)


def block_index(shape, m):
    H, W = shape
    by, bx = H // BLOCK + 1, W // BLOCK + 1
    bid = (np.arange(H)[:, None] // BLOCK) * bx + (np.arange(W)[None, :] // BLOCK)
    present = np.zeros(by * bx, bool); present[np.unique(bid[m])] = True   # row-major order == score_w016's
    remap = -np.ones(by * bx, np.int64); remap[present] = np.arange(present.sum())
    return remap[bid[m]], int(present.sum())


def hists(score, y, m, nbins, blk=None, nb=None):
    v = score[m].astype(np.int64); yy = y[m]
    if blk is None:
        return np.bincount(v[yy], minlength=nbins).astype(np.float64), np.bincount(v[~yy], minlength=nbins).astype(np.float64)
    pos = np.bincount(blk[yy] * nbins + v[yy], minlength=nb * nbins).reshape(nb, nbins).astype(np.float64)
    neg = np.bincount(blk[~yy] * nbins + v[~yy], minlength=nb * nbins).reshape(nb, nbins).astype(np.float64)
    return pos, neg


def auc_h(pos, neg):
    """pos/neg: (..., nbins). Exact Mann-Whitney AUC with half ties."""
    npos, nneg = pos.sum(-1), neg.sum(-1)
    below = np.cumsum(neg, -1) - neg
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pos * (below + 0.5 * neg)).sum(-1) / (npos * nneg)


def ap_h(pos, neg):
    P = pos.sum()
    tp = np.cumsum(pos[::-1]); fp = np.cumsum(neg[::-1])
    keep = (pos[::-1] + neg[::-1]) > 0
    tp, fp = tp[keep], fp[keep]
    prec = tp / (tp + fp); rec = tp / P
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def bootstrap(score, y, m, nbins, n=1000, seed=0):
    blk, nb = block_index(y.shape, m)
    pos, neg = hists(score, y, m, nbins, blk, nb)
    rng = np.random.default_rng(seed)
    picks = np.stack([np.bincount(rng.choice(nb, size=nb, replace=True), minlength=nb) for _ in range(n)]).astype(np.float64)
    P, N = picks @ pos, picks @ neg
    ok = (P.sum(1) > 0) & (N.sum(1) > 0)
    a = auc_h(P[ok], N[ok])
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), nb


def separation(score, k, m):
    p = score[m].astype(np.float64) / (255.0 * k)
    hi, lo = p[p > 0.5], p[p <= 0.5]
    return float(hi.mean() - lo.mean()) if hi.size and lo.size else float("nan")


def score_arm(files, y, m, n_boot=1000):
    s, k = load_sum(files)
    if s.shape != y.shape: raise SystemExit(f"prediction {s.shape} vs labels {y.shape}")
    nbins = 255 * k + 1
    pos, neg = hists(s, y, m, nbins)
    lo, hi, nb = bootstrap(s, y, m, nbins, n=n_boot)
    return dict(files=[str(f) for f in files], auc=float(auc_h(pos, neg)), ap=ap_h(pos, neg), auc_ci95=[lo, hi], blocks=nb,
                sep=separation(s, k, m), k=k)
