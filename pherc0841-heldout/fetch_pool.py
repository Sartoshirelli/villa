#!/usr/bin/env python3
"""Fetch the PHerc0841 label-bucket renders and pool them to 9.362 um, as #1867's
bench/xs_fetch.py::fetch_render does (crop = supervision bbox + 128 px margin snapped
to the 128 grid, even size; CT planes 0..63 mean-pooled 2x2x2 with (sum+4)//8;
labels >=2 of 4 ink; support all 4 supervised). Only the crop's chunks are read
(uncompressed 65x128x128 uint8 chunks, read directly over HTTP); only the pooled
result is stored.

usage: fetch_pool.py OUTROOT short=bucket_dir:ct_stem [...]
"""
from __future__ import annotations
import json, os, shutil, sys, tempfile, threading, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, requests, tifffile, zarr

RES = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
RATE = 8.0                      # resolver quota is 3000 / 300 s; stay at 8/s
_lock = threading.Lock(); _next = [time.time()]
S = requests.Session()


def pace():
    with _lock:
        now = time.time(); t = max(now, _next[0]); _next[0] = t + 1.0 / RATE
    if t > now: time.sleep(t - now)


def get(url, expect=None):
    for k in range(8):
        pace()
        try:
            r = S.get(url, timeout=120)
            if r.status_code == 404: return None
            if r.status_code == 429:
                print("  429, sleeping 130 s", flush=True); time.sleep(130); continue
            r.raise_for_status()
            if expect is not None and len(r.content) != expect: raise IOError(f"short read {len(r.content)} != {expect}")
            return r.content
        except Exception as exc:
            if k == 7: raise
            print(f"  retry {k}: {str(exc)[:100]}", flush=True); time.sleep(10 * 2 ** min(k, 4))


def download(path, dest: Path):
    if dest.exists() and dest.stat().st_size > 0: return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    b = get(RES + urllib.parse.quote(path)); assert b is not None, path
    tmp = dest.with_suffix(dest.suffix + ".partial"); tmp.write_bytes(b); tmp.replace(dest); return dest


def read2d(p):
    a = tifffile.imread(p); return a if a.ndim == 2 else a[..., 0]


def pool2(v):
    d, h, w = v.shape
    v = v[: d - d % 2, : h - h % 2, : w - w % 2].astype(np.uint16)
    s = v.reshape(d // 2, 2, h // 2, 2, w // 2, 2).sum(axis=(1, 3, 5))
    return ((s + 4) // 8).astype(np.uint8)


def fetch(outroot: Path, short: str, bdir: str, stem: str, margin=128):
    dest = outroot / short
    if (dest / "meta.json").exists(): print("have", short); return
    lab_dir = outroot / "_labels_raw" / short
    names = {k: f"{stem}_{k}.tif" for k in ("inklabels", "supervision_mask", "inklabels_v2", "supervision_mask_v2", "validation_mask_v2")}
    for k, n in names.items(): download(f"{bdir}/{n}", lab_dir / n)
    download(f"{bdir}/meta.json", lab_dir / "meta.json")
    zmeta = json.loads(get(RES + f"{bdir}/{stem}.zarr/0/.zarray"))
    D, Hz, Wz = zmeta["shape"]; cz, cy, cx = zmeta["chunks"]
    assert zmeta["compressor"] is None and zmeta["dtype"] == "|u1" and cz == D and zmeta.get("dimension_separator") == ".", zmeta
    L, M = read2d(lab_dir / names["inklabels"]), read2d(lab_dir / names["supervision_mask"])
    H, W = min(Hz, L.shape[0]), min(Wz, L.shape[1])
    ys, xs = np.nonzero(M[:H, :W] > 0)
    y0 = max(0, (ys.min() - margin) // 128 * 128); x0 = max(0, (xs.min() - margin) // 128 * 128)
    y1 = min(H, -(-(ys.max() + 1 + margin) // 128) * 128); x1 = min(W, -(-(xs.max() + 1 + margin) // 128) * 128)
    y1 -= (y1 - y0) % 2; x1 -= (x1 - x0) % 2
    assert y0 % cy == 0 and x0 % cx == 0
    nchunks = (-(-y1 // cy) - y0 // cy) * (-(-x1 // cx) - x0 // cx)
    print(f"{short}: render {zmeta['shape']} labels {L.shape} crop y[{y0},{y1}) x[{x0},{x1}) -> {nchunks} chunks, "
          f"{nchunks * D * cy * cx / 1e9:.2f} GB to read", flush=True)
    pooled = np.empty((D // 2, (y1 - y0) // 2, (x1 - x0) // 2), np.uint8)
    t = time.time(); nbytes = 0; missing = 0
    with ThreadPoolExecutor(24) as ex:
        for ry in range(y0 // cy, -(-y1 // cy)):
            sy0, sy1 = max(y0, ry * cy), min(y1, (ry + 1) * cy)
            cols = list(range(x0 // cx, -(-x1 // cx)))
            blobs = list(ex.map(lambda c: get(RES + f"{bdir}/{stem}.zarr/0/0.{ry}.{c}", expect=D * cy * cx), cols))
            strip = np.zeros((D, cy, len(cols) * cx), np.uint8)
            for i, b in enumerate(blobs):
                if b is None: missing += 1; continue           # absent chunk = fill_value 0
                strip[:, :, i * cx:(i + 1) * cx] = np.frombuffer(b, np.uint8).reshape(D, cy, cx)
                nbytes += len(b)
            sub = strip[:64, sy0 - ry * cy: sy1 - ry * cy, x0 - cols[0] * cx: x1 - cols[0] * cx]   # planes 0..63
            pooled[:, (sy0 - y0) // 2:(sy1 - y0) // 2] = pool2(sub)
            print(f"  {short} rows {sy1 - y0}/{y1 - y0}  {nbytes / 1e9:.2f} GB  {nbytes / 1e6 / (time.time() - t):.1f} MB/s", flush=True)
    dt = time.time() - t

    def pl(A):  # label >= 2 of 4
        return (A[y0:y1, x0:x1] > 0).reshape((y1 - y0) // 2, 2, (x1 - x0) // 2, 2).sum(axis=(1, 3)) >= 2

    def ps(A):  # support: all 4
        return (A[y0:y1, x0:x1] > 0).reshape((y1 - y0) // 2, 2, (x1 - x0) // 2, 2).all(axis=(1, 3))
    tmp = Path(tempfile.mkdtemp(prefix=short + ".", dir=outroot))
    z = zarr.open_array(str(tmp / "ct.zarr"), mode="w", shape=pooled.shape, chunks=(pooled.shape[0], 128, 128), dtype=np.uint8, zarr_format=2)
    z[:] = pooled
    lab, sup = pl(L), ps(M)
    L2, M2, V2 = (read2d(lab_dir / names[k]) for k in ("inklabels_v2", "supervision_mask_v2", "validation_mask_v2"))
    np.save(tmp / "label.npy", lab); np.save(tmp / "support.npy", sup)
    np.save(tmp / "label_v2.npy", pl(L2)); np.save(tmp / "support_v2.npy", ps(M2)); np.save(tmp / "validation_v2.npy", ps(V2))
    meta = dict(short=short, source=RES + bdir, ct=f"{stem}.zarr/0", native_um=4.681, pooled_um=9.362, render_shape=[D, Hz, Wz],
                label_shape=list(L.shape), bbox_native=[int(y0), int(y1), int(x0), int(x1)], ct_shape=list(pooled.shape),
                pooling="planes 0..63, 2x2x2 mean (sum+4)//8; label >=2 of 4 ink; support all 4 supervised (as #1867 xs_fetch.fetch_render)",
                labelled_plane_pooled=16, n_support=int(sup.sum()), n_ink=int((lab & sup).sum()),
                n_support_v2=int(ps(M2).sum()), n_ink_v2=int((pl(L2) & ps(M2)).sum()),
                missing_chunks=missing, bytes_read=nbytes, fetch_s=round(dt, 1), when=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(meta, open(tmp / "meta.json", "w"), indent=1)
    if dest.exists(): shutil.rmtree(dest)
    tmp.rename(dest)
    print("WROTE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[2:]:
        short, rest = spec.split("=", 1); bdir, stem = rest.split(":", 1)
        fetch(out, short, bdir, stem)
    print("DONE", time.strftime("%Y-%m-%dT%H:%M:%S%z"), flush=True)
