"""Level 2: how many distinct pitches a pitcher throws inside one Level 1 class.

A pitcher's (hb, ivb) density inside the class is shrunk toward the league density of that class, then its
secondary peaks are tested against a single-Gaussian null: the same number of pitches drawn from one bivariate
normal with the pitcher's mean and covariance. A secondary peak is a separate pitch when its persistence beats the
(1 - p) quantile of the null's largest secondary persistence. Watershed from the surviving peaks gives the groups.
"""
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .landscape import offsets, persistence, watershed

G2 = 50                      # 1-inch bins over hb, ivb in [-25, 25]
OFF2 = offsets(2)


def hist2(P, weights=None):
    H, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=G2, range=[[-25, 25], [-25, 25]], weights=weights)
    return H


def league_density(P):
    """Population density of one class, the shrinkage target."""
    f = gaussian_filter(hist2(P), 1.0)
    return f / f.sum()


def kde(P):
    """Gaussian KDE on the 1-inch grid, Scott bandwidth per axis, at least one bin."""
    h = np.maximum(P.std(axis=0) * len(P) ** (-1 / 6), 1.0)
    f = gaussian_filter(hist2(P), sigma=tuple(h))
    return f / max(f.sum(), 1e-12)


def secondary(f):
    pk, pers = persistence(f.ravel(), np.array(f.shape), OFF2, 0.0)
    fin = np.sort(pers[np.isfinite(pers)])[::-1]
    return pk, pers, (fin[0] if fin.size else 0.0)


def split(P, fpop, rng, lam=50, sims=500, p=0.01):
    """Group labels (0..k-1) for one pitcher-class and k. Fewer than 30 pitches: one group.
    Stops early once enough null draws reach the observed secondary persistence that no peak can pass."""
    n = len(P)
    if n < 30:
        return np.zeros(n, np.int64), 1
    w = n / (n + lam)
    f = w * kde(P) + (1 - w) * fpop
    pk, pers, _ = secondary(f)
    o = np.argsort(-pers)
    sec = pers[o[1:]]
    if sec.size == 0 or sec.max() <= 0:
        return np.zeros(n, np.int64), 1
    top = sec.max()
    need = sims - int(np.floor((sims - 1) * (1 - p)))     # draws >= top that guarantee the quantile >= top
    mu, cov = P.mean(0), np.cov(P.T) + np.eye(2) * 1e-6
    null, hits = [], 0
    for _ in range(sims):
        x = secondary(w * kde(rng.multivariate_normal(mu, cov, n)) + (1 - w) * fpop)[2]
        null.append(x)
        hits += x >= top
        if hits >= need:
            return np.zeros(n, np.int64), 1
    thr = np.quantile(null, 1 - p)
    seeds = [pk[o[0]]] + [pk[i] for i in o[1:] if pers[i] > thr]
    if len(seeds) == 1:
        return np.zeros(n, np.int64), 1
    lab = watershed(f.ravel(), np.array(f.shape), OFF2, np.array(seeds), 0.0).reshape(f.shape)
    v = np.clip(np.floor(P + 25).astype(int), 0, G2 - 1)
    return lab[v[:, 0], v[:, 1]] - 1, len(seeds)


def seed_of(pitcher, season, cls):
    return zlib.crc32(f"{pitcher}|{season}|{cls}".encode())


def _task(a):
    key, P, fpop = a
    return split(P, fpop, np.random.default_rng(seed_of(*key)))[0].astype(np.int8)


def run(D, l1, fpop, mask=None, workers=4):
    """Level 2 group id per pitch within (pitcher, season, class); -1 where unassigned or outside `mask`."""
    gid = np.full(len(D), -1, np.int8)
    ok = (l1 != "unassigned") if mask is None else (l1 != "unassigned") & mask
    pos = np.flatnonzero(ok)
    groups = pd.DataFrame({"p": D.pitcher.to_numpy()[pos], "s": D.season.to_numpy()[pos],
                           "c": l1[pos]}).groupby(["p", "s", "c"]).indices
    keys = list(groups)
    P = D[["hb", "ivb"]].to_numpy(np.float64)
    jobs = ((k, P[pos[groups[k]]], fpop[k[2]]) for k in keys)
    with ProcessPoolExecutor(workers) as ex:
        for k, lab in zip(keys, ex.map(_task, jobs, chunksize=16)):
            gid[pos[groups[k]]] = lab
    return gid
