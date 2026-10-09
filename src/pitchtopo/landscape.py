"""Level 1: the league movement landscape, its persistent peaks, and the watershed classes.

Following Maloof, Haumacher & Bonicki (2026): a smoothed 3-D histogram of (hb, ivb, ds), peaks found by 0-dim
superlevel-set persistence, basins drawn by watershed from the persistent peaks. Two rules are added here:
  * island rule: a peak whose basin holds under 0.5% of pitches is not a class (on MLB data one pitcher's submarine
    sinker forms its own disconnected component, so its persistence is infinite);
  * fastball-speed merge: an extra peak within 1.5 mph of the fastball peak's ds is a fastball sub-type (a sinker
    peak), which is Level 2's job, so its basin joins the fastball class.
"""
from dataclasses import dataclass, field

import numba
import numpy as np
from scipy.ndimage import gaussian_filter

LO = np.array([-25.0, -25.0, -25.0])
HI = np.array([25.0, 25.0, 5.0])
NB = 50
STEP = (HI - LO) / NB
CLASSES = ["unassigned", "FB", "BB", "CH"]


def offsets(ndim):
    """All 3^d - 1 neighbour offsets (26 in 3-D, 8 in 2-D)."""
    g = np.array(np.meshgrid(*[[-1, 0, 1]] * ndim, indexing="ij")).reshape(ndim, -1).T
    return g[np.abs(g).sum(1) > 0].astype(np.int64)


@numba.njit(cache=True)
def _neighbour(i, k, shape, offs):
    nd = shape.shape[0]
    coord = np.empty(nd, np.int64)
    r = i
    for d in range(nd - 1, -1, -1):
        coord[d] = r % shape[d]
        r //= shape[d]
    j = 0
    for d in range(nd):
        c = coord[d] + offs[k, d]
        if c < 0 or c >= shape[d]:
            return -1
        j = j * shape[d] + c
    return j


@numba.njit(cache=True)
def _root(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


@numba.njit(cache=True)
def persistence(f, shape, offs, floor):
    """0-dim persistence of superlevel sets (elder rule) on a grid. f is flattened.
    Returns peak voxels and their persistence (birth - death); the global maximum of each component is inf."""
    order = np.argsort(-f)
    parent = np.full(f.size, -1, np.int64)
    death = np.full(f.size, np.nan)
    is_peak = np.zeros(f.size, np.bool_)
    for i in order:
        if f[i] < floor:
            break
        cur = -1
        for k in range(offs.shape[0]):
            j = _neighbour(i, k, shape, offs)
            if j < 0 or parent[j] < 0:
                continue
            rj = _root(parent, j)
            if cur < 0:
                cur = rj
            elif rj != cur:
                young, old = (rj, cur) if f[rj] <= f[cur] else (cur, rj)
                death[young] = f[i]          # the younger (lower) peak dies where the two components meet
                parent[young] = old
                cur = old
        if cur < 0:
            parent[i] = i
            is_peak[i] = True
        else:
            parent[i] = cur
    peaks = np.flatnonzero(is_peak)
    pers = np.empty(peaks.size)
    for t in range(peaks.size):
        p = peaks[t]
        pers[t] = np.inf if np.isnan(death[p]) else f[p] - death[p]
    return peaks, pers


@numba.njit(cache=True)
def watershed(f, shape, offs, seeds, floor):
    """Seeded watershed: from high density to low, a voxel above `floor` takes the label of its densest labelled
    neighbour. Repeated sweeps pick up voxels that were cut off on the first pass. Label 0 = unassigned."""
    lab = np.zeros(f.size, np.int64)
    for t in range(seeds.size):
        lab[seeds[t]] = t + 1
    order = np.argsort(-f)
    changed = True
    while changed:
        changed = False
        for i in order:
            if f[i] < floor:
                break
            if lab[i] > 0:
                continue
            best, bl = -1.0, 0
            for k in range(offs.shape[0]):
                j = _neighbour(i, k, shape, offs)
                if j >= 0 and lab[j] > 0 and f[j] > best:
                    best, bl = f[j], lab[j]
            if bl > 0:
                lab[i] = bl
                changed = True
    return lab


def density(X, weights=None):
    H, _ = np.histogramdd(X, bins=NB, range=list(zip(LO, HI)), weights=weights)
    return gaussian_filter(H, 1.0)


def voxel(X):
    v = np.floor((np.asarray(X, float) - LO) / STEP).astype(np.int64)
    ok = ((v >= 0) & (v < NB)).all(axis=1)
    return np.where(ok, np.ravel_multi_index(np.clip(v, 0, NB - 1).T, (NB,) * 3), -1)


def center(flat):
    idx = np.array(np.unravel_index(np.atleast_1d(flat), (NB,) * 3)).T
    return LO + (idx + 0.5) * STEP


def name_peaks(xyz, merge_ds=1.5):
    """Article rule plus the fastball-speed merge. The peak with ds nearest 0 is FB, and any other peak within
    `merge_ds` mph of it joins FB (returned in `merged`). Of the rest, the most glove-side (min hb) is BB and the next
    is CH. Further peaks get codes 4, 5, ..."""
    idx = list(range(len(xyz)))
    fb = max(idx, key=lambda i: xyz[i][2])
    merged = [i for i in idx if i != fb and abs(xyz[i][2] - xyz[fb][2]) <= merge_ds]
    names = {i: 1 for i in [fb] + merged}
    rest = [i for i in idx if i not in names]
    if rest:
        bb = min(rest, key=lambda i: xyz[i][0])
        names[bb] = 2
        for t, i in enumerate(i for i in rest if i != bb):
            names[i] = 3 + t
    return names, merged


@dataclass
class Landscape:
    labels: np.ndarray                  # (NB, NB, NB) int8 class code per voxel, 0 = unassigned
    density: np.ndarray
    peaks: dict = field(default_factory=dict)
    barcode: list = field(default_factory=list)
    n_published_rule: int = 0
    islands: list = field(default_factory=list)
    merged: list = field(default_factory=list)

    def classify(self, X):
        v = voxel(X)
        return np.where(v >= 0, self.labels.ravel()[np.maximum(v, 0)], 0).astype(np.int8)

    def names(self, X):
        """Class names per pitch; an extra peak that survives both rules shows up as E4, E5, ..."""
        return np.array(CLASSES + [f"E{c}" for c in range(4, 10)], object)[self.classify(X)]


def build(X, weights=None, persist_frac=0.05, floor_frac=0.001, island_frac=0.005, merge_ds=1.5):
    f = density(X, weights)
    flat, shape, offs = f.ravel(), np.array(f.shape), offsets(3)
    floor = floor_frac * f.max()
    pk, pers = persistence(flat, shape, offs, floor)
    o = np.argsort(-pers)
    pk, pers = pk[o], pers[o]
    n = int((pers >= persist_frac * f.max()).sum())          # published rule: persistence >= 5% of the max
    lab = watershed(flat, shape, offs, pk[:n], floor)
    v = voxel(X)
    w = np.ones(len(X)) if weights is None else np.asarray(weights, float)
    basin = np.where(v >= 0, lab[np.maximum(v, 0)], 0)
    mass = np.bincount(basin, weights=w, minlength=n + 1)[1:] / w.sum()
    keep = mass >= island_frac
    xyz_all = center(pk[:n])
    islands = [{"xyz": xyz_all[i].round(2).tolist(), "pct_pitches": round(float(mass[i] * 100), 3)}
               for i in range(n) if not keep[i]]
    seeds = pk[:n][keep]
    lab = watershed(flat, shape, offs, seeds, floor)
    xyz = center(seeds)
    names, merged = name_peaks(xyz, merge_ds)
    remap = np.zeros(len(seeds) + 1, np.int8)
    for i, c in names.items():
        remap[i + 1] = c
    peaks = {}
    for i, c in names.items():
        key = CLASSES[c] if c <= 3 else f"E{c}"
        peaks[key + ("_merged" if i in merged else "")] = xyz[i].round(2).tolist()
    return Landscape(labels=remap[lab].reshape(f.shape), density=f, peaks=peaks,
                     barcode=[round(float(p / f.max()), 4) if np.isfinite(p) else "inf" for p in pers[:8]],
                     n_published_rule=n, islands=islands, merged=[xyz[i].round(2).tolist() for i in merged])
