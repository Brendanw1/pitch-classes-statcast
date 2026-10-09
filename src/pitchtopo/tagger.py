"""Individual pitch-type names on top of the topological classes.

A1  pitch-level LightGBM on shape, spin, release and differences from the pitcher's own fastball.
A2  the hybrid: Level 2 groups are the unit. A LightGBM scores each group from its mean shape relative to the
    pitcher's fastball, and the names inside a pitcher-season-class are chosen jointly (maximum total log
    probability) with distinct names while there are enough to go round. Every pitch in a group shares one name.
"""
import itertools

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PITCH_FEATURES = ["speed", "hb", "ivb", "spin", "axis_sin", "axis_cos", "ext", "rel_h", "rel_side", "d_velo", "d_ivb",
                  "d_hb", "d_spin"]
# fixed column order: LightGBM column sampling depends on it, so results are reproducible run to run
GROUP_STATS = ["hb", "ivb", "speed", "spin", "axis_sin", "axis_cos", "ext", "rel_h", "rel_side", "d_velo", "d_ivb",
               "d_hb", "d_spin", "ds"]
GROUP_FEATURES = ["cls_code", "frac"] + GROUP_STATS + ["sd_hb", "sd_ivb", "sd_speed"]
LGB = dict(n_estimators=300, learning_rate=0.07, num_leaves=63, min_child_samples=50, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=6, random_state=7)
LGB_GROUP = {**LGB, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20}
KEY = ["pitcher", "season", "cls", "gid"]


def fit_a1(X, y):
    return LGBMClassifier(**LGB).fit(X, y)


def group_table(D, l1, gid, y=None):
    """One row per Level 2 group: mean (and spread) of the group's pitches, its share of the pitcher's pitches."""
    m = gid >= 0
    G = D.loc[m, ["pitcher", "season"] + GROUP_STATS].copy()
    G["cls"], G["gid"] = l1[m], gid[m]
    if y is not None:
        G["y"] = y[m]
    g = G.groupby(KEY, observed=True)
    T = g[GROUP_STATS].mean()
    T[["sd_hb", "sd_ivb", "sd_speed"]] = g[["hb", "ivb", "speed"]].std().fillna(0).to_numpy()
    T["n"] = g.size()
    if y is not None:
        mode = g.y.agg(lambda s: s.mode().iloc[0] if s.notna().any() else None)
        T["y"] = mode
        T["purity"] = g.y.agg(lambda s: (s == s.mode().iloc[0]).mean() if s.notna().any() else np.nan)
    T = T.reset_index()
    T["frac"] = T.n / T.groupby(["pitcher", "season"]).n.transform("sum")
    T["cls_code"] = T.cls.map({"FB": 0, "BB": 1, "CH": 2})
    return T


def joint_names(logp, distinct=True):
    """Rows = groups, columns = names. The assignment with the largest total log probability; names distinct
    when there are at least as many names as groups. Falls back to per-group argmax for very large arsenals."""
    k, m = logp.shape
    if k > 6 or m ** k > 50000:
        return logp.argmax(1)
    best, score = None, -np.inf
    for combo in itertools.product(range(m), repeat=k):
        if distinct and k <= m and len(set(combo)) < k:
            continue
        s = logp[np.arange(k), combo].sum()
        if s > score + 1e-12:
            best, score = combo, s
    return np.array(best)


def fit_a2(T):
    tr = T[T.y.notna()]
    return LGBMClassifier(**LGB_GROUP).fit(tr[GROUP_FEATURES], tr.y.astype(str), sample_weight=tr.n)


def name_a2(model, T):
    """Names and the model probability of each chosen name, per group."""
    P = model.predict_proba(T[GROUP_FEATURES])
    L = np.log(P + 1e-6)
    names = np.empty(len(T), dtype=object)
    conf = np.empty(len(T))
    for ix in T.reset_index(drop=True).groupby(["pitcher", "season", "cls"], observed=True).indices.values():
        j = joint_names(L[ix])
        names[ix] = model.classes_[j]
        conf[ix] = P[ix, j]
    return names, conf


def to_pitches(D, l1, gid, T, col):
    """Broadcast a group column back to pitches (None where the pitch has no group)."""
    key = pd.MultiIndex.from_arrays([D.pitcher.to_numpy(), D.season.to_numpy(), l1, gid])
    ix = pd.MultiIndex.from_frame(T[KEY]).get_indexer(key)
    v = T[col].to_numpy()
    return np.where(ix >= 0, v[np.maximum(ix, 0)], None)
