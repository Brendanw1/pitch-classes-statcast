"""Scoring with pitcher-bootstrap intervals. Every interval resamples pitchers (the unit that carries correlation:
a pitcher's pitches share a label source, an arsenal and a tracking history) with replacement, 1,000 draws."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

B = 1000


def boot(A, stat, n=B, seed=11):
    """A: per-pitcher additive tallies, shape (pitchers, ...). stat maps the summed tally to a number.
    Returns [point, lo, hi] (2.5 / 97.5 percentiles)."""
    A = np.asarray(A, float)
    rng = np.random.default_rng(seed)
    flat = A.reshape(len(A), -1)
    vals = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(A), len(A)), minlength=len(A))
        vals.append(stat((w @ flat).reshape(A.shape[1:])))
    return [float(stat(A.sum(0))), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def rate(t):
    return 100 * t[0] / max(t[1], 1)


def tallies(pitcher, hit):
    """Per-pitcher (hits, n)."""
    g = pd.DataFrame({"p": pitcher, "h": np.asarray(hit, float)}).groupby("p").h.agg(["sum", "size"])
    return g.to_numpy()


def confusion_by_pitcher(pitcher, y, pred, labels):
    """(pitchers, true, predicted+none) counts; an unlabelled pitch is predicted 'none' and counts as wrong."""
    cols = list(labels) + ["none"]
    yi = pd.Categorical(y, categories=labels).codes
    pi = pd.Categorical(pd.Series(pred, dtype=object).where(lambda s: s.notna(), "none"), categories=cols).codes
    pi = np.where(pi < 0, len(cols) - 1, pi)
    ok = yi >= 0
    pc = pd.factorize(pd.Series(pitcher)[ok])[0]
    C = np.zeros((pc.max() + 1, len(labels), len(cols)))
    np.add.at(C, (pc, yi[ok], pi[ok]), 1)
    return C


def accuracy(C):
    return 100 * np.trace(C[:, :-1]) / max(C.sum(), 1)


def macro_f1(C):
    tp = np.diag(C[:, :-1])
    prec = tp / np.maximum(C[:, :-1].sum(0), 1)
    rec = tp / np.maximum(C.sum(1), 1)
    return float(np.mean(np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)))


def type_scores(C, labels):
    S = C.sum(0)
    tp = np.diag(S[:, :-1])
    return {"recall_pct": dict(zip(labels, (100 * tp / np.maximum(S.sum(1), 1)).round(2).tolist())),
            "precision_pct": dict(zip(labels, (100 * tp / np.maximum(S[:, :-1].sum(0), 1)).round(2).tolist())),
            "row_pct": {a: dict(zip(list(labels) + ["none"], (100 * S[i] / max(S[i].sum(), 1)).round(2).tolist()))
                        for i, a in enumerate(labels)}}


# ---------------------------------------------------------------- near-identical pairs
SCALE = np.array([1.5, 1.5, 1.0])          # hb, ivb inches; ds mph: a radius-1 ball in these units


def pairs(D, cols, keys, cap=400, seed=316):
    """Near-identical pitch pairs within each pitcher-season (hb/1.5, ivb/1.5, ds/1 within radius 1), split by
    same game / same park in another game / another park. Returns per pitcher-season tallies:
    {pair_kind: {label: array (keys, 2) of [mismatches, pairs]}} and the pitcher of each key."""
    kinds = ("same_game", "same_park_other_game", "other_park")
    out = {k: {c: np.zeros((len(keys), 2)) for c in cols} for k in kinds}
    sub = D.set_index(["pitcher", "season"]).loc[keys].reset_index()
    pos = {k: i for i, k in enumerate(keys)}
    for key, g in sub.groupby(["pitcher", "season"], observed=True):
        g = g.sample(min(len(g), cap), random_state=seed)
        pr = np.array(list(cKDTree(g[["hb", "ivb", "ds"]].to_numpy() / SCALE).query_pairs(1.0)))
        if not len(pr):
            continue
        gm, hm = g.game.to_numpy(), g.home.to_numpy()
        same = gm[pr[:, 0]] == gm[pr[:, 1]]
        other = hm[pr[:, 0]] != hm[pr[:, 1]]
        masks = {"same_game": same, "same_park_other_game": ~same & ~other, "other_park": other}
        i = pos[key]
        for c in cols:
            v = g[c].to_numpy()
            ok = pd.notna(v[pr[:, 0]]) & pd.notna(v[pr[:, 1]])
            mis = v[pr[:, 0]] != v[pr[:, 1]]
            for k, m in masks.items():
                out[k][c][i] = [(mis & m & ok).sum(), (m & ok).sum()]
    return out, np.array([k[0] for k in keys])


def by_pitcher(T, pitcher):
    """Collapse per-key tallies to per-pitcher tallies (a pitcher can appear in several seasons)."""
    codes = pd.factorize(pitcher)[0]
    out = np.zeros((codes.max() + 1,) + T.shape[1:])
    np.add.at(out, codes, T)
    return out


def cross_season_pairs(D, cols, y1, y2, cap=400, seed=316):
    """Near-identical pairs from the same pitcher with one pitch in season y1 and one in y2: does the label match?
    A label-free way to ask whether the same pitch keeps its name from one year to the next."""
    a, b = D[D.season == y1], D[D.season == y2]
    both = np.intersect1d(a.pitcher.unique(), b.pitcher.unique())
    T = {c: np.zeros((len(both), 2)) for c in cols}
    flips = {c: {} for c in cols}
    ga, gb = a.groupby("pitcher").indices, b.groupby("pitcher").indices
    for i, p in enumerate(both):
        x = a.iloc[ga[p]]
        z = b.iloc[gb[p]]
        x = x.sample(min(len(x), cap), random_state=seed)
        z = z.sample(min(len(z), cap), random_state=seed)
        hits = cKDTree(x[["hb", "ivb", "ds"]].to_numpy() / SCALE).query_ball_tree(
            cKDTree(z[["hb", "ivb", "ds"]].to_numpy() / SCALE), 1.0)
        ia = np.repeat(np.arange(len(x)), [len(h) for h in hits])
        ib = np.concatenate([np.asarray(h, int) for h in hits]) if len(ia) else np.zeros(0, int)
        for c in cols:
            u, v = x[c].to_numpy()[ia], z[c].to_numpy()[ib]
            ok = pd.notna(u) & pd.notna(v)
            T[c][i] = [(ok & (u != v)).sum(), ok.sum()]
            for a_, b_ in zip(u[ok & (u != v)], v[ok & (u != v)]):
                k = "/".join(sorted((str(a_), str(b_))))
                flips[c][k] = flips[c].get(k, 0) + 1
    return T, both, flips


def arsenal(df, col, min_n=10, min_share=0.05):
    """Set of labels with 10+ pitches and 5%+ of the pitcher-season."""
    def s(g):
        vc = g[col].dropna().value_counts()
        return frozenset(vc[(vc >= min_n) & (vc / len(g) >= min_share)].index)
    return df.groupby(["pitcher", "season"])[[col]].apply(s)


# ---------------------------------------------------------------- selective labelling
def coverage_curve(pitcher, y, pred, conf, levels=(1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5), labels=None, n=B):
    """Abstain on the lowest-confidence pitches. Unlabelled pitches have conf -inf, so they go first; at 100%
    coverage they count as wrong. Accuracy is over the pitches kept. Intervals resample pitchers at a fixed
    threshold."""
    conf = np.where(pd.notna(pred), np.nan_to_num(np.asarray(conf, float), nan=-np.inf), -np.inf)
    hit = (np.asarray(y, object) == np.asarray(pred, object))
    o = np.argsort(-conf, kind="stable")
    out = []
    for c in levels:
        k = int(round(c * len(y)))
        keep = np.zeros(len(y), bool)
        keep[o[:k]] = True
        T = np.stack([tallies(pitcher, hit & keep)[:, 0], tallies(pitcher, keep)[:, 0]], 1)
        row = {"coverage": c, "threshold": float(conf[o[k - 1]]) if k else None,
               "accuracy_pct": [round(x, 2) for x in boot(T, rate, n)]}
        if labels is not None and c < 1:
            ab = ~keep
            row["abstained_share_by_type_pct"] = {t: round(float(100 * (ab & (y == t)).sum() / max(ab.sum(), 1)), 1)
                                                  for t in labels}
            row["abstain_rate_by_type_pct"] = {t: round(float(100 * (ab & (y == t)).sum() / max((y == t).sum(), 1)), 1)
                                               for t in labels}
        out.append(row)
    return out
