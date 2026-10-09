"""Every number in README.md and ARTICLE.md. Aggregate output only: results/results.json, results/headline.csv,
figures/*.png. Nothing pitch-level or player-level is written.

Usage (from the repo root):
    nice -n 10 python scripts/run_all.py [--data data] [--workers 6] [--skip-phase-c] [--skip-2021]

Input: data/statcast_{2023,2024}.parquet and data/statcast_2025.parquet or data/gamefeed_2025.parquet
(see scripts/fetch_statcast.py). Optional data/statcast_{2021,2022}.parquet for the older-seasons check.
Runtime on a 2023 laptop with 6 workers: about 15 minutes. Peak memory about 4 GB.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pitchtopo import data, evaluate as ev, figures, landscape, level2, tagger  # noqa: E402
from pitchtopo.data import TYPES  # noqa: E402

R = {}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def r2(x):
    return [round(v, 2) for v in x]


def r4(x):
    return [round(v, 4) for v in x]


# ------------------------------------------------------------------------------------------------ Level 1
def level1(D, n_rebuild=100):
    X = D[["hb", "ivb", "ds"]].to_numpy(np.float64)
    L = landscape.build(X)
    D["l1"] = L.names(X)
    scored = D.cls_sc.isin(["FB", "BB", "CH"]).to_numpy()
    hit = (D.l1 == D.cls_sc).to_numpy()
    T = ev.tallies(D.pitcher[scored], hit[scored])
    per_type = (pd.crosstab(D.pitch_type, D.l1, normalize="index") * 100).round(2)
    per_type["n"] = D.pitch_type.value_counts()
    out = {"peaks": L.peaks, "barcode_frac_of_max": L.barcode, "published_rule_peaks": L.n_published_rule,
           "islands": L.islands, "merged_fastball_speed_peaks": L.merged,
           "class_share_pct": D.l1.value_counts(normalize=True).mul(100).round(2).to_dict(),
           "n_scored_excl_FC": int(scored.sum()),
           "agreement_pct_fixed_landscape": r2(ev.boot(T, ev.rate)),
           "per_statcast_type_pct": per_type[per_type.n >= 1000].to_dict("index")}
    # Rebuild the landscape on each pitcher resample, so the interval includes the landscape's own uncertainty.
    pc, inv = np.unique(D.pitcher.to_numpy(), return_inverse=True)
    rng = np.random.default_rng(5)
    vals, npk = [], []
    for _ in range(n_rebuild):
        w = np.bincount(rng.integers(0, len(pc), len(pc)), minlength=len(pc))[inv].astype(float)
        Lb = landscape.build(X, weights=w)
        lb = Lb.names(X)
        npk.append(len(Lb.peaks))
        vals.append(100 * np.sum(w[scored] * (lb[scored] == D.cls_sc.to_numpy()[scored])) / w[scored].sum())
    out["agreement_pct_landscape_rebuilt"] = r2([out["agreement_pct_fixed_landscape"][0],
                                                 np.quantile(vals, 0.025), np.quantile(vals, 0.975)])
    out["rebuilds"] = {"n": n_rebuild, "share_with_3_classes": float(np.mean(np.array(npk) == 3))}
    R["level1"] = out
    log("level1", out["peaks"], out["class_share_pct"], out["agreement_pct_fixed_landscape"],
        out["agreement_pct_landscape_rebuilt"], out["rebuilds"])
    return L


def park_test_all(D, n_keys=2000, seed=316):
    """Near-identical pairs on a sample of 2,000 pitcher-seasons with 2+ parks."""
    D["sc_collapsed"] = D.cls_sc.replace({"FC": "BB"})
    ps = D.groupby(["pitcher", "season"]).home.nunique()
    keys = list(ps[ps >= 2].sample(min(n_keys, int((ps >= 2).sum())), random_state=seed).index)
    cols = ["pitch_type", "sc_collapsed", "l1"]
    T, pit = ev.pairs(D, cols, keys)
    R["park_test_all_pitchers"] = summarize_pairs(T, pit, cols, len(keys))
    log("park test", {c: R["park_test_all_pitchers"]["mismatch_pct"][c] for c in cols})


def summarize_pairs(T, pit, cols, n_keys):
    out = {"pitcher_seasons": n_keys, "pairs": {k: int(T[k][cols[0]][:, 1].sum()) for k in T}, "mismatch_pct": {}}
    for c in cols:
        out["mismatch_pct"][c] = {}
        for k in T:
            P = ev.by_pitcher(T[k][c], pit)
            out["mismatch_pct"][c][k] = r2(ev.boot(P, ev.rate))
        # paired difference other park - same game, resampling pitchers jointly
        A = np.concatenate([ev.by_pitcher(T["other_park"][c], pit), ev.by_pitcher(T["same_game"][c], pit)], 1)
        out["mismatch_pct"][c]["other_park_minus_same_game"] = r2(ev.boot(A, lambda t: ev.rate(t[:2]) - ev.rate(t[2:])))
    return out


# ------------------------------------------------------------------------------------------------ taggers
def taggers(D, workers):
    y = D.type_sc.to_numpy(dtype=object)
    tr = ((~D.test) & D.season.isin([2023, 2024])).to_numpy()
    te = D.test.to_numpy()
    X = D[["hb", "ivb", "ds"]].to_numpy(np.float64)
    L = landscape.build(X[tr])                     # held-out pitchers never shape the landscape
    codes = L.classify(X)
    if (codes > 3).any():           # an extra class would need its own name map; treat it as unassigned
        log("WARNING extra Level 1 peak in the training landscape", L.peaks)
        codes = np.where(codes > 3, 0, codes)
    l1 = np.array(landscape.CLASSES, object)[codes]
    fpop = {c: level2.league_density(D.loc[tr & (l1 == c), ["hb", "ivb"]].to_numpy()) for c in ("FB", "BB", "CH")}
    R["tagger_setup"] = {"train_pitchers": int(D.pitcher[tr].nunique()), "train_pitches": int(tr.sum()),
                         "test_pitchers_2025": int(D.pitcher[te & (D.season == 2025)].nunique()),
                         "test_pitches_2025": int((te & (D.season == 2025)).sum()),
                         "landscape_peaks": L.peaks, "landscape_islands": L.islands, "barcode": L.barcode,
                         "class_share_train_pct": pd.Series(l1[tr]).value_counts(normalize=True).mul(100).round(2).to_dict()}
    log("tagger landscape", R["tagger_setup"])

    gid = level2.run(D, l1, fpop, mask=tr | te, workers=workers)
    log("level 2 done")
    G = tagger.group_table(D, l1, gid, y)
    test_g = G.pitcher.map(data.is_test_pitcher).to_numpy()
    Gtr, Gte = G[~test_g].reset_index(drop=True), G[test_g].reset_index(drop=True)
    R["level2"] = {"train_groups": len(Gtr), "test_groups": len(Gte),
                   "pitch_weighted_group_purity_test_pct": round(float(np.average(Gte.purity.fillna(0), weights=Gte.n) * 100), 2),
                   "groups_per_pitcher_class_test": round(float(Gte.groupby(["pitcher", "season", "cls"]).size().mean()), 3)}

    # A1: pitch-level booster on a 500k sample of labelled training pitches
    lab = np.flatnonzero(tr & pd.notna(y))
    samp = np.random.default_rng(7).choice(lab, min(500_000, len(lab)), replace=False)
    a1 = tagger.fit_a1(D.iloc[samp][tagger.PITCH_FEATURES], y[samp].astype(str))
    P = a1.predict_proba(D.loc[te, tagger.PITCH_FEATURES])
    D["A1"], D["A1_conf"] = None, np.nan
    D.loc[te, "A1"] = a1.classes_[P.argmax(1)]
    D.loc[te, "A1_conf"] = P.max(1)
    a1P = pd.DataFrame(P, index=D.index[te], columns=a1.classes_)
    log("A1 fit")

    # A2: group booster + joint naming
    a2 = tagger.fit_a2(Gtr)
    Gte["A2"], Gte["A2_conf"] = tagger.name_a2(a2, Gte)
    D["A2"] = tagger.to_pitches(D, l1, gid, Gte, "A2")
    D["A2_conf"] = pd.to_numeric(pd.Series(tagger.to_pitches(D, l1, gid, Gte, "A2_conf")), errors="coerce").to_numpy()
    # second confidence for A2: the pitch-level booster's mean probability of the group's name (an ensemble check)
    D["A2_conf_a1"] = np.nan
    m = te & pd.notna(D.A2).to_numpy()
    pr = a1P.loc[D.index[m]].to_numpy()
    col = pd.Index(a1.classes_).get_indexer(D.A2[m])
    p = np.where(col >= 0, pr[np.arange(int(m.sum())), np.maximum(col, 0)], 0.0)
    g = pd.DataFrame({"p": D.pitcher[m].to_numpy(), "s": D.season[m].to_numpy(), "c": l1[m], "g": gid[m], "x": p})
    D.loc[m, "A2_conf_a1"] = g.groupby(["p", "s", "c", "g"]).x.transform("mean").to_numpy()
    D["l1_tr"], D["gid"] = l1, gid
    log("A2 fit")

    t25 = D[te & (D.season == 2025).to_numpy() & pd.notna(y)]
    res = {}
    for k in ("A1", "A2"):
        C = ev.confusion_by_pitcher(t25.pitcher.to_numpy(), t25.type_sc.to_numpy(), t25[k].to_numpy(), TYPES)
        res[k] = {"accuracy_pct": r2(ev.boot(C, ev.accuracy)), "macro_f1": r4(ev.boot(C, ev.macro_f1)),
                  **ev.type_scores(C, TYPES)}
        log(k, res[k]["accuracy_pct"], res[k]["macro_f1"])
    Cd = np.concatenate([ev.confusion_by_pitcher(t25.pitcher.to_numpy(), t25.type_sc.to_numpy(), t25[k].to_numpy(),
                                                  TYPES) for k in ("A2", "A1")], 1)
    half = len(TYPES)
    res["A2_minus_A1_accuracy_pt"] = r2(ev.boot(Cd, lambda c: ev.accuracy(c[:half]) - ev.accuracy(c[half:])))
    res["A2_minus_A1_macro_f1"] = r4(ev.boot(Cd, lambda c: ev.macro_f1(c[:half]) - ev.macro_f1(c[half:])))
    # ceiling of any group-unit method: every group named by its own majority Statcast type
    Cc = ev.confusion_by_pitcher(t25.pitcher.to_numpy(), t25.type_sc.to_numpy(),
                                 tagger.to_pitches(t25, l1[t25.index], gid[t25.index], Gte, "y"), TYPES)
    res["group_majority_ceiling_accuracy_pct"] = r2(ev.boot(Cc, ev.accuracy))
    res["n_test_pitches"], res["n_test_pitchers"] = len(t25), int(t25.pitcher.nunique())
    R["tagger_2025_holdout"] = res

    # selective labelling
    cov = {}
    for k, c in (("A1", "A1_conf"), ("A2", "A2_conf"), ("A2_by_A1_agreement", "A2_conf_a1")):
        lbl = "A1" if k == "A1" else "A2"
        cov[k] = ev.coverage_curve(t25.pitcher.to_numpy(), t25.type_sc.to_numpy(), t25[lbl].to_numpy(),
                                   t25[c].to_numpy(), labels=TYPES)
        log("coverage", k, [(r["coverage"], r["accuracy_pct"][0]) for r in cov[k]])
    R["coverage"] = cov

    # park test on held-out pitchers, 2025
    t25p = t25.copy()
    ps = t25p.groupby(["pitcher", "season"]).home.nunique()
    keys = list(ps[ps >= 2].index)
    cols = ["type_sc", "A1", "A2", "l1_tr"]
    T, pit = ev.pairs(t25p, cols, keys)
    R["park_test_holdout_2025"] = summarize_pairs(T, pit, cols, len(keys))
    log("park test holdout", R["park_test_holdout_2025"]["mismatch_pct"])

    stability(D[te])
    level2_vs_statcast(D[te & (D.season == 2025).to_numpy()], l1[te & (D.season == 2025).to_numpy()],
                       gid[te & (D.season == 2025).to_numpy()])
    return dict(a1=a1, a2=a2, l1=l1, gid=gid, Gtr=Gtr, Gte=Gte, tr=tr, te=te, L=L, t25=t25)


def stability(Dt):
    """Year over year on held-out pitchers: 2023 -> 2024 and 2024 -> 2025."""
    out = {}
    cols = ["type_sc", "A1", "A2", "l1_tr"]
    for y1, y2 in ((2023, 2024), (2024, 2025)):
        T, pit, flips = ev.cross_season_pairs(Dt, cols, y1, y2)
        o = {"pitchers": len(pit), "pairs": int(T["type_sc"][:, 1].sum()),
             "cross_season_pair_mismatch_pct": {c: r2(ev.boot(T[c], ev.rate)) for c in cols}}
        # which label pairs flip, as % of that label's flipped pairs; and how concentrated flips are by pitcher
        o["flip_pairs_pct"] = {c: {k: round(100 * v / max(sum(flips[c].values()), 1), 1)
                                   for k, v in sorted(flips[c].items(), key=lambda kv: -kv[1])[:6]} for c in cols}
        o["share_of_flips_from_top_10pct_pitchers"] = {
            c: round(float(np.sort(T[c][:, 0])[::-1][:max(1, len(pit) // 10)].sum() / max(T[c][:, 0].sum(), 1) * 100), 1)
            for c in cols}
        A = np.concatenate([T["A2"], T["type_sc"]], 1)
        o["A2_minus_statcast_pt"] = r2(ev.boot(A, lambda t: ev.rate(t[:2]) - ev.rate(t[2:])))
        sub = Dt[Dt.season.isin([y1, y2]) & Dt.pitcher.isin(pit)]
        same = {}
        for c in cols:
            a = ev.arsenal(sub, c).unstack()
            same[c] = round(float((a[y1] == a[y2]).mean() * 100), 1)
        o["arsenal_identical_pct"] = same
        # does the tagger keep one majority name for a Statcast type cell (pitcher x type, 20+ pitches each year)?
        cell = sub[sub.type_sc.notna()].groupby(["pitcher", "type_sc", "season"])
        nn = cell.size()
        mj = {}
        for c in ("A1", "A2"):
            m = cell[c].agg(lambda s: s.mode().iloc[0] if s.notna().any() else None)[nn >= 20].unstack().dropna()
            mj[c] = round(float((m[y1] == m[y2]).mean() * 100), 1)
        mj["cells"] = int(len(m))
        o["same_majority_name_per_statcast_type_pct"] = mj
        out[f"{y1}_{y2}"] = o
        log("stability", y1, y2, o)
    R["stability_holdout"] = out


def level2_vs_statcast(D25, l1, gid, min_p=200):
    """Split counts vs distinct Statcast types (10+ pitches and 5%+ of the class) per pitcher-season-class."""
    big = D25.groupby("pitcher").size()
    m = D25.pitcher.isin(big[big >= min_p].index).to_numpy() & (gid >= 0)
    S = pd.DataFrame({"p": D25.pitcher.to_numpy()[m], "c": l1[m], "g": gid[m], "t": D25.type_sc.to_numpy()[m]})
    rows = []
    for (p, c), g in S.groupby(["p", "c"]):
        if len(g) < 30:
            continue
        vc = g.t.value_counts()
        rows.append((c, g.g.nunique(), int(((vc >= 10) & (vc / len(g) >= 0.05)).sum())))
    L = pd.DataFrame(rows, columns=["cls", "k", "types"])
    R["level2_vs_statcast_holdout_2025"] = {
        c: {"pitcher_classes": len(g), "topo_split_pct": round(float((g.k > 1).mean() * 100), 1),
            "statcast_multi_type_pct": round(float((g.types > 1).mean() * 100), 1),
            "split_agreement_pct": round(float(((g.k > 1) == (g.types > 1)).mean() * 100), 1)}
        for c, g in L.groupby("cls")}


# ------------------------------------------------------------------------------------------------ Phase C
SHAPE = ["speed", "hb", "ivb", "spin", "d_velo", "d_ivb", "d_hb", "d_spin", "balls", "strikes", "platoon"]


def boot_diff(pid, la, lb, n=1000, seed=7):
    g = pd.DataFrame({"p": pid, "d": lb - la}).groupby("p").d.agg(["sum", "size"]).to_numpy()
    return ev.boot(g, lambda t: t[0] / t[1], n, seed)


def phase_c(D, S):
    """Does the name add anything once shape is known? Shape-only booster vs shape + name, per class and outcome,
    trained on training pitchers (2023-24), scored on held-out pitchers (2025)."""
    s = D[D.season.isin([2023, 2024]) & D.xwoba.notna() & D.bip]
    s = s.sample(min(400_000, len(s)), random_state=7)
    xm = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, verbose=-1, n_jobs=6).fit(
        s[["ev", "la"]].to_numpy(), s.xwoba)
    R["phase_c_xwoba_model_r2"] = round(float(1 - np.mean((xm.predict(s[["ev", "la"]].to_numpy()) - s.xwoba) ** 2)
                                              / s.xwoba.var()), 4)
    D["xw"] = np.nan
    D.loc[D.bip, "xw"] = xm.predict(D.loc[D.bip, ["ev", "la"]].to_numpy())
    tr, te, l1, gid = S["tr"], S["te"], S["l1"], S["gid"]
    # out-of-fold tagger names on training pitchers (3 folds by pitcher)
    idx = np.flatnonzero(tr)
    yy = D.type_sc.to_numpy(dtype=object)
    D["A1_oof"] = None
    oof = np.empty(len(idx), dtype=object)
    for k, (a, b) in enumerate(GroupKFold(3).split(idx, groups=D.pitcher.to_numpy()[idx])):
        la = a[pd.notna(yy[idx[a]])]
        la = np.random.default_rng(k).choice(la, min(350_000, len(la)), replace=False)
        m = tagger.fit_a1(D.iloc[idx[la]][tagger.PITCH_FEATURES], yy[idx[la]].astype(str))
        oof[b] = m.classes_[m.predict_proba(D.iloc[idx[b]][tagger.PITCH_FEATURES]).argmax(1)]
    D.loc[D.index[idx], "A1_oof"] = oof
    Gtr = S["Gtr"].copy()
    Gtr["A2_oof"] = None
    for a, b in GroupKFold(3).split(Gtr, groups=Gtr.pitcher):
        Gtr.loc[b, "A2_oof"] = tagger.name_a2(tagger.fit_a2(Gtr.iloc[a]), Gtr.iloc[b].reset_index(drop=True))[0]
    D["A2_oof"] = tagger.to_pitches(D, l1, gid, Gtr, "A2_oof")
    log("phase C oof names")
    D["l1c"] = l1
    out = {}
    prm = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, verbose=-1, n_jobs=6,
               random_state=7)
    for cls in ("BB", "FB"):
        base = (D.l1c == cls) & D.type_sc.notna()
        Dtr, Dte = D[base & tr], D[base & te & (D.season == 2025) & D.A2.notna()]
        out[cls] = {}
        for oc, rows, yc, kind in (("whiff_per_swing", "swing", "whiff", "clf"),
                                   ("called_strike_per_take", "taken", "cstr", "clf"),
                                   ("xwoba_on_contact", "bip", "xw", "reg")):
            a, b = Dtr[Dtr[rows]], Dte[Dte[rows]]
            if len(a) > 400_000:
                a = a.sample(400_000, random_state=7)
            ya, yb = a[yc].astype(float).to_numpy(), b[yc].astype(float).to_numpy()

            def loss(Xa, Xb):
                if kind == "clf":
                    p = np.clip(LGBMClassifier(**prm).fit(Xa, ya).predict_proba(Xb)[:, 1], 1e-6, 1 - 1e-6)
                    return -(yb * np.log(p) + (1 - yb) * np.log(1 - p))
                return (LGBMRegressor(**prm).fit(Xa, ya).predict(Xb) - yb) ** 2

            l0 = loss(a[SHAPE], b[SHAPE])
            res = {"n_train": len(a), "n_test": len(b), "shape_only_loss": round(float(l0.mean()), 6)}
            for lab, (ca, cb) in (("statcast_name", ("type_sc", "type_sc")), ("A1_name", ("A1_oof", "A1")),
                                  ("A2_name", ("A2_oof", "A2"))):
                cats = sorted(set(a[ca].dropna()) | set(b[cb].dropna()))
                lb = loss(a[SHAPE].assign(name=pd.Categorical(a[ca], categories=cats)),
                          b[SHAPE].assign(name=pd.Categorical(b[cb], categories=cats)))
                d = boot_diff(b.pitcher.to_numpy(), l0, lb)
                res[f"plus_{lab}"] = {"diff_b_minus_a": [round(x, 6) for x in d],
                                      "rel_pct": round(float(d[0] / l0.mean() * 100), 3)}
            out[cls][oc] = res
            log("phase C", cls, oc, {k: v for k, v in res.items() if k.startswith("plus")})
    R["phase_c"] = out


# ------------------------------------------------------------------------------------------------ 2021-22
def older_seasons(data_dir, L_2325):
    try:
        D, info = data.load(data_dir, (2021, 2022))
    except FileNotFoundError as e:
        log("skip 2021-22:", e)
        return
    X = D[["hb", "ivb", "ds"]].to_numpy(np.float64)
    L = landscape.build(X)
    scored = D.cls_sc.isin(["FB", "BB", "CH"]).to_numpy()
    out = {"rows_by_season": info["rows_by_season"], "peaks_own_landscape": L.peaks, "islands": L.islands}
    for k, LL in (("own_landscape", L), ("landscape_2023_25", L_2325)):
        D["l1"] = LL.names(X)
        T = ev.tallies(D.pitcher[scored], (D.l1 == D.cls_sc).to_numpy()[scored])
        out[f"agreement_pct_{k}"] = r2(ev.boot(T, ev.rate))
        out[f"class_share_pct_{k}"] = D.l1.value_counts(normalize=True).mul(100).round(2).to_dict()
    D["sc_collapsed"] = D.cls_sc.replace({"FC": "BB"})
    ps = D.groupby(["pitcher", "season"]).home.nunique()
    keys = list(ps[ps >= 2].sample(min(2000, int((ps >= 2).sum())), random_state=316).index)
    cols = ["pitch_type", "sc_collapsed", "l1"]
    T, pit = ev.pairs(D, cols, keys)
    out["park_test"] = summarize_pairs(T, pit, cols, len(keys))
    R["seasons_2021_2022"] = out
    log("2021-22", {k: v for k, v in out.items() if k.startswith("agreement")}, out["park_test"]["mismatch_pct"])


def headline():
    rows = [("Level 1 class agreement vs Statcast, 2023-25 (%)", R["level1"]["agreement_pct_landscape_rebuilt"])]
    for c, nm in (("pitch_type", "Statcast pitch_type"), ("l1", "Level 1 class")):
        for k in ("same_game", "other_park"):
            rows.append((f"Park test {nm}, {k.replace('_', ' ')} (% pairs differing)",
                         R["park_test_all_pitchers"]["mismatch_pct"][c][k]))
    h = R["tagger_2025_holdout"]
    rows += [("A2 hybrid accuracy, held-out pitchers 2025 (%)", h["A2"]["accuracy_pct"]),
             ("A1 booster accuracy (%)", h["A1"]["accuracy_pct"]),
             ("A2 macro-F1", h["A2"]["macro_f1"]), ("A1 macro-F1", h["A1"]["macro_f1"]),
             ("A2 - A1 accuracy (pt)", h["A2_minus_A1_accuracy_pt"])]
    for c, nm in (("type_sc", "Statcast type"), ("A1", "A1"), ("A2", "A2")):
        rows.append((f"Park test holdout 2025, {nm}, other park (%)",
                     R["park_test_holdout_2025"]["mismatch_pct"][c]["other_park"]))
    for k in ("A1", "A2"):
        for r in R["coverage"][k]:
            if r["coverage"] in (0.95, 0.9, 0.8):
                rows.append((f"{k} accuracy at {int(r['coverage'] * 100)}% coverage (%)", r["accuracy_pct"]))
    for yy, o in R["stability_holdout"].items():
        for c, nm in (("type_sc", "Statcast"), ("A2", "A2"), ("A1", "A1")):
            rows.append((f"Cross-season pair mismatch {yy.replace('_', '->')}, {nm} (%)",
                         o["cross_season_pair_mismatch_pct"][c]))
    df = pd.DataFrame([(a, *b) for a, b in rows], columns=["quantity", "estimate", "ci_low", "ci_high"])
    df.to_csv(ROOT / "results/headline.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-phase-c", action="store_true")
    ap.add_argument("--skip-2021", action="store_true")
    ap.add_argument("--rebuilds", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="20%% of pitchers, 5 rebuilds: a quick end-to-end check")
    a = ap.parse_args()
    D, info = data.load(a.data, (2023, 2024, 2025))
    if a.smoke:
        D = D[D.pitcher.map(lambda p: p % 5 == 0)].reset_index(drop=True)
        a.rebuilds = 5
    R["data"] = info
    log("loaded", info)
    L = level1(D, a.rebuilds)
    park_test_all(D)
    S = taggers(D, a.workers)
    figures.make_all(ROOT / "figures", L, D, R, S)
    if not a.skip_phase_c:
        phase_c(D, S)
    if not a.skip_2021:
        older_seasons(a.data, L)
    headline()


if __name__ == "__main__":
    try:
        main()
    finally:
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results/results.json").write_text(json.dumps(R, indent=1, default=str))
        log("wrote results/results.json")
