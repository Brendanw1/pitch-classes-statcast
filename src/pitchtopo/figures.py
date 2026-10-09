"""Five aggregate figures. Nothing pitch-level is drawn: the landscape panels show binned league density."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import landscape  # noqa: E402
from .data import TYPES  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.titlesize": 11, "axes.titleweight": "regular"})


def _landscape(L, path):
    f, lab = L.density, L.labels
    edges = [np.linspace(lo, hi, landscape.NB + 1) for lo, hi in zip(landscape.LO, landscape.HI)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (keep, drop, xl, yl) in zip(axes, ((0, 1, "Horizontal break, arm side + (in)", "Induced vertical break (in)"),
                                               (0, 2, "Horizontal break, arm side + (in)", "Speed minus outing p95 (mph)"))):
        other = 3 - keep - drop
        m = f.sum(axis=other)
        # class of the densest voxel along the summed axis, for the boundary lines
        cl = np.take_along_axis(lab, np.expand_dims(f.argmax(axis=other), other), other).squeeze(other)
        cl = np.where(m > 1e-3 * m.max(), cl, 0)
        ax.pcolormesh(edges[keep], edges[drop], np.log10(m.T + 1), cmap="Blues", shading="flat", rasterized=True)
        for c in (1, 2, 3):
            ax.contour((edges[keep][:-1] + edges[keep][1:]) / 2, (edges[drop][:-1] + edges[drop][1:]) / 2,
                       (cl == c).T.astype(float), levels=[0.5], colors=INK, linewidths=0.8)
        for name, xyz in L.peaks.items():
            if name in ("FB", "BB", "CH"):
                ax.plot(xyz[keep], xyz[drop], "o", ms=7, mfc="white", mec=INK, mew=1.2)
                ax.annotate(name, (xyz[keep], xyz[drop]), xytext=(7, 5), textcoords="offset points", color=INK)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(False)
    axes[1].set_ylim(-20, 5)
    fig.suptitle("MLB 2023-25 movement landscape (log density) with the three watershed classes", x=0.01, ha="left")
    fig.savefig(path)
    plt.close(fig)


def _confusion(t25, path):
    ct = pd.crosstab(t25.type_sc, t25.A2.fillna("none"), normalize="index").reindex(index=TYPES,
                                                                                  columns=TYPES + ["none"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.imshow(ct.to_numpy() * 100, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(TYPES)):
        for j in range(len(TYPES) + 1):
            v = ct.iat[i, j] * 100
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 55 else INK)
    ax.set_xticks(range(len(TYPES) + 1), TYPES + ["none"])
    ax.set_yticks(range(len(TYPES)), TYPES)
    ax.set_xlabel("A2 label")
    ax.set_ylabel("Statcast pitch_type")
    ax.grid(False)
    ax.set_title("A2 hybrid on held-out pitchers, 2025 (% of each Statcast type)", loc="left")
    fig.savefig(path)
    plt.close(fig)


def _curve(y, pred, conf):
    conf = np.where(pd.notna(pred), np.nan_to_num(np.asarray(conf, float), nan=-np.inf), -np.inf)
    hit = (np.asarray(y, object) == np.asarray(pred, object))[np.argsort(-conf, kind="stable")]
    k = np.arange(1, len(hit) + 1)
    return k / len(hit), np.cumsum(hit) / k * 100


def _coverage(t25, R, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, pred, c, color, legend in (("A1", "A1", "A1_conf", ORANGE, "A1 pitch booster"),
                                        ("A2", "A2", "A2_conf", BLUE, "A2 hybrid, ranked by its own confidence"),
                                        ("A2_by_A1_agreement", "A2", "A2_conf_a1", AQUA,
                                         "A2 hybrid, ranked by A1 agreement")):
        x, a = _curve(t25.type_sc.to_numpy(), t25[pred].to_numpy(), t25[c].to_numpy())
        keep = x >= 0.5
        ax.plot(x[keep] * 100, a[keep], color=color, lw=2, label=legend)
        for r in R["coverage"][key]:
            ax.errorbar(r["coverage"] * 100, r["accuracy_pct"][0],
                        yerr=[[r["accuracy_pct"][0] - r["accuracy_pct"][1]], [r["accuracy_pct"][2] - r["accuracy_pct"][0]]],
                        fmt="o", ms=4, color=color, capsize=2, lw=1)
    ax.invert_xaxis()
    ax.set_xlabel("Coverage: % of pitches labelled (abstain on the rest)")
    ax.set_ylabel("Accuracy on labelled pitches (%)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Selective labelling, held-out pitchers 2025 (dots: pitcher-bootstrap 95% CI)", loc="left")
    fig.savefig(path)
    plt.close(fig)


def _bars(ax, groups, series, get, colors):
    w = 0.8 / len(series)
    for j, (s, lbl) in enumerate(series):
        vals = np.array([get(s, g) for g, _ in groups])
        x = np.arange(len(groups)) + (j - (len(series) - 1) / 2) * w
        ax.bar(x, vals[:, 0], w * 0.92, color=colors[j], label=lbl)
        ax.errorbar(x, vals[:, 0], yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], fmt="none",
                    ecolor=INK, capsize=2, lw=0.8)
    ax.set_xticks(range(len(groups)), [g[1] for g in groups])
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)


def _park(R, path):
    P = R["park_test_holdout_2025"]["mismatch_pct"]
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [("same_game", "Same game"), ("same_park_other_game", "Same park,\nother game"), ("other_park", "Other park")]
    _bars(ax, groups, [("type_sc", "Statcast type"), ("A1", "A1 pitch booster"), ("A2", "A2 hybrid")],
          lambda s, g: P[s][g], [VIOLET, ORANGE, BLUE])
    ax.set_ylabel("% of near-identical pairs labelled differently")
    ax.set_title("Park test, held-out pitchers 2025 (95% CI)", loc="left")
    fig.savefig(path)
    plt.close(fig)


def _stability(R, path):
    S = R["stability_holdout"]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    groups = [(k, k.replace("_", " to ")) for k in S]
    _bars(ax, groups, [("type_sc", "Statcast type"), ("A1", "A1 pitch booster"), ("A2", "A2 hybrid")],
          lambda s, g: S[g]["cross_season_pair_mismatch_pct"][s], [VIOLET, ORANGE, BLUE])
    ax.set_ylabel("% of near-identical cross-season pairs\nlabelled differently")
    ax.set_title("Same pitcher, same shape, next season (held-out pitchers, 95% CI)", loc="left")
    fig.savefig(path)
    plt.close(fig)


def make_all(out, L, D, R, S):
    out = Path(out)
    out.mkdir(exist_ok=True)
    _landscape(L, out / "landscape.png")
    _confusion(S["t25"], out / "confusion_a2.png")
    _coverage(S["t25"], R, out / "coverage.png")
    _park(R, out / "park_test.png")
    _stability(R, out / "stability.png")
