# pitch-classes-statcast

A label-free pitch classifier, checked against MLB Statcast. It reimplements the topological method of Maloof,
Haumacher & Bonicki (2026), "How to classify a pitch", adds two rules for MLB data, and puts a supervised
naming layer on top of it. Everything is computed from public data: Baseball Savant pitch-level CSVs for 2021-2024,
and the MLB Stats API game feed for 2025.

A full write-up is coming soon. All numbers come from `scripts/run_all.py` and are stored in
`results/results.json`. A flat table of the headline numbers is in `results/headline.csv`. Outputs are aggregate
only. No pitch-level rows, player names or player ids are written.

## Key results

MLB regular season 2023-25, 2,136,916 pitches. Brackets give pitcher-bootstrap 95% intervals (1,000 draws;
the Level 1 agreement interval also rebuilds the landscape on each of 100 resamples).

| Result | Estimate [95% CI] |
|---|---|
| Level 1 class (FB / BB / CH) agrees with Statcast, cutters excluded | 97.61% [97.28, 98.02] |
| Same, 2021-22 seasons, own landscape | 97.93% [97.60, 98.21] |
| Park test, Statcast `pitch_type`: same game / other park | 0.55% [0.44, 0.67] / 0.97% [0.80, 1.18] |
| Park test, Level 1 class: same game / other park | 0.56% [0.46, 0.67] / 0.52% [0.39, 0.69] |
| Eight-type accuracy on held-out pitchers, 2025: A2 hybrid / A1 booster | 86.84% [84.40, 89.25] / 85.16% [83.18, 86.92] |
| A2 minus A1 accuracy | +1.68 pt [-0.10, +3.57] |
| Macro-F1: A2 / A1 | .827 [.788, .859] / .799 [.771, .823] |
| A2 accuracy at 95 / 90 / 80% coverage | 89.09 / 91.15 / 93.54% |
| A1 accuracy at 95 / 90 / 80% coverage | 87.01 / 88.77 / 91.78% |
| Park test on held-out pitchers, other park: Statcast / A1 / A2 | 0.73% [0.45, 1.10] / 4.93% [4.04, 6.06] / 0.16% [0.10, 0.24] |
| Same pitcher, same shape, next season, label differs (2024 to 2025): Statcast / A1 / A2 | 1.27% [0.89, 1.73] / 5.88% [4.67, 7.16] / 8.05% [4.69, 11.95] |
| Same, Level 1 class | 0.67% [0.20, 1.42] |
| Largest loss reduction from adding any pitch name to a shape model | 0.094% (fastball called strikes) |

In short: the three classes match Statcast and do not move between parks or seasons. The individual-type names
match Statcast about as well as a plain booster does, and they stay put within a season. From one season to the
next they change much more than Statcast's labels. Once shape is known, the name adds almost nothing to outcome
models.

## Layout

```
src/pitchtopo/
  data.py        Savant and game-feed loaders, the arm-side sign check, relative features
  landscape.py   Level 1: density, persistence, watershed, island rule, fastball-speed merge
  level2.py      Level 2: per-pitcher split test against a single-Gaussian null
  tagger.py      A1 (pitch-level LightGBM) and A2 (group-level LightGBM with joint naming)
  evaluate.py    pitcher bootstrap, pair tests, cross-season pairs, coverage curves
  figures.py     the five figures
scripts/
  fetch_statcast.py   download a season from Baseball Savant
  run_all.py          every number and figure
results/   results.json, headline.csv
figures/   landscape.png, confusion_a2.png, coverage.png, park_test.png, stability.png
tests/     sign convention, island rule, synthetic landscape, Level 2 split, joint naming
```

## Reproducing

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python -m pytest -q                       # about 3 s
python scripts/fetch_statcast.py 2023     # data/statcast_2023.parquet, ~20-40 min per season
python scripts/fetch_statcast.py 2024
python scripts/fetch_statcast.py 2025
nice -n 10 python scripts/run_all.py      # about 7 minutes with 6 workers, peak memory about 4 GB
```

`run_all.py` reads `data/statcast_{year}.parquet` (or `.csv`) and falls back to `data/gamefeed_{year}.parquet`, an
MLB Stats API game-feed pitch table with the columns listed in `data.GAMEFEED_COLUMNS`. Add `--smoke` for a 20% sample
run (about 2 minutes), `--skip-phase-c` to skip the outcome models, `--skip-2021` to skip the older seasons check.
`data/` is git-ignored. Do not commit data.

The published run used Savant files for 2021-2024 and the game feed for 2025. I checked the game-feed loader
against `fetch_statcast.py` output for 1-2 June 2025: on 5,750 matched pitches, movement and release agree to a
standard deviation of 0.03 inches and 0.003 feet, swing, whiff and take flags agree on every pitch, and
`pitch_type` agrees on 99.4% (Savant revises some labels after the fact). Fetching 2025 from Savant instead should
move the 2025 numbers by about that much.

Conventions are checked, not assumed. Savant `pfx_x` is the catcher's view (positive toward first base), so
arm-side break is `-pfx_x * 12` for right-handers and `+pfx_x * 12` for left-handers. The loader confirms this per
season from four-seam medians and stops if a source has the other sign.

## Credit and references

The method is not mine. It comes from Adam Maloof, Joe Haumacher and Kyle Bonicki, who proposed classifying
pitches as peaks of a movement density landscape, found by persistent homology and bounded by a watershed. This
repo reimplements their published settings on Statcast, adds two rules for MLB data (an island rule and a
fastball-speed merge), and adds a supervised naming layer and the tests above.

- Maloof, A., Haumacher, J., and Bonicki, K. (2026). How to classify a pitch. *SABR Tooth Tigers*, March 2026.
  https://medium.com/sabr-tooth-tigers
- Edelsbrunner, H., Letscher, D., and Zomorodian, A. (2002). Topological persistence and simplification.
  *Discrete & Computational Geometry* 28, 511-533.
- Vincent, L., and Soille, P. (1991). Watersheds in digital spaces: an efficient algorithm based on immersion
  simulations. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 13(6), 583-598.
- Scott, D. W. (1992). *Multivariate Density Estimation*. Wiley.
- Ke, G., et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS* 30.
- Geifman, Y., and El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS* 30.

## Data

- Baseball Savant Statcast search, https://baseballsavant.mlb.com (column documentation:
  https://baseballsavant.mlb.com/csv-docs). Statcast's `pitch_type` is MLBAM's own classification. It is used here as
  the reference, not as ground truth.
- MLB Stats API game feed (2025 in the published run).

MIT licensed (code). The data belongs to MLB Advanced Media and is subject to its terms of use.
