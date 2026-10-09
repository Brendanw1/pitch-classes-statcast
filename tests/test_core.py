import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pitchtopo import data, landscape, level2, tagger  # noqa: E402

RNG = np.random.default_rng(0)
PEAKS = {"FB": (8.0, 16.0, -0.5), "BB": (-4.0, 2.0, -9.0), "CH": (15.0, 5.0, -8.0)}


def blob(center, n, sd=(2.0, 2.0, 1.0)):
    return RNG.normal(center, sd, size=(n, 3))


def three_peaks(extra=()):
    parts = [blob(PEAKS["FB"], 50_000), blob(PEAKS["BB"], 35_000), blob(PEAKS["CH"], 15_000)]
    return np.vstack(parts + list(extra))


def test_sign_convention():
    # Savant pfx_x is the catcher's view: a right-hander's four-seamer runs toward third base (negative).
    df = pd.DataFrame({"throws": ["R"] * 3 + ["L"] * 3, "pitch_type": ["FF"] * 6,
                       "pfx_x_in": [-7.0, -8.0, -9.0, 7.5, 8.0, 8.5]})
    assert data.arm_side_sign(df) == {"R": -1.0, "L": 1.0}
    df["pfx_x_in"] *= -1                       # a source with the opposite sign is caught, not silently mirrored
    assert data.arm_side_sign(df) == {"R": 1.0, "L": -1.0}


def test_synthetic_landscape_finds_and_names_three_peaks():
    L = landscape.build(three_peaks())
    assert set(L.peaks) == {"FB", "BB", "CH"}
    for name, c in PEAKS.items():
        assert np.allclose(L.peaks[name], c, atol=1.5), (name, L.peaks[name])
    lab = L.names(np.array([PEAKS["FB"], PEAKS["BB"], PEAKS["CH"], (0, 0, 4.9), (30, 0, 0)]))
    assert list(lab[:3]) == ["FB", "BB", "CH"]
    assert lab[4] == "unassigned"              # off the grid


def test_island_rule_drops_small_disconnected_basin():
    island = blob((0.0, -15.0, -0.5), 300, sd=(0.5, 0.5, 0.3))        # 0.3% of pitches, far from everything
    L = landscape.build(three_peaks([island]))
    assert set(L.peaks) == {"FB", "BB", "CH"}
    assert len(L.islands) == 1 and L.islands[0]["pct_pitches"] < 0.5
    assert L.n_published_rule == 4             # the published rule alone keeps it


def test_fastball_speed_peak_merges_into_fastball():
    sinker = blob((16.0, 7.0, -0.5), 30_000, sd=(1.5, 1.5, 1.0))
    L = landscape.build(three_peaks([sinker]))
    assert "FB_merged" in L.peaks
    assert L.names(np.array([[16.0, 7.0, -0.5]]))[0] == "FB"


def test_level2_splits_two_pitches_not_one():
    fpop = level2.league_density(RNG.normal((0, 5), (6, 6), size=(20_000, 2)))
    one = RNG.normal((0, 2), (2, 2), size=(400, 2))
    two = np.vstack([RNG.normal((-8, 0), 1.2, size=(200, 2)), RNG.normal((4, 8), 1.2, size=(200, 2))])
    assert level2.split(one, fpop, np.random.default_rng(1), sims=200)[1] == 1
    lab, k = level2.split(two, fpop, np.random.default_rng(1), sims=200)
    assert k == 2 and len(set(lab[:200])) == 1 and lab[0] != lab[-1]


def test_joint_names_are_distinct_and_best_as_a_set():
    # both groups prefer name 0; jointly the better assignment gives the second group name 1
    logp = np.log(np.array([[0.9, 0.1], [0.6, 0.4]]))
    assert list(tagger.joint_names(logp)) == [0, 1]
