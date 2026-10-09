"""Load public MLB pitch data into one internal schema.

Two sources are supported, both public:
  * Baseball Savant "statcast search" CSV rows (scripts/fetch_statcast.py writes data/statcast_{year}.parquet).
  * MLB Stats API game-feed pitch tables (data/gamefeed_{year}.parquet), the source of the 2025 season here.

Conventions (checked by `arm_side_sign`, never assumed):
  hb   horizontal break in inches, positive toward the pitcher's arm side (left-handers mirrored).
  ivb  induced vertical break in inches.
  ds   speed minus the pitcher's 95th-percentile speed in that game (mph), so the fastball sits near 0.
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

# Statcast pitch_type -> Level 1 class for the class-agreement test. FC is scored separately (the FB/BB border).
CLASS_OF = {**dict.fromkeys(["FF", "SI"], "FB"), **dict.fromkeys(["SL", "ST", "CU", "KC", "SV", "CS"], "BB"),
            **dict.fromkeys(["CH", "FS", "FO", "SC"], "CH"), "FC": "FC"}
# Statcast pitch_type -> the 8 individual types the taggers learn. KC/CS/SV fold into CU, SC into CH, FO into FS.
# KN, EP, FA, PO and unknown are dropped from the type tests (about 0.3% of pitches).
TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"]
TYPE_OF = {**{t: t for t in TYPES}, "KC": "CU", "CS": "CU", "SV": "CU", "SC": "CH", "FO": "FS"}

SAVANT_COLUMNS = ["game_type", "game_pk", "pitcher", "p_throws", "stand", "home_team", "pitch_type", "release_speed",
                  "pfx_x", "pfx_z", "release_spin_rate", "spin_axis", "release_extension", "release_pos_x",
                  "release_pos_z", "balls", "strikes", "description", "launch_speed", "launch_angle",
                  "estimated_woba_using_speedangle"]
GAMEFEED_COLUMNS = ["game_id", "pitcher_id", "pitcher_hand", "pitcher_team", "batter_hand", "ab_number", "pitch_type",
                    "start_speed", "hb", "ivb", "spin_rate", "spin_direction", "extension", "x0", "y0", "z0", "vx0",
                    "vy0", "vz0", "ax", "ay", "az", "is_pitch", "play_code", "balls", "strikes", "launch_speed",
                    "launch_angle"]

SAVANT_SWING = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "foul", "foul_bunt", "missed_bunt",
                "bunt_foul_tip", "hit_into_play"}
SAVANT_WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
SAVANT_TAKEN = {"ball", "blocked_ball", "called_strike"}
FEED_SWING, FEED_WHIFF, FEED_TAKEN = set("SWTFLMOXDE"), set("SWT"), {"B", "*B", "C"}


def from_savant(path, season):
    s = pd.read_parquet(path, columns=SAVANT_COLUMNS) if str(path).endswith(".parquet") else \
        pd.read_csv(path, usecols=SAVANT_COLUMNS)
    s = s[s.game_type == "R"]
    d = s.description.astype(str)
    return pd.DataFrame({
        "season": season, "pitcher": s.pitcher, "game": s.game_pk, "home": s.home_team, "throws": s.p_throws,
        "stand": s.stand, "pitch_type": s.pitch_type, "speed": s.release_speed,
        "pfx_x_in": s.pfx_x * 12, "ivb": s.pfx_z * 12,          # catcher's view, + toward first base
        "spin": s.release_spin_rate, "axis": s.spin_axis, "ext": s.release_extension,
        "rel_x": s.release_pos_x, "rel_z": s.release_pos_z, "balls": s.balls, "strikes": s.strikes,
        "swing": d.isin(SAVANT_SWING), "whiff": d.isin(SAVANT_WHIFF), "taken": d.isin(SAVANT_TAKEN),
        "cstr": d.eq("called_strike"), "in_play": d.eq("hit_into_play"),
        "ev": s.launch_speed, "la": s.launch_angle, "xwoba": s.estimated_woba_using_speedangle})


def release_from_fit(t):
    """Release point from the 9-parameter fit (given at y0 = 50 ft), evaluated at y = 60.5 - extension."""
    a, b, c = 0.5 * t.ay, t.vy0, t.y0 - (60.5 - t.extension)
    disc = np.sqrt(b * b - 4 * a * c)
    r1, r2 = (-b - disc) / (2 * a), (-b + disc) / (2 * a)
    tt = np.where(np.abs(r1) < np.abs(r2), r1, r2)
    return t.x0 + t.vx0 * tt + 0.5 * t.ax * tt ** 2, t.z0 + t.vz0 * tt + 0.5 * t.az * tt ** 2


def from_gamefeed(path, season):
    t = pd.read_parquet(path, columns=GAMEFEED_COLUMNS)
    # Home team = the fielding team on the game's first at-bat (top of the first).
    home = t.loc[t.groupby("game_id").ab_number.idxmin()].set_index("game_id").pitcher_team
    t = t[t.is_pitch]
    rx, rz = release_from_fit(t)
    c = t.play_code.astype(str)
    return pd.DataFrame({
        "season": season, "pitcher": t.pitcher_id, "game": t.game_id, "home": t.game_id.map(home),
        "throws": t.pitcher_hand, "stand": t.batter_hand, "pitch_type": t.pitch_type, "speed": t.start_speed,
        # The feed's hb is pitcher-relative with the opposite sign to Savant pfx_x (checked on 2024:
        # hb = -0.9999 * pfx_x * 12). Store the catcher's view so both sources go through the same sign check.
        "pfx_x_in": -t.hb, "ivb": t.ivb, "spin": t.spin_rate, "axis": t.spin_direction, "ext": t.extension,
        "rel_x": rx, "rel_z": rz, "balls": t.balls, "strikes": t.strikes,
        "swing": c.isin(FEED_SWING), "whiff": c.isin(FEED_WHIFF), "taken": c.isin(FEED_TAKEN), "cstr": c.eq("C"),
        "in_play": c.isin(set("XDE")), "ev": t.launch_speed, "la": t.launch_angle, "xwoba": np.nan})


def arm_side_sign(df):
    """Sign that turns catcher's-view inches into arm-side inches, per hand, from four-seam medians.
    A four-seamer runs to the arm side, so its arm-side break must be positive for both hands."""
    out = {}
    for h in ("R", "L"):
        med = df.pfx_x_in[(df.throws == h) & (df.pitch_type == "FF")].median()
        out[h] = -1.0 if med < 0 else 1.0
    return out


def load_season(data_dir, season):
    data_dir = Path(data_dir)
    for name, reader in ((f"statcast_{season}.parquet", from_savant), (f"statcast_{season}.csv", from_savant),
                         (f"gamefeed_{season}.parquet", from_gamefeed)):
        if (data_dir / name).exists():
            return reader(data_dir / name, season), name.split("_")[0]
    raise FileNotFoundError(f"no statcast_{season}.parquet/.csv or gamefeed_{season}.parquet in {data_dir}")


def load(data_dir, seasons):
    parts, info = [], {"source": {}, "sign_check": {}}
    for y in seasons:
        df, src = load_season(data_dir, y)
        df = df.dropna(subset=["speed", "pfx_x_in", "ivb", "throws", "pitcher", "game"])
        df = df[df.throws.isin(["R", "L"])]
        sign = arm_side_sign(df)
        assert sign == {"R": -1.0, "L": 1.0}, f"{y}: unexpected horizontal sign {sign}"
        info["source"][y], info["sign_check"][y] = src, sign
        parts.append(df)
    D = pd.concat(parts, ignore_index=True)
    for c in ("speed", "pfx_x_in", "ivb", "spin", "axis", "ext", "rel_x", "rel_z", "ev", "la", "xwoba"):
        D[c] = pd.to_numeric(D[c], errors="coerce").astype(np.float64)
    D["pitcher"] = D.pitcher.astype(np.int64)
    D["game"] = D.game.astype(np.int64)
    D["home"] = D.home.replace({"OAK": "ATH"})            # Oakland's code changed in 2025
    R = (D.throws == "R").to_numpy()
    D["hb"] = np.where(R, -D.pfx_x_in, D.pfx_x_in)
    D["axis"] = np.where(R, D.axis, 360 - D.axis)          # mirror left-handers' spin axis
    D["rel_side"] = np.where(R, -D.rel_x, D.rel_x)          # arm side positive
    D["rel_h"] = D.rel_z
    D["platoon"] = (D.stand == D.throws).astype(np.int8)
    D["bip"] = D.in_play & D.ev.notna() & D.la.notna()
    D["cls_sc"] = D.pitch_type.map(CLASS_OF)
    D["type_sc"] = D.pitch_type.map(TYPE_OF)
    D["test"] = D.pitcher.map(is_test_pitcher).astype(bool)
    D = add_relative_features(D)
    info["medians"] = {f"{h}_{pt}": [round(float(D[c][(D.throws == h) & (D.pitch_type == pt)].median()), 2)
                                     for c in ("hb", "ivb", "ds")] for h in "RL" for pt in ("FF", "SI")}
    info["rows_by_season"] = D.season.value_counts().sort_index().to_dict()
    return D.drop(columns=["pfx_x_in", "rel_x", "rel_z"]), info


def is_test_pitcher(pid):
    """Fixed pitcher holdout: a quarter of pitchers, by a hash of the MLBAM id, never seen in training."""
    return zlib.crc32(str(int(pid)).encode()) % 4 == 0


def add_relative_features(D):
    """ds = speed - outing p95 (the landscape axis). The tagger's fastball reference is tag-free: pitches within
    2.5 mph of the pitcher-season's p90 speed; d_* are the pitch minus that reference."""
    D["ds"] = D.speed - D.groupby(["pitcher", "game"]).speed.transform("quantile", 0.95)
    q90 = D.groupby(["pitcher", "season"]).speed.transform("quantile", 0.9)
    fb = D.speed >= q90 - 2.5
    for c in ("ivb", "hb", "spin"):
        D[f"d_{c}"] = D[c] - D[c].where(fb).groupby([D.pitcher, D.season]).transform("mean")
    D["d_velo"] = D.speed - q90
    D["axis_sin"], D["axis_cos"] = np.sin(np.radians(D.axis)), np.cos(np.radians(D.axis))
    return D
