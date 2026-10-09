"""Download one regular season of pitch-level Statcast data from Baseball Savant's public CSV search.

    python scripts/fetch_statcast.py 2024                 # -> data/statcast_2024.parquet
    python scripts/fetch_statcast.py 2025 --start 2025-06-01 --end 2025-06-07

Savant returns at most 25,000 rows per query, so the season is fetched one day at a time (a full slate is about
4,500 pitches). Only the columns the analysis reads are kept. A season is about 90 MB as parquet and takes
20-40 minutes; please be polite to the server (one request at a time, a short pause between days).

Columns kept (Savant names, see https://baseballsavant.mlb.com/csv-docs):
    game_type            R = regular season (the analysis keeps only R)
    game_pk              game id
    game_date            date
    pitcher              MLBAM pitcher id (no names are kept)
    p_throws, stand      pitcher and batter hand
    home_team            ballpark proxy for the park test
    pitch_type           Statcast's pitch classification (the reference labels)
    release_speed        mph at release
    pfx_x, pfx_z         movement in feet, catcher's view (pfx_x > 0 = toward first base); pfx_z is induced
                         vertical break. The analysis converts to inches and to the pitcher's arm side.
    release_spin_rate    rpm
    spin_axis            degrees
    release_extension    feet
    release_pos_x/z      release point in feet, catcher's view
    balls, strikes       count
    description          pitch result (swing / whiff / called strike / ball / in play)
    launch_speed, launch_angle, estimated_woba_using_speedangle   batted-ball outcome (Phase C)
"""
import argparse, datetime as dt, io, sys, time, urllib.parse, urllib.request
from pathlib import Path

import pandas as pd

KEEP = ["game_type", "game_pk", "game_date", "pitcher", "p_throws", "stand", "home_team", "pitch_type",
        "release_speed", "pfx_x", "pfx_z", "release_spin_rate", "spin_axis", "release_extension", "release_pos_x",
        "release_pos_z", "balls", "strikes", "description", "launch_speed", "launch_angle",
        "estimated_woba_using_speedangle"]
URL = "https://baseballsavant.mlb.com/statcast_search/csv"


def day(d):
    q = {"all": "true", "type": "details", "player_type": "pitcher", "hfGT": "R|", "hfSea": f"{d.year}|",
         "game_date_gt": d.isoformat(), "game_date_lt": d.isoformat()}
    req = urllib.request.Request(f"{URL}?{urllib.parse.urlencode(q)}", headers={"User-Agent": "research-fetch"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = r.read().decode("utf-8-sig")
            if not raw.strip() or raw.lstrip().startswith("<"):
                return pd.DataFrame(columns=KEEP)
            df = pd.read_csv(io.StringIO(raw), low_memory=False)
            if len(df) >= 25_000:
                raise RuntimeError(f"{d}: hit the 25,000-row cap; split the day")
            return df.reindex(columns=KEEP)
        except (OSError, ValueError) as e:
            time.sleep(5 * (attempt + 1))
            err = e
    raise RuntimeError(f"{d}: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("season", type=int)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    a = ap.parse_args()
    start = dt.date.fromisoformat(a.start) if a.start else dt.date(a.season, 3, 15)
    end = dt.date.fromisoformat(a.end) if a.end else dt.date(a.season, 10, 5)
    parts, d = [], start
    while d <= end:
        df = day(d)
        if len(df):
            parts.append(df)
            print(d, len(df), flush=True)
        d += dt.timedelta(days=1)
        time.sleep(1.0)
    if not parts:
        sys.exit("no rows")
    out = pd.concat(parts, ignore_index=True)
    out = out[out.game_type == "R"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    suffix = "" if not (a.start or a.end) else f"_{start}_{end}"
    path = Path(a.out) / f"statcast_{a.season}{suffix}.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path} ({len(out):,} pitches)")


if __name__ == "__main__":
    main()
