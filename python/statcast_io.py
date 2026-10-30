"""statcast_io.py — cached Statcast pitch-level acquisition shared by modules 04+.

Pulls Baseball Savant data with pybaseball one week at a time and caches each
season to data/raw/statcast/statcast_<season>.parquet (gitignored), so a
season is downloaded once and every later run is instant.

    from statcast_io import load_season
    sc = load_season(2026)               # regular season + postseason
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/raw/statcast"

# Columns the analysis modules need — keeps the cache small (~10x smaller).
KEEP = [
    "game_pk", "game_date", "game_type", "at_bat_number", "pitch_number",
    "inning", "pitcher", "batter", "player_name", "home_team", "away_team",
    "inning_topbot", "pitch_type", "release_speed", "description", "events",
    "balls", "strikes", "zone", "plate_x", "plate_z", "bat_speed",
    "swing_length", "stand", "p_throws",
]


def _weeks(start: date, end: date):
    d = start
    while d <= end:
        yield d, min(d + timedelta(days=6), end)
        d += timedelta(days=7)


def load_season(season: int, start: str | None = None, end: str | None = None,
                refresh: bool = False) -> pd.DataFrame:
    """Return all tracked pitches for a season (March 1 – November 30 by default)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"statcast_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    from pybaseball import statcast  # imported lazily: only needed on first pull
    from pybaseball import cache as pb_cache
    pb_cache.enable()

    s = date.fromisoformat(start) if start else date(season, 3, 1)
    e = date.fromisoformat(end) if end else min(date(season, 11, 30), date.today())
    chunks = []
    for a, b in _weeks(s, e):
        df = statcast(a.isoformat(), b.isoformat(), verbose=False)
        if df is not None and len(df):
            chunks.append(df[[c for c in KEEP if c in df.columns]])
        print(f"  {a} → {b}: {0 if df is None else len(df):,} pitches")
    sc = pd.concat(chunks, ignore_index=True).drop_duplicates(
        ["game_pk", "at_bat_number", "pitch_number"])
    sc.to_parquet(path, index=False)
    print(f"Cached {len(sc):,} pitches → {path.relative_to(ROOT)}")
    return sc
