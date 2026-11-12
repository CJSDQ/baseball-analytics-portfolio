# LIDOM database: design note

**Script:** `python/08_lidom_database.py`
**Status:** step 1. Tested on fixture data; first real load pending.

## Purpose

One clean source for every LIDOM analysis that follows: who is on each roster
now, and what each of those players has done in every other league, keyed by a
single player id (MLBAM).

## Coverage

| Source (MLB Stats API) | Covered |
|---|---|
| LIDOM rosters, `league 131` | Full rosters, about 60 players per team, as dated snapshots |
| LIDOM stats, `sportId 17` | Yes, in season |
| MLB and US minors, `sportId 1, 11–16` | Yes |
| Mexican League (LMB), `sportId 23, league 125` | Yes |
| Other winter leagues (LMP, LVBP, LBPRC, AFL) | Yes, through `sportId 17` |
| NPB (Japan), `sportId 31` | No: listed, but returns no stats |

The default load is the 2026–27 LIDOM season plus three earlier seasons of stats.

## Tables

| Table | Grain |
|---|---|
| `dim_player` | One row per player |
| `dim_team` | One row per team and season |
| `dim_league` | One row per league, with a short level label |
| `fact_lidom_roster` | One row per snapshot date, LIDOM team and player |
| `fact_batting_season` | One row per player, season, league and team |
| `fact_pitching_season` | One row per player, season, league and team; innings stored as outs |
| `v_latest_roster` | The latest roster snapshot |
| `v_lidom_player_seasons` | Every season, in every league, for each player on the latest roster |

Design choices:

- **Raw layer kept apart.** API responses are cached as JSON and never edited,
  so the database can always be rebuilt with `--offline`.
- **Rosters are snapshots.** LIDOM rosters change all winter; each run adds a
  dated snapshot instead of overwriting the last.
- **Per-team rows only.** A player traded mid-season has one row per team.
  Combined rows without a team are skipped, and duplicate rows are counted.
- **Portable SQL.** The DDL avoids SQLite-only features, so it moves to SQL
  Server or Snowflake with type renames.

## Quality checks

Each run fails if any of these find a problem:

- every LIDOM team has a roster in the latest snapshot
- every rostered player is in `dim_player`
- every stat row has a known league
- no negative or impossible counts (hits above at-bats, strikeouts above batters faced)

## Run it

The download needs normal internet access, so run it in your own Terminal:

```bash
python python/08_lidom_database.py
```

Rerun weekly during the season to add roster snapshots and refresh current stats.

## Next steps

- Japan: a separate NPB source.
- Statcast pitch-level data for rostered players with MLB or Triple-A time.
- League translation factors, which this database is built to support.
