"""08_lidom_database.py -- a small, layered database of LIDOM players.

Starts from the current LIDOM rosters and links every rostered player, by his
MLBAM id, to his season stats in every league the MLB Stats API covers: MLB,
the US minors, the Mexican League (LMB) and the winter leagues.

Layers
    raw   data/raw/lidom/        API responses cached as JSON, never edited
    core  data/lidom/lidom.db    SQLite tables with declared keys (DDL below)
    view  v_lidom_player_seasons every season, every league, for each player
                                 on the latest roster snapshot

The DDL avoids SQLite-only features so it ports to SQL Server or Snowflake with
type renames. Rosters are stored as dated snapshots because LIDOM rosters move
all season; reruns add a snapshot rather than overwrite one.

Known gap: NPB (Japan) is listed by the API but returns no stats.

Usage:
    python python/08_lidom_database.py                 # download, then build
    python python/08_lidom_database.py --offline       # rebuild from cache only
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/lidom"
DB_PATH = ROOT / "data/lidom/lidom.db"
API = "https://statsapi.mlb.com/api/v1"

LIDOM_LEAGUE_ID = 131
WINTER_SPORT_ID = 17
SPORTS = {1: "MLB", 11: "AAA", 12: "AA", 13: "A+", 14: "A", 16: "ROK",
          17: "WIN", 23: "IND"}
GROUPS = ("hitting", "pitching")
# League-level labels that are clearer than the sport's.
LEAGUE_LEVELS = {131: "LIDOM", 132: "LMP", 133: "LBPRC", 135: "LVBP",
                 119: "AFL", 162: "CS", 125: "LMB"}

DDL = """
CREATE TABLE IF NOT EXISTS dim_league (
    league_id   INTEGER PRIMARY KEY,
    sport_id    INTEGER NOT NULL,
    level       VARCHAR(8) NOT NULL,
    league_name VARCHAR(80)
);
CREATE TABLE IF NOT EXISTS dim_team (
    team_id   INTEGER NOT NULL,
    season    INTEGER NOT NULL,
    team_name VARCHAR(80),
    league_id INTEGER,
    PRIMARY KEY (team_id, season)
);
CREATE TABLE IF NOT EXISTS dim_player (
    player_id        INTEGER PRIMARY KEY,
    full_name        VARCHAR(80),
    birth_date       DATE,
    birth_country    VARCHAR(40),
    bats             CHAR(1),
    throws           CHAR(1),
    primary_position VARCHAR(4),
    mlb_debut_date   DATE
);
CREATE TABLE IF NOT EXISTS fact_lidom_roster (
    snapshot_date DATE NOT NULL,
    season        INTEGER NOT NULL,
    team_id       INTEGER NOT NULL,
    player_id     INTEGER NOT NULL,
    position      VARCHAR(4),
    status        VARCHAR(4),
    jersey        VARCHAR(4),
    PRIMARY KEY (snapshot_date, team_id, player_id)
);
CREATE TABLE IF NOT EXISTS fact_batting_season (
    player_id INTEGER NOT NULL,
    season    INTEGER NOT NULL,
    league_id INTEGER NOT NULL,
    team_id   INTEGER NOT NULL,
    g INTEGER, pa INTEGER, ab INTEGER, h INTEGER, b2 INTEGER, b3 INTEGER,
    hr INTEGER, bb INTEGER, ibb INTEGER, so INTEGER, hbp INTEGER, sf INTEGER,
    sb INTEGER, cs INTEGER,
    PRIMARY KEY (player_id, season, league_id, team_id)
);
CREATE TABLE IF NOT EXISTS fact_pitching_season (
    player_id INTEGER NOT NULL,
    season    INTEGER NOT NULL,
    league_id INTEGER NOT NULL,
    team_id   INTEGER NOT NULL,
    g INTEGER, gs INTEGER, outs INTEGER, bf INTEGER, h INTEGER, hr INTEGER,
    bb INTEGER, so INTEGER, hbp INTEGER, er INTEGER, pitches INTEGER,
    PRIMARY KEY (player_id, season, league_id, team_id)
);
DROP VIEW IF EXISTS v_latest_roster;
CREATE VIEW v_latest_roster AS
SELECT r.* FROM fact_lidom_roster r
WHERE r.snapshot_date = (SELECT MAX(snapshot_date) FROM fact_lidom_roster);
DROP VIEW IF EXISTS v_lidom_player_seasons;
CREATE VIEW v_lidom_player_seasons AS
SELECT lr.team_id AS lidom_team_id, lt.team_name AS lidom_team, p.player_id,
       p.full_name, p.primary_position, s.role, s.season, l.level,
       l.league_name, t.team_name, s.pa, s.bf, s.outs
FROM v_latest_roster lr
JOIN dim_player p ON p.player_id = lr.player_id
JOIN (SELECT player_id, season, league_id, team_id, 'bat' AS role, pa,
             NULL AS bf, NULL AS outs FROM fact_batting_season
      UNION ALL
      SELECT player_id, season, league_id, team_id, 'pitch', NULL, bf, outs
      FROM fact_pitching_season) s ON s.player_id = lr.player_id
JOIN dim_league l ON l.league_id = s.league_id
LEFT JOIN dim_team t ON t.team_id = s.team_id AND t.season = s.season
LEFT JOIN dim_team lt ON lt.team_id = lr.team_id AND lt.season = lr.season;
"""


# ------------------------------------------------------------------ raw layer
def get_json(url: str, path: Path, refresh: bool, offline: bool) -> dict:
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    if offline:
        raise FileNotFoundError(f"{path} is not cached; run once without --offline")
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "baseball-portfolio"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    path.write_text(json.dumps(payload))
    time.sleep(0.3)
    return payload


def fetch_lidom_teams(season: int, refresh: bool, offline: bool) -> list[dict]:
    payload = get_json(f"{API}/teams?sportId={WINTER_SPORT_ID}&season={season}",
                       RAW / f"teams_win_{season}.json", refresh, offline)
    return [t for t in payload.get("teams", [])
            if t.get("league", {}).get("id") == LIDOM_LEAGUE_ID]


def fetch_roster(team_id: int, season: int, snapshot: str, offline: bool) -> dict:
    url = f"{API}/teams/{team_id}/roster?rosterType=fullRoster&season={season}"
    return get_json(url, RAW / "rosters" / snapshot / f"team_{team_id}.json",
                    refresh=False, offline=offline)


def fetch_season_stats(sport_id: int, season: int, group: str,
                       refresh: bool, offline: bool) -> list[dict]:
    """All splits for one sport, season and group, following pagination."""
    splits, offset = [], 0
    while True:
        url = (f"{API}/stats?stats=season&group={group}&sportId={sport_id}"
               f"&season={season}&gameType=R&playerPool=ALL&limit=5000&offset={offset}")
        path = RAW / "stats" / f"{group}_{sport_id}_{season}_{offset}.json"
        payload = get_json(url, path, refresh, offline)
        block = (payload.get("stats") or [{}])[0]
        page = block.get("splits", [])
        splits.extend(page)
        total = block.get("totalSplits", len(page))
        offset += len(page)
        if not page or offset >= total:
            return splits


def fetch_people(ids: list[int], snapshot: str, offline: bool) -> list[dict]:
    people = []
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        url = f"{API}/people?personIds={','.join(map(str, chunk))}"
        path = RAW / "people" / snapshot / f"people_{i}.json"
        people.extend(get_json(url, path, refresh=False, offline=offline).get("people", []))
    return people


# ----------------------------------------------------------------- core layer
def outs_from_ip(ip) -> int | None:
    """'6.2' innings means 6 innings and 2 outs, so 20 outs."""
    if ip in (None, ""):
        return None
    whole, _, part = str(ip).partition(".")
    return int(whole) * 3 + int(part or 0)


def batting_row(split: dict, season: int) -> tuple:
    s = split.get("stat", {})
    return (split["player"]["id"], season, split["league"]["id"], split["team"]["id"],
            s.get("gamesPlayed"), s.get("plateAppearances"), s.get("atBats"),
            s.get("hits"), s.get("doubles"), s.get("triples"), s.get("homeRuns"),
            s.get("baseOnBalls"), s.get("intentionalWalks"), s.get("strikeOuts"),
            s.get("hitByPitch"), s.get("sacFlies"), s.get("stolenBases"),
            s.get("caughtStealing"))


def pitching_row(split: dict, season: int) -> tuple:
    s = split.get("stat", {})
    return (split["player"]["id"], season, split["league"]["id"], split["team"]["id"],
            s.get("gamesPlayed"), s.get("gamesStarted"), outs_from_ip(s.get("inningsPitched")),
            s.get("battersFaced"), s.get("hits"), s.get("homeRuns"), s.get("baseOnBalls"),
            s.get("strikeOuts"), s.get("hitByPitch"), s.get("earnedRuns"),
            s.get("numberOfPitches"))


def usable(split: dict) -> bool:
    """Combined multi-team rows have no team; per-team rows carry the data."""
    return all(split.get(k, {}).get("id") for k in ("player", "team", "league"))


def player_row(person: dict) -> tuple:
    return (person["id"], person.get("fullName"), person.get("birthDate"),
            person.get("birthCountry"), person.get("batSide", {}).get("code"),
            person.get("pitchHand", {}).get("code"),
            person.get("primaryPosition", {}).get("abbreviation"),
            person.get("mlbDebutDate"))


def build(con: sqlite3.Connection, season: int, snapshot: str, teams: list[dict],
          rosters: dict[int, dict], people: list[dict],
          stats: dict[tuple, list[dict]]) -> dict:
    """Load one snapshot into the core tables. Returns load counts."""
    con.executescript(DDL)
    rostered = set()
    for team in teams:
        con.execute("INSERT OR REPLACE INTO dim_team VALUES (?,?,?,?)",
                    (team["id"], season, team.get("name"), LIDOM_LEAGUE_ID))
        for entry in rosters.get(team["id"], {}).get("roster", []):
            pid = entry["person"]["id"]
            rostered.add(pid)
            con.execute("INSERT OR REPLACE INTO fact_lidom_roster VALUES (?,?,?,?,?,?,?)",
                        (snapshot, season, team["id"], pid,
                         entry.get("position", {}).get("abbreviation"),
                         entry.get("status", {}).get("code"), entry.get("jerseyNumber")))
    con.executemany("INSERT OR REPLACE INTO dim_player VALUES (?,?,?,?,?,?,?,?)",
                    [player_row(p) for p in people])

    counts = {"rostered": len(rostered), "duplicates": 0}
    for (sport_id, stat_season, group), splits in stats.items():
        table = "fact_batting_season" if group == "hitting" else "fact_pitching_season"
        make = batting_row if group == "hitting" else pitching_row
        seen, rows = set(), []
        for split in splits:
            if not usable(split) or split["player"]["id"] not in rostered:
                continue
            key = (split["player"]["id"], split["league"]["id"], split["team"]["id"])
            if key in seen:
                counts["duplicates"] += 1
                continue
            seen.add(key)
            rows.append(make(split, stat_season))
            league, team = split["league"], split["team"]
            con.execute("INSERT OR IGNORE INTO dim_league VALUES (?,?,?,?)",
                        (league["id"], sport_id,
                         LEAGUE_LEVELS.get(league["id"], SPORTS.get(sport_id, str(sport_id))),
                         league.get("name")))
            con.execute("INSERT OR IGNORE INTO dim_team VALUES (?,?,?,?)",
                        (team["id"], stat_season, team.get("name"), league["id"]))
        placeholders = ",".join("?" * len(rows[0])) if rows else ""
        if rows:
            con.executemany(f"INSERT OR REPLACE INTO {table} VALUES ({placeholders})", rows)
        counts[table] = counts.get(table, 0) + len(rows)
    con.commit()
    return counts


# -------------------------------------------------------------- quality checks
CHECKS = {
    "every LIDOM team has a roster in the latest snapshot":
        "SELECT COUNT(*) FROM dim_team t WHERE t.league_id = 131 AND t.season = "
        "(SELECT MAX(season) FROM v_latest_roster) AND t.team_id NOT IN "
        "(SELECT team_id FROM v_latest_roster)",
    "every rostered player is in dim_player":
        "SELECT COUNT(*) FROM v_latest_roster WHERE player_id NOT IN "
        "(SELECT player_id FROM dim_player)",
    "every stat row has a known league":
        "SELECT (SELECT COUNT(*) FROM fact_batting_season WHERE league_id NOT IN "
        "(SELECT league_id FROM dim_league)) + (SELECT COUNT(*) FROM fact_pitching_season "
        "WHERE league_id NOT IN (SELECT league_id FROM dim_league))",
    "no negative or impossible counts":
        "SELECT (SELECT COUNT(*) FROM fact_batting_season WHERE pa < 0 OR h > ab OR hr > h) "
        "+ (SELECT COUNT(*) FROM fact_pitching_season WHERE outs < 0 OR so > bf)",
}


def run_checks(con: sqlite3.Connection) -> dict[str, int]:
    """Each check returns the number of violating rows; 0 means it passed."""
    return {name: con.execute(sql).fetchone()[0] for name, sql in CHECKS.items()}


def coverage(con: sqlite3.Connection) -> list[tuple]:
    """Rostered players with any stats, by level and season."""
    return con.execute(
        "SELECT season, level, COUNT(DISTINCT player_id) FROM v_lidom_player_seasons "
        "GROUP BY season, level ORDER BY season, level").fetchall()


# ----------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--season", type=int, default=2026,
                        help="LIDOM season; 2026 is the 2026-27 winter")
    parser.add_argument("--history", type=int, default=3,
                        help="earlier seasons of stats to include")
    parser.add_argument("--snapshot", default=dt.date.today().isoformat(),
                        help="roster snapshot date to load (default today)")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    teams = fetch_lidom_teams(args.season, refresh=not args.offline, offline=args.offline)
    if not teams:
        raise RuntimeError("No LIDOM teams returned; check the API response.")
    rosters = {t["id"]: fetch_roster(t["id"], args.season, args.snapshot, args.offline)
               for t in teams}
    ids = sorted({e["person"]["id"] for r in rosters.values() for e in r.get("roster", [])})
    people = fetch_people(ids, args.snapshot, args.offline)

    stats = {}
    for season in range(args.season - args.history, args.season + 1):
        for sport_id in SPORTS:
            for group in GROUPS:
                stats[(sport_id, season, group)] = fetch_season_stats(
                    sport_id, season, group,
                    refresh=(season == args.season and not args.offline),
                    offline=args.offline)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        counts = build(con, args.season, args.snapshot, teams, rosters, people, stats)
        checks = run_checks(con)
        print(f"Snapshot {args.snapshot}: {len(teams)} LIDOM teams, {counts}")
        print("\nQuality checks (0 = pass)")
        for name, failures in checks.items():
            print(f"  {'PASS' if failures == 0 else 'FAIL'}  {name}: {failures}")
        print("\nRostered players with stats, by season and level")
        for season, level, players in coverage(con):
            print(f"  {season}  {level:<6} {players}")
        print(f"\nDatabase: {DB_PATH}")
    if any(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
