"""Fixture-data checks for the LIDOM database; no network needed."""
import importlib.util
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lidom", ROOT / "python" / "08_lidom_database.py")
lidom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lidom)

TEAMS = [{"id": 672, "name": "Tigres del Licey", "league": {"id": 131}},
         {"id": 671, "name": "Leones del Escogido", "league": {"id": 131}}]
ROSTERS = {
    672: {"roster": [{"person": {"id": 1}, "position": {"abbreviation": "P"},
                      "status": {"code": "A"}, "jerseyNumber": "40"},
                     {"person": {"id": 2}, "position": {"abbreviation": "SS"},
                      "status": {"code": "A"}, "jerseyNumber": "2"}]},
    671: {"roster": [{"person": {"id": 3}, "position": {"abbreviation": "CF"},
                      "status": {"code": "A"}, "jerseyNumber": "8"}]},
}
PEOPLE = [{"id": i, "fullName": f"Player {i}", "birthDate": "1998-05-01",
           "birthCountry": "Dominican Republic", "batSide": {"code": "R"},
           "pitchHand": {"code": "R"}, "primaryPosition": {"abbreviation": "P"}}
          for i in (1, 2, 3)]


def split(pid, league, team, **stat):
    return {"player": {"id": pid}, "league": {"id": league, "name": f"L{league}"},
            "team": {"id": team, "name": f"T{team}"}, "stat": stat}


STATS = {
    (11, 2025, "hitting"): [
        split(2, 112, 500, plateAppearances=400, atBats=360, hits=100, homeRuns=12),
        split(2, 112, 500, plateAppearances=400, atBats=360, hits=100, homeRuns=12),
        split(99, 112, 500, plateAppearances=300, atBats=270, hits=80),
        {"player": {"id": 3}, "league": {"id": 112}, "team": {}, "stat": {}},
    ],
    (23, 2025, "hitting"): [split(3, 125, 600, plateAppearances=350, atBats=310, hits=95)],
    (17, 2026, "pitching"): [split(1, 131, 672, inningsPitched="6.2", battersFaced=27,
                                   strikeOuts=7)],
}


def built():
    con = sqlite3.connect(":memory:")
    counts = lidom.build(con, 2026, "2026-11-12", TEAMS, ROSTERS, PEOPLE, STATS)
    return con, counts


def test_innings_become_outs():
    assert lidom.outs_from_ip("6.2") == 20
    assert lidom.outs_from_ip("0.1") == 1
    assert lidom.outs_from_ip("9") == 27
    assert lidom.outs_from_ip(None) is None


def test_only_rostered_players_and_usable_rows_load():
    con, counts = built()
    ids = {r[0] for r in con.execute("SELECT player_id FROM fact_batting_season")}
    assert ids == {2, 3}
    assert counts["duplicates"] == 1
    assert counts["rostered"] == 3


def test_pitching_outs_stored():
    con, _ = built()
    assert con.execute("SELECT outs FROM fact_pitching_season").fetchone()[0] == 20


def test_view_links_roster_to_other_leagues():
    con, _ = built()
    rows = con.execute("SELECT full_name, lidom_team, level, season FROM "
                       "v_lidom_player_seasons ORDER BY full_name, season").fetchall()
    assert ("Player 2", "Tigres del Licey", "AAA", 2025) in rows
    assert ("Player 3", "Leones del Escogido", "LMB", 2025) in rows
    assert ("Player 1", "Tigres del Licey", "LIDOM", 2026) in rows


def test_rerun_adds_snapshot_without_duplicating():
    con, _ = built()
    lidom.build(con, 2026, "2026-11-19", TEAMS, ROSTERS, PEOPLE, STATS)
    snaps = con.execute("SELECT COUNT(DISTINCT snapshot_date) FROM fact_lidom_roster").fetchone()[0]
    rows = con.execute("SELECT COUNT(*) FROM fact_batting_season").fetchone()[0]
    latest = con.execute("SELECT COUNT(*) FROM v_latest_roster").fetchone()[0]
    assert (snaps, rows, latest) == (2, 2, 3)


def test_quality_checks_pass_and_catch_problems():
    con, _ = built()
    assert all(v == 0 for v in lidom.run_checks(con).values())
    con.execute("INSERT INTO fact_lidom_roster VALUES ('2026-11-12',2026,672,77,'P','A','1')")
    con.execute("UPDATE fact_batting_season SET h = ab + 1 WHERE player_id = 2")
    checks = lidom.run_checks(con)
    assert checks["every rostered player is in dim_player"] == 1
    assert checks["no negative or impossible counts"] == 1


def test_missing_team_roster_is_flagged():
    con = sqlite3.connect(":memory:")
    lidom.build(con, 2026, "2026-11-12", TEAMS, {672: ROSTERS[672]}, PEOPLE, STATS)
    assert lidom.run_checks(con)["every LIDOM team has a roster in the latest snapshot"] == 1
