"""Synthetic-data checks for the DSL time-to-debut module; no network needed."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
spec = importlib.util.spec_from_file_location(
    "dsl", ROOT / "python" / "06_dsl_time_to_debut.py")
dsl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsl)


def row(pid, season, debut=None, pa=100, so=20, bb=10, team="A"):
    return {"player_id": pid, "name": f"P{pid}", "birth_date": "2000-01-01",
            "mlb_debut_date": debut, "season": season, "team": team, "pa": pa,
            "ab": pa - bb, "h": 25, "b2": 5, "b3": 0, "hr": 2, "bb": bb, "so": so}


def test_parse_keeps_only_dsl():
    payload = {"stats": [{"splits": [
        {"league": {"id": 130}, "team": {"name": "DSL X"},
         "player": {"id": 1, "fullName": "A", "birthDate": "2000-01-01"},
         "stat": {"plateAppearances": 60, "strikeOuts": 9}},
        {"league": {"id": 121}, "team": {"name": "AZL Y"},
         "player": {"id": 2}, "stat": {"plateAppearances": 70}},
    ]}]}
    parsed = dsl.parse_splits(payload, 2015)
    assert parsed.player_id.tolist() == [1]
    assert parsed.so.iloc[0] == 9


def test_two_teams_are_summed_within_season():
    raw = pd.DataFrame([row(1, 2010, pa=60, team="A"), row(1, 2010, pa=40, team="B")])
    assert dsl.season_totals(raw).pa.tolist() == [100]


def test_only_first_season_features_are_used():
    raw = pd.DataFrame([row(9, 2005), row(1, 2006, so=10), row(1, 2007, so=60, pa=300)])
    cohort = dsl.build_cohort(raw, as_of=2020)
    assert cohort.season.tolist() == [2006]
    assert cohort.pa.tolist() == [100]


def test_earliest_season_is_dropped_for_left_truncation():
    raw = pd.DataFrame([row(1, 2005), row(2, 2006)])
    assert dsl.build_cohort(raw, as_of=2020).player_id.tolist() == [2]


def test_future_debut_is_censored_at_as_of():
    raw = pd.DataFrame([row(9, 2005), row(1, 2010, debut="2019-05-01")])
    cohort = dsl.build_cohort(raw, as_of=2013)
    assert cohort.event.tolist() == [0]
    assert cohort.duration.tolist() == [3.0]


def test_no_debut_is_censored_not_failed():
    raw = pd.DataFrame([row(9, 2005), row(1, 2015)])
    cohort = dsl.build_cohort(raw, as_of=2020)
    assert cohort.event.tolist() == [0]
    assert cohort.duration.tolist() == [5.0]


def test_league_relative_rates_ignore_environment_shift():
    low = [row(i, 2010, bb=5) for i in range(20)]
    high = [row(i + 100, 2011, bb=20) for i in range(20)]
    out = dsl.add_league_relative(dsl.season_totals(pd.DataFrame(low + high)))
    assert out.groupby("season").bb_rel.mean().round(6).tolist() == [1.0, 1.0]


def test_auc_matches_known_value():
    assert dsl.rank_auc(np.array([0, 0, 1, 1]), np.array([.1, .4, .35, .8])) == 0.75


def test_validation_recovers_planted_effects_out_of_time():
    pytest.importorskip("lifelines")
    raw = dsl.synthetic_raw(players=6000, last=2025)
    results = dsl.run_validation(raw, split_year=2013, as_of=2025, reps=50)
    summary = results["summary"].iloc[0]
    ratios = results["model"].summary["exp(coef)"]
    assert ratios["age_rel"] < 1
    assert ratios["k_rel"] < 1
    assert ratios["iso_rel"] > 1
    assert summary.auc_full > summary.auc_age_only
    assert summary.train_debuts > 0
