"""Tests for python/05_decisions_vs_damage.py — run with:  python -m pytest tests -q"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_path = Path(__file__).resolve().parents[1] / "python/05_decisions_vs_damage.py"
_spec = importlib.util.spec_from_file_location("dvd", _path)
dvd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dvd)


def test_parse_window():
    assert dvd.parse_window("2025") == ("2025-03-15", "2025-11-15", 2025)
    assert dvd.parse_window("2023H2") == ("2023-07-15", "2023-11-15", 2023)
    assert dvd.parse_window("2024-04-01:2024-06-30") == ("2024-04-01", "2024-06-30", 2024)
    with pytest.raises(ValueError):
        dvd.parse_window("last year")


def test_plate_speed_is_about_92_pct_of_release():
    d = dvd.synthetic_batter_season(1, 2025)
    ratio = (dvd.plate_speed_mph(d) / d.release_speed).median()
    assert 0.88 < ratio < 0.95


def _one_bbe(bat, pitch_plate, ev):
    """A single tracked batted ball with a known plate speed (no kinematic columns)."""
    return pd.DataFrame({
        "game_pk": [1], "game_date": ["2025-05-01"], "stand": ["L"],
        "description": ["hit_into_play"], "events": ["single"], "zone": [5],
        "release_speed": [pitch_plate / 0.92], "bat_speed": [bat], "launch_speed": [ev],
        "launch_angle": [15.0], "launch_speed_angle": [4], "bb_type": ["line_drive"],
        "estimated_woba_using_speedangle": [.6], "woba_value": [.9], "woba_denom": [1.0],
    })


def test_squared_up_and_collision_definitions():
    bat, pitch = 72.0, 85.0
    max_ev = 1.23 * bat + 0.23 * pitch                      # Savant collision model
    d = dvd.prepare(_one_bbe(bat, pitch, 0.81 * max_ev))
    assert d.squared_up.iat[0]
    assert d.ev_frac.iat[0] == pytest.approx(0.81)
    # collision efficiency q solves EV = q*pitch + (1+q)*bat
    q = d.collision_eff.iat[0]
    assert q * pitch + (1 + q) * bat == pytest.approx(0.81 * max_ev)
    d2 = dvd.prepare(_one_bbe(bat, pitch, 0.79 * max_ev))
    assert not d2.squared_up.iat[0]


def test_missing_tracking_is_reported_not_invented():
    d = dvd.prepare(dvd.synthetic_batter_season(1, 2023))   # pre-2024: no bat tracking
    m = dvd.metrics(d)
    assert np.isnan(m["bat_speed"]) and np.isnan(m["sq_up_bbe"])
    assert not np.isnan(m["xwoba"])


def test_demo_finds_planted_break_and_flags_trivial_bat_speed():
    # Fixed seed: with ~30 metrics a few chance flags are expected at some seeds
    # (each layer carries a 10% family-wise error budget), so pin one.
    a = dvd.prepare(dvd.synthetic_batter_season(1, 2024, seed=8))
    b = dvd.prepare(dvd.synthetic_batter_season(1, 2026, seed=8))
    ch = dvd.compare(a, b, reps=300, seed=1).set_index("metric")
    assert ch.loc["bat_speed", "verdict"] in {"stable", "trivial"}
    assert ch.loc["attack_angle", "verdict"] == "shifted"
    contact = ch.loc[["sq_up_bbe", "sq_up_swing", "collision_eff"], "verdict"]
    assert (contact == "worse").any()
    assert ch.loc["chase_pct", "verdict"] in {"stable", "trivial"}
    diag = dvd.diagnose(ch.reset_index())
    assert diag[-1].endswith("3 Delivery")
