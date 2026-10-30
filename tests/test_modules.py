"""Synthetic-data tests for modules 04 and 05.

Each test plants a known relationship in fake data and checks the module
recovers it, so the pipelines are verified without network access.
Run from the project root:  python -m pytest tests -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "python" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pc = _load("04_pitch_cost")
oc = _load("05_opencommand_validation")


# ----------------------------------------------------------- fake Statcast
def fake_statcast(n_hitters=80, n_pitchers=30, pa_per_hitter=550, seed=1):
    """Hitters' two-strike foul skill is driven by z-contact ability and short
    swings (planted), while bat speed is pure noise."""
    rng = np.random.default_rng(seed)
    hit = pd.DataFrame({
        "batter": np.arange(1000, 1000 + n_hitters),
        "contact_skill": rng.normal(0, 1, n_hitters),
        "swing_length": rng.normal(7.2, .35, n_hitters),
        "bat_speed": rng.normal(72, 2.5, n_hitters),
    })
    pit_ids = np.r_[pc.MISIOROWSKI_ID, np.arange(500000, 500000 + n_pitchers - 1)]
    fb_mean = dict(zip(pit_ids, np.r_[100.5, rng.uniform(94, 100, n_pitchers - 1)]))
    rows, pa_counter = [], {}
    for _, h in hit.iterrows():
        p_foul = np.clip(.40 + .07 * h.contact_skill - .10 * (h.swing_length - 7.2), .05, .9)
        p_contact_z = np.clip(.82 + .05 * h.contact_skill, .4, .99)
        for _ in range(pa_per_hitter):
            pit = rng.choice(pit_ids)
            game = int(rng.integers(1, 400))
            ab = pa_counter.get(game, 0) + 1
            pa_counter[game] = ab
            balls = strikes = 0
            pn = 0
            while True:
                pn += 1
                hard = pit == pc.MISIOROWSKI_ID
                pt = rng.choice(["FF", "SI", "SL", "CH"], p=[.45, .15, .25, .15])
                velo = rng.normal(fb_mean[pit], 1.2) if pt in {"FF", "SI"} \
                    else rng.normal(86, 3)
                zone = int(rng.integers(1, 15))
                swing = rng.random() < (.65 if zone <= 9 else .3)
                ev = None
                if not swing:
                    desc = "called_strike" if zone <= 9 else "ball"
                elif zone <= 9 and rng.random() > p_contact_z:
                    desc = "swinging_strike"
                elif strikes == 2 and rng.random() < p_foul:
                    desc = "foul"
                elif rng.random() < .55:
                    desc = "foul" if strikes < 2 else "hit_into_play"
                else:
                    desc = "hit_into_play" if rng.random() < .6 else "swinging_strike"
                if desc in {"called_strike", "swinging_strike"}:
                    if strikes == 2:
                        ev = "strikeout"
                elif desc == "ball" and balls == 3:
                    ev = "walk"
                elif desc == "hit_into_play":
                    ev = "field_out"
                rows.append(dict(
                    game_pk=game, game_date="2026-06-01", game_type="R",
                    at_bat_number=ab, pitch_number=pn, inning=1 + ab // 6,
                    pitcher=pit, batter=h.batter,
                    player_name="Misiorowski, Jacob" if hard else f"P{pit}",
                    pitch_type=pt, release_speed=velo, description=desc, events=ev,
                    balls=balls, strikes=strikes, zone=zone,
                    bat_speed=h.bat_speed + rng.normal(0, 3) if swing else np.nan,
                    swing_length=h.swing_length + rng.normal(0, .3) if swing else np.nan))
                if ev:
                    break
                if desc == "ball":
                    balls += 1
                elif desc in {"called_strike", "swinging_strike"} or (desc == "foul" and strikes < 2):
                    strikes += 1
    sc = pd.DataFrame(rows)
    # Misiorowski "NLDS G1": two long, foul-heavy PAs
    g = sc[sc.pitcher.eq(pc.MISIOROWSKI_ID)].head(14).copy()
    g["game_pk"], g["game_date"], g["game_type"] = 999999, "2026-10-03", "D"
    return pd.concat([sc, g], ignore_index=True), hit


@pytest.fixture(scope="module")
def sc_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    pc.TABLES, pc.FIGS = out / "tables", out / "figures"
    sc, hit = fake_statcast()
    return pc.run(sc, 2026), hit, out


def test_flags_foul_tip_is_not_foul():
    d = pc.add_flags(pd.DataFrame({
        "description": ["foul", "foul_tip", "swinging_strike", "ball"],
        "strikes": [2, 2, 1, 0], "zone": [5, 5, 12, 14], "pitch_type": ["FF"] * 4,
        "release_speed": [99, 97, 99, 99], "events": [None, "strikeout", None, None],
        "game_pk": 1, "at_bat_number": 1}))
    assert d.foul.tolist() == [True, False, False, False]
    assert d.swing.tolist() == [True, True, True, False]
    assert d.elite_velo.tolist() == [True, False, True, True]


def test_pitcher_cost_columns(sc_out):
    res, _, _ = sc_out
    p = res["pitcher_cost_2026"]
    assert {"pitches_per_pa", "putaway_pct", "ppa_over_expected"} <= set(p.columns)
    assert p.pitches_per_pa.between(2, 7).all()


def test_predictors_recover_planted_signal(sc_out):
    res, _, _ = sc_out
    pr = res["foul_skill_predictors_2026"].set_index("trait")
    # planted: z_contact helps (+), longer swings hurt (-), bat speed is noise
    assert pr.loc["z_contact", "pearson_r"] > .2
    assert pr.loc["swing_length", "std_beta"] < 0
    assert abs(pr.loc["bat_speed", "std_beta"]) < abs(pr.loc["z_contact", "std_beta"])


def test_reliability_and_case_study(sc_out):
    res, _, out = sc_out
    rel = res["foul_skill_reliability_2026"]
    assert rel.spearman_brown.notna().any()
    case = res["misiorowski_nlds_g1"]
    assert "NLDS G1" in case.inning.astype(str).values
    assert (out / "tables/pitcher_cost_2026.csv").exists()


# ---------------------------------------------------------- fake OpenCommand
def fake_command(n=300, seed=2):
    """Miss distance drives walks; xERA driven by stuff + walks only."""
    rng = np.random.default_rng(seed)
    def alpha(i):  # letter-only unique names (real names carry no digits)
        return "".join(chr(97 + int(c)) for c in str(i))
    names = [f"Pitcher {alpha(i)}" for i in range(n - 1)] + ["Jacob Misiorowski"]
    skill = rng.normal(0, 1, n)
    cmds, fgs = [], []
    for season in (2025, 2026):
        miss = 10 + .6 * skill + rng.normal(0, .35, n)
        miss[-1] = 13.5                                   # glove-ignorer: overstated miss
        stuff = rng.normal(100, 7, n)
        stuff[-1] = 130
        bb = np.clip(.08 + .012 * skill + rng.normal(0, .01, n), .02, .2)
        bb[-1] = .07
        xera = 4.2 - .06 * (stuff - 100) + 12 * (bb - .08) + rng.normal(0, .35, n)
        cmds.append(pd.DataFrame({"pitcher": names, "pitch_type": "ALL",
                                  "n": rng.integers(800, 3000, n),
                                  "naive_in": miss + 1 + rng.normal(0, .3, n),
                                  "inferred_in": miss, "season": season}))
        cmds.append(pd.DataFrame({"pitcher": names, "pitch_type": "FF", "n": 400,
                                  "naive_in": miss, "inferred_in": miss - .5, "season": season}))
        fgs.append(pd.DataFrame({"Name": names, "IP": rng.uniform(40, 200, n),
                                 "BB%": (bb * 100).round(1), "Stuff+": stuff,
                                 "xERA": xera, "season": season}))
    return pd.concat(cmds, ignore_index=True), pd.concat(fgs, ignore_index=True)


@pytest.fixture(scope="module")
def oc_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("oc")
    oc.TABLES, oc.FIGS = out / "tables", out / "figures"
    cmd, fg = fake_command()
    return oc.run(cmd, fg)


def test_name_normalization():
    assert oc.norm_name("José Ramírez Jr.") == "jose ramirez"


def test_bb_pct_rescaled(oc_out):
    assert oc_out["opencommand_panel"].bb_pct.max() < 1


def test_replicates_walk_correlation(oc_out):
    c = oc_out["opencommand_correlations"]
    r = c.query("season=='pooled' and miss=='inferred_in' and vs=='bb_pct'").r.item()
    assert r > .4


def test_command_adds_little_beyond_stuff_and_walks(oc_out):
    inc = oc_out["opencommand_incremental_r2"].set_index("model")
    gain = inc.loc["stuff + walks + command", "r2"] - inc.loc["stuff + walks", "r2"]
    assert gain < .02


def test_stability_and_glove_ignorer(oc_out):
    st = oc_out["opencommand_stability"]
    assert st.query("metric=='inferred_in'").r.item() > .5
    gi = oc_out["opencommand_glove_ignorers"]
    assert (gi.resid_z > 2).all() and len(gi) == 2
