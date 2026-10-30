"""05_decisions_vs_damage.py — Decisions vs. Damage diagnostic (Statcast).

Inspired by Driveline Baseball's Kyle Tucker 2026 vs 2023 breakdown (Oct 2026):
"still swings at the right pitches, just isn't doing damage on them anymore."

The module reads a hitter through one causal chain and asks where it broke:

    1 Decisions  ->  2 Inputs  ->  3 Delivery  ->  4 Contact quality  ->  5 Outcomes
    (chase, z-swing)  (bat speed)  (attack angle,    (squared-up, collision   (EV, barrel,
                                    path tilt, depth)  efficiency, sweet spot)  xwOBA, GB%)

For two windows (e.g. 2023 vs 2026) every metric gets a game-level bootstrap
confidence interval on the change, so "bat speed is flat" and "barrels fell"
are statements with error bars, not eyeballing. The first layer in the chain
that moved meaningfully in the wrong direction is where the problem starts.

Outputs (output/tables, output/figures):
  dvd_<slug>_metrics.csv   every metric, every window
  dvd_<slug>_change.csv    base -> compare deltas with bootstrap CIs and flags
  dvd_<slug>_zones.csv     3x3 strike-zone grid: BBE, xwOBAcon, GB%, avg LA
  dvd_<slug>_reference.csv computed vs published reference values (if any)
  dvd_<slug>_chain.png     change by layer, with CIs
  dvd_<slug>_zones.png     zone heat maps (xwOBAcon, GB%), catcher's view

Run from the project root (needs internet to Baseball Savant):
    python python/05_decisions_vs_damage.py                       # Kyle Tucker case study
    python python/05_decisions_vs_damage.py --batter 665742 --name "Juan Soto" \
        --windows 2024 2025 2026 --base 2024 --compare 2026
    python python/05_decisions_vs_damage.py --demo                # synthetic, pipeline check only

Windows: "2025" (regular season), "2023H1" / "2023H2" (split at Jul 15),
or "2025-04-01:2025-06-30". Bat-tracking columns are public from 2024 and
richer from 2025 (attack angle, direction, path tilt, intercept). Earlier
windows report those metrics as missing; nothing is imputed.

Driveline's grades (Swing Decision, Hitting, Contact, Power) are proprietary
models and are NOT replicated. The public metrics here are the inputs those
grades are built on.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
TABLES = ROOT / "output/tables"
FIGS = ROOT / "output/figures"

SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
         "hit_into_play"}
BUNT = {"foul_bunt", "missed_bunt", "bunt_foul_tip"}
WHIFF = {"swinging_strike", "swinging_strike_blocked"}
WALKS = {"walk", "intent_walk"}
STRIKEOUTS = {"strikeout", "strikeout_double_play"}

# Collision model used by Savant's squared-up / blast definitions:
#   max possible EV = 1.23 * bat speed + 0.23 * pitch speed  (both mph)
COLL_BAT, COLL_PITCH = 1.23, 0.23
MIN_TRACKED = 200          # tracked competitive swings needed to compare bat-tracking metrics

# --- Metric catalogue: key, label, layer, kind, better ---------------------
# kind: "pct" (fraction shown as %), "num" (raw), "rate3" (.xxx)
# better: +1 higher is better, -1 lower is better, 0 descriptive (no value judgement)
METRICS = [
    ("swing_pct",       "Swing%",                "1 Decisions",        "pct",   0),
    ("z_swing_pct",     "Zone swing%",           "1 Decisions",        "pct",  +1),
    ("chase_pct",       "Chase%",                "1 Decisions",        "pct",  -1),
    ("zone_chase_gap",  "Z-swing minus chase",   "1 Decisions",        "pct",  +1),
    ("bb_pct",          "BB%",                   "1 Decisions",        "pct",  +1),
    ("bat_speed",       "Bat speed (mph)",       "2 Inputs",           "num",  +1),
    ("fast_swing_pct",  "Fast-swing% (75+)",     "2 Inputs",           "pct",  +1),
    ("swing_length",    "Swing length (ft)",     "2 Inputs",           "num",   0),
    ("attack_angle",    "Attack angle (deg)",    "3 Delivery",         "num",   0),
    ("ideal_aa_pct",    "Ideal attack angle%",   "3 Delivery",         "pct",  +1),
    ("swing_path_tilt", "Swing path tilt (deg)", "3 Delivery",         "num",   0),
    ("attack_dir_pull", "Attack dir. (+pull)",   "3 Delivery",         "num",   0),
    ("contact_depth",   "Contact depth (+front)", "3 Delivery",        "num",   0),
    ("z_contact_pct",   "Zone contact%",         "4 Contact quality",  "pct",  +1),
    ("whiff_pct",       "Whiff%",                "4 Contact quality",  "pct",  -1),
    ("sq_up_bbe",       "Squared-up% (BBE)",     "4 Contact quality",  "pct",  +1),
    ("sq_up_swing",     "Squared-up% (swings)",  "4 Contact quality",  "pct",  +1),
    ("collision_eff",   "Collision efficiency",  "4 Contact quality",  "rate3", +1),
    ("blast_bbe",       "Blast% (BBE)",          "4 Contact quality",  "pct",  +1),
    ("sweet_spot_pct",  "LA sweet-spot%",        "4 Contact quality",  "pct",  +1),
    ("ev",              "Avg EV (mph)",          "5 Outcomes",         "num",  +1),
    ("hard_hit_pct",    "HardHit%",              "5 Outcomes",         "pct",  +1),
    ("barrel_pct",      "Barrel% (BBE)",         "5 Outcomes",         "pct",  +1),
    ("gb_pct",          "GB%",                   "5 Outcomes",         "pct",  -1),
    ("launch_angle",    "Avg launch angle",      "5 Outcomes",         "num",   0),
    ("k_pct",           "K%",                    "5 Outcomes",         "pct",  -1),
    ("woba",            "wOBA",                  "5 Outcomes",         "rate3", +1),
    ("xwoba",           "xwOBA",                 "5 Outcomes",         "rate3", +1),
    ("xwobacon",        "xwOBAcon",              "5 Outcomes",         "rate3", +1),
]
META = {m[0]: dict(label=m[1], layer=m[2], kind=m[3], better=m[4]) for m in METRICS}
# Each layer gets a 10% family-wise error budget (Bonferroni): with k metrics in a
# layer, each metric's CI is two-sided at level 1 - 0.10/k. Without this, ~30 metrics
# at 90% would flag a few "changes" by chance in every comparison.
FAMILY_ALPHA = 0.10
# Smallest change worth acting on. A change needs BOTH a CI that excludes zero
# AND |delta| >= this to count; with thousands of swings, tiny drifts (e.g. -0.5 mph
# bat speed) are "statistically significant" but practically meaningless.
MIN_EFFECT = dict(
    swing_pct=.02, z_swing_pct=.02, chase_pct=.02, zone_chase_gap=.03, bb_pct=.015,
    bat_speed=1.0, fast_swing_pct=.05, swing_length=.15,
    attack_angle=2.0, ideal_aa_pct=.05, swing_path_tilt=2.0, attack_dir_pull=2.0,
    contact_depth=1.0,
    z_contact_pct=.02, whiff_pct=.02, sq_up_bbe=.03, sq_up_swing=.02,
    collision_eff=.010, blast_bbe=.02, sweet_spot_pct=.03,
    ev=1.0, hard_hit_pct=.03, barrel_pct=.02, gb_pct=.03, launch_angle=2.0,
    k_pct=.02, woba=.015, xwoba=.015, xwobacon=.020,
)
LAYER_ORDER = ["1 Decisions", "2 Inputs", "3 Delivery", "4 Contact quality", "5 Outcomes"]

# Published reference values for the case study: Driveline Baseball's Tucker post
# (Instagram carousel, Oct 2026; Savant-style percentile chart). Driveline's
# 2023 bat-tracking numbers come from 2023 H2 tracking, which may not be in the
# public pitch CSV - those rows will show as missing rather than be invented.
REFERENCE = {
    663656: {"source": "Driveline Baseball, Kyle Tucker 2026 vs 2023 (Oct 2026)",
             "values": {
                 "2023": dict(xwoba=.386, ev=90.2, barrel_pct=.106, hard_hit_pct=.459,
                              sweet_spot_pct=.357, bat_speed=71.9, sq_up_swing=.299,
                              chase_pct=.216, whiff_pct=.185, k_pct=.136, bb_pct=.119),
                 "2026": dict(xwoba=.331, ev=88.5, barrel_pct=.064, hard_hit_pct=.363,
                              sweet_spot_pct=.347, bat_speed=71.4, sq_up_swing=.270,
                              chase_pct=.239, whiff_pct=.208, k_pct=.167, bb_pct=.124)}},
}


# --- 1. Windows & data ---------------------------------------------------
def parse_window(w: str) -> tuple[str, str, int]:
    """Return (start, end, season) for '2025', '2025H1', '2025H2' or 'a:b'."""
    if m := re.fullmatch(r"(\d{4})", w):
        y = int(m[1]); return f"{y}-03-15", f"{y}-11-15", y
    if m := re.fullmatch(r"(\d{4})H([12])", w):
        y, h = int(m[1]), m[2]
        return (f"{y}-03-15", f"{y}-07-14", y) if h == "1" else (f"{y}-07-15", f"{y}-11-15", y)
    if m := re.fullmatch(r"(\d{4}-\d{2}-\d{2}):(\d{4}-\d{2}-\d{2})", w):
        return m[1], m[2], int(m[1][:4])
    raise ValueError(f"Bad window '{w}': use 2025, 2025H1, 2025H2 or YYYY-MM-DD:YYYY-MM-DD")


def load_batter_season(batter: int, season: int) -> pd.DataFrame:
    """One hitter's regular-season pitches, cached to data/raw/ (gitignored)."""
    cache = RAW / f"statcast_batter_{batter}_{season}.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    from pybaseball import statcast_batter, cache as pb_cache  # optional dependency
    pb_cache.enable()
    df = statcast_batter(f"{season}-03-15", f"{season}-11-15", player_id=batter)
    df = df[df["game_type"] == "R"]
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, compression="gzip")
    return df


def load_windows(batter: int, windows: list[str], loader) -> dict[str, pd.DataFrame]:
    seasons = {}
    out = {}
    for w in windows:
        start, end, yr = parse_window(w)
        if yr not in seasons:
            seasons[yr] = loader(batter, yr)
        s = seasons[yr]
        dates = pd.to_datetime(s["game_date"])
        out[w] = s[(dates >= start) & (dates <= end)].copy()
    return out


# --- 2. Synthetic data (pipeline check only) -----------------------------
def synthetic_batter_season(batter: int, season: int, seed: int = 7) -> pd.DataFrame:
    """Fake pitches with real Statcast column names. Numbers mean nothing.

    Seasons <= 2024 are built as the 'good' version (steeper path, efficient
    collisions); later seasons keep the same bat speed and decisions but get a
    flatter attack angle, lower collision efficiency and more low-zone grounders,
    so the diagnostic has a planted answer to find: Delivery -> Contact -> Outcomes.
    Bat-tracking columns are blank before 2024, as in the public data.
    """
    rng = np.random.default_rng(seed + season)
    good = season <= 2024
    n_games, per_game = 150, 26
    n = n_games * per_game
    game_pk = np.repeat(np.arange(n_games) + season * 1000, per_game)
    start = pd.Timestamp(f"{season}-03-28")
    game_date = (start + pd.to_timedelta(np.repeat(np.arange(n_games) * 1.2, per_game), "D")).strftime("%Y-%m-%d")

    zone = np.where(rng.random(n) < 0.48, rng.integers(1, 10, n), rng.integers(11, 15, n))
    in_z = zone <= 9
    row = np.where(in_z, (zone - 1) // 3, np.where(np.isin(zone, [11, 12]), 0, 2))   # 0 top .. 2 low
    sz_bot, sz_top = 1.6, 3.4
    plate_z = sz_top - (row + rng.random(n)) * (sz_top - sz_bot) / 3 + np.where(in_z, 0, np.where(row == 0, .4, -.4))
    rel_speed = rng.normal(89.5, 5, n)
    vy0 = -rel_speed * 1.4667 * 0.995
    balls = rng.choice(4, n, p=[.40, .30, .20, .10])
    strikes = rng.choice(3, n, p=[.38, .34, .28])

    p_swing = np.where(in_z, 0.70, 0.22)
    swung = rng.random(n) < p_swing
    p_contact = np.where(in_z, 0.86 if good else 0.84, 0.60)
    contact = swung & (rng.random(n) < p_contact)
    bip = contact & (rng.random(n) < 0.52)
    desc = np.where(~swung, np.where(in_z & (rng.random(n) < .9), "called_strike", "ball"),
            np.where(~contact, "swinging_strike", np.where(bip, "hit_into_play", "foul")))

    tracked = season >= 2024
    bat = rng.normal(71.8 if good else 71.4, 3.0, n)
    aa = rng.normal(12.0 if good else 8.0, 7.0, n)
    depth = rng.normal(1.0 if good else -0.8, 3.0, n)                 # + = out front
    plate_speed = rel_speed * 0.915
    max_ev = COLL_BAT * bat + COLL_PITCH * plate_speed
    frac = np.clip(rng.beta(9 if good else 7.5, 2.0, n), .35, 1.0)
    ev = frac * max_ev
    low_pen = np.where(row == 2, -4.0 if good else -11.0, np.where(row == 0, 4.0, 0.0))
    la = aa + low_pen + rng.normal(0, 15, n) - 0.6 * np.maximum(-depth, 0) * (not good)
    barrel = (ev >= 98) & (la >= 26 - (ev - 98)) & (la <= 30 + 1.2 * (ev - 98)) & (la >= 8) & (la <= 50)
    bb_type = np.select([la < 10, la < 25, la < 50], ["ground_ball", "line_drive", "fly_ball"], "popup")
    sweet = (la >= 8) & (la <= 32)
    xw = np.clip(0.05 + 0.012 * np.maximum(ev - 70, 0) * sweet + 0.9 * barrel - 0.12 * (la < 0), 0, 2.0)

    hit_val = np.where(rng.random(n) < xw / 1.4, rng.choice([.89, 1.27, 1.62, 2.1], n, p=[.6, .22, .02, .16]), 0)
    events = np.full(n, None, dtype=object)
    events[bip] = np.where(hit_val[bip] > 0, "single", "field_out")
    k_end = (desc == "swinging_strike") & (strikes == 2)
    bb_end = (desc == "ball") & (balls == 3) & (rng.random(n) < .75)
    events[k_end] = "strikeout"
    events[bb_end] = "walk"
    pa_end = pd.notna(events)
    woba_value = np.where(bip, hit_val, np.where(bb_end, 0.69, 0.0))

    df = pd.DataFrame({
        "game_pk": game_pk, "game_date": game_date, "game_year": season, "game_type": "R",
        "batter": batter, "player_name": "Demo, Hitter", "stand": "L",
        "description": desc, "events": events, "zone": zone, "plate_z": plate_z,
        "sz_top": sz_top, "sz_bot": sz_bot, "balls": balls, "strikes": strikes,
        "release_speed": rel_speed, "vx0": rng.normal(-5, 3, n), "vy0": vy0,
        "vz0": rng.normal(-4, 2, n), "ax": rng.normal(-8, 6, n), "ay": rng.normal(28, 3, n),
        "az": rng.normal(-22, 6, n),
        "bat_speed": np.where(swung & tracked, bat, np.nan),
        "swing_length": np.where(swung & tracked, rng.normal(7.3, .4, n), np.nan),
        "attack_angle": np.where(swung & tracked, aa, np.nan),
        "attack_direction": np.where(swung & tracked, rng.normal(0, 6, n) - 0.9 * depth, np.nan),
        "swing_path_tilt": np.where(swung & tracked, rng.normal(32 if good else 29, 4, n), np.nan),
        "intercept_ball_minus_batter_pos_y_inches": np.where(contact & tracked, 30 + depth, np.nan),
        "intercept_ball_minus_batter_pos_x_inches": np.where(contact & tracked, rng.normal(0, 3, n), np.nan),
        "launch_speed": np.where(bip, ev, np.nan), "launch_angle": np.where(bip, la, np.nan),
        "launch_speed_angle": np.where(bip, np.where(barrel, 6, 3), np.nan),
        "bb_type": np.where(bip, bb_type, None),
        "estimated_woba_using_speedangle": np.where(bip, xw, np.nan),
        "woba_value": np.where(pa_end, woba_value, np.nan),
        "woba_denom": np.where(pa_end, 1.0, np.nan),
        "hc_x": np.where(bip, 125.42 + 120 * np.sin(np.radians(rng.normal(0, 25, n) + 2 * depth)), np.nan),
        "hc_y": np.where(bip, 198.27 - 120 * np.cos(np.radians(rng.normal(0, 25, n))), np.nan),
    })
    return df


# --- 3. Feature engineering ---------------------------------------------
def plate_speed_mph(d: pd.DataFrame) -> pd.Series:
    """Pitch speed at the front of the plate (y = 17/12 ft) from the 9-parameter fit.

    Falls back to 0.92 x release speed when the kinematic columns are missing.
    """
    need = {"vx0", "vy0", "vz0", "ax", "ay", "az"}
    if not need <= set(d.columns):
        return 0.92 * d["release_speed"]
    y0, yf = 50.0, 17 / 12
    vyf = -np.sqrt(np.maximum(d.vy0 ** 2 - 2 * d.ay * (y0 - yf), 0))
    t = (vyf - d.vy0) / d.ay
    vxf, vzf = d.vx0 + d.ax * t, d.vz0 + d.az * t
    s = np.sqrt(vxf ** 2 + vyf ** 2 + vzf ** 2) * 0.681818
    return s.where(s.between(40, 110), 0.92 * d["release_speed"])


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for c in ["bat_speed", "swing_length", "attack_angle", "attack_direction",
              "swing_path_tilt", "intercept_ball_minus_batter_pos_y_inches",
              "launch_speed", "launch_angle", "launch_speed_angle",
              "estimated_woba_using_speedangle", "woba_value", "woba_denom",
              "hc_x", "hc_y", "zone", "plate_z"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["bunt"] = d.description.isin(BUNT)
    d["swing"] = d.description.isin(SWING)
    d["whiff"] = d.description.isin(WHIFF)
    d["contact"] = d.swing & ~d.whiff
    d["in_zone"] = d.zone.between(1, 9)
    d["out_zone"] = d.zone.between(11, 14)
    d["bbe"] = (d.description == "hit_into_play") & d.launch_speed.notna()
    d["pa_end"] = d.events.notna() & d.woba_denom.notna()
    d["plate_speed"] = plate_speed_mph(d)

    # Competitive swings (approximation of Savant's rule): no bunts, and drop the
    # hitter's slowest 10% of swings in the window unless they produced 90+ mph EV.
    d["competitive"] = False
    sw = d.swing & d.bat_speed.notna()
    if sw.any():
        p10 = d.loc[sw, "bat_speed"].quantile(0.10)
        d.loc[sw, "competitive"] = (d.loc[sw, "bat_speed"] >= p10) | (d.loc[sw, "launch_speed"] >= 90)

    max_ev = COLL_BAT * d.bat_speed + COLL_PITCH * d.plate_speed
    d["ev_frac"] = d.launch_speed / max_ev                      # share of max possible EV
    d["squared_up"] = d.ev_frac >= 0.80
    d["blast"] = (100 * d.ev_frac + d.bat_speed) >= 164
    d["collision_eff"] = (d.launch_speed - d.bat_speed) / (d.plate_speed + d.bat_speed)

    # Orientation: + depth = out front, + attack direction = pull side, set from
    # the correlation with pulled spray on this hitter's own batted balls.
    ang = np.degrees(np.arctan2(d.hc_x - 125.42, 198.27 - d.hc_y))
    d["spray_pull"] = np.where(d.stand == "R", -ang, ang)
    for col, new in [("intercept_ball_minus_batter_pos_y_inches", "contact_depth"),
                     ("attack_direction", "attack_dir_pull")]:
        sub = d.dropna(subset=[col, "spray_pull"])
        if len(sub) < 100:
            d[new] = np.nan
            continue
        sign = 1 if np.corrcoef(sub[col], sub.spray_pull)[0, 1] >= 0 else -1
        d[new] = sign * d[col]
    return d


# --- 4. Metrics ----------------------------------------------------------
def _ratio(num, den):
    return num / den if den else np.nan


def metrics(d: pd.DataFrame) -> dict:
    sw = d[d.swing & ~d.bunt]
    comp = sw[sw.competitive]
    bbe = d[d.bbe]
    bbe_t = bbe[bbe.bat_speed.notna()]              # batted balls with bat tracking
    pa = d[d.pa_end]
    z, o = d[d.in_zone], d[d.out_zone]
    z_sw = z[z.swing]
    x_num = np.where(pa.description == "hit_into_play",
                     pa.estimated_woba_using_speedangle.fillna(pa.woba_value), pa.woba_value)
    out = {
        "pitches": len(d), "pa": len(pa), "swings": len(sw), "bbe": len(bbe),
        "tracked_swings": int(comp.bat_speed.notna().sum()),
        "swing_pct": _ratio(len(sw), len(d)),
        "z_swing_pct": z.swing.mean() if len(z) else np.nan,
        "chase_pct": o.swing.mean() if len(o) else np.nan,
        "bb_pct": _ratio(pa.events.isin(WALKS).sum(), len(pa)),
        "k_pct": _ratio(pa.events.isin(STRIKEOUTS).sum(), len(pa)),
        "bat_speed": comp.bat_speed.mean(),
        "fast_swing_pct": (comp.bat_speed >= 75).mean() if comp.bat_speed.notna().any() else np.nan,
        "swing_length": comp.swing_length.mean(),
        "attack_angle": comp.attack_angle.mean(),
        "ideal_aa_pct": comp.attack_angle.between(5, 20).mean() if comp.attack_angle.notna().any() else np.nan,
        "swing_path_tilt": comp.swing_path_tilt.mean(),
        "attack_dir_pull": comp.attack_dir_pull.mean(),
        "contact_depth": d.contact_depth.mean(),
        "z_contact_pct": z_sw.contact.mean() if len(z_sw) else np.nan,
        "whiff_pct": _ratio(sw.whiff.sum(), len(sw)),
        "sq_up_bbe": bbe_t.squared_up.mean() if len(bbe_t) else np.nan,
        "sq_up_swing": _ratio(bbe_t.squared_up.sum(), len(comp)) if len(bbe_t) else np.nan,
        "collision_eff": bbe_t.collision_eff.mean() if len(bbe_t) else np.nan,
        "blast_bbe": bbe_t.blast.mean() if len(bbe_t) else np.nan,
        "sweet_spot_pct": bbe.launch_angle.between(8, 32).mean() if len(bbe) else np.nan,
        "ev": bbe.launch_speed.mean(),
        "hard_hit_pct": (bbe.launch_speed >= 95).mean() if len(bbe) else np.nan,
        "barrel_pct": (bbe.launch_speed_angle == 6).mean() if len(bbe) else np.nan,
        "gb_pct": (bbe.bb_type == "ground_ball").mean() if len(bbe) else np.nan,
        "launch_angle": bbe.launch_angle.mean(),
        "woba": _ratio(pa.woba_value.sum(), pa.woba_denom.sum()),
        "xwoba": _ratio(np.nansum(x_num), pa.woba_denom.sum()),
        "xwobacon": bbe.estimated_woba_using_speedangle.mean(),
    }
    out["zone_chase_gap"] = out["z_swing_pct"] - out["chase_pct"]
    return out


def bootstrap(d: pd.DataFrame, reps: int, rng: np.random.Generator) -> pd.DataFrame:
    """Game-level (cluster) bootstrap: resample whole games with replacement."""
    groups = list(d.groupby("game_pk").indices.values())     # positional indices
    k = len(groups)
    rows = []
    for _ in range(reps):
        pick = rng.integers(0, k, k)
        idx = np.concatenate([groups[i] for i in pick])
        rows.append(metrics(d.iloc[idx]))
    return pd.DataFrame(rows)


def compare(base: pd.DataFrame, comp: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a, b = metrics(base), metrics(comp)
    ba, bb = bootstrap(base, reps, rng), bootstrap(comp, reps, rng)
    rows = []
    k_layer = pd.Series([m["layer"] for m in META.values()]).value_counts()
    for key, m in META.items():
        delta = b[key] - a[key]
        diffs = (bb[key] - ba[key]).dropna()
        tail = FAMILY_ALPHA / k_layer[m["layer"]] / 2
        lo, hi = ((diffs.quantile(tail), diffs.quantile(1 - tail))
                  if len(diffs) > reps * .5 else (np.nan, np.nan))
        sig = bool(pd.notna(lo) and (lo > 0 or hi < 0))
        if pd.isna(delta):
            verdict = "missing"
        elif not sig:
            verdict = "stable"
        elif abs(delta) < MIN_EFFECT[key]:
            verdict = "trivial"           # real but too small to matter
        elif m["better"] == 0:
            verdict = "shifted"
        else:
            verdict = "better" if np.sign(delta) == m["better"] else "worse"
        rows.append(dict(metric=key, label=m["label"], layer=m["layer"],
                         base=a[key], compare=b[key], delta=delta,
                         ci_level=1 - 2 * tail, ci_lo=lo, ci_hi=hi, min_effect=MIN_EFFECT[key],
                         verdict=verdict))
    return pd.DataFrame(rows)


def diagnose(ch: pd.DataFrame) -> list[str]:
    """Walk the chain in order; report each layer and where the break starts."""
    lines, first = [], None
    for layer in LAYER_ORDER:
        sub = ch[ch.layer == layer]
        worse = sub[sub.verdict == "worse"].label.tolist()
        better = sub[sub.verdict == "better"].label.tolist()
        shifted = sub[sub.verdict == "shifted"].label.tolist()
        missing = sub[sub.verdict == "missing"].label.tolist()
        if len(missing) == len(sub):
            state = "no data (bat tracking not public for this window)"
        elif worse or shifted:
            state = "; ".join(x for x in [
                "WORSE: " + ", ".join(worse) if worse else "",
                "SHIFTED: " + ", ".join(shifted) if shifted else ""] if x)
        else:
            trivial = sub[sub.verdict == "trivial"].label.tolist()
            state = "stable" + (f" (better: {', '.join(better)})" if better else "") \
                + (f" (real but trivial: {', '.join(trivial)})" if trivial else "")
        lines.append(f"{layer:<18} {state}")
        if first is None and layer != "5 Outcomes" and (worse or shifted):
            first = layer
    lines.append("")
    lines.append(f"Chain breaks first at: {first}" if first else
                 "No upstream layer moved significantly; outcome change may be noise or luck.")
    return lines


# --- 5. Zone grid (replicates the heat-map slide) ------------------------
def zone_grid(d: pd.DataFrame) -> pd.DataFrame:
    """Statcast zones 1-9: row 0 = top, col 0 = catcher's left (third-base side)."""
    bbe = d[d.bbe & d.zone.between(1, 9)].copy()
    bbe["zrow"], bbe["zcol"] = (bbe.zone - 1) // 3, (bbe.zone - 1) % 3
    g = bbe.groupby(["zrow", "zcol"])
    out = pd.DataFrame({"bbe": g.size(),
                        "xwobacon": g.estimated_woba_using_speedangle.mean(),
                        "gb_pct": g.bb_type.apply(lambda s: (s == "ground_ball").mean()),
                        "launch_angle": g.launch_angle.mean(),
                        "ev": g.launch_speed.mean()})
    full = pd.MultiIndex.from_product([range(3), range(3)], names=["zrow", "zcol"])
    return out.reindex(full).reset_index()


# --- 6. Figures ----------------------------------------------------------
def _fmt(v, kind):
    if pd.isna(v):
        return "–"
    return f"{100 * v:.1f}%" if kind == "pct" else (f"{v:.3f}" if kind == "rate3" else f"{v:.2f}")


def plot_chain(ch: pd.DataFrame, title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = ch[ch.verdict != "missing"].copy()
    if c.empty:
        return
    # standardise each delta by its own CI half-width so layers share one axis
    half = ((c.ci_hi - c.ci_lo) / 2).replace(0, np.nan)
    sign = c.metric.map(lambda k: META[k]["better"] or 1)
    c["z"] = sign * c.delta / half                    # + = moved in the "better" direction
    c["z_lo"], c["z_hi"] = c.z - 1, c.z + 1
    colors = {"better": "#1a7f37", "worse": "#d1242f", "stable": "#8c959f",
              "trivial": "#c4c9cf", "shifted": "#9a6700"}
    c = c.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.33 * len(c) + 1.6))
    ax.barh(c.index, c.z, color=[colors[v] for v in c.verdict], height=.6)
    ax.errorbar(c.z, c.index, xerr=1, fmt="none", ecolor="#24292f", lw=.8, capsize=2)
    ax.axvline(0, c="#24292f", lw=.8)
    ax.set_yticks(c.index)
    ax.set_yticklabels([f"{r.label}  ({_fmt(r.base, META[r.metric]['kind'])} → "
                        f"{_fmt(r.compare, META[r.metric]['kind'])})" for r in c.itertuples()], fontsize=8)
    prev = None
    for i, r in c.iterrows():
        if prev is not None and r.layer != prev:
            ax.axhline(i - .5, c="#d0d7de", lw=.8)
        prev = r.layer
    ax.set_xlabel("Change in CI half-widths  (right = better; descriptive metrics: right = higher)")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_zones(grids: dict[str, pd.DataFrame], title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wins = list(grids)
    fig, axes = plt.subplots(len(wins), 2, figsize=(7, 3.1 * len(wins)), squeeze=False)
    specs = [("xwobacon", "xwOBAcon", "RdYlBu_r", (.200, .600)),
             ("gb_pct", "GB%", "RdYlBu_r", (0.0, 0.70))]
    for i, w in enumerate(wins):
        g = grids[w]
        for j, (col, lab, cmap, lim) in enumerate(specs):
            ax = axes[i, j]
            M = g.pivot(index="zrow", columns="zcol", values=col).to_numpy(dtype=float)
            N = g.pivot(index="zrow", columns="zcol", values="bbe").to_numpy(dtype=float)
            im = ax.imshow(M, cmap=cmap, vmin=lim[0], vmax=lim[1])
            for r in range(3):
                for cc in range(3):
                    if np.isnan(M[r, cc]):
                        continue
                    txt = f"{100 * M[r, cc]:.0f}%" if col == "gb_pct" else f"{M[r, cc]:.3f}"[1:]
                    ax.text(cc, r, f"{txt}\nn={int(N[r, cc])}", ha="center", va="center", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{w} · {lab}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle(title + "  (in-zone BBE, catcher's view)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- main ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batter", type=int, default=663656, help="MLBAM id (default Kyle Tucker)")
    ap.add_argument("--name", default="Kyle Tucker")
    ap.add_argument("--windows", nargs="+", default=["2023", "2024", "2025", "2026"])
    ap.add_argument("--base", default="2023")
    ap.add_argument("--compare", default="2026")
    ap.add_argument("--reps", type=int, default=1000, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--demo", action="store_true",
                    help="use synthetic data (pipeline check; results are meaningless)")
    args = ap.parse_args()
    for w in (args.base, args.compare):
        if w not in args.windows:
            args.windows.append(w)

    loader = synthetic_batter_season if args.demo else load_batter_season
    raw = load_windows(args.batter, args.windows, loader)
    data = {w: prepare(df) for w, df in raw.items() if len(df)}
    empty = [w for w in args.windows if w not in data]
    if empty:
        print("No pitches for windows:", empty)
    if args.base not in data or args.compare not in data:
        raise SystemExit("Base or compare window has no data.")

    slug = re.sub(r"[^a-z0-9]+", "_", args.name.lower()).strip("_") + ("_DEMO" if args.demo else "")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    m = pd.DataFrame({w: metrics(d) for w, d in data.items()})
    m.index.name = "metric"
    m.insert(0, "label", [META.get(k, {}).get("label", k) for k in m.index])
    m.insert(1, "layer", [META.get(k, {}).get("layer", "sample size") for k in m.index])
    m.to_csv(TABLES / f"dvd_{slug}_metrics.csv")

    ch = compare(data[args.base], data[args.compare], args.reps, args.seed)
    ch["base_window"] = args.base
    # Bat tracking is not public before 2024. If the base window has none, compare
    # the bat-tracking metrics from the earliest tracked window instead, and say so.
    track_base = None
    if m.loc["tracked_swings", args.base] < MIN_TRACKED:
        for w in args.windows:
            if w != args.compare and w in data and m.loc["tracked_swings", w] >= MIN_TRACKED:
                track_base = w
                break
    if track_base:
        ch2 = compare(data[track_base], data[args.compare], args.reps, args.seed + 1)
        miss = ch.verdict == "missing"
        repl = ch2.set_index("metric").loc[ch.loc[miss, "metric"]].reset_index()
        repl["base_window"] = track_base
        ch.loc[miss, repl.columns] = repl.to_numpy()
        print(f"Base {args.base} has no bat tracking; bat-tracking metrics compared "
              f"{track_base} → {args.compare} instead (see base_window column).")
    ch.to_csv(TABLES / f"dvd_{slug}_change.csv", index=False)

    grids = {w: zone_grid(data[w]) for w in (args.base, args.compare)}
    pd.concat([g.assign(window=w) for w, g in grids.items()]).to_csv(
        TABLES / f"dvd_{slug}_zones.csv", index=False)

    ref = REFERENCE.get(args.batter) if not args.demo else None
    if ref:
        rows = []
        for w, vals in ref["values"].items():
            for k, v in vals.items():
                got = m.loc[k, w] if w in m.columns else np.nan
                rows.append(dict(window=w, metric=k, label=META[k]["label"], published=v,
                                 computed=got, diff=got - v if pd.notna(got) else np.nan))
        pd.DataFrame(rows).to_csv(TABLES / f"dvd_{slug}_reference.csv", index=False)

    title = f"{args.name}: {args.base} → {args.compare}"
    if track_base:
        title += f" (bat tracking from {track_base})"
    if args.demo:
        title = "[DEMO, synthetic] " + title
    plot_chain(ch, f"Decisions vs. Damage — {title}", FIGS / f"dvd_{slug}_chain.png")
    plot_zones(grids, title, FIGS / f"dvd_{slug}_zones.png")

    # --- console report ---
    show = m.drop(columns=["layer"]).astype(object)
    for k in show.index:
        kind = META.get(k, {}).get("kind")
        for w in data:
            v = show.loc[k, w]
            show.loc[k, w] = _fmt(v, kind) if kind else (int(v) if pd.notna(v) else "–")
    print(f"\n{title}\n" + "=" * len(title))
    print(show.to_string())
    print(f"\nChange {args.base} → {args.compare} (game-bootstrap CIs, {args.reps} reps, {FAMILY_ALPHA:.0%} error budget per layer):")
    out = ch.copy()
    for col in ["base", "compare", "delta", "ci_lo", "ci_hi"]:
        out[col] = [_fmt(v, META[k]["kind"]) for v, k in zip(out[col], out.metric)]
    print(out[["layer", "label", "base_window", "base", "compare", "delta", "ci_lo", "ci_hi", "verdict"]]
          .to_string(index=False))
    print("\nDiagnosis:")
    print("\n".join("  " + s for s in diagnose(ch)))
    if ref:
        print(f"\nReference check vs {ref['source']}: see dvd_{slug}_reference.csv")


if __name__ == "__main__":
    main()
