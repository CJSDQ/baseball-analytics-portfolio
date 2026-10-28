"""04_decision_window.py — Decision Window profile (Statcast bat tracking).

Inspired by Chris Black's analysis of the 2026 Brewers: short, fast swings,
deep contact, opposite-field attack direction, and a "don't be early" timing
philosophy that accepts being late on fastballs to avoid being early on spin.

Three outputs:
  1. Hitter profiles  - bat speed vs swing length quadrant, attack angle and
                        direction, contact depth, early/late rates by pitch group,
                        performance vs breaking balls (before 2 strikes too).
  2. Team profiles    - same metrics rolled up by batting team and season.
  3. Breakeven model  - the trade-off only pays if pitchers throw enough spin.
                        Estimates the breaking-ball share at which "late on FB"
                        costs the same as "early on spin", and compares it with
                        the league's actual pitch mix by season.

Run from the project root (needs internet to Baseball Savant):
    python python/04_decision_window.py --seasons 2024 2025
    python python/04_decision_window.py --demo      # synthetic data, pipeline check only

Bat-tracking columns exist from 2024 (bat_speed, swing_length) and 2025
(attack_angle, attack_direction, swing_path_tilt, intercept_*). Missing columns
are reported and skipped, never invented.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
TABLES = ROOT / "output/tables"
FIGS = ROOT / "output/figures"

# --- Pitch groups --------------------------------------------------------
FB = {"FF", "SI", "FC"}
BRK = {"SL", "ST", "SV", "CU", "KC", "CS"}          # "spin" in Black's article
OFF = {"CH", "FS", "FO", "SC"}

SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
         "hit_into_play"}
WHIFF = {"swinging_strike", "swinging_strike_blocked"}

BAT_COLS = ["bat_speed", "swing_length", "attack_angle", "attack_direction",
            "swing_path_tilt", "intercept_ball_minus_batter_pos_x_inches",
            "intercept_ball_minus_batter_pos_y_inches"]

MIN_SWINGS = 200          # hitter qualification for profiles
MIN_CONTACT = 50          # tracked contacts needed for timing rates


# --- 1. Data -------------------------------------------------------------
def load_statcast(season: int) -> pd.DataFrame:
    """Regular-season pitch data, cached to data/raw/ (gitignored)."""
    cache = RAW / f"statcast_{season}.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    from pybaseball import statcast, cache as pb_cache  # local import: optional dep
    pb_cache.enable()
    df = statcast(f"{season}-03-15", f"{season}-11-15")
    df = df[df["game_type"] == "R"]
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, compression="gzip")
    return df


def synthetic_statcast(season: int, n_teams: int = 6, n_per_team: int = 6,
                       pitches: int = 900, seed: int = 0) -> pd.DataFrame:
    """Fake pitch data with the real column names, for pipeline testing only.

    One team ("MIL") is built with a short/fast, late-on-FB, not-early-on-spin
    profile so the outputs have something to find. Numbers mean nothing.
    """
    rng = np.random.default_rng(seed + season)
    teams = ["MIL"] + [f"T{i}" for i in range(1, n_teams)]
    rows = []
    pid = 100000
    for t in teams:
        brew = t == "MIL"
        for _ in range(n_per_team):
            pid += 1
            stand = rng.choice(["R", "L"])
            bs_mu = rng.normal(72.5 if brew else 71.5, 1.5)
            sl_mu = rng.normal(6.9 if brew else 7.3, 0.2)
            depth_mu = rng.normal(-1.5 if brew else 0.5, 1.0)   # + = out front
            n = pitches
            pt = rng.choice(["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"], n,
                            p=[.32, .15, .08, .17, .07, .08, .09, .04])
            is_fb = np.isin(pt, list(FB))
            # slower pitches -> contact further out front
            depth = rng.normal(depth_mu + np.where(is_fb, 0, 2.5), 3.0)
            swing_p = 0.43 if brew else 0.48
            swung = rng.random(n) < swing_p
            contact = swung & (rng.random(n) < 0.77)
            spray = rng.normal(0, 25, n) + 2.2 * depth        # out front -> pull
            ang = np.where(stand == "R", -spray, spray)       # raw field angle
            hc_x = 125.42 + 150 * np.sin(np.radians(ang))
            hc_y = 198.27 - 150 * np.cos(np.radians(ang))
            desc = np.where(~swung, "ball",
                    np.where(~contact, "swinging_strike",
                    np.where(rng.random(n) < 0.45, "foul", "hit_into_play")))
            early_spin = (~is_fb) & (depth > 4)
            late_fb = is_fb & (depth < -3)
            rv = (rng.normal(0, 0.12, n) - 0.05 * early_spin - 0.02 * late_fb
                  + np.where(desc == "swinging_strike", -0.06, 0))
            rows.append(pd.DataFrame({
                "game_year": season, "game_type": "R", "batter": pid,
                "player_name": f"Hitter {pid}", "stand": stand,
                "inning_topbot": "Bot", "home_team": t, "away_team": "OPP",
                "pitch_type": pt, "description": desc, "strikes": rng.integers(0, 3, n),
                "bat_speed": np.where(swung, rng.normal(bs_mu, 4, n), np.nan),
                "swing_length": np.where(swung, rng.normal(sl_mu, 0.5, n), np.nan),
                "attack_angle": np.where(swung, rng.normal(7 if brew else 10, 8, n), np.nan),
                "attack_direction": np.where(swung, rng.normal(0, 6, n) + 0.8 * depth, np.nan),
                "swing_path_tilt": np.where(swung, rng.normal(32, 4, n), np.nan),
                "intercept_ball_minus_batter_pos_y_inches": np.where(contact, depth + 30, np.nan),
                "intercept_ball_minus_batter_pos_x_inches": np.where(contact, rng.normal(0, 3, n), np.nan),
                "hc_x": np.where(desc == "hit_into_play", hc_x, np.nan),
                "hc_y": np.where(desc == "hit_into_play", hc_y, np.nan),
                "delta_run_exp": rv,
                "woba_value": np.where(desc == "hit_into_play", rng.choice([0, .9, 1.25, 2.0], n, p=[.68, .22, .07, .03]), np.nan),
                "woba_denom": np.where(desc == "hit_into_play", 1, np.nan),
            }))
    return pd.concat(rows, ignore_index=True)


# --- 2. Feature engineering ---------------------------------------------
def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add pitch group, swing flags, team, pull-oriented spray and timing.

    Sign conventions for contact depth and attack direction are not assumed:
    they are oriented from the data so that  + depth = contact further out
    front  and  + attack direction = toward the pull side,  using the fact
    that pulled batted balls are struck further out front.
    """
    d = df.copy()
    missing = [c for c in BAT_COLS if c not in d.columns]
    for c in missing:
        d[c] = np.nan
    d["pgroup"] = np.select([d.pitch_type.isin(FB), d.pitch_type.isin(BRK),
                             d.pitch_type.isin(OFF)], ["FB", "BRK", "OFF"], "OTH")
    d["swing"] = d.description.isin(SWING)
    d["whiff"] = d.description.isin(WHIFF)
    d["team"] = np.where(d.inning_topbot == "Bot", d.home_team, d.away_team)

    # spray angle, + = pull side
    ang = np.degrees(np.arctan2(d.hc_x - 125.42, 198.27 - d.hc_y))
    d["spray_pull"] = np.where(d.stand == "R", -ang, ang)

    info = {"missing_columns": missing}
    bip = d.dropna(subset=["spray_pull"])
    for col, new in [("intercept_ball_minus_batter_pos_y_inches", "contact_depth"),
                     ("attack_direction", "attack_dir_pull")]:
        sub = bip.dropna(subset=[col])
        if len(sub) < 500:
            d[new] = np.nan
            info[f"{new}_sign"] = "insufficient data"
            continue
        r = np.corrcoef(sub[col], sub.spray_pull)[0, 1]
        sign = 1 if r >= 0 else -1
        d[new] = sign * d[col]
        info[f"{new}_sign"] = f"{sign:+d} (corr with pull spray r={r:+.2f})"

    # Timing proxy: tercile of contact depth within league pitch group & season.
    # early = furthest out front, late = deepest. Classified on tracked contact
    # only - whiffs carry no intercept, so early-whiffs on spin are NOT counted.
    d["timing"] = pd.NA
    has = d.contact_depth.notna()
    for (yr, g), idx in d[has].groupby(["game_year", "pgroup"]).groups.items():
        q1, q2 = d.loc[idx, "contact_depth"].quantile([1 / 3, 2 / 3])
        d.loc[idx, "timing"] = np.select(
            [d.loc[idx, "contact_depth"] <= q1, d.loc[idx, "contact_depth"] >= q2],
            ["late", "early"], "on_time")
    return d, info


# --- 3. Profiles ---------------------------------------------------------
def _rates(g: pd.DataFrame) -> pd.Series:
    sw = g[g.swing]
    fb_c = g[(g.pgroup == "FB") & g.timing.notna()]
    sp_c = g[(g.pgroup == "BRK") & g.timing.notna()]
    brk = g[g.pgroup == "BRK"]
    brk02 = brk[brk.strikes < 2]
    wd = brk.woba_denom.sum()
    return pd.Series({
        "pitches": len(g),
        "swings": len(sw),
        "swing_pct": g.swing.mean(),
        "bat_speed": sw.bat_speed.mean(),
        "swing_length": sw.swing_length.mean(),
        "attack_angle": sw.attack_angle.mean(),
        "attack_dir_pull": sw.attack_dir_pull.mean(),
        "contact_depth": g.contact_depth.mean(),
        "early_pct_all": (g.timing == "early").sum() / max(g.timing.notna().sum(), 1),
        "late_pct_fb": (fb_c.timing == "late").mean() if len(fb_c) else np.nan,
        "early_pct_brk": (sp_c.timing == "early").mean() if len(sp_c) else np.nan,
        "not_early_brk": 1 - (sp_c.timing == "early").mean() if len(sp_c) else np.nan,
        "contacts_tracked": int(g.timing.notna().sum()),
        "brk_rv100": 100 * brk.delta_run_exp.mean(),    # delta_run_exp is batting-team view
        "brk_woba": brk.woba_value.sum() / wd if wd else np.nan,
        "brk_whiff_pct": brk.whiff.sum() / max(brk.swing.sum(), 1),
        "brk_pre2k_swing_pct": brk02.swing.mean(),
        "brk_pre2k_rv100": 100 * brk02.delta_run_exp.mean(),
    })


def profiles(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = d.groupby(keys).apply(_rates, include_groups=False).reset_index()
    # Short & fast quadrant, defined within season
    pct = out.groupby("game_year")[["bat_speed", "swing_length"]].rank(pct=True)
    out["bat_speed_pctl"] = pct.bat_speed
    out["swing_length_pctl"] = pct.swing_length
    out["short_fast"] = (out.bat_speed_pctl >= 0.6) & (out.swing_length_pctl <= 0.4)
    return out


# --- 4. Does "not early on spin" matter beyond bat speed? -----------------
def spin_model(h: pd.DataFrame) -> pd.DataFrame | None:
    import statsmodels.formula.api as smf
    m = h.dropna(subset=["brk_rv100", "not_early_brk", "bat_speed", "swing_length"])
    m = m[m.contacts_tracked >= MIN_CONTACT]
    if len(m) < 30:
        return None
    fit = smf.wls("brk_rv100 ~ not_early_brk + bat_speed + swing_length + C(game_year)",
                  data=m, weights=m.pitches).fit(cov_type="HC1")
    t = fit.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    t["n_hitters"] = len(m)
    t["r2"] = fit.rsquared
    return t


# --- 5. Breakeven pitch mix ----------------------------------------------
def breakeven(d: pd.DataFrame) -> pd.DataFrame:
    """Breaking-ball share at which sitting back stops paying.

    cost_fb   = run value lost per contact when late vs on time on fastballs
    cost_brk  = run value lost per contact when early vs on time on breaking balls
    With breaking-ball share s (of FB+BRK), sitting back breaks even when
        s * cost_brk = (1 - s) * cost_fb   ->   s* = cost_fb / (cost_fb + cost_brk)
    If the league's actual s > s*, accepting late fastballs is the cheaper error.

    Caveat: timing is only classified on contact, so early-on-spin WHIFFS are
    missing from cost_brk. That understates cost_brk, so s* is an UPPER bound
    - the real breakeven spin share is probably lower.
    """
    rows = []
    for yr, g in d.groupby("game_year"):
        rv = g.delta_run_exp      # already batting-team perspective (+ = good for hitter)
        def mean_rv(mask):
            return rv[mask].mean()
        fb, br = g.pgroup == "FB", g.pgroup == "BRK"
        cost_fb = mean_rv(fb & (g.timing == "on_time")) - mean_rv(fb & (g.timing == "late"))
        cost_brk = mean_rv(br & (g.timing == "on_time")) - mean_rv(br & (g.timing == "early"))
        s_actual = br.sum() / (fb.sum() + br.sum())
        s_star = cost_fb / (cost_fb + cost_brk) if (cost_fb + cost_brk) > 0 else np.nan
        rows.append({"game_year": yr, "cost_late_fb": cost_fb,
                     "cost_early_brk": cost_brk, "breakeven_brk_share": s_star,
                     "actual_brk_share": s_actual,
                     "sitting_back_pays": bool(s_actual > s_star) if pd.notna(s_star) else None})
    return pd.DataFrame(rows)


# --- 6. Figures ----------------------------------------------------------
def plot_quadrant(h: pd.DataFrame, season: int, path: Path, highlight: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = h[h.game_year == season].dropna(subset=["bat_speed", "swing_length"])
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(s.swing_length, s.bat_speed, s=14, c="#9aa5b1", alpha=.6, label="Hitters")
    sf = s[s.short_fast]
    ax.scatter(sf.swing_length, sf.bat_speed, s=18, c="#1f6feb", label="Short & fast")
    if highlight is not None and "team" in s:
        hl = s[s.team == highlight]
        ax.scatter(hl.swing_length, hl.bat_speed, s=40, facecolors="none",
                   edgecolors="#d1242f", linewidths=1.4, label=highlight)
    ax.axvline(s.swing_length.quantile(.4), ls="--", lw=.8, c="#57606a")
    ax.axhline(s.bat_speed.quantile(.6), ls="--", lw=.8, c="#57606a")
    ax.set(xlabel="Avg swing length (ft)", ylabel="Avg bat speed (mph)",
           title=f"Decision Window quadrant, {season}")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_breakeven(b: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if b.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(b.game_year, 100 * b.actual_brk_share, marker="o", c="#1f6feb",
            label="Actual breaking-ball share")
    ax.plot(b.game_year, 100 * b.breakeven_brk_share, marker="s", ls="--",
            c="#d1242f", label="Breakeven share (upper bound)")
    ax.set(ylabel="% of FB + breaking pitches", xlabel="Season",
           title="Does sitting back pay? Above the red line, yes")
    ax.set_xticks(b.game_year)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- main ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", type=int, nargs="+", default=[2025])
    ap.add_argument("--demo", action="store_true",
                    help="use synthetic data (pipeline check; results are meaningless)")
    ap.add_argument("--team", default="MIL", help="team to highlight in figures")
    args = ap.parse_args()

    load = synthetic_statcast if args.demo else load_statcast
    raw = pd.concat([load(s) for s in args.seasons], ignore_index=True)
    d, info = prepare(raw)
    print("Missing bat-tracking columns:", info["missing_columns"] or "none")
    print("Contact depth orientation:", info["contact_depth_sign"])
    print("Attack direction orientation:", info["attack_dir_pull_sign"])

    tag = "_DEMO" if args.demo else ""
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    # hitters: attach most-frequent team per season for highlighting
    team_of = (d.groupby(["game_year", "batter"]).team
                 .agg(lambda s: s.mode().iat[0]).rename("team"))
    h = profiles(d, ["game_year", "batter", "player_name"])
    h = h.merge(team_of.reset_index(), on=["game_year", "batter"])
    h = h[h.swings >= MIN_SWINGS]
    t = profiles(d, ["game_year", "team"])
    b = breakeven(d)
    m = spin_model(h)

    h.to_csv(TABLES / f"decision_window_hitters{tag}.csv", index=False)
    t.to_csv(TABLES / f"decision_window_teams{tag}.csv", index=False)
    b.to_csv(TABLES / f"decision_window_breakeven{tag}.csv", index=False)
    if m is not None:
        m.to_csv(TABLES / f"decision_window_spin_model{tag}.csv", index=False)

    last = max(args.seasons)
    plot_quadrant(h, last, FIGS / f"decision_window_quadrant{tag}.png", args.team)
    plot_breakeven(b, FIGS / f"decision_window_breakeven{tag}.png")

    cols = ["team", "bat_speed", "swing_length", "attack_angle", "attack_dir_pull",
            "contact_depth", "late_pct_fb", "early_pct_brk", "swing_pct", "brk_rv100"]
    print(f"\nTeams {last} (sorted by early% on breaking balls):")
    print(t[t.game_year == last].sort_values("early_pct_brk")[cols]
            .round(3).to_string(index=False))
    print("\nBreakeven:\n", b.round(4).to_string(index=False))
    if m is not None:
        print("\nSpin model (brk_rv100 ~ not_early_brk + bat_speed + swing_length):")
        print(m[["term", "Coef.", "Std.Err.", "P>|z|"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
