"""04_pitch_cost.py — Pitch cost: what elite velocity buys, and what foul balls take back.

Motivation: Jacob Misiorowski (MIL) threw the fastest postseason pitch on record
(104.8 mph) to open NLDS Game 1 on 2026-10-03, yet San Diego fouled off enough
two-strike pitches to end his night after four innings. Strikeout-centric
metrics (K%, whiff%) describe his stuff; they don't describe how *expensive*
each out was. This module measures that cost from both sides.

Outputs (output/tables, output/figures):
  pitcher_cost_<season>.csv     pitches/PA, 2-strike foul rate allowed, putaway%
  hitter_foul_skill_<season>.csv 2-strike foul rate overall and vs 98+ mph,
                                 z-contact, bat speed, swing length
  foul_skill_predictors_<season>.csv  which swing trait predicts fouling off velo
  foul_skill_reliability_<season>.csv split-half reliability of each metric
  misiorowski_nlds_g1.csv       pitch-by-PA case study + regular-season baseline
  foul_skill_predictors_<season>.png

Run from the project root:
  python python/04_pitch_cost.py --season 2026
First run downloads the season from Baseball Savant (cached afterwards).

DEFINITIONS (for the report)
* Swing      = swinging strike (+blocked), foul, foul tip, ball in play. Bunts excluded.
* 2-strike foul rate = fouls / swings with two strikes. A *foul tip* is a caught
  strike three, so it is a swing but NOT a foul here — it ends the PA.
* Elite velocity = four-seamer or sinker at >= VELO_MIN mph (default 98).
* Putaway% = strikeouts / two-strike pitches (Savant's definition).
* Reliability = split-half by odd/even plate appearance, Spearman-Brown
  corrected. Below ~0.5 the metric is mostly noise at that sample size —
  the same lesson as Marcel beating the fitted models in Phase 3.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES, FIGS = ROOT / "output/tables", ROOT / "output/figures"

MISIOROWSKI_ID = 694819
VELO_MIN = 98.0
FASTBALLS = {"FF", "SI"}

SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
         "hit_into_play"}
WHIFF = {"swinging_strike", "swinging_strike_blocked"}
CONTACT = {"foul", "foul_tip", "hit_into_play"}


# ---------------------------------------------------------------- pitch flags
def add_flags(sc: pd.DataFrame, velo_min: float = VELO_MIN) -> pd.DataFrame:
    d = sc.copy()
    desc = d["description"]
    d["swing"] = desc.isin(SWING)
    d["whiff"] = desc.isin(WHIFF)
    d["contact"] = desc.isin(CONTACT)
    d["foul"] = desc.eq("foul")
    d["two_k"] = d["strikes"].eq(2)
    d["in_zone"] = d["zone"].between(1, 9)
    d["elite_velo"] = d["pitch_type"].isin(FASTBALLS) & (d["release_speed"] >= velo_min)
    d["k"] = d["events"].isin({"strikeout", "strikeout_double_play"})
    d["pa_id"] = (d["game_pk"].astype(str) + "_" + d["at_bat_number"].astype(str))
    return d


def _safe(num, den):
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


# ------------------------------------------------------------- pitcher side
def pitcher_cost(d: pd.DataFrame, min_pa: int = 200) -> pd.DataFrame:
    g = d.groupby("pitcher")
    out = pd.DataFrame({
        "name": g["player_name"].first(),
        "pitches": g.size(),
        "PA": g["pa_id"].nunique(),
        "K": g["k"].sum(),
        "swings": g["swing"].sum(),
        "whiffs": g["whiff"].sum(),
        "two_k_pitches": g["two_k"].sum(),
        "two_k_swings": g.apply(lambda x: (x.swing & x.two_k).sum(), include_groups=False),
        "two_k_fouls": g.apply(lambda x: (x.foul & x.two_k).sum(), include_groups=False),
        "ff_velo": g.apply(lambda x: x.loc[x.pitch_type.eq("FF"), "release_speed"].mean(),
                           include_groups=False),
    })
    out["pitches_per_pa"] = out.pitches / out.PA
    out["K_pct"] = out.K / out.PA
    out["whiff_pct"] = _safe(out.whiffs, out.swings)
    out["two_k_foul_rate_allowed"] = _safe(out.two_k_fouls, out.two_k_swings)
    out["putaway_pct"] = _safe(out.K, out.two_k_pitches)
    # "Cost" residual: pitches/PA beyond what a pitcher's K% alone predicts.
    # Strikeout pitchers naturally run deep counts; the residual isolates
    # pitchers whose PAs are expensive *for their K rate*.
    q = out[out.PA >= min_pa].copy()
    if len(q) >= 10:
        b = np.polyfit(q.K_pct, q.pitches_per_pa, 1)
        q["ppa_over_expected"] = q.pitches_per_pa - np.polyval(b, q.K_pct)
    else:
        q["ppa_over_expected"] = np.nan
    return q.sort_values("ppa_over_expected", ascending=False).reset_index()


# -------------------------------------------------------------- hitter side
def hitter_foul_skill(d: pd.DataFrame) -> pd.DataFrame:
    def agg(x: pd.DataFrame) -> pd.Series:
        tk = x[x.two_k & x.swing]
        tkv = tk[tk.elite_velo]
        zs = x[x.in_zone & x.swing]
        sw = x[x.swing]
        return pd.Series({
            "PA": x.pa_id.nunique(),
            "pitches_per_pa": len(x) / max(x.pa_id.nunique(), 1),
            "two_k_swings": len(tk),
            "two_k_foul_rate": tk.foul.mean() if len(tk) else np.nan,
            "two_k_swings_vs_velo": len(tkv),
            "two_k_foul_rate_vs_velo": tkv.foul.mean() if len(tkv) else np.nan,
            "z_contact": zs.contact.mean() if len(zs) else np.nan,
            "bat_speed": sw.bat_speed.mean(),
            "swing_length": sw.swing_length.mean(),
        })
    return d.groupby("batter").apply(agg, include_groups=False).reset_index()


def foul_skill_predictors(h: pd.DataFrame, min_swings: int = 25) -> pd.DataFrame:
    """Which swing trait best predicts fouling off 98+ with two strikes?

    Reports each trait's simple correlation and its standardized coefficient
    in a joint OLS — the joint fit matters because bat speed and swing length
    are themselves correlated (longer swings generate more speed).
    """
    import statsmodels.api as sm

    X = ["bat_speed", "swing_length", "z_contact"]
    y = "two_k_foul_rate_vs_velo"
    q = h[h.two_k_swings_vs_velo >= min_swings].dropna(subset=X + [y])
    if len(q) < 10:
        return pd.DataFrame({"note": [f"only {len(q)} hitters with >= {min_swings} "
                                      "two-strike swings vs elite velo"]})
    z = (q[X] - q[X].mean()) / q[X].std()
    fit = sm.OLS((q[y] - q[y].mean()) / q[y].std(), sm.add_constant(z)).fit()
    return pd.DataFrame({
        "trait": X,
        "pearson_r": [q[c].corr(q[y]) for c in X],
        "std_beta": [fit.params[c] for c in X],
        "p_value": [fit.pvalues[c] for c in X],
        "n_hitters": len(q),
        "joint_r2": fit.rsquared,
    })


def split_half_reliability(d: pd.DataFrame, min_each: int = 15) -> pd.DataFrame:
    """Odd/even-PA split-half reliability (Spearman-Brown) for the hitter metrics."""
    d = d.copy()
    d["half"] = d.groupby("batter")["pa_id"].transform(
        lambda s: pd.factorize(s)[0] % 2)
    metrics = {
        "two_k_foul_rate": lambda x: x[x.two_k & x.swing].foul,
        "two_k_foul_rate_vs_velo": lambda x: x[x.two_k & x.swing & x.elite_velo].foul,
        "z_contact": lambda x: x[x.in_zone & x.swing].contact,
    }
    rows = []
    for name, sel in metrics.items():
        halves = {}
        for h in (0, 1):
            s = d[d.half == h].groupby("batter").apply(
                lambda x: pd.Series({"v": sel(x).mean(), "n": len(sel(x))}),
                include_groups=False)
            halves[h] = s
        m = halves[0].join(halves[1], lsuffix="0", rsuffix="1").dropna()
        m = m[(m.n0 >= min_each) & (m.n1 >= min_each)]
        r = m.v0.corr(m.v1) if len(m) >= 10 else np.nan
        rows.append({"metric": name, "n_hitters": len(m), "split_half_r": r,
                     "spearman_brown": 2 * r / (1 + r) if pd.notna(r) else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- case study
def misiorowski_case(d: pd.DataFrame, game_date: str = "2026-10-03") -> pd.DataFrame:
    m = d[d.pitcher.eq(MISIOROWSKI_ID)]
    g = m[m.game_date.astype(str).str[:10].eq(game_date)]
    if g.empty:
        return pd.DataFrame({"note": [f"no Misiorowski pitches on {game_date}"]})
    pa = (g.sort_values(["at_bat_number", "pitch_number"])
            .groupby(["inning", "at_bat_number"])
            .agg(batter=("batter", "first"), pitches=("pitch_number", "size"),
                 two_k_fouls=("foul", lambda s: int((s & g.loc[s.index, "two_k"]).sum())),
                 max_velo=("release_speed", "max"), result=("events", "last"))
            .reset_index())
    reg = m[m.game_type.eq("R")]
    base = {
        "inning": "REG-SEASON", "at_bat_number": np.nan, "batter": np.nan,
        "pitches": len(reg) / max(reg.pa_id.nunique(), 1),   # pitches per PA
        "two_k_fouls": _safe((reg.foul & reg.two_k).sum(),
                             (reg.swing & reg.two_k).sum()).item(),  # rate
        "max_velo": reg.release_speed.max(), "result": "baseline (per PA / rate)",
    }
    game = {
        "inning": "NLDS G1", "at_bat_number": np.nan, "batter": np.nan,
        "pitches": len(g) / max(g.pa_id.nunique(), 1),
        "two_k_fouls": _safe((g.foul & g.two_k).sum(), (g.swing & g.two_k).sum()).item(),
        "max_velo": g.release_speed.max(), "result": "game (per PA / rate)",
    }
    return pd.concat([pa, pd.DataFrame([game, base])], ignore_index=True)


# --------------------------------------------------------------------- plot
def plot_predictors(h: pd.DataFrame, season: int, min_swings: int = 25) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = h[h.two_k_swings_vs_velo >= min_swings]
    traits = [("bat_speed", "Bat speed (mph)"), ("swing_length", "Swing length (ft)"),
              ("z_contact", "In-zone contact rate")]
    q = q.dropna(subset=[t for t, _ in traits] + ["two_k_foul_rate_vs_velo"])
    if len(q) < 10:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (col, label) in zip(axes, traits):
        ax.scatter(q[col], q.two_k_foul_rate_vs_velo, s=12, alpha=.55, color="#2a6f97")
        b = np.polyfit(q[col], q.two_k_foul_rate_vs_velo, 1)
        xs = np.linspace(q[col].min(), q[col].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color="#c44536", lw=1.5)
        ax.set_xlabel(label)
        ax.set_title(f"r = {q[col].corr(q.two_k_foul_rate_vs_velo):+.2f}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"2-strike foul rate vs {VELO_MIN:.0f}+ mph")
    fig.suptitle(f"What predicts fouling off elite velocity? ({season}, "
                 f"hitters with ≥{min_swings} two-strike swings vs {VELO_MIN:.0f}+)",
                 fontsize=11)
    fig.tight_layout()
    path = FIGS / f"foul_skill_predictors_{season}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- main
def run(sc: pd.DataFrame, season: int, include_postseason: bool = False) -> dict:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    d = add_flags(sc[sc.description.notna()])
    reg = d if include_postseason else d[d.game_type.eq("R")]

    res = {
        f"pitcher_cost_{season}": pitcher_cost(reg),
        f"hitter_foul_skill_{season}": (h := hitter_foul_skill(reg)),
        f"foul_skill_predictors_{season}": foul_skill_predictors(h),
        f"foul_skill_reliability_{season}": split_half_reliability(reg),
        "misiorowski_nlds_g1": misiorowski_case(d),
    }
    for name, df in res.items():
        df.to_csv(TABLES / f"{name}.csv", index=False)
    plot_predictors(h, season)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--refresh", action="store_true", help="re-download from Savant")
    args = ap.parse_args()

    from statcast_io import load_season
    out = run(load_season(args.season, refresh=args.refresh), args.season)

    pd.set_option("display.width", 140)
    print("\nMost expensive pitchers for their K% (pitches/PA over expected):")
    print(out[f"pitcher_cost_{args.season}"].head(10)[
        ["name", "PA", "K_pct", "pitches_per_pa", "ppa_over_expected",
         "two_k_foul_rate_allowed"]].round(3).to_string(index=False))
    print("\nPredictors of 2-strike foul rate vs elite velo:")
    print(out[f"foul_skill_predictors_{args.season}"].round(3).to_string(index=False))
    print("\nReliability:")
    print(out[f"foul_skill_reliability_{args.season}"].round(3).to_string(index=False))
    print("\nMisiorowski NLDS G1 summary rows:")
    print(out["misiorowski_nlds_g1"].tail(2).to_string(index=False))
