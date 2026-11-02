"""06_dsl_time_to_debut.py -- how long until a DSL hitter debuts in MLB, if ever?

International signees have no draft pick, so the published MiLB-to-MLB models
do not apply to them. This module fits a Cox survival model on a hitter's FIRST
Dominican Summer League season and is built to avoid the leaks in that work:

- Features use the first DSL season only, so at-bats earned later cannot leak in.
- K%, BB% and ISO are shrunk toward, and expressed relative to, that season's
  league, because the DSL run environment has changed over time.
- Players still unresolved are censored, not counted as failures.
- Training labels stop at the split year. Validation uses later cohorts only.

Usage:
    python python/06_dsl_time_to_debut.py              # real data (needs internet)
    python python/06_dsl_time_to_debut.py --demo       # synthetic pipeline check

Data: MLB Stats API, sportId=16, league id 130. Hitters only; signing bonus is
not in this source. Outputs go to output/tables and output/figures; --demo
outputs carry a _DEMO suffix (gitignored) and say nothing about real players.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/milb"
TABLES, FIGURES = ROOT / "output/tables", ROOT / "output/figures"

DSL_LEAGUE_ID = 130
API = ("https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting"
       "&sportId=16&season={season}&gameType=R&playerPool=ALL&limit=10000"
       "&hydrate=person")

FEATURES = ["age_rel", "k_rel", "bb_rel", "iso_rel", "log_pa"]
BASELINE_FEATURES = ["age_rel"]
PRIOR_PA = 100          # shrinkage weight toward the league rate, in PA
MIN_PA = 50             # minimum PA in the first DSL season
DUPLICATE_KEYS = ["player_id", "season"]
SUM_COLS = ["pa", "ab", "h", "b2", "b3", "hr", "bb", "so"]


# ---------------------------------------------------------------- acquisition
def fetch_season(season: int, refresh: bool = False) -> dict:
    """Download one season of rookie-level hitting, cached in data/raw/milb."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"rookie_hitting_{season}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    request = urllib.request.Request(API.format(season=season),
                                     headers={"User-Agent": "baseball-portfolio"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    path.write_text(json.dumps(payload))
    time.sleep(0.5)
    return payload


def parse_splits(payload: dict, season: int) -> pd.DataFrame:
    """Flatten an API payload into one row per player, team and season."""
    rows = []
    for block in payload.get("stats", []):
        for split in block.get("splits", []):
            if split.get("league", {}).get("id") != DSL_LEAGUE_ID:
                continue
            stat, player = split.get("stat", {}), split.get("player", {})
            rows.append({
                "player_id": player.get("id"), "name": player.get("fullName"),
                "birth_date": player.get("birthDate"),
                "mlb_debut_date": player.get("mlbDebutDate"),
                "season": season, "team": split.get("team", {}).get("name"),
                "pa": stat.get("plateAppearances", 0),
                "ab": stat.get("atBats", 0), "h": stat.get("hits", 0),
                "b2": stat.get("doubles", 0), "b3": stat.get("triples", 0),
                "hr": stat.get("homeRuns", 0), "bb": stat.get("baseOnBalls", 0),
                "so": stat.get("strikeOuts", 0),
            })
    return pd.DataFrame(rows)


def load_real(first: int, last: int) -> pd.DataFrame:
    frames = [parse_splits(fetch_season(year), year) for year in range(first, last + 1)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No DSL rows returned; check the API response.")
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------ cohort building
def season_totals(raw: pd.DataFrame) -> pd.DataFrame:
    """Combine players who appeared for more than one DSL team in a season."""
    first_values = {column: "first" for column in
                    ["name", "birth_date", "mlb_debut_date"] if column in raw}
    return raw.groupby(DUPLICATE_KEYS, as_index=False).agg(
        {**{column: "sum" for column in SUM_COLS}, **first_values})


def add_league_relative(seasons: pd.DataFrame) -> pd.DataFrame:
    """Add shrunken K%, BB% and ISO, each as a ratio to that season's league."""
    out = seasons.copy()
    tb = out.h + out.b2 + 2 * out.b3 + 3 * out.hr
    out["_extra_bases"] = tb - out.h
    league = out.groupby("season")[["pa", "ab", "so", "bb", "_extra_bases"]].transform("sum")
    lg_k, lg_bb = league.so / league.pa, league.bb / league.pa
    lg_iso = league._extra_bases / league.ab
    out["k_rel"] = ((out.so + PRIOR_PA * lg_k) / (out.pa + PRIOR_PA)) / lg_k
    out["bb_rel"] = ((out.bb + PRIOR_PA * lg_bb) / (out.pa + PRIOR_PA)) / lg_bb
    out["iso_rel"] = ((out._extra_bases + PRIOR_PA * lg_iso)
                      / (out.ab + PRIOR_PA)) / lg_iso

    july_first = pd.to_datetime(out.season.astype(str) + "-07-01")
    out["age"] = (july_first - pd.to_datetime(out.birth_date)).dt.days / 365.25
    weights = out.pa.where(out.age.notna(), 0)
    league_age = ((out.age.fillna(0) * weights).groupby(out.season).transform("sum")
                  / weights.groupby(out.season).transform("sum"))
    out["age_rel"] = out.age - league_age
    out["log_pa"] = np.log(out.pa.clip(lower=1))
    return out.drop(columns="_extra_bases")


def build_cohort(raw: pd.DataFrame, as_of: int, min_pa: int = MIN_PA) -> pd.DataFrame:
    """One row per hitter: first-season features plus time to debut.

    The earliest loaded season is dropped because those players may have
    played in the DSL before the data starts. A debut after `as_of` is treated
    as not yet observed, which is what keeps future labels out of training.
    """
    seasons = add_league_relative(season_totals(raw))
    earliest = seasons.season.min()
    first = seasons.sort_values("season").drop_duplicates("player_id", keep="first")
    cohort = first[(first.season > earliest) & (first.pa >= min_pa)
                   & (first.season <= as_of)].dropna(subset=FEATURES).copy()

    debut_year = pd.to_datetime(cohort.mlb_debut_date).dt.year
    cohort = cohort[~(debut_year < cohort.season)].copy()
    debut_year = pd.to_datetime(cohort.mlb_debut_date).dt.year
    cohort["event"] = (debut_year.notna() & (debut_year <= as_of)).astype(int)
    cohort["duration"] = np.where(cohort.event == 1,
                                  debut_year - cohort.season,
                                  as_of - cohort.season).astype(float)
    cohort["duration"] = cohort.duration.clip(lower=0.5)
    return cohort.reset_index(drop=True)


# ---------------------------------------------------------------- evaluation
def rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    y, score = np.asarray(y), np.asarray(score)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_cox(frame: pd.DataFrame, features: list[str]):
    from lifelines import CoxPHFitter
    model = CoxPHFitter(penalizer=0.01)
    model.fit(frame[features + ["duration", "event"]],
              duration_col="duration", event_col="event")
    return model


def probability_by(model, frame: pd.DataFrame, horizon: float) -> np.ndarray:
    survival = model.predict_survival_function(frame, times=[horizon])
    return 1 - survival.to_numpy().ravel()


def baseline_rate(train: pd.DataFrame, horizon: float) -> float:
    from lifelines import KaplanMeierFitter
    km = KaplanMeierFitter().fit(train.duration, train.event)
    return 1 - float(km.predict(horizon))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 5) -> pd.DataFrame:
    table = pd.DataFrame({"y": y, "p": p})
    table["bin"] = pd.qcut(table.p.rank(method="first"), bins, labels=False) + 1
    return (table.groupby("bin").agg(players=("y", "size"), predicted=("p", "mean"),
                                     actual=("y", "mean")).reset_index())


def bootstrap_gain(y, p_full, p_base, reps: int = 1000, seed: int = 6) -> tuple:
    """95% interval for the AUC gain of the full model over age alone."""
    rng = np.random.default_rng(seed)
    y, p_full, p_base = map(np.asarray, (y, p_full, p_base))
    gains = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        gain = rank_auc(y[i], p_full[i]) - rank_auc(y[i], p_base[i])
        if not np.isnan(gain):
            gains.append(gain)
    return tuple(np.percentile(gains, [2.5, 97.5]))


def run_validation(raw: pd.DataFrame, split_year: int, as_of: int,
                   horizon: int = 6, reps: int = 1000) -> dict:
    """Train as if standing at the end of split_year; test later cohorts."""
    train = build_cohort(raw[raw.season <= split_year], as_of=split_year)
    full_view = build_cohort(raw, as_of=as_of)
    test = full_view[(full_view.season > split_year)
                     & (full_view.season + horizon <= as_of)].copy()
    if test.empty:
        raise ValueError("No test cohort has enough follow-up for this horizon.")

    # A player "made it" only if he debuted within the horizon.
    y = ((test.event == 1) & (test.duration <= horizon)).astype(int).to_numpy()
    full, base = fit_cox(train, FEATURES), fit_cox(train, BASELINE_FEATURES)
    p_full = probability_by(full, test, horizon)
    p_base = probability_by(base, test, horizon)
    rate = baseline_rate(train, horizon)

    summary = {
        "train_players": len(train), "train_debuts": int(train.event.sum()),
        "test_players": len(test), "test_debuts": int(y.sum()),
        "horizon_years": horizon, "train_rate": rate,
        "test_rate": float(y.mean()), "mean_predicted": float(p_full.mean()),
        "auc_full": rank_auc(y, p_full), "auc_age_only": rank_auc(y, p_base),
        "brier_full": float(np.mean((p_full - y) ** 2)),
        "brier_base_rate": float(np.mean((rate - y) ** 2)),
    }
    summary["auc_gain_low"], summary["auc_gain_high"] = bootstrap_gain(
        y, p_full, p_base, reps)
    return {"summary": pd.DataFrame([summary]), "model": full,
            "calibration": calibration_table(y, p_full), "train": train, "test": test}


def score_latest(raw: pd.DataFrame, as_of: int, horizon: int = 6) -> pd.DataFrame:
    """Refit on all resolved history and score the newest DSL cohort."""
    cohort = build_cohort(raw, as_of=as_of)
    latest = cohort[cohort.season == cohort.season.max()].copy()
    model = fit_cox(cohort[cohort.season < cohort.season.max()], FEATURES)
    latest["p_debut"] = probability_by(model, latest, horizon)
    columns = ["player_id", "name", "season", "age", "pa"] + FEATURES[:-1] + ["p_debut"]
    return latest.sort_values("p_debut", ascending=False)[columns]


# ------------------------------------------------------------------- demo data
def synthetic_raw(players: int = 6000, first: int = 2005, last: int = 2025,
                  seed: int = 6) -> pd.DataFrame:
    """Fake league with planted effects: youth, contact and power speed debuts."""
    rng = np.random.default_rng(seed)
    season = rng.integers(first, last + 1, players)
    age = rng.normal(18.8, 1.1, players).clip(16.5, 23)
    k_skill, power = rng.normal(0, 1, players), rng.normal(0, 1, players)
    pa = rng.integers(40, 280, players)
    k_rate = np.clip(0.21 + 0.05 * k_skill, 0.05, 0.5)
    bb_rate = np.clip(rng.normal(0.11, 0.03, players) + 0.002 * (season - first), 0.02, 0.3)
    bb = rng.binomial(pa, bb_rate)
    ab = np.maximum(pa - bb, 1)
    so = rng.binomial(ab, k_rate)
    h = rng.binomial(ab - so, 0.32)
    hr = rng.binomial(h, np.clip(0.03 + 0.02 * power, 0.0, 0.3))
    b2 = rng.binomial(h - hr, np.clip(0.18 + 0.05 * power, 0.02, 0.5))
    hazard = 0.006 * np.exp(-0.6 * (age - 18.8) - 0.55 * k_skill + 0.45 * power)
    wait = rng.exponential(1 / hazard)
    debut = np.where(wait < 40, season + np.ceil(wait).astype(int), 9999)
    birth = pd.to_datetime(season.astype(str) + "-07-01") - pd.to_timedelta(age * 365.25, unit="D")
    return pd.DataFrame({
        "player_id": np.arange(players), "name": [f"Demo {i}" for i in range(players)],
        "birth_date": birth.strftime("%Y-%m-%d"),
        "mlb_debut_date": [None if d > last else f"{d}-06-01" for d in debut],
        "season": season, "team": "DEMO", "pa": pa, "ab": ab, "h": h, "b2": b2,
        "b3": 0, "hr": hr, "bb": bb, "so": so,
    })


# ------------------------------------------------------------------------ main
def write_outputs(results: dict, latest: pd.DataFrame, suffix: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    results["summary"].to_csv(TABLES / f"dsl_validation{suffix}.csv", index=False)
    results["calibration"].to_csv(TABLES / f"dsl_calibration{suffix}.csv", index=False)
    results["model"].summary[["coef", "exp(coef)", "p"]].to_csv(
        TABLES / f"dsl_hazard_ratios{suffix}.csv")
    latest.to_csv(TABLES / f"dsl_latest_cohort{suffix}.csv", index=False)

    calibration = results["calibration"]
    top = max(calibration.predicted.max(), calibration.actual.max()) * 1.15
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, top], [0, top], color="#999999", linewidth=1)
    axis.scatter(calibration.predicted, calibration.actual, color="#1f4e79", s=45)
    axis.set(xlim=(0, top), ylim=(0, top), xlabel="Predicted chance of debut",
             ylabel="Actual debut rate", title="DSL model calibration" +
             (" (synthetic demo)" if suffix else ""))
    figure.tight_layout()
    figure.savefig(FIGURES / f"dsl_calibration{suffix}.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", type=int, default=2005)
    parser.add_argument("--last", type=int, default=2026)
    parser.add_argument("--split-year", type=int, default=2013)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    raw = synthetic_raw(last=args.last - 1) if args.demo else load_real(args.first, args.last)
    as_of = args.last - 1 if args.demo else args.last
    results = run_validation(raw, args.split_year, as_of, args.horizon)
    latest = score_latest(raw, as_of, args.horizon)
    write_outputs(results, latest, "_DEMO" if args.demo else "")
    label = "SYNTHETIC DEMO, not real players" if args.demo else "real DSL data"
    print(f"\n{label}\n{results['summary'].T.to_string()}")
    print("\nCalibration\n", results["calibration"].to_string(index=False))
    print("\nHazard ratios\n", results["model"].summary[["exp(coef)", "p"]].to_string())


if __name__ == "__main__":
    main()
