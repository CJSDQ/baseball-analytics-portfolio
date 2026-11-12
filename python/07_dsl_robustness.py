"""07_dsl_robustness.py -- does the DSL model survive two stress tests?

1. Drop plate appearances. PA partly reflects which players teams already
   favored, so the model's skill without it is closer to player skill alone.
2. Move the training cutoff later. More training debuts should give steadier
   effect sizes, at the cost of fewer test cohorts.

Reuses module 06 and its cached data in data/raw/milb (run 06 once first).
Writes output/tables/dsl_robustness.csv and dsl_robustness_hazards.csv.

Usage:
    python python/07_dsl_robustness.py
    python python/07_dsl_robustness.py --demo     # synthetic pipeline check
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dsl", ROOT / "python" / "06_dsl_time_to_debut.py")
dsl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsl)

FEATURE_SETS = {
    "full": ["age_rel", "k_rel", "bb_rel", "iso_rel", "log_pa"],
    "no_pa": dsl.FEATURES,
}
SPLIT_YEARS = [2013, 2015, 2016]


def top_fifth_share(y: np.ndarray, p: np.ndarray) -> float:
    """Share of all debuts that came from the top 20% of predictions."""
    cut = np.quantile(p, 0.8)
    return float(y[p >= cut].sum() / max(y.sum(), 1))


def run(raw: pd.DataFrame, as_of: int, horizon: int, reps: int) -> tuple:
    rows, hazards = [], []
    for split in SPLIT_YEARS:
        train = dsl.build_cohort(raw[raw.season <= split], as_of=split)
        view = dsl.build_cohort(raw, as_of=as_of)
        test = view[(view.season > split) & (view.season + horizon <= as_of)]
        if test.empty or train.event.sum() == 0:
            continue
        y = ((test.event == 1) & (test.duration <= horizon)).astype(int).to_numpy()
        p_age = dsl.probability_by(dsl.fit_cox(train, ["age_rel"]), test, horizon)
        for name, features in FEATURE_SETS.items():
            model = dsl.fit_cox(train, features)
            p = dsl.probability_by(model, test, horizon)
            low, high = dsl.bootstrap_gain(y, p, p_age, reps)
            rows.append({
                "split_year": split, "features": name,
                "train_debuts": int(train.event.sum()),
                "test_seasons": f"{test.season.min()}-{test.season.max()}",
                "test_players": len(test), "test_debuts": int(y.sum()),
                "auc": dsl.rank_auc(y, p), "auc_age_only": dsl.rank_auc(y, p_age),
                "gain_low": low, "gain_high": high,
                "top_fifth_share": top_fifth_share(y, p),
                "mean_predicted": float(p.mean()), "actual_rate": float(y.mean()),
            })
            ci = model.confidence_intervals_
            for factor in features:
                hazards.append({
                    "split_year": split, "features": name, "factor": factor,
                    "hazard_ratio": float(np.exp(model.params_[factor])),
                    "low": float(np.exp(ci.loc[factor].iloc[0])),
                    "high": float(np.exp(ci.loc[factor].iloc[1])),
                    "p": float(model.summary.loc[factor, "p"]),
                })
    return pd.DataFrame(rows), pd.DataFrame(hazards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", type=int, default=2005)
    parser.add_argument("--last", type=int, default=2026)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        raw, as_of = dsl.synthetic_raw(last=args.last - 1), args.last - 1
    else:
        raw, as_of = dsl.load_real(args.first, args.last), args.last
    summary, hazards = run(raw, as_of, args.horizon, args.reps)

    suffix = "_DEMO" if args.demo else ""
    dsl.TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(dsl.TABLES / f"dsl_robustness{suffix}.csv", index=False)
    hazards.to_csv(dsl.TABLES / f"dsl_robustness_hazards{suffix}.csv", index=False)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print()
    print(hazards.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
