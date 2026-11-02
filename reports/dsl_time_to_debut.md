# DSL time to debut: methodology note

**Script:** `python/06_dsl_time_to_debut.py`
**Status:** pipeline tested on synthetic data only. No real-data results yet.
**Source idea:** Lee & Lee (2023), *Baseball Informatics: From MiLB to MLB Debut*.

## Question

Given only a hitter's first Dominican Summer League season, how likely is he to
debut in MLB within six years?

Draft models cannot answer this because international signees have no draft pick.

## Design

| Risk in earlier work | Choice here |
|---|---|
| Career totals reflect playing time teams already chose to give | Features come from the first DSL season only |
| The DSL run environment shifts over time | K%, BB% and ISO are ratios to that season's league |
| Small samples are noisy | Rates shrink toward the league with a 100 PA prior |
| Unresolved players look like failures | Cox model with censoring |
| Random splits leak the future | Train through 2013 with labels stopped at 2013, test later cohorts |
| Accuracy flatters a rare outcome | Report AUC, Brier score against the base rate, and calibration |

Players in the earliest loaded season are dropped because they may have played
in the DSL before the data starts.

## Run it

Run this on a computer with normal internet access:

```bash
pip install -r requirements.txt
python python/06_dsl_time_to_debut.py
python -m pytest tests/test_06_dsl_time_to_debut.py -q
```

Results land in `output/tables/dsl_*.csv` and `output/figures/dsl_calibration.png`.
`--demo` runs on synthetic players and writes `_DEMO` files that git ignores.

## Limits

- Hitters only. Pitchers need their own feature set.
- Signing bonus is not in the MLB Stats API, so the model omits it.
- A debut is a team decision, and it says nothing about MLB value.
- The DSL lost its 2020 season, which affects those cohorts' development paths.
- Reaching MLB is rare, so subgroup and single-player estimates are uncertain.
