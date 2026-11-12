# DSL time to debut: methodology note

**Scripts:** `python/06_dsl_time_to_debut.py` (model), `python/07_dsl_robustness.py` (stress tests)
**Status:** first real-data run, November 2026. Data: MLB Stats API, DSL seasons 2005–2026.
**Source idea:** Lee & Lee (2023), *Baseball Informatics: From MiLB to MLB Debut*.

## Question

Given only a hitter's first Dominican Summer League season, how likely is he to
debut in MLB within six years?

Draft models cannot answer this because international signees have no draft pick.

## Design

| Risk in earlier work | Choice here |
|---|---|
| Career totals reflect playing time teams already chose to give | Features come from the first DSL season only, and plate appearances are not a feature |
| The DSL run environment shifts over time | K%, BB% and ISO are ratios to that season's league |
| Small samples are noisy | Rates shrink toward the league with a 100 PA prior |
| Unresolved players look like failures | Cox model with censoring |
| Random splits leak the future | Train with labels stopped at the cutoff year, test later cohorts |
| Accuracy flatters a rare outcome | Report AUC, Brier score against the base rate, and calibration |

Players in the earliest loaded season are dropped because they may have played
in the DSL before the data starts. Hitters need at least 50 PA in that first season.

## Results

Default model: age, K%, BB% and ISO relative to league. Training labels stop at
2013; the test set is the 2014–2019 DSL cohorts (there was no 2020 season), each
followed for six years.

| Measure | Value |
|---|---|
| Test players / debuts within 6 years | 2,139 / 53 (2.5%) |
| AUC, model | 0.84 |
| AUC, age only | 0.62 |
| AUC gain, 95% bootstrap range | 0.14 to 0.29 |
| Brier score, model vs base rate only | 0.0229 vs 0.0243 |
| Debuts from the top fifth of predictions | 39 of 53 (74%) |

**Calibration.** The model ranks well but overstates the odds below the top
fifth: those four fifths were predicted at 1.2% to 4.2% and debuted at 0.5% to
1.4%. The top fifth was predicted at 7.7% and debuted at 9.1%. Mean predicted
3.6% against an actual 2.5%, in every cutoff tested, so debut rates for recent
cohorts appear to have fallen.

**Hazard ratios, cutoff 2013** (per unit of each ratio; 1.0 = league average):

| Factor | Hazard ratio | p |
|---|---|---|
| K% vs league | 0.10 | 0.001 |
| Age vs league (per year) | 0.69 | 0.013 |
| ISO vs league | 2.79 | 0.048 |
| BB% vs league | 0.44 | 0.24 |

## Robustness (module 07)

| Cutoff | Training debuts | Test cohorts | AUC with PA | AUC without PA | AUC age only |
|---|---|---|---|---|---|
| 2013 | 21 | 2014–2019 | 0.82 | 0.84 | 0.62 |
| 2015 | 43 | 2016–2019 | 0.79 | 0.82 | 0.58 |
| 2016 | 57 | 2017–2019 | 0.85 | 0.86 | 0.59 |

- **Plate appearances add nothing out of time.** PA is significant in sample
  (hazard ratio 2.3 to 2.6) yet the model without it ranks the later cohorts
  slightly better at every cutoff. It was dropped from the default model.
- **Power was being masked by playing time.** Without PA, the ISO hazard ratio
  rises to 2.8, 3.1 and 4.1 across the three cutoffs, all significant. With more
  training debuts it is the second clearest skill signal after strikeouts.
- **Strikeouts are the steadiest signal.** The K% hazard ratio stays between
  0.06 and 0.10 across cutoffs.
- **Walks never matter.** BB% is not significant at any cutoff, in line with
  earlier work on DSL walk rates.

The AUC differences between the with- and without-PA models are small and were
not tested formally; the gains over age alone are well outside noise.

## Run it

Run module 06 once on a computer with normal internet access; it caches the API
responses in `data/raw/milb`, and module 07 then works offline:

```bash
pip install -r requirements.txt
python python/06_dsl_time_to_debut.py
python python/07_dsl_robustness.py
python -m pytest tests -q
```

Results land in `output/tables/dsl_*.csv` and `output/figures/dsl_calibration.png`.
`--demo` runs on synthetic players and writes `_DEMO` files that git ignores.

## Limits

- Hitters only. Pitchers need their own feature set.
- Signing bonus is not in the MLB Stats API, so the model omits it.
- A debut is a team decision, and it says nothing about MLB value.
- The DSL lost its 2020 season, which affects those cohorts' development paths.
- Only 21 to 57 training debuts, so exact effect sizes stay uncertain.
- Reaching MLB is rare, so subgroup and single-player estimates are uncertain.
