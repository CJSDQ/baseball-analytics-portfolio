# Decision Window — methodology note

**Script:** `python/04_decision_window.py`
**Source idea:** Chris Black's breakdown of the 2026 Milwaukee Brewers offense (Oct 2026).

## Hypothesis

A hitter can buy decision time by catching the ball deeper: a short, fast swing
lets him commit later, and a deep contact point plus opposite-field attack
direction is what that swing looks like. The price is being late on fastballs
more often. The payoff is being early on breaking balls less often.

## The trade-off depends on pitch mix

Being late on a fastball is not automatically the cheaper error. It is cheaper
only if the league throws enough breaking balls. Let

- `cost_fb` = run value lost per contact when late vs. on time on fastballs
- `cost_brk` = run value lost per contact when early vs. on time on breaking balls
- `s` = breaking-ball share of (fastball + breaking) pitches

Sitting back breaks even when `s · cost_brk = (1 − s) · cost_fb`, so

```
s* = cost_fb / (cost_fb + cost_brk)
```

When the league's actual `s` is above `s*`, accepting late fastballs is the
better bet. When pitchers lean on fastballs (`s < s*`), the same approach loses
runs. This makes the strategy league- and era-specific, which matters for
comparing MLB with LIDOM, LMB or NPB pitch mixes.

## Metrics

| Metric | Definition |
|---|---|
| `bat_speed`, `swing_length` | Mean over tracked swings |
| `short_fast` | Bat speed ≥ 60th pctl **and** swing length ≤ 40th pctl, within season |
| `attack_angle` | Mean vertical attack angle (Black: Brewers lowest at 7°) |
| `attack_dir_pull` | Horizontal attack direction, oriented so **+ = pull side** |
| `contact_depth` | Ball–batter intercept (y), oriented so **+ = further out front** |
| `early_*` / `late_*` | Contact depth in the front / back tercile of the league distribution for that pitch group and season |
| `brk_rv100` | Run value per 100 breaking balls, hitter perspective |
| `brk_pre2k_*` | Same, before two strikes (Black's strongest split) |

Sign conventions for intercept and attack direction are **derived from the
data** (correlation with pulled spray angle) rather than assumed; the script
prints the orientation it used on every run.

Pitch groups: fastballs = FF, SI, FC. Breaking ("spin") = SL, ST, SV, CU, KC, CS.
Offspeed = CH, FS, FO, SC.

## Model

Weighted least squares at the hitter-season level, robust (HC1) errors:

```
brk_rv100 ~ not_early_brk + bat_speed + swing_length + season
```

The question: does avoiding early contact on spin predict production against
spin **beyond** what bat speed and swing length already explain?

## Known limitations

1. **Whiffs carry no contact point.** Timing is classified on contact only, so
   early-on-spin whiffs are invisible. That understates `cost_brk`, which makes
   `s*` an **upper bound**.
2. **Tercile timing is a proxy.** Savant's own early/on-time/late labels are not
   in the public pitch CSV.
3. **Box depth is not in the pitch file.** Batter positioning lives on a
   separate Savant leaderboard; join it manually if needed.
4. **Coverage.** Bat tracking exists for MLB parks only (2024+ for speed/length,
   2025+ for attack angle/direction and intercept). No LMB, NPB or LIDOM data.
5. **Correlation, not causation.** Good swing deciders may also choose deeper
   contact; the model cannot separate the two.

## Running

```bash
pip install pybaseball statsmodels matplotlib pandas
python python/04_decision_window.py --seasons 2025 2026   # downloads & caches Savant data
python python/04_decision_window.py --demo                 # synthetic pipeline check
```

Outputs go to `output/tables/decision_window_*.csv` and
`output/figures/decision_window_*.png`. Demo outputs are suffixed `_DEMO` and
gitignored.
