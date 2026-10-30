# Baseball Analytics Portfolio

A SABR-style analytics project combining advanced metrics (Level 2 concepts) with
programming skills (Level 3): data pipelines, metric replication, player projection
modeling, a custom WAR variant, and an interactive dashboard.

## Structure

```
baseball-analytics-portfolio/
├── data/
│   ├── raw/            # Lahman core tables (Batting, Pitching, Fielding,
│   │                   #   People, Teams, Appearances) — seasons through 2025
│   └── processed/      # Cleaned, merged analysis panels
├── R/                  # Primary pipeline (mirrors SABR course workflow)
│   ├── 00_setup.R      # Package installation & project options
│   └── 01_acquire_data.R
├── python/             # Parallel/validation pipeline + Statcast modules
│   ├── 01_acquire_data.py
│   ├── statcast_io.py              # cached Baseball Savant pulls (pybaseball)
│   ├── 04_decision_window.py       # Statcast bat-tracking: swing profile, timing, pitch-mix breakeven
│   ├── 04_pitch_cost.py            # pitch cost & two-strike foul skill
│   ├── 05_decisions_vs_damage.py   # Where did a hitter break? Decisions → Inputs → Delivery → Contact → Outcomes
│   └── 05_opencommand_validation.py # independent test of the OpenCommand metric
├── tests/              # pytest: python -m pytest tests -q
├── output/
│   ├── figures/        # ggplot2 / matplotlib exports
│   └── tables/         # Summary tables for the report
└── reports/            # Final write-up (Rmd/Quarto → pdf/docx)
    ├── decision_window.md      # Methodology for the Decision Window module
    └── decisions_vs_damage.md  # Methodology + Kyle Tucker 2023/2025 vs 2026 case study
```

## Data sources (all public)

| Source | Access | Used for |
|---|---|---|
| Lahman / Chadwick Baseball Databank | `Lahman` R pkg (v14, thru 2025) | Season-level stats, bio, teams |
| FanGraphs leaderboards | `baseballr::fg_batter_leaders()` | wOBA/wRC+/WAR benchmarks |
| Baseball Savant (Statcast) | `baseballr::statcast_search()` / `pybaseball` | Pitch-level data, xStats |
| OpenCommand (Kim, 2026) | Hugging Face `tomdoyo/open-command` | Catcher targets & pitcher command (CC BY-NC-SA 4.0) |
| Retrosheet | `retrosheet` pkg (optional) | Play-by-play, run expectancy |

## Roadmap

1. ✅ Setup & data foundations
2. ✅ EDA + advanced metrics (regression linear weights, wOBA, FIP, park factors, aging curve)
3. Projection model: Marcel baseline → weighted regression → ML comparison
4. Custom WAR variant / scouting index
5. Dashboard (Shiny) + final report
6. Decision Window study — short/fast swings, deep contact, "not early" timing,
   and the breaking-ball share at which sitting back pays (`python/04_decision_window.py`)
7. Decisions vs. Damage — bootstrap diagnostic of where a hitter's production broke,
   replicating Driveline's Kyle Tucker 2026 vs 2023 read (`python/05_decisions_vs_damage.py`)
8. 🔄 Statcast extensions: pitch cost (04) and command-metric validation (05)

## Statcast modules

### 04 — Pitch cost (`python/04_pitch_cost.py`)
Velocity buys whiffs; foul balls take outs back. Prompted by Jacob Misiorowski's
2026 NLDS Game 1 (a 104.8 mph first pitch, out after four innings because San
Diego kept fouling him off). Measures:

- **Pitchers** — pitches/PA, two-strike foul rate allowed, putaway%, and
  *pitches/PA over expected* given K% (isolates arms whose outs are expensive).
- **Hitters** — two-strike foul rate overall and vs 98+ mph fastballs, z-contact,
  bat speed, swing length; which trait predicts fouling off elite velocity
  (simple r and standardized joint OLS); split-half reliability of each metric.
- **Case study** — Misiorowski's NLDS G1 by plate appearance vs his
  regular-season baseline.

### 05 — OpenCommand validation (`python/05_opencommand_validation.py`)
OpenCommand scores command as median miss distance from the catcher's glove,
inferred from broadcast video. This module re-tests it independently:
correlations with BB%, Stuff+ and xERA (bootstrap CIs, naive vs inferred
target); whether command adds anything to xERA once Stuff+ and BB% are known
(nested OLS, partial correlations); year-over-year stability vs BB%; and how
far glove-ignoring pitchers (e.g. Misiorowski) sit from what their walk rate
predicts.

> OpenCommand data © Tom Kim, CC BY-NC-SA 4.0 —
> [github.com/tomdoyo/open-command](https://github.com/tomdoyo/open-command).
> Used here non-commercially with attribution.

### Running

```bash
pip install -r requirements.txt
python python/04_pitch_cost.py --season 2026          # first run downloads Statcast (slow, cached)
python python/05_opencommand_validation.py --seasons 2025 2026
python -m pytest tests -q                              # synthetic-data checks, no network
```
Outputs land in `output/tables/` and `output/figures/`. Raw downloads go to
`data/raw/statcast/` and `data/raw/opencommand/` (gitignored).

