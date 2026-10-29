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
├── python/             # Parallel/validation pipeline
│   ├── 01_acquire_data.py
│   ├── 04_decision_window.py   # Statcast bat-tracking: swing profile, timing, pitch-mix breakeven
│   └── 05_decisions_vs_damage.py  # Where did a hitter break? Decisions → Inputs → Delivery → Contact → Outcomes
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
| FanGuards leaderboards | `baseballr::fg_batter_leaders()` | wOBA/wRC+/WAR benchmarks |
| Baseball Savant (Statcast) | `baseballr::statcast_search()` / `pybaseball` | Pitch-level data, xStats |
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
