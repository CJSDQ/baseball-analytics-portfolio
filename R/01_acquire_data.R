# 01_acquire_data.R — build the 2015–2025 hitter analysis panel
# Output: data/processed/hitters_2015_2025.csv
#
# Key cleaning decisions (documented for the report):
#   * Stints: players traded mid-season have multiple rows per year;
#     we SUM counting stats across stints so each row = one player-season.
#   * PA is derived (AB + BB + HBP + SH + SF) — Lahman doesn't store it.
#   * NA counting stats (rare, mostly pre-1955 SF/IBB) are treated as 0;
#     harmless for 2015+ but keeps the code reusable for history.

library(tidyverse)
library(Lahman)

YEARS <- 2015:2025

counting <- c("G","AB","R","H","X2B","X3B","HR","RBI","SB","CS",
              "BB","SO","IBB","HBP","SH","SF","GIDP")

hitters <- Batting |>
  filter(yearID %in% YEARS) |>
  mutate(across(all_of(counting), ~ replace_na(.x, 0))) |>
  group_by(playerID, yearID) |>
  summarise(across(all_of(counting), sum), .groups = "drop") |>
  mutate(
    PA  = AB + BB + HBP + SH + SF,
    X1B = H - X2B - X3B - HR,
    AVG = H / AB,
    OBP = (H + BB + HBP) / (AB + BB + HBP + SF),
    SLG = (X1B + 2*X2B + 3*X3B + 4*HR) / AB,
    OPS = OBP + SLG
  ) |>
  left_join(
    People |> select(playerID, nameFirst, nameLast, birthYear, bats),
    by = "playerID"
  ) |>
  mutate(Age = yearID - birthYear)   # baseball "seasonal age" refinement in Phase 2

write_csv(hitters, "data/processed/hitters_2015_2025.csv")

# --- Sanity check: should show Judge (1.144 OPS) atop 2025 ---
hitters |>
  filter(yearID == 2025, PA >= 502) |>
  slice_max(OPS, n = 5) |>
  select(nameFirst, nameLast, Age, PA, HR, AVG, OBP, SLG, OPS) |>
  print()

# --- Optional: FanGraphs benchmarks for later model validation ---
# library(baseballr)
# fg_2025 <- fg_batter_leaders(startseason = "2025", endseason = "2025", qual = "y")
# write_csv(fg_2025, "data/raw/fg_batters_2025.csv")
