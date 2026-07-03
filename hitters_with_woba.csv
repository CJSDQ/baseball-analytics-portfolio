# 02_advanced_metrics.R — Level 2 metrics, built from first principles
# Inputs : data/processed/hitters_2015_2025.csv, Lahman::Teams, Lahman::Pitching
# Outputs: hitters_with_woba.csv, pitchers_2015_2025.csv, park_factors.csv
#
# METHOD NOTES (for the report):
# * Linear weights are DERIVED by regressing team runs on team event counts
#   (2010–2025, excluding the 60-game 2020 season). This is the classic
#   approach when play-by-play RE24 data isn't available. Limitation:
#   multicollinearity slightly depresses the 1B coefficient vs. RE24-based
#   weights (~0.38 vs ~0.46); event coefficients otherwise match canon
#   (2B≈0.78, 3B≈0.97, HR≈1.42, BB≈0.33).
# * wOBA is scaled so league wOBA = league OBP each season (FanGraphs style).
# * FIP constant is computed per season from league totals: c = lgERA −
#   (13·HR + 3·(BB+HBP) − 2·K) / IP.
# * Park factors: Lahman's BPF/PPF (100 = neutral). These are multi-year,
#   run-based factors — coarse (no handedness or component splits), but fine
#   for season-level adjustment. Statcast park factors are a Phase 5 upgrade.

library(tidyverse)
library(Lahman)
library(broom)

# ---- 1) Linear weights via team-runs regression -----------------------------
t <- Teams |>
  filter(yearID %in% 2010:2025, yearID != 2020) |>
  mutate(X1B = H - X2B - X3B - HR,
         OUTS = AB - H + SF)

lw_fit <- lm(R ~ X1B + X2B + X3B + HR + BB + HBP + OUTS, data = t)
print(glance(lw_fit)$r.squared)          # ≈ 0.90
w <- coef(lw_fit)
print(round(w, 3))

# ---- 2) wOBA -----------------------------------------------------------------
hitters <- read_csv("data/processed/hitters_2015_2025.csv")

raw_num <- function(d) {
  w["BB"]*(d$BB - d$IBB) + w["HBP"]*d$HBP + w["X1B"]*d$X1B +
  w["X2B"]*d$X2B + w["X3B"]*d$X3B + w["HR"]*d$HR
}

lg <- hitters |>
  group_by(yearID) |>
  summarise(across(c(H,BB,IBB,HBP,AB,SF,X1B,X2B,X3B,HR), sum)) |>
  mutate(lgOBP = (H + BB + HBP) / (AB + BB + HBP + SF),
         raw   = raw_num(pick(everything())) / (AB + BB - IBB + SF + HBP),
         scale = lgOBP / raw)

hitters <- hitters |>
  left_join(select(lg, yearID, scale, lgOBP), by = "yearID") |>
  mutate(wOBA = raw_num(pick(everything())) /
                (AB + BB - IBB + SF + HBP) * scale)

write_csv(hitters, "data/processed/hitters_with_woba.csv")

# Sanity: 2025 leaders should be Judge (~.49), Ohtani, Raleigh
hitters |> filter(yearID == 2025, PA >= 502) |>
  slice_max(wOBA, n = 5) |>
  select(nameFirst, nameLast, PA, wOBA, OPS) |> print()

# ---- 3) FIP ------------------------------------------------------------------
pitchers <- Pitching |>
  filter(yearID %in% 2015:2025) |>
  mutate(across(c(HBP, BB, SO, HR, ER), ~ replace_na(.x, 0))) |>
  group_by(playerID, yearID) |>
  summarise(across(c(IPouts, HR, BB, HBP, SO, ER, G, GS, W, L, SV), sum),
            .groups = "drop") |>
  mutate(IP = IPouts / 3, ERA = 9 * ER / IP)

lgp <- pitchers |>
  group_by(yearID) |>
  summarise(across(c(IPouts, HR, BB, HBP, SO, ER), sum)) |>
  mutate(IP = IPouts / 3,
         lgERA = 9 * ER / IP,
         cFIP  = lgERA - (13*HR + 3*(BB + HBP) - 2*SO) / IP)

pitchers <- pitchers |>
  left_join(select(lgp, yearID, cFIP), by = "yearID") |>
  mutate(FIP = (13*HR + 3*(BB + HBP) - 2*SO) / IP + cFIP) |>
  left_join(select(People, playerID, nameFirst, nameLast, birthYear),
            by = "playerID") |>
  mutate(Age = yearID - birthYear)

write_csv(pitchers, "data/processed/pitchers_2015_2025.csv")

# Sanity: 2025 FIP leaders (≥162 IP) should start Skenes, Skubal
pitchers |> filter(yearID == 2025, IP >= 162) |>
  slice_min(FIP, n = 5) |>
  select(nameFirst, nameLast, IP, ERA, FIP) |> print()

# ---- 4) Park factors ----------------------------------------------------------
Teams |>
  filter(yearID %in% 2015:2025) |>
  select(yearID, teamID, name, BPF, PPF) |>
  write_csv("data/processed/park_factors.csv")
