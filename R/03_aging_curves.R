# 03_aging_curves.R — hitter aging curve via the delta method
# Input : data/processed/hitters_with_woba.csv
# Output: data/processed/season_pairs.csv, output/figures/aging_curve.png
#
# METHOD NOTES:
# * Delta method: for every player with consecutive seasons of ≥200 PA,
#   measure the change in wOBA from age A to A+1; average those deltas
#   (weighted by the harmonic mean of the two seasons' PA) at each age;
#   cumulate to build the curve, anchored so the peak = 0.
# * Known limitation — SURVIVORSHIP BIAS: players who decline hard enough
#   to lose their jobs never record the "next season," so observed old-age
#   declines are, if anything, UNDERSTATED. Discuss in the report; a
#   correction (e.g., phantom seasons / regression augmentation) is a
#   possible Phase 5 extension.
# * Finding on 2015–2025 data: peak ≈ age 26, roughly −5 to −12 points of
#   wOBA per year through the 30s — consistent with modern research showing
#   earlier peaks than the traditional 27–29 claim.

library(tidyverse)

hitters <- read_csv("data/processed/hitters_with_woba.csv")

pairs <- hitters |>
  select(playerID, yearID, Age, PA, wOBA) |>
  drop_na() |>
  arrange(playerID, yearID) |>
  group_by(playerID) |>
  mutate(next_year = lead(yearID),
         wOBA2     = lead(wOBA),
         PA2       = lead(PA)) |>
  ungroup() |>
  filter(next_year == yearID + 1, PA >= 200, PA2 >= 200) |>
  mutate(w      = 2 / (1/PA + 1/PA2),
         delta  = wOBA2 - wOBA,
         age_to = Age + 1)

write_csv(pairs, "data/processed/season_pairs.csv")

curve <- pairs |>
  group_by(age_to) |>
  summarise(delta = weighted.mean(delta, w), n = n()) |>
  filter(n >= 30) |>
  mutate(cum = cumsum(delta) - max(cumsum(delta)))

peak_age <- curve$age_to[which.max(curve$cum)]

p <- ggplot(curve, aes(age_to, cum)) +
  geom_line(color = "#1f4e79") +
  geom_point(color = "#1f4e79") +
  geom_vline(xintercept = peak_age, linetype = "dashed", alpha = .5) +
  annotate("text", x = peak_age + .3, y = -0.005,
           label = paste("peak ≈", peak_age), hjust = 0) +
  labs(title = "MLB Hitter Aging Curve, 2015–2025",
       subtitle = "Delta method, min 200 PA in consecutive seasons; PA-weighted",
       x = "Age", y = "Cumulative wOBA relative to peak") +
  theme_minimal()

ggsave("output/figures/aging_curve.png", p, width = 8, height = 5, dpi = 150)
print(curve)
