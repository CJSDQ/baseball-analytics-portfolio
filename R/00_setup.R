# 00_setup.R — one-time environment setup
# Run from the project root (use an RStudio Project or setwd() there).

pkgs <- c(
  "tidyverse",   # dplyr, ggplot2, tidyr, readr — core toolkit (SABR L3 stack)
  "Lahman",      # v14+: season-level database through 2025
  "baseballr",   # FanGraphs / Baseball Savant / Baseball-Reference scrapers
  "janitor",     # clean_names(), tabyl()
  "broom",       # tidy model outputs
  "gt",          # publication-quality tables
  "shiny"        # dashboard (Phase 6)
)

new <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(new)) install.packages(new)

invisible(lapply(pkgs, library, character.only = TRUE))

# Project directories (idempotent)
for (d in c("data/raw", "data/processed", "output/figures",
            "output/tables", "reports")) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

message("Setup complete. Lahman version: ",
        packageVersion("Lahman"),
        " | Max season in Batting: ", max(Lahman::Batting$yearID))
