"""01_acquire_data.py — Python mirror of R/01_acquire_data.R.

Builds the 2015-2025 hitter panel from Lahman CSVs in data/raw/.
(Locally you can instead use `pybaseball`; see footer.)
Run from the project root:  python python/01_acquire_data.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
COUNTING = ["G","AB","R","H","X2B","X3B","HR","RBI","SB","CS",
            "BB","SO","IBB","HBP","SH","SF","GIDP"]

bat = pd.read_csv(RAW / "Batting.csv")
ppl = pd.read_csv(RAW / "People.csv",
                  usecols=["playerID","nameFirst","nameLast","birthYear","bats"])

b = bat[bat.yearID.between(2015, 2025)].copy()
b[COUNTING] = b[COUNTING].fillna(0)

season = (b.groupby(["playerID","yearID"], as_index=False)[COUNTING].sum()
            .assign(PA =lambda d: d[["AB","BB","HBP","SH","SF"]].sum(axis=1),
                    X1B=lambda d: d.H - d.X2B - d.X3B - d.HR,
                    AVG=lambda d: d.H / d.AB,
                    OBP=lambda d: (d.H + d.BB + d.HBP) / (d.AB + d.BB + d.HBP + d.SF),
                    SLG=lambda d: (d.X1B + 2*d.X2B + 3*d.X3B + 4*d.HR) / d.AB)
            .assign(OPS=lambda d: d.OBP + d.SLG)
            .merge(ppl, on="playerID")
            .assign(Age=lambda d: d.yearID - d.birthYear))

season.to_csv(PROC / "hitters_2015_2025.csv", index=False)
print(f"Panel: {season.shape} | qualified (PA>=502) seasons: {(season.PA>=502).sum()}")
print(season[(season.yearID==2025) & (season.PA>=502)]
      .nlargest(5, "OPS")[["nameFirst","nameLast","Age","PA","HR","OPS"]])

# --- pybaseball alternative (run locally; needs internet to FG/Savant) ---
# from pybaseball import batting_stats, statcast
# fg = batting_stats(2015, 2025, qual=0)          # FanGraphs, incl. wOBA/wRC+/WAR
# sc = statcast("2025-04-01", "2025-04-07")       # pitch-level Statcast sample
