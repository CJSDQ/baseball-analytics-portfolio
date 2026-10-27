"""05_opencommand_validation.py — Does a vision-based command metric measure a skill?

OpenCommand (Tom Kim, 2026; github.com/tomdoyo/open-command) estimates the
catcher's target from broadcast video and scores command as the pitcher's
median miss distance (inches) from that target. Lower = better command.

This module independently re-tests its claims and pushes past them:
  1. Replicate: correlations of miss distance with BB%, Stuff+, xERA
     (bootstrap 95% CIs, naive vs inferred target).
  2. The puzzle: command tracks walks (~+0.55) but barely tracks xERA. Test
     whether command adds anything to run prevention once Stuff+ and BB%
     are known (incremental R², partial correlations).
  3. Skill or noise: year-over-year stability of median miss vs. BB%.
  4. Known blind spot: pitchers who ignore the glove (the author names
     Jacob Misiorowski) — where does he land, and is he an outlier vs. his BB%?

Data: CC BY-NC-SA 4.0 — cite OpenCommand; non-commercial use only.
  Files are fetched from the Hugging Face dataset tomdoyo/open-command, or put
  them by hand in data/raw/opencommand/ as command_scores_<season>.csv and
  fg_pitching_<season>.csv(.gz).

Run from the project root:
  python python/05_opencommand_validation.py --seasons 2025 2026
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/opencommand"
TABLES, FIGS = ROOT / "output/tables", ROOT / "output/figures"
REPO_ID = "tomdoyo/open-command"
GLOVE_IGNORERS = {"jacob misiorowski"}
RNG = np.random.default_rng(2026)


# ------------------------------------------------------------------ loading
def _fetch(candidates: list[str], local: Path) -> Path:
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    RAW.mkdir(parents=True, exist_ok=True)
    errors = []
    for fn in candidates:
        try:
            p = Path(hf_hub_download(REPO_ID, fn, repo_type="dataset"))
            local.write_bytes(p.read_bytes())
            return local
        except Exception as e:  # noqa: BLE001 — try the next known layout
            errors.append(f"{fn}: {type(e).__name__}")
    raise FileNotFoundError(f"Could not fetch {local.name} ({'; '.join(errors)}). "
                            f"Download it manually into {RAW}.")


def load_command(season: int) -> pd.DataFrame:
    p = _fetch([f"{season}/command_scores.csv", f"data/{season}/command_scores.csv"],
               RAW / f"command_scores_{season}.csv")
    return pd.read_csv(p).assign(season=season)


def load_fangraphs(season: int) -> pd.DataFrame:
    p = _fetch([f"fangraphs/fg_pitching_{season}.csv.gz", f"fg_pitching_{season}.csv.gz",
                f"data/fg_pitching_{season}.csv.gz"],
               RAW / f"fg_pitching_{season}.csv.gz")
    return pd.read_csv(p).assign(season=season)


# -------------------------------------------------------------- harmonizing
def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s.lower())
    return re.sub(r"[^a-z ]", "", s).strip()


FG_COLS = {  # canonical → accepted spellings in FanGraphs exports
    "name": ["Name", "PlayerName", "name"],
    "ip": ["IP", "ip"],
    "bb_pct": ["BB%", "BB_pct", "bb_pct", "BB%_"],
    "stuff_plus": ["Stuff+", "sp_stuff", "Stuff_plus", "stuff_plus"],
    "xera": ["xERA", "xera", "XERA"],
    "era": ["ERA", "era"],
}


def harmonize_fg(fg: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"season": fg["season"]})
    for canon, opts in FG_COLS.items():
        col = next((c for c in opts if c in fg.columns), None)
        out[canon] = fg[col] if col else np.nan
    if out.bb_pct.dtype == object:
        out["bb_pct"] = pd.to_numeric(out.bb_pct.astype(str).str.rstrip("% "), errors="coerce")
    if out.bb_pct.dropna().gt(1).mean() > .5:           # "8.2" → 0.082
        out["bb_pct"] = out.bb_pct / 100
    out["key"] = out.name.map(norm_name)
    return out


def build_panel(cmd: pd.DataFrame, fg: pd.DataFrame, min_ip: float = 50) -> pd.DataFrame:
    c = cmd[cmd.pitch_type.eq("ALL")].rename(columns={"n": "n_pitches"})
    c = c.assign(key=c.pitcher.map(norm_name))
    f = harmonize_fg(fg)
    # Name collisions inside a season (two "Luis Garcia"s) can't be resolved
    # by name — drop them rather than mis-join.
    f = f[~f.duplicated(["season", "key"], keep=False)]
    p = c.merge(f, on=["season", "key"], how="inner")
    return p[p.ip >= min_ip].reset_index(drop=True)


# ----------------------------------------------------------------- statistics
def boot_corr(x: pd.Series, y: pd.Series, n: int = 2000) -> tuple[float, float, float]:
    m = pd.concat([x, y], axis=1).dropna().to_numpy()
    if len(m) < 10:
        return (np.nan,) * 3
    r = np.corrcoef(m.T)[0, 1]
    idx = RNG.integers(0, len(m), (n, len(m)))
    bs = [np.corrcoef(m[i].T)[0, 1] for i in idx]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return r, lo, hi


def resid(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    beta, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    return pd.Series(y.to_numpy() - A @ beta, index=y.index)


def correlations(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, g in [("pooled", p), *p.groupby("season")]:
        for target in ["bb_pct", "stuff_plus", "xera"]:
            for miss in ["naive_in", "inferred_in"]:
                r, lo, hi = boot_corr(g[miss], g[target])
                rows.append({"season": season, "miss": miss, "vs": target,
                             "r": r, "ci_lo": lo, "ci_hi": hi, "n": g[[miss, target]].dropna().shape[0]})
        # partial correlations: the "puzzle" tests
        for label, ctrl in [("xera | stuff_plus", ["stuff_plus"]),
                            ("xera | stuff_plus, bb_pct", ["stuff_plus", "bb_pct"])]:
            q = g.dropna(subset=["inferred_in", "xera", *ctrl])
            if len(q) >= 10:
                r, lo, hi = boot_corr(resid(q.inferred_in, q[ctrl]), resid(q.xera, q[ctrl]))
                rows.append({"season": season, "miss": "inferred_in", "vs": label,
                             "r": r, "ci_lo": lo, "ci_hi": hi, "n": len(q)})
    return pd.DataFrame(rows)


def incremental_r2(p: pd.DataFrame) -> pd.DataFrame:
    """Nested OLS on xERA: how much does command add beyond stuff and walks?"""
    import statsmodels.formula.api as smf

    q = p.dropna(subset=["xera", "stuff_plus", "bb_pct", "inferred_in"])
    if len(q) < 20:
        return pd.DataFrame({"note": [f"only {len(q)} complete rows"]})
    specs = {
        "stuff": "xera ~ stuff_plus",
        "stuff + command": "xera ~ stuff_plus + inferred_in",
        "stuff + walks": "xera ~ stuff_plus + bb_pct",
        "stuff + walks + command": "xera ~ stuff_plus + bb_pct + inferred_in",
    }
    fits = {k: smf.ols(v, data=q).fit() for k, v in specs.items()}
    return pd.DataFrame([{
        "model": k, "r2": f.rsquared, "adj_r2": f.rsquared_adj, "n": int(f.nobs),
        "command_coef_per_inch": f.params.get("inferred_in", np.nan),
        "command_p": f.pvalues.get("inferred_in", np.nan),
    } for k, f in fits.items()])


def stability(p: pd.DataFrame, cmd: pd.DataFrame, min_pitches: int = 500) -> pd.DataFrame:
    """Year-over-year r for median miss (all pitchers with enough pitches)
    and for BB% (same pitchers) — a skill should repeat."""
    seasons = sorted(cmd.season.unique())
    rows = []
    for a, b in zip(seasons, seasons[1:]):
        c = cmd[cmd.pitch_type.eq("ALL") & (cmd.n >= min_pitches)]
        w = c.pivot_table(index="pitcher", columns="season", values="inferred_in")
        w = w[[a, b]].dropna() if {a, b} <= set(w.columns) else pd.DataFrame()
        rows.append({"pair": f"{a}->{b}", "metric": "inferred_in",
                     "r": w[a].corr(w[b]) if len(w) >= 10 else np.nan, "n": len(w)})
        bb = p.pivot_table(index="key", columns="season", values="bb_pct")
        bb = bb[[a, b]].dropna() if {a, b} <= set(bb.columns) else pd.DataFrame()
        rows.append({"pair": f"{a}->{b}", "metric": "bb_pct",
                     "r": bb[a].corr(bb[b]) if len(bb) >= 10 else np.nan, "n": len(bb)})
    return pd.DataFrame(rows)


def glove_ignorer_check(p: pd.DataFrame) -> pd.DataFrame:
    """Where do known glove-ignorers sit on miss distance vs. what their
    walk rate predicts? A large positive residual = metric overstates misses."""
    rows = []
    for season, g in p.groupby("season"):
        g = g.dropna(subset=["inferred_in", "bb_pct"])
        if len(g) < 10:
            continue
        b = np.polyfit(g.bb_pct, g.inferred_in, 1)
        g = g.assign(pctile=g.inferred_in.rank(pct=True),
                     resid_vs_bb=g.inferred_in - np.polyval(b, g.bb_pct))
        sd = g.resid_vs_bb.std()
        hit = g[g.key.isin(GLOVE_IGNORERS)]
        for _, r in hit.iterrows():
            rows.append({"season": season, "pitcher": r.pitcher, "inferred_in": r.inferred_in,
                         "miss_percentile": r.pctile, "bb_pct": r.bb_pct,
                         "resid_vs_bb_in": r.resid_vs_bb, "resid_z": r.resid_vs_bb / sd})
    return pd.DataFrame(rows)


def by_pitch_type(cmd: pd.DataFrame, min_n: int = 50) -> pd.DataFrame:
    c = cmd[~cmd.pitch_type.eq("ALL") & (cmd.n >= min_n)]
    return (c.groupby(["season", "pitch_type"]).inferred_in
              .describe(percentiles=[.1, .5, .9])
              .rename(columns={"count": "pitchers"}).reset_index())


# --------------------------------------------------------------------- plot
def plot_bb(p: pd.DataFrame, season: int) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = p[p.season.eq(season)].dropna(subset=["inferred_in", "bb_pct", "xera"])
    if len(g) < 10:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, (col, lab) in zip(axes, [("bb_pct", "BB%"), ("xera", "xERA")]):
        ax.scatter(g.inferred_in, g[col], s=12, alpha=.5, color="#2a6f97")
        b = np.polyfit(g.inferred_in, g[col], 1)
        xs = np.linspace(g.inferred_in.min(), g.inferred_in.max(), 50)
        ax.plot(xs, np.polyval(b, xs), color="#c44536", lw=1.5)
        for _, r in g[g.key.isin(GLOVE_IGNORERS)].iterrows():
            ax.annotate(r.pitcher, (r.inferred_in, r[col]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Inferred median miss (in) — lower = better command")
        ax.set_ylabel(lab)
        ax.set_title(f"r = {g.inferred_in.corr(g[col]):+.2f}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"OpenCommand miss distance vs walks and run prevention ({season}, "
                 f"n={len(g)}, min 50 IP)", fontsize=11)
    fig.tight_layout()
    path = FIGS / f"opencommand_bb_xera_{season}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------- main
def run(cmd: pd.DataFrame, fg: pd.DataFrame, min_ip: float = 50) -> dict:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    p = build_panel(cmd, fg, min_ip)
    res = {
        "opencommand_panel": p,
        "opencommand_correlations": correlations(p),
        "opencommand_incremental_r2": incremental_r2(p),
        "opencommand_stability": stability(p, cmd),
        "opencommand_glove_ignorers": glove_ignorer_check(p),
        "opencommand_by_pitch_type": by_pitch_type(cmd),
    }
    for name, df in res.items():
        df.to_csv(TABLES / f"{name}.csv", index=False)
    for s in sorted(p.season.unique()):
        plot_bb(p, int(s))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    ap.add_argument("--min-ip", type=float, default=50)
    a = ap.parse_args()

    cmd = pd.concat([load_command(s) for s in a.seasons], ignore_index=True)
    fg = pd.concat([load_fangraphs(s) for s in a.seasons], ignore_index=True)
    out = run(cmd, fg, a.min_ip)

    pd.set_option("display.width", 140)
    print(f"Matched pitchers: {len(out['opencommand_panel'])}")
    for k in ["opencommand_correlations", "opencommand_incremental_r2",
              "opencommand_stability", "opencommand_glove_ignorers"]:
        print(f"\n{k}:\n{out[k].round(3).to_string(index=False)}")
