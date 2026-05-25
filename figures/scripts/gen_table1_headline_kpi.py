"""Table 1 — Headline KPI table (LaTeX)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

OUT = ROOT / "figures" / "paper" / "table1_headline_kpi.tex"


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("baseline").agg(
        n=("feasible", "size"),
        feasible=("feasible", "mean"),
        mean_length_m=("length_m", lambda s: s[df.loc[s.index, "feasible"]].mean()
                       if df.loc[s.index, "feasible"].any() else float("nan")),
        mean_pops=("n_expansions", "mean"),
    )
    g["feasible"] *= 100
    return g


def main() -> int:
    frames = {}
    for ap in ("KLAX", "KSFO"):
        p = ROOT / "results" / "eval" / ap / "kpi_table.parquet"
        if p.exists():
            frames[ap] = pd.read_parquet(p)
    if not frames:
        print("No KPI tables")
        return 1

    klax = _agg(frames["KLAX"])
    ksfo = _agg(frames["KSFO"])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Real-data DREAM headline. Per-baseline corridor counts ($n=180$), "
        r"feasibility (\%, fraction of 12 vertiport pairs $\times$ 3 hours $\times$ "
        r"5 Aug-2024 Fridays that the planner returns a feasible corridor for), "
        r"mean corridor length (km, over feasible only), and mean A* expansions.}",
        r"\label{tab:headline}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{4}{c}{LAX (KLAX)} & \multicolumn{4}{c}{SFO (KSFO)} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"Baseline & $n$ & feas.\% & len.\,km & pops & "
        r"$n$ & feas.\% & len.\,km & pops \\",
        r"\midrule",
    ]
    for b in ("B0", "B1", "B2", "B3"):
        kl = klax.loc[b]
        ks = ksfo.loc[b]

        def fmt(v, fmtspec="{:.1f}"):
            try:
                if pd.isna(v):
                    return "—"
                return fmtspec.format(v)
            except Exception:
                return "—"

        klen = fmt(kl["mean_length_m"] / 1000.0)
        kpops = fmt(kl["mean_pops"], "{:.0f}")
        kfeas = fmt(kl["feasible"], "{:.1f}")
        slen = fmt(ks["mean_length_m"] / 1000.0)
        spops = fmt(ks["mean_pops"], "{:.0f}")
        sfeas = fmt(ks["feasible"], "{:.1f}")
        lines.append(
            f"{b} & {int(kl['n'])} & {kfeas} & {klen} & {kpops} & "
            f"{int(ks['n'])} & {sfeas} & {slen} & {spops} \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{9}{l}{\footnotesize B3 vs B2 feasibility lift: "
        r"\textbf{2.0$\times$} LAX, \textbf{4.5$\times$} SFO.} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"  saved {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
