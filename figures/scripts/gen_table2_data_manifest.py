"""Table 2 — Per-day ADS-B data manifest (LaTeX)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "figures" / "paper" / "table2_data_manifest.tex"

# Hard-coded from data-engineer's final M1 report (per refine-logs/EXPERIMENT_TRACKER.md
# and paper/narrative_report.md §3). Source: adsb.lol globe_history_2024.
ROWS = [
    # (date, klax_rows, klax_ac, klax_mb, ksfo_rows, ksfo_ac, ksfo_mb)
    ("2024-08-02", 1_124_625, 3763, 31.0, 851_836, 2069, 21.8),
    ("2024-08-09", 1_144_920, 3851, 31.8, 826_573, 2084, 21.1),
    ("2024-08-16",   565_660, 1834, 16.1, 404_611, 1004, 11.4),
    ("2024-08-23",   498_990, 1632, 14.5, 366_808,  899, 10.4),
    ("2024-08-30",   456_486, 1523, 13.4, 342_184,  865,  9.9),
]


def main() -> int:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Real-data ADS-B inventory per Friday in August 2024, "
        r"acquired from the open \texttt{adsb.lol} historical archive. "
        r"Within a 30 NM ARP-centred bounding box. "
        r"Note the $\sim$50\% coverage drop after 2024-08-09 at both airports, "
        r"reflecting fewer volunteer receivers on the later Fridays.}",
        r"\label{tab:data}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{LAX (KLAX)} & \multicolumn{3}{c}{SFO (KSFO)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Date (Fri) & rows & a/c & MB & rows & a/c & MB \\",
        r"\midrule",
    ]
    for d, kr, ka, km, sr, sa, sm in ROWS:
        lines.append(
            f"{d} & {kr:,} & {ka:,} & {km:.1f} & {sr:,} & {sa:,} & {sm:.1f} \\\\"
        )
    lines += [
        r"\midrule",
        r"\textbf{Sum} & \textbf{"
        + f"{sum(r[1] for r in ROWS):,}"
        + r"} & & & \textbf{"
        + f"{sum(r[4] for r in ROWS):,}"
        + r"} & & \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"  saved {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
