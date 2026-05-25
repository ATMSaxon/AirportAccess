"""Table 4 — Annex 14 OLS parameters used (LaTeX)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml

OUT = ROOT / "figures" / "paper" / "table4_annex14_params.tex"


def main() -> int:
    cfg = yaml.safe_load(
        (ROOT / "configs" / "annex14" / "code4_precision.yaml").read_text()
    )

    rows = [
        ("Approach surface (inner edge / divergence)",
         f"{cfg['approach_surface']['inner_edge_width_m']:.0f} m / "
         f"{cfg['approach_surface']['divergence_each_side']:.3f} rad"),
        ("Approach section 1 (slope $\\times$ length)",
         f"1:{1.0/cfg['approach_surface']['slope_first_section']:.0f} "
         f"$\\times$ {cfg['approach_surface']['length_first_section_m']:.0f} m"),
        ("Approach section 2 (slope $\\times$ length)",
         f"1:{1.0/cfg['approach_surface']['slope_second_section']:.0f} "
         f"$\\times$ {cfg['approach_surface']['length_second_section_m']:.0f} m"),
        ("Takeoff-climb (slope $\\times$ length)",
         f"1:{1.0/cfg['takeoff_climb_surface']['slope']:.0f} "
         f"$\\times$ {cfg['takeoff_climb_surface']['total_length_m']:.0f} m"),
        ("Transitional slope",
         f"1:{1.0/cfg['transitional_surface']['slope']:.0f}"),
        ("Inner horizontal (radius / height)",
         f"{cfg['inner_horizontal_surface']['radius_m']:.0f} m / "
         f"{cfg['inner_horizontal_surface']['height_above_arp_m']:.0f} m"),
        ("Conical (slope / height)",
         f"1:{1.0/cfg['conical_surface']['slope']:.0f} / "
         f"{cfg['conical_surface']['height_above_inner_horizontal_m']:.0f} m"),
        ("Runway strip (half-width / end length)",
         f"{cfg['runway_strip']['width_each_side_m']:.0f} m / "
         f"{cfg['runway_strip']['end_length_m']:.0f} m"),
        ("Runway end safety area (RESA)",
         f"{cfg['resa']['length_m']:.0f} $\\times$ "
         f"{cfg['resa']['width_m']:.0f} m"),
        ("OFZ inner-approach (slope $\\times$ length)",
         f"1:{1.0/cfg['ofz_inner_approach']['slope']:.0f} "
         f"$\\times$ {cfg['ofz_inner_approach']['length_m']:.0f} m"),
        ("OFZ inner-transitional slope",
         f"1:{1.0/cfg['ofz_inner_transitional']['slope']:.0f}"),
        ("Vertiport approach/departure (slope, FAA EB-105A)",
         f"1:{1.0/cfg['vertiport_approach_departure']['slope']:.0f}"),
        ("Vertiport OFV (FATO half-side / height)",
         f"{cfg['vertiport_ofv']['fato_side_m']:.0f} m / "
         f"{cfg['vertiport_ofv']['ofv_height_m']:.0f} m"),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Annex 14 OLS parameters used by DREAM "
        r"(generic Code 4 precision approach). These are public-secondary-source "
        r"placeholder values, not a copy of the ICAO normative text. "
        r"Per-airport calibration is supported via the YAML override in "
        r"\texttt{configs/annex14/}.}",
        r"\label{tab:ols-params}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Parameter & Value \\",
        r"\midrule",
    ]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"  saved {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
