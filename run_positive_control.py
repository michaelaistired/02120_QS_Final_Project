"""Run every positive control and write a report to results/positive_control_report.md.

    python run_positive_control.py

Steps:
    1) deterministic control (paper Fig. 4 setup) + correctness checks
    2) time-step convergence (tau vs tau/2)
    3) paper Fig. 4 perturbation test (alpha_U, beta_U, beta_L raised 20-50%)
    4) statistical control: single-dose paper experiment, 4 seeds, compared to paper numbers
Takes about 5 minutes.
"""

import json
from pathlib import Path

from qs_model.controls import (deterministic_checks, deterministic_control, perturbation_test,
                               statistical_control, time_step_convergence)
from qs_model.params import PAPER

OUT = Path(__file__).parent / "results"


def mark(ok) -> str:
    """PASS/FAIL label for the report (INFO for observations that are not checks)."""
    if ok is None:
        return "INFO"
    return "PASS" if ok else "FAIL"


def main() -> None:
    """Run all controls, print a summary, and save markdown + JSON reports."""
    OUT.mkdir(exist_ok=True)
    lines = ["# Positive-control report", "",
             "Model: Kuttler & Maslovskaya (2022), re-implemented in `qs_model/` (paper mode, "
             "feedback OFF).", ""]

    print("1) deterministic control ...")
    det = deterministic_checks(deterministic_control(PAPER, t_end=20.0))
    lines += ["## 1. Deterministic control (one centred colony, R = 10 um, N = 1, 20 h)", "",
              "| check | value | result |", "|---|---|---|"]
    lines += [f"| {k} | {v['value']:.3g} | {mark(v['pass'])} |" for k, v in det.items()]

    print("2) time-step convergence ...")
    conv = time_step_convergence(PAPER, t_end=5.0)
    lines += ["", "## 2. Time-step convergence (tau = 0.01 h vs 0.005 h, 5 h)", "",
              f"Relative change: AHL {conv['U']:.2e}, lactonase {conv['L']:.2e} -> "
              f"{mark(conv['pass'])} (threshold 1e-2)"]

    print("3) perturbation test (paper Fig. 4) ...")
    pert = perturbation_test(PAPER)
    lines += ["", "## 3. Perturbation test (paper Fig. 4)", "",
              "| parameter | +20% | +30% | +40% | +50% | grows steadily |", "|---|---|---|---|---|---|"]
    for name in ("alpha_U", "beta_U", "beta_L"):
        d = pert[name]
        lines.append(f"| {name} xi (AHL) | " + " | ".join(f"{x:.3f}" for x in d["xi"])
                     + f" | {mark(d['monotone'])} |")
        lines.append(f"| {name} phi (lactonase) | " + " | ".join(f"{x:.3f}" for x in d["phi"])
                     + " | |")

    print("4) statistical control (4 seeds x 50 h) ...")
    stat = statistical_control(PAPER)
    lines += ["", "## 4. Statistical control (single dose at 12 h, seeds 0-3, mean)", "",
              "| quantity | ours | paper | result |", "|---|---|---|---|"]
    for k, v in stat["checks"].items():
        ours = f"{v['ours']:.3g}" if isinstance(v["ours"], float) else v["ours"]
        paper = f"{v['paper']:.3g}" if isinstance(v["paper"], float) else v["paper"]
        lines.append(f"| {k} | {ours} | {paper} | {mark(v['pass'])} |")

    (OUT / "positive_control_report.md").write_text("\n".join(lines) + "\n")
    json.dump({"deterministic": det, "convergence": conv, "perturbation": pert,
               "statistical": {"runs": stat["runs"], "mean": stat["mean"],
                               "checks": stat["checks"]}},
              open(OUT / "positive_control_report.json", "w"), indent=2, default=bool)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
