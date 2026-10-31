"""Q7 demonstration: what changes when quorum sensing is allowed to change growth?

    python run_feedback_demo.py

Scenario (paper Fig. 10): triple antibiotic dose at 12, 24, 36 h; colonies regrow slowly
(r = 0.1) after 55 h; observe to 100 h. Same random seed in every run, so the ONLY
difference between paired runs is the feedback switch.

Two strains:
    paper     - Table 1 values. Our positive control shows AHL never reaches U_th here,
                so feedback should change nothing (a useful negative result).
    sensitive - autoinduction threshold U_th lowered 10x (7e-9 mol/L) while the lactonase
                threshold stays at the paper's 7.5e-8 (eps raised to 6.8e-8). QS turns ON.
Takes about 4 minutes. Writes results/feedback_demo.md.
"""

from pathlib import Path

from qs_model.params import PAPER
from qs_model.simulation import run_simulation

OUT = Path(__file__).parent / "results"
SCENARIO = dict(t_obs=100.0, dose_times=(12.0, 24.0, 36.0), regrow_time=55.0)
STRAINS = {
    "paper": PAPER.with_changes(**SCENARIO),
    "sensitive": PAPER.with_changes(**SCENARIO, U_th=7e-9, eps=6.8e-8),
}
REPORT_TIMES = (8.0, 12.0, 40.0, 80.0, 100.0)


def run_pairs(seed: int = 0) -> dict:
    """Run feedback OFF and ON for each strain.

    Input:  seed (int).
    Output: {(strain, "off"|"on"): Result}.
    Steps:  1) for each strain, 2) run with feedback_on False, then True.
    Types:  int -> dict[tuple[str, str], Result].
    Role in model: the controlled comparison that isolates the effect of the Q7 feedback.
    """
    out = {}
    for name, p in STRAINS.items():
        for mode in ("off", "on"):
            out[(name, mode)] = run_simulation(p.with_changes(feedback_on=(mode == "on")), seed=seed)
    return out


def main() -> None:
    """Run the demo and write a small markdown table."""
    OUT.mkdir(exist_ok=True)
    runs = run_pairs()
    lines = ["# Q7 feedback demo (seed 0)", "",
             "| strain | feedback | " + " | ".join(f"R_total @ {t:g} h" for t in REPORT_TIMES)
             + " | QS activation @ 12 h | DNA uptake (total) |",
             "|---|---|" + "---|" * (len(REPORT_TIMES) + 2)]
    for (name, mode), r in runs.items():
        s = r.series
        idx = [int(round(t / r.params.colony_dt)) for t in REPORT_TIMES]
        lines.append(f"| {name} | {mode} | " + " | ".join(f"{s['R_total'][i]:.3g}" for i in idx)
                     + f" | {s['activation_mean'][idx[1]]:.2f} | {s['dna_total'][-1]:.2f} |")
    (OUT / "feedback_demo.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
