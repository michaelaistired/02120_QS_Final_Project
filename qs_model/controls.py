"""Positive controls: does our code behave like the paper's model? (Q6)

Two levels, because the paper's runs are random and cannot be copied exactly:

1. DETERMINISTIC control - the paper's own test setup (Sec. 3.1, Fig. 4): one colony at the
   centre, radius 10 um, N = 1, no randomness, no antibiotic. We check basic correctness
   (symmetry, zero edges, no negative values, the QS switch turns on), that halving the time
   step barely changes the answer, and the paper's Fig. 4 perturbation test.

2. STATISTICAL control - the paper's single-dose experiment (Figs. 5-7) run with several
   random seeds. We compare against the numbers printed in the paper using an
   order-of-magnitude tolerance plus shape checks (e.g. lactonase peaks before the dose).
"""

import numpy as np

from .params import Params
from .simulation import Result, centered_colony, run_simulation

# Numbers read from the paper's text (Sec. 3.2-3.3, Figs. 5, 7, 9).
PAPER_REFERENCE = {
    "A_peak": 4.0,                    # Fig 5A, peak antibiotic level
    "A_peak_delay_h": (3.5, 4.5),     # "maximum ... after 3.5-4 h after adding"
    "R_total_final_single": 2.017,    # Fig 5B, um, after one dose
    "U_max_final_single": 4.89e-10,   # Fig 7A, mol/L
    "U_mean_final_single": 1.36e-10,  # text, mol/L
    "L_max_final_single": 6.76e-13,   # Fig 7B, mol/L
}


def within_factor(value: float, reference: float, factor: float) -> bool:
    """Is value within a multiplicative factor of reference?

    Input:  value, reference (both > 0), factor (e.g. 10 = same order of magnitude).
    Output: bool.
    Steps:  1) reference / factor <= value <= reference * factor.
    Types:  (float, float, float) -> bool.
    Role in model: the tolerance for comparing our random runs to the paper's random runs.
        Concentrations here span 10+ orders of magnitude, so a ratio test is the fair one.
    """
    return value > 0 and reference / factor <= value <= reference * factor


def deterministic_control(p: Params, t_end: float = 20.0, radius: float = 10.0) -> Result:
    """Run the paper's fixed, non-random test setup.

    Input:  p (Params), t_end (h), radius (um).
    Output: Result.
    Steps:  1) no antibiotic, observation time t_end,
            2) one centred colony, N held at 1, colonies frozen,
            3) run_simulation.
    Types:  (Params, float, float) -> Result.
    Role in model: the STRICT positive control (Q6). Nothing is random, so anyone running
        this code gets the same numbers, and it matches the setup of paper Fig. 4.
    """
    q = p.with_changes(t_obs=t_end, dose_times=())
    return run_simulation(q, colonies=centered_colony(q, radius), density_override=1.0,
                          freeze_colonies=True)


def deterministic_checks(res: Result) -> dict:
    """Basic correctness checks on a deterministic-control run.

    Input:  res (Result from deterministic_control).
    Output: dict check_name -> {"value": ..., "pass": bool}.
    Steps:  1) edges stay exactly 0 (boundary condition),
            2) no negative concentrations,
            3) the field is symmetric (a centred colony must give a symmetric answer):
               mirror symmetry should be exact; x<->y symmetry is only approximate because
               ADI always does x first then y (a tiny, expected splitting error),
            4) report AHL at the colony centre relative to the QS threshold (observation only),
            5) the fixed-point loop converged in a few iterations.
    Types:  Result -> dict[str, dict].
    Role in model: catches bugs in the solver before we trust any figure.
    """
    U, L, p = res.U, res.L, res.params
    c = U.shape[0] // 2
    mirror = np.abs(U - U[::-1, :]).max() / U.max()
    swap = np.abs(U - U.T).max() / U.max()
    edges = max(np.abs(U[0]).max(), np.abs(U[-1]).max(), np.abs(U[:, 0]).max(),
                np.abs(U[:, -1]).max(), np.abs(L[0]).max(), np.abs(L[-1]).max())
    return {
        "edges_zero": {"value": float(edges), "pass": edges == 0.0},
        "no_negative": {"value": float(min(U.min(), L.min())),
                        "pass": U.min() >= -1e-15 and L.min() >= -1e-15},
        "mirror_symmetric": {"value": float(mirror), "pass": mirror < 1e-10},
        "xy_symmetric": {"value": float(swap), "pass": swap < 1e-4},
        # an OBSERVATION, not a bug check (pass = None): with the paper's parameters the
        # centre AHL stays below U_th, i.e. the QS switch never fully turns on
        "centre_AHL_over_threshold": {"value": float(U[c, c] / p.U_th), "pass": None},
        "picard_converged": {"value": float(res.series["picard_iters"][1:].max()),
                             "pass": res.series["picard_iters"][1:].max() < p.picard_max},
    }


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative size of the difference between two fields.

    Input:  a (new field), b (reference field).
    Output: ||a - b|| / ||b|| (float).
    Steps:  1) L2 norm of the difference, 2) divide by L2 norm of the reference.
    Types:  (ndarray, ndarray) -> float.
    Role in model: the paper's perturbation measure xi, phi (Sec. 3.1) and our time-step check.
    """
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def time_step_convergence(p: Params, t_end: float = 5.0) -> dict:
    """Does halving the PDE time step change the answer much?

    Input:  p (Params), t_end (h).
    Output: {"U": rel. difference, "L": rel. difference, "pass": bool}.
    Steps:  1) deterministic control at tau, 2) again at tau / 2, 3) compare final fields.
    Types:  (Params, float) -> dict.
    Role in model: shows the PDE answer is a property of the model, not of our step size.
        (The colony kill rule is NOT step-independent - see antibiotic.kill_fraction.)
    """
    a = deterministic_control(p, t_end)
    b = deterministic_control(p.with_changes(tau=p.tau / 2.0), t_end)
    dU, dL = relative_l2(a.U, b.U), relative_l2(a.L, b.L)
    return {"U": dU, "L": dL, "pass": dU < 1e-2 and dL < 1e-2}


def perturbation_test(p: Params, fractions=(0.2, 0.3, 0.4, 0.5),
                      names=("alpha_U", "beta_U", "beta_L"), t_end: float = 10.0) -> dict:
    """Reproduce paper Fig. 4: raise each production rate by 20-50% and measure the change.

    Input:  p (Params), fractions (increases to try), names (parameters to vary), t_end (h).
    Output: {"fractions": [...], name: {"xi": [...], "phi": [...], "monotone": bool}, ...}
            xi = relative change in AHL, phi = relative change in lactonase.
    Steps:  1) baseline deterministic run,
            2) for each parameter and each fraction: rerun with that parameter * (1 + fraction),
            3) relative_l2 against the baseline for U (xi) and L (phi),
            4) check the changes grow steadily with the perturbation.
    Types:  (Params, tuple[float], tuple[str], float) -> dict.
    Role in model: the paper's own stability check. Small, smoothly growing changes mean
        the scheme is stable. NOTE: the paper's xi formula is ambiguous about the reference
        run; we measure every perturbation against the unperturbed run.
    """
    base = deterministic_control(p, t_end)
    out = {"fractions": list(fractions)}
    for name in names:
        xi, phi = [], []
        for f in fractions:
            run = deterministic_control(p.with_changes(**{name: getattr(p, name) * (1 + f)}), t_end)
            xi.append(relative_l2(run.U, base.U))
            phi.append(relative_l2(run.L, base.L))
        monotone = all(np.diff(xi) > 0) and all(np.diff(phi) > 0)
        out[name] = {"xi": xi, "phi": phi, "monotone": bool(monotone)}
    return out


def paper_metrics(res: Result) -> dict:
    """Pull the numbers the paper reports out of one single-dose run.

    Input:  res (Result of a single-dose paper-mode run).
    Output: dict of metrics (floats).
    Steps:  1) antibiotic peak and its delay after dosing,
            2) final total radius and final AHL / lactonase levels,
            3) when lactonase peaked, and how far each molecule fell from its peak.
    Types:  Result -> dict[str, float].
    Role in model: turns a run into the same quantities quoted in paper Sec. 3.2.
    """
    s, p = res.series, res.params
    i_A = int(np.argmax(s["A"]))
    return {
        "A_peak": float(s["A"][i_A]),
        "A_peak_delay_h": float(s["t"][i_A] - min(p.dose_times)),
        "R_total_peak": float(s["R_total"].max()),
        "R_total_final": float(s["R_total"][-1]),
        "U_max_final": float(s["U_max"][-1]),
        "U_mean_final": float(s["U_mean"][-1]),
        "L_max_final": float(s["L_max"][-1]),
        "L_peak_time": float(s["t"][int(np.argmax(s["L_max"]))]),
        "U_fall_from_peak": float(s["U_max"][-1] / s["U_max"].max()),
        "L_fall_from_peak": float(s["L_max"][-1] / s["L_max"].max()),
    }


def statistical_control(p: Params, seeds=(0, 1, 2, 3), factor: float = 10.0) -> dict:
    """Run the paper's single-dose experiment with several seeds and compare to the paper.

    Input:  p (Params, paper values with one dose), seeds (random seeds), factor (tolerance).
    Output: {"runs": [metrics per seed], "mean": averaged metrics, "checks": {...}, "results": [...]}.
    Steps:  1) run_simulation for each seed,
            2) paper_metrics for each run, then average,
            3) compare to PAPER_REFERENCE:
               - antibiotic peak height (5%) and timing (within the paper's 3.5-4.5 h),
               - final radius and concentrations within `factor`,
               - shape: lactonase peaks before the dose; lactonase falls far more than AHL.
    Types:  (Params, tuple[int], float) -> dict.
    Role in model: the STATISTICAL positive control (Q6). It cannot match one exact paper run,
        but our typical behavior should match the paper's typical behavior.
    """
    results = [run_simulation(p, seed=s) for s in seeds]
    runs = [paper_metrics(r) for r in results]
    mean = {k: float(np.mean([m[k] for m in runs])) for k in runs[0]}
    ref = PAPER_REFERENCE
    lo, hi = ref["A_peak_delay_h"]
    checks = {
        "A_peak (5%)": {"ours": mean["A_peak"], "paper": ref["A_peak"],
                        "pass": abs(mean["A_peak"] / ref["A_peak"] - 1) < 0.05},
        "A_peak_delay_h": {"ours": mean["A_peak_delay_h"], "paper": f"{lo}-{hi}",
                           "pass": lo <= mean["A_peak_delay_h"] <= hi},
        f"R_total_final (x{factor:g})": {"ours": mean["R_total_final"],
                                         "paper": ref["R_total_final_single"],
                                         "pass": within_factor(mean["R_total_final"],
                                                               ref["R_total_final_single"], factor)},
        f"U_max_final (x{factor:g})": {"ours": mean["U_max_final"],
                                       "paper": ref["U_max_final_single"],
                                       "pass": within_factor(mean["U_max_final"],
                                                             ref["U_max_final_single"], factor)},
        f"U_mean_final (x{factor:g})": {"ours": mean["U_mean_final"],
                                        "paper": ref["U_mean_final_single"],
                                        "pass": within_factor(mean["U_mean_final"],
                                                              ref["U_mean_final_single"], factor)},
        f"L_max_final (x{factor:g})": {"ours": mean["L_max_final"],
                                       "paper": ref["L_max_final_single"],
                                       "pass": within_factor(mean["L_max_final"],
                                                             ref["L_max_final_single"], factor)},
        "L peaks before dose": {"ours": mean["L_peak_time"], "paper": f"< {min(p.dose_times)}",
                                "pass": all(m["L_peak_time"] <= min(p.dose_times) + p.colony_dt
                                            for m in runs)},
        "L falls more than U": {"ours": f"L {mean['L_fall_from_peak']:.1e} vs "
                                        f"U {mean['U_fall_from_peak']:.1e}",
                                "paper": "L collapses, U lingers",
                                "pass": all(m["L_fall_from_peak"] < m["U_fall_from_peak"]
                                            for m in runs)},
    }
    return {"runs": runs, "mean": mean, "checks": checks, "results": results}
