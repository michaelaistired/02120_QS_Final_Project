"""N(t): the normalized bacterial density that scales how MUCH signal is produced (Q4).

In the paper N(t) is written down from data fits (Eqs. 15-16), not computed from the colonies.
This is the "two disconnected population descriptions" limitation we discussed.
"""

import math

from .params import Params


def growth_phase_density(t: float, b1: float, mu: float) -> float:
    """Density while the population grows (before the first antibiotic dose).

    Input:  t (h), b1 = midpoint time (h), mu = steepness (1/h).
    Output: N in [0, 1].
    Steps:  1) logistic sigmoid 1 / (1 + exp(-mu (t - b1))).
    Types:  (float, float, float) -> float.
    Role in model: paper Eq. 15. The population rises from ~0 to ~1 with its half-way point at
        b1 = 6 h, so production of AHL/lactonase ramps up as the culture fills in.
    """
    return 1.0 / (1.0 + math.exp(-mu * (t - b1)))


def decline_coefficients(N_before: float, drop_fraction: float, t_dose: float,
                         b2: float, mu: float) -> tuple:
    """Pick (a, b) so the decline curve starts exactly where the previous curve ended.

    Input:  N_before = density at the dose time, drop_fraction (0-1), t_dose (h), b2 (h), mu.
    Output: (a, b): a = level the density falls to, b = size of the drop.
    Steps:  1) a = (1 - drop) * N_before,
            2) solve a + b / (1 + exp(mu (t_dose - b2))) = N_before for b (continuity).
    Types:  (float, float, float, float, float) -> tuple[float, float].
    Role in model: the paper says a, b are chosen to give "smooth behavior of N(t) at t_d" and
        a 30% drop, but does not print their values. This function makes that rule explicit.
    """
    a = (1.0 - drop_fraction) * N_before
    b = (N_before - a) * (1.0 + math.exp(mu * (t_dose - b2)))
    return a, b


def decline_phase_density(t: float, a: float, b: float, b2: float, mu: float) -> float:
    """Density while antibiotic is reducing the population.

    Input:  t (h), a and b from decline_coefficients, b2 = midpoint of decline (h), mu.
    Output: N (float).
    Steps:  1) a + b / (1 + exp(mu (t - b2))).
    Types:  (float, float, float, float, float) -> float.
    Role in model: paper Eq. 16. After a dose, density slides smoothly down to level a,
        so production drops by ~30% over ~10 h.
    """
    return a + b / (1.0 + math.exp(mu * (t - b2)))


def density(t: float, p: Params) -> float:
    """Full N(t) for any number of antibiotic doses.

    Input:  t (h), p (Params) - uses b1, mu, b2_offset, drop_fraction, dose_times.
    Output: N(t) (float, normalized so a saturated, untreated population is ~1).
    Steps:  1) growth sigmoid up to the first dose,
            2) for each dose in order: compute (a, b) from the density at that dose,
               then use the decline curve until the next dose,
            3) return the value of whichever piece contains t.
    Types:  (float, Params) -> float.
    Role in model: chains growth_phase_density -> decline_coefficients -> decline_phase_density.
        With one dose this is exactly paper Fig. 3; with three doses it gives Fig. 8B's staircase.
    """
    doses = sorted(p.dose_times)
    if not doses or t <= doses[0]:
        return growth_phase_density(t, p.b1, p.mu)

    N_now = growth_phase_density(doses[0], p.b1, p.mu)
    value = N_now
    for k, t_dose in enumerate(doses):
        b2 = t_dose + p.b2_offset
        a, b = decline_coefficients(N_now, p.drop_fraction, t_dose, b2, p.mu)
        t_next = doses[k + 1] if k + 1 < len(doses) else math.inf
        if t <= t_next:
            return decline_phase_density(t, a, b, b2, p.mu)
        N_now = decline_phase_density(t_next, a, b, b2, p.mu)
        value = N_now
    return value
