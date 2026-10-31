"""A(t): the antibiotic. Uniform over the whole surface, so it is only a function of time."""

import math

from .params import Params


def rayleigh_pulse(t_since_dose: float, A0: float, A1: float) -> float:
    """Concentration from ONE dose, given how long ago it was added.

    Input:  t_since_dose (h; negative means not added yet), A0 amplitude, A1 width (h^2).
    Output: antibiotic concentration (arbitrary units, peak ~4.0 with paper values).
    Steps:  1) 0 before the dose, 2) otherwise A0 * t * exp(-t^2 / A1).
    Types:  (float, float, float) -> float.
    Role in model: paper Eq. 13, fit to ciprofloxacin data. It peaks sqrt(A1/2) = 4 h after
        dosing and fades over ~12-16 h, which is paper Fig. 5A.
    """
    if t_since_dose <= 0.0:
        return 0.0
    return A0 * t_since_dose * math.exp(-t_since_dose ** 2 / A1)


def antibiotic_concentration(t: float, p: Params) -> float:
    """Total antibiotic at time t from all doses.

    Input:  t (h), p (Params) - uses dose_times, A0, A1.
    Output: A(t) (float).
    Steps:  1) one Rayleigh pulse per dose, 2) add them up.
    Types:  (float, Params) -> float.
    Role in model: lets the same code run the single-dose (Fig 5) and the triple-dose
        (Fig 8A) treatment strategies.
    """
    return sum(rayleigh_pulse(t - td, p.A0, p.A1) for td in p.dose_times)


def kill_fraction(A: float, theta: float, m: float) -> float:
    """Fraction of a colony's radius lost in one colony step at antibiotic level A.

    Input:  A (antibiotic level), theta (max fraction, 0.2), m (half-saturation constant).
    Output: fraction in [0, theta).
    Steps:  1) theta * A^2 / (A^2 + m)  (saturating: more drug helps less and less).
    Types:  (float, float, float) -> float.
    Role in model: the bracket in paper Eq. 14. NOTE: it is applied once per colony step
        (0.5 h), so the total kill depends on the step size - a limitation we discussed.
    """
    return theta * A * A / (A * A + m)


def shrink_radius(R: float, A: float, p: Params) -> float:
    """New colony radius after one colony step under antibiotic.

    Input:  R (um), A (antibiotic level), p (Params) - uses theta, m.
    Output: new radius R * (1 - kill_fraction) (um).
    Steps:  1) compute kill fraction, 2) scale the radius.
    Types:  (float, float, Params) -> float.
    Role in model: paper Eq. 14. Smaller radius -> smaller Gaussian footprint -> less area
        producing AHL/lactonase. This is how antibiotic reaches the chemistry.
    """
    return R * (1.0 - kill_fraction(A, p.theta, p.m))


def treatment_started(t: float, p: Params) -> bool:
    """Has the first antibiotic dose been added by time t?

    Input:  t (h), p (Params).
    Output: bool.
    Steps:  1) compare t to the earliest dose time.
    Types:  (float, Params) -> bool.
    Role in model: the flowchart's "The antibiotics have been added?" diamond. Once True,
        no new colonies are generated and colonies only shrink (unless regrowth is enabled).
    """
    return bool(p.dose_times) and t >= min(p.dose_times)
