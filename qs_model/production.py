"""Where and how fast AHL and lactonase are produced: the source terms F1 and F2 (paper Eqs. 5-6).

F_m(x, y, t, U) = N(t) * f_m(U) * sum_v exp(-((x - xc_v)^2 + (y - yc_v)^2) / sigma_v^2)
                  ^^^^   ^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  HOW    QS       WHERE (colony footprints; sigma_v = R_v / 3)
                  MUCH   switch
"""

import numpy as np

from .params import Params


def hill(U: np.ndarray, K: float, n: float) -> np.ndarray:
    """The QS switch: ~0 below the threshold K, ~1 above it.

    Input:  U (AHL concentration, scalar or array, mol/L), K (threshold, mol/L), n (sharpness).
    Output: values in [0, 1], same shape as U.
    Steps:  1) clip U at 0 (tiny negative round-off must not break the power),
            2) U^n / (K^n + U^n).
    Types:  (float | ndarray, float, float) -> float | ndarray.
    Role in model: this is the "quorum" decision. When local AHL passes K the cells switch
        from low to high production. n = 2.5 makes the switch fairly sharp.
    """
    U = np.maximum(U, 0.0)
    Un = U ** n
    return Un / (K ** n + Un)


def ahl_production(U: np.ndarray, p: Params) -> np.ndarray:
    """f1(U): AHL production rate per unit footprint.

    Input:  U (AHL field or value), p (Params) - uses alpha_U, beta_U, U_th, n.
    Output: rate in mol/(L*h), same shape as U.
    Steps:  1) baseline alpha_U, 2) plus beta_U times the Hill switch at threshold U_th.
    Types:  (ndarray, Params) -> ndarray.
    Role in model: POSITIVE feedback (autoinduction). More AHL -> switch ON -> ~10x more AHL.
        The baseline alpha_U lets a quiet colony slowly build up toward the threshold.
    """
    return p.alpha_U + p.beta_U * hill(U, p.U_th, p.n)


def lactonase_production(U: np.ndarray, p: Params) -> np.ndarray:
    """f2(U): lactonase production rate per unit footprint.

    Input:  U (AHL field or value), p (Params) - uses beta_L, U_th, eps, n.
    Output: rate in mol/(L*h), same shape as U.
    Steps:  1) beta_L times the Hill switch at the SHIFTED threshold U_th + eps.
    Types:  (ndarray, Params) -> ndarray.
    Role in model: the NEGATIVE-feedback brake. No baseline term, and the threshold is
        shifted by eps (Q2): eps = 0 makes the brake engage at the same AHL level as the
        accelerator; a large eps delays the brake so AHL climbs higher first.
    """
    return p.beta_L * hill(U, p.U_th + p.eps, p.n)


def colony_sigma(R: float, p: Params) -> float:
    """Width of a colony's Gaussian footprint.

    Input:  R (colony radius, um), p (Params) - uses sigma_factor (1/3).
    Output: sigma (um).
    Steps:  1) sigma = R / 3 ("3-sigma rule": almost all production falls inside radius R).
    Types:  (float, Params) -> float.
    Role in model: connects colony SIZE (from colonies.py / antibiotic) to the chemistry.
    """
    return R * p.sigma_factor


def colony_footprint(X: np.ndarray, Y: np.ndarray, xc: float, yc: float,
                     sigma: float) -> np.ndarray:
    """Gaussian bump marking where one colony produces molecules.

    Input:  X, Y ((n, n) coordinate arrays, um), colony center (xc, yc) um, sigma (um).
    Output: (n, n) array, 1 at the center falling toward 0 away from it.
    Steps:  1) squared distance from the center, 2) exp(-d^2 / sigma^2).
    Types:  (ndarray, ndarray, float, float, float) -> ndarray.
    Role in model: answers WHERE production happens (Q4). The paper uses sigma^2, not the
        textbook 2 sigma^2, in the denominator; we copy the paper exactly.
    """
    if sigma <= 0.0:
        return np.zeros_like(X)
    d2 = (X - xc) ** 2 + (Y - yc) ** 2
    return np.exp(-d2 / sigma ** 2)


def total_footprint(X: np.ndarray, Y: np.ndarray, colonies: list, p: Params) -> np.ndarray:
    """Sum of every colony's footprint (overlapping colonies simply add).

    Input:  X, Y (coordinate arrays), colonies (list of colonies.Colony), p (Params).
    Output: (n, n) array.
    Steps:  1) for each colony: sigma from its radius, 2) its Gaussian bump, 3) add them up.
    Types:  (ndarray, ndarray, list[Colony], Params) -> ndarray.
    Role in model: the sum over v in Eq. 5. Because f_m(U) does not depend on which colony
        produced it, we can add footprints once per colony step and reuse the result for
        every PDE step in between (a big speed-up, same math).
    """
    footprint = np.zeros_like(X)
    for c in colonies:
        footprint += colony_footprint(X, Y, c.x, c.y, colony_sigma(c.R, p))
    return footprint


def source_terms(U: np.ndarray, footprint: np.ndarray, N: float, p: Params) -> tuple:
    """F1 and F2: production of AHL and lactonase at every grid point.

    Input:  U ((n, n) AHL field), footprint ((n, n) from total_footprint), N (density, 0-1), p.
    Output: (F1, F2), each an (n, n) array in mol/(L*h).
    Steps:  1) f1(U) and f2(U) pointwise, 2) multiply by footprint (WHERE) and N (HOW MUCH).
    Types:  (ndarray, ndarray, float, Params) -> tuple[ndarray, ndarray].
    Role in model: paper Eq. 5 (Q4). N(t) scales HOW MUCH everything produces; the footprint
        (set by colony radius) says WHERE. A bigger radius also means more total output,
        because the footprint covers more area.
    """
    F1 = N * ahl_production(U, p) * footprint
    F2 = N * lactonase_production(U, p) * footprint
    return F1, F2
