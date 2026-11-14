"""Source and sink terms that connect agents to the fields.

    QS chemistry (paper Eqs. 5-6, with agents replacing Gaussian colonies):
        F1 = f1(U) * rho / rho_ref        f1(U) = alpha_U + beta_U * hill(U, U_th)
        F2 = f2(U) * rho / rho_ref        f2(U) = beta_L * hill(U, U_th + eps)
    Nutrient (new):   dS/dt = D_S Lap(S) - uptake,  uptake = sum over agents of V_max*pause*S/(K_S+S)
    Signal (new):     dG/dt = D_G Lap(G) - gamma_G G + G_emit * (starving agents per cell)
"""

import numpy as np


def hill(U, K: float, n: float):
    """Switch: ~0 below K, 1/2 at K, ~1 above.

    Input:  U (scalar/array >= 0), K (threshold), n (sharpness).   Output: same shape, in [0, 1].
    Steps:  1) clip at 0, 2) U^n / (K^n + U^n).
    Types:  (float|ndarray, float, float) -> float|ndarray.
    Role in model: the QS decision (paper), reused for the starvation-signal pause.
    """
    U = np.maximum(U, 0.0)
    Un = U ** n
    return Un / (K ** n + Un)


def ahl_production(U, p):
    """f1(U): AHL made per unit coverage (baseline + QS-ON boost).

    Input:  U (AHL), p.   Output: mol/(L h), same shape.   Steps: alpha_U + beta_U * hill(U, U_th).
    Types:  (ndarray, Params) -> ndarray.
    Role in model: positive feedback (autoinduction), unchanged from the paper.
    """
    return p.alpha_U + p.beta_U * hill(U, p.U_th, p.n)


def lactonase_production(U, p):
    """f2(U): lactonase made per unit coverage, switching on at the shifted threshold.

    Input:  U, p.   Output: mol/(L h).   Steps: beta_L * hill(U, U_th + eps).
    Types:  (ndarray, Params) -> ndarray.
    Role in model: the negative-feedback brake (Q2), unchanged from the paper.
    """
    return p.beta_L * hill(U, p.U_th + p.eps, p.n)


def agent_qs_sources(rho: np.ndarray, p):
    """Build the source function the QS solver needs, for agents.

    Input:  rho ((n, n) agents per cell), p (Params: rho_ref).
    Output: function (U, k) -> (F1, F2), each (n, n).
    Steps:  1) coverage = rho / rho_ref (plays the role of the paper's N(t) * footprint),
            2) return a closure that multiplies f1(U), f2(U) by coverage (k is unused here).
    Types:  (ndarray, Params) -> Callable[[ndarray, int], tuple[ndarray, ndarray]].
    Role in model: the only change to the paper's chemistry - production now comes from where
        the individual cells actually are, instead of from prescribed Gaussian blobs.
    """
    coverage = rho / p.rho_ref

    def sources(U, k):
        return ahl_production(U, p) * coverage, lactonase_production(U, p) * coverage

    return sources


def pause_factor(G_at_agent: np.ndarray, starving: np.ndarray, p) -> np.ndarray:
    """How much of its normal feeding each agent does (hypothesis b).

    Input:  G_at_agent ((N,) signal at each agent), starving ((N,) bool), p (pause_on, G_th,
            pause_strength).
    Output: (N,) factors in [1 - pause_strength, 1].
    Steps:  1) all ones if the pause mechanism is off,
            2) fed agents: 1 - pause_strength * hill(G, G_th, 2); starving agents keep eating (1).
    Types:  (ndarray, ndarray, Params) -> ndarray.
    Role in model: B. subtilis-style coordination (Liu et al. 2015): the starving centre's
        signal makes well-fed cells eat less, so nutrient can diffuse past them to the centre.
    """
    if not p.pause_on:
        return np.ones_like(G_at_agent)
    f = 1.0 - p.pause_strength * hill(G_at_agent, p.G_th, 2.0)
    return np.where(starving, 1.0, f)


def nutrient_uptake(S: np.ndarray, capacity: np.ndarray, p, dt: float) -> np.ndarray:
    """Nutrient removed from each cell during dt (exactly what the agents there gain).

    Input:  S ((n, n) nutrient), capacity ((n, n) = sum over agents in the cell of their pause
            factor, i.e. how many "full-rate eaters" the cell holds), p (V_max, K_S), dt (h).
    Output: (n, n) amount removed (never more than is there).
    Steps:  1) Monod rate V_max * capacity * S / (K_S + S), 2) times dt, 3) cap at S.
    Types:  (ndarray, ndarray, Params, float) -> ndarray.
    Role in model: the sink that empties the colony centre. Doing it as its own explicit step
        (operator splitting) makes food conserved exactly: removed here == gained by agents.
    """
    removed = p.V_max * capacity * S / (p.K_S + S) * dt
    return np.minimum(removed, S)


def relay_strength(G_at_agent: np.ndarray, starving: np.ndarray, p) -> np.ndarray:
    """How strongly each fed agent re-emits the starvation signal it senses.

    Input:  G_at_agent ((N,)), starving ((N,) bool), p (G_th).
    Output: (N,) values in [0, 1] (0 for starving agents - they already emit at G_emit).
    Steps:  1) hill(G, G_th, 2) for fed agents.
    Types:  (ndarray, ndarray, Params) -> ndarray.
    Role in model: turns passive diffusion (reaches ~sqrt(D_G/gamma_G) = 22 um) into a relayed
        wave that can cross the whole colony, like potassium waves in B. subtilis biofilms.
    """
    return np.where(starving, 0.0, hill(G_at_agent, p.G_th, 2.0))


def signal_source(starving_per_cell: np.ndarray, relay_per_cell: np.ndarray, p) -> np.ndarray:
    """Starvation signal production per cell.

    Input:  starving_per_cell ((n, n) count of starving agents), relay_per_cell ((n, n) sum of
            relay_strength over fed agents), p (G_emit, G_relay).
    Output: (n, n) production rate.
    Steps:  1) G_emit * starving count + G_relay * relay sum.
    Types:  (ndarray, ndarray, Params) -> ndarray.
    Role in model: hypothesis (b)'s message from the hungry centre to the fed edge. It obeys the
        same diffuse + decay equation as AHL, so it is solved with the same ADI code.
    """
    return p.G_emit * starving_per_cell + p.G_relay * relay_per_cell
