"""Monte-Carlo colonies: random birth, logistic growth, antibiotic shrinkage (paper Sec. 2.3).

This is the dashed box on the right of the paper's flowchart:
    antibiotic added?  -> yes: R(k+1) = shrink(R(k), A(k+1))
                       -> no : reached limit size? -> yes: stop growing
                                                   -> no : logistic growth
"""

from dataclasses import dataclass
import math

import numpy as np

from .antibiotic import antibiotic_concentration, shrink_radius, treatment_started
from .feedback import apply_dna_uptake, effective_growth_rate, sense_colonies
from .params import Params


@dataclass
class Colony:
    """One circular colony.

    Fields: x, y (centre, um), R (radius, um), activation (QS switch 0-1, readout),
            dna (accumulated DNA-uptake events, readout).
    Role in model: the discrete, stochastic half of the hybrid model. The PDEs only see a
        colony through its centre and radius (via the Gaussian footprint).
    """

    x: float
    y: float
    R: float
    activation: float = 0.0
    dna: float = 0.0


def draw_initial_count(rng: np.random.Generator, probs: tuple) -> int:
    """How many colonies exist at t = 0.

    Input:  rng (numpy random generator), probs (P(1), P(2), P(3), ...).
    Output: an int (1, 2 or 3 with paper values).
    Steps:  1) random choice weighted by probs.
    Types:  (Generator, tuple[float, ...]) -> int.
    Role in model: paper: "up to three colonies start ... p1 = 0.6, p2 = 0.3, p3 = 0.1".
    """
    return int(rng.choice(np.arange(1, len(probs) + 1), p=probs))


def random_position(rng: np.random.Generator, length: float) -> tuple:
    """A uniformly random point in the square domain.

    Input:  rng, length (um).
    Output: (x, y) in um.
    Steps:  1) two uniform draws in [0, length].
    Types:  (Generator, float) -> tuple[float, float].
    Role in model: colonies land anywhere, including near the absorbing edges - one reason
        runs with different seeds look different (Q6).
    """
    return float(rng.uniform(0.0, length)), float(rng.uniform(0.0, length))


def new_colony(rng: np.random.Generator, p: Params) -> Colony:
    """A fresh colony of radius R0 at a random position.

    Input:  rng, p (Params) - uses length, R0.
    Output: Colony.
    Steps:  1) random position, 2) radius R0.
    Types:  (Generator, Params) -> Colony.
    Role in model: the "probabilistically generate a new bacterial population" box.
    """
    x, y = random_position(rng, p.length)
    return Colony(x=x, y=y, R=p.R0)


def seed_colonies(rng: np.random.Generator, p: Params) -> list:
    """The colonies present at t = 0.

    Input:  rng, p (Params) - uses p_initial.
    Output: list[Colony].
    Steps:  1) draw how many, 2) make that many new colonies.
    Types:  (Generator, Params) -> list[Colony].
    Role in model: the flowchart's "generate ... initial positions and initial size".
    """
    return [new_colony(rng, p) for _ in range(draw_initial_count(rng, p.p_initial))]


def maybe_spawn(rng: np.random.Generator, p: Params):
    """With small probability, a new colony appears this colony step.

    Input:  rng, p (Params) - uses p_new.
    Output: Colony or None.
    Steps:  1) uniform draw, 2) if below p_new make a new colony.
    Types:  (Generator, Params) -> Colony | None.
    Role in model: paper's Monte-Carlo birth, p_v = 0.1 per 0.5 h step.
    """
    return new_colony(rng, p) if rng.uniform() < p.p_new else None


def logistic_step(R: float, r: float, I: float, dt: float) -> float:
    """Advance a radius along the logistic curve by dt.

    Input:  R (current radius, um), r (growth rate, 1/h), I (limit radius, um), dt (h).
    Output: radius after dt (um).
    Steps:  1) exact logistic solution started from the CURRENT radius:
               R_new = I R e^{r dt} / (I + R (e^{r dt} - 1)).
    Types:  (float, float, float, float) -> float.
    Role in model: paper Eq. 12. Starting from the current R (instead of from R0 at t = 0)
        gives the same curve but also works when r changes over time (Q7 feedback, regrowth).
    """
    g = math.exp(r * dt)
    return I * R * g / (I + R * (g - 1.0))


def reached_limit(R: float, I: float, rel_tol: float = 1e-3) -> bool:
    """Has the colony stopped growing?

    Input:  R, I (um), rel_tol (fraction of I).
    Output: bool.
    Steps:  1) R >= (1 - rel_tol) * I.
    Types:  (float, float, float) -> bool.
    Role in model: the flowchart's "bacterial populations reach the limit size" diamond.
    """
    return R >= (1.0 - rel_tol) * I


def grow_colony(c: Colony, r_base: float, dt: float, p: Params) -> None:
    """Grow one colony for dt, applying QS feedback to its growth rate.

    Input:  c (Colony, modified in place), r_base (1/h), dt (h), p (Params).
    Output: None (c.R is updated).
    Steps:  1) stop if at the limit, 2) r_eff from feedback (unchanged in paper mode),
            3) logistic step.
    Types:  (Colony, float, float, Params) -> None.
    Role in model: the place where the Q7 feedback actually touches the population.
    """
    if reached_limit(c.R, p.I):
        return
    c.R = logistic_step(c.R, effective_growth_rate(r_base, c.activation, p), p.I, dt)


def update_colonies(colonies: list, t: float, rng: np.random.Generator, p: Params,
                    U=None, grid=None) -> list:
    """Advance every colony by one colony step (t -> t + colony_dt).

    Input:  colonies (list[Colony]), t (h, start of the step), rng, p (Params),
            U ((n, n) AHL field, optional) and grid (Grid, optional) for QS sensing.
    Output: the updated list of colonies (may contain one new colony).
    Steps:  1) sense: each colony reads local AHL -> activation; add DNA-uptake readout,
            2) if no antibiotic yet: maybe spawn a colony; grow every colony (logistic),
            3) if antibiotic added: shrink every colony by A(t + dt),
               and if past regrow_time also regrow slowly (Fig 10 scenario).
    Types:  (list[Colony], float, Generator, Params, ndarray|None, Grid|None) -> list[Colony].
    Role in model: chains sense_colonies -> maybe_spawn / grow_colony / shrink_radius in the
        same order as the paper's flowchart. This is the whole stochastic population layer.
    """
    dt = p.colony_dt
    t_next = t + dt

    # 1) sensing + readouts
    if U is not None and grid is not None:
        sense_colonies(colonies, U, grid, p)
        apply_dna_uptake(colonies, dt, p)

    # 2) before treatment: birth + logistic growth
    if not treatment_started(t_next, p):
        baby = maybe_spawn(rng, p)
        if baby is not None:
            colonies.append(baby)
        for c in colonies:
            grow_colony(c, p.r, dt, p)
        return colonies

    # 3) after treatment: shrink, and optionally regrow
    A = antibiotic_concentration(t_next, p)
    for c in colonies:
        c.R = shrink_radius(c.R, A, p)
        if t_next >= p.regrow_time:
            grow_colony(c, p.r_regrow, dt, p)
    return colonies


def total_radius(colonies: list) -> float:
    """Sum of all colony radii.

    Input:  colonies (list[Colony]).
    Output: total linear size R(t) in um.
    Steps:  1) add up c.R.
    Types:  list[Colony] -> float.
    Role in model: the paper's population-size readout (Figs. 5B, 9A, 11A).
    """
    return float(sum(c.R for c in colonies))
