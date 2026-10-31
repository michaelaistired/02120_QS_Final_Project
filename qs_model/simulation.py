"""The main loop: the paper's Figure 1 flowchart, chaining every other module.

    start -> parameters -> seed colonies
    repeat every colony step (0.5 h):
        footprint  <- colonies              (production.total_footprint)
        repeat 50 PDE steps (0.01 h):
            N      <- density(t)            (density.density)
            U, L   <- adi_step(...)         (solver.adi_step)
        colonies   <- update_colonies(...)  (colonies.update_colonies; includes Q7 sensing)
        record readouts / snapshots
    stop at t_obs
"""

from dataclasses import dataclass, field
import copy

import numpy as np

from .antibiotic import antibiotic_concentration
from .colonies import Colony, seed_colonies, total_radius, update_colonies
from .density import density
from .feedback import sense_colonies
from .grid import Grid, make_grid, zero_field
from .params import Params
from .production import total_footprint
from .solver import adi_step

SERIES_KEYS = ("t", "U_max", "U_mean", "L_max", "L_mean", "R_total", "N", "A",
               "n_colonies", "activation_mean", "dna_total", "picard_iters")


@dataclass
class Result:
    """Everything a run produces.

    Fields:
        params    : the Params used
        grid      : the Grid
        series    : dict[str, np.ndarray] - one value per colony step for each SERIES_KEYS entry
        snapshots : dict[float, dict] - time -> {"U", "L" (n, n arrays), "colonies" (list)}
        U, L      : final fields ((n, n) arrays)
        colonies  : final list[Colony]
    Role in model: the "output space-time distributions" box of the flowchart; every plot and
        positive-control check reads from this object.
    """

    params: Params
    grid: Grid
    series: dict
    snapshots: dict = field(default_factory=dict)
    U: np.ndarray = None
    L: np.ndarray = None
    colonies: list = None


def resolve_density(t: float, p: Params, density_override) -> float:
    """N(t), unless a control experiment fixes it.

    Input:  t (h), p (Params), density_override (None, a number, or a function of t).
    Output: N (float).
    Steps:  1) None -> paper's N(t), 2) number -> that constant, 3) function -> call it.
    Types:  (float, Params, None|float|Callable[[float], float]) -> float.
    Role in model: lets the deterministic positive control hold N = 1 (paper Fig. 4 setup)
        without touching the paper's density code.
    """
    if density_override is None:
        return density(t, p)
    if callable(density_override):
        return float(density_override(t))
    return float(density_override)


def record(series: dict, t: float, U: np.ndarray, L: np.ndarray, colonies: list,
           p: Params, N: float, picard_iters: float) -> None:
    """Append one row of readouts.

    Input:  series (dict of lists, modified in place), t (h), U, L (fields), colonies, p, N,
            picard_iters (mean fixed-point iterations over the last colony step).
    Output: None.
    Steps:  1) compute max/mean of each field, total radius, antibiotic, QS and DNA readouts,
            2) append each to its list.
    Types:  (dict, float, ndarray, ndarray, list[Colony], Params, float, float) -> None.
    Role in model: the time series behind paper Figs. 5, 7, 9, 11 (max AHL, total R, ...).
    """
    acts = [c.activation for c in colonies]
    row = {
        "t": t,
        "U_max": float(U.max()), "U_mean": float(U.mean()),
        "L_max": float(L.max()), "L_mean": float(L.mean()),
        "R_total": total_radius(colonies), "N": N,
        "A": antibiotic_concentration(t, p),
        "n_colonies": len(colonies),
        "activation_mean": float(np.mean(acts)) if acts else 0.0,
        "dna_total": float(sum(c.dna for c in colonies)),
        "picard_iters": picard_iters,
    }
    for k in SERIES_KEYS:
        series[k].append(row[k])


def advance_chemistry(U: np.ndarray, L: np.ndarray, footprint: np.ndarray, t: float,
                      n_sub: int, p: Params, density_override) -> tuple:
    """Run the PDE solver for one colony step (n_sub PDE steps) with colonies held still.

    Input:  U, L (fields at time t), footprint (fixed during this colony step), t (h),
            n_sub (PDE steps per colony step), p (Params), density_override (see resolve_density).
    Output: (U, L, mean fixed-point iterations per half step).
    Steps:  1) for each PDE step: N at start and end of the step,
            2) adi_step advances U and L by tau.
    Types:  (ndarray, ndarray, ndarray, float, int, Params, ...) -> tuple[ndarray, ndarray, float].
    Role in model: the "calculate N, U, L (numerically solve PDEs)" box. Colonies only change
        every 0.5 h, while chemistry moves every 0.01 h; this is how the two clocks fit together.
    """
    total_iters = 0
    for s in range(n_sub):
        ts = t + s * p.tau
        N0 = resolve_density(ts, p, density_override)
        N1 = resolve_density(ts + p.tau, p, density_override)
        U, L, iters = adi_step(U, L, footprint, N0, N1, p)
        total_iters += iters
    return U, L, total_iters / (2 * n_sub)


def run_simulation(p: Params, seed: int = 0, snapshot_times: tuple = (),
                   colonies: list = None, density_override=None,
                   freeze_colonies: bool = False, verbose: bool = False) -> Result:
    """Run the full hybrid model from t = 0 to p.t_obs.

    Input:  p (Params); seed (int, makes the random colonies reproducible);
            snapshot_times (times in h to save full U/L maps);
            colonies (optional fixed starting colonies, list[Colony]);
            density_override (optional, fixes N - used by the deterministic control);
            freeze_colonies (True = colonies never change; deterministic control);
            verbose (print progress).
    Output: Result.
    Steps:  1) build grid, zero fields (Eq. 3), random generator, starting colonies,
            2) loop over colony steps:
               a) footprint from current colonies,
               b) advance_chemistry for 50 PDE steps,
               c) update_colonies (sense AHL, spawn/grow or shrink, Q7 feedback),
               d) record readouts and any snapshots,
            3) package everything in a Result.
    Types:  (Params, int, tuple[float], list[Colony]|None, ..., bool, bool) -> Result.
    Role in model: the top-level chain of the paper's flowchart. Every experiment - the
        positive controls, multi-dose treatment, and the Q7 feedback - is just a call to this
        function with different Params or options.
    """
    grid = make_grid(p)
    rng = np.random.default_rng(seed)
    U, L = zero_field(grid), zero_field(grid)
    cols = copy.deepcopy(colonies) if colonies is not None else seed_colonies(rng, p)

    n_sub = int(round(p.colony_dt / p.tau))
    n_steps = int(round(p.t_obs / p.colony_dt))
    series = {k: [] for k in SERIES_KEYS}
    snapshots = {}
    pending = sorted(snapshot_times)

    record(series, 0.0, U, L, cols, p, resolve_density(0.0, p, density_override), 0.0)
    for k in range(n_steps):
        t = k * p.colony_dt
        footprint = total_footprint(grid.X, grid.Y, cols, p)
        U, L, iters = advance_chemistry(U, L, footprint, t, n_sub, p, density_override)
        if freeze_colonies:
            sense_colonies(cols, U, grid, p)  # readout only; radii never change
        else:
            cols = update_colonies(cols, t, rng, p, U, grid)
        t_end = t + p.colony_dt
        record(series, t_end, U, L, cols, p, resolve_density(t_end, p, density_override), iters)
        while pending and pending[0] <= t_end + 1e-9:
            snapshots[pending.pop(0)] = {"U": U.copy(), "L": L.copy(),
                                         "colonies": copy.deepcopy(cols)}
        if verbose and (k + 1) % 20 == 0:
            print(f"  t = {t_end:6.1f} h  max U = {U.max():.3e}  max L = {L.max():.3e}  "
                  f"R_total = {total_radius(cols):.3g}")

    return Result(params=p, grid=grid, series={k: np.array(v) for k, v in series.items()},
                  snapshots=snapshots, U=U, L=L, colonies=cols)


def centered_colony(p: Params, radius: float = 10.0) -> list:
    """One colony in the middle of the domain.

    Input:  p (Params), radius (um).
    Output: list with a single Colony at (l/2, l/2).
    Steps:  1) build the colony.
    Types:  (Params, float) -> list[Colony].
    Role in model: the paper's deterministic test setup (Sec. 3.1, Fig. 4): "only one
        bacterial colony located at the central position ... with a linear size of 10 um".
    """
    return [Colony(x=p.length / 2.0, y=p.length / 2.0, R=radius)]
