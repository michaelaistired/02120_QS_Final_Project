"""Main loop for the agent model: chains every other module, calls a frame callback for video.

    start: grid, nucleation-site agents, fields (U = L = G = 0, S = S_init, edges S_edge)
    every PDE step (tau = 0.01 h):
        eat         nutrient_uptake -> S goes down, agent food goes up by the same amount
        nutrient    diffuse_step(S)                    (food diffuses in from the edges)
        signal      diffuse_step(G)  if pause_on       (hypothesis b)
        QS          qs_step(U, L) with agent sources   (paper chemistry)
    every agent step (agent_dt = 0.05 h):
        sense_qs -> burn_and_die -> divide -> share_food (hypothesis a) -> move -> refresh
    every frame_stride PDE steps: record readouts, label subnetworks, call on_frame(state)
"""

from dataclasses import dataclass, field

import numpy as np

from .agents import Agents, burn_and_die, divide, move, seed_nucleation, sense_qs
from .fields import (agent_qs_sources, nutrient_uptake, pause_factor, relay_strength,
                     signal_source)
from .grid import Grid, cell_index, count_per_cell, make_grid, sample
from .network import contact_subnetworks, share_food, transport_subnetworks
from .solver import diffuse_step, qs_step

HISTORY_KEYS = ("t", "n_agents", "n_starving", "food_mean", "food_centre", "food_edge",
                "S_centre", "radius95", "U_max", "activation_mean", "n_contact",
                "n_transport", "share_flow", "births", "deaths")


@dataclass
class State:
    """Everything the renderer needs at one moment.

    Fields: t (h), grid, U, L, S, G ((n, n) fields), agents, rho ((n, n) agents per cell),
            Jx, Jy (food flows on links), contact, transport ((n, n) subnetwork labels),
            strong_x, strong_y (which links are supply routes), history (dict of lists), p.
    Role in model: one video frame's worth of data. Passed to on_frame and then discarded,
        which is why nothing needs to be stored for every step (render while simulating).
    """

    t: float
    grid: Grid
    U: np.ndarray
    L: np.ndarray
    S: np.ndarray
    G: np.ndarray
    agents: Agents
    rho: np.ndarray
    Jx: np.ndarray
    Jy: np.ndarray
    p: object
    history: dict = field(default_factory=dict)
    contact: np.ndarray = None
    transport: np.ndarray = None
    strong_x: np.ndarray = None
    strong_y: np.ndarray = None


def initial_nutrient(grid: Grid, p) -> np.ndarray:
    """Starting nutrient field with the supply held at the edges.

    Input:  grid, p (S_init, S_edge).   Output: (n, n) array.
    Steps:  1) fill with S_init, 2) set boundary rows/cols to S_edge.
    Types:  (Grid, Params) -> ndarray.
    Role in model: a colony sitting in fresh medium, fed from all around (like agar).
    """
    S = np.full((grid.n, grid.n), p.S_init)
    S[0, :] = S[-1, :] = S[:, 0] = S[:, -1] = p.S_edge
    return S


def agent_cell_info(agents: Agents, G: np.ndarray, grid: Grid, p) -> dict:
    """Recompute everything that depends on where agents are.

    Input:  agents, G (signal field), grid, p.
    Output: dict with i, j (cells), rho (agents per cell), starving ((N,) bool),
            starving_cells ((n, n) counts), relay_cells ((n, n) summed relay strength),
            factor ((N,) feeding factor), capacity ((n, n)).
    Steps:  1) cells, 2) density, 3) who is starving, 4) feeding factor from the signal
            (hypothesis b), 5) per-cell feeding capacity.
    Types:  (Agents, ndarray, Grid, Params) -> dict.
    Role in model: agents -> fields bookkeeping; refreshed only after agents act/move.
    """
    i, j = cell_index(agents.x, agents.y, grid)
    starving = agents.food < p.food_starve
    G_a = sample(G, i, j)
    factor = pause_factor(G_a, starving, p)
    return {"i": i, "j": j, "rho": count_per_cell(i, j, grid),
            "starving": starving, "starving_cells": count_per_cell(i, j, grid, starving.astype(float)),
            "relay_cells": count_per_cell(i, j, grid, relay_strength(G_a, starving, p)),
            "factor": factor, "capacity": count_per_cell(i, j, grid, factor)}


def eat(S: np.ndarray, agents: Agents, info: dict, p) -> np.ndarray:
    """One PDE step of feeding: nutrient leaves the field and enters the agents.

    Input:  S, agents, info (from agent_cell_info), p.
    Output: new S (agents.food updated in place).
    Steps:  1) removed per cell = nutrient_uptake, 2) S -= removed,
            3) each agent gets removed[cell] * (its factor / cell capacity).
    Types:  (ndarray, Agents, dict, Params) -> ndarray.
    Role in model: creates the food-depleted centre. Food is conserved exactly
        (sum of agent gains == sum removed), which a unit test checks.
    """
    removed = nutrient_uptake(S, info["capacity"], p, p.tau)
    i, j, cap = info["i"], info["j"], info["capacity"]
    share = np.divide(info["factor"], cap[i, j], out=np.zeros_like(info["factor"]), where=cap[i, j] > 0)
    agents.food = agents.food + removed[i, j] * share
    return S - removed


def readouts(t: float, agents: Agents, S: np.ndarray, U: np.ndarray, contact, transport,
             Jx, Jy, births: int, deaths: int, grid: Grid, p) -> dict:
    """One row of numbers describing the colony.

    Input:  current state pieces, births/deaths since the last row.
    Output: dict keyed by HISTORY_KEYS.
    Steps:  1) counts and food, 2) centre (within 40 um of the nucleation site) vs edge
            (outermost 20% of agents by distance) food, 3) colony radius (95th percentile
            distance), 4) QS readouts, 5) number of subnetworks, 6) total food flow.
    Types:  (...) -> dict[str, float].
    Role in model: the time series shown beside the video, and the basis for comparing
        the ON/OFF control runs.
    """
    c = p.length / 2.0
    d = np.hypot(agents.x - c, agents.y - c)
    nA = agents.n
    centre = d < 40.0
    edge = d >= np.percentile(d, 80) if nA else d > 0
    mid = grid.n // 2
    return {
        "t": t, "n_agents": nA,
        "n_starving": int((agents.food < p.food_starve).sum()),
        "food_mean": float(agents.food.mean()) if nA else 0.0,
        "food_centre": float(agents.food[centre].mean()) if centre.any() else 0.0,
        "food_edge": float(agents.food[edge].mean()) if nA else 0.0,
        "S_centre": float(S[mid, mid]),
        "radius95": float(np.percentile(d, 95)) if nA else 0.0,
        "U_max": float(U.max()),
        "activation_mean": float(agents.activation.mean()) if nA else 0.0,
        "n_contact": int(contact.max()), "n_transport": int(transport.max()),
        "share_flow": float(np.abs(Jx).sum() + np.abs(Jy).sum()),
        "births": births, "deaths": deaths,
    }


def run_emergence(p, on_frame=None, verbose: bool = False) -> dict:
    """Run the agent model from t = 0 to t_obs.

    Input:  p (Params), on_frame (optional function State -> None, called every frame_stride
            PDE steps - the video renderer), verbose (print progress).
    Output: history (dict of numpy arrays keyed by HISTORY_KEYS) plus "final" (last State).
    Steps:  1) set up grid, agents, fields,
            2) loop over PDE steps; every agent step run the agent life cycle and refresh,
            3) every PDE step: eat, nutrient, signal, QS chemistry,
            4) every frame: label subnetworks, record readouts, call on_frame.
    Types:  (Params, Callable[[State], None] | None, bool) -> dict.
    Role in model: the top-level chain. The four Stage 1 experiments are this function with
        share_on / pause_on switched on or off.
    """
    grid = make_grid(p)
    rng = np.random.default_rng(p.seed)
    agents = seed_nucleation(rng, p)
    U, L, G = (np.zeros((grid.n, grid.n)) for _ in range(3))
    S = initial_nutrient(grid, p)
    zero = np.zeros_like(S)
    Jx, Jy = np.zeros((grid.n - 1, grid.n)), np.zeros((grid.n, grid.n - 1))
    info = agent_cell_info(agents, G, grid, p)
    history = {k: [] for k in HISTORY_KEYS}
    births = deaths = 0
    n_steps = int(round(p.t_obs / p.tau))
    agent_every = max(1, int(round(p.agent_dt / p.tau)))

    def frame(t):
        contact = contact_subnetworks(info["rho"])
        transport, sx, sy = transport_subnetworks(Jx, Jy)
        row = readouts(t, agents, S, U, contact, transport, Jx, Jy, births, deaths, grid, p)
        for k in HISTORY_KEYS:
            history[k].append(row[k])
        if on_frame is not None:
            on_frame(State(t=t, grid=grid, U=U, L=L, S=S, G=G, agents=agents, rho=info["rho"],
                           Jx=Jx, Jy=Jy, p=p, history=history, contact=contact,
                           transport=transport, strong_x=sx, strong_y=sy))

    frame(0.0)
    for k in range(n_steps):
        t = k * p.tau
        if k % agent_every == 0 and agents.n:
            sense_qs(agents, U, grid, p)
            agents, died = burn_and_die(agents, p.agent_dt, p)
            agents, born = divide(agents, rng, t, p)
            Jx, Jy = share_food(agents, grid, p, p.agent_dt)
            move(agents, S, info["rho"], grid, rng, p.agent_dt, p)
            info = agent_cell_info(agents, G, grid, p)
            births, deaths = births + born, deaths + died

        S = eat(S, agents, info, p)
        S = diffuse_step(S, 0.0, zero, p.D_S, p.tau, p.h)
        if p.pause_on:
            G = diffuse_step(G, p.gamma_G, signal_source(info["starving_cells"], info["relay_cells"], p), p.D_G, p.tau, p.h)
        U, L, _ = qs_step(U, L, agent_qs_sources(info["rho"], p), p)

        if (k + 1) % p.frame_stride == 0:
            frame((k + 1) * p.tau)
            births = deaths = 0
            if verbose and (k + 1) % (p.frame_stride * 40) == 0:
                h = history
                print(f"  t={h['t'][-1]:5.1f} h  agents={h['n_agents'][-1]:5d}  "
                      f"starving={h['n_starving'][-1]:5d}  R95={h['radius95'][-1]:5.1f} um  "
                      f"S_centre={h['S_centre'][-1]:.3f}  U_max={h['U_max'][-1]:.2e}")

    out = {k: np.array(v) for k, v in history.items()}
    out["final"] = State(t=n_steps * p.tau, grid=grid, U=U, L=L, S=S, G=G, agents=agents,
                         rho=info["rho"], Jx=Jx, Jy=Jy, p=p)
    return out
