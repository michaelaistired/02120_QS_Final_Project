"""Individual cells (agents). Each agent is a small cluster (~10^3) of bacteria.

Stored "structure of arrays" style (one numpy array per property) so thousands of agents
update with vector operations instead of a Python loop.

Life cycle each agent step (agent_dt):
    sense (QS activation)  ->  burn food / die  ->  divide  ->  [share food: network.py]  ->  move
Eating happens every PDE step in engine.py (food comes from the nutrient field).
"""

from dataclasses import dataclass

import numpy as np

from .fields import hill
from .grid import Grid, cell_index, count_per_cell, gradient, sample


@dataclass
class Agents:
    """All agents.

    Fields (each shape (N,)):
        x, y        position, um
        food        stored food reserve (same units as nutrient S x one grid cell)
        activation  QS switch state 0-1 (AHL sensed at its cell)
        born        time of birth, h
    Role in model: the individual cells the video draws as dots. Each one senses, eats,
        divides, dies and moves on its own; colony shape and networks emerge from that.
    """

    x: np.ndarray
    y: np.ndarray
    food: np.ndarray
    activation: np.ndarray
    born: np.ndarray

    @property
    def n(self) -> int:
        """Number of agents."""
        return len(self.x)

    def keep(self, mask: np.ndarray) -> "Agents":
        """Agents where mask is True (used for death).

        Input: mask ((N,) bool).  Output: new Agents.  Steps: index every array.
        Types: ndarray[bool] -> Agents.  Role: removes dead cells.
        """
        return Agents(self.x[mask], self.y[mask], self.food[mask], self.activation[mask],
                      self.born[mask])

    def extend(self, other: "Agents") -> "Agents":
        """These agents plus other (used for division).

        Input: other (Agents).  Output: new Agents.  Steps: concatenate every array.
        Types: Agents -> Agents.  Role: adds daughter cells.
        """
        return Agents(*(np.concatenate([getattr(self, f), getattr(other, f)])
                        for f in ("x", "y", "food", "activation", "born")))


def seed_nucleation(rng: np.random.Generator, p) -> Agents:
    """Place the founding agents in a small disc at the domain centre.

    Input:  rng, p (n_seed, seed_radius, length, food_init).
    Output: Agents.
    Steps:  1) uniform random points in a disc (sqrt for even area density), 2) equal food.
    Types:  (Generator, Params) -> Agents.
    Role in model: the nucleation site the colony expands out of.
    """
    r = p.seed_radius * np.sqrt(rng.uniform(size=p.n_seed))
    th = rng.uniform(0, 2 * np.pi, size=p.n_seed)
    c = p.length / 2.0
    return Agents(c + r * np.cos(th), c + r * np.sin(th), np.full(p.n_seed, p.food_init),
                  np.zeros(p.n_seed), np.zeros(p.n_seed))


def sense_qs(agents: Agents, U: np.ndarray, grid: Grid, p) -> None:
    """Each agent reads AHL at its cell and updates its QS activation (in place).

    Input:  agents, U ((n, n) AHL), grid, p (U_th, n).
    Output: None (agents.activation updated).
    Steps:  1) find cells, 2) sample U, 3) Hill switch at U_th.
    Types:  (Agents, ndarray, Grid, Params) -> None.
    Role in model: QS state per cell - shown in the video and (optionally) boosting foraging.
    """
    i, j = cell_index(agents.x, agents.y, grid)
    agents.activation = hill(sample(U, i, j), p.U_th, p.n)


def burn_and_die(agents: Agents, dt: float, p) -> tuple:
    """Pay the cost of staying alive; agents with no food left die.

    Input:  agents, dt (h), p (maintenance).
    Output: (surviving Agents, number that died).
    Steps:  1) food -= maintenance * dt, 2) keep agents with food > 0.
    Types:  (Agents, float, Params) -> tuple[Agents, int].
    Role in model: starvation is lethal, so a depleted centre slowly empties unless food
        reaches it - the pressure that makes the two food-route hypotheses matter.
    """
    agents.food = agents.food - p.maintenance * dt
    alive = agents.food > 0.0
    return agents.keep(alive), int((~alive).sum())


def divide(agents: Agents, rng: np.random.Generator, t: float, p) -> tuple:
    """Well-fed agents split in two.

    Input:  agents, rng, t (h, birth time for daughters), p (food_divide, max_agents, h).
    Output: (Agents including daughters, number of divisions).
    Steps:  1) pick agents with food >= food_divide (respecting max_agents),
            2) halve their food, 3) place a daughter up to half a cell away with the other half.
    Types:  (Agents, Generator, float, Params) -> tuple[Agents, int].
    Role in model: growth. Division happens only where food is, so the colony grows where
        nutrient is - the root of outward expansion from the nucleation site.
    """
    idx = np.flatnonzero(agents.food >= p.food_divide)
    idx = idx[: max(0, p.max_agents - agents.n)]
    if idx.size == 0:
        return agents, 0
    agents.food[idx] *= 0.5
    ang = rng.uniform(0, 2 * np.pi, idx.size)
    off = 0.5 * p.h
    kids = Agents(agents.x[idx] + off * np.cos(ang), agents.y[idx] + off * np.sin(ang),
                  agents.food[idx].copy(), agents.activation[idx].copy(), np.full(idx.size, t))
    return agents.extend(kids), int(idx.size)


def move(agents: Agents, S: np.ndarray, rho: np.ndarray, grid: Grid, rng: np.random.Generator,
         dt: float, p) -> None:
    """Move every agent: random walk + chemotaxis toward food + push out of crowds (in place).

    Input:  agents, S ((n, n) nutrient), rho ((n, n) agents per cell), grid, rng, dt, p.
    Output: None (agents.x, agents.y updated).
    Steps:  1) hungry = food below half the division threshold,
            2) random step sqrt(2 D dt) * normal, D = motility_hungry (x QS boost) or motility_fed,
            3) hungry agents drift up the nutrient gradient (chemotaxis * grad S),
            4) everyone drifts down the gradient of crowding above crowd_max,
            5) cap each step at one grid cell, keep agents inside the interior.
    Types:  (Agents, ndarray, ndarray, Grid, Generator, float, Params) -> None.
    Role in model: foraging. Starving cells at the edge search outward for food; fed cells
        mostly stay put. Branches and fronts are not programmed - they come from this rule.
    """
    i, j = cell_index(agents.x, agents.y, grid)
    hungry = agents.food < 0.5 * p.food_divide
    D = np.where(hungry, p.motility_hungry, p.motility_fed)
    if p.qs_motility_on:
        D = D * (1.0 + (p.qs_motility_boost - 1.0) * agents.activation)
    step = np.sqrt(2.0 * D * dt)[:, None] * rng.normal(size=(agents.n, 2))

    Sx, Sy = gradient(S, grid.h)
    chem = p.chemotaxis * hungry[:, None] * np.stack([sample(Sx, i, j), sample(Sy, i, j)], axis=1)
    Cx, Cy = gradient(np.maximum(rho - p.crowd_max, 0.0), grid.h)
    crowd = -p.crowd_push * np.stack([sample(Cx, i, j), sample(Cy, i, j)], axis=1)

    disp = step + (chem + crowd) * dt
    norm = np.linalg.norm(disp, axis=1, keepdims=True)
    disp *= np.minimum(1.0, grid.h / np.maximum(norm, 1e-12))
    lo, hi = grid.h, p.length - grid.h
    agents.x = np.clip(agents.x + disp[:, 0], lo, hi)
    agents.y = np.clip(agents.y + disp[:, 1], lo, hi)


def per_cell(agents: Agents, grid: Grid, weights=None) -> np.ndarray:
    """Shortcut: count (or sum a property of) agents in each cell.

    Input:  agents, grid, weights (optional (N,)).   Output: (n, n).
    Steps:  1) cell_index, 2) count_per_cell.
    Types:  (Agents, Grid, ndarray|None) -> ndarray.
    Role in model: rho (density), food per cell, starving per cell all come from this.
    """
    i, j = cell_index(agents.x, agents.y, grid)
    return count_per_cell(i, j, grid, weights)
