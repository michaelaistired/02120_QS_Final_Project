"""Contact network between cells: food sharing (hypothesis a) and subnetwork labels.

Two agents are LINKED when they sit in the same grid cell or in side-by-side cells (the grid
spacing h = 4 um is about one contact distance). Links are handled cell-to-cell so thousands
of agents can share food with vector operations.

    share_food          move food along links from better-fed to hungrier cells (conserved)
    components          group linked cells into connected pieces (union-find)
    contact_subnetworks every connected patch of cells
    transport_subnetworks only the links currently carrying a lot of food - the "supply routes"
"""

import numpy as np

from .agents import Agents
from .grid import Grid, cell_index, count_per_cell


def share_food(agents: Agents, grid: Grid, p, dt: float) -> tuple:
    """Pass food along links from better-fed to hungrier neighbouring cells (in place).

    Input:  agents, grid, p (share_on, share_rate), dt (h).
    Output: (Jx, Jy) food flow rates per link (food/h); Jx has shape (n-1, n) for the link
            between cells (a, b) and (a+1, b) (positive = toward +x); Jy is (n, n-1).
    Steps:  0) split dt into sub-steps so share_rate * sub_dt <= 0.2 (keeps the exchange stable),
            then for each sub-step:
            1) per cell: number of agents n_c, total food F_c, food per agent f_c,
            2) each link between two occupied cells moves  share_rate * dt * (f_a - f_b) * m
               with m = n_a n_b / (n_a + n_b) (so per-agent food evens out),
            3) add up each cell's net gain/loss,
            4) gains are split equally between the cell's agents; losses are taken in
               proportion to each agent's reserve (nobody is pushed below zero).
    Types:  (Agents, Grid, Params, float) -> tuple[ndarray, ndarray].
    Role in model: hypothesis (a). Food really moves from cell to cell, so a fed edge can
        feed a starving centre through chains of cells. Total food is conserved exactly.
    """
    n = grid.n
    Jx_tot, Jy_tot = np.zeros((n - 1, n)), np.zeros((n, n - 1))
    if not p.share_on or agents.n == 0:
        return Jx_tot, Jy_tot
    n_sub = max(1, int(np.ceil(p.share_rate * dt / 0.2)))
    i, j = cell_index(agents.x, agents.y, grid)
    for _ in range(n_sub):
        Jx, Jy = _share_once(agents, i, j, grid, p, dt / n_sub)
        Jx_tot += Jx
        Jy_tot += Jy
    return Jx_tot / dt, Jy_tot / dt


def _share_once(agents: Agents, i: np.ndarray, j: np.ndarray, grid: Grid, p, dt: float) -> tuple:
    """One stable sub-step of share_food (see share_food for the rule); returns food moved per link."""
    n_c = count_per_cell(i, j, grid)
    F_c = count_per_cell(i, j, grid, agents.food)
    f = np.divide(F_c, n_c, out=np.zeros_like(F_c), where=n_c > 0)

    def link(na, nb, fa, fb):
        both = (na > 0) & (nb > 0)
        m = np.divide(na * nb, na + nb, out=np.zeros_like(na), where=both)
        return p.share_rate * dt * (fa - fb) * m

    Jx = link(n_c[:-1, :], n_c[1:, :], f[:-1, :], f[1:, :])
    Jy = link(n_c[:, :-1], n_c[:, 1:], f[:, :-1], f[:, 1:])
    delta = np.zeros_like(F_c)
    delta[:-1, :] -= Jx
    delta[1:, :] += Jx
    delta[:, :-1] -= Jy
    delta[:, 1:] += Jy

    d, nc, Fc = delta[i, j], n_c[i, j], F_c[i, j]
    gain = np.where(d > 0, d / nc, 0.0)
    scale = np.where(d < 0, 1.0 + np.divide(d, Fc, out=np.zeros_like(d), where=Fc > 0), 1.0)
    agents.food = agents.food * scale + gain
    return Jx, Jy


def components(n_nodes: int, a: np.ndarray, b: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Connected components of a graph (union-find).

    Input:  n_nodes (size of the flat index space), a, b (link endpoints, flat indices),
            nodes (flat indices that belong to the graph, linked or not).
    Output: labels, shape (n_nodes,): 0 = not in graph, 1..k = component number.
    Steps:  1) every node starts as its own parent, 2) for each link merge the two roots,
            3) find each node's root, 4) renumber roots 1..k.
    Types:  (int, ndarray[int], ndarray[int], ndarray[int]) -> ndarray[int].
    Role in model: turns "who touches whom" into "which subnetwork am I in".
    """
    parent = np.arange(n_nodes)

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for u, v in zip(a.tolist(), b.tolist()):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[max(ru, rv)] = min(ru, rv)
    labels = np.zeros(n_nodes, dtype=int)
    roots = np.array([find(v) for v in nodes.tolist()], dtype=int)
    if roots.size:
        _, labels[nodes] = np.unique(roots, return_inverse=True)
        labels[nodes] += 1
    return labels


def contact_subnetworks(rho: np.ndarray) -> np.ndarray:
    """Label every connected patch of occupied cells.

    Input:  rho ((n, n) agents per cell).
    Output: (n, n) int labels (0 = empty).
    Steps:  1) occupied cells, 2) links between side-by-side occupied cells, 3) components.
    Types:  ndarray -> ndarray[int].
    Role in model: the coarse picture - separate islands of cells (e.g. fragments left after
        the centre dies, or branches that pinched off).
    """
    n = rho.shape[0]
    occ = rho > 0
    idx = np.arange(n * n).reshape(n, n)
    ax = occ[:-1, :] & occ[1:, :]
    ay = occ[:, :-1] & occ[:, 1:]
    a = np.concatenate([idx[:-1, :][ax], idx[:, :-1][ay]])
    b = np.concatenate([idx[1:, :][ax], idx[:, 1:][ay]])
    return components(n * n, a, b, idx[occ]).reshape(n, n)


def transport_subnetworks(Jx: np.ndarray, Jy: np.ndarray, frac: float = 0.05) -> tuple:
    """Label the "supply routes": links carrying at least frac of the largest food flow.

    Input:  Jx ((n-1, n)), Jy ((n, n-1)) food flows from share_food, frac (0-1).
    Output: (labels (n, n) int, 0 = not on a route; strong_x, strong_y bool masks of the links).
    Steps:  1) threshold = frac * max |flow|, 2) keep strong links, 3) components of their cells.
    Types:  (ndarray, ndarray, float) -> tuple[ndarray, ndarray, ndarray].
    Role in model: the "subnetworks that send food to the centre". These are not drawn by
        hand - they are whichever chains of cells end up carrying the food.
    """
    n = Jy.shape[0]
    peak = max(np.abs(Jx).max(initial=0.0), np.abs(Jy).max(initial=0.0))
    if peak <= 0:
        return np.zeros((n, n), dtype=int), np.zeros_like(Jx, bool), np.zeros_like(Jy, bool)
    sx, sy = np.abs(Jx) >= frac * peak, np.abs(Jy) >= frac * peak
    idx = np.arange(n * n).reshape(n, n)
    a = np.concatenate([idx[:-1, :][sx], idx[:, :-1][sy]])
    b = np.concatenate([idx[1:, :][sx], idx[:, 1:][sy]])
    nodes = np.unique(np.concatenate([a, b]))
    return components(n * n, a, b, nodes).reshape(n, n), sx, sy
