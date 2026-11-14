"""The 2D grid, and the two-way link between individual agents and grid fields."""

from dataclasses import dataclass

import numpy as np

from .params import Params


@dataclass(frozen=True)
class Grid:
    """Grid points of [0, l] x [0, l].

    Fields: x, y (1D coords, shape (n,)), X, Y ((n, n), X[i, j] = x[i]), h (spacing), n (points/side).
    Role in model: every field (AHL, lactonase, nutrient, signal) is an (n, n) array here;
        index i runs along x, j along y. Each grid point also stands for a square "cell" of
        side h around it - agents in the same or neighbouring cells are in contact.
    """

    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    h: float
    n: int


def make_grid(p: Params) -> Grid:
    """Build the grid.

    Input:  p (Params: length, h).   Output: Grid with n = length/h + 1 points per side.
    Steps:  1) count points, 2) 1D coordinates, 3) 2D coordinate arrays.
    Types:  Params -> Grid.
    Role in model: the space that both the PDEs and the agents live in.
    """
    n = int(round(p.length / p.h)) + 1
    x = np.linspace(0.0, p.length, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return Grid(x=x, y=x.copy(), X=X, Y=Y, h=p.h, n=n)


def cell_index(x: np.ndarray, y: np.ndarray, grid: Grid) -> tuple:
    """Which grid cell each agent sits in.

    Input:  x, y (agent positions, shape (N,), um), grid.
    Output: (i, j) integer arrays, shape (N,), clipped to the interior (never on the edge).
    Steps:  1) round position / h, 2) clip to 1..n-2.
    Types:  (ndarray, ndarray, Grid) -> tuple[ndarray[int], ndarray[int]].
    Role in model: the bridge between agents and fields. Agents are kept off the boundary
        row because edge values are fixed by the boundary conditions.
    """
    i = np.clip(np.rint(x / grid.h).astype(int), 1, grid.n - 2)
    j = np.clip(np.rint(y / grid.h).astype(int), 1, grid.n - 2)
    return i, j


def count_per_cell(i: np.ndarray, j: np.ndarray, grid: Grid, weights=None) -> np.ndarray:
    """Add up agents (or an agent property) in each grid cell.

    Input:  i, j (cell indices, shape (N,)), grid, weights (optional (N,) values to add instead of 1).
    Output: (n, n) array.
    Steps:  1) flatten (i, j) to one index, 2) np.bincount with weights, 3) reshape.
    Types:  (ndarray, ndarray, Grid, ndarray|None) -> ndarray[n, n].
    Role in model: agents -> field. Agent density rho = count_per_cell(...) is what produces
        AHL/lactonase and eats nutrient; summed food per cell drives network sharing.
    """
    flat = i * grid.n + j
    out = np.bincount(flat, weights=weights, minlength=grid.n * grid.n)
    return out.reshape(grid.n, grid.n).astype(float)


def sample(field: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Read a field at each agent's cell.

    Input:  field ((n, n)), i, j (cell indices, shape (N,)).   Output: (N,) values.
    Steps:  1) fancy indexing.   Types: (ndarray, ndarray, ndarray) -> ndarray.
    Role in model: field -> agents. How each cell "senses" nutrient, AHL and the signal.
    """
    return field[i, j]


def gradient(field: np.ndarray, h: float) -> tuple:
    """Spatial gradient of a field (central differences).

    Input:  field ((n, n)), h (spacing, um).   Output: (d/dx, d/dy), each (n, n), per um.
    Steps:  1) numpy.gradient along both axes.
    Types:  (ndarray, float) -> tuple[ndarray, ndarray].
    Role in model: chemotaxis (move up the nutrient gradient) and crowding (move down the
        density gradient) both need the direction a field increases in.
    """
    gx, gy = np.gradient(field, h)
    return gx, gy
