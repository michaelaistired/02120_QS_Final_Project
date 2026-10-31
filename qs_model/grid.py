"""The 2D space grid that AHL (U) and lactonase (L) live on."""

from dataclasses import dataclass
import numpy as np

from .params import Params


@dataclass(frozen=True)
class Grid:
    """Grid points of the square domain [0, l] x [0, l].

    Fields:
        x, y : 1D arrays of coordinates, shape (n,), um
        X, Y : 2D coordinate arrays, shape (n, n); X[i, j] = x[i], Y[i, j] = y[j]
        h    : spacing, um
        n    : points per side (including the two boundary points)
    Role in model: every concentration field is an (n, n) array on this grid. Index i runs
        along x and j along y, matching the paper's u_{i,j}. Boundary rows/columns are always 0
        (the paper's zero Dirichlet condition: edges act as perfect sinks).
    """

    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    h: float
    n: int


def make_grid(p: Params) -> Grid:
    """Build the space grid from the parameters.

    Input:  p (Params) - uses p.length and p.h.
    Output: Grid with n = length/h + 1 points per side.
    Steps:  1) count points, 2) make 1D coordinates, 3) make 2D coordinate arrays.
    Types:  Params -> Grid (numpy float64 arrays).
    Role in model: the paper's grid Omega_{h1,h2} (100 x 100 um, h = 1 um gives 101 x 101
        points). Colony footprints and both PDEs are evaluated on these points.
    """
    n = int(round(p.length / p.h)) + 1
    x = np.linspace(0.0, p.length, n)
    y = np.linspace(0.0, p.length, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return Grid(x=x, y=y, X=X, Y=Y, h=p.h, n=n)


def zero_field(grid: Grid) -> np.ndarray:
    """A concentration field that is zero everywhere.

    Input:  grid (Grid).
    Output: (n, n) float array of zeros.
    Steps:  1) allocate zeros.
    Types:  Grid -> np.ndarray[float64] of shape (n, n).
    Role in model: the paper's initial condition U(x,y,0) = L(x,y,0) = 0 (Eq. 3).
    """
    return np.zeros((grid.n, grid.n))


def value_at(field: np.ndarray, grid: Grid, x: float, y: float) -> float:
    """Read a field at the grid point closest to (x, y).

    Input:  field ((n, n) array), grid, x and y in um.
    Output: the field value (float) at the nearest grid point.
    Steps:  1) convert coordinates to indices, 2) clip to the grid, 3) look up.
    Types:  (ndarray, Grid, float, float) -> float.
    Role in model: lets a colony "sense" the AHL concentration at its center. Only used by
        the Q7 feedback and the QS-activation readout; the paper itself never needs it.
    """
    i = int(np.clip(round(x / grid.h), 0, grid.n - 1))
    j = int(np.clip(round(y / grid.h), 0, grid.n - 1))
    return float(field[i, j])
