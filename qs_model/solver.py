"""Numerical solution of the two reaction-diffusion PDEs (paper Eqs. 1-2, scheme Eqs. 7-11).

    dU/dt = D_U * Laplacian(U) - (gamma_U + gamma_LU * L) * U + F1      (AHL)
    dL/dt = D_L * Laplacian(L) -  gamma_L                 * L + F2      (lactonase)

Both have the general form  du/dt = D * Laplacian(u) - q * u + F  (paper Eq. 7), so ONE solver
handles both; only (D, q, F) change. The pieces, from smallest to largest:

    thomas_solve            solve many "chain" (tridiagonal) systems at once        (Q3)
    second_difference_y     the explicit (known) half of the Laplacian
    implicit_x_half_step    half a time step: implicit in x, explicit in y           (Q3)
    implicit_y_half_step    half a time step: implicit in y, explicit in x
    ahl_loss_rate           q for AHL = natural decay + destruction by lactonase    (Q1)
    coupled_half_step       L -> U (fixed-point loop) -> L ordering from the paper   (Q1)
    adi_step                one full Peaceman-Rachford step = x half + y half
"""

import numpy as np

from .params import Params
from .production import source_terms


def thomas_solve(lower, diag, upper, rhs: np.ndarray) -> np.ndarray:
    """Solve many tridiagonal (chain-shaped) linear systems at the same time.

    Input:  lower, diag, upper - coefficients of each chain, scalars or arrays broadcastable to
                rhs; rhs - array of shape (m, k): m unknowns along each chain, k separate chains.
            Equation for point i of a chain:
                lower * x[i-1] + diag[i] * x[i] + upper * x[i+1] = rhs[i]
    Output: x, array of shape (m, k).
    Steps:  1) forward sweep: walk down the chain, eliminating the "lower" neighbour so each
               equation only mentions itself and the next point,
            2) back substitution: the last point is now known; walk back up filling in the rest.
    Types:  (float|ndarray, ndarray, float|ndarray, ndarray[m, k]) -> ndarray[m, k].
    Role in model: answer to Q3. Because ADI handles one direction at a time, every grid row
        becomes a chain where each point only touches its two neighbours. Thomas solves such a
        chain in ~m steps (instead of ~m^3 for a general solver). All k rows are solved together
        with numpy vector operations, which is what keeps a 50 h run to about a minute.
    """
    m = rhs.shape[0]
    lower = np.broadcast_to(lower, rhs.shape)
    diag = np.broadcast_to(diag, rhs.shape)
    upper = np.broadcast_to(upper, rhs.shape)

    c_prime = np.empty(rhs.shape)  # modified upper coefficients
    d_prime = np.empty(rhs.shape)  # modified right-hand side

    # 1) forward sweep
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    for i in range(1, m):
        denom = diag[i] - lower[i] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denom

    # 2) back substitution
    x = np.empty(rhs.shape)
    x[m - 1] = d_prime[m - 1]
    for i in range(m - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def second_difference_y(u: np.ndarray, hy: float) -> np.ndarray:
    """Discrete d^2u/dy^2 at interior points, using values already known.

    Input:  u ((n, n) field, boundary included), hy (grid spacing in y, um).
    Output: ((n-2), (n-2)) array for interior points.
    Steps:  1) (u[j-1] - 2 u[j] + u[j+1]) / hy^2 along axis 1 (y), interior rows only.
    Types:  (ndarray[n, n], float) -> ndarray[n-2, n-2].
    Role in model: the EXPLICIT half of the Laplacian in an x-implicit half-step. Using known
        values in y is what lets the x-direction stay a simple chain for thomas_solve.
    """
    return (u[1:-1, :-2] - 2.0 * u[1:-1, 1:-1] + u[1:-1, 2:]) / hy ** 2


def implicit_x_half_step(u: np.ndarray, q, F: np.ndarray, D: float, tau: float,
                         hx: float, hy: float) -> np.ndarray:
    """Advance du/dt = D Laplacian(u) - q u + F by tau/2: implicit in x, explicit in y.

    Input:  u ((n, n) field at the start of the half step), q (loss rate: scalar or (n, n)),
            F ((n, n) source), D (diffusion), tau (FULL time step; this advances tau/2), hx, hy.
    Output: (n, n) field after the half step, with zero boundary values.
    Steps:  1) right-hand side = u + (D tau / 2) * d2u/dy2 (explicit) + (tau / 2) * F,
            2) chain coefficients along x: neighbours -D tau / (2 hx^2),
               centre 1 + D tau / hx^2 + tau q / 2,
            3) thomas_solve along x for every interior y-row at once,
            4) put the solution back inside a zero boundary (Dirichlet U = L = 0 on the edges).
    Types:  (ndarray, float|ndarray, ndarray, float, float, float, float) -> ndarray[n, n].
    Role in model: paper Eq. 8 (first half-step of Peaceman-Rachford). Treating x implicitly
        keeps the step stable even though the AHL loss rate can be very large inside colonies
        where lactonase is high (gamma_LU * L can reach ~10^3 per hour).
    """
    q_int = np.broadcast_to(q, u.shape)[1:-1, 1:-1]
    rhs = u[1:-1, 1:-1] + 0.5 * D * tau * second_difference_y(u, hy) + 0.5 * tau * F[1:-1, 1:-1]
    off = -D * tau / (2.0 * hx ** 2)
    diag = 1.0 + D * tau / hx ** 2 + 0.5 * tau * q_int
    new = np.zeros_like(u)
    new[1:-1, 1:-1] = thomas_solve(off, diag, off, rhs)  # axis 0 = x is the chain direction
    return new


def implicit_y_half_step(u: np.ndarray, q, F: np.ndarray, D: float, tau: float,
                         hx: float, hy: float) -> np.ndarray:
    """Advance by tau/2: implicit in y, explicit in x (the second Peaceman-Rachford half-step).

    Input/Output/Types: same as implicit_x_half_step.
    Steps:  1) transpose (swap x and y), 2) reuse implicit_x_half_step, 3) transpose back.
    Role in model: paper Eq. 9. Alternating the implicit direction every half step is what
        makes the method second-order accurate and symmetric in x and y.
    """
    q_t = q.T if np.ndim(q) == 2 else q
    return implicit_x_half_step(u.T, q_t, F.T, D, tau, hy, hx).T


def ahl_loss_rate(L: np.ndarray, p: Params) -> np.ndarray:
    """q for the AHL equation: how fast AHL disappears per unit AHL.

    Input:  L ((n, n) lactonase field, mol/L), p (Params) - uses gamma_U, gamma_LU.
    Output: (n, n) array, 1/h.
    Steps:  1) natural decay gamma_U plus lactonase destruction gamma_LU * L.
    Types:  (ndarray, Params) -> ndarray.
    Role in model: answer to Q1. The product gamma_LU * L * U is the ONLY place where L
        affects U, and it is nonlinear (two unknowns multiplied). We handle it by holding L
        fixed while solving for U, which is why coupled_half_step solves L first.
    """
    return p.gamma_U + p.gamma_LU * L


def coupled_half_step(U: np.ndarray, L: np.ndarray, footprint: np.ndarray, N: float,
                      p: Params, direction: str) -> tuple:
    """One half step for BOTH molecules, using the paper's L -> U -> L ordering.

    Input:  U, L ((n, n) fields at the start), footprint ((n, n) colony footprints),
            N (density at this time), p (Params), direction ("x" or "y" = implicit direction).
    Output: (U_new, L_new, iterations) - fields after the half step and the number of
            fixed-point iterations the AHL equation needed.
    Steps:  1) lactonase: solve with production from the OLD AHL,
            2) AHL: hold that new lactonase fixed (Q1) and repeat
                   F1 <- source from current guess; U_guess <- solve
               until the guess stops changing (fixed-point / Picard iteration; handles the
               nonlinear Hill switch inside F1),
            3) lactonase again: re-solve with production from the NEW AHL.
    Types:  (ndarray, ndarray, ndarray, float, Params, str) -> tuple[ndarray, ndarray, int].
    Role in model: the paper's iterative procedure (text below Eq. 8). It turns two coupled
        nonlinear PDEs into a sequence of linear chain solves that thomas_solve can do.
    """
    half_step = implicit_x_half_step if direction == "x" else implicit_y_half_step
    h = p.h

    # 1) lactonase from old AHL
    _, F2 = source_terms(U, footprint, N, p)
    L_new = half_step(L, p.gamma_L, F2, p.D_L, p.tau, h, h)

    # 2) AHL with lactonase held fixed; fixed-point loop for the Hill nonlinearity
    q_U = ahl_loss_rate(L_new, p)
    U_guess = U
    iterations = 0
    for iterations in range(1, p.picard_max + 1):
        F1, _ = source_terms(U_guess, footprint, N, p)
        U_next = half_step(U, q_U, F1, p.D_U, p.tau, h, h)
        change = np.linalg.norm(U_next - U_guess) / max(np.linalg.norm(U_next), 1e-300)
        U_guess = U_next
        if change < p.picard_tol:
            break
    U_new = U_guess

    # 3) lactonase again from new AHL
    _, F2 = source_terms(U_new, footprint, N, p)
    L_new = half_step(L, p.gamma_L, F2, p.D_L, p.tau, h, h)
    return U_new, L_new, iterations


def adi_step(U: np.ndarray, L: np.ndarray, footprint: np.ndarray, N_start: float,
             N_end: float, p: Params) -> tuple:
    """One full PDE time step (tau) with the Peaceman-Rachford ADI method.

    Input:  U, L ((n, n) fields at time t), footprint, N_start = N(t), N_end = N(t + tau), p.
    Output: (U, L, iterations) at time t + tau.
    Steps:  1) coupled half step implicit in x (density at t),
            2) coupled half step implicit in y (density at t + tau).
    Types:  (ndarray, ndarray, ndarray, float, float, Params) -> tuple[ndarray, ndarray, int].
    Role in model: the "numerically solve a system of PDEs" box in the paper's flowchart.
        simulation.py calls this colony_dt / tau = 50 times between colony updates.
    """
    U, L, it1 = coupled_half_step(U, L, footprint, N_start, p, "x")
    U, L, it2 = coupled_half_step(U, L, footprint, N_end, p, "y")
    return U, L, it1 + it2
