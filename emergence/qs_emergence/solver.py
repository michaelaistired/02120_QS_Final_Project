"""Peaceman-Rachford ADI + Thomas algorithm for  du/dt = D Laplacian(u) - q u + F.

Same scheme as the paper (Eqs. 7-11) and as qs_model/solver.py, with one generalization:
edge values are whatever is stored in u's boundary rows (held fixed in time). For AHL and
lactonase they are 0 (the paper's absorbing edges); for nutrient they are S_edge (a food
supply surrounding the colony, like the agar around a biofilm).

    thomas_solve          many tridiagonal chains at once
    implicit_x_half_step  half step, implicit in x, explicit in y, fixed edges
    implicit_y_half_step  same with x and y swapped (transpose trick)
    diffuse_step          one full ADI step for a LINEAR field (nutrient, signal)
    qs_step               one full ADI step for the coupled AHL/lactonase pair (L -> U -> L)
"""

import numpy as np


def thomas_solve(lower, diag, upper, rhs: np.ndarray) -> np.ndarray:
    """Solve many tridiagonal systems  lower*x[i-1] + diag*x[i] + upper*x[i+1] = rhs[i].

    Input:  lower, diag, upper (scalars or arrays broadcastable to rhs), rhs ((m, k): m unknowns
            along each of k chains).
    Output: x, shape (m, k).
    Steps:  1) forward sweep eliminating the lower neighbour, 2) back substitution.
    Types:  (float|ndarray, ndarray, float|ndarray, ndarray[m, k]) -> ndarray[m, k].
    Role in model: the fast inner solver (Q3). ADI turns each grid row into a chain where every
        point touches only 2 neighbours; Thomas solves a chain in ~m steps.
    """
    m = rhs.shape[0]
    lower = np.broadcast_to(lower, rhs.shape)
    diag = np.broadcast_to(diag, rhs.shape)
    upper = np.broadcast_to(upper, rhs.shape)
    c = np.empty(rhs.shape)
    d = np.empty(rhs.shape)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, m):
        denom = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / denom
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom
    x = np.empty(rhs.shape)
    x[m - 1] = d[m - 1]
    for i in range(m - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def implicit_x_half_step(u: np.ndarray, q, F: np.ndarray, D: float, tau: float,
                         hx: float, hy: float) -> np.ndarray:
    """Advance by tau/2: implicit in x, explicit in y; boundary rows of u are kept fixed.

    Input:  u ((n, n), boundary rows/cols hold the fixed edge values), q (loss rate, scalar or
            (n, n)), F ((n, n) source), D, tau (FULL step), hx, hy.
    Output: (n, n) field after the half step (same edge values as u).
    Steps:  1) rhs = u + (D tau/2) d2u/dy2 + (tau/2) F at interior points,
            2) move the known edge values in x onto the rhs (zero for AHL/lactonase),
            3) chain coefficients: neighbours -D tau/(2 hx^2), centre 1 + D tau/hx^2 + tau q/2,
            4) thomas_solve along x for all interior rows, 5) copy edges back.
    Types:  (ndarray, float|ndarray, ndarray, float, float, float, float) -> ndarray[n, n].
    Role in model: paper Eq. 8. With zero edges this is exactly qs_model's half step (checked
        by the Stage 0 control); with S_edge it lets food diffuse in from the surroundings.
    """
    r = D * tau / (2.0 * hx ** 2)
    q_int = np.broadcast_to(q, u.shape)[1:-1, 1:-1]
    rhs = (u[1:-1, 1:-1]
           + 0.5 * D * tau * (u[1:-1, :-2] - 2.0 * u[1:-1, 1:-1] + u[1:-1, 2:]) / hy ** 2
           + 0.5 * tau * F[1:-1, 1:-1])
    rhs[0] += r * u[0, 1:-1]
    rhs[-1] += r * u[-1, 1:-1]
    diag = 1.0 + 2.0 * r + 0.5 * tau * q_int
    new = u.copy()
    new[1:-1, 1:-1] = thomas_solve(-r, diag, -r, rhs)
    return new


def implicit_y_half_step(u: np.ndarray, q, F: np.ndarray, D: float, tau: float,
                         hx: float, hy: float) -> np.ndarray:
    """Advance by tau/2: implicit in y. Transpose, reuse the x version, transpose back.

    Input/Output/Types: as implicit_x_half_step.
    Steps:  1) swap axes, 2) implicit_x_half_step, 3) swap back.
    Role in model: paper Eq. 9, the second half of each Peaceman-Rachford step.
    """
    q_t = q.T if np.ndim(q) == 2 else q
    return implicit_x_half_step(u.T, q_t, F.T, D, tau, hy, hx).T


def diffuse_step(u: np.ndarray, q, F: np.ndarray, D: float, tau: float, h: float) -> np.ndarray:
    """One full ADI step for a linear field (q and F do not depend on u).

    Input:  u ((n, n)), q (scalar or (n, n)), F ((n, n)), D, tau, h.
    Output: (n, n) field at t + tau.
    Steps:  1) x-implicit half step, 2) y-implicit half step.
    Types:  (ndarray, float|ndarray, ndarray, float, float, float) -> ndarray.
    Role in model: moves nutrient S and starvation signal G. Same diffuse/decay equation form
        as the paper's AHL and lactonase, so it reuses the same solver.
    """
    u = implicit_x_half_step(u, q, F, D, tau, h, h)
    return implicit_y_half_step(u, q, F, D, tau, h, h)


def qs_step(U: np.ndarray, L: np.ndarray, sources, p) -> tuple:
    """One full ADI step for AHL (U) and lactonase (L), with the paper's L -> U -> L ordering.

    Input:  U, L ((n, n)), sources (function (U, k) -> (F1, F2), the production terms;
            k = 0 for the x half step, 1 for the y half step), p (Params:
            D_U, D_L, gamma_U, gamma_L, gamma_LU, tau, h, picard_tol, picard_max).
    Output: (U, L, iterations) at t + tau.
    Steps:  for each half step (x then y):
            1) lactonase from old AHL,
            2) AHL with that lactonase held fixed (Q1: gamma_LU * L * U), repeating
               "source from current guess -> solve" until stable (Hill nonlinearity),
            3) lactonase again from the new AHL.
    Types:  (ndarray, ndarray, Callable[[ndarray, int], tuple], Params) -> tuple[ndarray, ndarray, int].
    Role in model: the paper's chemistry. Passing `sources` as a function lets the SAME code
        run the paper's Gaussian colonies (Stage 0) and individual agents (Stage 1).
    """
    total = 0
    for k, half in enumerate((implicit_x_half_step, implicit_y_half_step)):
        _, F2 = sources(U, k)
        L_new = half(L, p.gamma_L, F2, p.D_L, p.tau, p.h, p.h)
        q_U = p.gamma_U + p.gamma_LU * L_new
        guess = U
        for it in range(1, p.picard_max + 1):
            F1, _ = sources(guess, k)
            nxt = half(U, q_U, F1, p.D_U, p.tau, p.h, p.h)
            change = np.linalg.norm(nxt - guess) / max(np.linalg.norm(nxt), 1e-300)
            guess = nxt
            if change < p.picard_tol:
                break
        total += it
        U = guess
        _, F2 = sources(U, k)
        L = half(L, p.gamma_L, F2, p.D_L, p.tau, p.h, p.h)
    return U, L, total
