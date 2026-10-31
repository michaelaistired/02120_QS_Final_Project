"""Fast unit tests for every building block. Run with:  python -m tests.test_model
(also works with pytest if it is installed).

Each test checks one subroutine against an answer we know independently.
"""

import math

import numpy as np

from qs_model.antibiotic import antibiotic_concentration, kill_fraction, rayleigh_pulse
from qs_model.colonies import Colony, logistic_step
from qs_model.controls import deterministic_checks, deterministic_control
from qs_model.density import density
from qs_model.feedback import effective_growth_rate, qs_activation
from qs_model.params import PAPER, half_life
from qs_model.production import hill, lactonase_production, source_terms, total_footprint
from qs_model.grid import make_grid
from qs_model.solver import implicit_x_half_step, implicit_y_half_step, thomas_solve


def test_thomas_matches_dense_solver():
    """Q3: Thomas must give the same answer as a general solver on random chains."""
    rng = np.random.default_rng(1)
    m, k = 30, 5
    lower, upper = -rng.uniform(0.1, 1, (m, k)), -rng.uniform(0.1, 1, (m, k))
    diag = 3.0 + rng.uniform(0, 1, (m, k))          # diagonally dominant -> stable
    rhs = rng.normal(size=(m, k))
    x = thomas_solve(lower, diag, upper, rhs)
    for col in range(k):
        A = np.diag(diag[:, col]) + np.diag(lower[1:, col], -1) + np.diag(upper[:-1, col], 1)
        assert np.allclose(A @ x[:, col], rhs[:, col])


def test_adi_pure_diffusion_matches_exact_solution():
    """A Gaussian blob spreading by diffusion has a known exact answer; ADI should match it."""
    p = PAPER
    g = make_grid(p)
    D, t0, t_run, tau = 10.0, 1.0, 1.0, 0.01
    r2 = (g.X - 50) ** 2 + (g.Y - 50) ** 2
    u = np.exp(-r2 / (4 * D * t0))
    zero = np.zeros_like(u)
    for _ in range(int(round(t_run / tau))):
        u = implicit_x_half_step(u, 0.0, zero, D, tau, 1.0, 1.0)
        u = implicit_y_half_step(u, 0.0, zero, D, tau, 1.0, 1.0)
    exact = t0 / (t0 + t_run) * np.exp(-r2 / (4 * D * (t0 + t_run)))
    assert np.abs(u - exact).max() < 2e-3


def test_hill_switch():
    """Q2: switch is 0 at U=0, exactly 1/2 at the threshold, near 1 far above it."""
    assert hill(0.0, 7e-8, 2.5) == 0.0
    assert math.isclose(hill(7e-8, 7e-8, 2.5), 0.5)
    assert hill(7e-6, 7e-8, 2.5) > 0.999


def test_epsilon_delays_lactonase():
    """Q2: at the AHL threshold, lactonase is below half-on because its threshold is shifted."""
    U = PAPER.U_th
    assert lactonase_production(U, PAPER) < 0.5 * PAPER.beta_L
    assert math.isclose(lactonase_production(U, PAPER.with_changes(eps=0.0)), 0.5 * PAPER.beta_L)


def test_source_scales_with_density_and_footprint():
    """Q4: N scales HOW MUCH (linear), a bigger radius widens WHERE."""
    g = make_grid(PAPER)
    small = total_footprint(g.X, g.Y, [Colony(50, 50, 5.0)], PAPER)
    big = total_footprint(g.X, g.Y, [Colony(50, 50, 15.0)], PAPER)
    U = np.zeros_like(small)
    F1_half, _ = source_terms(U, small, 0.5, PAPER)
    F1_full, _ = source_terms(U, small, 1.0, PAPER)
    assert np.allclose(2 * F1_half, F1_full)
    assert big.sum() > small.sum()


def test_half_lives():
    """Q5: AHL lingers ~125 h, lactonase ~1.4 h."""
    assert 120 < half_life(PAPER.gamma_U) < 130
    assert 1.3 < half_life(PAPER.gamma_L) < 1.5


def test_logistic_step_matches_closed_form():
    """Paper Eq. 12: stepping 0.5 h at a time equals the closed-form curve."""
    R, p = PAPER.R0, PAPER
    for _ in range(20):
        R = logistic_step(R, p.r, p.I, 0.5)
    t = 10.0
    exact = p.I * p.R0 * math.exp(p.r * t) / (p.I + p.R0 * (math.exp(p.r * t) - 1))
    assert math.isclose(R, exact, rel_tol=1e-12)


def test_density_is_continuous_and_drops_30_percent():
    """Paper Fig. 3: smooth at the dose, ~30% lower ~10 h later."""
    td = PAPER.dose_times[0]
    assert abs(density(td - 1e-9, PAPER) - density(td + 1e-9, PAPER)) < 1e-6
    assert math.isclose(density(td + 20, PAPER) / density(td, PAPER), 0.7, abs_tol=0.01)


def test_antibiotic_pulse():
    """Paper Fig. 5A: peak ~4.0, 4 h after dosing; zero before dosing."""
    assert rayleigh_pulse(-1.0, PAPER.A0, PAPER.A1) == 0.0
    assert math.isclose(rayleigh_pulse(4.0, PAPER.A0, PAPER.A1), 4.0, rel_tol=0.01)
    assert antibiotic_concentration(5.0, PAPER) == 0.0
    assert kill_fraction(1e6, PAPER.theta, PAPER.m) < PAPER.theta


def test_feedback_off_is_paper_mode():
    """Q7: with feedback off, growth is untouched (keeps the positive control valid)."""
    assert effective_growth_rate(0.4, 1.0, PAPER) == 0.4
    on = PAPER.with_changes(feedback_on=True, k_growth=1.0)
    assert effective_growth_rate(0.4, 1.0, on) == 0.8
    assert math.isclose(qs_activation(PAPER.U_th, PAPER), 0.5)


def test_deterministic_control_short():
    """Q6: the strict positive control passes its basic checks (short 2 h run)."""
    res = deterministic_control(PAPER, t_end=2.0)
    checks = deterministic_checks(res)
    for name in ("edges_zero", "no_negative", "mirror_symmetric", "xy_symmetric",
                 "picard_converged"):
        assert checks[name]["pass"], (name, checks[name])


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
