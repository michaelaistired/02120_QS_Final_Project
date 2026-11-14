"""Unit tests for the agent model. Run from the emergence/ folder:  python -m tests.test_emergence

Each test checks one building block against an answer known independently.
"""

import numpy as np

from qs_emergence.agents import Agents, burn_and_die, divide, seed_nucleation
from qs_emergence.engine import agent_cell_info, eat, initial_nutrient
from qs_emergence.fields import hill, pause_factor
from qs_emergence.grid import make_grid
from qs_emergence.network import components, contact_subnetworks, share_food
from qs_emergence.params import DEFAULT
from qs_emergence.solver import diffuse_step, implicit_x_half_step, thomas_solve


def test_thomas_matches_dense():
    """Thomas == general solver on random diagonally dominant chains."""
    rng = np.random.default_rng(0)
    m, k = 25, 4
    lo, up = -rng.uniform(0.1, 1, (m, k)), -rng.uniform(0.1, 1, (m, k))
    dg = 3 + rng.uniform(0, 1, (m, k))
    rhs = rng.normal(size=(m, k))
    x = thomas_solve(lo, dg, up, rhs)
    for c in range(k):
        A = np.diag(dg[:, c]) + np.diag(lo[1:, c], -1) + np.diag(up[:-1, c], 1)
        assert np.allclose(A @ x[:, c], rhs[:, c])


def test_zero_edges_match_original_solver():
    """With zero edges the generalized half step equals qs_model's exactly."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from qs_model.solver import implicit_x_half_step as original
    rng = np.random.default_rng(1)
    u = rng.uniform(size=(30, 30))
    u[0], u[-1], u[:, 0], u[:, -1] = 0, 0, 0, 0
    q, F = rng.uniform(size=(30, 30)), rng.uniform(size=(30, 30))
    assert np.array_equal(implicit_x_half_step(u, q, F, 50.0, 0.01, 1.0, 1.0),
                          original(u, q, F, 50.0, 0.01, 1.0, 1.0))


def test_fixed_edge_fills_domain():
    """Nutrient with a fixed edge value and no sink relaxes to that value everywhere."""
    p = DEFAULT.with_changes(length=40.0, h=4.0)
    g = make_grid(p)
    S = np.zeros((g.n, g.n))
    S[0], S[-1], S[:, 0], S[:, -1] = 1, 1, 1, 1
    for _ in range(400):
        S = diffuse_step(S, 0.0, np.zeros_like(S), 200.0, 0.05, p.h)
    assert np.allclose(S, 1.0, atol=1e-3)


def test_eating_conserves_food():
    """Nutrient removed from the field == food gained by agents."""
    p = DEFAULT
    g = make_grid(p)
    ag = seed_nucleation(np.random.default_rng(2), p)
    S = initial_nutrient(g, p)
    info = agent_cell_info(ag, np.zeros_like(S), g, p)
    before = S.sum() + ag.food.sum()
    S = eat(S, ag, info, p)
    assert np.isclose(S.sum() + ag.food.sum(), before, rtol=1e-12)


def test_sharing_conserves_and_stays_positive():
    """Food sharing moves food but never creates, destroys or overdraws it."""
    p = DEFAULT.with_changes(share_on=True)
    g = make_grid(p)
    rng = np.random.default_rng(3)
    ag = Agents(200 + rng.normal(0, 15, 500), 200 + rng.normal(0, 15, 500),
                rng.uniform(0, 1, 500), np.zeros(500), np.zeros(500))
    total = ag.food.sum()
    Jx, Jy = share_food(ag, g, p, p.agent_dt)
    assert np.isclose(ag.food.sum(), total, rtol=1e-12)
    assert ag.food.min() >= 0
    assert np.abs(Jx).sum() + np.abs(Jy).sum() > 0


def test_division_and_death_conserve_food():
    """Division splits food exactly; death removes only empty agents."""
    p = DEFAULT
    ag = Agents(np.full(4, 100.0), np.full(4, 100.0), np.array([2.0, 0.5, 1.2, 0.01]),
                np.zeros(4), np.zeros(4))
    total = ag.food.sum()
    ag, born = divide(ag, np.random.default_rng(0), 0.0, p)
    assert born == 2 and np.isclose(ag.food.sum(), total)
    ag, died = burn_and_die(ag, 1.0, p)
    assert died == 1 and ag.food.min() > 0


def test_pause_off_is_identity():
    """With pause_on False every agent feeds at full rate."""
    G = np.array([0.0, 5.0, 50.0])
    assert np.array_equal(pause_factor(G, np.zeros(3, bool), DEFAULT), np.ones(3))
    on = DEFAULT.with_changes(pause_on=True)
    f = pause_factor(G, np.array([False, False, True]), on)
    assert f[0] > f[1] and f[2] == 1.0 and np.isclose(hill(on.G_th, on.G_th, 2), 0.5)


def test_components():
    """Two separate patches -> two subnetworks."""
    rho = np.zeros((10, 10))
    rho[1:3, 1:3] = 1
    rho[6:9, 6] = 2
    lab = contact_subnetworks(rho)
    assert lab.max() == 2 and len(np.unique(lab[rho > 0])) == 2
    assert components(5, np.array([0, 1]), np.array([1, 2]), np.array([0, 1, 2, 4])).max() == 2


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
