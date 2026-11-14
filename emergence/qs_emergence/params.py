"""Every number in the emergence model, in one place.

Units: length um, time h, AHL/lactonase mol/L (as in the paper).
Nutrient S and the "starvation signal" G are NORMALIZED (S = 1 is the supply concentration at
the domain edge), and cell food reserves are in the same units as S x one grid cell, so food
eaten by cells equals nutrient removed from the field exactly.
Values marked ASSUMPTION are exploratory choices (not from the paper) meant to put the model
in a regime where nutrient depletion matters; they are the first things to sweep.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Params:
    """All parameters for a run.

    Input:  nothing required; defaults give the Stage 1 nutrient experiment.
    Output: immutable Params; make variants with with_changes(...).
    Steps:  groups below: chemistry (paper), domain, nutrient, agents, network/sharing,
            starvation signal, QS coupling, video.
    Types:  float/int/bool/str fields.
    Role in model: the single source of truth for numbers. Every experiment (and every
        ON/OFF control) is one with_changes(...) call away from the defaults.
    """

    # ---- paper Table 1 chemistry (AHL = U, lactonase = L) ----
    D_U: float = 100.0
    D_L: float = 1.0
    gamma_U: float = 0.005545
    gamma_L: float = 0.5
    gamma_LU: float = 0.65e9
    alpha_U: float = 1.058e-7
    beta_U: float = 1.058e-6
    beta_L: float = 1.38e-6
    U_th: float = 7e-8
    eps: float = 5e-9
    n: float = 2.5

    # ---- domain and numerics ----
    length: float = 480.0          # square side, um (bigger than the paper's 100 um: colonies spread)
    h: float = 4.0                 # grid spacing, um (also the cell-to-cell contact distance)
    tau: float = 0.01              # PDE time step, h (paper value)
    t_obs: float = 36.0            # observation time, h (before the colony reaches the edge)
    picard_tol: float = 1e-6
    picard_max: int = 10
    seed: int = 0

    # ---- nutrient S (normalized) ----
    S_edge: float = 1.0            # supply concentration held at the domain edge
    S_init: float = 1.0            # starting concentration everywhere
    D_S: float = 2000.0            # ASSUMPTION: effective nutrient diffusion in matrix, um^2/h
    V_max: float = 0.6             # ASSUMPTION: max uptake per agent, food units/h
    K_S: float = 0.2               # ASSUMPTION: half-saturation of uptake (Monod)

    # ---- agents (each agent = a cluster of ~10^3 bacteria) ----
    n_seed: int = 40               # agents placed at the nucleation site
    seed_radius: float = 12.0      # radius of the nucleation site, um
    food_init: float = 0.5         # starting food reserve per agent
    maintenance: float = 0.08      # food burned per agent per hour just to stay alive
    food_divide: float = 1.0       # reserve needed to divide (each daughter gets half)
    food_starve: float = 0.15      # below this an agent counts as starving
    max_agents: int = 80000        # safety cap (should never be reached; checked in tests/runs)
    agent_dt: float = 0.05         # how often agents act, h (every 5 PDE steps)
    motility_hungry: float = 40.0  # random-walk diffusivity when starving (foraging), um^2/h
    motility_fed: float = 2.0      # random-walk diffusivity when fed, um^2/h
    chemotaxis: float = 3000.0     # drift up the nutrient gradient (hungry agents), um^2/h
    crowd_max: float = 4.0         # agents per grid cell before crowding pushes them apart
    crowd_push: float = 60.0       # strength of the crowding push, um^2/h per agent

    # ---- food route hypothesis (a): sharing along the contact network ----
    share_on: bool = False
    share_rate: float = 20.0       # ASSUMPTION: food exchange rate between linked cells, 1/h
                               # (2/h only spreads food ~30 um^2/h - too slow to reach the centre)

    # ---- food route hypothesis (b): starvation signal pauses edge feeding ----
    pause_on: bool = False
    D_G: float = 500.0             # ASSUMPTION: starvation-signal diffusion, um^2/h
    gamma_G: float = 1.0           # starvation-signal decay, 1/h
    G_emit: float = 0.5            # signal made per starving agent, units/h
    G_relay: float = 0.4           # ASSUMPTION: fed cells that sense the signal re-emit it
                                   # (relay, as for potassium in Prindle et al. 2015); 0 = passive
    G_th: float = 0.2              # signal level at which fed cells half-pause feeding
    pause_strength: float = 0.9    # max fraction of feeding paused

    # ---- QS coupling of agents to the paper's chemistry ----
    rho_ref: float = 4.0           # agents per cell that count as "fully covered" (footprint = 1)
    qs_motility_on: bool = False   # optional: QS-ON agents forage more (swarming-like)
    qs_motility_boost: float = 2.0

    # ---- video ----
    frame_stride: int = 10         # render every Nth PDE step (10 -> every 0.1 h)
    fps: int = 24

    def with_changes(self, **changes) -> "Params":
        """Copy with some fields changed.

        Input:  field=value keywords.  Output: new Params.
        Steps:  dataclasses.replace.   Types: **kwargs -> Params.
        Role in model: makes every ON/OFF control explicit and readable.
        """
        return replace(self, **changes)


DEFAULT = Params()

# The four Stage 1 runs: each food-route hypothesis alone, both, and neither (the control).
STAGE1_RUNS = {
    "neither": DEFAULT,
    "share": DEFAULT.with_changes(share_on=True),
    "pause": DEFAULT.with_changes(pause_on=True),
    "both": DEFAULT.with_changes(share_on=True, pause_on=True),
}
