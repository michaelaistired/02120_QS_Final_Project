"""Every number the model uses, in one place.

Units used everywhere in this package:
    length  -> micrometers (um)
    time    -> hours (h)
    concentration -> mol/L
"""

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Params:
    """All model and computational parameters.

    Input:  nothing required; every field has the paper's value (or a documented assumption).
    Output: an immutable Params object passed to every other module.
    Steps:  1) Table 1 chemistry, 2) domain and time grid, 3) colony growth,
            4) density curve N(t), 5) antibiotic, 6) numerics, 7) Q7 feedback switches.
    Types:  all fields are float/int/bool/tuple; use `with_changes` to make a modified copy.
    Role in model: this is the single source of truth for numbers, so the positive control
        (paper values) and any experiment (changed values) differ only by an explicit
        `with_changes(...)` call. Fields marked ASSUMPTION are not given in the paper.
    """

    # ---- Table 1: chemistry of P. putida (paper Table 1) ----
    D_U: float = 100.0            # AHL diffusion rate, um^2/h
    D_L: float = 1.0              # lactonase diffusion rate, um^2/h
    gamma_U: float = 0.005545     # AHL natural decay, 1/h   (half-life ~125 h, see Q5)
    gamma_L: float = 0.5          # lactonase natural decay, 1/h (half-life ~1.4 h, see Q5)
    gamma_LU: float = 0.65e9      # AHL destroyed by lactonase, L/(mol*h)   (Q1 coupling)
    alpha_U: float = 1.058e-7     # low (baseline) AHL production, mol/(L*h)
    beta_U: float = 1.058e-6      # extra AHL production when QS is ON, mol/(L*h)
    beta_L: float = 1.38e-6       # lactonase production when QS is ON, mol/(L*h)
    U_th: float = 7e-8            # AHL threshold of the QS switch, mol/L
    eps: float = 5e-9             # lactonase threshold shift above U_th, mol/L (Q2)
    n: float = 2.5                # Hill exponent (switch sharpness)

    # ---- domain and numerics (paper Sec. 2.2 / 3.1) ----
    length: float = 100.0         # square side l, um
    h: float = 1.0                # grid spacing h1 = h2, um
    tau: float = 0.01             # PDE time step, h
    t_obs: float = 50.0           # observation time, h
    picard_tol: float = 1e-6      # stop the nonlinear fixed-point loop at this relative change
    picard_max: int = 10          # ...or after this many iterations

    # ---- colonies (paper Sec. 2.3 / 3.2) ----
    R0: float = 1.0               # initial colony radius, um
    I: float = 20.0               # limiting colony radius, um
    r: float = 0.4                # logistic growth rate, 1/h
    colony_dt: float = 0.5        # colony (Monte-Carlo) time step, h
    p_initial: tuple = (0.6, 0.3, 0.1)  # P(start with 1, 2, 3 colonies)
    p_new: float = 0.1            # P(new colony appears) per colony step
    sigma_factor: float = 1.0 / 3.0     # sigma = R/3 ("3-sigma rule")

    # ---- density curve N(t) (paper Eqs. 15-16, Sec. 3.1) ----
    b1: float = 6.0               # midpoint of the growth sigmoid, h
    b2_offset: float = 6.0        # midpoint of decline = dose time + 6 h (paper: td=12, b2=18)
    mu: float = 1.4               # steepness of N(t) sigmoids, 1/h
    drop_fraction: float = 0.3    # ASSUMPTION from text: "30% decrease ... within 10 h"

    # ---- antibiotic (paper Eqs. 13-14, Sec. 3.2) ----
    dose_times: tuple = (12.0,)   # when antibiotic is added, h (paper: t_d = 12 h)
    A0: float = 1.65              # Rayleigh pulse amplitude
    A1: float = 32.0              # Rayleigh pulse width, h^2  (peak at sqrt(A1/2) = 4 h)
    theta: float = 0.2            # max fraction of radius lost per colony step
    m: float = 4.0                # ASSUMPTION: calibrated so one dose shrinks total R ~10x (Fig 5B)
    regrow_time: float = math.inf # Fig 10 scenario: colonies regrow after this time (inf = never)
    r_regrow: float = 0.1         # slower regrowth rate after treatment, 1/h (Fig 10)

    # ---- Q7 extension: QS feeds back on behavior (OFF = paper positive control) ----
    feedback_on: bool = False     # False reproduces the paper exactly (QS is only a readout)
    k_growth: float = 1.0         # QS fully ON multiplies growth rate by (1 + k_growth)
    k_dna: float = 1.0            # DNA-uptake events per hour for a fully QS-ON colony (readout)

    def with_changes(self, **changes) -> "Params":
        """Return a copy of these parameters with some fields changed.

        Input:  keyword arguments naming fields, e.g. with_changes(beta_U=2e-6).
        Output: a new Params object (the original is untouched).
        Steps:  1) dataclasses.replace builds the copy.
        Types:  **changes: field_name=value -> Params.
        Role in model: experiments (perturbations, feedback ON, multi-dose) are expressed as
            small, readable differences from the paper's parameters.
        """
        return replace(self, **changes)


PAPER = Params()  # the positive-control parameter set


def half_life(decay_rate: float) -> float:
    """Time for a molecule to fall to half when nothing makes it.

    Input:  decay_rate (1/h), e.g. gamma_U or gamma_L.
    Output: half-life in hours.
    Steps:  1) half-life = ln(2) / rate.
    Types:  float -> float.
    Role in model: this is the answer to Q5. AHL (gamma_U) lasts ~125 h but lactonase
        (gamma_L) lasts ~1.4 h, so when bacteria stop producing, lactonase vanishes
        quickly while AHL lingers. That timescale gap, not resistance, drives the paper's result.
    """
    return math.log(2.0) / decay_rate
