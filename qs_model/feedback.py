"""Q7 extension: let quorum sensing change what the bacteria DO.

In the paper the arrow only points one way: bacteria -> signal. QS is a readout.
Here we add the arrow back: signal -> bacteria, through the same Hill switch the cells
already use to decide to produce more AHL:

    local AHL at colony centre --hill--> activation a in [0, 1]
        a -> growth rate   r_eff = r * (1 + k_growth * a)        (only if p.feedback_on)
        a -> DNA uptake    dDNA/dt = k_dna * a                   (readout only, always on)

The growth arrow closes a loop that the paper does not have:
    more AHL -> faster growth -> bigger radius -> wider footprint -> more AHL -> ...
That loop is the first step toward emergent, collective behavior.

With p.feedback_on = False (the default) the growth rate is untouched, so the model is
exactly the paper's - that is what keeps the positive control valid.
"""

from .grid import Grid, value_at
from .params import Params
from .production import hill


def qs_activation(U_local: float, p: Params) -> float:
    """How "switched on" a colony's quorum sensing is.

    Input:  U_local (AHL at the colony centre, mol/L), p (Params) - uses U_th, n.
    Output: activation in [0, 1] (0 = OFF, 1 = fully ON, 0.5 exactly at the threshold).
    Steps:  1) the same Hill function used for AHL production, at threshold U_th.
    Types:  (float, Params) -> float.
    Role in model: reuses the cells' own QS switch as the signal that changes behavior,
        so we add no new biology beyond "the switch that controls production also
        controls growth/uptake".
    """
    return float(hill(U_local, p.U_th, p.n))


def sense_colonies(colonies: list, U, grid: Grid, p: Params) -> list:
    """Each colony reads the AHL at its centre and stores its activation.

    Input:  colonies (list[Colony]), U ((n, n) AHL field), grid (Grid), p (Params).
    Output: list of activations (floats), one per colony; also stored as colony.activation.
    Steps:  1) look up U at each colony centre, 2) convert to activation with qs_activation.
    Types:  (list[Colony], ndarray, Grid, Params) -> list[float].
    Role in model: the "sensing" step. It runs in every mode so we can always report
        how many colonies are quorate, but it only changes growth when feedback_on is True.
    """
    activations = []
    for c in colonies:
        c.activation = qs_activation(value_at(U, grid, c.x, c.y), p)
        activations.append(c.activation)
    return activations


def effective_growth_rate(r_base: float, activation: float, p: Params) -> float:
    """Growth rate after QS feedback.

    Input:  r_base (1/h), activation (0-1), p (Params) - uses feedback_on, k_growth.
    Output: growth rate (1/h).
    Steps:  1) if feedback is off, return r_base unchanged (paper mode),
            2) else r_base * (1 + k_growth * activation).
    Types:  (float, float, Params) -> float.
    Role in model: the core of the Q7 answer. A quorate colony grows up to (1 + k_growth)
        times faster, so signal now changes the population that makes the signal.
    """
    if not p.feedback_on:
        return r_base
    return r_base * (1.0 + p.k_growth * activation)


def dna_uptake_increment(activation: float, dt: float, p: Params) -> float:
    """DNA-uptake events a colony gains over dt.

    Input:  activation (0-1), dt (h), p (Params) - uses k_dna.
    Output: events (float, arbitrary units).
    Steps:  1) k_dna * activation * dt.
    Types:  (float, float, Params) -> float.
    Role in model: a placeholder for "collective uptake of environmental DNA" from the
        proposal. It is only a READOUT for now (it changes nothing else), so it is a safe
        hook for later work on competence circuits.
    """
    return p.k_dna * activation * dt


def apply_dna_uptake(colonies: list, dt: float, p: Params) -> float:
    """Add this step's DNA uptake to every colony.

    Input:  colonies (list[Colony]), dt (h), p (Params).
    Output: total uptake this step (float).
    Steps:  1) per colony: dna += dna_uptake_increment(activation), 2) sum.
    Types:  (list[Colony], float, Params) -> float.
    Role in model: accumulates the DNA-uptake readout over the run.
    """
    total = 0.0
    for c in colonies:
        inc = dna_uptake_increment(c.activation, dt, p)
        c.dna += inc
        total += inc
    return total
