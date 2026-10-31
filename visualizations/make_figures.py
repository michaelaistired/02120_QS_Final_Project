"""Draw every figure of the core logic into visualizations/figures/.

    python -m visualizations.make_figures

One function per figure. Each says which model step / quiz question it illustrates.
Takes about 8 minutes (it runs ~25 simulations).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qs_model.antibiotic import antibiotic_concentration
from qs_model.colonies import Colony
from qs_model.controls import (PAPER_REFERENCE, deterministic_control, perturbation_test,
                               statistical_control)
from qs_model.density import density
from qs_model.grid import make_grid
from qs_model.params import PAPER, half_life
from qs_model.production import ahl_production, lactonase_production, total_footprint
from qs_model.simulation import run_simulation
from run_feedback_demo import STRAINS

FIG = Path(__file__).parent / "figures"
AHL, LAC, GREY = "#2a6fb0", "#c2502b", "#888888"


def save(fig, name: str) -> None:
    """Write a figure to visualizations/figures/<name>.png and close it."""
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=130)
    plt.close(fig)
    print("  wrote", name)


def fig01_hill_switch() -> None:
    """Q2 + production: the QS switch for AHL and the shifted lactonase switch (eps)."""
    U = np.logspace(-10, -6, 400)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].semilogx(U, ahl_production(U, PAPER), color=AHL, label="f1: AHL production")
    ax[0].semilogx(U, lactonase_production(U, PAPER), color=LAC, label="f2: lactonase production")
    ax[0].axvline(PAPER.U_th, ls="--", color=GREY)
    ax[0].text(PAPER.U_th, 1.2e-6, " U_th", color=GREY)
    ax[0].set(xlabel="AHL concentration U (mol/L)", ylabel="production rate (mol/(L h))",
              title="QS switch: positive (AHL) and negative (lactonase) feedback")
    ax[0].legend()
    for eps, ls in ((0.0, ":"), (PAPER.eps, "-"), (1e-7, "--")):
        ax[1].semilogx(U, lactonase_production(U, PAPER.with_changes(eps=eps)), ls=ls,
                       color=LAC, label=f"eps = {eps:.0e}")
    ax[1].axvline(PAPER.U_th, ls="--", color=GREY)
    ax[1].set(xlabel="AHL concentration U (mol/L)", ylabel="lactonase production",
              title="Q2: larger eps = the brake engages later")
    ax[1].legend()
    save(fig, "01_hill_switch_Q2")


def fig02_footprint() -> None:
    """Q4: radius sets WHERE (footprint width), N sets HOW MUCH (amplitude)."""
    g = make_grid(PAPER)
    cols = [Colony(30, 35, 15.0), Colony(65, 60, 8.0), Colony(40, 70, 20.0)]
    fp = total_footprint(g.X, g.Y, cols, PAPER)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    im = ax[0].imshow(fp.T, origin="lower", extent=[0, 100, 0, 100], cmap="Greens")
    for c in cols:
        ax[0].add_patch(plt.Circle((c.x, c.y), c.R, fill=False, color="k", lw=0.8))
    fig.colorbar(im, ax=ax[0])
    ax[0].set(title="Footprint: sum of Gaussians (circle = radius R)", xlabel="x (um)",
              ylabel="y (um)")
    for R in (5, 10, 20):
        one = total_footprint(g.X, g.Y, [Colony(50, 50, R)], PAPER)[:, 50]
        for N, ls in ((1.0, "-"), (0.5, "--")):
            ax[1].plot(g.x, N * one, ls=ls, label=f"R = {R} um, N = {N}")
    ax[1].set(xlabel="x (um) through colony centre", ylabel="N * footprint",
              title="Q4: R widens (where), N scales (how much)")
    ax[1].legend(fontsize=8)
    save(fig, "02_footprint_Q4")


def fig03_adi_thomas() -> None:
    """Q3: the matrix a fully implicit 2D step needs vs the chain ADI needs."""
    m = 8                                                   # tiny 8 x 8 interior for display
    n = m * m
    full = np.zeros((n, n))
    for i in range(m):
        for j in range(m):
            k = i * m + j
            full[k, k] = 4
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= i + di < m and 0 <= j + dj < m:
                    full[k, (i + di) * m + (j + dj)] = -1
    chain = np.diag(np.full(m, 2.0)) + np.diag(np.full(m - 1, -1.0), 1) + np.diag(np.full(m - 1, -1.0), -1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].spy(full, markersize=3)
    ax[0].set_title(f"Fully implicit 2D: one {n}x{n} system\n(each point has 4 neighbours)")
    ax[1].spy(chain, markersize=10)
    ax[1].set_title(f"ADI half-step: {m} separate {m}x{m} chains\n(each point has 2 neighbours -> Thomas)")
    save(fig, "03_adi_vs_full_implicit_Q3")


def fig04_density_antibiotic() -> None:
    """Paper Figs. 3, 5A, 8A/B: prescribed density N(t) and antibiotic A(t)."""
    t = np.linspace(0, 50, 1001)
    triple = PAPER.with_changes(dose_times=(12.0, 24.0, 36.0))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(t, [density(s, PAPER) for s in t], label="single dose (Fig 3)")
    ax[0].plot(t, [density(s, triple) for s in t], "--", label="triple dose (Fig 8B)")
    ax[0].set(xlabel="t (h)", ylabel="N(t)", title="Normalized bacterial density")
    ax[0].legend()
    ax[1].plot(t, [antibiotic_concentration(s, PAPER) for s in t], label="single dose (Fig 5A)")
    ax[1].plot(t, [antibiotic_concentration(s, triple) for s in t], "--", label="triple (Fig 8A)")
    ax[1].set(xlabel="t (h)", ylabel="A(t)", title="Antibiotic (Rayleigh pulse)")
    ax[1].legend()
    save(fig, "04_density_and_antibiotic")


def fig05_to_07_baseline() -> None:
    """Paper Figs. 5B, 6, 7, 9A: one single-dose and one triple-dose run (seed 0)."""
    single = run_simulation(PAPER, seed=0, snapshot_times=(5.0, 12.0, 50.0))
    triple = run_simulation(PAPER.with_changes(dose_times=(12.0, 24.0, 36.0)), seed=0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for res, lab in ((single, "single dose (Fig 5B)"), (triple, "triple dose (Fig 9A)")):
        ax[0].plot(res.series["t"], res.series["R_total"], label=lab)
        ax[1].semilogy(res.series["t"], res.series["R_total"], label=lab)
    ax[0].set(xlabel="t (h)", ylabel="total radius R (um)", title="Colony size")
    ax[1].set(xlabel="t (h)", ylabel="total radius R (um, log)", title="Same, log scale")
    ax[0].legend()
    save(fig, "05_colony_size")

    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    for k, t in enumerate((5.0, 12.0, 50.0)):
        snap = single.snapshots[t]
        for row, (key, cmap, lab) in enumerate((("U", "Blues", "AHL"), ("L", "Oranges", "lactonase"))):
            im = ax[row, k].imshow(snap[key].T, origin="lower", extent=[0, 100, 0, 100], cmap=cmap)
            for c in snap["colonies"]:
                ax[row, k].add_patch(plt.Circle((c.x, c.y), max(c.R, 0.5), fill=False, color="k", lw=0.6))
            fig.colorbar(im, ax=ax[row, k], format="%.0e")
            ax[row, k].set_title(f"{lab} at {t:g} h")
    save(fig, "06_fields_snapshots")

    s = single.series
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].semilogy(s["t"], s["U_max"], color=AHL, label="max AHL")
    ax[0].axhline(PAPER.U_th, ls="--", color=GREY, label="QS threshold U_th")
    ax[0].axhline(PAPER_REFERENCE["U_max_final_single"], ls=":", color=AHL, label="paper final value")
    ax[0].set(xlabel="t (h)", ylabel="mol/L", title="Max AHL (Fig 7A) - never reaches U_th")
    ax[0].legend(fontsize=8)
    ax[1].semilogy(s["t"], s["L_max"], color=LAC, label="max lactonase")
    ax[1].axhline(PAPER_REFERENCE["L_max_final_single"], ls=":", color=LAC, label="paper final value")
    ax[1].axvline(PAPER.dose_times[0], ls="--", color=GREY, label="dose")
    ax[1].set(xlabel="t (h)", ylabel="mol/L", title="Max lactonase (Fig 7B)")
    ax[1].legend(fontsize=8)
    save(fig, "07_max_concentrations")
    return single


def fig08_half_lives(single) -> None:
    """Q5: why lactonase collapses but AHL lingers - decay timescales, not resistance."""
    t = np.linspace(0, 20, 400)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(t, np.exp(-PAPER.gamma_U * t), color=AHL,
               label=f"AHL, half-life {half_life(PAPER.gamma_U):.0f} h")
    ax[0].plot(t, np.exp(-PAPER.gamma_L * t), color=LAC,
               label=f"lactonase, half-life {half_life(PAPER.gamma_L):.1f} h")
    ax[0].set(xlabel="hours after production stops", ylabel="fraction left",
              title="Q5: natural decay alone")
    ax[0].legend()
    s = single.series
    ax[1].semilogy(s["t"], s["U_max"] / s["U_max"].max(), color=AHL, label="AHL / its peak")
    ax[1].semilogy(s["t"], s["L_max"] / s["L_max"].max(), color=LAC, label="lactonase / its peak")
    ax[1].axvline(PAPER.dose_times[0], ls="--", color=GREY, label="dose")
    ax[1].set(xlabel="t (h)", ylabel="relative to peak", title="In the full model (seed 0)")
    ax[1].legend()
    save(fig, "08_half_lives_Q5")


def fig09_positive_control() -> None:
    """Q6 strict control: deterministic single colony + paper Fig. 4 perturbation test."""
    det = deterministic_control(PAPER, t_end=20.0)
    g = det.grid
    pert = perturbation_test(PAPER)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(g.x, det.U[:, 50], color=AHL, label="AHL")
    ax[0].plot(g.x, det.L[:, 50], color=LAC, label="lactonase")
    ax[0].axhline(PAPER.U_th, ls="--", color=GREY, label="U_th")
    ax[0].set(xlabel="x (um)", ylabel="mol/L", title="Deterministic control at 20 h (centre line)")
    ax[0].legend()
    for name, mk in (("alpha_U", "o"), ("beta_U", "s"), ("beta_L", "^")):
        ax[1].plot(pert["fractions"], pert[name]["xi"], marker=mk, label=name)
        ax[2].plot(pert["fractions"], pert[name]["phi"], marker=mk, label=name)
    ax[1].set(xlabel="parameter increase", ylabel="xi (AHL change)", title="Fig 4A analogue")
    ax[2].set(xlabel="parameter increase", ylabel="phi (lactonase change)", title="Fig 4B analogue")
    ax[1].legend()
    ax[2].legend()
    save(fig, "09_positive_control_deterministic_Q6")


def fig10_seed_variability() -> None:
    """Q6 statistical control: same parameters, different random seeds, vs paper numbers."""
    stat = statistical_control(PAPER)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, r in enumerate(stat["results"]):
        ax[0].semilogy(r.series["t"], r.series["U_max"], alpha=0.8, label=f"seed {k}")
        ax[1].plot(r.series["t"], r.series["R_total"], alpha=0.8, label=f"seed {k}")
    ax[0].axhline(PAPER_REFERENCE["U_max_final_single"], ls=":", color="k", label="paper final")
    ax[1].axhline(PAPER_REFERENCE["R_total_final_single"], ls=":", color="k", label="paper final")
    ax[0].set(xlabel="t (h)", ylabel="max AHL (mol/L)", title="Q6: every seed is a different run")
    ax[1].set(xlabel="t (h)", ylabel="total radius (um)", title="Colony size by seed")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    save(fig, "10_seed_variability_Q6")


def fig11_coupling() -> None:
    """Q1: the gamma_LU * L * U term - AHL with and without lactonase destroying it."""
    with_L = deterministic_control(PAPER, t_end=20.0)
    without_L = deterministic_control(PAPER.with_changes(gamma_LU=0.0), t_end=20.0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(with_L.series["t"], with_L.series["U_max"], color=AHL, label="with lactonase")
    ax[0].plot(without_L.series["t"], without_L.series["U_max"], "--", color=AHL,
               label="gamma_LU = 0 (no coupling)")
    ax[0].axhline(PAPER.U_th, ls=":", color=GREY, label="U_th")
    ax[0].set(xlabel="t (h)", ylabel="max AHL (mol/L)", title="Q1: lactonase holds AHL down")
    ax[0].legend()
    g = with_L.grid
    ax[1].plot(g.x, with_L.U[:, 50], color=AHL, label="with lactonase")
    ax[1].plot(g.x, without_L.U[:, 50], "--", color=AHL, label="no coupling")
    ax[1].set(xlabel="x (um)", ylabel="AHL at 20 h (mol/L)", title="Centre line at 20 h")
    ax[1].legend()
    save(fig, "11_coupling_Q1")


def fig12_feedback() -> None:
    """Q7: QS feedback on growth - paper strain vs a QS-sensitive strain, OFF vs ON."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, color in (("paper", GREY), ("sensitive", "#2f8f4e")):
        for mode, ls in (("off", "--"), ("on", "-")):
            r = run_simulation(STRAINS[name].with_changes(feedback_on=(mode == "on")), seed=0)
            s = r.series
            lab = f"{name}, feedback {mode}"
            ax[0].semilogy(s["t"], np.maximum(s["R_total"], 1e-6), ls=ls, color=color, label=lab)
            ax[1].plot(s["t"], s["activation_mean"], ls=ls, color=color, label=lab)
            ax[2].plot(s["t"], s["dna_total"], ls=ls, color=color, label=lab)
    ax[0].set(xlabel="t (h)", ylabel="total radius (um, log)",
              title="Q7: colony size (doses 12/24/36 h, regrow 55 h)")
    ax[1].set(xlabel="t (h)", ylabel="mean QS activation", title="Is QS ON?")
    ax[2].set(xlabel="t (h)", ylabel="DNA-uptake events (readout)", title="Competence readout")
    for a in ax:
        a.legend(fontsize=7)
    save(fig, "12_feedback_Q7")


def main() -> None:
    """Make every figure."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig01_hill_switch()
    fig02_footprint()
    fig03_adi_thomas()
    fig04_density_antibiotic()
    single = fig05_to_07_baseline()
    fig08_half_lives(single)
    fig11_coupling()
    fig09_positive_control()
    fig10_seed_variability()
    fig12_feedback()


if __name__ == "__main__":
    main()
