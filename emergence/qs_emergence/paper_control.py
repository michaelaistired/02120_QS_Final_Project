"""Stage 0 positive control: the paper's own model, animated step by step.

This is figure 06 of the original project (AHL and lactonase maps) turned into a video.
The PDE steps use this package's solver (solver.qs_step); the colony, density N(t) and
antibiotic rules are taken unchanged from qs_model (the reference implementation). The check:
with the same random seed, our final AHL and lactonase fields must equal qs_model's to
round-off. If they do, the new solver is a faithful copy of the paper's chemistry.
"""

import copy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, to import qs_model
from qs_model.antibiotic import antibiotic_concentration  # noqa: E402
from qs_model.colonies import seed_colonies, total_radius, update_colonies  # noqa: E402
from qs_model.density import density  # noqa: E402
from qs_model.grid import make_grid  # noqa: E402
from qs_model.params import PAPER  # noqa: E402
from qs_model.production import source_terms, total_footprint  # noqa: E402
from qs_model.simulation import run_simulation  # noqa: E402

from .solver import qs_step  # noqa: E402
from .video import Mp4Writer  # noqa: E402


def run_paper(p=PAPER, seed: int = 0, stride: int = 10, on_frame=None,
              keep_times=(5.0, 12.0, 50.0)) -> dict:
    """The paper's simulation loop, PDE step by PDE step, with a frame callback.

    Input:  p (qs_model Params, default PAPER), seed, stride (call on_frame every stride PDE
            steps), on_frame (function(t, U, L, colonies, series) or None), keep_times (h).
    Output: {"U", "L" (final fields), "kept" {t: (U, L)}, "series" {t, U_max, L_max, R, A}}.
    Steps:  1) same set-up as qs_model.run_simulation (grid, seeded colonies, zero fields),
            2) each colony step: footprint; 50 x qs_step with sources = N(t) f(U) footprint
               (N at t for the x half step, at t + tau for the y half step - same as qs_model),
            3) every `stride` PDE steps: record and call on_frame,
            4) update_colonies exactly as qs_model does.
    Types:  (Params, int, int, Callable|None, tuple) -> dict.
    Role in model: lets us watch the paper's model at every time step and proves our solver
        reproduces the reference implementation.
    """
    grid = make_grid(p)
    rng = np.random.default_rng(seed)
    U, L = np.zeros((grid.n, grid.n)), np.zeros((grid.n, grid.n))
    cols = seed_colonies(rng, p)
    n_sub = int(round(p.colony_dt / p.tau))
    n_col = int(round(p.t_obs / p.colony_dt))
    series = {k: [] for k in ("t", "U_max", "L_max", "R", "A")}
    kept, step = {}, 0
    for c in range(n_col):
        t0 = c * p.colony_dt
        fp = total_footprint(grid.X, grid.Y, cols, p)
        for s in range(n_sub):
            ts = t0 + s * p.tau
            N = (density(ts, p), density(ts + p.tau, p))
            U, L, _ = qs_step(U, L, lambda u, k: source_terms(u, fp, N[k], p), p)
            step += 1
            if step % stride == 0:
                t = step * p.tau
                for k, v in (("t", t), ("U_max", U.max()), ("L_max", L.max()),
                             ("R", total_radius(cols)), ("A", antibiotic_concentration(t, p))):
                    series[k].append(v)
                if on_frame is not None:
                    on_frame(t, U, L, cols, series)
        cols = update_colonies(cols, t0, rng, p, U, grid)
        t_end = t0 + p.colony_dt
        for kt in keep_times:
            if abs(kt - t_end) < 1e-9:
                kept[kt] = (U.copy(), L.copy())
    return {"U": U, "L": L, "kept": kept, "series": series}


def compare_with_reference(seed: int = 0, p=PAPER, keep_times=(5.0, 12.0, 50.0)) -> dict:
    """Positive control: our step-by-step run vs qs_model.run_simulation, same seed.

    Input:  seed, p, keep_times (the figure 06 snapshot times).
    Output: {t: {"U": rel. L2 difference, "L": rel. L2 difference}} for each snapshot time.
    Steps:  1) run_paper (no video), 2) qs_model.run_simulation with snapshots,
            3) relative L2 difference of each field at each time.
    Types:  (int, Params, tuple) -> dict.
    Role in model: if every difference is ~1e-12 the new solver IS the paper's solver, so
        anything new seen in Stage 1 comes from the new biology, not from a changed numerics.
    """
    ours = run_paper(p, seed=seed, stride=10 ** 9, keep_times=keep_times)
    ref = run_simulation(p, seed=seed, snapshot_times=keep_times)
    out = {}
    for t in keep_times:
        U, L = ours["kept"][t]
        rU, rL = ref.snapshots[t]["U"], ref.snapshots[t]["L"]
        out[t] = {"U": float(np.linalg.norm(U - rU) / np.linalg.norm(rU)),
                  "L": float(np.linalg.norm(L - rL) / np.linalg.norm(rL))}
    return out


def render_paper_video(out_path, seed: int = 0, stride: int = 10, fps: int = 24, p=PAPER) -> int:
    """Write the Stage 0 video: AHL map, lactonase map, and time series, every `stride` steps.

    Input:  out_path (.mp4), seed, stride (10 -> one frame per 0.1 h), fps, p.
    Output: number of frames written.
    Steps:  1) build a 3-panel figure (figure 06 colours), 2) run_paper with an on_frame that
            updates the images, colony circles and curves, then appends the frame, 3) close.
    Types:  (str|Path, int, int, int, Params) -> int.
    Role in model: Stage 0 - the paper's figure 06 as a movie.
    """
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), dpi=90)
    ext = [0, p.length, 0, p.length]
    imU = ax[0].imshow(np.zeros((2, 2)), origin="lower", extent=ext, cmap="Blues")
    imL = ax[1].imshow(np.zeros((2, 2)), origin="lower", extent=ext, cmap="Oranges")
    cbU, cbL = fig.colorbar(imU, ax=ax[0], format="%.0e"), fig.colorbar(imL, ax=ax[1], format="%.0e")
    ax[0].set_title("AHL (mol/L)")
    ax[1].set_title("lactonase (mol/L)")
    lu, = ax[2].semilogy([], [], color="#2a6fb0", label="max AHL")
    ll, = ax[2].semilogy([], [], color="#c2502b", label="max lactonase")
    ax[2].axhline(p.U_th, ls="--", color="grey", label="QS threshold")
    ax[2].set(xlim=(0, p.t_obs), ylim=(1e-16, 1e-6), xlabel="t (h)", title="maxima over time")
    ax[2].legend(loc="lower right", fontsize=8)
    for d in p.dose_times:
        ax[2].axvline(d, color="k", lw=0.8, ls=":")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    writer = Mp4Writer(out_path, fps)
    circles = []

    def on_frame(t, U, L, cols, series):
        imU.set_data(U.T)
        imU.set_clim(0, max(U.max(), 1e-20))
        imL.set_data(L.T)
        imL.set_clim(0, max(L.max(), 1e-20))
        for c in circles:
            c.remove()
        circles.clear()
        for col in cols:
            for a in ax[:2]:
                circles.append(a.add_patch(plt.Circle((col.x, col.y), max(col.R, 0.5), fill=False, lw=0.6)))
        lu.set_data(series["t"], series["U_max"])
        ll.set_data(series["t"], np.maximum(series["L_max"], 1e-30))
        fig.suptitle(f"Stage 0 - paper model (Kuttler & Maslovskaya 2022), seed {seed}    "
                     f"t = {t:5.1f} h    antibiotic A = {series['A'][-1]:.2f}    "
                     f"total colony radius = {series['R'][-1]:.2f} um", fontsize=12)
        writer.add(fig)

    try:
        run_paper(p, seed=seed, stride=stride, on_frame=on_frame)
    finally:
        writer.close()
        plt.close(fig)
    return writer.frames
