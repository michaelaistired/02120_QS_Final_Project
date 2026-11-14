"""Comparison figures + a side-by-side video for the Stage 1 runs.

    python -m visualizations.make_figures      (from emergence/, after run_stage1.py; ~2 min)

Reads results/stage1_<run>.npz and videos/stage1_<run>.mp4. Writes to visualizations/figures/:
    s1_timeseries.png        six readouts over time, four runs overlaid
    s1_kymographs.png        radius-vs-time maps of nutrient and cell food (a movie in one image)
    s1_final_snapshots.png   where every cell is at the end, coloured by its food reserve
    s1_compare.mp4           the four movies tiled 2 x 2 and played in sync (panel A + B crops)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from qs_emergence.params import DEFAULT

SIDE = DEFAULT.length
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "figures"
RUNS = {"neither": ("control (neither)", "#555555"), "share": ("(a) sharing", "#1f77b4"),
        "pause": ("(b) pause + relay", "#d62728"), "both": ("(a) + (b)", "#9467bd")}


def load_runs() -> dict:
    """Load every Stage 1 result file that exists.

    Input: none (reads results/stage1_<run>.npz).   Output: {run: dict of arrays}.
    Steps: 1) for each known run, load if present.   Types: () -> dict[str, dict].
    Role: the single data source for every figure here.
    """
    out = {}
    for name in RUNS:
        f = ROOT / "results" / f"stage1_{name}.npz"
        if f.exists():
            out[name] = dict(np.load(f))
    return out


def fig_timeseries(runs: dict) -> None:
    """Six readouts over time for all runs.

    Input: runs.  Output: s1_timeseries.png.
    Steps: one panel per readout; one line per run.
    Role: the quick answer to "which mechanism keeps the centre alive, and at what cost".
    """
    keys = [("n_agents", "cells"), ("n_starving", "starving cells"),
            ("food_centre", "mean food, centre cells"), ("S_centre", "nutrient at centre"),
            ("n_contact", "contact subnetworks (islands)"),
            ("n_transport", "supply-route subnetworks")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, (k, lab) in zip(ax.ravel(), keys):
        for name, h in runs.items():
            a.plot(h["t"], h[k], color=RUNS[name][1], label=RUNS[name][0])
        a.set(xlabel="t (h)", title=lab)
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "s1_timeseries.png", dpi=120)
    plt.close(fig)


def fig_kymographs(runs: dict) -> None:
    """Radius-vs-time maps: nutrient and cell food per 5 um ring.

    Input: runs.  Output: s1_kymographs.png.
    Steps: 1) row per run, 2) left = nutrient S(r, t), right = mean cell food(r, t),
           3) white line = colony radius (95% of cells inside).
    Role: shows the expanding front, the centre emptying, and whether food returns inward,
        all in one picture per run.
    """
    fig, ax = plt.subplots(len(runs), 2, figsize=(12, 3 * len(runs)), squeeze=False)
    for row, (name, h) in enumerate(runs.items()):
        r = h["r_bins"][:-1] + 2.5
        ext = [h["t"][0], h["t"][-1], r[0], r[-1]]
        im0 = ax[row, 0].imshow(h["kymo_S"].T, origin="lower", aspect="auto", extent=ext,
                                cmap="YlGn", vmin=0, vmax=1)
        im1 = ax[row, 1].imshow(h["kymo_food"].T, origin="lower", aspect="auto", extent=ext,
                                cmap="RdYlGn", vmin=0, vmax=1)
        for a in ax[row]:
            a.plot(h["t"], h["radius95"], "w-", lw=1)
            a.set(ylabel=f"{RUNS[name][0]}\nradius (um)")
        fig.colorbar(im0, ax=ax[row, 0], label="nutrient")
        fig.colorbar(im1, ax=ax[row, 1], label="cell food")
    ax[0, 0].set_title("nutrient by distance from nucleation site")
    ax[0, 1].set_title("cell food reserve by distance (white = colony edge)")
    for a in ax[-1]:
        a.set_xlabel("t (h)")
    fig.tight_layout()
    fig.savefig(FIG / "s1_kymographs.png", dpi=110)
    plt.close(fig)


def fig_final(runs: dict) -> None:
    """Final cells over the final nutrient field.

    Input: runs.  Output: s1_final_snapshots.png.
    Steps: per run: nutrient background, cells as dots coloured by food reserve.
    Role: the end state side by side - fragmentation, starving core, recovered centre.
    """
    fig, ax = plt.subplots(1, len(runs), figsize=(5 * len(runs), 5), squeeze=False)
    for a, (name, h) in zip(ax[0], runs.items()):
        extent = [0, SIDE, 0, SIDE]
        a.imshow(h["final_S"].T, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1, alpha=0.5)
        sc = a.scatter(h["final_x"], h["final_y"], s=0.5, c=h["final_food"], cmap="RdYlGn", vmin=0, vmax=1)
        a.set(title=f"{RUNS[name][0]}: {len(h['final_x'])} cells", aspect="equal",
              xlim=(0, SIDE), ylim=(0, SIDE))
    fig.colorbar(sc, ax=ax[0, -1], label="food reserve")
    fig.tight_layout()
    fig.savefig(FIG / "s1_final_snapshots.png", dpi=110)
    plt.close(fig)


def compare_video(names=("neither", "share", "pause", "both")) -> int:
    """Tile the four Stage 1 movies 2 x 2 (panels A and B of each), in sync.

    Input: names of runs whose videos exist.  Output: number of frames written.
    Steps: 1) open each MP4, 2) per frame crop the top-left two panels (cells + food),
           3) downsample by 2, 4) tile 2 x 2 and write.
    Types: tuple[str] -> int.
    Role: watch the four hypotheses side by side at the same moment in time.
    """
    readers = [imageio.get_reader(str(ROOT / "videos" / f"stage1_{n}.mp4")) for n in names]
    writer = imageio.get_writer(str(FIG / "s1_compare.mp4"), fps=24, codec="libx264",
                                quality=7, macro_block_size=1, pixelformat="yuv420p")
    count = 0
    try:
        for frames in zip(*readers):
            tiles = []
            for f in frames:
                H, W = f.shape[:2]
                crop = f[: H // 2, : 2 * W // 3]
                tiles.append(crop[::2, ::2])
            h = min(t.shape[0] for t in tiles)
            w = min(t.shape[1] for t in tiles)
            tiles = [t[:h, :w] for t in tiles]
            grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
            grid = grid[: grid.shape[0] // 2 * 2, : grid.shape[1] // 2 * 2]
            writer.append_data(grid)
            count += 1
    finally:
        writer.close()
        for r in readers:
            r.close()
    return count


def main() -> None:
    """Make every Stage 1 figure and the comparison video."""
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("no results/stage1_*.npz found - run run_stage1.py first")
    fig_timeseries(runs)
    fig_kymographs(runs)
    fig_final(runs)
    print("figures written")
    if all((ROOT / "videos" / f"stage1_{n}.mp4").exists() for n in RUNS):
        print(f"s1_compare.mp4: {compare_video()} frames")


if __name__ == "__main__":
    main()
