"""Stage 1 - nutrients: does food reach the starving centre, and by which route?

    python run_stage1.py            (from the emergence/ folder; ~15 min for all four runs)
    python run_stage1.py --runs share both

Four runs, identical except for two switches (same seed, so the same founding cells):
    neither  - control: no sharing, no pause
    share    - hypothesis (a): food passed cell-to-cell along the contact network
    pause    - hypothesis (b): starvation signal (relayed) makes fed cells eat less
    both     - (a) + (b)

Writes:
    videos/stage1_<run>.mp4             six-panel movie, a frame every 0.1 h
    results/stage1_<run>.npz            time series + radial profiles (kymographs)
    results/stage1_summary.md           table comparing the runs
Then run  python -m visualizations.make_figures  for the comparison figures.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from qs_emergence.params import STAGE1_RUNS
from qs_emergence.video import render_agent_run

HERE = Path(__file__).parent
TITLES = {"neither": "Stage 1 CONTROL - no sharing, no pause",
          "share": "Stage 1 (a) - food shared along the cell network",
          "pause": "Stage 1 (b) - starvation signal pauses fed cells",
          "both": "Stage 1 (a)+(b) - sharing and pause"}
R_BINS = np.arange(0, 245, 5.0)  # radial bins, um from the nucleation site


class Kymograph:
    """Collect radial profiles every frame: how nutrient and cell food vary with distance from
    the nucleation site over time.

    Input:  p (Params).  Output: arrays .S, .food, .cells, each (frames, radial bins).
    Steps:  each frame: 1) distance of every grid point / agent from the centre, 2) average
            nutrient and agent food per 5 um ring, 3) count agents per ring.
    Types:  Params -> Kymograph; __call__(State) -> None.
    Role in model: a whole movie squeezed into one picture (radius vs time). Expansion fronts,
        centre starvation and stop-and-go growth show up as stripes and slopes.
    """

    def __init__(self, p):
        self.c = p.length / 2.0
        self.S, self.food, self.cells = [], [], []

    def __call__(self, s):
        d_grid = np.hypot(s.grid.X - self.c, s.grid.Y - self.c).ravel()
        k = np.digitize(d_grid, R_BINS) - 1
        nb = len(R_BINS) - 1
        ok = k < nb
        S = np.bincount(k[ok], s.S.ravel()[ok], nb) / np.maximum(np.bincount(k[ok], minlength=nb), 1)
        d = np.hypot(s.agents.x - self.c, s.agents.y - self.c)
        ka = np.clip(np.digitize(d, R_BINS) - 1, 0, nb - 1)
        cnt = np.bincount(ka, minlength=nb).astype(float)
        food = np.divide(np.bincount(ka, s.agents.food, nb), cnt, out=np.full(nb, np.nan), where=cnt > 0)
        self.S.append(S)
        self.food.append(food)
        self.cells.append(cnt)


def summary_rows(name: str, h: dict) -> list:
    """Markdown rows for one run at 12, 24 and 36 h."""
    rows = []
    for t in (12, 24, 36):
        k = int(np.argmin(np.abs(h["t"] - t)))
        rows.append(f"| {name} | {t} | {h['n_agents'][k]} | {h['n_starving'][k]} | "
                    f"{h['radius95'][k]:.0f} | {h['food_centre'][k]:.3f} | {h['food_edge'][k]:.3f} | "
                    f"{h['S_centre'][k]:.3f} | {h['n_contact'][k]} | {h['n_transport'][k]} |")
    return rows


def main() -> None:
    """Render each requested run, save its data, then write the summary table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(STAGE1_RUNS))
    args = ap.parse_args()
    (HERE / "videos").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)

    for name in args.runs:
        p = STAGE1_RUNS[name]
        kymo = Kymograph(p)
        t0 = time.time()
        h = render_agent_run(p, HERE / "videos" / f"stage1_{name}.mp4", TITLES[name],
                             verbose=True, extra=kymo)
        fin = h.pop("final")
        np.savez_compressed(HERE / "results" / f"stage1_{name}.npz",
                            **{k: v for k, v in h.items() if isinstance(v, np.ndarray)},
                            r_bins=R_BINS, kymo_S=np.array(kymo.S), kymo_food=np.array(kymo.food),
                            kymo_cells=np.array(kymo.cells), final_x=fin.agents.x,
                            final_y=fin.agents.y, final_food=fin.agents.food, final_S=fin.S,
                            final_U=fin.U, final_G=fin.G, final_Jx=fin.Jx, final_Jy=fin.Jy)
        print(f"{name}: {h['frames']} frames in {time.time() - t0:.0f} s, "
              f"max cells {h['n_agents'].max()} (cap {p.max_agents})", flush=True)

    rows = []
    for name in STAGE1_RUNS:
        f = HERE / "results" / f"stage1_{name}.npz"
        if f.exists():
            rows += summary_rows(name, dict(np.load(f)))
    lines = ["# Stage 1 summary (seed 0)", "",
             "centre = cells within 40 um of the nucleation site; edge = outermost 20% of cells.", "",
             "| run | t (h) | cells | starving | R95 (um) | centre food | edge food | nutrient at centre | "
             "contact subnetworks | supply-route subnetworks |",
             "|---|---|---|---|---|---|---|---|---|---|"] + rows
    (HERE / "results" / "stage1_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
