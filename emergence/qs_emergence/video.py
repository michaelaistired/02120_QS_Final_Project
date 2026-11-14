"""Render frames while the simulation runs and write them to MP4 (imageio + ffmpeg).

    Mp4Writer           collects matplotlib figures as video frames
    AgentFrameRenderer  six-panel frame for the agent model (cells, food, nutrient, AHL, signal, plots)
    render_agent_run    run_emergence + AgentFrameRenderer + Mp4Writer in one call
"""

import matplotlib
matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from .engine import run_emergence


class Mp4Writer:
    """Append matplotlib figures to an MP4 file.

    Input:  path (str/Path), fps (int).
    Output: an .mp4 file once close() is called.
    Steps:  1) open an ffmpeg writer, 2) add(fig): draw the figure, grab its pixels, append,
            3) close().
    Types:  (str, int) -> Mp4Writer; add(Figure) -> None.
    Role in model: "render while simulating" - each frame is written and forgotten, so a run
        with thousands of frames never needs gigabytes of stored fields.
    """

    def __init__(self, path, fps: int):
        self.writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7,
                                         macro_block_size=1, pixelformat="yuv420p")
        self.frames = 0

    def add(self, fig) -> None:
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        self.writer.append_data(rgba[:, :, :3])
        self.frames += 1

    def close(self) -> None:
        self.writer.close()


def link_segments(mask_x: np.ndarray, mask_y: np.ndarray, Jx: np.ndarray, Jy: np.ndarray,
                  h: float) -> tuple:
    """Line segments (and widths) for the links flagged as supply routes.

    Input:  mask_x ((n-1, n) bool), mask_y ((n, n-1) bool), Jx, Jy (food flows), h (spacing, um).
    Output: (segments (K, 2, 2) array of [[x0, y0], [x1, y1]] in um, widths (K,)).
    Steps:  1) x-links join cell (a, b) to (a+1, b); y-links join (a, b) to (a, b+1),
            2) line width proportional to |flow| relative to the largest.
    Types:  (ndarray, ndarray, ndarray, ndarray, float) -> tuple[ndarray, ndarray].
    Role in model: draws the food supply routes the network carries.
    """
    ax, bx = np.nonzero(mask_x)
    ay, by = np.nonzero(mask_y)
    seg = np.concatenate([
        np.stack([np.stack([ax * h, bx * h], 1), np.stack([(ax + 1) * h, bx * h], 1)], 1),
        np.stack([np.stack([ay * h, by * h], 1), np.stack([ay * h, (by + 1) * h], 1)], 1)])
    flow = np.concatenate([np.abs(Jx[ax, bx]), np.abs(Jy[ay, by])])
    widths = 0.3 + 2.5 * flow / flow.max() if flow.size else flow
    return seg, widths


class AgentFrameRenderer:
    """Six-panel video frame for the agent model.

    Panels: (A) cells coloured by supply-route subnetwork, with the routes drawn as lines,
            (B) each cell's food reserve (red = starving), (C) nutrient field S,
            (D) AHL with the QS threshold contour, (E) starvation signal G,
            (F) time series: cells, starving cells, centre vs edge food.
    Role in model: what you watch - colony expanding from the nucleation site, the centre
        emptying of nutrient, food routes forming, and the signal wave (if pause_on).
    """

    def __init__(self, p, title: str):
        self.p = p
        self.title = title
        self.fig, ax = plt.subplots(2, 3, figsize=(15, 9), dpi=90)
        self.ax = ax.ravel()
        ext = [0, p.length, 0, p.length]
        a = self.ax
        self.cells = a[0].scatter([], [], s=2, c=[], cmap="tab20", vmin=0, vmax=20)
        self.routes = LineCollection([], colors="k", alpha=0.6)
        a[0].add_collection(self.routes)
        self.food = a[1].scatter([], [], s=2, c=[], cmap="RdYlGn", vmin=0, vmax=p.food_divide)
        self.fig.colorbar(self.food, ax=a[1], label="food reserve")
        self.S = a[2].imshow(np.zeros((2, 2)), origin="lower", extent=ext, cmap="YlGn", vmin=0, vmax=p.S_edge)
        self.fig.colorbar(self.S, ax=a[2], label="nutrient S")
        self.U = a[3].imshow(np.full((2, 2), 1e-12), origin="lower", extent=ext, cmap="Blues",
                             norm=LogNorm(1e-11, 1e-7))
        self.fig.colorbar(self.U, ax=a[3], label="AHL (mol/L)")
        self.G = a[4].imshow(np.zeros((2, 2)), origin="lower", extent=ext, cmap="Purples", vmin=0, vmax=2.0)
        self.fig.colorbar(self.G, ax=a[4], label="starvation signal G")
        self.u_contour = None
        for k, t in enumerate(["A  cells + food supply routes", "B  food reserve per cell",
                               "C  nutrient", f"D  AHL (contour = QS threshold)",
                               "E  starvation signal" + ("" if p.pause_on else " (mechanism OFF)")]):
            a[k].set(xlim=(0, p.length), ylim=(0, p.length), title=t, aspect="equal")
        self.l_cells, = a[5].plot([], [], "k-", label="cells")
        self.l_starv, = a[5].plot([], [], "r-", label="starving cells")
        a[5].set(xlim=(0, p.t_obs), xlabel="t (h)", ylabel="cells", title="F  colony over time")
        self.tw = a[5].twinx()
        self.l_fc, = self.tw.plot([], [], "g--", label="centre food")
        self.l_fe, = self.tw.plot([], [], "b--", label="edge food")
        self.tw.set(ylim=(0, 1.5 * p.food_divide), ylabel="mean food reserve")
        a[5].legend(loc="upper left", fontsize=8)
        self.tw.legend(loc="upper right", fontsize=8)
        self.fig.tight_layout(rect=(0, 0, 1, 0.95))

    def draw(self, s) -> None:
        """Update every panel from a State (engine.State)."""
        ag, p = s.agents, self.p
        xy = np.column_stack([ag.x, ag.y])
        i = np.clip(np.rint(ag.x / s.grid.h).astype(int), 1, s.grid.n - 2)
        j = np.clip(np.rint(ag.y / s.grid.h).astype(int), 1, s.grid.n - 2)
        lab = s.transport[i, j] if s.transport is not None and s.transport.max() > 0 else s.contact[i, j]
        self.cells.set_offsets(xy)
        self.cells.set_array(np.where(lab > 0, lab % 20, 0).astype(float))
        if p.share_on and s.strong_x is not None:
            seg, w = link_segments(s.strong_x, s.strong_y, s.Jx, s.Jy, s.grid.h)
            self.routes.set_segments(seg)
            self.routes.set_linewidths(w)
        self.food.set_offsets(xy)
        self.food.set_array(ag.food)
        self.S.set_data(s.S.T)
        self.U.set_data(np.maximum(s.U, 1e-12).T)
        if self.u_contour is not None:
            self.u_contour.remove()
        self.u_contour = self.ax[3].contour(s.grid.x, s.grid.y, s.U.T, levels=[p.U_th], colors="r") \
            if s.U.max() > p.U_th else None
        self.G.set_data(s.G.T)
        h = s.history
        self.l_cells.set_data(h["t"], h["n_agents"])
        self.l_starv.set_data(h["t"], h["n_starving"])
        self.l_fc.set_data(h["t"], h["food_centre"])
        self.l_fe.set_data(h["t"], h["food_edge"])
        self.ax[5].set_ylim(0, max(10, 1.1 * max(h["n_agents"])))
        self.fig.suptitle(f"{self.title}    t = {s.t:5.1f} h    cells = {ag.n}    "
                          f"subnetworks: contact {int(s.contact.max())}, supply routes "
                          f"{int(s.transport.max())}", fontsize=13)


def render_agent_run(p, out_path, title: str, verbose: bool = False, extra=None) -> dict:
    """Run the agent model and write its video.

    Input:  p (Params), out_path (.mp4 path), title (shown on every frame), verbose,
            extra (optional function State -> None, e.g. to collect kymograph data).
    Output: the run's history (dict from run_emergence).
    Steps:  1) build renderer and writer, 2) run_emergence with on_frame = draw + add,
            3) close the video.
    Types:  (Params, str|Path, str, bool, Callable|None) -> dict.
    Role in model: one call = one experiment = one movie.
    """
    renderer = AgentFrameRenderer(p, title)
    writer = Mp4Writer(out_path, p.fps)

    def on_frame(state):
        renderer.draw(state)
        writer.add(renderer.fig)
        if extra is not None:
            extra(state)

    try:
        hist = run_emergence(p, on_frame=on_frame, verbose=verbose)
    finally:
        writer.close()
        plt.close(renderer.fig)
    hist["frames"] = writer.frames
    return hist
