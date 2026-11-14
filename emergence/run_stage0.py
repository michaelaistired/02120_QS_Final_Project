"""Stage 0 - positive control: the paper's model as a video, plus an exact-match check.

    python run_stage0.py            (from the emergence/ folder; ~3 min)

Writes:
    videos/stage0_paper_model.mp4          AHL + lactonase maps every 0.1 h (stride 10)
    results/stage0_control.md              our solver vs qs_model at 5, 12, 50 h (relative L2)
Use --stride 1 for a frame at every single 0.01 h PDE step (5000 frames).
"""

import argparse
from pathlib import Path

from qs_emergence.paper_control import compare_with_reference, render_paper_video

HERE = Path(__file__).parent


def main() -> None:
    """Render the Stage 0 video and write the control report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    (HERE / "videos").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)

    print("comparing with qs_model ...")
    diff = compare_with_reference(args.seed)
    lines = ["# Stage 0 positive control", "",
             f"Paper model, seed {args.seed}. This package's ADI solver vs the reference "
             "`qs_model` (relative L2 difference of the whole field, same seed):", "",
             "| time (h) | AHL | lactonase | result |", "|---|---|---|---|"]
    for t, d in diff.items():
        ok = d["U"] < 1e-10 and d["L"] < 1e-10
        lines.append(f"| {t:g} | {d['U']:.1e} | {d['L']:.1e} | {'PASS' if ok else 'FAIL'} |")
    (HERE / "results" / "stage0_control.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    print("rendering video ...")
    n = render_paper_video(HERE / "videos" / "stage0_paper_model.mp4", seed=args.seed,
                           stride=args.stride)
    print(f"wrote videos/stage0_paper_model.mp4 ({n} frames)")


if __name__ == "__main__":
    main()
