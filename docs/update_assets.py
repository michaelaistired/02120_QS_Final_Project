"""Copy figures and videos into docs/assets/ so the GitHub Pages site can show them.

    python docs/update_assets.py      (from the repo root, after re-running any experiment)

GitHub Pages serves only the docs/ folder, so the site cannot link to files elsewhere in the
repo. Only the videos the site embeds are copied, to keep the repository small.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent / "assets"

COPY = {
    "figures": [*sorted((ROOT / "visualizations" / "figures").glob("*.png")),
                *sorted((ROOT / "emergence" / "visualizations" / "figures").glob("*.png"))],
    "videos": [ROOT / "emergence" / "videos" / "stage0_paper_model.mp4",
               ROOT / "emergence" / "visualizations" / "figures" / "s1_compare.mp4",
               ROOT / "emergence" / "videos" / "stage1_pause.mp4"],
}


def main() -> None:
    """Copy every listed file, replacing older copies."""
    for folder, files in COPY.items():
        dest = ASSETS / folder
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)
            print(f"{folder}/{f.name}")


if __name__ == "__main__":
    main()
