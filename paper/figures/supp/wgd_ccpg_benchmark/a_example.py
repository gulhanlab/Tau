"""wgd_ccpg_benchmark figure (supp), panel a — worked example (SIM_3744): Tau overview vs ground truth.
SNAPSHOT / object-based (NOT table-driven). This panel is rendered from a simulation GENOME PKL
(<data>/objects/sim/SIM_3744/SIM_3744.genome.pkl.gz) by paper/upstream/simulation/panel_A_overview_compare.py, not from the composite
bench tables, so it is intentionally kept as a snapshot. To regenerate the underlying image, run the
dev script directly (needs the pixi env + the v8 sim output):

    python paper/upstream/simulation/panel_A_overview_compare.py

`make()` copies the existing snapshot into the supp panels dir and returns a figure displaying it.
"""
import shutil
import sys
from pathlib import Path
from paper.config import panels_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import (S, apply as _apply_style, save as _save_panel,
                                       _svg_longhand_fonts, MM, SINGLE, DOUBLE, HALF)
_apply_style()

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
from paper.config import FROZEN as SNAP   # frozen render; see the docstring
SOURCE_SCRIPT = "paper/upstream/simulation/panel_A_overview_compare.py"


def make(save=True):
    for ext in ("png", "svg", "pdf"):
        src = SNAP / f"fig3_a_example.{ext}"
        if src.exists():
            shutil.copyfile(src, PANELS / f"a_example.{ext}")
    # the dev snapshot is written by matplotlib too, so its SVG carries the same `font:` shorthand
    # that Illustrator ignores (falling back to 12 pt) — rewrite it to longhand like the other panels
    if (PANELS / "a_example.svg").exists():
        _svg_longhand_fonts(PANELS / "a_example.svg")
    png = PANELS / "a_example.png"
    fig, ax = plt.subplots(figsize=(SINGLE, 58 * MM))
    if png.exists():
        ax.imshow(mpimg.imread(png))
    ax.axis("off")
    ax.set_title("a_example (object-based snapshot — from SIM_3744 genome pkl)", fontsize=S.title)
    return fig


if __name__ == "__main__":
    make(); print("a_example: snapshot copied into panels/ (object-based, not table-driven)")
