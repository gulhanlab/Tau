"""extra_oncogenes figure (supp), panel b — EGFR per-gain swimmer. Table-driven (see _swimmer.py). make() -> fig + PNG/SVG."""
import sys
from pathlib import Path
R = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(R))   # repo root
from paper.config import panels_dir_of  # noqa: E402
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
from paper.figures._shared._swimmer import make as _make  # noqa: E402


def make(save=True):
    return _make("EGFR", save=save, panels_dir=panels_dir_of(__file__), stem="b_egfr_swimmer")


if __name__ == "__main__":
    make(); print("b_egfr_swimmer: PNG+SVG written")
