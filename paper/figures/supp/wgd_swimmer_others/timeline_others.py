"""wgd_swimmer_others figure (supp) — per-sample WGD/ccPG event timelines for the tumour types NOT
shown in the wgd_dynamics figure's panel a.

Same drawing as main/wgd_dynamics/a_timeline.py (imported, not copied): that panel shows
eight selected blocks (Lung adeno + SCC pooled); this one shows every other eligible type (the
`_all_types()` minus the selected ones — a type is dropped only if it has fewer than 5 event-bearing
samples), in the same landscape layout, at the same per-sample column width as the main panel so the
two figures read on one scale. Blocks ordered by mean first-event time like the main panel.
One script -> two figures, so the two can never drift.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))                                     # repo root
sys.path.insert(0, str(HERE.parents[1] / "main/wgd_dynamics"))   # a_timeline + _wgdpgd_common
from paper.config import panels_dir_of  # noqa: E402
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
import a_timeline as g_timeline  # noqa: E402

STEM = "timeline_others"
COL_MM = 0.39   # = the main panel's column width (200 mm / ~510 sample+gap units)
N_ROWS = 2      # as in the assembled supplementary figure: the 16 blocks over two rows of ~equal sample count


def make(save=True):
    return g_timeline.make(save=save, types=g_timeline.other_types(), stem=STEM,
                           panels_dir=panels_dir_of(__file__), col_mm=COL_MM,
                           rotate_labels=True, n_rows=N_ROWS)


if __name__ == "__main__":
    print("types:", g_timeline.other_types())
    make(); print(f"{STEM}: PNG+SVG written")
