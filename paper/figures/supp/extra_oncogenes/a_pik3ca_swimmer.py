"""extra_oncogenes figure (supp), panel a — PIK3CA per-gain swimmer. Table-driven (shared swimmer over
figures/data/landscape/oncogene_gains.tsv.gz). make() -> fig + PNG/SVG."""
import sys
from pathlib import Path
R = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(R))   # repo root
from paper.config import panels_dir_of  # noqa: E402
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
from paper.figures._shared._swimmer import make as _make  # noqa: E402


def make(save=True):
    return _make("PIK3CA", save=save, panels_dir=panels_dir_of(__file__), stem="a_pik3ca_swimmer")


if __name__ == "__main__":
    make(); print("a_pik3ca_swimmer: PNG+SVG written")
