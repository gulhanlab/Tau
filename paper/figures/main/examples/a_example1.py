"""examples figure (main), panel a — dc537fcf (Stomach.AdenoCA, double WGD): per-sample two-level overview.
See _example.py: this is `tau.overview.overview_figure` on the sample's objects, plus the assembled figure's titles and labels.
"""
from _example import make
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()

SHORT, STEM, TITLE = "dc537fcf", "a_example1", "Stomach Adenocarcinoma"

if __name__ == "__main__":
    make(SHORT, STEM, TITLE); print(f"{STEM}: PNG+SVG written")
