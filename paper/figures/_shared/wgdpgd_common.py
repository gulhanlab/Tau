"""Shared loaders + styling for the punctuated-gain figures (main/punctuated, main/wgd_dynamics, supp/wgd_swimmer_others).

Imported through the three-line `_wgdpgd_common.py` stub in each figure directory, which binds the
directory panels are written to.

Origin: reproduces the analysis-tree `wgdpgd_common` API on top of the paper composite tables.

The Fig. 3 / Fig. 4 panel scripts were ported near-verbatim from `dev/pcawg/FigWGDPGD/*`. Those dev scripts
import `load_events` / `load_roster` + styling from `dev/pcawg/FigWGDPGD/wgdpgd_common.py`, which reads
`events_allpcawg.tsv` / `samples_allpcawg.tsv`. This shim serves the SAME API but sources everything
from the Tier-1 tables via `paper.data_tables.tables`, renaming columns to the dev names:

    event_time_frac -> time      genome_frac -> gf

so the plotting logic can stay unchanged. Styling (NatRev palette, Arial, rcParams, mm units) is copied
verbatim from the dev module.

Donor de-duplication: the dev module drops multi-aliquot donors to one representative aliquot
(`donor_dedup.representative_samples()`). The paper tables carry `is_representative_aliquot` for this,
but if that column is unpopulated (all-NA) we fall back to the full cohort and warn once — the panel
still renders, just over the un-deduplicated sample set.
"""
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[3]          # _shared -> figures -> paper -> repo root
sys.path.insert(0, str(R))
from paper.config import panels_dir_of  # noqa: E402
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
from paper.data_tables.tables import load_events as _load_events_tbl, load_samples as _load_samples_tbl  # noqa: E402

PANELS = None      # set by the per-figure stub (_wgdpgd_common.py in each figure directory)

MM = 1 / 25.4
# Nature canvas (mm): single 89 / double 183.
COL1, COL2 = 89 * MM, 183 * MM
GUTTER = 7 * MM
WA, WB, WC, WD, WE = 76 * MM, 100 * MM, 56 * MM, 56 * MM, 183 * MM

# NatRev palette (matches FigHighCN / dev fig4 colours)
WGD = "#0b6979"; PGD = "#cd9f2c"; BOTH = "#7b3f9e"; NONE = "#cacdd9"; CS = "#e7e6ec"
TXT = "#273246"; GREY = "#717c91"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "font.size": 6.0, "axes.titlesize": 7.0, "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 6.0,
    "figure.titlesize": 7.0, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "patch.linewidth": 0.4,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 300, "savefig.dpi": 450,
})
_style_apply()   # the shared ladder wins over the sizes in the block above


def _reps():
    """Set of representative-aliquot samples, or None if the column is unpopulated (no dedup possible)."""
    s = _load_samples_tbl()
    if "is_representative_aliquot" not in s.columns:
        return None
    m = s["is_representative_aliquot"]
    if m.notna().sum() == 0:                       # column present but entirely NA -> cannot dedup
        warnings.warn("tau_samples.is_representative_aliquot is all-NA; multi-aliquot donor dedup skipped "
                      "(panels use the full cohort).", stacklevel=2)
        return None
    return set(s.loc[m.fillna(False).astype(bool), "sample"])


def load_events(dedup=True):
    """Per-event table with the dev column names (time, gf). One row per clustered event."""
    ev = _load_events_tbl().rename(columns={"event_time_frac": "time", "genome_frac": "gf"})
    ev = ev.astype({"sample": str, "tumor_type": str, "classification": str})
    if dedup:
        reps = _reps()
        if reps is not None:
            ev = ev[ev["sample"].isin(reps)]
    return ev.reset_index(drop=True)


def load_roster(dedup=True):
    """Per-sample roster with the dev columns (sample, tumor_type)."""
    s = _load_samples_tbl()[["sample", "tumor_type"]].astype({"sample": str, "tumor_type": str})
    if dedup:
        reps = _reps()
        if reps is not None:
            s = s[s["sample"].isin(reps)]
    return s.reset_index(drop=True)


def save_panel(fig, stem, save=True, exact=True, panels_dir=None):
    """Write PNG + SVG for a panel into the calling figure's directory (svg.fonttype=none keeps text editable).

    exact=True mirrors the dev panels (no bbox crop; content kept inside the canvas via subplots_adjust);
    exact=False crops to tight bounds.
    """
    if not save:
        return fig
    bb = None if exact else "tight"
    fig.savefig((Path(panels_dir) if panels_dir else PANELS) / f"{stem}.png", dpi=300, bbox_inches=bb)
    fig.savefig((Path(panels_dir) if panels_dir else PANELS) / f"{stem}.svg", bbox_inches=bb)
    print(f"  saved {stem} -> {Path(panels_dir) if panels_dir else PANELS}", flush=True)
    return fig
