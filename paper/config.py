"""Central paths for the paper/ package.

Every script imports its locations from here instead of hardcoding them:

    from paper.config import DATA, REF, FROZEN, OBJECTS_PCAWG, panels_dir_of, cache_dir_of

Panel scripts read only from the data bundle: the composite tables and reference files under DATA,
and the per-figure cache tables under figures/data/<figure>/. The full per-sample Tau outputs are needed
only by the cache builders and the table-building scripts.
"""
import os
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parent
REPO = PAPER_ROOT.parent


def _env(name, default) -> Path:
    """A path that an environment variable may override (so the data can live outside the checkout)."""
    return Path(os.environ[name]).expanduser() if os.environ.get(name) else Path(default)


# ---- where the data lives --------------------------------------------------------------------------
# TAU_PAPER_DATA   the data bundle: Tier 1 tables (tau_*.tsv, bench_*), ref/, frozen/, objects/, per_sample/.
#                  Everything a PANEL script reads is under here.
DATA = _env("TAU_PAPER_DATA", PAPER_ROOT / "data_tables")
REF = DATA / "ref"                          # curated reference tables (OncoKB, cytoband, COSMIC spectra)
FROZEN = DATA / "frozen"                    # small analysis outputs frozen from upstream/ (see upstream/PROVENANCE.md)
PER_SAMPLE = DATA / "per_sample"            # per-sample bundles (re-clustered objects + per-sample tables)
OBJECTS = DATA / "objects"                  # the few raw Tau objects the object-based panels render
OBJECTS_PCAWG = OBJECTS / "pcawg"           #   <tumor_type>/<sample>/<sample>.genome.pkl.gz ...
OBJECTS_SIM = OBJECTS / "sim"               #   SIM_<n>/SIM_<n>.genome.pkl.gz ...
FIGURES = PAPER_ROOT / "figures"

# ---- full runs: needed ONLY to rebuild tables/caches from scratch, never by a panel script ----------
# TAU_PCAWG_OUTPUTS  per-sample Tau outputs for the whole cohort (<tumor_type>/<sample>/...); derived from
#                    controlled-access PCAWG data, so not distributed.
# TAU_SIM_OUTPUTS    the simulated cohort (SIM_*/...).
# TAU_ANALYSIS_DIR   working directory of the upstream analysis scripts (upstream/), whose outputs are
#                    what FROZEN holds a copy of.
TAU_OUTPUTS = _env("TAU_PCAWG_OUTPUTS", REPO / "dev/pcawg/tau_outputs/output_sbs1sbs5_soft_allpcawg")
SIM_OUTPUTS = _env("TAU_SIM_OUTPUTS", REPO / "dev/benchmarking/output_v8")
ANALYSIS = _env("TAU_ANALYSIS_DIR", REPO / "dev")
PCAWG_SAMPLES = PAPER_ROOT / "pcawg_run" / "samples"   # symlink to the Tau per-sample outputs (batch input only)


# ---------------------------------------------------------------------------------------------
# A panel script finds its own figure directory and cache directory from its file path, so scripts are
# movable and the figure directories are keyed by LABEL, not number.
def figure_dir_of(script_file) -> Path:
    """The figure directory a panel script lives in: paper/figures/{main,supp}/<label>/ (scripts and their
    panels sit together, flat). Scripts inside a figure's subdirectory (e.g. punctuated/wgd_timing_model/)
    resolve to the figure directory too."""
    p = Path(script_file).resolve().parent
    while p.parent.name not in ("main", "supp") and p != p.anchor:
        p = p.parent
    return p


def figures_root() -> Path:
    return Path(__file__).resolve().parent / "figures"


def panels_dir_of(script_file) -> Path:
    """Where a script writes its panels: its own figure directory."""
    return figure_dir_of(script_file)


def cache_dir_of(script_file) -> Path:
    """The figure's data directory: paper/figures/data/<label>/ (cache tables + their build scripts).
    Also works for a build script that lives IN data/<label>/."""
    p = Path(script_file).resolve()
    if "data" in p.parts and p.parts[p.parts.index("data") - 1] == "figures":
        d = figures_root() / "data" / p.parts[p.parts.index("data") + 1]
    else:
        d = figures_root() / "data" / figure_dir_of(script_file).name
    d.mkdir(parents=True, exist_ok=True)
    return d
