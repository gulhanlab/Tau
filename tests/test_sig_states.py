"""tau.sig_states is a port of D. Gulhan's estimate_cluster_timing.R — these tests check it against
the R's own output rather than against itself.

Reference material lives in dev/pcawg/signatures/R_reference/ (the R script and the estimates it
produced); the inputs are the repo's own sig_intervals_range_*_cluster.tsv, which are byte-identical to
the tables the R was run on. The whole-cohort test is skipped when those are absent, so the suite still
runs on a checkout without dev/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from tau.sig_states import StateFitParams, fit_cluster_table, fit_directory, fit_sample_signature

TABLES = R / "dev/pcawg/signatures/timing_tables"
REF = R / "dev/pcawg/signatures/R_reference/cluster_timing_estimates_BIC.tsv"
KEY = ["tumor_type", "sample", "signature"]
needs_dev = pytest.mark.skipif(not (TABLES.exists() and REF.exists()),
                               reason="dev/ signature tables or R reference not present")


@pytest.fixture(scope="module")
def fits():
    return fit_directory(TABLES), pd.read_csv(REF, sep="\t")


@needs_dev
def test_same_fits_selected(fits):
    """Every (tumour type, sample, signature) the R fitted, and no others."""
    py, r = fits
    assert set(map(tuple, py[KEY].drop_duplicates().values)) == \
           set(map(tuple, r[KEY].drop_duplicates().values))


@needs_dev
def test_same_number_of_states(fits):
    """The BIC/greedy search must stop in the same place — this is the part most likely to drift."""
    py, r = fits
    a = py.drop_duplicates(KEY).set_index(KEY).n_states
    b = r.drop_duplicates(KEY).set_index(KEY).n_states
    assert (a.sort_index() == b.sort_index()).all()


@needs_dev
def test_fitted_values_match_to_machine_precision(fits):
    py, r = fits
    j = py.merge(r, on=KEY + ["state"], suffixes=("_py", "_r"))
    assert len(j) == len(r)
    for c in ("log2_fold", "log2_fold_se", "baseline", "ratio"):
        assert np.nanmax(np.abs(j[f"{c}_py"] - j[f"{c}_r"])) < 1e-10, c


@needs_dev
def test_state_boundaries_match_except_degenerate_fits(fits):
    """Boundaries agree wherever the fit has any structure.

    One fit (Skin.Melanoma 0ab4d782, SBS7) is exactly flat: every state takes the same level and the
    residual sum of squares is ~0, so n*log(RSS/n) diverges, every partition ties, and the changepoint
    is arbitrary. R and this port break that tie differently. The fitted VALUES still agree, so the
    tolerance is on structure, not on the criterion.
    """
    py, r = fits
    j = py.merge(r, on=KEY + ["state"], suffixes=("_py", "_r"))
    spread = j.groupby(KEY).log2_fold_py.transform(lambda v: v.max() - v.min())
    # a single-state fit has zero spread trivially; degeneracy means MULTI-state with no spread
    degenerate = (spread <= 1e-9) & (j.n_states_py > 1)
    structured = j[~degenerate]
    assert np.nanmax(np.abs(structured.lo_py - structured.lo_r)) < 1e-9
    assert np.nanmax(np.abs(structured.hi_py - structured.hi_r)) < 1e-9
    assert j[degenerate].drop_duplicates(KEY).shape[0] <= 1


@needs_dev
def test_overlap_matrix_rows_sum_to_one():
    """Each interval's span must be fully accounted for by the states it overlaps."""
    from tau.sig_states import _build_B
    S0 = np.array([0.0, 0.2, 0.5]); E0 = np.array([0.4, 0.8, 1.0])
    B, lo, hi = _build_B(S0, E0, [0.3, 0.6], 0.0, 1.0)
    assert np.allclose(B.sum(axis=1), 1.0)
    assert lo[0] == 0.0 and hi[-1] == 1.0


@needs_dev
def test_more_states_never_selected_without_support():
    """Raising min_support_mut can only make the fit simpler, never more complex."""
    f = next(TABLES.glob("sig_intervals_range_*_cluster.tsv"))
    loose = fit_cluster_table(f, p=StateFitParams(min_support_mut=10.0))
    tight = fit_cluster_table(f, p=StateFitParams(min_support_mut=1e6))
    a = loose.drop_duplicates(KEY).set_index(KEY).n_states
    b = tight.drop_duplicates(KEY).set_index(KEY).n_states
    common = a.index.intersection(b.index)
    assert (b.loc[common] <= a.loc[common]).all()
