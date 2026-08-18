"""Sage-free unit tests for tau.sig_timing.

These tests exercise the exposure estimate, signature grouping and soft-posterior
weighting logic, which require neither Sage nor route files.
"""
import numpy as np
import pandas as pd
import pytest

from tau.core import Segment, Genome
from tau import sig_timing


def _make_genome():
    """Tiny two-segment genome with synthetic emission columns + hard calls."""
    sig_cols = ["SBS1", "SBS5", "SBS17a", "SBS17b", "SBS2", "SBS13"]

    def seg_table(n, hard, ctx_emit):
        df = pd.DataFrame({
            "chrom": ["1"] * n,
            "pos": np.arange(n) * 100 + 1,
            "ref": ["C"] * n,
            "alt": ["T"] * n,
            "nalt": [20] * n,
            "nref": [20] * n,
            "start": [0] * n,
            "end": [1000] * n,
            "major_cn": [2] * n,
            "minor_cn": [1] * n,
            "segment_id": ["s"] * n,
            "hard_sig_choice": hard,
        })
        for c in sig_cols:
            df[c] = ctx_emit[c]
        return df

    n1, n2 = 6, 4
    # segment 1: mostly SBS17b; segment 2: mostly SBS1
    t1 = seg_table(
        n1,
        hard=["SBS17b", "SBS17b", "SBS17a", "SBS5", "SBS1", "SBS2"],
        ctx_emit={
            "SBS1": np.full(n1, 0.1), "SBS5": np.full(n1, 0.1),
            "SBS17a": np.full(n1, 0.3), "SBS17b": np.full(n1, 0.4),
            "SBS2": np.full(n1, 0.05), "SBS13": np.full(n1, 0.05),
        },
    )
    t2 = seg_table(
        n2,
        hard=["SBS1", "SBS5", "SBS1", "SBS13"],
        ctx_emit={
            "SBS1": np.full(n2, 0.5), "SBS5": np.full(n2, 0.3),
            "SBS17a": np.full(n2, 0.05), "SBS17b": np.full(n2, 0.05),
            "SBS2": np.full(n2, 0.05), "SBS13": np.full(n2, 0.05),
        },
    )
    t1["segment_id"] = "segA"
    t2["segment_id"] = "segB"
    g = Genome([
        Segment(chrom="1", start=0, end=1000, major_cn=2, minor_cn=1,
                purity=0.8, seg_id="segA", snv_table=t1.reset_index(drop=True)),
        Segment(chrom="1", start=2000, end=3000, major_cn=2, minor_cn=1,
                purity=0.8, seg_id="segB", snv_table=t2.reset_index(drop=True)),
    ])
    return g


def test_exposure_estimate_sums_to_one():
    g = _make_genome()
    e = sig_timing.sample_exposure_estimate(g)
    assert abs(sum(e.values()) - 1.0) < 1e-9
    # 10 SNVs total; SBS17b appears twice in seg1 -> 2/10
    assert e["SBS17b"] == pytest.approx(2 / 10)
    assert e["SBS1"] == pytest.approx(3 / 10)  # 1 in seg1 + 2 in seg2
    assert all(v >= 0 for v in e.values())


def test_grouping():
    g = _make_genome()
    assert sig_timing.group_members(g, "SBS17") == ["SBS17a", "SBS17b"]
    assert sig_timing.group_members(g, "clock") == ["SBS1", "SBS5"]
    assert sig_timing.group_members(g, "APOBEC") == ["SBS13", "SBS2"]
    assert sig_timing._base_signature("SBS7d") == "SBS7"
    assert sig_timing._base_signature("SBS5") == "SBS5"


def test_active_signatures_includes_clock_and_sbs17():
    g = _make_genome()
    specs = sig_timing.active_signatures(g, thresh=0.15)
    names = {s["name"] for s in specs}
    assert "clock" in names                     # always present
    assert "SBS17" in names                      # 3/10 share > 0.15
    clock_spec = next(s for s in specs if s["name"] == "clock")
    assert clock_spec["members"] == ["SBS1", "SBS5"]


def test_soft_posterior_normalization():
    g = _make_genome()
    expo = sig_timing.sample_exposure_estimate(g)
    members = sig_timing.group_members(g, "SBS17")
    wmap = sig_timing.signature_mut_w(g, members, expo)
    for seg_id, w in wmap.items():
        assert np.all(w >= -1e-12)
        assert np.all(w <= 1.0 + 1e-9)          # soft posterior is a fraction


def test_soft_posterior_partition_to_one():
    """Per SNV, summing the soft weights over a partition of all signatures = 1."""
    g = _make_genome()
    expo = sig_timing.sample_exposure_estimate(g)
    clock = sig_timing.group_members(g, "clock")
    sbs17 = sig_timing.group_members(g, "SBS17")
    apobec = sig_timing.group_members(g, "APOBEC")
    w_clock = sig_timing.signature_mut_w(g, clock, expo)
    w_17 = sig_timing.signature_mut_w(g, sbs17, expo)
    w_ap = sig_timing.signature_mut_w(g, apobec, expo)
    # these three groups are the only signatures present -> weights sum to 1
    for seg_id in w_clock:
        total = w_clock[seg_id] + w_17[seg_id] + w_ap[seg_id]
        assert np.allclose(total, 1.0, atol=1e-9)
