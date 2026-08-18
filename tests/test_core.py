"""Tests for tau.core module."""

import numpy as np
import pandas as pd
import pytest

from tau.core import Segment, Genome, truncated_binom_pmf, _em_pi, _ess


class TestTruncatedBinomPmf:
    """Tests for truncated_binom_pmf function."""

    def test_below_threshold_returns_zero(self):
        """PMF should return 0 for observations below detection threshold."""
        assert truncated_binom_pmf(2, 50, 0.3, min_alt=3) == 0.0
        assert truncated_binom_pmf(1, 50, 0.3, min_alt=3) == 0.0

    def test_at_threshold_returns_nonzero(self):
        """PMF should return nonzero for observations at/above threshold."""
        result = truncated_binom_pmf(3, 50, 0.3, min_alt=3)
        assert result > 0

    def test_invalid_inputs_return_zero(self):
        """PMF should handle edge cases gracefully."""
        assert truncated_binom_pmf(5, 0, 0.3, min_alt=3) == 0.0
        assert truncated_binom_pmf(5, 50, 0.0, min_alt=3) == 0.0
        assert truncated_binom_pmf(5, 50, 1.0, min_alt=3) == 0.0


class TestEmPi:
    """Tests for EM mixture weight estimation."""

    def test_uniform_initialization(self):
        """EM should converge from uniform initialization."""
        np.random.seed(42)
        L = np.random.rand(100, 3) + 0.1
        pi, P = _em_pi(L)
        assert len(pi) == 3
        assert np.isclose(pi.sum(), 1.0)

    def test_posterior_shape(self):
        """Posterior matrix should have correct shape."""
        np.random.seed(42)
        L = np.random.rand(100, 4) + 0.1
        pi, P = _em_pi(L)
        assert P.shape == (100, 4)
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)


class TestEss:
    """Tests for effective sample size calculation."""

    def test_uniform_weights(self):
        """ESS of uniform weights should equal N."""
        w = np.ones(100)
        assert np.isclose(_ess(w), 100.0)

    def test_single_weight(self):
        """ESS of single nonzero weight should be 1."""
        w = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.isclose(_ess(w), 1.0)


class TestSegment:
    """Tests for Segment class."""

    def test_segment_creation(self, sample_segment):
        """Segment should be created with correct attributes."""
        assert sample_segment.chrom == "1"
        assert sample_segment.major_cn == 3
        assert sample_segment.minor_cn == 1
        assert sample_segment.total_cn == 4
        assert sample_segment.n_snvs == 50

    def test_multiplicity_likelihoods_computed(self, sample_segment):
        """Multiplicity likelihoods should be computed on creation."""
        assert "multiplicity_likelihoods" in sample_segment.snv_table.columns
        likelihoods = sample_segment.snv_table["multiplicity_likelihoods"].iloc[0]
        assert len(likelihoods) == sample_segment.major_cn

    def test_estimate_pi_em(self, sample_segment):
        """EM estimation should populate pi_em and N_counts."""
        sample_segment.estimate_pi_em()
        assert sample_segment.pi_em is not None
        assert sample_segment.N_counts is not None
        assert len(sample_segment.pi_em) == sample_segment.major_cn
        assert np.isclose(sample_segment.pi_em.sum(), 1.0)

    def test_bootstrap_replicates(self, sample_segment):
        """Bootstrap should create multiple N_counts replicates."""
        sample_segment.estimate_pi_em(bootstrap_B=10, random_state=42)
        assert sample_segment.N_counts_boot is not None
        assert sample_segment.N_counts_boot.shape[0] == 10


class TestGenome:
    """Tests for Genome class."""

    def test_genome_creation(self, sample_genome):
        """Genome should be created with segments."""
        assert len(sample_genome.segments) > 0

    def test_genome_create_missing_columns(self):
        """Genome.create should raise error for missing columns."""
        df = pd.DataFrame({"chrom": ["1"], "pos": [1000]})
        with pytest.raises(ValueError, match="missing required columns"):
            Genome.create(df, purity=0.8)

    def test_calculate_multiplicities(self, sample_genome):
        """calculate_multiplicities should run on all segments."""
        sample_genome.calculate_multiplicities(bootstrap_B=1, random_state=42)
        for seg in sample_genome.segments:
            assert seg.pi_em is not None
            assert seg.N_counts is not None

    def test_to_segment_summary(self, sample_genome):
        """to_segment_summary should return DataFrame."""
        sample_genome.calculate_multiplicities(bootstrap_B=1, random_state=42)
        df = sample_genome.to_segment_summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_genome.segments)

    def test_normal_cn_map(self, sample_snv_table):
        """normal_cn_map should set normal_cn per chromosome."""
        g = Genome.create(sample_snv_table, purity=0.8, normal_cn_map={"1": 1})
        assert g.segments[0].normal_cn == 1

    def test_normal_cn_default(self, sample_genome):
        """normal_cn defaults to 2 when no map supplied."""
        assert all(s.normal_cn == 2 for s in sample_genome.segments)

    def test_calculate_gof(self, sample_genome):
        """calculate_gof populates gof dict on all EM-estimated segments."""
        sample_genome.calculate_multiplicities(random_state=42)
        sample_genome.calculate_gof()
        for seg in sample_genome.segments:
            assert seg.gof is not None
            assert "mean_U" in seg.gof
            assert "ks_p" in seg.gof
            assert 0.0 <= seg.gof["mean_U"] <= 1.0

    def test_gof_in_summary(self, sample_genome):
        """GOF columns appear in to_segment_summary after calculate_gof."""
        sample_genome.calculate_multiplicities(random_state=42)
        sample_genome.calculate_gof()
        df = sample_genome.to_segment_summary()
        assert "gof_mean_U" in df.columns
        assert "gof_ks_p" in df.columns
        assert "gof_n" in df.columns


class TestSegmentNormalCn:
    """Tests for normal_cn and GOF on Segment."""

    def test_normal_cn_affects_p_vec(self, sample_snv_table):
        """normal_cn=1 should give higher expected VAFs than normal_cn=2."""
        from tau.core import Segment
        seg2 = Segment(chrom="1", start=1000000, end=2000000,
                       major_cn=2, minor_cn=0, purity=0.5,
                       snv_table=sample_snv_table.copy(), normal_cn=2)
        seg1 = Segment(chrom="1", start=1000000, end=2000000,
                       major_cn=2, minor_cn=0, purity=0.5,
                       snv_table=sample_snv_table.copy(), normal_cn=1)
        # With haploid normal, denominator is smaller → larger expected VAF
        L2 = seg2.snv_table["multiplicity_likelihoods"].iloc[0]
        L1 = seg1.snv_table["multiplicity_likelihoods"].iloc[0]
        assert L1[0] != L2[0], "Likelihoods should differ between normal_cn=1 and 2"

    def test_compute_gof_returns_dict(self, sample_segment):
        """compute_gof should populate self.gof after EM."""
        sample_segment.estimate_pi_em()
        sample_segment.compute_gof()
        assert sample_segment.gof is not None
        assert set(sample_segment.gof.keys()) >= {"mean_U", "ks_stat", "ks_p", "n"}

    def test_compute_gof_before_em(self, sample_segment):
        """compute_gof before EM sets gof=None."""
        sample_segment.compute_gof()
        assert sample_segment.gof is None

    def test_compute_gof_mean_u_range(self, sample_segment):
        """mean_U should be in [0, 1]."""
        sample_segment.estimate_pi_em()
        sample_segment.compute_gof()
        assert 0.0 <= sample_segment.gof["mean_U"] <= 1.0


def test_genome_carries_exposures_explicitly_and_via_attrs():
    """A pickled Genome should be self-describing about its signature exposures.

    Two routes in, because pandas drops DataFrame.attrs on most operations (filter, concat, merge):
    any caller that touches the mutation table between preprocess_sample and Genome.create loses the
    attrs copy silently, which is why runs predating this carry genome.exposures = None.
    """
    import numpy as np
    import pandas as pd
    from tau.core import Genome

    df = pd.DataFrame({
        "chrom": ["1"] * 4, "pos": [100, 200, 300, 400],
        "ref": list("ACGT"), "alt": list("TGCA"),
        "nalt": [10, 12, 9, 11], "nref": [10, 8, 11, 9],
        "start": [1] * 4, "end": [1000] * 4,
        "major_cn": [2] * 4, "minor_cn": [1] * 4,
        "segment_id": ["1:1-1000"] * 4, "mut_w": [1.0] * 4,
    })

    # 1. explicit argument wins
    g = Genome.create(df, purity=0.8, exposures={"SBS1": 100.0, "SBS5": 50.0})
    assert g.exposures == {"SBS1": 100.0, "SBS5": 50.0}

    # 2. falls back to attrs when no argument is given
    d2 = df.copy()
    d2.attrs["exposures"] = {"SBS17a": 7.0}
    assert Genome.create(d2, purity=0.8).exposures == {"SBS17a": 7.0}

    # 3. explicit argument overrides a stale attrs payload
    assert Genome.create(d2, purity=0.8, exposures={"SBS3": 1.0}).exposures == {"SBS3": 1.0}

    # 4. absent everywhere -> None, not a missing attribute
    assert Genome.create(df, purity=0.8).exposures is None
