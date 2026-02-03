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
