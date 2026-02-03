"""Pytest fixtures for Tau tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_snv_table():
    """Create a sample SNV table for testing."""
    np.random.seed(42)
    n_snvs = 50
    return pd.DataFrame({
        "chrom": ["1"] * n_snvs,
        "pos": np.arange(1000000, 1000000 + n_snvs * 1000, 1000),
        "ref": ["A"] * n_snvs,
        "alt": ["T"] * n_snvs,
        "nalt": np.random.poisson(15, n_snvs),
        "nref": np.random.poisson(35, n_snvs),
        "start": [1000000] * n_snvs,
        "end": [2000000] * n_snvs,
        "major_cn": [3] * n_snvs,
        "minor_cn": [1] * n_snvs,
        "segment_id": ["1:1000000-2000000"] * n_snvs,
        "mut_w": [1.0] * n_snvs,
        "subclonal": [False] * n_snvs,
        "total_cn": [4] * n_snvs,
    })


@pytest.fixture
def sample_segment(sample_snv_table):
    """Create a sample Segment for testing."""
    from tau.core import Segment
    return Segment(
        chrom="1",
        start=1000000,
        end=2000000,
        major_cn=3,
        minor_cn=1,
        purity=0.8,
        snv_table=sample_snv_table,
    )


@pytest.fixture
def sample_genome(sample_snv_table):
    """Create a sample Genome for testing."""
    from tau.core import Genome
    return Genome.create(sample_snv_table, purity=0.8)
