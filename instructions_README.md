# Tau

Tau is a Python bioinformatics tool that infers copy number amplification timing in tumor genomes. It uses the density of clock-like mutations (SNVs) in copy number segments to determine when genomic alterations occurred during tumor evolution.

## Installation

### From source (development)

```bash
git clone https://github.com/parklab/tau.git
cd tau
pip install -e .
```

### Dependencies

Tau requires Python 3.9+ and the following packages (automatically installed):

- numpy >= 1.21
- pandas >= 1.3
- scipy >= 1.7
- h5py >= 3.0
- pysam >= 0.19
- matplotlib >= 3.4
- seaborn >= 0.11
- tqdm >= 4.62
- scikit-learn >= 1.0

**Additional requirement:** SageMath must be installed separately (via conda or system package manager) for timing analysis.

### Git LFS (for large data files)

The package includes large data files that require Git LFS:

```bash
git lfs install
git lfs pull
```

## Quick Start

```python
from tau import Segment, Genome, preprocess_sample
from tau import timing

# Load pre-computed routes (required for timing)
env = timing.load_routes(
    "tau/data/7_5_solutions_updated.sobj",
    "tau/data/matrices_7_7.h5"
)

# Preprocess VCF and CNV data
mut_df = preprocess_sample(
    sample="SAMPLE1",
    vcf_path="sample.vcf.gz",
    cnv_tsv="sample.cna.txt",
    purity=0.8,
    ref_fasta="/path/to/reference.fasta",
    cosmic_csv="/path/to/COSMIC_signatures.csv",
    exposures_tsv="/path/to/sample_exposures.txt",
    signatures=["SBS1", "SBS5"],
)

# Create genome and run analysis
g = Genome.create(mut_df, purity=0.8)
g.calculate_multiplicities(bootstrap_B=10)
g.time_segments(env=env)

# Get timing results
from tau.utils import make_time_df
times_df = make_time_df(g)
```

## Command-line Interface

```bash
python -m scripts.run_tau \
  --sample SAMPLE_ID \
  --vcf sample.vcf.gz \
  --cnv sample.cna.txt \
  --purity 0.8 \
  --ref_fasta /path/to/reference.fasta \
  --cosmic_csv /path/to/COSMIC_signatures.csv \
  --exposures /path/to/exposures.txt \
  --output_times sample.segment_times.tsv \
  --output_fig sample.overview.png \
  --output_gpkl sample.genome.pkl.gz \
  --output_cluster_tsv sample.segment_clusters.tsv \
  --output_cluster_plot sample.segment_clusters.png \
  --output_cluster_gpkl sample.clustered_genome.pkl.gz \
  --output_time_cluster_plot sample.time_clusters.png \
  --output_time_cluster_df sample.time_clusters.tsv
```

## Pipeline Overview

1. **Preprocessing** (`tau.preprocessing`): Load SNVs from VCF, extract SBS96 context, apply COSMIC signature weights
2. **Multiplicity estimation** (`tau.core`): EM algorithm for estimating SNV multiplicity mixture proportions
3. **Timing** (`tau.timing`): Load Sage solutions and constraint matrices, solve timing equations via grid sampling
4. **Clustering** (`tau.clustering`): EM-based clustering of timing estimates, classify events as WGD/PGD/SCA
5. **Visualization** (`tau.plotting`): Genome-wide timing visualization

## Output Files

Per-sample outputs include:

- `*.segment_times.tsv` - Per-segment timing estimates
- `*.segment_clusters.tsv` - Segment cluster assignments
- `*.time_clusters.tsv` - Timepoint cluster summary
- `*.overview.png` - Genome-wide timing visualization
- `*.genome.pkl.gz` - Pickled Genome object

## Environment Variables

You can override default data file paths using environment variables:

- `TAU_ROUTES_SAGE`: Path to Sage solutions file
- `TAU_MATRICES_H5`: Path to constraint matrices HDF5 file

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Citation

If you use Tau in your research, please cite:

[Citation to be added]

## License

MIT License - see LICENSE file for details.
