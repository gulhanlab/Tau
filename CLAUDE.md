# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Tau is a Python bioinformatics tool that infers copy number amplification timing in tumor genomes. It uses the density of clock-like mutations (SNVs) in copy number segments to determine when genomic alterations occurred during tumor evolution.

## Build & Run Commands

### Install the package (development mode)
```bash
pip install -e .
```

### Run tests
```bash
pytest tests/
```

### Run on a sample
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

## Architecture

### Package Structure
```
tau/
├── __init__.py       # Public API exports, method attachment
├── core.py           # Segment, Genome classes, EM multiplicity
├── preprocessing.py  # VCF/CNV parsing, signature weighting
├── timing.py         # Route loading, timing analysis
├── clustering.py     # Event clustering (WGD/PGD)
├── plotting.py       # Genome-wide visualization
├── utils.py          # Helper functions
├── simulation.py     # Simulation framework
└── data/             # Reference data files
    ├── hg19.centromeres.tsv
    ├── hg19.chrom.sizes
    ├── 7_5_solutions_updated.sobj  # Sage solutions
    └── matrices_7_7.h5             # Constraint matrices
```

### Key Modules
- **core.py**: `Segment` and `Genome` dataclasses, EM algorithm for SNV multiplicity estimation
- **preprocessing.py**: Load SNVs from VCF, extract SBS96 context, apply COSMIC signature weights
- **timing.py**: Load Sage solutions and constraint matrices, solve timing equations via grid sampling
- **clustering.py**: EM-based clustering of timing estimates, WGD/PGD classification
- **plotting.py**: Genome-wide visualization (`plot_overview`)
- **utils.py**: Helper functions (`order_t`, `pick_best_key`, `extract_mat`, `make_time_df`)
- **simulation.py**: Synthetic tumor genome generation for validation

### Method Attachment
Timing methods are attached to Segment/Genome classes in `__init__.py`:
- `Segment.time_segment()` - Time a single segment
- `Genome.time_segments()` - Time all segments in a genome
- `Genome.times_to_df()` - Export timing results to DataFrame

### Data Dependencies
The `tau/data/` directory contains required reference files:
- `7_5_solutions_updated.sobj` (39MB) - Sage symbolic solutions for CN state routes
- `matrices_7_7.h5` (367MB) - HDF5 constraint matrices for route feasibility
- `hg19.centromeres.tsv` - Centromere positions
- `hg19.chrom.sizes` - Chromosome sizes

Environment variables can override default data paths:
- `TAU_ROUTES_SAGE` - Path to Sage solutions file
- `TAU_MATRICES_H5` - Path to constraint matrices file

## Dependencies

Core dependencies:
- numpy, pandas, scipy, h5py, pysam, matplotlib, seaborn, tqdm, scikit-learn

Additional requirement:
- SageMath (installed separately via conda or system package manager)

## Testing

Basic tests in `tests/`:
- `test_core.py` - Tests for Segment/Genome classes and EM algorithm
- `conftest.py` - Pytest fixtures
