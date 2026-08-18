# Timing analysis with Tau

- [Background](#background)
	- [Diagram representation](#diagram-representation)
	- [Matrix representation of diagrams](#matrix-representation-of-diagrams)
	- [Systematically compose the A matrices](#systematically-compose-the-A-matrices)
	- [Solving for time when there is no exact solution](#solving-for-time-when-there-is-no-exact-solution)
- [Usage and Installation](#usage-and-installation)
	- [Installation](#installation)
		- [Dependencies](#dependencies)
	- [Try it: zero-input demo](#try-it-zero-input-demo)
	- [Quick Start (Python API)](#quick-start-python-api)
	- [Command-line Interface](#command-line-interface)
	- [Input file formats](#input-file-formats)
	- [Interactive viewer](#interactive-viewer)
	- [Pipeline Overview](#pipeline-overview)
	- [Environment Variables](#environment-variables)
	- [Citation](#citation)
	- [License](#license)

# Background

<img align="right" src="https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/Mutations_as_clocks_at_amplified_segments.png" width="400">

Point mutations can be used to time copy number amplifications based on:
* the assumption that mutation counts reflect molecular clock, meaning older cells carry more mutations.
* the simple observation that the early amplifications will carry late point mutations, meaning they were generated after the amplification, on single copies of the chromosome. While, late amplifications will carry early mutations, generated before the amplifications, that are duplicated on multiple chromosomes.

## Diagram representation

Here is a diagram showing the time evolution for a diploid chromosome segment (on the left hand side) that evolves into a major copy number 3 and minor copy number 2 state.

![Image description](https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/Diagram_to_equations.png)

Assuming molecular time is measured by number of mutations, the time intervals are expected to be proportional to number of mutations based on the equations listed above (add the t intervals for each color). 

### Single state can be achieved in multiple ways

For the example above we have three diagrams that differ from each other based on when minor allele is amplified with respect to major allele.

![Image description](https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/diagrams.png)

Two of the diagrams results in the same set of equations, while third one where amplification in minor segments happen first differs.

### Amplifications lead to more DNA material leading to a higher rate of mutations than diploid chromosome

Despite the lack of exact solutions for several copy number evolution trajectories, it is always possible to apply a normalization to the mutation counts such that the boost on the number of mutations due to more DNA material can be cancelled out. This normalization has a geometric interpretation as demonstrated for the M = 3, N = 2 state in the diagram below by the dashed lines that complete the missing lines in our grid. The green lines are duplicated (meaning a factor of 2 for N2) and the red line tripled (meaning a factor of 3 for N3). N<sub>1</sub> + 2N<sub>2</sub> + 3N<sub>3</sub> 

![Image description](https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/Normalization.png)

## Matrix representation of diagrams

The set of linear equations can be represented as a matrix equation, for the example above:

<img class="center" src="https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/Matrix_representation.png" width=500>

In the A matrix, the placement of values in each row indicates final CN values, for example a value of 1 in row 3 indicates that the segment will be amplified to 3 copies, a value of 2 in row 2 indicates two segments that will be amplified to 2 copies, a diagonal change in the numbers mean a branching from a given state to another, see the arrows below. In addition, matrix A can be expressed as the sum of contributions of minor (N) and major (M) chromosome, which are composed of branching (B<sub>M</sub> and B<sub>N</sub>) and propagation matrices (P<sub>M</sub> and P<sub>N</sub>)

<img class="center" src="https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/factorizing_A_matrix.png" width = 700>

For the above matrices the graphical represantation of propagation and branching matrices are as shown below:

<img class="center" src="https://github.com/gulhanlab/Tau_R/blob/main/inst/extdata/image/propagations_branchings.png" width="600">

## Systematically compose the A matrices

The decomposition into branching and propagation matrices makes the calculation of A matrices easier.

*Branching matrices:* Given an initial copy number state a propagation matrix is supposed to take all the copy numbers from the bottom to the top rows while moving from left to right such that the next column will have a change in higher rows, say of index i, of magnitude -1, and this will be compensated by the increase of + 1in rows that represent the branching products in the following column. In the example above for the major copy number, the first column contains a value of 1 at the row 3, meaning that this allele will eventually generate 3 copies, which branches into a 2 and a 1, which means these branches will later lead to 2 and 1 copies.

*Propagation matrices*: are constructed by selecting different permutations of how gains will be distributed among major and minor allele across the different time points. They exist in pairs for major and minor alleles as one defines the other.

Note that branching, propagation and composite A matrices have already been calculated and are saved in the package for major copy number of 7 and minor copy number of 7 and above this value for major copy number of 10 and minor copy number of 2 for capturing focal amplifications. These copy number states cover 99.8% of the genome on average and 81% of all segments in the consesnsus copy number calls in WGS data from Pancancer Analysis of Whole Genomes (PCAWG) project. See below the functions on how to access these matrices.

## Solving for time when there is no exact solution

The matrices were solved using [SageMath](https://www.sagemath.org/) and saved in the Tau package which provides expressions either in terms of mutation counts N<sub>i</sub> and in addition some of the unknown t<sub>i</sub> values due to remaining degrees of freedom when equations are not exactly solvable. The solution also provides conditions that need to be satisfied. These solutions define intervals for each t<sub>i</sub> value.

Another important observation is the lack of a unique diagram for each final copy number state, instead a solution is provided for each diagram. Later either some of these solutions can be preselected based on its agreement with other segments (or copy number states) or all solutions can be treated equally (up to the multiplicative scale that defines how many times the corresponding matrix naturally occurs from distinct diagrams, e.g. for M = 3, N = 2 we saw that two diagrams provide the same solution, so this solution will be twice as likely as the other solution). The solutions obtained by each unique diagram can be used to define an average time value for the time intervals as well as upper and lower bounds. 


# Usage and Installation

## Installation

### With pixi (recommended)

Tau depends on [SageMath](https://www.sagemath.org/) (and Singular) for the route
solutions. The reliable way to get a fully-working environment — including Sage on
`$PATH` — is [pixi](https://pixi.sh):

```bash
git clone https://github.com/parklab/tau.git
cd tau
pixi install           # creates .pixi/envs/default with Sage + all deps
pixi run test          # sanity check: runs the test suite
pixi run demo          # zero-input end-to-end demo (see below)
```

Run any command inside the environment with `pixi run <cmd>`, e.g.
`pixi run tau --help` or `pixi run python my_script.py`.

> **Note on `pip install`:** `pip install -e .` installs the Python dependencies but
> **not SageMath**, so route loading (and therefore timing) will fail. Use pixi (or a
> conda environment that provides `sage`) for a functional install.

### Dependencies

Tau requires Python 3.9+. The pinned versions live in `pyproject.toml` (the
`[tool.pixi.dependencies]` table is the source of truth). Key dependencies:

- numpy, pandas, scipy, h5py, statsmodels, scikit-learn
- pysam (VCF reading)
- matplotlib, seaborn (plotting)
- tqdm
- **SageMath** (route solutions — provided by pixi/conda, *not* by pip)
- snakemake (optional; for batch workflows)

## Try it: zero-input demo

No data files required. This synthesises a small WGD genome and runs the whole
pipeline (multiplicity EM → timing → clustering → plot):

```bash
pixi run demo
# or, inside the environment:
tau demo --output_dir tau_demo_out
```

It writes `DEMO.segment_times.tsv`, `DEMO.events.tsv`, and `DEMO.overview.png`, and
prints the recovered event vs. the ground-truth WGD time. The first run takes a
couple of minutes because it imports Sage and loads route solutions.

## Quick Start (Python API)

The recommended entry point is `run_sample()`, which chains
preprocessing → EM → timing → clustering in one call:

```python
from tau import run_sample, timing

# Load only the CN states you expect (fast); or use load_routes(...) for everything.
env = timing.load_routes_for_states([(2, 1), (2, 2), (3, 1), (4, 2)])

genome, events = run_sample(
    "SAMPLE1",
    vcf_path="sample.vcf.gz",
    cnv_tsv="sample.cna.txt",
    purity=0.8,
    env=env,
    ref_fasta="/path/to/reference.fasta",   # optional (needed only with signatures)
    cosmic_csv="/path/to/COSMIC_signatures.csv",  # optional
    exposures_tsv="/path/to/sample_exposures.txt",  # optional
    signatures=["SBS1", "SBS5"],             # optional; omit for equal-weight mutations
)

# `events` is one row per detected event (time, classification, gf, n_chroms, ...)
print(events[["time", "classification", "gf"]])

# Long-form per-segment timing table:
times_df = genome.times_to_df("SAMPLE1")

# Genome-wide overview plot:
from tau.plotting import plot_overview
event_times, seg_cluster_ids, orig_times, _ = genome._cluster_result
plot_overview(genome, cluster_times=event_times,
              segment_cluster_ids=seg_cluster_ids, original_times=orig_times,
              output_file="timing_overview.png")
```

**Signatures are optional.** If you omit `signatures` (and `cosmic_csv`/`exposures_tsv`),
every mutation is weighted equally — no COSMIC download or signature-fitting step is
needed. Provide `signatures=["SBS1", "SBS5"]` (plus the COSMIC CSV, exposures TSV, and
reference FASTA) only if you want to down-weight non-clock mutations.

If you need finer control, the same pipeline is available step by step
(`preprocess_sample → Genome.create → g.run(env)`); see `examples/quickstart.ipynb`.

## Command-line Interface

The `tau` command has two subcommands: `tau run` (real data) and `tau demo`
(synthetic). `python -m scripts.run_tau ...` is a thin alias for `tau run`.

```bash
tau run \
  --sample SAMPLE_ID \
  --vcf sample.vcf.gz \
  --cnv sample.cna.txt \
  --purity 0.8 \
  --output_dir results/
  # signatures off by default (equal-weight). To weight by clock signatures, add:
  #   --signatures SBS1 SBS5 \
  #   --ref_fasta ref.fa --cosmic_csv COSMIC.csv --exposures exposures.tsv
```

A run writes **15 files** into `--output_dir`, and nothing else:

```
{sample}.tau_samples.tsv                {sample}.tau_merged_segments.tsv
{sample}.tau_segments.tsv               {sample}.tau_merged_multiplicities.tsv
{sample}.tau_multiplicities.tsv         {sample}.tau_merged_routes.tsv
{sample}.tau_routes.tsv                 {sample}.tau_merged_gains.tsv.gz
{sample}.tau_gains.tsv.gz               {sample}.tau_merged_events.tsv
{sample}.tau_events.tsv

{sample}.overview.png  +  .svg          the two-level overview figure
{sample}.genome.pkl.gz                  the full Genome object
{sample}.clustered_genome.pkl.gz        the pooled genome
```

Eleven tables (see [Composite tables](#composite-tables)), the overview, and the two pickles. The
tables are what most analyses want; the pickles keep everything the tables flatten, and let
`tau export-tables` rebuild the whole set later without re-timing.

Give each sample its own directory and point `tau aggregate` at the parent — it searches recursively,
so any nesting works:

```bash
tau run --sample S1 ... --output_dir results/S1/
tau run --sample S2 ... --output_dir results/S2/
tau aggregate results/ --out cohort/
```

Add `--interactive` for `{sample}.overview_interactive.html`, the self-contained viewer
(see [Interactive viewer](#interactive-viewer)). Off by default — it precomputes a tree per route.

`--bootstrap_B` defaults to **10**, matching the Python API (`run_sample`/`Genome.run`).
Lower it to `1` to skip bootstrap uncertainty for a faster run.

The interactive HTML is opt-in: add `--interactive`. Use `--tree-top-k 0` to omit the per-route tree
explorer (faster; no Sage needed for the viewer step). `--tree-top-k N` (default 5) sets how many
routes per segment get a tree.

## Input file formats

| Input | Required? | Format |
|---|---|---|
| `--vcf` | yes | Somatic SNV VCF (`.vcf`/`.vcf.gz`). Read with pysam; needs standard `CHROM/POS/REF/ALT` and a sample genotype column carrying alt/ref depths. |
| `--cnv` | yes | Tab-separated CN segments. Columns: `chromosome` (or `chrom`), `start`, `end`, and either `major_cn`/`minor_cn` (PCAWG-style integers) **or** `majorAlleleCopyNumber`/`minorAlleleCopyNumber` (PURPLE-style floats). `chr` prefixes are stripped automatically. |
| `--purity` / `--purity_file` | yes (one of) | A float in `(0, 1]`, or a TSV with `samplename` and `purity` columns. |
| `--ref_fasta` | only with `--signatures` | Reference FASTA matching the VCF build (e.g. hg19), for SBS96 context extraction. |
| `--cosmic_csv` | only with `--signatures` | COSMIC SBS definitions CSV (download from <https://cancer.sanger.ac.uk/signatures/>). First column = SBS96 type, columns `SBS1`, `SBS5`, … |
| `--exposures` | only with `--signatures` | Per-sample exposures TSV: a `sample` (or `aliquot_id`) column plus one column per `SBS*` signature. |

If `--signatures` is omitted (the default), the last three files are not needed and
all mutations are weighted equally.

## Subclonal mutations

A subclonal mutation is present in only part of the tumour, so its VAF understates its true
multiplicity. Left in, such mutations pile into the multiplicity-1 bin and pull amplification
times **earlier**. Tau offers two independent ways to remove them, and they **combine by union** —
a mutation is dropped if either source flags it. Flagged mutations get zero weight in the EM.

**Both are off by default**, which is what every published Tau run used.

### 1. Built-in test — `--subclonal-filter`

A one-sided confidence test. For each mutation Tau takes the upper 99% bound of the VAF posterior,
`Beta(nalt+1, nref+1)`, converts it to a CCF assuming multiplicity 1

```
ccf_upper = p₉₉ · (purity·total_cn + 2(1−purity)) / purity
```

and flags the mutation only if `ccf_upper < 1` — i.e. even the optimistic reading cannot reach
clonality. Set the confidence with `--subclonal-alpha` (default `0.01`).

> **Its power depends on purity.** At low purity the VAF posterior is too wide for the test to
> ever conclude CCF < 1, so nothing is flagged; at high purity a few percent are. Measured across
> PCAWG samples: 0.0% flagged at purity 0.20, 2.3% at 0.63, 7.8% at 0.77. Enabling it across a
> cohort therefore applies a correction whose strength scales with purity — fine within a sample,
> something to control for when comparing samples.

### 2. Your own list — `--subclonal-list`

Supply an external caller's assignment (PyClone-VI, DPClust, …) as a text file with one mutation ID
per line, formatted `{chrom}:{pos}:{ref}/{alt}`:

```
1:2489573:C/T
7:55086725:G/A
```

```bash
tau run --sample SAMPLE --vcf snvs.vcf.gz --cnv cn.tsv --purity 0.8 \
  --subclonal-filter --subclonal-list pyclone_subclonal.txt \
  --output_dir results/
```

This is generally the better option when you have it: an external caller uses the whole VAF
distribution and copy-number context to fit clusters, rather than testing each mutation alone.
Use both together to catch what either misses.

The Python API takes the same three arguments:

```python
g, events = run_sample(..., apply_subclonal_filter=True,
                       subclonal_list="pyclone_subclonal.txt")
```

## Composite tables

The pickles hold everything, but they need Python and the Tau classes to open. Every `tau run` also
writes the same results as flat tables — this is the analysis-ready output, and what `tau aggregate`
consumes to build a cohort.

| Table | One row per | Key columns |
|---|---|---|
| `{sample}.tau_samples.tsv` | sample | `purity`, `ploidy_seg`, `n_snvs`, `n_segments`, `n_timed_segments`, `mean_ess` |
| `{sample}.tau_segments.tsv` | CN segment | coords, `major_cn`/`minor_cn`, `ess`, `is_timed`, `discard_reason`, `event_id`, `merged_id` |
| `{sample}.tau_multiplicities.tsv` | segment × multiplicity | `n_count` (the EM `N_counts`), `pi` |
| `{sample}.tau_routes.tsv` | segment × route | `n_events`, `n_draws`, `free_var_count`, `dist_rel`, `inside_margin`, `is_best_route` |
| `{sample}.tau_gains.tsv.gz` | segment × route × gain × solution | `gain_index`, `draw_id`, `gain_time_frac`, `event_id` |
| `{sample}.tau_events.tsv` | detected event | `event_time_frac`, `classification`, `genome_frac`, `n_chroms` |

A parallel `tau_merged_*` set carries the same six tables computed on the **pooled** genome, where
segments sharing a CN state and a timing cluster are merged into one higher-power unit. Merged tables
drop genomic coordinates (a pooled unit spans many loci) and add `n_segments`.

`tau_gains` is the one to reach for in most analyses. Each row is one polytope solution for one gain —
these are equally valid points in an underdetermined solution space, **not** posterior draws, so a
segment with `n_draws = 50` has 50 equally good answers rather than a distribution to average. Filter
to `is_best_route` for one route per segment.

Skip the tables with `--no-tables`. Set `--tumor_type` to label them, and `--output_tables_dir` to
send them elsewhere. To build them for runs that predate this (from the pickles, no re-timing):

```bash
tau export-tables results/SAMPLE.genome.pkl.gz --out tables/
```

`tau aggregate <dir> --out cohort/` concatenates every sample it finds under `<dir>`, at any depth —
one flat directory of per-sample files, a directory per sample, or a tree nested by tumour type all
work the same way.

## Signature activity over time

`--signatures SBS1 SBS5` picks the clock used to *tell the time*. A separate, optional step
asks the reverse question: **given that clock, when was each other signature active?**

Add `--include-signatures` to `tau run`. Tau finds every signature whose exposure share
exceeds `--sig-thresh` (default `0.15`), re-times the whole genome under each one, and
compares its per-interval widths against the clock's:

```bash
tau run --sample SAMPLE --vcf snvs.vcf.gz --cnv cn.tsv --purity 0.8 \
  --ref_fasta hg19.fa --cosmic_csv COSMIC_v3.4_SBS_GRCh37.txt \
  --exposures exposures.tsv --signatures SBS1 SBS5 \
  --include-signatures --sig-thresh 0.15 \
  --output_dir out/
```

Three extra files land in `--output_dir`:

| File | Contents |
|---|---|
| `{sample}.sig_activity.tsv` | one row per (signature, segment, gain interval): `t0`/`t1` (the interval's span in clock time), `ct`/`st` (clock and signature interval widths), `ratio`, `ess`, `ok` |
| `{sample}.sig_summary.tsv` | one row per signature: exposure `share`, segments compared, and `genome_ratio` — its genome-wide abundance relative to the clock |
| `{sample}.sig_timing.json.gz` | the same analysis as the viewer cache; picked up automatically by the interactive HTML, which gains a signature toggle and an activity-over-time curve |

**Reading `ratio`.** It is `(sig width / clock width) x (D_sig / D_clock)` over one interval —
above 1 means the signature deposited mutations faster than the clock there, below 1 slower.
It is a density over the `[t0, t1]` span, not a value at a point. Compare it against that
signature's `genome_ratio`, not against 1: a signature twice as abundant everywhere scores
2 everywhere without being time-structured at all.

**Reading `ok`.** False means the segment's polytope is too wide at that interval for the
value to be read on its own. Those rows are excluded from the aggregate and have no `ess`.
Filter to `ok` before pooling.

Cost: one full re-time per active signature, so expect minutes per sample rather than
seconds. It needs signature exposures on the genome; without them Tau falls back to
hard-assignment fractions.

## Interactive viewer

`tau run --interactive` writes a **self-contained interactive HTML** (`{sample}.overview_interactive.html`)
that reproduces the static overview as a fully interactive page. It inlines all of its
JavaScript and data, so it opens offline in any browser and can be emailed or shared as a
single file — no server, no install on the other end.

It is the genome-wide timing overview (filled per-segment stack coloured by copy number,
the multiplicity-cluster track, copy-number legend, WGD/PGD event lines, and the
timepoint histogram), made clickable:

- **Click a segment** → its mutation count + multiplicity histogram, VAF distribution
  (with the expected-VAF lines for each multiplicity), the SBS96 mutational spectrum with a
  signature-weighting selector (e.g. clock vs SBS17), a spatial *rainfall / VAF-vs-position*
  panel, and a table of **all route (polytope) solutions** sorted by fit (`dist_rel`).
- **Expand a route** → its evolutionary **tree with a slider per free variable**: dial the
  knobs to walk the route's polytope of valid solutions while the tree redraws live;
  WGD/PGD times are drawn as landmark lines so you can read each gain against them.
- **Click a multiplicity cluster** (top track) → the cluster's aggregate VAF / multiplicity /
  spectrum, its timing, and a scrollable list of member segments.
- **Compare** → Shift-click (or box-select) several segments to overlay their VAF and SBS96
  spectra with a side-by-side stats table.
- **Browse** → a *VAF distribution by CN state* section (pooled across the sample), plus a
  chromosome zoom and copy-number filters; the cluster track zooms in sync with the segments.

### Generating it on its own

To (re)build the viewer for outputs you already have:

```bash
python -m tau.viz \
  --genome    SAMPLE.genome.pkl.gz \
  --clustered SAMPLE.clustered_genome.pkl.gz \
  --clusters  SAMPLE.time_clusters.tsv \
  --out       SAMPLE.overview_interactive.html \
  --sample    SAMPLE
# after install, the `tau-viz` console script is equivalent.
```

Or from Python:

```python
from tau.viz import genome_to_html, load_genome
import pandas as pd
genome     = load_genome("SAMPLE.genome.pkl.gz")            # all individual segments
clustered  = load_genome("SAMPLE.clustered_genome.pkl.gz")  # drives the cluster track
cluster_df = pd.read_csv("SAMPLE.time_clusters.tsv", sep="\t")
genome_to_html(genome, cluster_df=cluster_df, clustered_genome=clustered,
               output_html="SAMPLE.overview_interactive.html", sample="SAMPLE")
```

The per-route tree explorer is precomputed at build time and needs SageMath (already in the
pixi environment). It adds a little time per sample; pass `tree_top_k=0` (or `--tree-top-k 0`)
to skip it for a faster, Sage-free viewer. The viewer requires `plotly`, which is included in
the pixi environment.

## Pipeline Overview

1. **Preprocessing** (`tau.preprocessing`): Load SNVs from VCF, extract SBS96 context, apply COSMIC signature weights
2. **Multiplicity estimation** (`tau.core`): EM algorithm for estimating SNV multiplicity mixture proportions
3. **Timing** (`tau.timing`): Load Sage solutions and constraint matrices, solve timing equations via grid sampling
4. **Clustering** (`tau.clustering`): EM-based clustering of timing estimates, classify events as WGD/PGD/SCA
5. **Visualization** (`tau.plotting`): Genome-wide timing visualization

## Environment Variables

You can override default data file paths using environment variables:

- `TAU_ROUTES_SAGE`: Path to Sage solutions file
- `TAU_MATRICES_H5`: Path to constraint matrices HDF5 file

## Citation

If you use Tau in your research, please cite:

[Citation to be added]

## License

MIT License - see LICENSE file for details.

