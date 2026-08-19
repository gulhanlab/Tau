# Timing analysis with Tau

- [Background](#background)
	- [Diagram representation](#diagram-representation)
	- [Matrix representation of diagrams](#matrix-representation-of-diagrams)
	- [Systematically compose the A matrices](#systematically-compose-the-A-matrices)
	- [Solving for time when there is no exact solution](#solving-for-time-when-there-is-no-exact-solution)
- [Usage and Installation](#usage-and-installation)
	- [Installation](#installation)
	- [Try it](#try-it)
	- [Step through it yourself](#step-through-it-yourself)
	- [Running on your own data](#running-on-your-own-data)
		- [Input formats](#input-formats)
		- [Output](#output)
		- [Common options](#common-options)
		- [Multiple samples](#multiple-samples)
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

Tau depends on [SageMath](https://www.sagemath.org/) (and Singular) for its route
solutions. `pip install` does **not** provide SageMath, so timing will fail — use
[pixi](https://pixi.sh), which builds a complete environment including Sage:

```bash
git clone https://github.com/gulhanlab/Tau.git
cd Tau
pixi install

# put tau (and Sage) on your PATH -- add this line to ~/.bashrc to make it permanent
export PATH="$PWD/.pixi/envs/default/bin:$PATH"
```

`pixi install` downloads a full SageMath stack — expect roughly **20 minutes** and about
6 GB on first use, longer on a slow connection. It is a one-off.

Check it worked:

```bash
tau --help
pytest tests/          # expect: 40 passed
```

`tau` now works from any directory, so you can run it wherever your data lives.

> Prefer not to change `PATH`? Every command below also works as `pixi run <cmd>` from
> inside the repo (e.g. `pixi run tau demo`), or from anywhere with
> `pixi run --manifest-path /path/to/Tau/pyproject.toml tau ...`.

## Try it

A small simulated dataset ships with Tau, so this needs no input files:

```bash
tau demo
```

It runs the full pipeline and prints the recovered event against the known truth:

```
26/26 segments timed | 1 event(s): WGD t=0.29
Ground truth: WGD @ t=0.30
```

Look through the output directory before going further — it contains every file a real
run produces. `DEMO.tau_events.tsv` is the answer (one row per detected event) and
`DEMO.overview_interactive.html` opens in a browser.

## Step through it yourself

`tau demo` is just `tau run` on data that ships with the package. Running it yourself is
the quickest way to see what Tau expects. The two input files are in the repo:

```bash
$ zcat tau/data/demo/DEMO.snv_table.tsv.gz | head -3
Chromosome  Position  Ref  Alt  Tumor_Ref_Count  Tumor_Alt_Count
1           43147441  A    T    48               10
1           38216075  A    T    11               4

$ head -3 tau/data/demo/DEMO.cn_table.tsv
Chromosome  Segment_Start  Segment_End  Major_CN  Minor_CN
1           1              125000000    2         2
1           125000000      249250621    2         2
```

One row per somatic SNV with its read counts, and one row per copy-number segment. That
is all Tau needs, plus the sample's purity:

```bash
tau run \
  --sample DEMO \
  --snv_table tau/data/demo/DEMO.snv_table.tsv.gz \
  --cn_table  tau/data/demo/DEMO.cn_table.tsv \
  --purity    0.8 \
  --signatures none \
  --output_dir demo_manual/
```

This reproduces `tau demo` exactly. `--signatures none` asks for equal-weight
mutations because no signature exposures ship with the demo; drop it and Tau will say
so and fall back to the same thing.

Now swap in your own three inputs and you are running Tau on real data.

## Running on your own data

```bash
tau run \
  --sample SAMPLE_ID \
  --snv_table snvs.vcf.gz \
  --cn_table cn.tsv \
  --purity 0.8 \
  --output_dir results/
```

Add clock-signature weighting (recommended — see [Background](#background)) with
`--exposures exposures.tsv --ref_fasta hg19.fa`.

### Input formats

Column names are matched case-insensitively, ignoring spaces and hyphens, and the
delimiter (tab, comma or whitespace) is detected — so tables from other tools
usually work unedited.

| Input | Required | Format |
|---|---|---|
| `--snv_table` | yes | Somatic SNVs as **a VCF or a plain table**. VCF: counts are read from INFO `t_alt_count`/`t_ref_count` or FORMAT `AD` of the tumour sample. Table: needs chromosome, position and both read counts (e.g. `Chromosome`, `Position`, `Tumor_Ref_Count`, `Tumor_Alt_Count`). `ref`/`alt` are optional, as is a precomputed `sbs96`/`context` column (which removes the need for `--ref_fasta`). |
| `--cn_table` | yes | Allele-specific copy number: chromosome, segment start/end, major and minor CN (e.g. `Chromosome`, `Segment_Start`, `Segment_End`, `Major_CN`, `Minor_CN`). Float CN is rounded; `chr` prefixes are stripped. |
| `--purity` | yes | Tumour purity, a float in `(0, 1]`. |
| `--exposures` | for signatures | Per-sample signature exposures: a sample-ID column plus one column per `SBS*` signature. |
| `--ref_fasta` | for signatures | Reference FASTA matching your SNV coordinates, used to derive trinucleotide context. |

Signature weighting defaults to `--signatures SBS1 SBS5` and turns itself on once it has
exposures and a context source; otherwise Tau warns and weights all mutations equally.
Pass `--signatures none` to ask for equal weighting explicitly. COSMIC definitions ship
with Tau.

### Output

One directory per sample. `{sample}` is the `--sample` value.

| File | Contents |
|---|---|
| `{sample}.tau_samples.tsv` | one row for the sample: purity, ploidy, SNV/segment counts, number of events |
| `{sample}.tau_events.tsv` | **the main result** — one row per detected event: time, classification (WGD/PGD/chrom_specific), genome fraction, chromosomes |
| `{sample}.tau_segments.tsv` | per segment: coordinates, CN state, mutation weight, assigned event, or why it was not timed |
| `{sample}.tau_routes.tsv` | per candidate gain ordering: free variables, fit distance, whether it was chosen |
| `{sample}.tau_gains.tsv.gz` | per individual gain: its time, one row per polytope solution |
| `{sample}.tau_multiplicities.tsv` | per segment: the SNV multiplicity distribution the timing is fitted to |
| `{sample}.tau_merged_*.tsv` | the same five tables at the pooled level, where segments sharing a CN state and time are combined |
| `{sample}.overview.png` / `.svg` | genome-wide overview: per-segment timing, allele trees, event histogram |
| `{sample}.overview_interactive.html` | self-contained interactive version of the overview (`--no-interactive` to skip) |
| `{sample}.genome.pkl.gz` | the full result object, for the Python API |
| `{sample}.clustered_genome.pkl.gz` | the pooled result object |

Times run from **0 (early)** to **1 (late)**.

### Common options

| Option | Default | Purpose |
|---|---|---|
| `--signatures` | `SBS1 SBS5` | clock signatures; `none` for equal weighting |
| `--signature-analysis` | off | also re-time under every signature above `--signature-threshold` (default `0.15`), giving activity-over-time tables |
| `--bootstrap_B` | `10` | bootstrap replicates; `1` is much faster |
| `--sex` | — | `male` corrects chrX/chrY for hemizygosity |
| `--subclonal-filter` | off | drop probable subclonal mutations |
| `--no-interactive` | — | skip the interactive HTML |
| `--tumor_type` | `NA` | label carried into the output tables |

Full list: `tau run --help`.

### Multiple samples

`tau run` handles one sample. Run it per sample (in parallel if you like), pointing each
at its own `--output_dir`, then combine:

```bash
tau aggregate results/ --out cohort/
```

`tau aggregate` searches recursively, so it picks up however you nested the directories.

## Citation

If you use Tau in your research, please cite:

[Citation to be added]

## License

MIT License - see LICENSE file for details.

