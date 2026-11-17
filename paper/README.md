# Reproducing the figures of the Tau paper

This directory holds every script behind the figures of

> Brew J., Park P.J., Gulhan D.C. *Complete timing of copy-number amplification histories reveals
> whole-genome duplication dynamics and coincident oncogene amplification bursts.*

It is organised so that the panels can be regenerated at three depths, depending on what data you have:

| level | you need | you can regenerate |
|---|---|---|
| 1. panels | the **data bundle** (derived tables, a few example objects) | every scripted panel |
| 2. figure caches | the bundle + the cohort's per-sample Tau outputs | the per-figure cache tables |
| 3. from scratch | PCAWG calls (controlled access) | the Tau runs, the tables, everything |

Level 1 is the supported, verified path: with the bundle in place, `run_all.py` reproduces the panels
used in the paper.

## Layout

```
paper/
├── config.py            all paths; the data location is overridable by environment variable
├── data_tables/         tables.py (typed loaders), SCHEMA.md, DATA_TABLES.md  -- the DATA itself is not in git
├── figures/
│   ├── manifest.py      figure -> panel -> script output (single source of truth)
│   ├── run_all.py       run every panel script; --check confirms every manifest panel was produced
│   ├── export_panels.py collect the panels into export/<main|supp>/<figure>/ named by panel letter
│   ├── _style.py        fonts, type sizes, column widths shared by all panels
│   ├── main/<figure>/   one script per panel, named <panel letter>_<name>.py; writes its output beside itself
│   ├── supp/<figure>/   same, for the supplementary figures
│   └── data/<figure>/   per-figure cache tables (not in git) and the build scripts that make them
├── batch_scripts/       per-sample Tau outputs -> composite tables in data_tables/
└── upstream/            analysis scripts that produced the frozen inputs (see upstream/PROVENANCE.md)
```

A script named `alt_*.py` draws an alternative that is not in the paper; a file starting with `_` is a
helper shared within its directory.

## Figure map

| paper | directory | panels |
|---|---|---|
| Fig. 1 | `figures/main/overview` | schematic, drawn by hand (no script) |
| Fig. 2 | `figures/main/examples` | a, b |
| Fig. 3 | `figures/main/punctuated` | a–f |
| Fig. 4 | `figures/main/wgd_dynamics` | a–c |
| Fig. 5 | `figures/main/landscape` | a, b |
| Fig. 6 | `figures/main/burst` | a–i |
| Fig. 7 | `figures/main/signatures` | a–c |
| Supp. 1 | `figures/supp/route_complexity` | a (b is a hand-drawn schematic) |
| Supp. 2 | `figures/supp/multiplicity_clustering` | a–c |
| Supp. 3 | `figures/supp/benchmarking_segment` | a–f |
| Supp. 4 | `figures/supp/underdetermined` | a, b |
| Supp. 5 | `figures/supp/indeterminacy` | a, b |
| Supp. 6 | `figures/supp/wgd_ccpg_benchmark` | a–e |
| Supp. 7 | `figures/supp/wgd_swimmer_others` | one panel |
| Supp. 8 | `figures/supp/gain_landscape` | five sheets |
| Supp. 9 | `figures/supp/loss_landscape` | a–c |
| Supp. 10 | `figures/supp/extra_oncogenes` | a, b |
| Supp. 11 | `figures/supp/burst_gallery` | one panel |
| Supp. 12 | `figures/supp/burst_characteristics` | a–c |
| Supp. 13 | `figures/supp/signature` | a–e |

`figures/manifest.py` lists the exact file behind every panel letter.

## Level 1: regenerate the panels

1. **Install.** From the repository root, `pixi install` (or `pip install -e .` into an environment
   with the dependencies in `pyproject.toml`). The editable install is what makes `import paper` and
   `import tau` work from any directory. Panels b and c of Fig. 7 additionally need R with `ggplot2`,
   `reshape` and `svglite` (included in the pixi environment).

2. **Get the data bundle** and unpack it at the repository root. It provides

   ```
   paper/data_tables/    tau_*.tsv(.gz), bench_*.tsv, sim/, ref/, frozen/, objects/, per_sample/
   paper/figures/data/   <figure>/ cache tables
   paper/figures/main/signatures/signatures_arrows/clustered_signature_input/
   ```

   To keep the tables somewhere else, set `TAU_PAPER_DATA=/path/to/data_tables`.

3. **Run.**

   ```bash
   python paper/figures/run_all.py            # all panels (~1 h on one core; --fast skips the three slowest)
   python paper/figures/run_all.py --only main/burst/      # one figure
   python paper/figures/main/burst/d_burst_density.py      # one panel
   python paper/figures/run_all.py --check    # confirm every manifest panel exists
   python paper/figures/export_panels.py      # collect into paper/figures/export/
   ```

   Each script writes `<name>.svg` and `<name>.png` (some also `.pdf`) next to itself. `--alt` also
   runs the `alt_*.py` alternatives.

   The scripts are deterministic: in a fixed environment, rerunning them gives pixel-identical output,
   and this is how they were checked against the panels placed in the paper. Those panels were rendered
   with Python 3.12, matplotlib 3.9.0, numpy 2.4.2, pandas 3.0.0 and scipy 1.13.1, with Nimbus Sans
   standing in for Helvetica. Under another matplotlib version or font the plotted data are the same but
   text is rasterised at slightly different positions, so images will not match pixel for pixel.

4. **Check the bundle** at any time with `python paper/data_bundle.py check`, which lists every data
   file the panel scripts read and whether it is present.

### What the panels are, and are not

The scripts produce the individual panels at their final size. The published figures were then
assembled in Illustrator, where panel letters were added and, in places, labels were moved or
legends rearranged by hand. The data shown in a panel is exactly the script's; the page layout is not
scripted. Fig. 1 and Supplementary Fig. 1b are schematics with no script. Supplementary Fig. 6a is a
frozen render of one simulated genome (`upstream/simulation/panel_A_overview_compare.py`).

Most panels are **table-driven**: they read only the composite tables. A few are **object-based** and
draw directly from a sample's Tau objects, which the bundle carries for exactly the samples shown:
Fig. 2a,b, Fig. 7a, Supp. 2, Supp. 3b, Supp. 4a. Supplementary Fig. 11 draws 24 frozen exemplars
(`figures/data/burst_gallery/exemplars.pkl`).

### Conventions that affect the numbers

- **Cohort.** 2,703 PCAWG tumour samples. Gain- and segment-level analyses use all of them. Sample-level
  prevalence and timing statistics (Fig. 3d,e, Fig. 4, Supp. 7) use one representative aliquot per
  donor and exclude seven prostate donors represented only by multiple metastases (n = 2,580);
  `tau_samples.is_representative_aliquot` carries the choice.
- **Routes.** Burst statistics use the best-fitting route of each segment
  (`load_gains(best_route_only=True)`).
- **Solutions.** Tau's output for an underdetermined route is a set of equally valid solutions, not a
  distribution. Classifications ask whether the solution set contains a pattern (e.g. a gain is
  WGD-linked if any solution places it within 0.20 of a WGD), never what fraction of solutions do.
- **Linkage windows.** WGD-linked: within 0.20 molecular time of a detected WGD; ccPG-linked: within 0.10
  of a ccPG and not WGD-linked.

## Level 2: rebuild the figure caches

`figures/data/<figure>/build.py` (or `build/*.py`) regenerates that figure's cache tables. The landscape
and burst builders need only the composite tables; the others also read the cohort's per-sample Tau
outputs and the upstream analysis outputs, located through

```
TAU_PCAWG_OUTPUTS   per-sample Tau outputs, <tumor_type>/<sample>/<sample>.genome.pkl.gz ...
TAU_SIM_OUTPUTS     the simulated cohort, SIM_<n>/...
TAU_ANALYSIS_DIR    working directory of the upstream analysis scripts
```

`upstream/PROVENANCE.md` maps each frozen input and cache to the script that produced it.

## Level 3: from the PCAWG calls

1. Run Tau on every sample: `upstream/pcawg_run/run_tau_sample.py` (consensus SNVs, allele-specific
   copy number and purity; SBS1+SBS5 weighting from MuSiCal exposures), then event detection with
   `upstream/pcawg_run/run_bottomup_final.py` and the per-signature reruns
   (`upstream/pcawg_run/setup_persig_runs.py`).
2. Build the composite tables: `batch_scripts/` (see its README).
3. Run the upstream analyses (`upstream/`) and the figure cache builders (Level 2).
4. Simulated cohort: `upstream/simulation/` (Snakefile + `simulate_sample.py`, configuration `config_v8.yaml`),
   exported with `batch_scripts/export_sim_chunk.py` and `build_ref_and_bench.py`.

PCAWG data are available under controlled access through the ICGC and dbGaP; the per-sample Tau
outputs derived from them are therefore not distributed here.
