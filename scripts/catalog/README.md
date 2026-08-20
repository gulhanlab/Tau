# PCAWG catalog

A browsable, static website of Tau's interactive per-sample viewers, with a searchable
index. Everything is plain HTML/JS — no server-side code, no build step, no dependencies
at serve time.

## Layout

```
catalog_v2/
├── index.html      # the catalog page — sortable/filterable table, metadata embedded inline
├── catalog.json    # the same metadata as a sidecar, for programmatic use
└── viewers/
    └── <tumour_type>/<sample>.overview_interactive.html
```

`index.html` embeds its metadata inline, so it works over `file://` as well as HTTP.
Each viewer is self-contained: all CSS, JS and data are inlined, nothing is fetched from
a CDN.

## Serving it

Any static host works — S3, GitHub Pages, nginx, an institutional web directory. There is
nothing to configure beyond serving the directory. To check locally:

```bash
cd catalog_v2 && python -m http.server 8000     # then open http://localhost:8000
```

## ⚠ The viewers are symlinks by default

To avoid duplicating ~17 GB on disk, `viewers/` contains **symlinks** into the run
directory. Copy with dereferencing, or the upload will contain broken links:

```bash
rsync -aL catalog_v2/ user@host:/var/www/tau-catalog/     # -L follows symlinks
# or
cp -rL catalog_v2 /path/to/upload
```

Alternatively rebuild with real files: `--link copy` (see below).

## Size

| | |
|---|---|
| samples | 2,703 |
| tumour types | 37 |
| viewers | 16.8 GB total; 5.5 MB median, 70.9 MB largest |
| index.html | ~1 MB |

The largest viewers are hypermutated, highly-rearranged genomes with many timed segments.
If total size is a problem, rebuilding the run with `--tree-top-k 0` drops the per-route
tree explorer and shrinks the viewers substantially, at the cost of that feature.

## Rebuilding

The viewers are produced by `tau run` itself, so the catalog is only an assembly step:

```bash
# 1. metadata + directory layout (fast; no re-rendering, no Sage needed)
python scripts/catalog/build_catalog_from_run.py \
    --samples paper/pcawg_run/samples_v2 \
    --cohort  paper/pcawg_run/cohort_v2 \
    --out     paper/pcawg_run/catalog_v2 \
    [--link copy]        # real files instead of symlinks

# 2. the index page
python scripts/catalog/build_catalog_index.py \
    --out paper/pcawg_run/catalog_v2 \
    --title "Tau — PCAWG catalog (2,703 samples)"
```

Metadata is read from the cohort tables (`tau aggregate` output), so the catalog cannot
disagree with the run it describes.

## Metadata fields

Per sample, in `catalog.json` and inline in `index.html`:

`uuid`, `tumor_type`, `ploidy`, `purity`, `n_mutations`, `n_segs_timed`, `n_events`,
`n_wgd`, `n_pgd`, `wgd`, `pgd`, `wgd_times`, `pgd_times`, `html`, `size_mb`.

Note: the `pgd` / `n_pgd` / `pgd_times` keys hold **ccPG** (cross-chromosomal punctuated
gain) events. The key names predate the rename and are kept so the existing index page
keeps working.

## Data note

The viewers show timing results derived from PCAWG somatic variant calls. They contain
per-sample mutation counts and copy-number profiles. Check the applicable data-access
terms before publishing this to a public URL.
