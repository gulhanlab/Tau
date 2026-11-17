# paper/figures — one directory per manuscript figure

See `../README.md` for how to run everything. This file describes the conventions inside this tree.

```
manifest.py          figure -> panel letter -> output file (single source of truth)
run_all.py           run every panel script; --check lists each manifest panel and whether it exists
export_panels.py     copy the manifest panels into export/<main|supp>/<figure>/<letter>_<name>.*
_style.py            the type ladder (S.title 8 / S.label 7 / S.tick 6.5 / S.legend 6.5 / S.annot 6.5 / S.tiny 5 pt),
                     fonts (Helvetica), column widths, save(). Every panel calls apply() and uses fontsize=S.<role>.
_svgfix.py           SVG post-processing so the panels open correctly in Illustrator
_shared/             helpers used by more than one figure (gain swimmer; punctuated-gain loaders)
main/<figure>/       <letter>_<name>.py writes <letter>_<name>.{svg,png[,pdf]} next to itself
supp/<figure>/       same
data/<figure>/       the figure's cache tables (not in git) + build.py or build/*.py that makes them
```

- **The letter in a file name is the manuscript panel letter.** `main/burst/b_bursts_pile_at_wgd.py` draws
  Fig. 6b. Where one script draws several panels its name carries all the letters (`bc_timing.py`,
  `abc_solo.py`) or the manifest note names it.
- `alt_*.py` are alternatives that are not in the paper. `_*.py` are helpers.
- Every script can be run on its own from any directory and exposes `make(save=True)`, which returns the
  matplotlib figure.
- Panels are drawn at final size (mm) with editable text; they were placed into the page layouts by hand.
- Two subdirectories hold self-contained model code: `main/wgd_dynamics/wgd_timing_model/` (the
  time-to-WGD Weibull models behind Fig. 4c) and `main/signatures/signatures_arrows/` (the R scripts
  fitting the stepwise signature-activity model and drawing Fig. 7b,c).
