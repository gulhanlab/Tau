"""signatures figure (main), panel b — per-sample signature activity over molecular time, SBS17 and APOBEC,
by tumour type (BIC state fits). One column per sample, y = molecular time (0 bottom -> 1 top), each
column piecewise-constant over the sample's fitted states, colour = log2(signature/clock ratio) clipped
at +-2; samples ordered left->right by fitted slope (early-enriched first); tumour-type panels with
>= MIN_SAMPLES samples, ordered by mean slope; panel widths proportional to sample count.

This is `signatures_arrows/plot_cluster_heatmap_bysig.R` on the same `cluster_timing_estimates_BIC.tsv`
that panel c (arrows) is drawn from, restricted to SBS17 + APOBEC -- the two heatmaps she placed side by
side in the assembled figure (counts match: Eso 52, Stomach 17,
Breast 31, Head 22, ...). Runs a localised COPY of her script (original untouched): the personal
`common_ggplot.R` source line dropped, output prefix made local, signature list and sizes set here, an SVG
emitted next to each PDF. Writes b_sbs17.{pdf,svg} and b_apobec.{pdf,svg}; the two are placed side by side
(same per-sample column width, so they share one scale).

CAVEAT (as for panel c): `clustered_signature_input/` is an Aug-2026 snapshot not routed through
paper/data_tables or the donor dedup.
"""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN = HERE / "signatures_arrows"
RUN = CHAIN / "run"
sys.path.insert(0, str(HERE.parents[3]))
from paper.config import panels_dir_of  # noqa: E402
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()
from c_direction_arrows import _localised  # noqa: E402

PANELS = panels_dir_of(__file__)
RSCRIPT = HERE.parents[3] / ".pixi/envs/default/bin/Rscript"
SIGS = {"SBS17": "b_sbs17", "APOBEC": "b_apobec"}
MIN_SAMPLES = 8
COL_IN = 0.055          # inches per sample column (both signatures)
PANEL_PAD_IN = 0.30     # per tumour-type panel (strip gap)


def make(save=True):
    if not (RUN / "cluster_timing_estimates_BIC.tsv").exists():
        import c_direction_arrows
        c_direction_arrows.make(save=False)           # produces the BIC table in run/
    t = _localised(CHAIN / "plot_cluster_heatmap_bysig.R")
    t = t.replace('sigs <- c("SBS17", "APOBEC", "SBS18", "SBS3", "SBS4", "SBS7")',
                  'sigs <- c("SBS17", "APOBEC")')
    t = t.replace("min_samples <- 8", f"min_samples <- {MIN_SAMPLES}")
    # panel order as in the assembled figure: mean-slope sort, except Cervix.SCC before Bladder.TCC
    # (she swapped those two by hand; they are adjacent in the sort)
    t = t.replace("tord <- names(sort(tmean, decreasing = TRUE))",
                  'tord <- names(sort(tmean, decreasing = TRUE))\n'
                  '  if (all(c("Bladder.TCC", "Cervix.SCC") %in% tord)) {\n'
                  '    i <- match("Bladder.TCC", tord); j <- match("Cervix.SCC", tord)\n'
                  '    if (i < j) tord[c(i, j)] <- tord[c(j, i)] }')
    # Geometry measured on the assembled figure:
    # 0.86 mm per sample column, 27.3 mm heatmap height, 0.8 mm between tumour-type panels, strip labels
    # 7.5 pt at 45 deg, y ticks 6 pt, "Molecular time" 7.5 pt, "samples" 6 pt, colourbar 1.8 x 17.8 mm with a
    # 6 pt rotated title (APOBEC plot only; the SBS17 plot has none). Panel sizes are forced through the
    # gtable so they are absolute, not a function of the device size.
    t = t.replace('title = sprintf("Cluster-based %s timing across tumor types (BIC state-fit)", sg)',
                  'title = NULL')
    t = t.replace('scale_y_continuous("molecular time  (0 early -> 1 late)"', 'scale_y_continuous("Molecular time"')
    t = t.replace('labs(x = "samples  (left = early-enriched / front-loaded)",', 'labs(x = "samples",')
    t = t.replace('name = expression(log[2]~"fold")', 'name = expression(log[2]~"signature/clock ratio")')
    t = t.replace("theme_bw(base_size = 11)", "theme_bw(base_size = 7.5)")
    t = t.replace('panel.grid = element_blank(), panel.spacing = unit(4, "pt"), strip.clip = "off",',
                  'panel.grid = element_blank(), strip.clip = "off",')
    t = t.replace('strip.background = element_rect(fill = "#f0efec", color = NA),',
                  'strip.background = element_blank(), plot.margin = margin(t = 30, r = 2, b = 2, l = 2, unit = "pt"), '
                  'panel.spacing = unit(0.8, "mm"), axis.title.x = element_text(size = 6, hjust = 0), '
                  'axis.title.y = element_text(size = 7.5), axis.text.y = element_text(size = 6), '
                  'axis.ticks.y = element_line(linewidth = 0.3), axis.ticks.length = unit(1, "mm"), '
                  'legend.title = element_text(size = 6, angle = 90), legend.text = element_text(size = 6), '
                  'legend.position = if (sg == "APOBEC") "right" else "none",')
    t = t.replace('strip.text = element_text(face = "bold", size = 7.5, angle = 90, hjust = 0)',
                  'strip.text = element_text(size = 7.5, angle = 45, hjust = 0, vjust = 0)')
    t = t.replace('panel.border = element_rect(color = "#c3c2b7"))',
                  'panel.border = element_rect(color = "#c3c2b7", linewidth = 0.3)) +\n'
                  '    guides(fill = guide_colourbar(title.position = "left", barheight = unit(17.8, "mm"), barwidth = unit(1.8, "mm")))')
    # absolute panel sizes via the gtable, then a device exactly as large as the table
    i0 = t.index("  nsamp <- length(unique(d$sample)); npan <- length(keep_t)")
    i1 = t.rindex("}")                                            # end of the per-signature loop
    t = t[:i0] + '''  ncols <- sapply(levels(d$panel), function(pn) length(unique(d$sample[d$panel == pn])))
  gt <- ggplotGrob(p)
  pl <- gt$layout[grepl("^panel", gt$layout$name), ]
  pl <- pl[order(pl$l), ]
  for (k in seq_len(nrow(pl))) gt$widths[pl$l[k]] <- grid::unit(ncols[k] * 0.86, "mm")
  gt$heights[unique(pl$t)] <- grid::unit(27.3, "mm")
  w_in <- grid::convertWidth(sum(gt$widths), "in", valueOnly = TRUE)
  h_in <- grid::convertHeight(sum(gt$heights), "in", valueOnly = TRUE)
  ggsave(sprintf('cluster_heatmap_%s_bytumor.pdf', sg), gt, width = w_in, height = h_in)
  ggsave(sprintf('cluster_heatmap_%s_bytumor.svg', sg), gt, width = w_in, height = h_in)
  message(sprintf("%s: %d panels, %d samples, %.1f x %.1f mm", sg, length(ncols), sum(ncols), w_in * 25.4, h_in * 25.4))
''' + t[i1:]
    (RUN / "plot_cluster_heatmap_bysig.R").write_text(t)
    subprocess.run([str(RSCRIPT), "plot_cluster_heatmap_bysig.R"], cwd=RUN, check=True)
    if save:
        import shutil
        for sg, stem in SIGS.items():
            for ext in ("pdf", "svg"):
                src = RUN / f"cluster_heatmap_{sg}_bytumor.{ext}"
                if src.exists():
                    shutil.copy2(src, PANELS / f"{stem}.{ext}")
            print(f"wrote {PANELS}/{stem}.pdf (+ .svg)")


if __name__ == "__main__":
    make()
