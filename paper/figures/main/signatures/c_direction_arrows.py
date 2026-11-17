"""signatures figure (main), panel c — direction of signature-activity change across successive
pre-MRCA intervals, by signature x tumour type (arrow grid).

Wrapper over the R scripts in `signatures_arrows/` (relocated intact; see its README). Runs
    estimate_cluster_timing.R      clustered_signature_input/*.tsv -> cluster_timing_estimates_BIC.tsv
    plot_cluster_pancancer_arrows.R  -> cluster_timing_pancancer_arrows.{pdf,tsv}, ..._trends.tsv
in a scratch `signatures_arrows/run/` directory, with two edits applied on the fly to COPIES of her
scripts (the originals are never modified):
  * the `source('~/Dropbox/MSI/indels/common_ggplot.R')` line is dropped -- verified 2026-09-15 that
    nothing in the script comes from it (state_trend is defined in-file; all else is base/ggplot2)
  * her `signature_code_results_for_Joseph/` output prefix is made local
then copies the PDF to panels/d_direction_arrows.pdf and, where the R build has an SVG device, an
SVG alongside. Needs r-ggplot2 + r-reshape in the pixi env (added 2026-09-15).

Verified against her 2026-08-10 outputs: arrows table byte-identical (79 rows), 574/574 trend
classifications identical, 1120/1120 BIC fits identical (float noise only).

CAVEAT: `clustered_signature_input/` is a snapshot of Tau signature output from ~2026-08-10 that does
not pass through paper/data_tables or the donor dedup. Regenerating here reproduces the snapshot;
it does not refresh it. Rebuilding the input from the current tau_sig_timing table is a separate task.
"""
import os, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAIN = HERE / "signatures_arrows"
RUN = CHAIN / "run"
sys.path.insert(0, str(HERE.parents[3]))
from paper.config import panels_dir_of  # noqa: E402
from paper.figures._style import apply as _style_apply, S  # noqa: E402
_style_apply()

PANELS = panels_dir_of(__file__)
STEM = "c_direction_arrows"
RSCRIPT = HERE.parents[3] / ".pixi/envs/default/bin/Rscript"

PLOT_R = r"""
# ---- each arrow is ONE polygon in mm, from the geometry measured on the assembled figure panel c:
#   shaft width  w  = 0.077 + 1.80 f   (mm; f = fraction of the cell's samples)
#   shaft length    = 0.41 + 1.96 w
#   head length     = 0.20 + 2.16 w,  head half-width = 0.18 + 1.95 w,  tip 0.07 + 0.90 w beyond the shaft end
# centred on the same point her segments were centred on; cells are 9.51 x 8.92 mm.
CELL_W <- 9.51; CELL_H <- 8.92
DIRANG <- c(increasing = atan(0.45), flat = 0, decreasing = -atan(0.45))
arrow_poly <- function(xc, yc, dir, f, id) {
  # EXACT geometry of her arrows (measured on three of them; all scale with the shaft width w):
  #   w = 0.077 + 1.80 f;  shaft length L = 0.41 + 1.96 w (round-capped, centred on (xc, yc))
  #   head = a chevron of six points in the arrow frame, x measured from the shaft START:
  #     back corners (0.30 L, +-1.36 w), inner notch (0.90 L, 0), front corners (0.80 L, +-1.36 w), tip (1.40 L, 0)
  w <- 0.077 + 1.80 * f; L <- 0.41 + 1.96 * w; hw <- 1.36 * w
  ang <- DIRANG[dir]
  x0 <- -L / 2                                                           # shaft start in the arrow frame
  hx <- x0 + c(0.30, 0.90, 0.30, 0.80, 1.40, 0.80) * L; hy <- c(hw, 0, -hw, -hw, 0, hw)
  # shaft as a round-capped bar from x0 to x0 + L (the head covers its front end)
  th1 <- seq(pi/2, 3*pi/2, length.out = 9); th2 <- seq(-pi/2, pi/2, length.out = 9)
  sx <- c(x0 + (w/2) * cos(th1), x0 + L + (w/2) * cos(th2)); sy <- c((w/2) * sin(th1), (w/2) * sin(th2))
  rot <- function(px, py) { rx <- px * cos(ang) - py * sin(ang); ry <- px * sin(ang) + py * cos(ang)
                            data.frame(x = xc + rx / CELL_W, y = yc + ry / CELL_H) }
  rbind(cbind(id = paste0(id, "s"), dir = dir, rot(sx, sy)),
        cbind(id = paste0(id, "h"), dir = dir, rot(hx, hy)))
}
tip_x <- function(xc, dir, f) { w <- 0.077 + 1.80 * f; L <- 0.41 + 1.96 * w; xc + (0.90 * L) * cos(DIRANG[dir]) / CELL_W }
tip_y <- function(yc, dir, f) { w <- 0.077 + 1.80 * f; L <- 0.41 + 1.96 * w; yc + (0.90 * L) * sin(DIRANG[dir]) / CELL_H }
polys <- do.call(rbind, lapply(seq_len(nrow(a)), function(i) arrow_poly(a$xc[i], a$yb[i], as.character(a$dir[i]), a$frac[i], i)))
polys$dir <- factor(polys$dir, levels = c("increasing", "flat", "decreasing"))
dom$tipx <- tip_x(dom$xc, as.character(dom$dir), dom$frac); dom$tipy <- tip_y(dom$yb, as.character(dom$dir), dom$frac)

# ---- legend row above the grid: three arrows (f = 0.6) + a four-step thickness key ----
ytop <- length(sig_present) + 1.25
leg_txt <- c(decreasing = "decreasing (early-enriched)", flat = "flat", increasing = "increasing (late-enriched)")
leg_c <- c(decreasing = 0.56, flat = 4.75, increasing = 6.0)
lab_x <- c(decreasing = 1.02, flat = 5.2, increasing = 6.45)
legp <- do.call(rbind, lapply(names(leg_c), function(d) arrow_poly(leg_c[d], ytop, d, 0.6, paste0("leg_", d))))
legp$dir <- factor(legp$dir, levels = levels(polys$dir))
leg_lab <- data.frame(x = lab_x, y = ytop, lab = leg_txt)
keyf <- c(0.25, 0.5, 0.75, 1.0); kx0 <- 9.9; kc <- c(13.0, 14.1, 15.25, 16.45)
keyp <- do.call(rbind, lapply(seq_along(keyf), function(i) arrow_poly(kc[i], ytop, "flat", keyf[i], paste0("key", i))))
keyp$dir <- "flat"; keyp$dir <- factor(keyp$dir, levels = levels(polys$dir))
key_lab <- data.frame(x = c(kx0, kc + 0.42), y = ytop, lab = c("percentage of samples", keyf * 100))

cells <- unique(agg[, c("xc", "yc")])
p <- ggplot() +
  geom_tile(data = cells, aes(x = xc, y = yc), fill = "grey97", color = "grey85", linewidth = 0.3, width = 1, height = 1) +
  geom_polygon(data = polys, aes(x = x, y = y, group = id, fill = dir), color = NA) +
  geom_text(data = dom, aes(x = tipx + 0.08, y = tipy, label = paste0(pct, "%")),
            hjust = 0, vjust = 0.5, size = 7 / .pt, fontface = "bold", color = "grey15") +
  geom_polygon(data = legp, aes(x = x, y = y, group = id, fill = dir), color = NA) +
  geom_polygon(data = keyp, aes(x = x, y = y, group = id), fill = "grey15", color = NA) +
  geom_text(data = leg_lab, aes(x = x, y = y, label = lab), hjust = 0, vjust = 0.5, size = 7 / .pt, color = "grey15") +
  geom_text(data = key_lab, aes(x = x, y = y, label = lab), hjust = 0, vjust = 0.5, size = 7 / .pt, color = "grey15") +
  scale_fill_manual(values = COL, guide = "none") +
  scale_x_continuous(breaks = seq_along(tord), labels = tord) +
  scale_y_continuous(breaks = seq_along(sig_present), labels = sig_label[sig_present]) +
  coord_cartesian(xlim = c(0.5, length(tord) + 0.5), ylim = c(0.5, ytop + 0.5), expand = FALSE, clip = "off") +
  labs(x = NULL, y = NULL) +
  theme_bw(base_size = 7.5) +
  theme(panel.grid = element_blank(), panel.border = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 7.5), axis.text.y = element_text(size = 7.5),
        axis.ticks = element_blank(), plot.margin = margin(2, 60, 2, 2, unit = "pt"))
gt <- ggplotGrob(p)
pl <- gt$layout[gt$layout$name == "panel", ]
gt$widths[pl$l] <- grid::unit(length(tord) * CELL_W, "mm")
gt$heights[pl$t] <- grid::unit(ytop * CELL_H, "mm")
w_in <- grid::convertWidth(sum(gt$widths), "in", valueOnly = TRUE)
h_in <- grid::convertHeight(sum(gt$heights), "in", valueOnly = TRUE)
ggsave('cluster_timing_pancancer_arrows.pdf', gt, width = w_in, height = h_in)
ggsave('cluster_timing_pancancer_arrows.svg', gt, width = w_in, height = h_in)
message(sprintf("arrows: %.1f x %.1f mm", w_in * 25.4, h_in * 25.4))
"""


def _localised(src: Path) -> str:
    t = src.read_text()
    t = "\n".join(l for l in t.splitlines() if "common_ggplot.R" not in l)
    return t.replace("signature_code_results_for_Joseph/", "")


def make(save=True, estimate=True):
    RUN.mkdir(exist_ok=True)
    (RUN / "clustered_signature_input").unlink(missing_ok=True)
    os.symlink("../clustered_signature_input", RUN / "clustered_signature_input")
    if estimate:
        (RUN / "estimate_cluster_timing.R").write_text(_localised(CHAIN / "estimate_cluster_timing.R"))
        subprocess.run([str(RSCRIPT), "estimate_cluster_timing.R"], cwd=RUN, check=True)
    else:
        shutil.copy2(CHAIN / "cluster_timing_estimates_BIC.tsv", RUN)
    plot = _localised(CHAIN / "plot_cluster_pancancer_arrows.R")
    # Paper styling (as in the assembled figure, where the rounded segments are drawn as true
    # arrows in Illustrator): the data step is hers untouched; from `cells <- unique(` on, the plot is
    # rebuilt with ARROW POLYGONS (body + head, both scaling with the fraction), the legend drawn as a
    # row of the same arrows above the grid, no title/subtitle, 7-8 pt text, 180 mm wide.
    i0 = plot.index("cells <- unique(agg["); i1 = plot.index("ggsave(")
    i1 = plot.index("\n", i1) + 1
    plot = plot[:i0] + PLOT_R + plot[i1:]
    (RUN / "plot_cluster_pancancer_arrows.R").write_text(plot)
    subprocess.run([str(RSCRIPT), "plot_cluster_pancancer_arrows.R"], cwd=RUN, check=True)
    if save:
        for ext in ("pdf", "svg"):
            src = RUN / f"cluster_timing_pancancer_arrows.{ext}"
            if src.exists():
                shutil.copy2(src, PANELS / f"{STEM}.{ext}")
        for t in ("cluster_timing_pancancer_arrows.tsv", "cluster_timing_pancancer_trends.tsv",
                  "cluster_timing_estimates_BIC.tsv"):
            shutil.copy2(RUN / t, CHAIN / t)          # keep the chain's own tables current
        print(f"wrote {PANELS}/{STEM}.pdf" + (" (+ .svg)" if (PANELS / f'{STEM}.svg').exists() else ""))


if __name__ == "__main__":
    make()
