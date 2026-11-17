library('ggplot2'); library('grid')
source('~/Dropbox/MSI/indels/common_ggplot.R')

# FINAL pan-cancer directional summary, built from the CLUSTER BIC state-fit
# itself (cluster_timing_estimates_BIC.tsv), not from an independent linear
# regression on the raw intervals. Per (tumor_type, sample, signature): treat
# the fitted states as points (x = state midpoint, y = log2_fold, weight =
# 1/log2_fold_se^2) and fit a weighted trend through them; t = trend/se(trend)
# classifies increasing/flat/decreasing (|t| >= T_FLAT). A single-state fit
# (BIC found no changepoint) is "flat" by construction.
#   -> signature_code_results_for_Joseph/cluster_timing_pancancer_arrows.pdf/.tsv
#   -> signature_code_results_for_Joseph/cluster_timing_pancancer_trends.tsv

min_samples <- 8; T_FLAT <- 2
sig_levels  <- c("SBS17", "APOBEC", "SBS18", "SBS3", "SBS4", "SBS7")
sig_label   <- c("SBS17"="SBS17", "APOBEC"="APOBEC", "SBS18"="SBS18",
                 "SBS3"="SBS3 (HRD)", "SBS4"="SBS4 (smoking)", "SBS7"="SBS7 (UV)")
COL <- c(increasing = "#c0392b", flat = "#8a8a8a", decreasing = "#2a6fb0")

est <- read.delim('signature_code_results_for_Joseph/cluster_timing_estimates_BIC.tsv', stringsAsFactors = FALSE)

state_trend <- function(d) {
  d <- d[order(d$lo), ]
  if (nrow(d) == 1) return(data.frame(slope = 0, se = NA_real_, t = 0, n_states = 1L))
  x <- (d$lo + d$hi) / 2; y <- d$log2_fold
  se <- pmax(d$log2_fold_se, 1e-3); w <- 1 / se^2
  X <- cbind(1, x); M <- t(X) %*% (X * w)
  if (qr(M, tol = 1e-8)$rank < 2) return(data.frame(slope = 0, se = NA_real_, t = 0, n_states = nrow(d)))
  b <- solve(M, t(X) %*% (w * y)); V <- solve(M); se_b <- sqrt(V[2, 2])
  data.frame(slope = b[2], se = se_b, t = b[2] / se_b, n_states = nrow(d))
}

per <- do.call(rbind, by(est, list(est$tumor_type, est$sample, est$signature), function(d) {
  cbind(tumor_type = d$tumor_type[1], sample = d$sample[1], signature = d$signature[1], state_trend(d))
}, simplify = FALSE))
rownames(per) <- NULL
per$dir <- ifelse(per$t >= T_FLAT, "increasing", ifelse(per$t <= -T_FLAT, "decreasing", "flat"))
per <- per[per$signature %in% sig_levels, ]
write.table(per, 'signature_code_results_for_Joseph/cluster_timing_pancancer_trends.tsv',
            sep = "\t", quote = FALSE, row.names = FALSE)

agg <- as.data.frame(table(tumor_type = per$tumor_type, signature = per$signature, dir = per$dir),
                     stringsAsFactors = FALSE); names(agg)[4] <- "count"
tot <- aggregate(count ~ tumor_type + signature, agg, sum)
keep <- tot[tot$count >= min_samples, c("tumor_type", "signature")]
agg <- merge(agg, keep, by = c("tumor_type", "signature"))
write.table(agg[order(agg$tumor_type, agg$signature, agg$dir), ],
            'signature_code_results_for_Joseph/cluster_timing_pancancer_arrows.tsv',
            sep = "\t", quote = FALSE, row.names = FALSE)

mslope <- tapply(per$slope, per$tumor_type, mean, na.rm = TRUE)
tord <- intersect(names(sort(mslope, decreasing = TRUE)), unique(keep$tumor_type))
agg <- agg[agg$tumor_type %in% tord, ]
agg$xc <- match(agg$tumor_type, tord)
sig_present <- rev(sig_levels[sig_levels %in% unique(keep$signature)])
agg$yc <- match(agg$signature, sig_present)

yoff <- c(increasing = 0.27, flat = 0, decreasing = -0.27)
diag <- c(increasing = 0.45, flat = 0, decreasing = -0.45)
a <- agg[agg$count > 0, ]
a$cell <- paste(a$xc, a$yc)
tot_cell <- tapply(a$count, a$cell, sum)
a$frac <- a$count / tot_cell[a$cell]
a$pct  <- round(100 * a$frac)
h <- 0.04 + 0.42 * a$frac
a$ux <- h; a$uy <- h * diag[a$dir]
a$yb <- a$yc + yoff[a$dir]
a$x0 <- a$xc - a$ux; a$x1 <- a$xc + a$ux
a$y0 <- a$yb - a$uy; a$y1 <- a$yb + a$uy
dom <- do.call(rbind, by(a, a$cell, function(d) d[which.max(d$frac), ]))
a$dir <- factor(a$dir, levels = c("increasing", "flat", "decreasing"))

cells <- unique(agg[, c("xc", "yc")])
p <- ggplot() +
  geom_tile(data = cells, aes(x = xc, y = yc), fill = "grey97", color = "grey85", linewidth = 0.3, width = 1, height = 1) +
  geom_segment(data = a, aes(x = x0, y = y0, xend = x1, yend = y1, color = dir, linewidth = frac), lineend = "round") +
  geom_text(data = dom, aes(x = x1 + 0.03, y = y1, label = paste0(pct, "%")),
            hjust = 0, vjust = 0.5, size = 2.3, fontface = "bold", color = "grey15") +
  scale_color_manual(values = COL, name = "trend",
                     labels = c("increasing (late)", "flat", "decreasing (early)")) +
  scale_linewidth(range = c(0.1, 4), name = "fraction\nof samples", limits = c(0, 1), breaks = c(0.25, 0.5, 0.75, 1)) +
  scale_x_continuous(breaks = seq_along(tord), labels = tord, expand = expansion(add = 0.5)) +
  scale_y_continuous(breaks = seq_along(sig_present), labels = sig_label[sig_present], expand = expansion(add = 0.5)) +
  coord_equal(clip = "off") +
  guides(color = guide_legend(override.aes = list(linewidth = 2.2))) +
  labs(x = NULL, y = NULL,
       title = "Cluster-based signature timing: directional summary across cancer types (BIC state-fit)",
       subtitle = "per cell: red up = increasing/late, grey = flat, blue down = decreasing/early;  arrow length & thickness proportional to the fraction of that cell's samples;  % = dominant arrow's share") +
  theme_bw(base_size = 11) +
  theme(panel.grid = element_blank(), panel.border = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1), axis.ticks = element_blank(),
        plot.subtitle = element_text(size = 7.5, color = "grey35"))
ggsave('signature_code_results_for_Joseph/cluster_timing_pancancer_arrows.pdf', p, width = 11, height = 4.2)
message(sprintf("wrote state-based cluster arrow summary: %d fits, %d cells (n>=%d); inc/flat/dec = %d/%d/%d",
                nrow(per), nrow(cells), min_samples,
                sum(per$dir=="increasing"), sum(per$dir=="flat"), sum(per$dir=="decreasing")))
