library('ggplot2'); library('scales')
source('~/Dropbox/MSI/indels/common_ggplot.R')

# One heatmap PER SIGNATURE from the cohort-wide CLUSTER BIC state fits:
# panels = tumor types, x = samples (ordered by timing slope), y = molecular
# time 0 bottom -> 1 top, fill = log2 fold (RdBu).
#   -> signature_code_results_for_Joseph/cluster_heatmap_<SIG>_bytumor.pdf
min_samples <- 8
sigs <- c("SBS17", "APOBEC", "SBS18", "SBS3", "SBS4", "SBS7")

est <- read.delim('signature_code_results_for_Joseph/cluster_timing_estimates_BIC.tsv')

slope_of <- function(d) { d <- d[order(d$lo), ]; x <- (d$lo+d$hi)/2; y <- d$log2_fold
  se <- pmax(d$log2_fold_se, 1e-3); if (nrow(d) == 1) return(0)
  W <- 1/se^2; X <- cbind(1, x); solve(t(X) %*% (X*W), t(X) %*% (W*y))[2] }

for (sg in sigs) {
  d <- est[est$signature == sg, ]
  if (!nrow(d)) next
  ns <- tapply(d$sample, d$tumor_type, function(s) length(unique(s)))
  keep_t <- names(ns)[ns >= min_samples]
  d <- d[d$tumor_type %in% keep_t, ]
  if (!length(keep_t)) next

  d$col <- NA_integer_; tmean <- c()
  for (tt in keep_t) {
    dt <- d[d$tumor_type == tt, ]
    sl <- sapply(split(dt, dt$sample), slope_of)
    ord <- names(sort(sl, decreasing = FALSE))   # early-enriched (lowest slope) on the left
    d$col[d$tumor_type == tt] <- match(d$sample[d$tumor_type == tt], ord)
    tmean[tt] <- mean(sl)
  }
  tord <- names(sort(tmean, decreasing = TRUE))
  d$panel <- factor(sprintf("%s (%d)", d$tumor_type, ns[d$tumor_type]),
                    levels = sprintf("%s (%d)", tord, ns[tord]))

  p <- ggplot(d, aes(xmin = col - 0.5, xmax = col + 0.5, ymin = lo, ymax = hi, fill = log2_fold)) +
    geom_rect() +
    facet_grid(. ~ panel, scales = "free_x", space = "free_x") +
    scale_fill_distiller(palette = "RdBu", direction = -1, limits = c(-2, 2),
                         oob = squish, name = expression(log[2]~"fold")) +
    scale_y_continuous("molecular time  (0 early -> 1 late)", expand = c(0, 0),
                       breaks = c(0, 0.25, 0.5, 0.75, 1), limits = c(0, 1)) +
    scale_x_continuous(expand = c(0, 0)) +
    labs(x = "samples  (left = early-enriched / front-loaded)",
         title = sprintf("Cluster-based %s timing across tumor types (BIC state-fit)", sg)) +
    theme_bw(base_size = 11) +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
          panel.grid = element_blank(), panel.spacing = unit(4, "pt"), strip.clip = "off",
          strip.background = element_rect(fill = "#f0efec", color = NA),
          strip.text = element_text(face = "bold", size = 7.5, angle = 90, hjust = 0),
          panel.border = element_rect(color = "#c3c2b7"))
  nsamp <- length(unique(d$sample)); npan <- length(keep_t)
  w <- max(6, min(20, 0.045 * nsamp + 0.55 * npan + 1.5))
  ggsave(sprintf('signature_code_results_for_Joseph/cluster_heatmap_%s_bytumor.pdf', sg), p, width = w, height = 4.8)
  message(sprintf("%-7s: %d tumor panels, %d samples -> cluster_heatmap_%s_bytumor.pdf", sg, npan, nsamp, sg))
}
