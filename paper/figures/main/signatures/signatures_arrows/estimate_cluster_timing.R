# FINAL cluster-based per-sample state fit (window-overlap deconvolution +
# support-constrained greedy changepoint search), for every tumor type in
# clustered_signature_input and every signature with a *_ratio column.
#
# Model-selection criterion is true BIC: n*log(rss/n) + S*log(n), matching
# segment_signatures/estimate_segment_timing.R (so cluster- and segment-level
# fits use the same penalty convention). This supersedes the earlier flat
# per-state penalty (n*log(rss/n) + 1.5*S) used during development.
#   -> signature_code_results_for_Joseph/cluster_timing_estimates_BIC.tsv

library('reshape')

indir <- 'clustered_signature_input'
interval_min_clock <- 5; sample_min_clock <- 20
min_span <- 0.05; max_span <- 0.75; min_state_width <- 0.05; min_segments <- 5
state_penalty <- 1.0; support_tol <- 0.05; min_support_mut <- 10

fit_one <- function(molten, sid, sig) {
  wcol <- paste0(sig, "_raw_interval")
  allrows <- molten[molten$sample == sid & molten$variable == paste0(sig, "_ratio"), ]
  if (!nrow(allrows)) return(NULL)
  total_clock <- sum(allrows$clocklike_raw_interval, na.rm = TRUE)
  if (total_clock <= sample_min_clock) return(NULL)
  baseline <- sum(allrows[[wcol]], na.rm = TRUE) / total_clock

  span <- molten$clocklike_cumulative_end - molten$clocklike_cumulative_start
  sub <- molten[molten$sample == sid & molten$clocklike_raw_interval >= interval_min_clock &
                span > min_span & span <= max_span & molten$variable == paste0(sig, "_ratio"), ]
  sub$w <- sub[[wcol]]
  sub <- sub[is.finite(sub$w) & sub$w > 0 & is.finite(sub$value) & sub$value > 0, ]
  if (nrow(sub) < min_segments || !is.finite(baseline) || baseline <= 0) return(NULL)

  z <- log2(sub$value / baseline); w <- sub$w
  S0 <- sub$clocklike_cumulative_start; E0 <- sub$clocklike_cumulative_end; n <- nrow(sub)
  bp <- sort(unique(c(S0, E0))); dom_lo <- min(bp); dom_hi <- max(bp); cand <- bp[bp > dom_lo & bp < dom_hi]
  build_B <- function(cps) { edges <- sort(unique(c(dom_lo, cps, dom_hi))); lo <- edges[-length(edges)]; hi <- edges[-1]
    B <- matrix(0, n, length(lo)); for (i in seq_len(n)) { len <- E0[i]-S0[i]; ov <- pmin(E0[i],hi)-pmax(S0[i],lo); ov[ov<0]<-0; B[i,] <- ov/len }; list(B=B, lo=lo, hi=hi) }
  wls <- function(B) { M <- t(B) %*% (B * w); if (qr(M, tol=1e-8)$rank < ncol(B)) return(NULL)
    th <- tryCatch(solve(M, t(B) %*% (w*z)), error=function(e) NULL); if (is.null(th)) return(NULL); list(theta=th, rss=sum(w*(z-B%*%th)^2), M=M) }
  width_ok <- function(cps) all(diff(sort(unique(c(dom_lo, cps, dom_hi)))) >= min_state_width)
  crit <- function(rss, S) n * log(rss / n) + state_penalty * S * log(n)
  supported <- function(o, th) all(vapply(seq_along(o$lo), function(k) {
    ov <- which(S0 < o$hi[k] & E0 > o$lo[k]); if (!length(ov)) return(FALSE)
    zz <- z[ov]; ww <- w[ov]
    oh <- order(zz, decreasing = TRUE); hi <- zz[oh][which(cumsum(ww[oh]) >= min_support_mut)[1]]; if (is.na(hi)) hi <- min(zz)
    ol <- order(zz);                    lo <- zz[ol][which(cumsum(ww[ol]) >= min_support_mut)[1]]; if (is.na(lo)) lo <- max(zz)
    th[k] <= hi + support_tol && th[k] >= lo - support_tol }, logical(1)))

  base <- wls(build_B(numeric(0))$B); if (is.null(base)) return(NULL)
  sel <- numeric(0); cur <- crit(base$rss, 1)
  repeat {
    best_r <- Inf; best_cp <- NA
    for (cp in setdiff(cand, sel)) {
      if (!width_ok(sort(c(sel, cp)))) next
      o <- build_B(sort(c(sel, cp))); f <- wls(o$B); if (is.null(f)) next
      if (!supported(o, as.numeric(f$theta))) next
      if (f$rss < best_r) { best_r <- f$rss; best_cp <- cp }
    }
    if (is.na(best_cp) || crit(best_r, length(sel) + 2) >= cur - 1e-9) break
    sel <- sort(c(sel, best_cp)); cur <- crit(best_r, length(sel) + 1)
  }
  o <- build_B(sel); f <- wls(o$B); S <- length(o$lo)
  se <- if (n - S >= 1) sqrt(diag((f$rss / (n - S)) * solve(f$M))) else rep(NA_real_, S)
  fold <- as.numeric(f$theta)
  data.frame(tumor_type = allrows$tumor_type[1], sample = sid, signature = sig, n_segments = n,
             n_states = S, state = seq_len(S), lo = o$lo, hi = o$hi, width = o$hi - o$lo, baseline = baseline,
             log2_fold = fold, log2_fold_se = se, log2_fold_lo = fold - se, log2_fold_hi = fold + se,
             ratio = baseline * 2^fold)
}

files <- list.files(indir, pattern = 'sig_intervals_range_.*_cluster\\.tsv$', full.names = TRUE)
out <- list()
for (f in files) {
  clust <- read.delim(f, stringsAsFactors = FALSE)
  molten <- melt(clust, id.vars = colnames(clust)[!grepl('_ratio$', colnames(clust))])
  sigs <- sub('_ratio$', '', grep('_ratio$', colnames(clust), value = TRUE))
  for (sig in sigs) for (s in unique(clust$sample)) {
    r <- fit_one(molten, s, sig); if (!is.null(r)) out[[length(out) + 1]] <- r
  }
}
est <- do.call(rbind, out); rownames(est) <- NULL
write.table(est, 'signature_code_results_for_Joseph/cluster_timing_estimates_BIC.tsv', sep = "\t", quote = FALSE, row.names = FALSE)
fits <- unique(est[, c("tumor_type","sample","signature","n_states")])
message(sprintf("%d states across %d fits, %d tumor types, %d signatures",
                nrow(est), nrow(fits), length(unique(est$tumor_type)), length(unique(est$signature))))
for (sg in sort(unique(fits$signature))) message(sprintf("  %-7s: %d fits, median states %g",
  sg, sum(fits$signature==sg), median(fits$n_states[fits$signature==sg])))
