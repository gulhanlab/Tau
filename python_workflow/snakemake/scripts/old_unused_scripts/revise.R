#!/usr/bin/env Rscript

# Load required packages
library(cluster)

# Parse arguments from command line
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]
sample <- args[3]

# Hierarchical clustering helper function
hc <- function(x, k, d.meth = "euclidean", ...) {
  list(cluster = cutree(hclust(dist(x, method = d.meth), ...), k = k))
}

# Function to calculate clonal and subclonal distances
clon_sub_distance <- function(df, by_state = TRUE) {
  if (by_state) {
    df$clonal_cn_state <- paste0(df$major_cn, '_', df$minor_cn)
    states <- unique(df$clonal_cn_state)
  } else {
    states <- 'all'
    df$clonal_cn_state <- "all"
  }

  df$dist_clon <- NA
  df$dist_sub <- NA
  df$mean_count <- NA
  df$sd_count <- NA
  df$mean_count_sub <- NA
  df$sd_count_sub <- NA

  for (state in states) {
    if (all(is.na(df$best_cn[df$categ == "clonal" & df$clonal_cn_state == state]))) {
      next  # Skip invalid states
    }
    
    min_clonal_cn <- min(df$best_cn[df$categ == "clonal" & df$clonal_cn_state == state], na.rm = TRUE)
    inds <- which(df$best_cn == min_clonal_cn & df$categ == "clonal" & df$clonal_cn_state == state)
    
    mean_count <- mean((1 / df$vaf[inds]) / df$total_cn[inds], na.rm = TRUE)
    sd_count <- sd((1 / df$vaf[inds]) / df$total_cn[inds], na.rm = TRUE)
    mean_count_sub <- mean((1 / df$vaf[df$clonal_cn_state == state & df$categ == "subclonal1"]) /
                             df$total_cn[df$clonal_cn_state == state & df$categ == "subclonal1"], na.rm = TRUE)
    sd_count_sub <- sd((1 / df$vaf[df$clonal_cn_state == state & df$categ == "subclonal1"]) /
                         df$total_cn[df$clonal_cn_state == state & df$categ == "subclonal1"], na.rm = TRUE)
    inds_state <- which(df$clonal_cn_state == state)
    if (length(inds_state) > 10) {
      df$mean_count[inds_state] <- mean_count
      df$sd_count[inds_state] <- sd_count
      df$mean_count_sub[inds_state] <- mean_count_sub
      df$sd_count_sub[inds_state] <- sd_count_sub

      df$dist_clon[inds_state] <- abs((1 / df$vaf[inds_state]) / df$total_cn[inds_state] - df$mean_count[inds_state]) / df$sd_count[inds_state]
      df$dist_sub[inds_state] <- abs((1 / df$vaf[inds_state]) / df$total_cn[inds_state] - df$mean_count_sub[inds_state]) / df$sd_count_sub[inds_state]

      df$dist_clon[inds_state[df$best_cn[inds_state] > min_clonal_cn]] <- 0
    }
  }
  return(df)
}

# Function to cluster subclones
cluster_subclones <- function(df, sample, output_dir, min_vaf_diff = 0.05) {
  # Extract values for clustering
  subclonal_rows <- grepl("subclonal", df$categ)
  vals <- (1 / df$vaf[subclonal_rows]) / df$total_cn[subclonal_rows]
  
  # Filter valid values
  valid_indices <- which(is.finite(vals) & vals > 0)
  vals <- vals[valid_indices]
  
  if (length(vals) == 0) {
    stop("No valid data for clustering. Check VAF and total CN values.")
  }

  # Prepare data for clustering
  df_x <- data.frame(x = log2(vals + 1))

  # Perform clustering using clusGap
  gs <- cluster::clusGap(df_x, FUNcluster = hc, K.max = 5, B = 20, verbose = FALSE)
  best_k <- cluster::maxSE(gs$Tab[, "gap"], gs$Tab[, "SE.sim"], method = "Tibs2001SEmax")
  hclust <- hclust(dist(df_x))
  cluster_inds <- cutree(hclust, k = best_k)

  # Calculate mean values per cluster
  mean_vec <- numeric()
  for (j in 1:best_k) {
    mean_vec <- c(mean_vec, mean(vals[cluster_inds == j], na.rm = TRUE))
  }

  # Create subclonal CCF dataframe
  df_subclonal_CCF <- data.frame(CCFs_subclone = 1 / mean_vec, index = 1:length(mean_vec))
  if (nrow(df_subclonal_CCF) > 1) {
    df_subclonal_CCF <- df_subclonal_CCF[order(-df_subclonal_CCF$CCFs_subclone), ]
  }
  df_subclonal_CCF$cluster <- 1:nrow(df_subclonal_CCF)
  write.table(df_subclonal_CCF, paste0(output_dir, '/', sample, '_CCF_subclones.txt'), row.names = FALSE, sep = '\t', quote = FALSE)

  # Update `df$categ` with subclone clusters
  indices <- df_subclonal_CCF$cluster[match(cluster_inds, df_subclonal_CCF$index)]
  replacement_indices <- which(subclonal_rows)[valid_indices]  # Match valid rows
  if (length(indices) != length(replacement_indices)) {
    stop("Replacement length mismatch: check indices and subclonal categories.")
  }
  df$categ[replacement_indices] <- paste0("subclonal", indices)
  return(df)
}

# Main revise function
revise <- function(sample, input_file, output_file, n_sub_thresh = 20, distance_scale = 3) {
  df <- read.delim(input_file)

  has_subclone <- FALSE
  if (sum(grepl('subclonal', df$categ), na.rm = TRUE) > n_sub_thresh) {
    print('Clustering subclones (Step 1)')
    df <- cluster_subclones(df, sample, output_dir = dirname(output_file))
    has_subclone <- TRUE
  } else {
    df$categ[grepl('subclonal', df$categ)] <- 'undefined'
  }

  if (!has_subclone) {
    write.table(df, file = output_file, row.names = FALSE, sep = '\t', quote = FALSE)
    return(0)
  }

  df <- clon_sub_distance(df)

  # Recategorize subclonal1 to clonal if closer
  inds <- which(df$dist_clon < distance_scale * df$dist_sub & df$categ == "subclonal1" & df$max_likelihood > 0.02)
  if (length(inds) > 0) {
    df$categ[inds] <- 'clonal'
    print('Re-clustering subclones (Step 2)')
    df <- cluster_subclones(df, sample, output_dir = dirname(output_file))
    df <- clon_sub_distance(df)
  }

  # Recategorize undefined to subclonal1 if closer
  inds <- which(df$dist_clon > distance_scale * df$dist_sub & !grepl('subclonal', df$categ) & df$dist_clon > 0 & df$vaf < 1 / df$total_cn)
  if (length(inds) > 0) {
    df$categ[inds] <- 'subclonal1'
    print('Re-clustering subclones (Step 3)')
    df <- cluster_subclones(df, sample, output_dir = dirname(output_file))
    df <- clon_sub_distance(df)
  }

  if (sum(grepl('subclonal', df$categ), na.rm = TRUE) > n_sub_thresh) {
    print('Final subclone clustering (Step 4)')
    df <- cluster_subclones(df, sample, output_dir = dirname(output_file))
  }

  write.table(df, file = output_file, row.names = FALSE, sep = '\t', quote = FALSE)
  return(0)
}

# Execute the revise function with parsed arguments
revise(sample, input_file, output_file)
