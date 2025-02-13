#!/usr/bin/env Rscript

# Parse arguments from command line
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]

# Normalize function
normalize <- function(input_file, output_file) {
  # Read input file
  df <- read.delim(input_file)
  
  # Initialize weight column
  df$weight <- 0
  
  # Ensure chromosome column is character
  df$chrom <- as.character(df$chrom)
  
  # Check if the sample is male (presence of chromosome Y)
  male <- "Y" %in% df$chrom
  
  # Assign weights to clonal mutations
  clonal_indices <- which(df$categ == "clonal")
  df$weight[clonal_indices] <- df$best_cn[clonal_indices] / df$total_cn[clonal_indices]
  
  # Adjust weights based on male/female chromosome content
  if (male) {
    autosomal_indices <- which(!(df$chrom %in% c("X", "Y")))
    df$weight[autosomal_indices] <- 2 * df$weight[autosomal_indices]
  } else {
    df$weight <- 2 * df$weight
  }
  
  # Replace infinite weights with zero
  df$weight[is.infinite(df$weight)] <- 0
  
  # Save the updated data
  write.table(df, file = output_file, sep = "\t", row.names = FALSE, quote = FALSE)
}

# Execute the normalize function with parsed arguments
normalize(input_file, output_file)
