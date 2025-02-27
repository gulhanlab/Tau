#!/usr/bin/env Rscript

# Parse arguments from command line
args <- commandArgs(trailingOnly = TRUE)
sample <- args[1]
txt_file <- args[2]
vcf_file <- args[3]
output_file <- args[4]

# Required libraries
library(VariantAnnotation)
library(GenomicRanges)

# Function to annotate VCF with categories and weights
annotate_vcf <- function(txt_file, vcf_file, output_file) {
  
  # Read normalized data
  df <- read.delim(txt_file)
  df$state <- paste0(df$major_cn, '_', df$minor_cn)

  # Check if the input VCF exists
  if (!file.exists(vcf_file)) {
    stop(paste("VCF file not found:", vcf_file))
  }
  
  # Read the input VCF
  vcf <- VariantAnnotation::readVcf(vcf_file)

  # Generate mutation IDs to match
  mut_ids <- paste0(
    as.character(GenomicRanges::seqnames(vcf)), '_',
    GenomicRanges::start(vcf), '_',
    VariantAnnotation::ref(vcf), '_',
    as.character(unlist(VariantAnnotation::alt(vcf)))
  )
  inds <- match(mut_ids, paste0(df$chrom, '_', df$pos, '_', df$ref, '_', df$alt))

  # Annotate VCF info fields
  VariantAnnotation::info(vcf)$state <- df$state[inds]
  VariantAnnotation::info(vcf)$best_cn <- df$best_cn[inds]
  VariantAnnotation::info(vcf)$weight <- df$weight[inds]
  VariantAnnotation::info(vcf)$categ <- df$categ[inds]

  # Add metadata for new fields
  newInfo <- DataFrame(
    Number = c(1, 1, 1, 1),
    Type = c("String", "Integer", "String", "Float"),
    Description = c(
      "Major and minor copy number values (M_N)",
      "Copy number matching the VAF for clonal mutations",
      "Subclonal/clonal category",
      "Copy number weight"
    ),
    row.names = c("state", "best_cn", "categ", "weight")
  )
  VariantAnnotation::info(VariantAnnotation::header(vcf)) <- rbind(
    VariantAnnotation::info(VariantAnnotation::header(vcf)),
    newInfo
  )

  # Write updated VCF
  VariantAnnotation::writeVcf(vcf, file = output_file)
}

# Execute the function
annotate_vcf(txt_file, vcf_file, output_file)
