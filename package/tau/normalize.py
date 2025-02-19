#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys

def normalize(input_file, output_file, df = None):
    # Read input file
    if df is None:
        df = pd.read_csv(input_file, sep='\t')
    
    # Initialize weight column
    df['weight'] = 0.0
    
    # Ensure chromosome column is string type
    df['chrom'] = df['chrom'].astype(str)
    
    # Check if the sample is male (presence of chromosome Y)
    male = 'Y' in df['chrom'].values
    
    # Assign weights to clonal mutations
    clonal_mask = df['categ'] == 'clonal'
    df.loc[clonal_mask, 'weight'] = (
        df.loc[clonal_mask, 'best_cn'] / df.loc[clonal_mask, 'total_cn']
    )
    
    # Adjust weights based on male/female chromosome content
    if male:
        # Autosomal chromosomes (not X or Y) have their weights doubled
        autosomal_mask = ~df['chrom'].isin(['X', 'Y'])
        df.loc[autosomal_mask, 'weight'] *= 2
    else:
        # All weights are doubled for female samples
        df['weight'] *= 2
    
    # Replace infinite weights with zero (due to division by zero)
    df['weight'] = df['weight'].replace([np.inf, -np.inf], 0.0)
    
    # Save the updated data
    if output_file:
        df.to_csv(output_file, sep='\t', index=False, float_format='%.6f')
        print(f"Normalize step completed successfully. Output saved to {output_file}")

    return df

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    normalize(input_file, output_file)
