# tau/preprocessing/normalize.py
import pandas as pd
import numpy as np

def normalize_data(df, output_file=None):
    """
    Normalize mutation weights based on CN and gender.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        output_file (str, optional): Path to save the output. If None, no file is written.
    
    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
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
    
    # Save to file if output_file is provided
    if output_file:
        df.to_csv(output_file, sep='\t', index=False, float_format='%.6f')
        print(f"Normalization step completed successfully. Output saved to {output_file}")
    
    return df
