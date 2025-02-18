# tau/analysis/calculate_likelihoods.py
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from Bio.Seq import Seq
import os
import sys

def calculate_likelihoods(df, sample=None, exposures=None, signatures=None, ref_genome_path=None, output_file=None):
    """
    Calculate signature likelihoods for mutations.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        sample (str, optional): Sample name.
        exposures (pd.DataFrame): Signature exposures.
        signatures (pd.DataFrame): Signature probabilities.
        ref_genome_path (str): Path to reference genome.
        output_file (str, optional): Path to save the output. If None, no file is written.
    
    Returns:
        pd.DataFrame: DataFrame with likelihoods.
    """
    # Get context for mutations
    df = get_context_96(df, ref_genome_path)

    # Filter by state and best_cn if provided
    if 'state' in df.columns and 'best_cn' in df.columns:
        df = df[(df['state'] == state) & (df['best_cn'] == best_cn)]

    # Calculate likelihoods per mutation spectrum index
    df_llh = likelihoods_per_index(signatures, exposures)

    # Merge `df` with `df_llh`
    df['index_spectrum'] = df['index_spectrum'].astype(int)
    df = df.merge(
        df_llh,
        how="left",
        left_on="index_spectrum",
        right_index=True
    )

    # Ensure all signature likelihood columns are present
    for sig in signatures.columns:
        if sig not in df.columns:
            df[sig] = 0  # Add missing columns with 0 values

    # Rename columns to explicitly have "_llh" suffix for signature likelihoods
    df.rename(columns={sig: f"{sig}_llh" for sig in signatures.columns}, inplace=True)

    # Save to file if output_file is provided
    if output_file:
        df.to_csv(output_file, sep='\t', index=False)
        print(f"Processed file saved to {output_file}")
    
    return df
