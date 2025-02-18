# tau/preprocessing/categorize.py
import pandas as pd
import numpy as np
from scipy.stats import betabinom

def categorize_data(df, sample=None, output_file=None, **kwargs):
    """
    Categorize mutations based on VAF and CN data.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        sample (str, optional): Sample name.
        output_file (str, optional): Path to save the output. If None, no file is written.
        **kwargs: Additional arguments for categorization.
    
    Returns:
        pd.DataFrame: Categorized DataFrame.
    """
    # Use the beta-binomial version of calculate_prob_cn
    df = calculate_prob_cn_beta_binomial(df, kwargs.get('dispersion', 30))

    df['categ'] = 'undefined'

    # Calculate CCF
    df['CCF'] = df['vaf'] * df['total_cn'] / df['best_cn']
    df.loc[(df['best_cn'] <= 0) | (df['best_cn'].isna()), 'CCF'] = np.nan

    # Categorize based on clonal and subclonal CCF and likelihood
    df.loc[df['max_likelihood'] >= kwargs.get('likelihood_threshold', 0.015), 'categ'] = 'clonal'
    df.loc[(df['CCF'] < 1) & (df['max_likelihood'] < kwargs.get('likelihood_threshold', 0.015)), 'categ'] = 'subclonal'

    # Handle negative or NaN values explicitly
    invalid_conditions = (
        (df['nalt'] < 0) |
        (df['nref_corrected'] < 0) |
        (df['vaf'] < 0) |
        (df['max_likelihood'].isna()) |
        (df['vaf'].isna()) |
        (df['best_cn'].isna()) |
        (df['total_cn'] <= 0)
    )
    df.loc[invalid_conditions, 'categ'] = 'undefined'

    # Handle scaling to reduce "undefined" mutations if necessary
    undefined_frac = (df['categ'] == 'undefined').mean()
    if undefined_frac > kwargs.get('undefined_max', 0.15) and kwargs.get('vaf_scale') is None and kwargs.get('vaf_scale_small') is None:
        best_scale = 1
        scales = np.linspace(1 - kwargs.get('scale_margin', 0.4), 1 + kwargs.get('scale_margin', 0.4), num=5)
        for scale in scales:
            df_tmp = categorize_data(df, sample, output_file=None, vaf_scale=scale, **kwargs)
            undefined_frac_this = (df_tmp['categ'] == 'undefined').mean()
            if undefined_frac_this < undefined_frac and (df_tmp['categ'] == 'clonal').sum() > 0:
                best_scale = scale
                undefined_frac = undefined_frac_this
        df['scale'] = best_scale

    # Check undefined fraction in the final result
    undefined_frac = (df['categ'] == 'undefined').mean()
    if undefined_frac > kwargs.get('undefined_max', 0.15):
        print(f"Warning: Undefined fraction remains high ({undefined_frac:.2%}) even after scaling.")

    # Save to file if output_file is provided
    if output_file:
        df.to_csv(output_file, sep='\t', index=False)
        print(f"Categorization step completed successfully for sample {sample}. Output saved to {output_file}")
    
    return df
