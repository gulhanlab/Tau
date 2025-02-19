#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.stats import betabinom
import argparse

def calculate_prob_cn_beta_binomial(df, dispersion=150):
    # Cache beta-binomial probabilities for common parameters to save computation time
    beta_binom_cache = {}

    def get_beta_binom_pmf(x, n, a, b):
        """Fetch or compute beta-binomial PMF to reduce redundant calculations."""
        key = (x, n, a, b)
        if key not in beta_binom_cache:
            beta_binom_cache[key] = betabinom.pmf(x, n, a, b)
        return beta_binom_cache[key]

    def cn_likelihood(row):
        # Check for missing or invalid data early
        if row.isna().any() or row['nalt'] < 0 or row['nref'] < 0 or row['major_cn'] < 0 or row['minor_cn'] < 0:
            return -1, -1, -1

        nalt = int(row['nalt'])
        nref = int(row['nref'])
        total_reads = nalt + nref
        total_cn = int(row['total_cn'])
        major_cn = int(row['major_cn'])
        purity = float(row['purity'])

        # Define possible CN states
        cn_states = np.arange(1, major_cn + 1)

        # Calculate likelihood for each CN state
        likelihoods = []
        for y in cn_states:
            # Expected probability of alternate reads for CN state `y`
            p = purity * y / total_cn

            # Adjust alpha and beta based on the mean probability `p`
            a = p * dispersion
            b = (1 - p) * dispersion  # Adjust `b` for failure likelihood

            # Calculate beta-binomial likelihood for observing `nalt` alternate reads
            likelihood = get_beta_binom_pmf(nalt, total_reads, a, b)
            likelihoods.append(likelihood)

        # Determine the highest likelihood and corresponding CN state
        max_likelihood_idx = np.argmax(likelihoods) if len(likelihoods) > 0 else None
        if max_likelihood_idx is None:
            return -1, -1, -1

        return likelihoods[max_likelihood_idx], cn_states[max_likelihood_idx], likelihoods

    # Apply the cn_likelihood function to each row in df
    df[['max_likelihood', 'best_cn', 'likelihoods']] = df.apply(cn_likelihood, axis=1, result_type="expand")

    return df

def categorize(sample, input_file, output_file, vaf_scale=None, vaf_scale_small=None,
               scale_margin=0.4, scale_margin_small=0.2, undefined_max=0.15,
               likelihood_threshold=0.015, dispersion=30):

    # Load the data
    if not input_file:
        raise FileNotFoundError(f"Input file not found: {input_file}")
    df = pd.read_csv(input_file, delimiter='\t')

    # Use the beta-binomial version of calculate_prob_cn
    df = calculate_prob_cn_beta_binomial(df, dispersion)

    df['categ'] = 'undefined'

    # Calculate CCF
    df['CCF'] = df['vaf'] * df['total_cn'] / df['best_cn']
    df.loc[(df['best_cn'] <= 0) | (df['best_cn'].isna()), 'CCF'] = np.nan

    # Categorize based on clonal and subclonal CCF and likelihood
    df.loc[df['max_likelihood'] >= likelihood_threshold, 'categ'] = 'clonal'
    df.loc[(df['CCF'] < 1) & (df['max_likelihood'] < likelihood_threshold), 'categ'] = 'subclonal'

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
    if undefined_frac > undefined_max and vaf_scale is None and vaf_scale_small is None:
        best_scale = 1
        scales = np.linspace(1 - scale_margin, 1 + scale_margin, num=5)
        for scale in scales:
            df_tmp = categorize(sample, input_file, output_file, vaf_scale=scale, dispersion=dispersion)
            undefined_frac_this = (df_tmp['categ'] == 'undefined').mean()
            if undefined_frac_this < undefined_frac and (df_tmp['categ'] == 'clonal').sum() > 0:
                best_scale = scale
                undefined_frac = undefined_frac_this
        df['scale'] = best_scale

    # Check undefined fraction in the final result
    undefined_frac = (df['categ'] == 'undefined').mean()
    if undefined_frac > undefined_max:
        print(f"Warning: Undefined fraction remains high ({undefined_frac:.2%}) even after scaling.")

    # Save the updated data
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Categorization step completed successfully for sample {sample}. Output saved to {output_file}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Categorize script for Tau workflow")
    parser.add_argument("sample", help="Sample name")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")
    parser.add_argument("--vaf_scale", type=float, default=None, help="VAF scaling factor")
    parser.add_argument("--vaf_scale_small", type=float, default=None, help="Small VAF scaling factor")
    parser.add_argument("--scale_margin", type=float, default=0.4, help="Margin for scaling")
    parser.add_argument("--undefined_max", type=float, default=0.15, help="Maximum undefined fraction allowed")
    parser.add_argument("--likelihood_threshold", type=float, default=0.015, help="Likelihood threshold for categorization")
    parser.add_argument("--dispersion", type=float, default=30, help="Dispersion for beta-binomial calculation")
    args = parser.parse_args()
    categorize(
        args.sample, args.input_file, args.output_file, 
        vaf_scale=args.vaf_scale, vaf_scale_small=args.vaf_scale_small,
        scale_margin=args.scale_margin, undefined_max=args.undefined_max,
        likelihood_threshold=args.likelihood_threshold, dispersion=args.dispersion
    )
