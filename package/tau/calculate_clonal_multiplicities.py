import pandas as pd
import argparse

def calculate_multiplicities(df):
    """
    Calculate clonal multiplicities (N1, N2, ..., major_cn, minor_cn) grouped by segment_id and sig_max.
    Exclude invalid best_cn entries (e.g., best_cn = -1).
    """
    # Filter out invalid best_cn entries
    valid_df = df[df['best_cn'] >= 1]

    # Group by segment_id and sig_max
    grouped = valid_df.groupby(['segment_id', 'sig_max'])

    # Initialize results list
    results = []

    for (segment_id, sig_max), group in grouped:
        # Prepare the row dictionary
        row = {'segment_id': segment_id, 'sig_max': sig_max}
        
        # Count N1, N2, ..., based on `best_cn`
        for cn in sorted(group['best_cn'].unique()):
            row[f'N{int(cn)}'] = (group['best_cn'] == cn).sum().astype(int)
        
        # Add major_cn and minor_cn (assuming they are consistent within the group)
        row['major_cn'] = group['major_cn'].iloc[0].astype(int)
        row['minor_cn'] = group['minor_cn'].iloc[0].astype(int)
        
        # Append the row to results
        results.append(row)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Fill missing N-columns with 0
    n_columns = [col for col in results_df.columns if col.startswith('N')]
    results_df[n_columns] = results_df[n_columns].fillna(0).astype(int)

    # Sort the results
    results_df = results_df.sort_values(['segment_id', 'sig_max']).reset_index(drop=True)
    
    return results_df

def process_likelihood_file(input_file=None, output_file=None, df=None):
    """
    Process a likelihood file to calculate clonal multiplicities and
    add minor/major allele copy number summaries.
    """
    # Load the likelihood file
    if df is None:
        df = pd.read_csv(input_file, sep="\t")

    # Calculate multiplicities
    df_result = calculate_multiplicities(df)

    columns_order = (
                ['segment_id', 'sig_max', 'major_cn', 'minor_cn'] +  # Fixed columns first
                    [col for col in df_result.columns if col.startswith('N')]  # All N* columns
                    )

    df_result = df_result[columns_order]

    # Save the results
    if output_file:
        df_result.to_csv(output_file, sep="\t", index=False)
        print(f"Results saved to {output_file}")

    return df_result

def main():
    parser = argparse.ArgumentParser(description="Process likelihood files for clonal multiplicities.")
    parser.add_argument("--input", required=True, help="Path to the input likelihood file.")
    parser.add_argument("--output", required=True, help="Path to the output file.")
    args = parser.parse_args()

    process_likelihood_file(args.input, args.output)

if __name__ == "__main__":
    main()
