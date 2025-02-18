# tau/analysis/calculate_clonal_multiplicities.py
import pandas as pd

def calculate_clonal_multiplicities(df, output_file=None):
    """
    Calculate clonal multiplicities (N1, N2, ..., major_cn, minor_cn) grouped by segment_id and sig_max.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        output_file (str, optional): Path to save the output. If None, no file is written.
    
    Returns:
        pd.DataFrame: DataFrame with clonal multiplicities.
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
    
    # Save to file if output_file is provided
    if output_file:
        results_df.to_csv(output_file, sep="\t", index=False)
        print(f"Results saved to {output_file}")
    
    return results_df
