import pandas as pd
import numpy as np
from scipy.optimize import nnls
import os

def reassign_signatures(matrix_state_path, exposure_file, catalog_file, output_dir):
    # Load data
    matrix_state = pd.read_csv(matrix_state_path)
    exposures = pd.read_csv(exposure_file, sep="\t")
    catalog = pd.read_csv(catalog_file)
    
    # Extract catalog signature names
    signature_names = [col for col in catalog.columns if col.startswith('SBS')]
    
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metadata
    clonalities = matrix_state['sample_id'].str.split(':').str[1]
    samples = matrix_state['sample_id'].str.split(':').str[0]
    categ_cns = matrix_state['sample_id'].str.split(':').str[2]
    states = matrix_state['sample_id'].str.split(':').str[3]
    
    # Filter for clonal data
    matrix_state['clonality'] = clonalities
    clonal_matrix = matrix_state[matrix_state['clonality'] == 'clonal']
    
    unique_samples = clonal_matrix['sample_id'].str.split(':').str[0].unique()
    
    # Reassign signatures for each sample
    for sample in unique_samples:
        reassigned_rows = []
        
        sample_rows = clonal_matrix[clonal_matrix['sample_id'].str.startswith(sample)]
        sample_exposures = exposures[exposures['sample_id'] == sample][signature_names]
        
        if sample_exposures.empty:
            print(f"Sample {sample} not found in exposure file. Skipping.")
            continue
        
        for _, row in sample_rows.iterrows():
            state_mutations = row.iloc[:96].values  # First 96 columns correspond to mutation counts
            total_snvs = np.sum(state_mutations)
            
            if total_snvs == 0:
                print(f"No mutations in state {row['sample_id']}. Skipping.")
                continue
            
            # Filter catalog and exposures for non-zero entries
            valid_signatures = [sig for sig in signature_names if sample_exposures[sig].values[0] > 0]
            filtered_catalog = catalog[valid_signatures]
            filtered_exposures = sample_exposures[valid_signatures].values.flatten()
            
            # Perform NNLS
            nnls_result = nnls(filtered_catalog.values, state_mutations)
            reassigned_exposures = nnls_result[0]
            
            # Format output row
            reassigned_row = {
                "sample_id": sample,
                "state": row['sample_id'],
                "total_snvs": total_snvs,
                **{f"{sig}_exposure": exp for sig, exp in zip(valid_signatures, reassigned_exposures)}
            }
            reassigned_rows.append(reassigned_row)
        
        # Save results for the sample
        if reassigned_rows:
            reassigned_df = pd.DataFrame(reassigned_rows)
            reassigned_df.to_csv(
                os.path.join(output_dir, f"{sample}_final_reassign.csv"),
                index=False
            )

# Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reassign Signatures Per State")
    parser.add_argument("--matrix_state", required=True, help="Path to mutation signature matrix state CSV")
    parser.add_argument("--exposure_file", required=True, help="Path to exposure file (TSV)")
    parser.add_argument("--catalog_file", required=True, help="Path to catalog file (CSV)")
    parser.add_argument("--output_dir", required=True, help="Directory to save reassigned signature files")
    
    args = parser.parse_args()
    
    reassign_signatures(
        matrix_state_path=args.matrix_state,
        exposure_file=args.exposure_file,
        catalog_file=args.catalog_file,
        output_dir=args.output_dir
    )
