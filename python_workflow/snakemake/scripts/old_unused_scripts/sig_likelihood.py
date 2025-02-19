import pandas as pd
import numpy as np
from scipy.optimize import nnls

# Main function
def calculate_likelihoods(matrix_file, exposure_file, signature_file, output_file):
    matrix = pd.read_csv(matrix_file, index_col=0)
    exposures = pd.read_csv(exposure_file, sep="\t")
    signatures = pd.read_csv(signature_file)

    likelihoods = []
    for sample in matrix.columns:
        sample_exposures = exposures[exposures["sample_id"] == sample]
        if sample_exposures.empty:
            continue

        active_signatures = signatures[sample_exposures.iloc[0] > 0]
        for context in matrix.index:
            nnls_result = nnls(active_signatures.values, matrix.loc[context, sample])
            likelihoods.append({
                "sample": sample,
                "context": context,
                "likelihoods": nnls_result[0].tolist()
            })

    pd.DataFrame(likelihoods).to_csv(output_file, index=False)
    print(f"Saved likelihoods to {output_file}")

if __name__ == "__main__":
    import sys
    matrix_file, exposure_file, signature_file, output_file = sys.argv[1:]
    calculate_likelihoods(matrix_file, exposure_file, signature_file, output_file)
