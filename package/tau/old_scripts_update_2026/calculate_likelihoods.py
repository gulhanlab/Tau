import numpy as np
import pandas as pd
from pyfaidx import Fasta
from Bio.Seq import Seq
import argparse
import os
import sys

#def parse_info_field(info):
#    """Parse the INFO field into a dictionary."""
#    return dict(item.split("=") if "=" in item else (item, True) for item in info.split(";"))

def generate_96_types():
    """Generate the 96 trinucleotide mutation types (COSMIC-style)."""
    bases = ['A', 'C', 'G', 'T']  # Use uppercase to match the signature file
    mutations = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    types = [f"{b1}[{mut}]{b2}" for mut in mutations for b1 in bases for b2 in bases]
    return types

def ensure_signature_order(signatures):
    """Reorder the signatures DataFrame to match the standard 96 trinucleotide mutation types."""
    expected_types = generate_96_types()  # Uppercase types
    if not all(typ in signatures['Type'].values for typ in expected_types):
        raise ValueError("Signature file is missing some standard 96 trinucleotide types.")
    
    # Reorder signatures to match expected order
    signatures = signatures.set_index('Type').reindex(expected_types).reset_index()
    return signatures

def calc_llh(index_spectrum, signatures, exposures):
    """Calculate likelihood for a given mutation index."""
    likelihood = signatures.iloc[index_spectrum] * exposures.values.flatten()
    llh_rat = likelihood / likelihood.sum()
    ind_max = llh_rat.idxmax()
    return {
        "probs": llh_rat,
        "ind_max": ind_max,
        "sig_max": ind_max,  # Signature with max likelihood
        "max_val": llh_rat[ind_max]
    }

def likelihoods_per_index(signatures, exposures):
    """Calculate likelihoods for each index in the mutation spectrum."""
    #print("calling likelihoods_per_index")
    #print(signatures.head())
    length_spectrum = signatures.shape[0]
    df_llhs_index = pd.DataFrame(0.0, index=range(length_spectrum), columns=signatures.columns)
    #print('df_llhs_index:')
    #print(df_llhs_index)
    sig_max_vec = []
    llh_max_vec = []

    for ind in range(length_spectrum):
        llhs = calc_llh(ind, signatures, exposures)
        df_llhs_index.loc[ind] = llhs["probs"]
        sig_max_vec.append(llhs["sig_max"])
        llh_max_vec.append(llhs["max_val"])

    df_llhs_index['sig_max'] = sig_max_vec
    df_llhs_index['max_val'] = llh_max_vec
    return df_llhs_index

def get_context_96(df, ref_genome_path, ref_col="ref", alt_col="alt"):
    """Assign context and 96-based index to mutations."""
    #print("calling get_context_96")
    ref_genome = Fasta(ref_genome_path)
    types = generate_96_types()
    #print(types)
    type_index_map = {type_str: idx for idx, type_str in enumerate(types)}
    #print(type_index_map)
    #print("type_index_map")
    #print(type_index_map)

    context_list = []
    index_spectrum_list = []

    for _, row in df.iterrows():
        chrom = str(row['chrom'])
        pos = int(row['pos'])
        ref = row[ref_col]
        alt = row[alt_col]

        # Get the surrounding trinucleotide context
        start = pos - 1
        end = pos + 1
        try:
            context_seq = ref_genome[chrom][start - 1:end].seq
        except KeyError:
            context_seq = "NNN"

        if not len(context_seq):
            print(chrom, pos, ref, alt)

        # Flip strand if needed
        if ref in ["A", "G"]:
            ref_flipped = str(Seq(ref).complement())
            alt_flipped = str(Seq(alt).complement())
            context_seq_flipped = str(Seq(context_seq).reverse_complement())
            ref, alt, context_seq = ref_flipped, alt_flipped, context_seq_flipped

        before = context_seq[0]
        after = context_seq[2]
        #ref, alt = ref, alt
        type_vec = f"{before}[{ref}>{alt}]{after}"
        #print(type_vec)
        #print(type_vec)

        index_spectrum = type_index_map.get(type_vec, None)
        #print(index_spectrum)

        context_list.append(context_seq)
        index_spectrum_list.append(index_spectrum)

    df['context'] = context_list
    df['index_spectrum'] = index_spectrum_list
    return df

def sig_likelihoods(sample, input_file=None, exposure_file=None, signature_file=None,
                    state=None, best_cn=None, df=None, ref_genome_path=None, output_file=None, sample_column='aliquot_id'):
    
    sample_id = sample

    print("sample:",sample_id)

    # Load exposures
    df_exposure = pd.read_csv(exposure_file, sep="\t")
    #df_exposure = df_exposure.rename(columns={'aliquot_id': 'sample_id', 'sample_id': 'icgc_specimen_id'})

    df_exposure = df_exposure.rename(columns={sample_column : 'unique_sample_id_column'})

    # Load signatures and ensure the correct order
    catalog = pd.read_csv(signature_file)
    catalog = ensure_signature_order(catalog)
    signames = list(set(catalog.columns[1:]) & set(df_exposure.columns))  # Exclude the 'Type' column
    #print("signames:", signames)

    # Filter exposures for the current sample
    exposures = df_exposure.query('unique_sample_id_column == @sample_id')[signames]

    # Filter signatures for non-zero exposures
    exposures = exposures.loc[:, (exposures > 0).values.flatten()]
    signatures = catalog.loc[:, exposures.columns]

    if (state is None) ^ (best_cn is None):
        sys.exit('Provide state and best_cn together or set both to None')

    if df is None:
        df = pd.read_csv(input_file, sep='\t')

    # Get context for mutations
    df = get_context_96(df, ref_genome_path)

    # Filter by state and best_cn if provided
    if state is not None and best_cn is not None:
        df['state'] = df.apply(lambda row: f"{row['major_cn']}_{row['minor_cn']}", axis=1)
        df = df[(df['state'] == state) & (df['best_cn'] == best_cn)]

    # Ensure signature order matches
    #signatures = ensure_signature_order(signatures)

    # Calculate likelihoods per mutation spectrum index
    df_llh = likelihoods_per_index(signatures, exposures)
    #print("df_llh complete:")
    #print(df_llh.head())
    #print("df current state:")
    #print(df.head())
    #print(df['index_spectrum'])
    df['index_spectrum'] = df['index_spectrum'].astype(int)
    # Merge `df` with `df_llh`
    df = df.merge(
        df_llh,
        how="left",
        left_on="index_spectrum",
        right_index=True
    )

    # Ensure all signature likelihood columns are present
    for sig in signatures.columns:  # Exclude 'Type' column
        if sig not in df.columns:  # Check if the likelihood column for this signature exists
            df[sig] = 0  # Add the missing column with 0 values

    # Rename columns to explicitly have "_llh" suffix for signature likelihoods
    df.rename(columns={sig: f"{sig}_llh" for sig in signatures.columns}, inplace=True)

    # Assign maximum likelihood signature and value
    #df['sig_max'] = df_llh.loc[df['index_spectrum'], 'sig_max'].values
    #df['max_val'] = df_llh.loc[df['index_spectrum'], 'max_val'].values 

    # Save to file
    if output_file: 
        if state is not None and best_cn is not None:
            output_path = f"{output_file}_state_{state}_cn_{best_cn}.txt"
        else:
            output_path = output_file
        df.to_csv(output_path, sep='\t', index=False)
        print(f"Processed file saved to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate signature likelihoods.")
    #parser.add_argument("--vcf", required=True, help="Input annotated VCF file.")
    parser.add_argument("--sample", required=True, hekp ="Sample name")
    parser.add_argument("--exposure", required=True, help="Exposure file.")
    parser.add_argument("--signature", required=True, help="Signature probabilities.")
    parser.add_argument("--output", required=True, help="Output likelihood file.")
    parser.add_argument("--normalize_file", required=True, help="Normalized file with mutation information")
    parser.add_argument("--ref_path", default='/n/data1/hms/dbmi/park/jbrew/ref/hg19_decoy/human_g1k_v37_decoy.fasta', help="Path to indexed reference genome")
    args = parser.parse_args()
 
    # Call the sig_likelihoods function
    sig_likelihoods(
        sample=sample,
        input_file=args.normalize_file,
        exposures=args.exposure,
        signatures=args.signature,
        ref_genome_path=args.ref_path,
        output_file=args.output
    )
