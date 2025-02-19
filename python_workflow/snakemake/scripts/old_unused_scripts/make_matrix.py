import os
import numpy as np
import pandas as pd
import pysam
from collections import defaultdict

def flip_base(base):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return complement.get(base.upper(), base)

def flip_strand(combinations):
    flipped = []
    for comb in combinations:
        ref = comb[0]
        if ref in ['C', 'T']:
            flipped.append(comb)
        else:
            ref = flip_base(comb[0])
            alt = flip_base(comb[1])
            prime5 = flip_base(comb[2])
            prime3 = flip_base(comb[3])
            flipped.append(f"{ref}{alt}{prime5}{prime3}")
    return flipped

def make_type(ncontext=3, nstrand=1):
    components = ['A', 'C', 'G', 'T']
    base_in = ['C', 'T'] if nstrand == 1 else components
    types = []
    for ref in base_in:
        for alt in components:
            if ref != alt:
                for prime5 in components:
                    for prime3 in components:
                        types.append(f"{ref}{alt}{prime5}{prime3}")
    return sorted(types)

def read_vcf(vcf_file):
    vcf = pysam.VariantFile(vcf_file)
    records = []
    for rec in vcf.fetch():
        records.append({
            "chrom": rec.chrom,
            "pos": rec.pos,
            "ref": rec.ref,
            "alt": str(rec.alts[0])
        })
    return pd.DataFrame(records)

def convert_to_vector(vcf_data, ref_genome, types, ncontext=3, nstrand=1):
    count_vector = defaultdict(int)
    for _, row in vcf_data.iterrows():
        context = get_context(row['chrom'], row['pos'], ref_genome, ncontext)
        combined = f"{row['ref']}{row['alt']}{context[:1]}{context[-1:]}"
        if nstrand == 1:
            combined = flip_strand([combined])[0]
        if combined in types:
            count_vector[combined] += 1
    return [count_vector[snv] for snv in types]

def get_context(chrom, pos, ref_genome, ncontext):
    half_context = (ncontext - 1) // 2
    seq = ref_genome.fetch(chrom, pos - half_context - 1, pos + half_context)
    return seq.upper()

def make_matrix(vcf_dir, ref_genome, output_matrix, ncontext=3, nstrand=1, mode=None):
    types = make_type(ncontext, nstrand)
    matrix = []
    sample_ids = []

    for vcf_file in os.listdir(vcf_dir):
        if not vcf_file.endswith(".vcf"):
            continue

        vcf_path = os.path.join(vcf_dir, vcf_file)
        vcf_data = read_vcf(vcf_path)
        vector = convert_to_vector(vcf_data, ref_genome, types, ncontext, nstrand)
        matrix.append(vector)
        sample_ids.append(os.path.splitext(vcf_file)[0])

    matrix_df = pd.DataFrame(matrix, columns=types, index=sample_ids)
    matrix_df.to_csv(output_matrix)

# Example usage
# from pyfaidx import Fasta
# ref_genome = Fasta("path/to/reference_genome.fa")
# make_matrix("path/to/vcf_dir", ref_genome, "output_matrix.csv")

