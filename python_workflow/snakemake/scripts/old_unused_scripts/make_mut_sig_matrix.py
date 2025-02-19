import argparse
import os
import pandas as pd
import numpy as np
from Bio import SeqIO

# Helper Functions
def flip_base(base):
    """Flip DNA base to its complement."""
    complement = {
        'a': 't',
        'c': 'g',
        'g': 'c',
        't': 'a'
    }
    return complement.get(base, base)

def flip_strand(seq):
    """Flip strand for a sequence."""
    return ''.join(flip_base(base) for base in reversed(seq))

def make_types(ncontext=3, nstrand=1):
    """Generate all possible mutation contexts."""
    components = ['a', 'c', 'g', 't']
    bases = ['c', 't'] if nstrand == 1 else components
    types = []
    for ref in bases:
        for alt in [b for b in components if b != ref]:
            for left in components:
                for right in components:
                    types.append(f"{ref}{alt}{left}{right}")
    types.sort()
    return types

def convert_seq_to_vector(context, ref_vector, alt_vector, types):
    """Convert mutation data to a vector of counts."""
    count_vector = np.zeros(len(types), dtype=int)
    combined = [f"{ref}{alt}{context[i-1]}{context[i+1]}" for i, (ref, alt) in enumerate(zip(ref_vector, alt_vector))]

    type_to_index = {t: i for i, t in enumerate(types)}
    for mutation in combined:
        if mutation in type_to_index:
            count_vector[type_to_index[mutation]] += 1
    return count_vector

def parse_vcf(vcf_file, ref_genome, ncontext=3):
    """Parse VCF file and compute mutation context."""
    mutations = []
    ref_genome = SeqIO.to_dict(SeqIO.parse(ref_genome, "fasta"))

    with open(vcf_file) as vcf:
        for line in vcf:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            chrom, pos, _, ref, alt = fields[:5]

            pos = int(pos)
            if chrom not in ref_genome:
                continue

            sequence = ref_genome[chrom].seq
            left = sequence[pos - ncontext // 2 - 1:pos - 1].lower()
            right = sequence[pos:pos + ncontext // 2].lower()
            context = f"{left}{ref}{right}"

            if ref not in "ACGT" or alt not in "ACGT":
                continue

            if ref in "AG":
                ref = flip_base(ref)
                alt = flip_base(alt)
                context = flip_strand(context)

            mutations.append((ref, alt, context))

    return mutations

def make_matrix(vcf_files, ref_genome, ncontext=3, nstrand=1):
    """Create mutation signature matrix from VCF files."""
    types = make_types(ncontext, nstrand)
    matrix = pd.DataFrame(0, index=types, columns=[os.path.basename(f).replace(".vcf", "") for f in vcf_files])

    for vcf_file in vcf_files:
        mutations = parse_vcf(vcf_file, ref_genome, ncontext)
        ref_vector, alt_vector, context_vector = zip(*mutations)
        count_vector = convert_seq_to_vector(context_vector, ref_vector, alt_vector, types)
        sample_name = os.path.basename(vcf_file).replace(".vcf", "")
        matrix[sample_name] = count_vector

    return matrix

def main():
    parser = argparse.ArgumentParser(description="Create mutation signature matrix.")
    parser.add_argument("--vcf_dir", required=True, help="Directory containing VCF files.")
    parser.add_argument("--ref_genome", required=True, help="Reference genome in FASTA format.")
    parser.add_argument("--output", required=True, help="Output path for the mutation signature matrix.")
    parser.add_argument("--ncontext", type=int, default=3, help="Number of bases for context (default: 3).")
    parser.add_argument("--nstrand", type=int, default=1, help="Number of strands to consider (default: 1).")
    args = parser.parse_args()

    # List VCF files
    vcf_files = [os.path.join(args.vcf_dir, f) for f in os.listdir(args.vcf_dir) if f.endswith(".vcf")]

    # Create mutation signature matrix
    matrix = make_matrix(vcf_files, args.ref_genome, args.ncontext, args.nstrand)

    # Save to file
    matrix.to_csv(args.output, sep=",")

if __name__ == "__main__":
    main()

