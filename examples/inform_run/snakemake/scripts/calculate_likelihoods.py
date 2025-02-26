import sys
#sys.path.append('../../../../package/')
sys.path.append("/n/data1/hms/dbmi/park/jbrew/Tau/Tau_github/package")
import tau
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample", help = "Sample name")
parser.add_argument("--preprocessed_file", help = "preprocessed input file")
parser.add_argument("--exposure", help = "exposures file")
parser.add_argument("--signature", help = "signatures file")
parser.add_argument("--output", help = "output file")
parser.add_argument("--ref_genome", help = "reference genome")
args = parser.parse_args()

sample=args.sample
preprocessed_file=args.preprocessed_file
exposure_file=args.exposure
signature_file=args.signature
output_file=args.output
ref_genome=args.ref_genome

signature_df = tau.sig_likelihoods(sample, 
        exposure_file = exposure_file,
        signature_file = signature_file,
        ref_genome_path = ref_genome)

signature_df.to_csv(output_file, sep='\t')
