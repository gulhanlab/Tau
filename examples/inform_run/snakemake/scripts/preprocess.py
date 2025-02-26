import sys
#sys.path.append('../../../../package/')
sys.path.append("/n/data1/hms/dbmi/park/jbrew/Tau/Tau_github/package")
import tau
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample", help = "Sample name")
parser.add_argument("--snv", help = "SNV file")
parser.add_argument("--cnv", help = "CNV file")
parser.add_argument("--purity", help = "purity file")
parser.add_argument("--output", help = "output file")

args = parser.parse_args()

sample=args.sample
snv_file=args.snv
cnv_file=args.cnv
purity_file=args.purity
output_file=args.output

trim_df = tau.trim(sample, snv_file, cnv_file, purity_file)
categorize_df = tau.categorize(sample, df=trim_df)
revise_df, subclonal_CCF_df = tau.revise(sample, df=categorize_df)
normalize_df = tau.normalize(sample, df=revise_df)

normalize_df.to_csv(output_file, sep='\t')


