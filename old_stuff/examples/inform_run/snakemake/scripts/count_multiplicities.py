import sys
#sys.path.append('../../../../package/')
sys.path.append("/n/data1/hms/dbmi/park/jbrew/Tau/Tau_github/package")
import tau
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample", help = "Sample name")
parser.add_argument("--likelihood_file", help = "input likelihood file")
parser.add_argument("--output", help = "output multiplicities file")
args = parser.parse_args()

sample=args.sample
likelihood_file=args.likelihood_file
output=args.output

multiplicities_df = tau.count_multiplicities(likelihood_file)

multiplicities_df.to_csv(output, sep='\t')
