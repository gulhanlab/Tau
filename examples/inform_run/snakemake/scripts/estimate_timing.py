import sys
#sys.path.append('../../../../package/')
sys.path.append("/n/data1/hms/dbmi/park/jbrew/Tau/Tau_github/package")
import tau
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--multiplicities", help = "Sample name")
parser.add_argument("--output_tsv", help = "output data table of timing results")
parser.add_argument("--output_plot", help = "output plot of timing results")
args = parser.parse_args()

multiplicities=args.multiplicities
output_tsv=args.output_tsv
output_plot=args.output_plot

#calculating timing solutions (dictionary format, per chromosomal segment)
timing_results = tau.calculate_timing_solutions(multiplicities)

#processing data into a table
timing_df = tau.process_solutions(timing_results)
timing_df.write_csv(output_tsv, sep='\t')

#calculating breakpoints
breakpoints, all_segments = tau.calculate_breakpoints(timing_results)

#plotting timing results 
tau.plot_timing_results(all_segments, np.median(breakpoints), output_plot) 

#plotting trees?
