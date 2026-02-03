import sys
sys.path.append('../../package/')
import tau
import argparse

#run all of Tau at once, rather than snakemake step by step version?
parser = argparse.ArgumentParser()

parser.add_argument("-s", "--sample", help = "Sample name")
parser.add_argument("--snv", help = "SNV file") 
parser.add_argument("--cnv", help = "CNV file") 
parser.add_argument("--purity", help = "purity file")
parser.add_argument("--exposure", help = "exposure file")
parser.add_argument("--signature", help = "signatures file")

##INCOMPLETE
