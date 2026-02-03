#!/usr/bin/env python3
import sys
sys.path.insert(0, '/n/data1/hms/dbmi/park/jbrew/pcawg_run')
from tau_multiplicity import Segment, Genome
from tau_preprocessing import preprocess_sample
from tau_utils import order_t, pick_best_key, extract_mat, time_matrix, make_time_df
from tau_cluster import cluster_times, cluster_segment_multiplicities, make_clustered_genome
from tau_plot import plot_overview
import tau_time


import pandas as pd
from importlib import reload
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle, gzip

DEFAULT_ROUTES_SAGE = "/n/data1/hms/dbmi/park/jbrew/Tau/GENERALIZED_SOLUTIONS/7_5_solutions_updated.sobj"
DEFAULT_MATRICES_H5 = "/n/data1/hms/dbmi/park/jbrew/tools/Tau/inst/extdata/matrices/matrices_7_7.h5"
routes = tau_time.load_routes(DEFAULT_ROUTES_SAGE, DEFAULT_MATRICES_H5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Tau on a single PCAWG sample.')
    parser.add_argument('--sample', type=str, required=True, help='Sample ID to process')
    parser.add_argument('--vcf', type=str, required=True, help='Path to the VCF file')
    parser.add_argument('--cnv', type=str, required=True, help='Path to the CNV file')
    parser.add_argument('--output_times', type=str, required=True, help='Path to the output times file')
    parser.add_argument('--output_fig', type=str, required=True, help='Path to the output overview figure')
    parser.add_argument('--output_gpkl', type=str, required=True, help='Path to the genome pickle file')
    parser.add_argument('--output_cluster_tsv', type=str, required=True, help='Path to the segment clusters file')
    parser.add_argument('--output_cluster_plot', type=str, required=True, help='Path to the segment clusters file')
    parser.add_argument('--output_cluster_gpkl', type=str, required=True, help='Path to the clustered genome pickle file')
    parser.add_argument('--output_time_cluster_plot', type=str, required=True, help='Path to the timepoint clusters plot')
    parser.add_argument('--output_time_cluster_df', type=str, required=True, help='Path to the timepoint clusters dataframe')

    args = parser.parse_args()

    #make empty files if not provided of all args outputs
    output_args = [x for x in args.__dict__.keys() if x.startswith('output_')]

    for out_arg in output_args:
        out_path = args.__dict__[out_arg]
        with open(out_path, 'w') as f:
            pass

    sample = args.sample
    vcf_file = args.vcf
    cn_file = args.cnv

    ref_fasta = "/n/data1/hms/dbmi/park/jbrew/ref/hg19_decoy/human_g1k_v37_decoy.fasta"
    cosmic_csv = "/n/data1/hms/dbmi/park/jbrew/Tau/NEW_TAU/COSMIC_v3p2_SBS_WGS_MuSiCal.csv"
    exposures = "/n/data1/hms/dbmi/park/jbrew/Tau/NEW_TAU/MuSiCal_assignment_PCAWG.txt"
    purity_file = "/n/data1/hms/dbmi/park/jbrew/Tau/NEW_TAU/consensus.20170217.purity.ploidy.txt"

    purity_df = pd.read_csv(purity_file, sep='\t')
    purity = float(purity_df.loc[purity_df['samplename'] == sample, 'purity'])

    #subclonal_list = 'dc537fcf-d910-4c4b-8af9-e7da429f2633_subclonal_list.txt'

    #build preprocessed dataframe from VCF and CN files
    mut_df = preprocess_sample(
            sample=sample, #sample name
            vcf_path=vcf_file, #VCF file path
            cnv_tsv=cn_file, #copy number file path
            purity=purity, #tumor purity
            ref_fasta=ref_fasta, #reference genome path
            cosmic_csv=cosmic_csv, #COSMIC signatures file path
            exposures_tsv=exposures, #pre-computed MuSiCal exposures file path
            apply_subclonal_filter=False, #whether to apply built-in (minimal) low-VAF subclonal filtering
            detect_min_alt=3, #minimum alt reads to consider a mutation detectable
            signatures=['SBS1','SBS5'], #mutational signatures to consider
            mode='soft', #weights mode for signature fitting (default: soft), soft is continuous (down-weights unlikely signatures), hard is binary (only keeps selected signatures)
            subclonal_list=None #optional: path to a list of mutations to flag as subclonal
    )

    #run Tau time estimation
    g = Genome.create(mut_df,
                   purity=purity, 
                   detect_min_alt=3,
                   detect_min_vaf=0)
    g.calculate_multiplicities(bootstrap_B=1, random_state=42)
    g.time_segments(env=routes)


    #post-processing: cluster segment multiplicities
    seg_cluster_df = cluster_segment_multiplicities(g, output_plot_file=args.output_cluster_plot)

    times_df = make_time_df(g)

    #save times dataframe to tsv
    times_df.to_csv(args.output_times, sep='\t', index=False)

    #save genome as pkl
    with gzip.open(args.output_gpkl, "wb") as f:
        pickle.dump(g, f)

    #cluster timepoints
    cluster_times, segment_cluster_ids, original_times, cluster_summary_df = cluster_times(g, routes, cluster_output_file=args.output_time_cluster_plot)

    cluster_summary_df.to_csv(args.output_time_cluster_df, sep='\t', index=False)

    cluster_definitions = {}
    for idx, row in cluster_summary_df.iterrows():
        cluster_definitions[row['cluster_id']] = row['classification']

    if seg_cluster_df.empty:
        plot_overview(g, cluster_times=cluster_times, segment_cluster_ids=segment_cluster_ids, original_times=original_times, clustered_genome=None, cluster_definitions=cluster_definitions, output_file=args.output_fig)
        exit(0)

    clustered_genome = make_clustered_genome(g, seg_cluster_df)
    clustered_genome.calculate_multiplicities(bootstrap_B=1, random_state=42)
    clustered_genome.time_segments(env=routes)

    #save clustered genome as pkl
    with gzip.open(args.output_cluster_gpkl, "wb") as f:
        pickle.dump(clustered_genome, f)

    #save segment cluster assignments
    seg_cluster_df.to_csv(args.output_cluster_tsv, sep='\t', index=False)

    #plot overview with clustered genome
    plot_overview(g, cluster_times=cluster_times, segment_cluster_ids=segment_cluster_ids, original_times=original_times, clustered_genome=clustered_genome, cluster_definitions=cluster_definitions, output_file=args.output_fig)