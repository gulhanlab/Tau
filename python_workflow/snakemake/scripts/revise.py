#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import betabinom
import argparse
import os

def hc(x, k, d_meth='euclidean', **kwargs):
    Z = linkage(x, method='ward', metric=d_meth)
    return {'cluster': cut_tree(Z, n_clusters=k).flatten()}

def clon_sub_distance(df, by_state=True):
    if by_state:
        df['clonal_cn_state'] = df['major_cn'].astype(str) + '_' + df['minor_cn'].astype(str)
        states = df['clonal_cn_state'].unique()
    else:
        states = ['all']
        df['clonal_cn_state'] = 'all'

    df['dist_clon'] = np.nan
    df['dist_sub'] = np.nan
    df['mean_count'] = np.nan
    df['sd_count'] = np.nan
    df['mean_count_sub'] = np.nan
    df['sd_count_sub'] = np.nan

    for state in states:
        state_df = df[(df['categ'] == 'clonal') & (df['clonal_cn_state'] == state)]
        if state_df['best_cn'].isna().all():
            continue

        min_clonal_cn = state_df['best_cn'].min()
        inds = df[(df['best_cn'] == min_clonal_cn) & (df['categ'] == 'clonal') & (df['clonal_cn_state'] == state)].index

        mean_count = ((1 / df.loc[inds, 'vaf']) / df.loc[inds, 'total_cn']).mean()
        sd_count = ((1 / df.loc[inds, 'vaf']) / df.loc[inds, 'total_cn']).std()
        
        # Fix: Separate filtering and column access
        subclonal1_mask = (df['clonal_cn_state'] == state) & (df['categ'] == 'subclonal1')
        mean_count_sub = ((1 / df.loc[subclonal1_mask, 'vaf']) / df.loc[subclonal1_mask, 'total_cn']).mean()
        sd_count_sub = ((1 / df.loc[subclonal1_mask, 'vaf']) / df.loc[subclonal1_mask, 'total_cn']).std()

        inds_state = df[df['clonal_cn_state'] == state].index
        if len(inds_state) > 10:
            df.loc[inds_state, 'mean_count'] = mean_count
            df.loc[inds_state, 'sd_count'] = sd_count
            df.loc[inds_state, 'mean_count_sub'] = mean_count_sub
            df.loc[inds_state, 'sd_count_sub'] = sd_count_sub

            df.loc[inds_state, 'dist_clon'] = np.abs((1 / df.loc[inds_state, 'vaf']) / df.loc[inds_state, 'total_cn'] - df.loc[inds_state, 'mean_count']) / df.loc[inds_state, 'sd_count']
            df.loc[inds_state, 'dist_sub'] = np.abs((1 / df.loc[inds_state, 'vaf']) / df.loc[inds_state, 'total_cn'] - df.loc[inds_state, 'mean_count_sub']) / df.loc[inds_state, 'sd_count_sub']

            df.loc[inds_state[df.loc[inds_state, 'best_cn'] > min_clonal_cn], 'dist_clon'] = 0

    return df

def cluster_subclones(df, sample, output_dir, min_vaf_diff=0.05):
    subclonal_rows = df['categ'].str.contains('subclonal')
    vals = (1 / df.loc[subclonal_rows, 'vaf']) / df.loc[subclonal_rows, 'total_cn']

    valid_indices = vals.index[np.isfinite(vals) & (vals > 0)]
    vals = vals[valid_indices]

    if len(vals) == 0:
        raise ValueError("No valid data for clustering. Check VAF and total CN values.")

    df_x = pd.DataFrame({'x': np.log2(vals + 1)})

    # Perform clustering using hierarchical clustering
    Z = linkage(df_x, method='ward')
    best_k = 5  # Placeholder for actual gap statistic calculation
    cluster_inds = fcluster(Z, best_k, criterion='maxclust')

    mean_vec = []
    for j in range(1, best_k + 1):
        mean_vec.append(vals[cluster_inds == j].mean())

    df_subclonal_CCF = pd.DataFrame({'CCFs_subclone': 1 / np.array(mean_vec), 'index': range(1, len(mean_vec) + 1)})
    if len(df_subclonal_CCF) > 1:
        df_subclonal_CCF = df_subclonal_CCF.sort_values(by='CCFs_subclone', ascending=False)
    df_subclonal_CCF['cluster'] = range(1, len(df_subclonal_CCF) + 1)
    df_subclonal_CCF.to_csv(os.path.join(output_dir, f'{sample}_CCF_subclones.txt'), sep='\t', index=False)

    indices = df_subclonal_CCF['cluster'].values[cluster_inds - 1]
    replacement_indices = subclonal_rows[valid_indices].index
    if len(indices) != len(replacement_indices):
        raise ValueError("Replacement length mismatch: check indices and subclonal categories.")
    df.loc[replacement_indices, 'categ'] = ['subclonal' + str(i) for i in indices]

    return df

def revise(sample, input_file, output_file, n_sub_thresh=20, distance_scale=3):
    df = pd.read_csv(input_file, sep='\t')

    has_subclone = False
    if df['categ'].str.contains('subclonal').sum() > n_sub_thresh:
        print('Clustering subclones (Step 1)')
        df = cluster_subclones(df, sample, output_dir=os.path.dirname(output_file))
        has_subclone = True
    else:
        df.loc[df['categ'].str.contains('subclonal'), 'categ'] = 'undefined'

    if not has_subclone:
        df.to_csv(output_file, sep='\t', index=False)
        return 0

    df = clon_sub_distance(df)

    inds = df[(df['dist_clon'] < distance_scale * df['dist_sub']) & (df['categ'] == "subclonal1") & (df['max_likelihood'] > 0.02)].index
    if len(inds) > 0:
        df.loc[inds, 'categ'] = 'clonal'
        print('Re-clustering subclones (Step 2)')
        df = cluster_subclones(df, sample, output_dir=os.path.dirname(output_file))
        df = clon_sub_distance(df)

    inds = df[(df['dist_clon'] > distance_scale * df['dist_sub']) & (~df['categ'].str.contains('subclonal')) & (df['dist_clon'] > 0) & (df['vaf'] < 1 / df['total_cn'])].index
    if len(inds) > 0:
        df.loc[inds, 'categ'] = 'subclonal1'
        print('Re-clustering subclones (Step 3)')
        df = cluster_subclones(df, sample, output_dir=os.path.dirname(output_file))
        df = clon_sub_distance(df)

    if df['categ'].str.contains('subclonal').sum() > n_sub_thresh:
        print('Final subclone clustering (Step 4)')
        df = cluster_subclones(df, sample, output_dir=os.path.dirname(output_file))

    df.to_csv(output_file, sep='\t', index=False)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Revise script for Tau workflow")
    parser.add_argument("sample", help="Sample name")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")
    args = parser.parse_args()
    revise(args.sample, args.input_file, args.output_file)
