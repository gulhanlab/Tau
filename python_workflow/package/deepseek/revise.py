# tau/preprocessing/revise.py
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import betabinom
import os

def revise_data(df, sample=None, output_file=None, n_sub_thresh=20, distance_scale=3):
    """
    Revise mutation categorization based on subclonal clustering.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        sample (str, optional): Sample name.
        output_file (str, optional): Path to save the output. If None, no file is written.
        n_sub_thresh (int): Threshold for subclonal clustering.
        distance_scale (float): Distance scaling factor.
    
    Returns:
        pd.DataFrame: Revised DataFrame.
    """
    has_subclone = False
    if df['categ'].str.contains('subclonal').sum() > n_sub_thresh:
        print('Clustering subclones (Step 1)')
        df = cluster_subclones(df, sample, output_dir=os.path.dirname(output_file))
        has_subclone = True
    else:
        df.loc[df['categ'].str.contains('subclonal'), 'categ'] = 'undefined'

    if not has_subclone:
        if output_file:
            df.to_csv(output_file, sep='\t', index=False)
        return df

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

    if output_file:
        df.to_csv(output_file, sep='\t', index=False)
    
    return df
