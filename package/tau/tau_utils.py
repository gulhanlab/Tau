import sys
import tau_time
import pandas as pd
from importlib import reload
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tau_multiplicity import Segment, Genome

import re
_t_pat = re.compile(r"t(\d+)$")

def order_t(tdict):
    """Dense [t1..tK] in order of t-index, row sums to 1."""
    if not tdict: 
        return np.array([1.0], float)
    items = []
    for k, v in tdict.items():
        m = _t_pat.search(str(k))
        if m:
            items.append((int(m.group(1)), float(v)))
    if not items:
        return np.array([1.0], float)
    K = max(i for i, _ in items)
    arr = np.zeros(K, float)
    for i, v in items:
        if i >= 1:
            arr[i-1] = v
    s = arr.sum()
    return arr / s if s > 0 else arr

def time_matrix(time_list):
    return np.vstack([order_t(t_dict) for t_dict in time_list])

def pick_best_key(seg):
    timing = getattr(seg, "timing_result", {}) or {}
    if not timing:
        return None

    best = None
    best_d = np.inf
    for k, v in timing.items():
        qc = v.get("qc", [])
        if not qc:
            continue
        d = qc[0].get("dist_rel", np.inf)
        if d < best_d and len(v.get("draws", [])) > 0:
            best_d = d
            best = k
    return best

def extract_mat(seg, key, boot_id=0):
    draws = seg.timing_result[key].get("draws", [])
    boot_draws = [d for d in draws if int(d.get("boot_id", 0)) == boot_id]
    if not boot_draws:
        return None
    mat = time_matrix([d["t"] for d in boot_draws])  # (#draws, K)
    return mat


def make_time_df(genome):
    time_df = pd.DataFrame()

    for seg in genome.segments:
        chrom = seg.chrom
        s0 = seg.start
        s1 = seg.end

        keys = seg.timing_result.keys()
        timing_result = getattr(seg, "timing_result", {}) or {}

        euclidean_distances = {}

        for key, value in timing_result.items():
            distance = next(x for x in value['qc'])['dist_rel']
            euclidean_distances[key] = distance
        
        keys = np.array(list(timing_result.keys()))
        distance_sorted_keys = keys[np.argsort(list(euclidean_distances.values()))]
        keys_with_results = [k for k in distance_sorted_keys if len(timing_result.get(k, {}).get('draws', [])) > 0]

        #metric = 'loglik'
        #logliks_per_route = [(next(x for x in timing_result[k]['qc'] if x[metric] is not None)[metric], k) for k in keys]
        #max_loglik = max(logliks_per_route, key=lambda x: x[0])[0]

        #probs = [np.exp(ll - max_loglik) for ll, _ in logliks_per_route]
        #total_prob = sum(probs)
        #probs = {key : p / total_prob for key, p in zip(keys, probs)}

        #ll_sorted_keys = sorted(keys, key=lambda k: probs[k], reverse=True)

        lls = {k: next(x for x in timing_result[k]['qc'] if x['loglik'] is not None)['loglik'] for k in keys}
        if len(lls) == 0:
            continue
        else:
            max_ll = max(lls.values())
            probs = {k: np.exp(lls[k] - max_ll) for k in keys}
            total_prob = sum(probs.values())
            probs = {k: p / total_prob for k, p in probs.items()}

        for key in distance_sorted_keys:
        #key = distance_sorted_keys[0] #pick top key
            all_draws = timing_result[key].get('draws', [])
            if len(all_draws) == 0:
                continue

            draws_by_bootstrap = {}
            for i, draw in enumerate(all_draws, 1):
                b_id = draw.get('boot_id', 0)
                if b_id not in draws_by_bootstrap:
                    draws_by_bootstrap[b_id] = []
                draws_by_bootstrap[b_id].append(draw)

            for b_id in draws_by_bootstrap:
                mat = time_matrix([x['t'] for x in draws_by_bootstrap[b_id]])

                cumulative_times = mat.cumsum(axis=1)[:, :-1]
                num_draws = cumulative_times.shape[0]

                for k in range(cumulative_times.shape[1]):
                    time_df = pd.concat([time_df, pd.DataFrame({
                        'bootstrap_id': b_id,
                        'draw_id': np.arange(1, num_draws + 1),
                        'chrom': chrom,
                        'start': s0,
                        'end': s1,
                        'segment_id': seg.seg_id,
                        'key': key,
                        'major_cn': seg.major_cn,
                        'minor_cn': seg.minor_cn,
                        'time_point': k + 1,
                        'time_fraction': cumulative_times[:, k],
                        't': np.diff(np.pad(cumulative_times, ((0, 0), (1, 0)), mode='constant'), axis=1)[:, k],
                        'mutation_count': sum(seg.N_counts),
                        'num_draws': num_draws,
                        'distance_from_route_boundary': euclidean_distances[key],
                        'route_probability': probs[key],
                        'length': seg.total_length if hasattr(seg, 'total_length') else (seg.end - seg.start),
                    })])
    if time_df.empty:
        return time_df
    
    time_df['draw_weight'] = 1/time_df['num_draws']
    time_df['w'] = time_df['draw_weight'] * time_df['route_probability'] * time_df['length']
    time_df['best_route'] = time_df.groupby('segment_id')['route_probability'].transform('max') == time_df['route_probability']
    time_df['min_distance'] = time_df.groupby('segment_id')['distance_from_route_boundary'].transform('min')
    return time_df