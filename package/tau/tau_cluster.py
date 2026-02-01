import sys
import tau_time
import pandas as pd
from importlib import reload
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tau_multiplicity import Segment, Genome
from tau_plot import compute_offsets, plot_tau_stack
from tau_utils import pick_best_key, order_t, extract_mat
import tau_time

HG19_ORDER = [f"{i}" for i in range(1, 23)] + ["X", "Y"]
HG19_SIZES = {
    "1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276,
    "5": 180915260, "6": 171115067, "7": 159138663, "8": 146364022,
    "9": 141213431, "10": 135534747, "11": 135006516, "12": 133851895,
    "13": 115169878, "14": 107349540, "15": 102531392, "16": 90354753,
    "17": 81195210, "18": 78077248, "19": 59128983, "20": 63025520,
    "21": 48129895, "22": 51304566}#, "X": 155270560, "Y": 59373566,
#}


#CLUSTERING TIMEPOINTS

def calculate_likelihood(seg, assignments, env, key, num_private=0, private_penalty=.01):
    #if assignments are not monotonic, return small
    if not np.all(np.diff(assignments) >= -1e3):
        return -1e300
    seg_snvs = seg.snv_table
    new_times = np.diff(np.pad(assignments, (1, 1), constant_values=(0, 1)))
    A = env.MATRICES[key].T
    mult_dist = A @ new_times
    mult_dist /= sum(mult_dist)
    likelihoods = np.vstack(seg_snvs['multiplicity_likelihoods'])
    weights = np.array(seg_snvs['mut_w'])
    marg = likelihoods @ mult_dist
    marg = np.clip(marg, 1e-300, None)
    full_likelihood = np.sum(weights * np.log(marg))
    full_likelihood -= private_penalty * num_private
    return full_likelihood

def build_times_from_indices(seg_id, idx_arr, T, original_times, cluster_output_file=None):
    og = original_times[seg_id]
    return np.array([og[j] if idx == -1 else float(T[idx]) for j, idx in enumerate(idx_arr)], float)

def cluster_times(g, env, cluster_output_file=None, ess_thresh=20):
    #initialization!
    usable_segments = [seg for seg in g.segments if seg.major_cn > 1 and len(seg.timing_result) > 0 and np.sum(seg.N_counts) > ess_thresh]

    original_times = {}
    curr_assignment_indices = {}
    num_mutations_per_seg = {seg.seg_id: sum(seg.N_counts) for seg in usable_segments}
    seg_length = {seg.seg_id: seg.end - seg.start for seg in usable_segments}
    curr_ll = {}

    total_snvs = sum(num_mutations_per_seg.values())

    for seg in usable_segments:
        key = pick_best_key(seg)

        #pick random draw
        draw_idx = np.random.choice(len(seg.timing_result[key]['draws']))
        times = seg.timing_result[key]['draws'][draw_idx]['t']

        time_array = order_t(times)
        cumulative_times = np.cumsum(time_array)[:-1]
        original_times[seg.seg_id] = cumulative_times
        curr_ll[seg.seg_id] = calculate_likelihood(seg, cumulative_times, env, key, num_private=len(cumulative_times))
        curr_assignment_indices[seg.seg_id] = np.repeat(-1, len(cumulative_times))

    #calculate AIC with zero events
    num_params = 0
    total_ll = sum(curr_ll.values())
    AIC = - 2 * total_ll

    best_event_num = 0
    best_AIC = AIC
    cluster_times = []
    segment_cluster_ids = curr_assignment_indices.copy()

    print(f'{best_event_num} events: total LL={total_ll}, AIC={AIC}')

    for event_num in range(1, 5):
        curr_assignment_indices = {sid: np.repeat(-1, len(original_times[sid])) for sid in original_times}
        curr_ll = {}

        for seg in usable_segments:
            key = pick_best_key(seg)
            times = original_times[seg.seg_id]
            curr_ll[seg.seg_id] = calculate_likelihood(seg, times, env, key, num_private=len(times))
            
        #make T spaced out widely between 0 and 1
        T = np.linspace(0.1, 0.9, event_num) if event_num > 1 else np.array([0.5], float)
        print(f'T:{T}')
        total_lik = sum(curr_ll.values())

        T_history = []
        idx_history = []
        lik_history = []

        #developing the assignments and the likelihood assessment
        for i in range(10):
            #plot things
            #fig, ax = plt.subplots(figsize=(25,10))

            #plot_tau_stack(ax, segments_ordered, offsets, ess_thresh=10, cluster_times=T, segment_cluster_ids=curr_assignment_indices)

            T_history.append(T.copy())
            idx_history.append({sid: curr_assignment_indices[sid].copy() for sid in curr_assignment_indices})
            lik_history.append(sum(curr_ll.values()))
            
            #assignment update
            for seg in usable_segments:
                key = pick_best_key(seg)
                #print(key)
                #key = next(iter(seg.timing_result.keys()))
                idx_arr = curr_assignment_indices[seg.seg_id]

                for t_idx in range(len(idx_arr)):
                    old_idx = int(idx_arr[t_idx])
                    best_idx = old_idx
                    best_ll  = curr_ll[seg.seg_id]

                    base_times = build_times_from_indices(seg.seg_id, idx_arr, T, original_times)
                    lo = base_times[t_idx-1] if t_idx > 0 else 0.0
                    hi = base_times[t_idx+1] if t_idx < len(base_times)-1 else 1.0

                    for cand in [-1] + list(range(len(T))):
                        v = original_times[seg.seg_id][t_idx] if cand == -1 else float(T[cand])
                        if not (lo <= v <= hi):
                            continue #invalid time assignment

                        idx_arr[t_idx] = cand
                        num_private = int(np.sum(idx_arr == -1))
                        new_times = build_times_from_indices(seg.seg_id, idx_arr, T, original_times)

                        new_ll = calculate_likelihood(seg, new_times, env, key, num_private=num_private)

                        if new_ll > best_ll:
                            best_ll = new_ll
                            best_idx = cand

                    idx_arr[t_idx] = best_idx
                    curr_ll[seg.seg_id] = best_ll
                    
            #T update
            #make T values equal to the mean value of all og times assigned to index from T
            new_T = T.copy()
            for T_idx in range(len(T)):
                assigned_times = []
                segments = []
                for seg in usable_segments:
                    seg_curr_indices = curr_assignment_indices[seg.seg_id]
                    seg_og_times = original_times[seg.seg_id]
                    assigned_times.extend([t for idx, t in enumerate(seg_og_times) if seg_curr_indices[idx] == T_idx])
                    segments.extend([seg for idx, t in enumerate(seg_og_times) if seg_curr_indices[idx] == T_idx])
                if assigned_times:
                    #print(f"T[{T_idx}] # assigned times: {len(assigned_times)}, # segments: {len(segments)}")
                    new_T[T_idx] = np.average(assigned_times, weights=[num_mutations_per_seg[seg.seg_id]* seg_length[seg.seg_id] for seg in segments])
                else:
                    new_T[T_idx] = float(np.random.choice(np.concatenate(list(original_times.values()))))
            
            T = new_T.copy()

            for seg in usable_segments:
                key = pick_best_key(seg)
                idx_arr = curr_assignment_indices[seg.seg_id]
                num_private = int(np.sum(idx_arr == -1))
                times = build_times_from_indices(seg.seg_id, idx_arr, T, original_times)
                curr_ll[seg.seg_id] = calculate_likelihood(seg, times, env, key, num_private=num_private)

            total_lik = sum(curr_ll.values())
            #print('T has been updated:', T)
            #print('Updated likelihood:', total_lik,'\n')

            #plot things
            #fig, ax = plt.subplots(figsize=(25,10))

            #plot_tau_stack(ax, segments_ordered, offsets, ess_thresh=10, cluster_times=T, segment_cluster_ids=curr_assignment_indices)
        
        AIC = 2 * event_num - 2 * total_lik
        BIC = event_num * np.log(total_snvs) - 2 * total_lik
        print(f'{event_num} Events, Total Likelihood: {total_lik}, AIC: {AIC}, BIC: {BIC}')
        print(f'Times: {T}')
        if AIC < best_AIC:
            best_AIC = AIC
            best_event_num = event_num
            cluster_times = T.copy()
            segment_cluster_ids = {sid: curr_assignment_indices[sid].copy() for sid in curr_assignment_indices}

    print(f'Best model has {best_event_num} events with AIC {best_AIC}')

    #plot best model
    fig, ax = plt.subplots(figsize=(25,10))
    segments_ordered, offsets = compute_offsets(g.segments)

    plot_tau_stack(ax, segments_ordered, offsets, ess_thresh=10, cluster_times=cluster_times, segment_cluster_ids=segment_cluster_ids, output_file=cluster_output_file)
    
    segment_list_per_cluster = {}
    for seg_id, idx_arr in segment_cluster_ids.items():
        for idx in idx_arr:
            if idx not in segment_list_per_cluster:
                segment_list_per_cluster[idx] = []
            segment_list_per_cluster[idx].append(seg_id)
            
    cluster_sizes = {}
    import re
    for cluster_idx, seg_list in segment_list_per_cluster.items():
        total_cluster_length = np.array([np.diff(list(map(int, re.split(':|-', x)[1:]))) for x in seg_list]).sum()
        print(f'Cluster {cluster_idx}: # segments = {len(seg_list)}, total length = {total_cluster_length}')
        cluster_sizes[cluster_idx] = total_cluster_length

    best_event_num = len(cluster_times)
    #add clusters that have zero segments
    for cid in range(-1, best_event_num):
        if cid not in cluster_sizes:
            cluster_sizes[cid] = 0

    total_length = 0
    for seg in [s for s in g.segments if sum(s.N_counts) > 20]:
        seg_length = seg.end - seg.start
        total_length += seg_length

    total_genome_length = sum(HG19_SIZES.values())
    cluster_pct_of_observed = {cid: length / total_length * 100 for cid, length in cluster_sizes.items()}
    cluster_pct_of_theoretical = {cid: length / total_genome_length * 100 for cid, length in cluster_sizes.items()}
    cluster_definitions = {cid: 'PGD' if pct < 40 else 'WGD' for cid, pct in cluster_pct_of_observed.items()}
    cluster_definitions[-1] = 'Private'  # private mutations

    #make a dataframe summarizing the clusters
    cluster_summary_df = pd.DataFrame({
        'cluster_id': list(cluster_sizes.keys()),
        'time': list(cluster_times)[::-1] + [None],
        'num_segments': [len(segment_list_per_cluster[cid]) if cid in segment_list_per_cluster else 0 for cid in cluster_sizes.keys()],
        'total_length': list(cluster_sizes.values()),
        'pct_of_observed_genome': [cluster_pct_of_observed[cid] for cid in cluster_sizes.keys()],
        'pct_of_theoretical_genome': [cluster_pct_of_theoretical[cid] for cid in cluster_sizes.keys()],
        'classification': [cluster_definitions[cid] for cid in cluster_sizes.keys()],
    })
    
    return cluster_times, segment_cluster_ids, original_times, cluster_summary_df


#CLUSTERING SEGMENTS BY MULTIPLICITY PROFILES
hg19_centromeres = pd.read_csv('/n/data1/hms/dbmi/park/jbrew/Tau/NEW_TAU/hg19.centromeres.tsv', sep='\t')
hg19_centromeres['chrom'] = hg19_centromeres['chrom'].astype(str)

#build a classifier for p arm or q arm based on centromere positions
def assign_chrom_arm(chrom, start, end):
    centromere_row = hg19_centromeres[hg19_centromeres['chrom'] == chrom]
    if centromere_row.empty:
        return chrom
    cent_start = centromere_row['p_end'].values[0]
    cent_end = centromere_row['q_start'].values[0]
    if end < cent_start:
        return 'p'
    elif start > cent_end:
        return 'q'
    else:
        return 'centromere'

#write function that does PCA and clustering on each major_cn and minor_cn combination and assigns a cluster label to each segment and save the PCA so we can plot later
#returns data frame with multiplicity cluster labels and multiplicities and PCA values for each segment

def cluster_segment_multiplicities(genome, output_plot_file=None):
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    import pandas as pd
    import numpy as np

    records = []
    for major_cn in range(2, 7):
        for minor_cn in range(0, major_cn+1):
            seg_subset = [seg for seg in genome.segments if seg.major_cn == major_cn and seg.minor_cn == minor_cn]
            points = []
            seg_ids = []
            for seg in seg_subset:
                seg_id = seg.seg_id
                N_counts = np.array(seg.N_counts)
                total = N_counts.sum()
                if total < 40:
                    continue

                norm_counts = N_counts / total
                eff_N = total
                points.append(np.concatenate([norm_counts, [eff_N]]))
                seg_ids.append(seg_id)

            if len(points) <= 2:
                continue

            points = np.array(points)
            X = points[:, :len(seg.N_counts)]
            weights = points[:, -1]
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            #quantify contributions of each PC
            explained_variance = pca.explained_variance_ratio_
            labels = DBSCAN(eps=0.1, min_samples=5).fit_predict(X_pca, sample_weight=weights)
            #instead of applying PCA, try clustering on raw normalized counts with DBSCAN
            #labels = DBSCAN(eps=0.1, min_samples=1).fit_predict(X, sample_weight=weights)

            for i, seg_id in enumerate(seg_ids):
                seg = next((s for s in genome.segments if s.seg_id == seg_id), None)
                records.append({
                    'seg_id': seg_id,
                    'major_cn': major_cn,
                    'minor_cn': minor_cn,
                    'cluster_label': labels[i],
                    'pca_1': X_pca[i, 0],
                    'pca_2': X_pca[i, 1],
                    'pca_1_contribution': explained_variance[0],
                    'pca_2_contribution': explained_variance[1],
                    'N_counts': seg.N_counts,
                })

    seg_cluster_df = pd.DataFrame(records)

    if not output_plot_file or seg_cluster_df.empty:
        return seg_cluster_df
    
    #plot the PCA results colored by chromosome arm (p or q) and cluster label side by side for each major_cn, minor_cn combo
    for (major_cn, minor_cn), sub in seg_cluster_df.groupby(['major_cn','minor_cn']):
        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        arms = []
        for _, row in sub.iterrows():
            seg = next((s for s in genome.segments if s.seg_id == row['seg_id']), None)
            if seg is not None:
                arm = assign_chrom_arm(str(seg.chrom), seg.start, seg.end)
                arms.append(str(seg.chrom) + arm)
            else:
                arms.append('unknown')
        sub = sub.assign(chrom_arm=arms)

        unique_arms = sorted(set(arms))
        colors = sns.color_palette('hsv', len(unique_arms))
        arm_color_map = {arm: colors[i] for i, arm in enumerate(unique_arms)}

        for arm in unique_arms:
            mask = sub['chrom_arm'] == arm
            ax[0].scatter(sub.loc[mask, 'pca_1'], sub.loc[mask, 'pca_2'], s=20, alpha=0.5, color=arm_color_map[arm], label=f'Arm {arm}')
        ax[0].set_title(f'Major CN={major_cn}, Minor CN={minor_cn} colored by Chromosome Arm')
        ax[0].set_xlabel('PCA 1')
        ax[0].set_ylabel('PCA 2')
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        unique_labels = sorted(sub['cluster_label'].unique())
        colors = sns.color_palette('hsv', len(unique_labels))
        label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

        for label in unique_labels:
            mask = sub['cluster_label'] == label
            ax[1].scatter(sub.loc[mask, 'pca_1'], sub.loc[mask, 'pca_2'], s=20, alpha=0.5, color=label_color_map[label], label=f'Cluster {label}')
        ax[1].set_title(f'Major CN={major_cn}, Minor CN={minor_cn} colored by Cluster Label')
        ax[1].set_xlabel('PCA 1')
        ax[1].set_ylabel('PCA 2')
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
    
    plt.savefig(f'{output_plot_file}', dpi=400)

    return seg_cluster_df

def make_clustered_genome(g, seg_cluster_df): 
    clustered_genome = Genome()

    #merge segments that have the same cluster label and major_cn and minor_cn
    for (major_cn, minor_cn, cluster_label), group in seg_cluster_df.groupby(['major_cn','minor_cn','cluster_label']):
        if cluster_label == -1:
            #just return standard segment
            for _, row in group.iterrows():
                seg = next((s for s in g.segments if s.seg_id == row['seg_id']), None)
                if seg is not None:
                    clustered_genome.segments.append(seg)
            continue

        snv_tables = pd.concat([s.snv_table for s in g.segments if s.seg_id in group['seg_id'].values], ignore_index=True)

        merged_seg = Segment(
            chrom=','.join(group['seg_id'].apply(lambda x: x.split(':')[0])),  #concatenate all together with commas
            start=min(next((s.start for s in g.segments if s.seg_id == seg_id), None) for seg_id in group['seg_id']),
            end=max(next((s.end for s in g.segments if s.seg_id == seg_id), None) for seg_id in group['seg_id']),
            major_cn=major_cn,
            minor_cn=minor_cn,
            seg_id=f'merged_{major_cn}_{minor_cn}_{cluster_label}',
            purity=g.segments[0].purity,
            snv_table=snv_tables
        )
        merged_seg.num_segs = len(group)
        merged_seg.seg_ids = group['seg_id'].tolist()
        merged_seg.total_length = sum(next((s.end - s.start) for s in g.segments if s.seg_id == seg_id) for seg_id in group['seg_id'])

        clustered_genome.segments.append(merged_seg)

    return clustered_genome