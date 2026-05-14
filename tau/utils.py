"""Utility functions for Tau timing analysis."""

import re
import numpy as np
import pandas as pd

from tau.core import Segment, Genome

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
            arr[i - 1] = v
    s = arr.sum()
    return arr / s if s > 0 else arr


def time_matrix(time_list):
    """Convert list of time dicts to matrix."""
    return np.vstack([order_t(t_dict) for t_dict in time_list])


def pick_best_key(seg, rng=None):
    """Pick the best timing key for a segment based on distance.

    When multiple routes share the minimum dist_rel (feasible region overlap),
    the winner is chosen uniformly at random rather than by dict iteration order.
    Pass an np.random.Generator via rng for reproducibility.
    """
    timing = getattr(seg, "timing_result", {}) or {}
    if not timing:
        return None

    _TIEBREAK_EPS = 1e-9
    best_d = np.inf
    candidates = []
    for k, v in timing.items():
        qc = v.get("qc", [])
        if not qc or not len(v.get("draws", [])):
            continue
        d = qc[0].get("dist_rel", np.inf)
        if d < best_d - _TIEBREAK_EPS:
            best_d = d
            candidates = [k]
        elif d <= best_d + _TIEBREAK_EPS:
            candidates.append(k)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    _rng = rng if rng is not None else np.random.default_rng()
    return candidates[int(_rng.integers(len(candidates)))]


def _sort_draws_for_display(mat, n_free, n_fixed_steps=5):
    """Re-order draws for coherent visual sweep in genome overview.

    n_free is the true number of free variables for this route (from the
    constraint matrix rank: n_free = n_cols - rank(A)).

    For 0-1 free variables: sort by largest-range column ascending.
    For 2+ free variables: bins_per_level = [5, 3, 2, 1, ...] for secondary
    vars; sweep the largest-range var continuously within each bin.
    """
    if mat.shape[0] <= 2 or n_free == 0:
        return mat
    col_ranges = mat.max(axis=0) - mat.min(axis=0)
    # Sort columns by range descending; index 0 = largest range (sweep var)
    cols_sorted = np.argsort(col_ranges)[::-1]
    if n_free == 1:
        return mat[np.argsort(mat[:, cols_sorted[0]])]
    bins_per_level = ([n_fixed_steps, 3, 2] + [1] * max(0, n_free - 3))[: n_free - 1]
    bin_indices = np.zeros(mat.shape[0], dtype=int)
    for level, col in enumerate(cols_sorted[1:n_free]):
        n_bins = bins_per_level[level]
        col_vals = mat[:, col]
        col_min, col_max = col_vals.min(), col_vals.max()
        if col_max > col_min:
            breakpoints = np.linspace(col_min, col_max, n_bins + 1)
            bin_i = np.searchsorted(breakpoints[1:-1], col_vals)
        else:
            bin_i = np.zeros(mat.shape[0], dtype=int)
        bin_indices = bin_indices * n_bins + bin_i
    order = np.lexsort((mat[:, cols_sorted[0]], bin_indices))
    return mat[order]


def extract_mat(seg, key, boot_id=0):
    """Extract time matrix from segment timing results."""
    draws = seg.timing_result[key].get("draws", [])
    boot_draws = [d for d in draws if int(d.get("boot_id", 0)) == boot_id]
    if not boot_draws:
        return None
    mat = time_matrix([d["t"] for d in boot_draws])  # (#draws, K)
    n_free = seg.timing_result[key].get("meta", {}).get("free_var_count", 1)
    return _sort_draws_for_display(mat, n_free)


def make_time_df(genome):
    """Create a DataFrame of timing results for all segments in a genome.

    .. deprecated::
        Use ``genome.times_to_df(sample)`` instead, which reads the current
        timing_result format correctly and supports the summarize parameter.
    """
    import warnings
    warnings.warn(
        "make_time_df() is deprecated. Use genome.times_to_df(sample) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    rows = []

    for seg in genome.segments:
        timing_result = getattr(seg, "timing_result", {}) or {}
        if not timing_result:
            continue

        chrom = seg.chrom
        s0 = seg.start
        s1 = seg.end
        seg_length = getattr(seg, "total_length", seg.end - seg.start)
        mut_count = sum(seg.N_counts) if seg.N_counts is not None else 0

        euclidean_distances = {
            k: v["qc"][0]["dist_rel"] if v.get("qc") else np.nan
            for k, v in timing_result.items()
        }

        keys_arr = np.array(list(timing_result.keys()))
        dist_vals = np.array([euclidean_distances[k] for k in keys_arr])
        distance_sorted_keys = keys_arr[np.argsort(dist_vals)]

        lls = {}
        for k in keys_arr:
            qc = timing_result[k].get("qc", [])
            qc0 = next((q for q in qc if q.get("loglik") is not None), None)
            if qc0 is not None:
                lls[k] = float(qc0["loglik"])

        if not lls:
            continue
        max_ll = max(lls.values())
        raw = {k: np.exp(lls[k] - max_ll) for k in lls}
        total_prob = sum(raw.values())
        probs = {k: p / total_prob for k, p in raw.items()}

        for key in distance_sorted_keys:
            all_draws = timing_result[key].get("draws", [])
            if not all_draws:
                continue

            draws_by_bootstrap: dict = {}
            for draw in all_draws:
                b_id = draw.get("boot_id", 0)
                draws_by_bootstrap.setdefault(b_id, []).append(draw)

            for b_id, b_draws in draws_by_bootstrap.items():
                mat = time_matrix([x["t"] for x in b_draws])
                cumulative_times = mat.cumsum(axis=1)[:, :-1]
                num_draws = cumulative_times.shape[0]
                t_increments = np.diff(
                    np.pad(cumulative_times, ((0, 0), (1, 0)), mode="constant"),
                    axis=1,
                )

                for k in range(cumulative_times.shape[1]):
                    for d in range(num_draws):
                        rows.append({
                            "bootstrap_id": b_id,
                            "draw_id": d + 1,
                            "chrom": chrom,
                            "start": s0,
                            "end": s1,
                            "segment_id": seg.seg_id,
                            "key": key,
                            "major_cn": seg.major_cn,
                            "minor_cn": seg.minor_cn,
                            "time_point": k + 1,
                            "time_fraction": float(cumulative_times[d, k]),
                            "t": float(t_increments[d, k]),
                            "mutation_count": mut_count,
                            "num_draws": num_draws,
                            "distance_from_route_boundary": euclidean_distances[key],
                            "route_probability": probs.get(key, 0.0),
                            "length": seg_length,
                        })

    if not rows:
        return pd.DataFrame()

    time_df = pd.DataFrame(rows)
    time_df["draw_weight"] = 1.0 / time_df["num_draws"]
    time_df["w"] = (
        time_df["draw_weight"] * time_df["route_probability"] * time_df["length"]
    )
    time_df["best_route"] = (
        time_df.groupby("segment_id")["route_probability"].transform("max")
        == time_df["route_probability"]
    )
    time_df["min_distance"] = time_df.groupby("segment_id")[
        "distance_from_route_boundary"
    ].transform("min")
    return time_df
