#!/usr/bin/env python3
"""Simulation framework for generating synthetic tumor genomes."""

from __future__ import annotations
import os
import re
import argparse
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from importlib import resources

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from tau.core import Segment, Genome
from tau.preprocessing import preprocess_sample


def _get_data_path(filename):
    """Get path to package data file."""
    try:
        with resources.files("tau.data").joinpath(filename) as p:
            return str(p)
    except (FileNotFoundError, TypeError):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        return os.path.join(data_dir, filename)


# Default paths - can be overridden via environment variables
DEFAULT_MATRIX_H5 = os.environ.get("TAU_MATRICES_H5", _get_data_path("matrices_7_7.h5"))
DEFAULT_DIFF_ALLELES = os.environ.get(
    "TAU_DIFF_ALLELES",
    os.path.join(os.path.dirname(__file__), "..", "solutions", "counts_diagram_As.txt"),
)

CHR_LEN = {
    **{
        f"chr{i}": l
        for i, l in zip(
            range(1, 23),
            [
                249250621, 243199373, 198022430, 191154276, 180915260, 171115067,
                159138663, 146364022, 141213431, 135534747, 135006516, 133851895,
                115169878, 107349540, 102531392, 90354753, 81195210, 78077248,
                59128983, 63025520, 48129895, 51304566,
            ],
        )
    }
}

AUTOSOMES = list(CHR_LEN)
ARM_BOUNDARY = {
    "chr1": 125e6, "chr2": 93e6, "chr3": 91e6, "chr4": 50e6, "chr5": 48e6,
    "chr6": 60e6, "chr7": 60e6, "chr8": 45e6, "chr9": 49e6, "chr10": 40e6,
    "chr11": 53e6, "chr12": 35e6, "chr13": 17e6, "chr14": 16e6, "chr15": 17e6,
    "chr16": 36e6, "chr17": 25e6, "chr18": 17e6, "chr19": 27e6, "chr20": 28e6,
    "chr21": 13e6, "chr22": 15e6,
}

# Lazy-loaded globals
_diff_allele_dict = None
_MATRICES = None


def _load_diff_alleles(path=None):
    """Load diff allele lookup table."""
    global _diff_allele_dict
    if _diff_allele_dict is not None:
        return _diff_allele_dict

    path = path or DEFAULT_DIFF_ALLELES
    if not os.path.exists(path):
        # Try alternative location
        alt_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "solutions", "counts_diagram_As.txt"
        )
        if os.path.exists(alt_path):
            path = alt_path
        else:
            raise FileNotFoundError(f"Could not find diff alleles file at {path}")

    diff_df = pd.read_csv(path, sep="\t")
    diff_alleles = diff_df["diff_allele_indices"].apply(
        lambda x: np.unique(re.split("[;,]", str(x))) if not pd.isnull(x) else np.full(0, 0)
    )
    _diff_allele_dict = dict(zip(diff_df["tag_A"], diff_alleles))
    return _diff_allele_dict


def _load_matrices(path=None):
    """Load constraint matrices from HDF5."""
    global _MATRICES
    if _MATRICES is not None:
        return _MATRICES

    path = path or DEFAULT_MATRIX_H5
    with h5py.File(path, "r") as h5:
        _MATRICES = {k: h5[k][()] for k in h5}
    return _MATRICES


_t_pat = re.compile(r"t(\d+)$")
_rng = np.random.default_rng()
which_arm = lambda c, p: "p" if p < ARM_BOUNDARY[c] else "q"

MUT_DIST = (-5.5, 0)


def choose_allele_delta(seg, delta, loss=False):
    """Choose which allele to modify for gain/loss events."""
    if loss:
        if seg.minor > 0:
            drop = min(delta, seg.minor)
            return (0, -drop)
        return (-delta, 0)
    if seg.minor == 0:
        return (delta, 0)
    return (
        (0, delta)
        if _rng.random() < seg.minor / (seg.major + seg.minor) * 0.25
        else (delta, 0)
    )


@dataclass
class Event:
    """Base class for genomic events."""
    time: float

    def apply(self, genome: "SimGenome"):
        raise NotImplementedError


@dataclass
class SimSegment:
    """Simulated genomic segment with copy number state history."""
    chrom: str
    start: int
    end: int
    major: int
    minor: int
    state_history: list
    history: list = field(default_factory=list)
    muts: np.ndarray | None = None

    def copy(self):
        return SimSegment(
            self.chrom,
            self.start,
            self.end,
            self.major,
            self.minor,
            self.state_history.copy(),
            self.history.copy(),
        )

    def get_diff_allele_indices(self):
        """Calculate diff allele indices based on state history."""
        diff_allele_indices = []
        path = []
        sh = self.state_history
        history = self.history
        for i in range(len(sh) - 1):
            major_delta = sh[i + 1][0] - sh[i][0]
            minor_delta = sh[i + 1][1] - sh[i][1]
            path.append(["major"] * major_delta + ["minor"] * minor_delta)

        final_state = sh[-1]
        curr_t_index = 1
        for step in path:
            t_change = len(step)
            both_alleles = "major" in step and "minor" in step
            if both_alleles:
                potential_indices_to_zero = np.arange(t_change - 1) + 1
                t_indices = potential_indices_to_zero + curr_t_index
                allele_counts = np.unique(step, return_counts=True)[1]
                diff_allele_indices.append((t_indices, min(allele_counts)))
            curr_t_index += t_change

        self.allele_indices = diff_allele_indices

    def calculate_t_values(self):
        """Calculate timing values from event history."""
        t_vals = [0] * (max(self.major - 1, 0) + max(self.minor - 1, 0) + 1)
        if len(t_vals) == 1:
            self.t_vals = [1]
            return

        prev_state = (1, 1)
        prev_t = 0
        prev_event_time = 0
        for i, (state, event) in enumerate(zip(self.state_history[1:], self.history)):
            if isinstance(event, Loss):
                continue
            num_gains_in_event = sum(np.array(state) - np.array(prev_state))
            curr_t = num_gains_in_event + prev_t
            t_vals[prev_t] = event.time - prev_event_time
            t_vals[prev_t + 1 : curr_t] = [0] * (num_gains_in_event - 1)
            prev_state = state
            prev_t = curr_t
            prev_event_time = event.time

        t_vals[-1] = 1 - sum(t_vals[:-1])
        self.t_vals = t_vals

    def simulate_N_vals(self, mut_dist, n_mutations=None):
        """Simulate mutation counts per multiplicity."""
        MATRICES = _load_matrices()

        seg_len = self.end - self.start
        mut_mu = seg_len * 10 ** np.random.normal(*mut_dist)

        if self.major == 1 and self.minor == 1:
            return np.asarray([np.random.poisson(2 * mut_mu)])
        if self.major == 1 and self.minor == 0:
            return np.asarray([np.random.poisson(mut_mu)])
        if (self.major == 0 and self.minor == 0) or pd.isna(self.major) or pd.isna(self.minor):
            return np.asarray([0])

        if self.tag not in MATRICES:
            return np.asarray([np.nan])

        matrix = MATRICES[self.tag].T
        N_vals = matrix @ np.array(self.t_vals)

        if n_mutations is not None:
            N_vals = N_vals / N_vals.sum() * n_mutations

        mu_N = N_vals if n_mutations is not None else N_vals * mut_mu
        return np.random.poisson(mu_N).astype(int)


@dataclass
class SimGenome:
    """Simulated tumor genome."""
    segments: list = field(default_factory=list)
    purity: float = 1.0

    def __post_init__(self):
        if self.segments:
            return
        for chrom, length in CHR_LEN.items():
            b = int(ARM_BOUNDARY[chrom])
            self.segments += [
                SimSegment(chrom, 1, b, 1, 1, [(1, 1)]),
                SimSegment(chrom, b, length, 1, 1, [(1, 1)]),
            ]

    def _split_at(self, chrom: str, pos: int):
        for i, s in enumerate(self.segments):
            if s.chrom == chrom and s.start < pos < s.end:
                left = s.copy()
                right = s.copy()
                left.end = pos
                right.start = pos
                self.segments[i : i + 1] = [left, right]
                break

    def _alter_segmental(
        self, chrom: str, start: int, end: int, delta: int, loss: bool, ev: Event
    ):
        self._split_at(chrom, start)
        self._split_at(chrom, end)
        for seg in self.segments:
            if seg.chrom == chrom and start < seg.end and end > seg.start:
                dM, dN = choose_allele_delta(seg, delta, loss)
                seg.major = max(0, seg.major + dM)
                seg.minor = max(0, seg.minor + dN)
                seg.history.append(ev)
                seg.state_history.append((seg.major, seg.minor))

    def apply(self, ev: Event):
        ev.apply(self)

    def swap_minor_major(self):
        """Ensure major >= minor for all segments."""
        for s in self.segments:
            if s.minor > s.major:
                s.major, s.minor = s.minor, s.major
                s_hist = s.state_history
                minor = [x[0] for x in s_hist]
                major = [x[1] for x in s_hist]
                s.state_history = list(zip(minor, major))

    def unify_history_per_state(self):
        """Unify histories for segments with same CN state on same arm."""
        buckets = {}
        for s in self.segments:
            arm = which_arm(s.chrom, s.start)
            key = (s.chrom, arm, s.major, s.minor)
            buckets.setdefault(key, []).append(s)
        for segs in buckets.values():
            canon = segs[0].history
            canon_state_history = segs[0].state_history
            for s in segs[1:]:
                s.history = canon.copy()
                s.state_history = canon_state_history.copy()

    def calculate_t_values(self):
        for seg in self.segments:
            seg.calculate_t_values()

    def assign_tags(self):
        """Assign route tags to all segments."""
        diff_allele_dict = _load_diff_alleles()
        segs = self.segments
        for seg in segs:
            seg.get_diff_allele_indices()
            indices = seg.allele_indices
            tag_str = f"{seg.major}_{seg.minor}."
            potential_tags = {
                tag: idx for tag, idx in diff_allele_dict.items() if tag.startswith(tag_str)
            }
            final_tags = []
            multiple_tags = len(potential_tags) > 1
            if multiple_tags and seg.minor > 1:
                for tag, idx in potential_tags.items():
                    satisfied = True
                    remaining = set(np.array(idx, dtype=int))
                    for i, (event, num_both_allele) in enumerate(indices):
                        if any(x in remaining for x in event):
                            remaining -= set(event)
                        if len(remaining) == 0 and i < len(indices) - 1:
                            satisfied = False
                    if satisfied:
                        final_tags.append(tag)
                seg.tag = _rng.choice(final_tags)
            elif multiple_tags:
                seg.tag = _rng.choice(list(potential_tags.keys()))
            else:
                seg.tag = tag_str + "1"

        self._unify_tags()

    def _unify_tags(self):
        buckets = {}
        for s in self.segments:
            arm = which_arm(s.chrom, s.start)
            key = (s.chrom, arm, s.major, s.minor)
            buckets.setdefault(key, []).append(s)
        for segs in buckets.values():
            canon = segs[0].tag
            for s in segs[1:]:
                s.tag = canon

    def simulate_N_vals(self):
        """Simulate mutation counts for all segments."""
        for s in self.segments:
            s.muts = s.simulate_N_vals(MUT_DIST)
        print("Simulation complete!")


@dataclass
class WGD(Event):
    """Whole genome duplication event."""
    def apply(self, g):
        for s in g.segments:
            s.major *= 2
            s.minor *= 2
            s.history.append(self)
            s.state_history.append((s.major, s.minor))


@dataclass
class PGD(Event):
    """Partial genome duplication event."""
    chroms: list

    def apply(self, g):
        for s in g.segments:
            if s.chrom in self.chroms:
                s.major += 1
                s.history.append(self)
                s.state_history.append((s.major, s.minor))


@dataclass
class Gain(Event):
    """Segmental copy number gain."""
    chrom: str
    start: int
    end: int
    delta: int

    def apply(self, g):
        g._alter_segmental(self.chrom, self.start, self.end, self.delta, False, self)


@dataclass
class Loss(Event):
    """Segmental copy number loss."""
    chrom: str
    start: int
    end: int
    delta: int

    def apply(self, g):
        g._alter_segmental(self.chrom, self.start, self.end, self.delta, True, self)


def random_arm_span(chrom: str, focal: bool, prop_range=(0.2, 1)):
    """Generate random arm span coordinates."""
    arm_q = _rng.random() < 0.5
    a_start = 0 if not arm_q else int(ARM_BOUNDARY[chrom])
    a_end = int(ARM_BOUNDARY[chrom]) if not arm_q else CHR_LEN[chrom]
    if not focal:
        return a_start, a_end
    arm_len = a_end - a_start
    length = min(arm_len, int(_rng.uniform(1e5, 1e7)))
    offset = int(_rng.integers(0, arm_len - length))
    return a_start + offset, a_start + offset + length


def generate_mutation_dataframe(segment, purity):
    """Generate mutation dataframe from simulated segment."""
    mut_df = []
    mut_counts = segment.muts
    mut_positions = np.arange(segment.start, segment.end)
    rng = np.random.default_rng()
    depth_mu, depth_std = 1.69, 0.22
    depth = lambda: max(1, int(round(10 ** rng.normal(depth_mu, depth_std))))
    total_cn = segment.major + segment.minor

    for multiplicity, count in enumerate(mut_counts, 1):
        if np.isnan(count):
            continue
        for _ in range(int(count)):
            pos = rng.choice(mut_positions)
            n = depth()
            alt_count = rng.binomial(
                n, (multiplicity * purity) / (2 * (1 - purity) + total_cn * purity)
            )
            ref_count = n - alt_count
            vaf = alt_count / n
            mut_df.append(
                {
                    "chrom": segment.chrom,
                    "pos": pos,
                    "ref": "A",
                    "alt": "T",
                    "nalt": alt_count,
                    "nref": ref_count,
                    "start": segment.start,
                    "end": segment.end,
                    "major_cn": segment.major,
                    "minor_cn": segment.minor,
                    "segment_id": f"{segment.chrom}:{segment.start}-{segment.end}",
                    "mut_w": 1,
                    "subclonal": False,
                    "total_cn": segment.major + segment.minor,
                    "vaf": vaf,
                    "multiplicity": multiplicity,
                }
            )

    return pd.DataFrame(mut_df)


def simulate_and_write():
    """Simulate a tumor genome with WGD/PGD events and focal gains/losses."""
    g = SimGenome()
    g.purity = _rng.uniform(0.2, 0.9)
    num_WGD = _rng.choice(2)
    if num_WGD == 0:
        num_PGD = _rng.choice([1, 2, 3])
    else:
        num_PGD = 0
    times = _rng.choice(np.arange(0.1, 1, 0.1), num_WGD + num_PGD, replace=False)
    wgds = [WGD(t) for t in times[:num_WGD]]
    pgd_times = times[num_WGD:]
    chroms = list(CHR_LEN)
    pgds = [
        PGD(t, _rng.choice(chroms, _rng.integers(4, 8), replace=False).tolist())
        for t in pgd_times
    ]
    events = wgds + pgds
    for _ in range(_rng.poisson(20)):
        c = _rng.choice(AUTOSOMES)
        s, e = random_arm_span(c, _rng.random() < 0.3)
        delta = int(_rng.integers(2, 5) if e - s < 1e7 else 1)
        events.append(Gain(_rng.uniform(0.05, 0.95), c, s, e, delta))
    for _ in range(_rng.poisson(20)):
        c = _rng.choice(AUTOSOMES)
        s, e = random_arm_span(c, False)
        events.append(Loss(_rng.uniform(0.05, 0.95), c, s, e, 1))

    for ev in sorted(events, key=lambda x: x.time):
        g.apply(ev)

    g.swap_minor_major()
    g.unify_history_per_state()
    g.assign_tags()
    g.calculate_t_values()
    print("Simulation of tumor complete!")
    print(f"WGDs: {wgds}")
    print(f"PGDs: {pgds}")
    for s in g.segments:
        s.seg_id = f"{s.chrom}:{s.start}-{s.end}"
        s.muts = s.simulate_N_vals(MUT_DIST)
        s.mut_df = generate_mutation_dataframe(s, purity=g.purity)
    return g, events


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


def make_time_df(genome):
    """Create timing dataframe from genome segments."""
    time_df = pd.DataFrame()

    for seg in genome.segments:
        chrom = seg.chrom
        s0 = seg.start
        s1 = seg.end

        timing_result = getattr(seg, "timing_result", {}) or {}
        if not timing_result:
            continue

        euclidean_distances = {}
        for key, value in timing_result.items():
            distance = next(x for x in value["qc"])["dist_rel"]
            euclidean_distances[key] = distance

        keys = np.array(list(timing_result.keys()))
        distance_sorted_keys = keys[np.argsort(list(euclidean_distances.values()))]

        lls = {
            k: next(x for x in timing_result[k]["qc"] if x["loglik"] is not None)["loglik"]
            for k in keys
        }
        if len(lls) == 0:
            continue
        max_ll = max(lls.values())
        probs = {k: np.exp(lls[k] - max_ll) for k in keys}
        total_prob = sum(probs.values())
        probs = {k: p / total_prob for k, p in probs.items()}

        for key in distance_sorted_keys:
            all_draws = timing_result[key].get("draws", [])
            if len(all_draws) == 0:
                continue

            draws_by_bootstrap = {}
            for draw in all_draws:
                b_id = draw.get("boot_id", 0)
                if b_id not in draws_by_bootstrap:
                    draws_by_bootstrap[b_id] = []
                draws_by_bootstrap[b_id].append(draw)

            for b_id in draws_by_bootstrap:
                mat = time_matrix([x["t"] for x in draws_by_bootstrap[b_id]])
                cumulative_times = mat.cumsum(axis=1)[:, :-1]
                num_draws = cumulative_times.shape[0]

                for k in range(cumulative_times.shape[1]):
                    time_df = pd.concat(
                        [
                            time_df,
                            pd.DataFrame(
                                {
                                    "bootstrap_id": b_id,
                                    "chrom": chrom,
                                    "start": s0,
                                    "end": s1,
                                    "segment_id": seg.seg_id,
                                    "key": key,
                                    "major_cn": seg.major_cn,
                                    "minor_cn": seg.minor_cn,
                                    "time_point": k + 1,
                                    "time_fraction": cumulative_times[:, k],
                                    "t": np.diff(
                                        np.pad(cumulative_times, ((0, 0), (1, 0)), mode="constant"),
                                        axis=1,
                                    )[:, k],
                                    "mutation_count": sum(seg.N_counts),
                                    "num_draws": num_draws,
                                    "distance_from_route_boundary": euclidean_distances[key],
                                    "route_probability": probs[key],
                                    "length": seg.total_length
                                    if hasattr(seg, "total_length")
                                    else (seg.end - seg.start),
                                }
                            ),
                        ]
                    )

    if not time_df.empty:
        time_df["draw_weight"] = 1 / time_df["num_draws"]
        time_df["w"] = time_df["draw_weight"] * time_df["route_probability"] * time_df["length"]
        time_df["best_route"] = (
            time_df.groupby("segment_id")["route_probability"].transform("max")
            == time_df["route_probability"]
        )
    return time_df


def cluster_wgd_pgd_events(all_time_df):
    """Cluster timing events into WGD and PGD events."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.cluster import DBSCAN

    MIN_COV_CHR = 0.7
    EPS_CHR = 0.1
    MIN_CLUST = 5

    chr_peaks = []
    good_idx = np.zeros(len(all_time_df), bool)

    for chrom, sub in all_time_df.groupby("chrom"):
        db = DBSCAN(eps=EPS_CHR, min_samples=MIN_CLUST).fit(
            sub[["time_fraction"]], sample_weight=sub["w"]
        )
        sub = sub.assign(cluster=db.labels_)
        print(f"Found {sub['cluster'].nunique()} clusters in {chrom}")

        for c, grp in sub.query("cluster != -1").groupby("cluster"):
            cov_bp = grp.drop_duplicates(subset=["chrom", "start", "end"])["length"].sum()
            chrom_frac = cov_bp / CHR_LEN[chrom]

            if chrom_frac < MIN_COV_CHR:
                continue

            peak_t = np.average(grp["time_fraction"], weights=grp["w"])
            chr_peaks.append(
                dict(chrom=chrom, time=peak_t, cov_frac=chrom_frac, total_weight=grp["w"].sum())
            )
            good_idx[grp.index] = True

    chrom_df = pd.DataFrame(chr_peaks).assign(chr_len=lambda d: d["chrom"].map(CHR_LEN))

    if chrom_df.empty:
        return pd.DataFrame(columns=["event_time", "chroms", "n_chr", "cov_genome", "event_type"])

    DELTA_T = 0.12
    Z = linkage(chrom_df[["time"]], method="single", metric="euclidean")
    chrom_df["event_id"] = fcluster(Z, t=DELTA_T, criterion="distance")

    def summarise(group):
        size_bp = (group["chr_len"] * group["cov_frac"]).sum()
        genome_bp = sum(CHR_LEN.values())
        return pd.Series(
            {
                "event_time": np.average(group["time"], weights=group["total_weight"]),
                "chroms": sorted(group["chrom"], key=lambda c: int(c.replace("chr", ""))),
                "n_chr": group["chrom"].nunique(),
                "cov_genome": size_bp / genome_bp,
            }
        )

    events_df = chrom_df.groupby("event_id").apply(summarise).reset_index(drop=True)

    WGD_CHR_THRESH = 15
    WGD_GENOME_FRAC = 0.60

    events_df["event_type"] = np.where(
        (events_df["n_chr"] >= WGD_CHR_THRESH) | (events_df["cov_genome"] >= WGD_GENOME_FRAC),
        "WGD",
        "PGD",
    )
    return events_df


def cluster_segment_multiplicities(genome):
    """Cluster segments by multiplicity profiles using PCA and DBSCAN."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN

    max_cn = max(seg.major_cn for seg in genome.segments)

    records = []
    for major_cn in range(2, max_cn):
        for minor_cn in range(0, major_cn + 1):
            seg_subset = [
                seg
                for seg in genome.segments
                if seg.major_cn == major_cn and seg.minor_cn == minor_cn
            ]
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

            if len(points) <= 1:
                for seg in seg_subset:
                    records.append(
                        {
                            "seg_id": seg.seg_id,
                            "major_cn": major_cn,
                            "minor_cn": minor_cn,
                            "cluster_label": -1,
                            "pca_1": np.nan,
                            "pca_2": np.nan,
                            "pca_1_contribution": np.nan,
                            "pca_2_contribution": np.nan,
                            "N_counts": seg.N_counts,
                        }
                    )
                continue

            points = np.array(points)
            X = points[:, : len(seg.N_counts)]
            weights = points[:, -1]
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            explained_variance = pca.explained_variance_ratio_
            labels = DBSCAN(eps=0.05, min_samples=5).fit_predict(X_pca, sample_weight=weights)

            for i, seg_id in enumerate(seg_ids):
                seg = next((s for s in genome.segments if s.seg_id == seg_id), None)
                records.append(
                    {
                        "seg_id": seg_id,
                        "major_cn": major_cn,
                        "minor_cn": minor_cn,
                        "cluster_label": labels[i],
                        "pca_1": X_pca[i, 0],
                        "pca_2": X_pca[i, 1],
                        "pca_1_contribution": explained_variance[0],
                        "pca_2_contribution": explained_variance[1],
                        "N_counts": seg.N_counts,
                    }
                )

    return pd.DataFrame(records)
