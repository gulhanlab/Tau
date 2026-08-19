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
from tau.lineage import LineageTracker, class_of
from tau.preprocessing import preprocess_sample


def _get_data_path(filename):
    """Get path to package data file."""
    try:
        return str(resources.files("tau.data").joinpath(filename))
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
    if not os.path.exists(path):
        # Normal installs ship only the ~3 MB packed matrices for solution-backed
        # routes; the full HDF5 is a build-time artifact.
        from tau.timing import load_packed_matrices
        packed = load_packed_matrices()
        if packed:
            _MATRICES = dict(packed)
            return _MATRICES
        raise FileNotFoundError(
            f"No route matrices found: neither {path} nor the packaged "
            f"route_matrices.npz is available."
        )
    with h5py.File(path, "r") as h5:
        _MATRICES = {k: h5[k][()] for k in h5}
    return _MATRICES


_t_pat = re.compile(r"t(\d+)$")
_rng = np.random.default_rng()
which_arm = lambda c, p: "p" if p < ARM_BOUNDARY[c] else "q"

MUT_DIST = (-5.5, 0)


def route_degeneracy_set(t_vals, tags, matrices, tol=1e-6):
    """Group candidate routes by the N they produce for these t_vals; return the
    correct compatible set + its N.

    Two routes are indistinguishable for a given history iff MATRIX.T @ t_vals is
    equal. Co-temporal events (zeroed t-intervals) collapse intermediate
    multiplicities, so the correct cluster is the one occupying the FEWEST
    multiplicities (sparsest N). Returns (list_of_tags, normalised_N) when a
    uniquely-sparsest cluster exists, else (None, None) -> caller falls back."""
    t_vals = np.asarray(t_vals, float)
    Ns = {}
    for tag in tags:
        N = matrices[tag].T @ t_vals
        if N.sum() > 0:
            Ns[tag] = N / N.sum()
    if not Ns:
        return None, None
    clusters = []  # [normalised_N, [tags]]
    for tag, N in Ns.items():
        for cl in clusters:
            if np.allclose(cl[0], N, atol=1e-5):
                cl[1].append(tag)
                break
        else:
            clusters.append([N, [tag]])
    nnz = lambda N: int(np.sum(N > tol))
    clusters.sort(key=lambda cl: nnz(cl[0]))
    if len(clusters) == 1 or nnz(clusters[0][0]) < nnz(clusters[1][0]):
        return sorted(clusters[0][1], key=lambda t: int(str(t).split(".")[1])), clusters[0][0]
    return None, None


def choose_allele(seg, loss=False):
    """Which PHYSICAL allele ("A"/"B") an event lands on.

    Replaces choose_allele_delta, which returned a (d_major, d_minor) pair. Under lineage tracing the
    major/minor LABEL is emergent — whichever allele ends with more surviving copies is the major one —
    so the simulator has to name a physical allele rather than a role. Naming a role is precisely what
    produced the phantom gains: a WGD on (1,1) counted a major and a minor gain, and if the minor allele
    was later lost the surviving (4,0) kept a gain slot for a copy that no longer existed.

    Probabilities mirror the old behaviour with `hi`/`lo` (current copy counts) standing in for
    major/minor: losses target an allele in proportion to its copy number, gains land on the larger
    allele about three times out of four.
    """
    n = seg.lin.counts()
    hi, lo = ("A", "B") if n["A"] >= n["B"] else ("B", "A")
    tot = n[hi] + n[lo]
    if tot == 0:
        return hi
    if loss:
        if n[lo] == 0:
            return hi
        if n[hi] == 0:
            return lo
        return lo if _rng.random() < n[lo] / tot else hi
    if n[lo] == 0:
        return hi
    return lo if _rng.random() < n[lo] / tot * 0.25 else hi


@dataclass
class Event:
    """Base class for genomic events."""
    time: float

    def apply(self, genome: "SimGenome"):
        raise NotImplementedError


@dataclass
class SimSegment:
    """Simulated genomic segment. Its history is a traced lineage, not a list of CN states.

    `major`/`minor` are a CACHE of `lin.final_state()`, refreshed after every event so code reading the
    current state (and the .cna.txt writer) keeps working. They are OUTPUTS of the tree, never the
    thing an event mutates.
    """
    chrom: str
    start: int
    end: int
    major: int
    minor: int
    history: list = field(default_factory=list)
    muts: np.ndarray | None = None
    lin: object = None

    def __post_init__(self):
        if self.lin is None:
            self.lin = LineageTracker(rng=_rng)

    def copy(self):
        return SimSegment(self.chrom, self.start, self.end, self.major, self.minor,
                          self.history.copy(), lin=self.lin.copy())

    def refresh(self):
        """Re-read (major, minor) from the surviving copies."""
        self.major, self.minor = self.lin.final_state()

    def finalize(self, matrices):
        """Read the traced tree into the fields the rest of the pipeline expects.

        Sets t_vals, tag, route_class (STRUCTURAL truth — what actually happened) and equiv_class
        (what a method could have RECOVERED, since co-temporal splits let structurally different
        routes predict an identical N). Route selection should be scored against equiv_class;
        route_class is what the simulator did.
        """
        self.refresh()
        _evs, t_vals = self.lin.topology()
        n_int = max(self.major - 1, 0) + max(self.minor - 1, 0) + 1
        self.route_class, self.equiv_class, self.route_N = [], [], None
        if n_int <= 1 or self.major <= 0:
            self.t_vals, self.tag = [1.0], None
            return
        # A state's interval count is fixed by (major, minor); topology() yields one interval per
        # split plus the trailing one. They agree whenever the tree is consistent with the state.
        if len(t_vals) != n_int:
            self.t_vals, self.tag = None, None
            return
        self.t_vals = t_vals
        self.route_class = self.lin.route_class()
        self.tag = self.route_class[0] if self.route_class else None
        if self.tag is None or self.tag not in matrices:
            self.equiv_class = list(self.route_class)
            return
        self.equiv_class = class_of(self.tag, t_vals, self.major, self.minor, matrices)
        N = matrices[self.tag].T @ np.asarray(t_vals, float)
        self.route_N = (N / N.sum()) if N.sum() > 0 else None

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

        # Prefer the route_set's shared N: when co-temporal events zero a t-interval several routes
        # collapse to the SAME multiplicity vector, and that cluster's N is the physically correct one.
        # Falling back to MATRICES[tag] would re-introduce the inconsistency this fixes.
        route_N = getattr(self, "route_N", None)
        if route_N is not None:
            N_vals = np.asarray(route_N, float)
        else:
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
                SimSegment(chrom, 1, b, 1, 1),
                SimSegment(chrom, b, length, 1, 1),
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
        hit = [s for s in self.segments
               if s.chrom == chrom and start < s.end and end > s.start]
        if not hit:
            return
        # ONE allele for the whole event, not one per segment. A focal gain or loss amplifies or
        # removes a copy of a single physical chromosome across its whole span, so every segment it
        # covers must move on the SAME allele; drawing per segment would let one event gain allele A
        # in one segment and allele B in the next. Chosen from the first covered segment.
        allele = choose_allele(hit[0], loss)
        for seg in hit:
            if loss:
                for _ in range(max(1, int(delta))):
                    seg.lin.loss(ev.time, allele)
            else:
                seg.lin.gain(ev.time, allele, delta)
            seg.history.append(ev)
            seg.refresh()

    def apply(self, ev: Event):
        ev.apply(self)

    def finalize(self):
        """Read every segment's traced tree into t_vals / tag / route_class / equiv_class.

        Replaces the old swap_minor_major + unify_history_per_state + assign_tags +
        calculate_t_values + assign_route_sets chain. Nothing to swap (final_state returns
        (max, min), so major >= minor by construction) and nothing to unify (each segment's history
        is its own tree; forcing neighbours to share one was a way of papering over the fact that
        state histories could not represent what actually happened).
        """
        matrices = _load_matrices()
        for s in self.segments:
            s.finalize(matrices)

    def simulate_N_vals(self):
        """Simulate mutation counts for all segments."""
        for s in self.segments:
            s.muts = s.simulate_N_vals(MUT_DIST)
        print("Simulation complete!")


@dataclass
class WGD(Event):
    """Whole genome duplication event.

    Every LIVING copy splits in two — which is what makes a doubling a doubling, and why this cannot
    be written as `major *= 2; minor *= 2`. Under the arithmetic form a doubling of (1,1) booked one
    major and one minor gain; if the minor allele was lost afterwards the surviving (4,0) still
    carried a gain slot for a copy that no longer existed (the phantom gain, 180 of 401 v8 (4,0)
    segments). A copy that dies is simply absent from the tree, so it cannot occupy a slot.
    """
    def apply(self, g):
        for s in g.segments:
            s.lin.wgd(self.time)
            s.history.append(self)
            s.refresh()


@dataclass
class PGD(Event):
    """Partial genome duplication event — one extra copy of one allele on the named chromosomes."""
    chroms: list

    def apply(self, g):
        # One allele per CHROMOSOME, drawn once and applied to every segment on it: a ccPG duplicates
        # whole chromosomes, so a chromosome cannot gain allele A in one segment and allele B in the
        # next. Chromosomes are independent physical entities, so the draw is per chromosome rather
        # than one for the entire event.
        for chrom in self.chroms:
            segs = [s for s in g.segments if s.chrom == chrom]
            if not segs:
                continue
            allele = choose_allele(segs[0], loss=False)
            for s in segs:
                s.lin.gain(self.time, allele, 1)
                s.history.append(self)
                s.refresh()


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
    # Seed this segment's RNG off the (seeded) global stream so VCF positions/depths are
    # reproducible from the run seed; previously default_rng() was unseeded.
    rng = np.random.default_rng(np.random.randint(2 ** 31))
    depth_mu, depth_std = 1.69, 0.22
    depth = lambda: max(1, int(round(10 ** rng.normal(depth_mu, depth_std))))
    total_cn = segment.major + segment.minor

    for multiplicity, count in enumerate(mut_counts, 1):
        if np.isnan(count):
            continue
        for _ in range(int(count)):
            pos = int(rng.integers(segment.start, segment.end))   # avoid a per-segment arange (could be ~1.5 GB for whole-chrom segments)
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


def simulate_demo(seed: int = 0, allowed_states=((2, 1), (2, 2), (4, 2))):
    """Synthesise a small, fast, self-contained demo genome (no external files).

    Builds a SimGenome with a single whole-genome duplication plus a couple of
    focal gains, restricted to a handful of chromosomes and to a small set of
    copy-number states so that route loading stays bounded (one WGD at t=0.30,
    a focal gain producing a (3,1) and a (4,2) state).

    Returns
    -------
    mut_df : pd.DataFrame
        Per-mutation table ready for ``Genome.create`` (same columns the
        preprocessing pipeline emits, with ``mut_w=1`` so all mutations are
        weighted equally — equivalent to ``signatures=None``).
    purity : float
        The purity used to draw the read counts.
    truth : dict
        Ground-truth event times, e.g. ``{"WGD": 0.30}`` — handy for the demo
        message and for sanity-checking the recovered timing.

    The set of CN states that actually carry simulated mutations is exactly
    ``allowed_states``; pass these to ``load_routes_for_states`` to keep the
    Sage load fast.
    """
    global _rng
    _rng = np.random.default_rng(seed)
    np.random.seed(seed)

    allowed = set(tuple(s) for s in allowed_states)
    purity = 0.80
    wgd_time = 0.30

    # Use a subset of chromosomes. Route loading only depends on the (small) set
    # of CN *states* (allowed_states), not the number of chromosomes, so we keep
    # enough chromosomes for the WGD signal to cluster cleanly.
    demo_chroms = [
        "chr1", "chr2", "chr3", "chr4", "chr5", "chr6",
        "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr17",
    ]

    g = SimGenome()
    g.purity = purity
    # Drop chromosomes we don't use so simulate_N_vals stays cheap.
    g.segments = [s for s in g.segments if s.chrom in demo_chroms]

    events = [WGD(wgd_time)]
    # Pre-WGD gain on chr17p: (1,1) -> (2,1) at t=0.15, doubled by WGD -> (4,2).
    events.append(Gain(0.15, "chr17", 1, int(ARM_BOUNDARY["chr17"]), 1))
    # Post-WGD single-allele loss on chr2p: (2,2) -> (2,1) at t=0.70.
    events.append(Loss(0.70, "chr2", 1, int(ARM_BOUNDARY["chr2"]), 1))

    for ev in sorted(events, key=lambda x: x.time):
        g.apply(ev)

    # One call: lineage tracing yields (major, minor), t_vals and the true route together, so there
    # is no separate tag lookup to keep consistent with the timing.
    g.finalize()

    frames = []
    for s in g.segments:
        s.seg_id = f"{s.chrom}:{s.start}-{s.end}"
        if s.t_vals is None or (s.major, s.minor) not in allowed:
            continue
        s.muts = s.simulate_N_vals(MUT_DIST)
        df = generate_mutation_dataframe(s, purity=g.purity)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("Demo simulation produced no mutations; try a different seed.")
    mut_df = pd.concat(frames, ignore_index=True)
    # Match the real preprocessing pipeline, which strips the "chr" prefix
    # (downstream clustering does int(seg.chrom)).
    mut_df["chrom"] = mut_df["chrom"].astype(str).str.replace("^chr", "", regex=True)
    mut_df["segment_id"] = (
        mut_df["chrom"] + ":"
        + mut_df["start"].astype(int).astype(str) + "-"
        + mut_df["end"].astype(int).astype(str)
    )
    return mut_df, purity, {"WGD": wgd_time}


def simulate_and_write():
    """Simulate a tumor genome with WGD/PGD events and focal gains/losses.

    WGD count drawn from {0, 1, 2} with probabilities (0.40, 0.40, 0.20).
    For 2-WGD: times are sampled so they are at least 0.3 apart.
    """
    g = SimGenome()
    g.purity = _rng.uniform(0.2, 0.9)
    num_WGD = int(_rng.choice([0, 1, 2], p=[0.40, 0.40, 0.20]))
    if num_WGD == 0:
        num_PGD = _rng.choice([0, 1, 2, 3], p=[0.40, 0.35, 0.15, 0.10])
    else:
        # WGD samples can also carry PGD events (~30% chance of at least one PGD)
        num_PGD = _rng.choice([0, 1, 2], p=[0.70, 0.20, 0.10])

    # Sample WGD times: for 2 WGDs enforce a minimum gap of 0.6
    if num_WGD == 2:
        t1 = _rng.uniform(0.1, 0.60)
        t2 = _rng.uniform(t1 + 0.30, min(t1 + 0.30 + 0.40, 0.95))
        wgd_times = [t1, t2]
    else:
        # Continuous event times (v7): de-discretized from the old 0.1 grid so the
        # timing-accuracy and resolution panels are measured continuously.
        wgd_times = list(_rng.uniform(0.1, 0.9, num_WGD))

    wgds = [WGD(t) for t in wgd_times]

    pgd_times = list(_rng.uniform(0.1, 0.9, num_PGD))
    chroms = list(CHR_LEN)
    pgds = [
        PGD(t, _rng.choice(chroms, _rng.integers(2, 6), replace=False).tolist())
        for t in pgd_times
    ]
    events = wgds + pgds
    for _ in range(_rng.poisson(10)):
        c = _rng.choice(AUTOSOMES)
        s, e = random_arm_span(c, _rng.random() < 0.7)
        delta = int(_rng.integers(2, 5) if e - s < 1e7 else 1)
        events.append(Gain(_rng.uniform(0.05, 0.95), c, s, e, delta))
    for _ in range(_rng.poisson(10)):
        c = _rng.choice(AUTOSOMES)
        s, e = random_arm_span(c, False)
        events.append(Loss(_rng.uniform(0.05, 0.95), c, s, e, 1))
    # Post-WGD arm-level losses: two independent passes per WGD.
    # Pass 1 (p=0.45): converts (2,2) → (2,1) — increases LOH, no change to frac_major_ge2.
    # Pass 2 (p=0.40): applied independently; arms that already lost once go (2,1) → (1,1)
    #   with 2/3 probability (random major targeting), reducing frac_major_ge2 by ~0.12.
    # Together these bring simulated WGD=1 frac_major_ge2 from ~0.95 down to ~0.83,
    # matching the PCAWG WGD cluster.
    for wgd in wgds:
        t_wgd = wgd.time
        t_lo1 = min(t_wgd + 0.05, 0.94)
        t_lo2 = min(t_wgd + 0.15, 0.94)
        for chrom in AUTOSOMES:
            for arm_is_q in [False, True]:
                a_start = int(ARM_BOUNDARY[chrom]) if arm_is_q else 0
                a_end = CHR_LEN[chrom] if arm_is_q else int(ARM_BOUNDARY[chrom])
                if _rng.random() < 0.45:
                    t = _rng.uniform(t_lo1, 0.95) if t_lo1 < 0.95 else 0.94
                    events.append(Loss(t, chrom, a_start, a_end, 1))
                if _rng.random() < 0.40:
                    t = _rng.uniform(t_lo2, 0.95) if t_lo2 < 0.95 else 0.94
                    events.append(Loss(t, chrom, a_start, a_end, 1))

    for ev in sorted(events, key=lambda x: x.time):
        g.apply(ev)

    # The route now comes from the traced lineage, so it IS the simulated history rather than a
    # lookup that has to agree with it. counts_diagram_As.txt (no minor==0 rows, so every LOH state
    # fell through to "<M>_<m>.1") is out of the loop entirely.
    g.finalize()
    print("Simulation of tumor complete!")
    print(f"WGDs: {wgds}")
    print(f"PGDs: {pgds}")
    for s in g.segments:
        s.seg_id = f"{s.chrom}:{s.start}-{s.end}"
        if s.t_vals is None:
            # Segment has an over-complex history (gains reversed by losses).
            # No valid Tau route maps to it; skip mutation simulation.
            s.muts = np.zeros(1, dtype=int)
            s.mut_df = pd.DataFrame(columns=["chrom","pos","ref","alt","nalt","nref",
                                             "start","end","major_cn","minor_cn",
                                             "segment_id","mut_w","vaf","multiplicity"])
            continue
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
