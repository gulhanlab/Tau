#!/usr/bin/env python3
"""
tau/core.py - Core data structures for Tau timing analysis.

Contains Segment and Genome classes with EM-based multiplicity estimation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binom
from tqdm import tqdm

EPS = 1e-12


def truncated_binom_pmf(
    x: int, n: int, p: float, *, min_alt: int = 3, min_vaf: float | None = None
) -> float:
    """Binomial PMF truncated below the detection threshold."""
    if n <= 0 or p <= 0 or p >= 1:
        return 0.0
    kmin = max(min_alt, int(np.ceil(min_vaf * n))) if min_vaf is not None else int(min_alt)
    if x < kmin:
        return 0.0
    base = binom.pmf(x, n, p)
    norm = 1.0 - binom.cdf(kmin - 1, n, p)
    return float(base / max(norm, EPS))


def _em_pi(
    L: np.ndarray,
    w: np.ndarray | None = None,
    tol: float = 1e-7,
    max_iter: int = 200,
    pi0: np.ndarray | None = None,
):
    """EM for mixture weights π given per-SNV likelihoods L (N x K) and SNV weights w (size N)."""
    L = np.asarray(L, float)
    n, K = L.shape
    w = np.ones(n, float) if (w is None) else np.maximum(np.asarray(w, float), 0.0)
    pi = (np.ones(K) / K) if pi0 is None else np.maximum(np.asarray(pi0, float), 0.0)
    pi /= (pi.sum() + EPS)

    for _ in range(max_iter):
        f = L @ pi + EPS
        P = (L * pi[None, :]) / f[:, None]
        pi_new = (w[:, None] * P).sum(axis=0) / (w.sum() + EPS)
        if np.max(np.abs(pi_new - pi)) < tol:
            pi = pi_new
            break
        pi = pi_new
    return pi, P


def _ess(w: np.ndarray) -> float:
    """Effective sample size for weights w."""
    w = np.asarray(w, float)
    s1 = w.sum()
    s2 = (w * w).sum()
    return float((s1 * s1) / s2) if s2 > 0 else 0.0


def _segment_length(start: int, end: int) -> int:
    return int(max(0, int(end) - int(start)))


@dataclass
class Segment:
    """One CN segment with its SNVs and EM-estimated multiplicity mixture."""
    chrom: str
    start: int
    end: int
    major_cn: int
    minor_cn: int
    purity: float
    snv_table: pd.DataFrame = field(repr=False)
    seg_id: str | None = None
    N_counts: np.ndarray | None = None
    pi_em: np.ndarray | None = None
    N_counts_boot: np.ndarray | None = None   # bootstrapped N_counts replicates
    min_alt: int = 3
    min_vaf: float | None = None

    def __post_init__(self):
        if self.seg_id is None:
            self.seg_id = f"{self.chrom}:{self.start}-{self.end}"
        self._compute_multiplicity_likelihoods()

    @property
    def total_cn(self) -> int:
        return int(self.major_cn + self.minor_cn)

    @property
    def n_snvs(self) -> int:
        return int(len(self.snv_table))

    def _compute_multiplicity_likelihoods(self):
        """Pre-compute per-SNV likelihoods over multiplicities 1..major_cn."""
        if self.major_cn <= 0 or self.n_snvs == 0:
            self.snv_table["multiplicity_likelihoods"] = [[]] * self.n_snvs
            return

        K = int(self.major_cn)
        y_vals = np.arange(1, K + 1)
        denom = self.purity * self.total_cn + (1.0 - self.purity) * 2.0
        p_vec = (self.purity * y_vals) / max(denom, EPS)

        likes = []
        nalt_col = pd.to_numeric(self.snv_table.get("nalt", np.nan), errors="coerce").to_numpy()
        nref_col = pd.to_numeric(self.snv_table.get("nref", np.nan), errors="coerce").to_numpy()
        for nalt, nref in zip(nalt_col, nref_col):
            if np.isnan(nalt) or np.isnan(nref):
                likes.append([EPS] * K)
                continue
            nalt = int(nalt); nref = int(nref); n = nalt + nref
            L = [
                truncated_binom_pmf(nalt, n, float(p), min_alt=self.min_alt, min_vaf=self.min_vaf)
                for p in p_vec
            ]
            likes.append([max(v, EPS) for v in L])

        self.snv_table["multiplicity_likelihoods"] = likes

    def estimate_pi_em(self, bootstrap_B: int | None = None, random_state: int | None = None):
        """
        Run EM on this segment; fill pi_em, N_counts, and per-SNV posteriors.
        Optionally perform fixed-ESS bootstrap with B replicates.
        """
        K = int(self.major_cn)
        if K <= 0 or self.n_snvs == 0:
            K = max(K, 1)
            self.pi_em = np.ones(K) / K
            self.N_counts = np.zeros(K)
            return

        if "multiplicity_likelihoods" not in self.snv_table.columns:
            self._compute_multiplicity_likelihoods()

        L = np.vstack(self.snv_table["multiplicity_likelihoods"].to_numpy()).astype(float)

        # SNV weights
        if "mut_w" in self.snv_table.columns:
            w = pd.to_numeric(self.snv_table["mut_w"], errors="coerce").fillna(0.0).to_numpy().astype(float)
            if not np.any(w > 0):
                w = np.ones(len(L), float)
        else:
            w = np.ones(len(L), float)

        # Remove subclonal SNVs from EM
        if "subclonal" in self.snv_table.columns:
            keep = ~self.snv_table["subclonal"].fillna(False).to_numpy()
            w = w * keep.astype(float)

        pi, P = _em_pi(L, w=w)
        ess = _ess(w)
        self.pi_em = pi
        self.N_counts = pi * ess

        idx = self.snv_table.index
        self.snv_table.loc[idx, "posterior"] = pd.Series([row.tolist() for row in P], index=idx, dtype=object)
        self.snv_table.loc[idx, "MAP_multiplicity"] = pd.Series((P.argmax(1) + 1).astype(int), index=idx, dtype="int64")

        rng = np.random.default_rng(random_state)
        B = 1 if bootstrap_B is None or bootstrap_B <= 0 else int(bootstrap_B)
        boots = np.zeros((B, K), float)
        boots[0, :] = self.N_counts

        if B > 1:
            p = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
            for b in range(1, B):
                # weighted bootstrap resample
                ix = rng.choice(len(L), size=len(L), replace=True, p=p)
                Lb = L[ix]
                wb = w[ix]
                pi_b, _ = _em_pi(Lb, w=wb)
                boots[b, :] = pi_b * ess  # fixed ESS

        self.N_counts_boot = boots


@dataclass
class Genome:
    segments: list[Segment] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        snv_table: pd.DataFrame,
        *,
        purity: float,
        detect_min_alt: int = 3,
        detect_min_vaf: float | None = None,
    ) -> "Genome":
        """Build Genome from a preprocessed mutation table (output of tau_preprocessing.py)."""
        required = {"chrom","pos","nalt","nref","start","end","major_cn","minor_cn","segment_id"}
        missing = required - set(snv_table.columns)
        if missing:
            raise ValueError(f"Preprocessed table missing required columns: {sorted(missing)}")

        segs: list[Segment] = []
        keys = ["chrom", "start", "end", "major_cn", "minor_cn", "segment_id"]
        for _, sub in snv_table.groupby(keys, sort=False):
            chrom, start, end, maj, mino, sid = sub.iloc[0][keys]
            segs.append(
                Segment(
                    chrom=str(chrom), start=int(start), end=int(end),
                    major_cn=int(maj), minor_cn=int(mino), purity=float(purity),
                    seg_id=str(sid),
                    snv_table=sub.reset_index(drop=True).copy(),
                    min_alt=detect_min_alt, min_vaf=detect_min_vaf,
                )
            )
        return cls(segs)

    def calculate_multiplicities(self, bootstrap_B: int = 1, random_state: int | None = None):
        for s in tqdm(self.segments, desc="Estimating multiplicities", unit="segment"):
            s.estimate_pi_em(bootstrap_B=bootstrap_B, random_state=random_state)

    def to_segment_summary(self) -> pd.DataFrame:
        """One row per segment with π and N_counts exploded into columns."""
        rows = []
        for s in self.segments:
            if s.pi_em is None or s.N_counts is None:
                continue
            K = int(s.major_cn)
            row = dict(
                seg_id=s.seg_id, chrom=s.chrom, start=int(s.start), end=int(s.end),
                major_cn=int(s.major_cn), minor_cn=int(s.minor_cn),
                n_snvs=int(s.n_snvs)
            )
            for k in range(1, K + 1):
                row[f"pi_{k}"] = float(s.pi_em[k-1])
                row[f"N_{k}"]  = float(s.N_counts[k-1])
            if s.N_counts_boot is not None:
                # add bootstrapped summary stats
                row["boot_mean_effN"] = float(np.nanmean(s.N_counts_boot.sum(axis=1)))
                row["boot_sd_effN"]   = float(np.nanstd(s.N_counts_boot.sum(axis=1)))
            rows.append(row)
        return pd.DataFrame(rows)


def _cli():
    ap = argparse.ArgumentParser(description="Tau multiplicity EM on a preprocessed mutation table.")
    ap.add_argument("--in", dest="in_path", required=True, help="Preprocessed table (.tsv)")
    ap.add_argument("--purity", type=float, required=True)
    ap.add_argument("--min-alt", type=int, default=3)
    ap.add_argument("--min-vaf", type=float, default=None)
    ap.add_argument("--bootstrap", type=int, default=0, help="Number of fixed-ESS bootstrap replicates")
    ap.add_argument("--out-seg", default=None, help="Write segment summary TSV")
    args = ap.parse_args()

    in_path = Path(args.in_path)

    df = pd.read_csv(in_path, sep="\t")

    g = Genome.create(
        df,
        purity=float(args.purity),
        detect_min_alt=int(args.min_alt),
        detect_min_vaf=args.min_vaf,
    )
    g.calculate_multiplicities(bootstrap_B=int(args.bootstrap))

    if args.out_seg:
        seg_df = g.to_segment_summary()
        Path(args.out_seg).parent.mkdir(parents=True, exist_ok=True)
        seg_df.to_csv(args.out_seg, sep="\t", index=False)


if __name__ == "__main__":
    _cli()
