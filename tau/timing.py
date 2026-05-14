#!/usr/bin/env python3
"""
tau/timing.py - Timing analysis for copy number segments.

Loads pre-computed Sage solutions and constraint matrices, solves timing equations via grid sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional
import importlib.resources

import operator
import os
from functools import lru_cache
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
from sage.all import var, load as sage_load

from tau.core import Segment, Genome


def _get_package_data_path(filename: str) -> Path:
    """Get the path to a package data file."""
    try:
        # Python 3.9+
        with importlib.resources.files("tau.data").joinpath(filename) as p:
            return Path(p)
    except (AttributeError, TypeError):
        # Fallback for older Python
        import pkg_resources
        return Path(pkg_resources.resource_filename("tau", f"data/{filename}"))


def _default_paths() -> tuple[Path, Path]:
    """
    Return default paths for route solutions and matrices.

    Priority:
    1. Environment variables TAU_ROUTES_SAGE and TAU_MATRICES_H5
    2. Package data files in tau/data/
    """
    # Check environment variables first
    sage_env = os.environ.get("TAU_ROUTES_SAGE")
    h5_env = os.environ.get("TAU_MATRICES_H5")

    if sage_env and h5_env:
        return Path(sage_env), Path(h5_env)

    # Fall back to package data
    sage_path = sage_env or _get_package_data_path("7_5_solutions_updated.sobj")
    h5_path = h5_env or _get_package_data_path("matrices_7_7.h5")

    return Path(sage_path), Path(h5_path)


@dataclass
class RouteEnv:
    """Container for route solutions and constraint matrices."""
    SOL: Dict[str, Any]
    MATRICES: Dict[str, np.ndarray]
    free_var_cache: Dict[str, int]
    route_cache: Dict[str, Any]


def load_routes(sol_file: str | Path, matrix_h5: str | Path) -> RouteEnv:
    """Load route solutions and matrices from files.

    If *sol_file* is a directory, it is assumed to contain per-CN-state
    .sobj files produced by ``tau/data/split_solutions.py``
    (e.g. ``solutions/5_1.sobj``).  All files in the directory are loaded
    and merged into a single SOL dict — this is much faster for large
    analyses that only need a subset of CN states because callers can pass
    a list of filenames rather than the full monolithic file.
    """
    sol_file = Path(sol_file)
    if sol_file.is_dir():
        SOL = {}
        for p in sorted(sol_file.glob("*.sobj")):
            SOL.update(sage_load(str(p)))
        print(f"Loaded {len(SOL):,} routes from {sol_file}/")
    else:
        print(f"Loading route solutions from {sol_file}...")
        SOL = sage_load(str(sol_file))

    print(f"Loading route matrices from {matrix_h5}...")
    with h5py.File(str(matrix_h5), "r") as h5:
        MATRICES = {k: h5[k][()] for k in h5}
    return RouteEnv(SOL=SOL, MATRICES=MATRICES, free_var_cache={}, route_cache={})


def load_routes_for_states(
    cn_states: list[tuple[int, int]],
    matrix_h5: str | Path | None = None,
) -> RouteEnv:
    """Load solutions only for the requested CN states.

    Looks for per-state .sobj files in ``tau/data/solutions/``.
    Falls back to the monolithic file if the split directory does not exist.

    Parameters
    ----------
    cn_states:
        List of (major, minor) tuples, e.g. ``[(5, 1), (4, 0)]``.
    matrix_h5:
        Path to the HDF5 matrices file.  Defaults to the package data file.
    """
    split_dir = _get_package_data_path("solutions")
    if matrix_h5 is None:
        _, matrix_h5 = _default_paths()

    if not Path(split_dir).is_dir():
        # Fall back to monolithic load
        sage_path, _ = _default_paths()
        return load_routes(sage_path, matrix_h5)

    SOL: Dict[str, Any] = {}
    for major, minor in cn_states:
        p = Path(split_dir) / f"{major}_{minor}.sobj"
        if p.exists():
            SOL.update(sage_load(str(p)))
        else:
            print(f"  Warning: no split file for ({major},{minor}), skipping")

    print(f"Loaded {len(SOL):,} routes for {len(cn_states)} CN states")
    with h5py.File(str(matrix_h5), "r") as h5:
        MATRICES = {k: h5[k][()] for k in h5}
    return RouteEnv(SOL=SOL, MATRICES=MATRICES, free_var_cache={}, route_cache={})


@lru_cache(maxsize=1)
def _get_route_env(
    sage_path: str | Path | None = None,
    matrices_path: str | Path | None = None,
) -> RouteEnv:
    """Lazy, cached default RouteEnv; respects env overrides if provided."""
    if sage_path is None or matrices_path is None:
        sage_default, h5_default = _default_paths()
        sage_path = sage_default if sage_path is None else sage_path
        matrices_path = h5_default if matrices_path is None else matrices_path

    return load_routes(sage_path, matrices_path)


def calculate_free_vars(A: np.ndarray) -> int:
    """Number of free variables in the linear system determined by A."""
    rank = np.linalg.matrix_rank(A)
    return int(A.shape[1] - rank)


def calc_total_sum(n_vals: Iterable[float], total_cn: float) -> float:
    """Weighted total sum for normalizing t-variables."""
    n_vals = list(n_vals)
    return float(sum((i + 1) * n for i, n in enumerate(n_vals)) / total_cn)


def generate_grid(n: int, bounds_list: list[dict], idx: int | None = None, eps=1e-8):
    """
    Build a Cartesian grid over free t-variable bounds. Guaranteed non-empty:
    if an interval is narrower than 2*eps, pick its midpoint once.
    """
    if not bounds_list:
        return [dict()]

    if idx is None:
        m = len(bounds_list)
        idx = m - 1
        num_per = int(np.ceil(n ** (1 / max(m, 1))))
        return generate_grid(num_per, bounds_list, idx, eps=eps)

    t_var = bounds_list[idx]['t_var']
    upper = bounds_list[idx]['upper']
    lower = bounds_list[idx]['lower']

    if idx == 0:
        try:
            lo = float(lower); hi = float(upper)
        except (TypeError, ValueError):
            return []
        if not np.isfinite(lo) or not np.isfinite(hi):
            return []
        if hi - lo <= 2*eps:
            mid = 0.5*(lo + hi)
            return [{t_var: mid}]
        xs = np.linspace(lo+eps, hi-eps, n)
        if xs.size == 0:
            xs = np.array([0.5*(lo+hi)])
        return [{t_var: float(x)} for x in xs]

    grid = generate_grid(n, bounds_list, idx - 1, eps=eps)
    out = []
    for g in grid:
        try:
            lo = float(lower.subs(g)); hi = float(upper.subs(g))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if hi - lo <= 2*eps:
            out.append({**g, t_var: 0.5*(lo+hi)})
        else:
            xs = np.linspace(lo+eps, hi-eps, n)
            if xs.size == 0:
                xs = np.array([0.5*(lo+hi)])
            for val in xs:
                out.append({**g, t_var: float(val)})
    return out


def _counts_from(seg: Segment) -> np.ndarray:
    """Use EM-derived counts by default (stable, weighted)."""
    K = int(seg.major_cn)
    if getattr(seg, "N_counts", None) is None:
        raise ValueError("seg.N_counts missing; run EM first on multiplicities.")
    v = np.asarray(seg.N_counts, float)
    if len(v) != K:
        raise ValueError("N_counts length != major_cn.")
    return v


def _counts_iter(seg: Segment, method: str = "em"):
    """
    Unified iterator: always iterate over seg.N_counts_boot (B replicates),
    which includes the original EM counts as the first replicate.
    """
    Nboot = getattr(seg, "N_counts_boot", None)
    if isinstance(Nboot, np.ndarray) and Nboot.ndim == 2:
        for b in range(Nboot.shape[0]):
            yield np.asarray(Nboot[b, :], float), {"boot": int(b)}
        return

    # fallback: no boot info (should be rare)
    nn = getattr(seg, "N_counts", None)
    if nn is not None:
        yield np.asarray(nn, float), {"boot": 0}


def prepare_route(env: RouteEnv, key: str, major: int, minor: int):
    """
    Extracts and caches route-specific components:
      sols:   equalities for t-variables (possibly involving free vars)
      t_ineqs / N_ineqs: bounds/constraints
      t_vals, num_t_vals: number and symbolic t-variables
      free_var_bounds: dict[t_var] -> {'lower': expr, 'upper': expr}
    """
    if key in env.route_cache:
        return env.route_cache[key]

    exprs = env.SOL[key]
    N_ineqs = [e for e in exprs if e.operator() in [operator.ge, operator.eq] and not str(e.lhs()).startswith('t')]
    sols = [e for e in exprs if (e.operator() is operator.eq) and str(e.lhs()).startswith('t')]
    t_ineqs = [e for e in exprs if e.operator() is operator.le]

    if key in env.free_var_cache:
        num_free_vars = env.free_var_cache[key]
    else:
        A = env.MATRICES[key].T
        num_free_vars = calculate_free_vars(A)
        env.free_var_cache[key] = num_free_vars

    num_t_vals = max(major - 1, 0) + max(minor - 1, 0) + 1
    t_vals = [var(f't{i + 1}') for i in range(num_t_vals)]

    # bounds for free t-variables
    free_var_bounds: Dict[Any, Dict[str, Any]] = {}
    for i, expr in enumerate(t_ineqs[::-1]):
        if i % 2 == 0:
            free_var_bounds.setdefault(expr.lhs(), {})['upper'] = expr.rhs()
        else:
            free_var_bounds.setdefault(expr.rhs(), {})['lower'] = expr.lhs()

    free_vars = [x.lhs() for x in t_ineqs if (x.operator() is operator.le) and str(x.lhs()).startswith('t')]
    env.route_cache[key] = (sols, t_ineqs, N_ineqs, t_vals, num_t_vals, free_vars, num_free_vars, free_var_bounds)
    return env.route_cache[key]


def _Ab_from_N_ineqs(N_ineqs, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert N-constraints into row-normalized halfspaces A x >= b."""
    Ni = [var(f"N{i+1}") for i in range(K)]
    A_rows, b_rows = [], []
    zero_sub = {Ni[j]: 0 for j in range(K)}
    for e in N_ineqs:
        expr = e.lhs() - e.rhs()
        a = np.array([float(expr.coefficient(Ni[j])) for j in range(K)], float)
        c = float(expr.subs(zero_sub))
        if e.operator() is operator.ge:
            A_rows.append(a);  b_rows.append(-c)
        elif e.operator() is operator.eq:
            A_rows.append(a);   b_rows.append(-c)  # a·x >= -c
            A_rows.append(-a);  b_rows.append(c)   # -a·x >=  c   (i.e., a·x <= -c)
    if not A_rows:
        return np.zeros((0, K), float), np.zeros(0, float)
    A = np.vstack(A_rows)
    b = np.asarray(b_rows, float)
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    nrm[nrm == 0.0] = 1.0
    return A / nrm, (b / nrm.ravel())


def _project_active_set(
    A: np.ndarray, b: np.ndarray, x: np.ndarray, *,
    tol: float = 1e-9, max_active: int | None = None
) -> tuple[float, np.ndarray, list[int], np.ndarray]:
    """Project x (Euclidean) onto {y : A y >= b}. Returns (||y-x||, y, active_ids, final_slacks)."""
    K = x.shape[0]
    if A.size == 0:
        return 0.0, x.copy(), [], np.array([])
    max_active = max_active or K
    active: list[int] = []
    y = x.copy()

    for _ in range(max_active):
        sl = A @ y - b
        i = int(np.argmin(sl))          # most violated
        if sl[i] >= -tol or i in active:
            break
        active.append(i)
        Aact = A[active, :]
        bact = b[active]
        M = Aact @ Aact.T
        rhs = Aact @ x - bact
        try:
            lam = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            lam = np.linalg.pinv(M) @ rhs
        y = x - Aact.T @ lam

    dist = float(np.linalg.norm(y - x))
    return dist, y, active, (A @ y - b)


def project_preserving_sum(
    A: np.ndarray, b: np.ndarray, N: np.ndarray, *, tol: float = 1e-9,
) -> tuple[float, float, np.ndarray, list[int]]:
    """
    Project N onto {y : A y >= b} while preserving total sum(N).
    Steps:
      1) x̂ = N / sumN
      2) project x̂ in normalized space
      3) rescale back: N* = sumN · x̂*
    Returns (d_abs, d_rel, N_proj, active_ids).
    """
    N = np.asarray(N, float)
    sumN = float(N.sum())
    if sumN <= 0 or A.size == 0:
        return 0.0, 0.0, N.copy(), []

    xhat = N / sumN
    d_norm, xhat_proj, active, _ = _project_active_set(A, b, xhat, tol=tol)

    # rescale and clean up numerics
    N_proj = np.maximum(xhat_proj * sumN, 0.0)
    # exact renorm to preserve sum
    tot = float(N_proj.sum())
    if tot > 0:
        N_proj *= sumN / tot

    d_abs = float(d_norm * sumN)
    d_rel = float(d_norm)
    return d_abs, d_rel, N_proj, active


def n_slack_metrics(N_ineqs, nvars: dict, *, norm: str = "sumN", eps: float = 1e-12) -> tuple[float, float]:
    """Compute raw and relative slack metrics for N-constraints."""
    raw = []
    for e in N_ineqs:
        val = float((e.lhs() - e.rhs()).subs(nvars))
        raw.append(-abs(val) if e.operator() is operator.eq else val)
    raw_min = min(raw) if raw else 0.0
    denom = sum(float(v) for v in nvars.values()) + (eps if norm == "sumN" else 0.0)
    rel_min = raw_min / (denom if denom > 0 else 1.0)
    return raw_min, rel_min


def time_solution(env: RouteEnv, key: str, nvars: dict, major: int, minor: int,
                  total_sum: float, n_draws: int = 20):
    if total_sum <= 0:
        return None
    sols, _, N_ineqs, *_ = prepare_route(env, key, major, minor)
    sols_subs = [x.subs(nvars) for x in sols]

    # No free vars → direct normalization
    if env.free_var_cache.get(key, 0) == 0:
        sols_norm = {x.lhs(): x.rhs() / total_sum for x in sols_subs}
        return {'t_norm': sols_norm, 'total_sum': total_sum, 'num_free_vars': 0, 'nvars': nvars}

    # Free vars → build bounds and grid
    bounds_list = [{
        't_var': t_var,
        'lower': b['lower'].subs(nvars),
        'upper': b['upper'].subs(nvars)
    } for t_var, b in prepare_route(env, key, major, minor)[7].items()]

    draws = []
    for free_vals in generate_grid(n_draws, bounds_list):
        full_sol = {x.lhs(): x.rhs().subs(free_vals) for x in sols_subs}
        final_sol = {**free_vals, **full_sol}
        t_norm = {k: v / total_sum for k, v in final_sol.items()}
        draws.append(t_norm)

    # Fallback: if grid somehow ended up empty, plug midpoints once
    if not draws:
        mids = {}
        try:
            for bd in bounds_list:
                lo = float(bd['lower']); hi = float(bd['upper'])
                mids[bd['t_var']] = 0.5*(lo + hi)
        except (TypeError, ValueError):
            return None
        full_sol = {x.lhs(): x.rhs().subs(mids) for x in sols_subs}
        final_sol = {**mids, **full_sol}
        t_norm = {k: v / total_sum for k, v in final_sol.items()}
        draws = [t_norm]

    return {'t_norm': draws, 'total_sum': total_sum, 'num_free_vars': len(bounds_list), 'nvars': nvars}


def _effective_n(seg: Segment) -> float:
    """ESS proxy: sum of EM counts (keep stable across boot/no-boot)."""
    return float(np.sum(np.asarray(seg.N_counts, float))) if getattr(seg, "N_counts", None) is not None else 0.0


def _segment_fail(self, reason: str, **extra) -> None:
    """Attach a standardized discard_info and empty timing_result, then return."""
    self.timing_result = {}
    self.discard_info = dict(
        reason=reason,
        seg_id=getattr(self, "seg_id", None),
        chrom=str(self.chrom),
        start=int(self.start),
        end=int(self.end),
        major_cn=int(self.major_cn),
        minor_cn=int(self.minor_cn),
        n_snvs=int(getattr(self, "n_snvs", 0)),
        **extra,
    )


def _route_keys_for_state(env, major_cn: int, minor_cn: int) -> list[str]:
    prefix = f"{int(major_cn)}_{int(minor_cn)}."
    return [k for k in env.SOL.keys() if str(k).startswith(prefix)]


def _segment_time_segment(
    self: Segment,
    env: RouteEnv | None = None,
    *,
    counts: str = "em",
    n_draws: int = 20,
    dist_cut: float = 0.20,
    effN_cut: float = 10.0,
    tol: float = 1e-9,
) -> None:
    """
    Store *all* per-boot QC for each route, and only time (boot, route) pairs
    that pass the per-boot distance gate. Streamlined to avoid repeated work.
    """
    env = _get_route_env() if env is None else env

    # ---- quick validity checks ----
    if int(self.major_cn) <= 0 or int(self.minor_cn) < 0:
        _segment_fail(self, "nonpositive_cn")
        return

    # Route set for this state
    try:
        route_keys = _route_keys_for_state(env, self.major_cn, self.minor_cn)
    except NameError:
        route_keys = [k for k in env.SOL.keys() if str(k).startswith(f"{self.major_cn}_{self.minor_cn}.")]
    if not route_keys:
        _segment_fail(self, "no_routes_for_state")
        return

    # ---- per-segment precomputations reused everywhere ----
    # Gate by effective N from the (non-bootstrapped) counts
    N_gate = _counts_from(self)  # EM counts for gating only
    effN_gate = float(np.sum(N_gate))
    if effN_gate < float(effN_cut):
        _segment_fail(self, "low_effective_N", effN=effN_gate, effN_cut=float(effN_cut))
        return

    if len(self.snv_table):
        # Per-SNV likelihoods matrix + weights once
        seg_df = self.snv_table
        L = np.vstack(seg_df["multiplicity_likelihoods"].to_numpy())
        if "mut_w" in seg_df.columns:
            w = pd.to_numeric(seg_df["mut_w"], errors="coerce").fillna(0.0).to_numpy().astype(float)
            if not np.any(w > 0):
                w = np.ones(L.shape[0], float)
        else:
            w = np.ones(L.shape[0], float)

        # Base log-likelihood under EM mixture pi (single computation)
        pi_em = np.asarray(self.pi_em, float)
        loglik_base = float(np.sum(w * np.log(L @ pi_em + 1e-300)))

    # ---- per-route cache (computed ONCE) ----
    route_cache: dict[str, dict] = {}
    for rk in route_keys:
        _, _, N_ineqs, _, _, _, n_free, bounds_map = prepare_route(
            env, rk, self.major_cn, self.minor_cn
        )
        K = int(self.major_cn)
        A, b = _Ab_from_N_ineqs(N_ineqs, K)
        route_cache[rk] = dict(
            A=A, b=b, K=K,
            n_free=int(n_free),
            bounds_map=bounds_map,
        )

    # ---- coarse pre-gate per route using original EM counts ----
    prekept = []
    sumN_gate = float(np.sum(N_gate))
    for rk in route_keys:
        A = route_cache[rk]["A"]; b = route_cache[rk]["b"]
        if A.size == 0:
            d_rel = 0.0
        else:
            d_rel, _, _, _ = _project_active_set(A, b, np.asarray(N_gate, float) / sumN_gate, tol=tol)
        d_abs = float(d_rel * sumN_gate)
        if d_rel <= float(dist_cut):
            prekept.append(rk)

    if not prekept:
        # report the nearest route merely for diagnostics
        best_rk = None
        best_d = float("inf")
        for rk in route_keys:
            A = route_cache[rk]["A"]; b = route_cache[rk]["b"]
            if A.size == 0:
                d_abs = 0.0
            else:
                d_abs, d_rel, _Nproj, _ = project_preserving_sum(
                    A, b, np.asarray(N_gate, float), tol=tol
                )
            if d_abs < best_d:
                best_d, best_rk = d_abs, rk
        _segment_fail(
            self,
            "no_route_within_dist_cut",
            effN=effN_gate,
            dist_cut=float(dist_cut),
            best_route=str(best_rk),
            best_d_abs=float(best_d),
        )
        return

    # ---- initialize bins for ALL routes (QC for every route) ----
    out: dict[str, dict] = {
        rk: dict(
            draws=[],
            qc=[],
            meta=dict(
                free_var_count=0,
                had_projection=False,
                accepted_draws=0,
                total_draws=0,
                boots_seen=0,
                boots_passed=0,
            ),
        )
        for rk in route_keys
    }

    # ---- main loop over (original + bootstrap) count vectors ----
    for N_vec, meta in _counts_iter(self, method=counts):
        if N_vec is None or np.sum(N_vec) <= 0:
            continue
        boot_id = int(meta.get("boot", 0))

        for rk in route_keys:
            A = route_cache[rk]["A"]; b = route_cache[rk]["b"]; K = route_cache[rk]["K"]

            if A.size == 0:
                d_abs = 0.0
                d_rel = 0.0
                N_use = np.asarray(N_vec, float).copy()
                active_idx = []
            else:
                d_abs, d_rel, N_proj, active_idx = project_preserving_sum(A, b, np.asarray(N_vec, float), tol=tol)
                projected = bool(d_abs > tol)
                N_use = N_proj if projected else np.asarray(N_vec, float).copy()

            # Inside-margin computed on the N we actually use (projected or not)
            if A.size == 0:
                inside_margin = float("inf")
            else:
                slack = A @ N_use - b
                inside_margin = float(slack.min()) if slack.size else float("inf")
                if inside_margin < 0 and inside_margin > -1e-10:
                    inside_margin = 0.0

            # Likelihoods
            proj_pi = N_use / max(N_use.sum(), 1e-300)
            if not len(self.snv_table):
                proj_loglik = float("nan")
                loglik_diff = float("nan")
                loglik_base = float("nan")
            else:
                proj_loglik = float(np.sum(w * np.log(L @ proj_pi + 1e-300)))
                loglik_diff = proj_loglik - loglik_base

            # QC record for this (boot, route)
            projected = (d_abs > tol)
            out[rk]["meta"]["boots_seen"] += 1
            out[rk]["meta"]["had_projection"] = out[rk]["meta"]["had_projection"] or projected
            out[rk]["qc"].append(dict(
                boot_id=boot_id,
                N_used=N_use,
                N_sum=float(N_use.sum()),
                dist_abs=float(d_abs),
                dist_rel=float(d_rel),
                inside_margin=float(inside_margin if inside_margin >= -tol else 0.0),
                projected=projected,
                loglik=float(proj_loglik if projected else loglik_base),
                loglik_diff=float(loglik_diff),
                active_constraints=list(map(int, active_idx)),
            ))

            # Timing gate
            if (rk not in prekept) or (d_rel > float(dist_cut)):
                continue

            # Run timing
            Ndict = {var(f"N{i+1}"): float(N_use[i]) for i in range(K)}
            total_sum = calc_total_sum(N_use, self.total_cn)
            res = time_solution(env, rk, Ndict, self.major_cn, self.minor_cn, total_sum, n_draws=n_draws)
            if res is None:
                continue

            t_out = res["t_norm"]
            t_list = [t_out] if isinstance(t_out, dict) else list(t_out)
            free_var_count = 0 if isinstance(t_out, dict) else int(res.get("num_free_vars", 0))

            for draw_id, tdict in enumerate(t_list):
                out[rk]["draws"].append(dict(
                    boot_id=boot_id,
                    draw_id=draw_id,
                    t=tdict,
                    N_used=N_use.tolist(),
                    N_sum=float(N_use.sum()),
                    dist_abs=float(d_abs),
                    dist_rel=float(d_rel),
                    inside_margin=float(inside_margin if inside_margin >= -tol else 0.0),
                    projected=projected,
                    active_constraints=list(map(int, active_idx)),
                ))
                out[rk]["meta"]["accepted_draws"] += 1
                out[rk]["meta"]["total_draws"] += 1

            out[rk]["meta"]["boots_passed"] += 1
            out[rk]["meta"]["free_var_count"] = max(out[rk]["meta"]["free_var_count"], free_var_count)

    self.timing_result = out
    if hasattr(self, "discard_info"):
        delattr(self, "discard_info")


def _genome_time_segments(
    self: Genome,
    env: RouteEnv | None = None,
    *,
    counts: str = "em",
    n_draws: int = 50,
    dist_cut: float = 0.20,
    effN_cut: float = 10.0,
    tol: float = 1e-9,
) -> None:
    """Time every segment in the genome and store results on each Segment."""
    env = _get_route_env() if env is None else env
    for seg in tqdm(getattr(self, "segments", []), desc="Timing segments"):
        _segment_time_segment(
            seg,
            env,
            counts=counts,
            n_draws=n_draws,
            dist_cut=dist_cut,
            effN_cut=effN_cut,
            tol=tol,
        )


def _genome_times_to_df(
    self: Genome,
    sample: str,
    *,
    accept_thresh: float = 0.0,
    summarize: bool = False,
    ci: tuple[float, float, float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """
    Long-form timing draws (default), or summarized per-(seg,route,interval,boot_id) CIs.

    Long-form columns:
      sample, seg_id, key, boot_id, draw_id, time_interval, time, seg_len, eff_N, w, is_bootstrap
    """
    from tau.utils import order_t
    rows: list[dict] = []

    for seg in getattr(self, "segments", []):
        if not getattr(seg, "timing_result", None):
            continue

        seg_len = float(int(seg.end) - int(seg.start))
        eff_N = float(_effective_n(seg))

        for key, info in seg.timing_result.items():
            if not key:
                continue
            draws = info.get("draws", [])
            if not draws:
                continue

            for draw in draws:
                tdict = draw.get("t")
                if not tdict:
                    continue
                boot_id = int(draw.get("boot_id", 0))
                draw_id = int(draw.get("draw_id", 0))
                is_bootstrap = bool(boot_id >= 1)

                ts = order_t(tdict)
                if ts.sum() <= 0:
                    continue
                cum = np.cumsum(ts)[:-1]

                for k_idx, s_k in enumerate(cum, 1):
                    rows.append(dict(
                        sample=sample,
                        seg_id=getattr(seg, "seg_id", f"{seg.chrom}:{seg.start}-{seg.end}"),
                        key=key,
                        boot_id=boot_id,
                        draw_id=draw_id,
                        draw=draw_id,
                        time_interval=int(k_idx),
                        time=float(s_k),
                        seg_len=seg_len,
                        eff_N=eff_N,
                        is_bootstrap=is_bootstrap,
                    ))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # weights: seg length × ESS proxy
    w = (df["seg_len"] * df["eff_N"]).astype(float)
    df["w"] = w / (w.sum() if w.sum() > 0 else 1.0)

    if not summarize:
        return df

    # Summarize
    qs = sorted(set(ci))
    gcols = ["seg_id", "key", "time_interval", "boot_id", "is_bootstrap"]
    qdf = (
        df.groupby(gcols, as_index=False)
          .agg(
              time_q=("time", lambda x: np.quantile(x.to_numpy(), qs)),
              seg_len=("seg_len", "first"),
              eff_N=("eff_N", "first"),
              w=("w", "mean"),
          )
    )

    out_rows = []
    for _, r in qdf.iterrows():
        tq = np.asarray(r["time_q"], float)
        out = dict(
            seg_id=r["seg_id"],
            key=r["key"],
            time_interval=int(r["time_interval"]),
            boot_id=int(r["boot_id"]),
            is_bootstrap=bool(r["is_bootstrap"]),
            seg_len=float(r["seg_len"]),
            eff_N=float(r["eff_N"]),
            w=float(r["w"]),
        )
        for q, val in zip(qs, tq):
            out[f"time_q{q:.3f}"] = float(val)
        out_rows.append(out)

    return pd.DataFrame(out_rows)


_DISCARD_REASON_LABELS: dict[str, str] = {
    "nonpositive_cn":           "Non-positive CN state (major ≤ 0 or minor < 0)",
    "low_effective_N":          "Too few mutations — effective N below threshold",
    "no_routes_for_state":      "CN state not modelled (major > 7)",
    "no_route_within_dist_cut": "Multiplicity distribution incompatible with all routes",
}


def _genome_summarize_discards(self: Genome, sample: str) -> pd.DataFrame:
    """Collects segments that were discarded during timing with reasons and basics."""
    rows = []
    for seg in getattr(self, "segments", []):
        info = getattr(seg, "discard_info", None)
        if not info:
            continue
        row = dict(sample=sample)
        row.update(info)
        row["seg_len"] = float(int(seg.end) - int(seg.start))
        rows.append(row)
    return pd.DataFrame(rows)


def _genome_discard_report(self: Genome, sample: str = "") -> pd.DataFrame:
    """Print a human-readable discard summary and return the raw discard DataFrame.

    Shows how many segments were skipped during timing and why.

    Parameters
    ----------
    sample : str, optional
        Sample label to include in the header line.

    Returns
    -------
    pd.DataFrame
        Same as summarize_discards(sample) — one row per discarded segment.

    Examples
    --------
    >>> g.discard_report("MY_SAMPLE")
    Discard report (12 segments skipped) — MY_SAMPLE
    --------------------------------------------------
       10  Too few mutations — effective N below threshold
        2  Multiplicity distribution incompatible with all routes
    """
    df = _genome_summarize_discards(self, sample)
    n_timed = sum(1 for s in getattr(self, "segments", [])
                  if getattr(s, "timing_result", None))
    n_disc = len(df)
    header = f"Timing summary: {n_timed} segments timed, {n_disc} discarded"
    if sample:
        header += f" — {sample}"
    print(header)
    print("-" * len(header))
    if df.empty:
        print("  (no segments discarded)")
    else:
        for reason, count in df["reason"].value_counts().items():
            label = _DISCARD_REASON_LABELS.get(reason, reason)
            print(f"  {count:4d}  {label}")
    print()
    return df


def _genome_summarize_constraint_violations(self: Genome, sample: str) -> pd.DataFrame:
    """
    Thin flattener that pulls already-stored QC from timing_result.
    Reads from qc[boot_id==0] and meta sub-keys stored by _segment_time_segment.
    """
    rows: list[dict] = []
    for seg in getattr(self, "segments", []):
        if not getattr(seg, "timing_result", None):
            continue
        for key, info in seg.timing_result.items():
            qc = info.get("qc", [])
            meta = info.get("meta", {})
            draws = info.get("draws", [])
            qc0 = next((q for q in qc if q.get("boot_id", 0) == 0), qc[0] if qc else {})
            rows.append(dict(
                sample=sample,
                seg_id=getattr(seg, "seg_id", f"{seg.chrom}:{seg.start}-{seg.end}"),
                chrom=str(seg.chrom),
                start=int(seg.start),
                end=int(seg.end),
                major_cn=int(seg.major_cn),
                minor_cn=int(seg.minor_cn),
                route=key,
                seg_len=float(int(seg.end) - int(seg.start)),
                eff_N=float(qc0.get("N_sum", _effective_n(seg))),
                N_sum=float(qc0.get("N_sum", np.nan)),
                dist_abs=float(qc0.get("dist_abs", np.nan)),
                dist_rel=float(qc0.get("dist_rel", np.nan)),
                inside_margin=float(qc0.get("inside_margin", np.nan)),
                proj_used=bool(qc0.get("projected", False)),
                free_vars=int(meta.get("free_var_count", 0)),
                n_draws=len(draws),
                boots_passed=int(meta.get("boots_passed", 0)),
            ))
    return pd.DataFrame(rows)


def summarize_constraint_violations_batch(
    items: List[Tuple[str, Any]], env: RouteEnv, *, counts: str = "em", routes: str = "all", tol: float = 1e-9
) -> pd.DataFrame:
    """Batch summarize constraint violations for multiple (sample, genome) pairs."""
    from tau.timing import summarize_constraint_violations
    parts = [summarize_constraint_violations(sample, genome, env, counts=counts, routes=routes, tol=tol)
             for sample, genome in items]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize_constraint_violations(
    sample: str, genome: Genome, env: RouteEnv, *, counts: str = "em", routes: str = "all", tol: float = 1e-9
) -> pd.DataFrame:
    """Summarize constraint violations for a single genome."""
    rows = []
    for seg in getattr(genome, "segments", []):
        K = int(seg.major_cn)
        if K <= 0:
            continue
        Nvec = _counts_from(seg)
        sumN = float(np.sum(Nvec))
        effN = _effective_n(seg)
        keys = [k for k in env.SOL.keys() if str(k).startswith(f"{seg.major_cn}_{seg.minor_cn}.")]
        if routes == "timed_only" and getattr(seg, "timing_result", None):
            keys = [k for k in keys if k in seg.timing_result]
        for key in keys:
            _, _, N_ineqs, *_ = prepare_route(env, key, seg.major_cn, seg.minor_cn)
            A, b = _Ab_from_N_ineqs(N_ineqs, K)
            raw_slack, rel_slack = n_slack_metrics(N_ineqs, {var(f"N{i+1}"): Nvec[i] for i in range(K)}, norm="sumN")
            d_abs, d_rel, _projN, active = project_preserving_sum(A, b, Nvec, tol=tol)
            feasible = (d_abs <= tol)
            rows.append(dict(
                sample=sample, seg_id=getattr(seg, "seg_id", f"{seg.chrom}:{seg.start}-{seg.end}"),
                chrom=str(seg.chrom), start=int(seg.start), end=int(seg.end),
                major_cn=int(seg.major_cn), minor_cn=int(seg.minor_cn),
                K=K, route=key, n_snvs=int(seg.n_snvs), seg_len=float(seg.end - seg.start),
                purity=float(getattr(seg, "purity", np.nan)), counts_method=counts,
                eff_N=effN, sumN=sumN, feasible=bool(feasible),
                proj_dist=float(d_abs), proj_dist_per_N=float(d_rel),
                num_free_vars=int(env.free_var_cache.get(key, calculate_free_vars(env.MATRICES[key].T))),
                active_faces=int(len(active)),
            ))
    return pd.DataFrame(rows)
