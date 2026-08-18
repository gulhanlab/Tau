#!/usr/bin/env python3
"""
tau/tree_polytope.py — extract a JSON-serialisable "tree solution with knobs" payload.

For a single CN segment and one route, Tau's polytope of valid timing solutions is
parameterised by a small set of FREE VARIABLES.  After substituting the segment's
N-counts, each timing interval ``t_i`` (and therefore each cumulative EVENT TIME =
tree-node height) is an *affine* function of the free variables, and the feasible
region of the free variables is an intersection of *affine* half-spaces.  This module
extracts those affine coefficients ONCE (using Sage to evaluate the route's symbolic
solution) so that all live recomputation — dialing a knob, redrawing the tree,
re-clamping the other knobs — can be done with plain arithmetic (e.g. in a browser).

The returned payload is pure-Python / JSON-serialisable and is designed to be embedded
directly into an interactive viewer (see ``tau/viz.py`` and the demo in
``examples/tree_explorer_demo.html``).

TERMINOLOGY
-----------
The entries of a segment's ``timing_result`` are **polytope solutions** — equally valid
points in an underdetermined solution space.  They are NOT posterior draws, samples, or a
distribution.  This module never weights them.

KEY MATH
--------
With ``nvars`` = the route's per-segment N-count dict and ``S`` = total_sum:

    t_i / S  =  (rhs_i.subs(nvars)) / S            # affine in free vars
    G_j      =  sum_{i<=j} t_i / S                 # cumulative event time, affine

    free_var_bounds[t_var]['lower'].subs(nvars) <= t_var <= ...['upper'].subs(nvars)
                                                   # affine half-spaces in the OTHER free vars

The affine coefficients are recovered EXACTLY by evaluating an affine expression at the
origin (all free vars = 0) and at each free-var unit vector:

    const   = expr(0, 0, ..., 0)
    coef_k  = expr(e_k) - const

PUBLIC API
----------
``build_tree_polytope_payload(genome, seg_id, route_key=None, *, n_samples=12, env=None)``
    Returns the payload dict.

``verify_payload(payload, genome, seg_id, env=None)``
    Round-trip / feasibility / topology checks; returns a dict of metrics.
"""
from __future__ import annotations

import gzip
import json
import operator
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# NOTE: this module must NOT import tau.viz (another component owns that file).
from tau.timing import (
    RouteEnv,
    _Ab_from_N_ineqs,
    _counts_from,
    calc_total_sum,
    load_routes_for_states,
    prepare_route,
    project_preserving_sum,
)
from tau.utils import pick_best_key

_TOPOLOGY_GZ = Path(__file__).parent / "data" / "route_topology.json.gz"


# ---------------------------------------------------------------------------
# Route topology (allele / parent_mult / split per event) — small cached table
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_topology() -> Dict[str, Dict[str, Any]]:
    """Load the route -> events topology table (allele/parent_mult/split/col_index).

    The table is a compact gzipped JSON keyed by route_key.  It encodes only the
    discrete tree *topology* of each route (which allele gains, from which parent
    multiplicity, splitting how) — the same ``events`` structure consumed by
    ``tau.trees.build_allele_trees``.  Event *times* are computed live from the
    affine maps in this module, never read from the table.
    """
    if not _TOPOLOGY_GZ.exists():
        return {}
    with gzip.open(_TOPOLOGY_GZ, "rt") as fh:
        return json.load(fh)


def _events_for_route(route_key: str) -> List[Dict[str, Any]]:
    """Return the ordered list of event dicts (topology) for *route_key*.

    Each event dict has ``allele`` ('major'/'minor'), ``parent_mult`` (int),
    ``split`` ([k, l]) — exactly what ``build_allele_trees`` needs — plus the
    ``col_index`` (1-based matrix column / interval that the event opens).
    Events are returned sorted by ``col_index``.
    """
    topo = _load_topology()
    rec = topo.get(route_key)
    if rec is None:
        return []
    events = list(rec.get("events", []))
    events.sort(key=lambda e: int(e.get("col_index", 0)))
    return events


# ---------------------------------------------------------------------------
# Affine extraction helpers
# ---------------------------------------------------------------------------
def _affine_coefs(expr, free_vars) -> Tuple[float, List[float]]:
    """Exactly recover (const, [coef_k]) of an affine Sage expression in *free_vars*.

    Works for any expression that is affine in the free variables (which is always
    the case after the segment's N-counts have been substituted in).  Constants are
    evaluated symbolically via ``.subs`` so plain Python floats and Sage expressions
    are both handled.
    """
    zero = {v: 0 for v in free_vars}
    try:
        const = float(expr.subs(zero))
    except AttributeError:
        # expr is a plain number
        return float(expr), [0.0] * len(free_vars)
    coefs = []
    for v in free_vars:
        unit = {u: (1 if u is v else 0) for u in free_vars}
        coefs.append(float(expr.subs(unit)) - const)
    return const, coefs


def _affine_to_dict(const: float, coefs: List[float]) -> Dict[str, Any]:
    return {"const": float(const), "coef": [float(c) for c in coefs]}


def _affine_branches(expr, free_vars) -> List[Tuple[float, List[float]]]:
    """Expand a bound expression — possibly ``min(...)`` / ``max(...)`` — into a list
    of affine branches ``[(const, coefs), ...]``.

    A free-variable bound after ``.subs(nvars)`` may be a ``min`` of several affine
    upper bounds or a ``max`` of several affine lower bounds (e.g. Tau's
    ``t4 <= min(N1/4, N2-N4)`` or ``t2 >= max(0, ...)``).  Each operand of the
    ``min``/``max`` is itself affine in the free variables, so the bound is
    equivalent to a *conjunction* of affine half-spaces — one per operand.  This
    returns those operand branches; ``_affine_coefs`` would otherwise mis-read the
    non-affine ``min``/``max`` wrapper.
    """
    s = str(expr)
    if (s.startswith("min(") or s.startswith("max(")) and hasattr(expr, "operands"):
        return [_affine_coefs(op, free_vars) for op in expr.operands()]
    return [_affine_coefs(expr, free_vars)]


# ---------------------------------------------------------------------------
# N-counts the polytope solutions were actually computed from
# ---------------------------------------------------------------------------
def _route_nvars(seg, route_key: str, env: RouteEnv) -> Tuple[Dict[Any, float], np.ndarray, float]:
    """Reproduce the per-route N-count vector that timing used for this route.

    ``_segment_time_segment`` projects the EM counts onto each route's feasible
    region (preserving the total) before solving.  To get affine maps whose
    round-trip matches the stored polytope solutions exactly, we must use the SAME
    projected counts.  We re-derive them here from ``prepare_route``'s constraints.

    Returns ``(nvars_dict, N_used, total_sum)`` where ``nvars_dict`` maps the Sage
    symbols ``N1..NK`` to the projected float counts.
    """
    from sage.all import var as _svar

    K = int(seg.major_cn)
    N_em = _counts_from(seg)
    _, _, N_ineqs, *_ = prepare_route(env, route_key, seg.major_cn, seg.minor_cn)
    A, b = _Ab_from_N_ineqs(N_ineqs, K)
    if A.size == 0:
        N_used = np.asarray(N_em, float).copy()
    else:
        _d_abs, _d_rel, N_proj, _active = project_preserving_sum(A, b, np.asarray(N_em, float))
        N_used = N_proj if _d_abs > 1e-9 else np.asarray(N_em, float).copy()
    nvars = {_svar(f"N{i + 1}"): float(N_used[i]) for i in range(K)}
    total_sum = calc_total_sum(N_used, seg.total_cn)
    return nvars, N_used, float(total_sum)


# ---------------------------------------------------------------------------
# Main payload builder
# ---------------------------------------------------------------------------
def build_tree_polytope_payload(
    genome,
    seg_id: str,
    route_key: Optional[str] = None,
    *,
    n_samples: int = 12,
    env: Optional[RouteEnv] = None,
) -> Dict[str, Any]:
    """Build the JSON-serialisable 'tree solution with knobs' payload.

    Parameters
    ----------
    genome : tau.core.Genome
        A timed genome (segments carry ``timing_result``).
    seg_id : str
        Segment id to explore (e.g. ``"13:61635570-71312996"``).
    route_key : str, optional
        Route to explore.  Default: the best route for the segment that has at
        least one free variable (an underdetermined route, so the knobs are
        meaningful).  Falls back to ``pick_best_key`` if none are underdetermined.
    n_samples : int
        Number of polytope solutions to sample for the optional faint overlay.
    env : RouteEnv, optional
        Pre-loaded route environment.  If ``None`` one is loaded for this CN state
        (requires Sage).

    Returns
    -------
    dict
        See module docstring / README for the schema.  All values are plain
        Python types (floats, ints, str, lists, dicts) — JSON-serialisable.
    """
    seg = next((s for s in genome.segments if getattr(s, "seg_id", None) == seg_id), None)
    if seg is None:
        raise ValueError(f"segment {seg_id!r} not found in genome")
    timing = getattr(seg, "timing_result", {}) or {}
    if not timing:
        raise ValueError(f"segment {seg_id!r} has no timing_result")

    # ---- choose route ----
    if route_key is None:
        # prefer an underdetermined best route (>=1 free var)
        underdet = {
            k: v for k, v in timing.items()
            if v.get("meta", {}).get("free_var_count", 0) >= 1 and v.get("draws")
        }
        if underdet:
            # among underdetermined routes, the one with smallest dist_rel
            def _d(k):
                qc = timing[k].get("qc", [])
                return qc[0].get("dist_rel", np.inf) if qc else np.inf
            route_key = min(underdet, key=_d)
        else:
            route_key = pick_best_key(seg)
    if route_key is None or route_key not in timing:
        raise ValueError(f"no usable route for segment {seg_id!r}")

    major, minor = int(seg.major_cn), int(seg.minor_cn)
    if env is None:
        env = load_routes_for_states([(major, minor)])

    (sols, t_ineqs, N_ineqs, t_vals, num_t_vals,
     free_vars, num_free_vars, free_var_bounds) = prepare_route(env, route_key, major, minor)

    nvars, N_used, total_sum = _route_nvars(seg, route_key, env)

    # ordered free-variable symbols / names (the knobs)
    free_var_syms = list(free_var_bounds.keys())  # only t-vars that are actually free
    free_var_names = [str(v) for v in free_var_syms]

    # ---- substitute N-counts into the symbolic solution; each t_i now affine ----
    sols_subs = {str(x.lhs()): x.rhs().subs(nvars) for x in sols}

    # ordered t-interval symbols t1..t_{num_t_vals}
    t_names = [f"t{i + 1}" for i in range(num_t_vals)]

    # affine map for each interval fraction t_i / total_sum (in terms of free vars)
    interval_maps: Dict[str, Dict[str, Any]] = {}
    interval_affine: Dict[str, Tuple[float, List[float]]] = {}
    for tn in t_names:
        if tn in sols_subs:
            const, coefs = _affine_coefs(sols_subs[tn], free_var_syms)
        else:
            # a free variable itself: t = t (const 0, unit coef on itself)
            const = 0.0
            coefs = [1.0 if str(v) == tn else 0.0 for v in free_var_syms]
        # normalise by total_sum
        const /= total_sum
        coefs = [c / total_sum for c in coefs]
        interval_affine[tn] = (const, coefs)
        interval_maps[tn] = _affine_to_dict(const, coefs)

    # ---- cumulative EVENT TIMES G_j = sum_{i<=j} t_i / S (drop terminal G_n == 1) ----
    # events are tied to matrix columns (col_index); event j opens interval j and its
    # node height is the cumulative time up to and INCLUDING interval j.
    events = _events_for_route(route_key)
    n_events = len(events)

    event_time_maps: List[Dict[str, Any]] = []
    cum_const = 0.0
    cum_coefs = [0.0] * len(free_var_syms)
    event_time_affine: List[Tuple[float, List[float]]] = []
    for j in range(1, num_t_vals + 1):
        c, k = interval_affine[f"t{j}"]
        cum_const += c
        cum_coefs = [a + b for a, b in zip(cum_coefs, k)]
        if j <= n_events:  # keep only the first n_events cumulative times (node heights)
            event_time_affine.append((cum_const, list(cum_coefs)))
            event_time_maps.append(_affine_to_dict(cum_const, list(cum_coefs)))

    # ---- feasible region: affine half-spaces (>= 0 form) over the free vars ----
    # (a) per-free-variable lower/upper bounds (each affine in the OTHER free vars)
    # A bound may be min()/max() of several affine exprs; each operand becomes its own
    # half-space (a min-upper / max-lower is a CONJUNCTION of half-spaces).
    bound_constraints: List[Dict[str, Any]] = []
    for sym in free_var_syms:
        bd = free_var_bounds.get(sym, {})
        name = str(sym)
        self_unit = [1.0 if str(v) == name else 0.0 for v in free_var_syms]
        # lower:  sym - lower_branch >= 0   for every branch of max(...)
        if "lower" in bd:
            for lc, lk in _affine_branches(bd["lower"].subs(nvars), free_var_syms):
                coefs = [self_unit[i] - lk[i] for i in range(len(free_var_syms))]
                bound_constraints.append({
                    "const": float(-lc), "coef": [float(c) for c in coefs],
                    "kind": "lower", "var": name,
                })
        # upper:  upper_branch - sym >= 0   for every branch of min(...)
        if "upper" in bd:
            for uc, uk in _affine_branches(bd["upper"].subs(nvars), free_var_syms):
                coefs = [uk[i] - self_unit[i] for i in range(len(free_var_syms))]
                bound_constraints.append({
                    "const": float(uc), "coef": [float(c) for c in coefs],
                    "kind": "upper", "var": name,
                })

    # (b) every interval fraction must be >= 0 (t_i / S >= 0).  These are implied by the
    #     bounds for valid routes but make the JS feasibility test self-contained.
    interval_nonneg: List[Dict[str, Any]] = []
    for tn in t_names:
        c, k = interval_affine[tn]
        interval_nonneg.append({"const": float(c), "coef": [float(x) for x in k], "interval": tn})

    # ---- sampled polytope solutions (free-var vectors) for faint overlay ----
    # Sample the SAME polytope Tau's grid sampler walks, but directly from the affine
    # half-spaces we just extracted — a fast, Sage-free Cartesian grid that visits each
    # free var in turn and intersects the per-knob feasible interval (identical to
    # ``generate_grid``'s recursion, but in pure numpy).  Guaranteed feasible.
    sampled_free = _grid_sample_polytope(
        len(free_var_syms), bound_constraints + interval_nonneg,
        n_per=max(int(round(max(n_samples, 2) ** (1.0 / max(len(free_var_syms), 1)))), 2),
        cap=max(n_samples, 1),
    )

    # ---- precomputed node list for the tree ----
    # build the tree once at the midpoint of the polytope to get a concrete node
    # ordering / parent indices; JS will only update node HEIGHTS from the affine maps.
    nodes = _build_node_list(events, major, minor, event_time_affine, sampled_free, free_var_syms)

    # initial free-var assignment = midpoint of sampled solutions (or zeros)
    if sampled_free:
        init_free = [float(np.mean([v[k] for v in sampled_free])) for k in range(len(free_var_syms))]
    else:
        init_free = [0.0] * len(free_var_syms)

    payload = {
        "seg": {
            "seg_id": seg_id,
            "chrom": str(seg.chrom),
            "start": int(seg.start),
            "end": int(seg.end),
            "major_cn": major,
            "minor_cn": minor,
            "total_cn": int(seg.total_cn),
            "n_snvs": int(getattr(seg, "n_snvs", 0)),
            "total_sum": float(total_sum),
            "N_counts": [float(x) for x in _counts_from(seg)],
            "N_used": [float(x) for x in N_used],
        },
        "route_key": route_key,
        "num_t_vals": int(num_t_vals),
        "t_vars": t_names,
        "free_vars": free_var_names,
        "num_free_vars": len(free_var_names),
        "events": [
            {"allele": e["allele"], "parent_mult": int(e["parent_mult"]),
             "split": [int(e["split"][0]), int(e["split"][1])],
             "col_index": int(e.get("col_index", i + 1))}
            for i, e in enumerate(events)
        ],
        "interval_maps": interval_maps,        # t_i / S  (affine in free vars)
        "event_time_maps": event_time_maps,    # node heights G_j (affine), len == n_events
        "constraints": {
            "bounds": bound_constraints,       # per-knob lower/upper half-spaces (>=0)
            "interval_nonneg": interval_nonneg,  # t_i >= 0 half-spaces
        },
        "sampled_solutions": sampled_free,     # free-var vectors for faint overlay
        "init_free": init_free,
        "nodes": nodes,
    }
    return payload


def _interval_1d(constraints, free, ki):
    """1-D feasible interval for free var *ki* with the others fixed at *free*.

    Mirrors the browser's ``feasibleInterval``: each affine half-space
    ``const + sum coef*free >= 0`` becomes a bound on ``free[ki]``.
    """
    lo, hi = -np.inf, np.inf
    for c in constraints:
        base = c["const"] + sum(c["coef"][k] * free[k] for k in range(len(free)) if k != ki)
        a = c["coef"][ki]
        if abs(a) < 1e-12:
            continue
        if a > 0:
            lo = max(lo, -base / a)
        else:
            hi = min(hi, -base / a)
    return lo, hi


def _grid_sample_polytope(n_free, constraints, *, n_per=4, cap=16, eps=1e-9):
    """Cartesian grid over the polytope defined by affine half-space *constraints*.

    Walks the free variables in order, intersecting each one's feasible interval given
    the partial assignment (exactly like ``tau.timing.generate_grid`` but using the
    pre-extracted affine constraints, so no Sage calls).  Returns a list of free-var
    vectors that are all strictly inside (or on) the polytope.
    """
    if n_free == 0:
        return []
    grids = [[]]
    for ki in range(n_free):
        # Prefer constraints that do not reference a not-yet-chosen var (index > ki) —
        # this mirrors generate_grid's recursion where each var's bounds depend only on
        # EARLIER vars (true for every Tau route).  If those alone leave the var
        # unbounded (generic polytopes), fall back to the full constraint set.
        usable = [c for c in constraints
                  if all(abs(c["coef"][j]) < 1e-12 for j in range(ki + 1, n_free))]
        nxt = []
        for partial in grids:
            free = list(partial) + [0.0] * (n_free - len(partial))
            lo, hi = _interval_1d(usable, free, ki)
            if not (np.isfinite(lo) and np.isfinite(hi)):
                lo, hi = _interval_1d(constraints, free, ki)
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo - eps:
                continue
            if hi - lo <= 2 * eps:
                vals = [0.5 * (lo + hi)]
            else:
                vals = list(np.linspace(lo + eps, hi - eps, n_per))
            for v in vals:
                nxt.append(list(partial) + [float(v)])
        grids = nxt
        if not grids:
            return []
    # de-dupe + cap
    uniq, seen = [], set()
    for vec in grids:
        key = tuple(round(x, 9) for x in vec)
        if key not in seen:
            seen.add(key)
            uniq.append(vec)
    return uniq[:cap]


def _build_node_list(events, major, minor, event_time_affine, sampled_free, free_var_syms):
    """Precompute the tree's node list (topology + which event drives each height).

    Builds the allele trees once with concrete event times (midpoint of the sampled
    polytope) to fix a node ordering / parent indices, then records for each node the
    index of the cumulative event-time that determines its height.  JS redraws heights
    purely from ``event_time_maps`` without re-deriving topology.

    Returns a list of dicts: ``{index, parent_index, allele, event_index, x}`` where
    ``event_index`` is the 0-based index into ``event_time_maps`` (or ``-1`` for the
    allele roots, which sit at height 0).
    """
    from tau.trees import build_allele_trees, sort_nodes

    if not events:
        return []

    # Build the topology with STRICTLY increasing synthetic times so that birth-time
    # order within each allele equals event-application order exactly (no ties) — this
    # makes the node -> event_index mapping unambiguous even when real event times tie.
    n = len(events)
    synth_times = [(i + 1) / (n + 1) for i in range(n)]
    try:
        maj_root, min_root = build_allele_trees(events, int(major), int(minor), times=synth_times)
    except (ValueError, IndexError):
        return []

    # Map each node (by identity) to the event index that set its gain time.
    # Events are applied in order; the j-th event creates exactly one new branch.
    # With strictly increasing synth_times, per-allele birth-time order == event order.
    sorted_maj = sort_nodes(maj_root)
    # LOH (minor_cn == 0): the minor allele is gone — there is no minor lineage and
    # an LOH route has no minor events, so drop the phantom minor root node entirely
    # (otherwise the viewer draws a minor branch for a segment that has none).
    sorted_min = sort_nodes(min_root) if int(minor) > 0 else []
    all_nodes = sorted_maj + sorted_min
    node_index = {id(n): i for i, n in enumerate(all_nodes)}

    # which event index produced each non-root node?  Non-root nodes were added in
    # event order within each allele; reconstruct by sorting non-root nodes per allele
    # by birth time and pairing with the event indices for that allele.
    allele_of = {}
    for n in sorted_maj:
        allele_of[id(n)] = "major"
    for n in sorted_min:
        allele_of[id(n)] = "minor"

    # event index per allele, in application order
    maj_event_idxs = [j for j, e in enumerate(events) if e["allele"] == "major"]
    min_event_idxs = [j for j, e in enumerate(events) if e["allele"] == "minor"]

    def _assign(root_nodes, event_idxs):
        # non-root nodes of this allele, in tree-build (birth-time) order
        non_root = [n for n in root_nodes if n.parent is not None]
        non_root.sort(key=lambda n: (n.time, node_index[id(n)]))
        mapping = {}
        for n, j in zip(non_root, event_idxs):
            mapping[id(n)] = j
        return mapping

    ev_idx_map = {}
    ev_idx_map.update(_assign(sorted_maj, maj_event_idxs))
    ev_idx_map.update(_assign(sorted_min, min_event_idxs))

    nodes = []
    for i, n in enumerate(all_nodes):
        parent_idx = node_index[id(n.parent)] if n.parent is not None else -1
        nodes.append({
            "index": i,
            "parent_index": parent_idx,
            "allele": allele_of[id(n)],
            "event_index": int(ev_idx_map.get(id(n), -1)),
            "mult": int(n.mult) if n.mult is not None else None,
            "x": float(i),  # placeholder horizontal slot; JS lays out per allele
        })
    return nodes


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _eval_affine_dict(d: Dict[str, Any], free_vals: List[float]) -> float:
    return float(d["const"]) + float(np.dot(d["coef"], free_vals))


def verify_payload(payload: Dict[str, Any], genome, seg_id: str,
                   env: Optional[RouteEnv] = None) -> Dict[str, Any]:
    """Validate the extracted payload against what Tau actually computes.

    Checks
    ------
    1. Round-trip: for each sampled polytope solution, the affine interval/event-time
       maps reproduce the values Tau computes (via ``prepare_route``/``time_solution``
       arithmetic) to < 1e-6 absolute.
    2. Feasibility: every sampled solution satisfies all extracted half-spaces (>= -tol).
    3. Topology: #events == #event_time_maps == #non-root tree nodes, tree builds.

    Returns a metrics dict and prints a summary.
    """
    from sage.all import var as _svar

    seg = next((s for s in genome.segments if getattr(s, "seg_id", None) == seg_id), None)
    route_key = payload["route_key"]
    major, minor = payload["seg"]["major_cn"], payload["seg"]["minor_cn"]
    if env is None:
        env = load_routes_for_states([(major, minor)])
    (sols, _ti, _Ni, _tv, num_t_vals, _fv, _nfv, _fvb) = prepare_route(
        env, route_key, major, minor)
    nvars, N_used, total_sum = _route_nvars(seg, route_key, env)
    sols_subs = {str(x.lhs()): x.rhs().subs(nvars) for x in sols}
    free_syms = [_svar(n) for n in payload["free_vars"]]

    # ---- 1. round-trip ----
    max_interval_err = 0.0
    max_event_err = 0.0
    t_names = payload["t_vars"]
    for free_vec in payload["sampled_solutions"]:
        fv_dict = {free_syms[k]: free_vec[k] for k in range(len(free_syms))}
        # Tau's truth: solve each interval symbolically, normalise
        true_int = {}
        for tn in t_names:
            if tn in sols_subs:
                val = float(sols_subs[tn].subs(fv_dict)) / total_sum
            else:
                # free variable itself
                val = float(fv_dict[_svar(tn)]) / total_sum
            true_int[tn] = val
        # affine prediction
        for tn in t_names:
            pred = _eval_affine_dict(payload["interval_maps"][tn], free_vec)
            max_interval_err = max(max_interval_err, abs(pred - true_int[tn]))
        # event times: cumulative
        true_cum = np.cumsum([true_int[tn] for tn in t_names])
        for j, emap in enumerate(payload["event_time_maps"]):
            pred = _eval_affine_dict(emap, free_vec)
            max_event_err = max(max_event_err, abs(pred - float(true_cum[j])))

    # ---- 2. feasibility ----
    tol = 1e-7
    feas_min = np.inf
    feas_pass = True
    all_constraints = (payload["constraints"]["bounds"]
                       + payload["constraints"]["interval_nonneg"])
    for free_vec in payload["sampled_solutions"]:
        for c in all_constraints:
            val = _eval_affine_dict(c, free_vec)
            feas_min = min(feas_min, val)
            if val < -tol:
                feas_pass = False

    # ---- 3. topology ----
    n_events = len(payload["events"])
    n_event_maps = len(payload["event_time_maps"])
    n_nonroot = sum(1 for nd in payload["nodes"] if nd["parent_index"] >= 0)
    topo_ok = (n_events == n_event_maps == n_nonroot) and len(payload["nodes"]) > 0

    metrics = {
        "route_key": route_key,
        "num_free_vars": payload["num_free_vars"],
        "n_sampled": len(payload["sampled_solutions"]),
        "roundtrip_max_interval_err": float(max_interval_err),
        "roundtrip_max_event_err": float(max_event_err),
        "roundtrip_pass": bool(max(max_interval_err, max_event_err) < 1e-6),
        "feasibility_min_slack": float(feas_min if np.isfinite(feas_min) else 0.0),
        "feasibility_pass": bool(feas_pass),
        "n_events": n_events,
        "n_event_maps": n_event_maps,
        "n_nonroot_nodes": n_nonroot,
        "topology_pass": bool(topo_ok),
    }

    print(f"[verify] route={route_key}  free_vars={payload['num_free_vars']}  "
          f"sampled={metrics['n_sampled']}")
    print(f"[verify] round-trip max interval err = {max_interval_err:.3e}, "
          f"max event-time err = {max_event_err:.3e}  -> "
          f"{'PASS' if metrics['roundtrip_pass'] else 'FAIL'}")
    print(f"[verify] feasibility min slack = {metrics['feasibility_min_slack']:.3e}  -> "
          f"{'PASS' if feas_pass else 'FAIL'}")
    print(f"[verify] topology: {n_events} events == {n_event_maps} event-maps == "
          f"{n_nonroot} non-root nodes  -> {'PASS' if topo_ok else 'FAIL'}")
    return metrics
