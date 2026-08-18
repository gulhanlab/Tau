#!/usr/bin/env python3
"""
convert_solutions_to_analytical.py

Convert all Tau Sage solutions into an analytical coefficient form:

    G_j = total_cn × (a_base · N  +  c1·t_free1  +  c2·t_free2  + ...) / D

where D = Σᵢ i·Nᵢ, total_cn = major_cn + minor_cn, and G_j is the j-th
event time (cumulative sum of timing intervals up to and including interval j).

a_base[i] is the coefficient on N_i in the unnormalized cumulative sum.
free_coefs[k] is the coefficient on the k-th free variable.

This form is directly comparable to learned mixture model coefficients a_k:
    predicted_G_j ≈ total_cn × (a_k · m) / D_norm
where m_i = N_i / total_muts are the normalised multiplicity fractions.

Output: tau/data/solutions_analytical/<major>_<minor>.sobj
Each file is a pickle dict keyed by route_key → route_form dict.

Route form dict:
    major_cn, minor_cn, total_cn, n_mult
    n_free_vars          : int
    free_var_names       : [str, ...]
    structural_zero_mults: [int, ...]  e.g. [3] means N3==0 for this route
    events: {
        "G1": {a_base: [float,...], free_coefs: [float,...], is_determined: bool},
        ...
    }
    free_var_bounds: {
        "t4": {
            "lb": [{"N_coefs":[...], "free_coefs":{var:coef,...}, "clamp_zero":bool}],
            "ub": [{"N_coefs":[...], "free_coefs":{var:coef,...}}],
        }, ...
    }
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from tau.timing import load_routes_for_states

SOL_DIR  = Path(__file__).parent / "solutions"
OUT_DIR  = Path(__file__).parent / "solutions_analytical"
OUT_DIR.mkdir(exist_ok=True)

# Accept optional --state major_minor argument for SLURM array parallelism
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--state", default=None,
                    help="Single CN state to convert, e.g. '4_2'. If omitted, converts all.")
args = parser.parse_args()

if args.state:
    major, minor = map(int, args.state.split("_"))
    cn_states = [(major, minor)]
else:
    cn_states = []
    for p in sorted(SOL_DIR.glob("*.sobj")):
        parts = p.stem.split("_")
        cn_states.append((int(parts[0]), int(parts[1])))

print(f"Converting {len(cn_states)} CN state(s)...")


def _is_equality(eq):
    s = str(eq)
    return "==" in s and "<=" not in s and ">=" not in s


def _extract_linear_coefs(expr, N_vars, free_vars):
    """Return (N_coefs, free_coefs_dict) from a linear Sage expression."""
    N_coefs = []
    for v in N_vars:
        try:
            c = float(expr.coefficient(v))
        except Exception:
            c = 0.0
        N_coefs.append(c)
    free_coefs = {}
    for name, v in free_vars.items():
        try:
            c = float(expr.coefficient(v))
        except Exception:
            c = 0.0
        if c != 0.0:
            free_coefs[name] = c
    return N_coefs, free_coefs


def _parse_bound_expr(expr, N_vars, free_vars):
    """
    Parse a bound expression — handles plain linear exprs and min()/max() wrappers.
    Returns a list of {"N_coefs":..., "free_coefs":..., "clamp_zero":bool} dicts,
    one entry per branch of min() (upper bound may be min of several exprs).
    """
    s = str(expr)
    # Handle max(0, inner) — lower bound clamped at 0
    if s.startswith("max("):
        operands = expr.operands()
        # Find the non-zero operand
        for op in operands:
            if str(op) != "0":
                N_c, f_c = _extract_linear_coefs(op, N_vars, free_vars)
                return [{"N_coefs": N_c, "free_coefs": f_c, "clamp_zero": True}]
        return [{"N_coefs": [0.]*len(N_vars), "free_coefs": {}, "clamp_zero": True}]
    # Handle min(expr1, expr2) — upper bound is minimum of multiple exprs
    if s.startswith("min("):
        results = []
        for op in expr.operands():
            N_c, f_c = _extract_linear_coefs(op, N_vars, free_vars)
            results.append({"N_coefs": N_c, "free_coefs": f_c, "clamp_zero": False})
        return results
    # Plain linear expression
    N_c, f_c = _extract_linear_coefs(expr, N_vars, free_vars)
    return [{"N_coefs": N_c, "free_coefs": f_c, "clamp_zero": False}]


def convert_route(route_key, sol_equations, major_cn, minor_cn):
    total_cn = major_cn + minor_cn
    n_mult   = major_cn  # multiplicities run 1..major_cn

    # ── Identify variables ────────────────────────────────────────────────────
    # Collect all symbolic variables appearing in the solution
    all_vars = {}
    for eq in sol_equations:
        try:
            for v in eq.variables():
                all_vars[str(v)] = v
        except Exception:
            pass

    N_vars   = {f"N{i}": all_vars[f"N{i}"] for i in range(1, n_mult+1)
                if f"N{i}" in all_vars}
    N_var_list = [N_vars.get(f"N{i}") for i in range(1, n_mult+1)]

    # Timing interval equality equations: t_j == expr
    t_eq_lhs  = {}
    structural_zero_mults = []
    for eq in sol_equations:
        if not _is_equality(eq):
            continue
        lhs_s = str(eq.lhs())
        rhs   = eq.rhs()
        if lhs_s.startswith("t"):
            t_eq_lhs[lhs_s] = rhs
        elif lhs_s.startswith("N") and str(rhs) == "0":
            idx = int(lhs_s[1:])
            structural_zero_mults.append(idx)

    # Free variables: t_j vars that appear in expressions but are NOT
    # defined by an equality equation (they only appear in inequalities)
    t_all_names = {str(v) for v in all_vars if str(v).startswith("t")}
    free_var_names = sorted(t_all_names - set(t_eq_lhs.keys()))
    free_vars = {n: all_vars[n] for n in free_var_names if n in all_vars}

    # ── Parse bounds on free variables ────────────────────────────────────────
    free_var_bounds = {n: {"lb": [], "ub": []} for n in free_var_names}
    for eq in sol_equations:
        if _is_equality(eq):
            continue
        s = str(eq)
        try:
            lhs_s = str(eq.lhs())
            rhs_s = str(eq.rhs())
            # lower bound: something <= t_free
            if rhs_s in free_var_names:
                parsed = _parse_bound_expr(eq.lhs(), N_var_list, free_vars)
                free_var_bounds[rhs_s]["lb"].extend(parsed)
            # upper bound: t_free <= something
            elif lhs_s in free_var_names:
                parsed = _parse_bound_expr(eq.rhs(), N_var_list, free_vars)
                free_var_bounds[lhs_s]["ub"].extend(parsed)
        except Exception:
            pass

    # ── Compute event times (cumulative sums of intervals) ────────────────────
    # Sort interval names: t1 < t2 < t3 ...
    interval_names = sorted(t_eq_lhs.keys(), key=lambda x: int(x[1:]))
    n_intervals = len(interval_names)

    events = {}
    for j in range(1, n_intervals + 1):  # G_j = cumsum of t_1..t_j (include last interval)
        cumsum_expr = sum(t_eq_lhs[interval_names[i]] for i in range(j))
        # Extract a_base (coefficients on N_i) and free_coefs
        a_base = []
        for v in N_var_list:
            if v is None:
                a_base.append(0.0)
                continue
            try:
                a_base.append(float(cumsum_expr.coefficient(v)))
            except Exception:
                a_base.append(0.0)
        free_coefs = []
        for fname in free_var_names:
            fv = free_vars.get(fname)
            if fv is None:
                free_coefs.append(0.0)
                continue
            try:
                free_coefs.append(float(cumsum_expr.coefficient(fv)))
            except Exception:
                free_coefs.append(0.0)

        # Skip degenerate terminal event: for determined routes, cumsum of all
        # intervals = 1.0 always (G_n = 1 is not an informative event time).
        # Detected when a_base == [k/total_cn for k in 1..n_mult] AND all free_coefs == 0.
        terminal_a = [float(k) / total_cn for k in range(1, n_mult + 1)]
        if (all(abs(c) < 1e-9 for c in free_coefs) and
                len(a_base) == len(terminal_a) and
                all(abs(a_base[k] - terminal_a[k]) < 1e-9 for k in range(n_mult))):
            continue

        is_determined = all(c == 0.0 for c in free_coefs)
        events[f"G{j}"] = {
            "a_base":       a_base,
            "free_coefs":   free_coefs,
            "is_determined": is_determined,
        }

    return {
        "route_key":             route_key,
        "major_cn":              major_cn,
        "minor_cn":              minor_cn,
        "total_cn":              total_cn,
        "n_mult":                n_mult,
        "n_free_vars":           len(free_var_names),
        "free_var_names":        free_var_names,
        "structural_zero_mults": structural_zero_mults,
        "events":                events,
        "free_var_bounds":       free_var_bounds,
    }


# ── Main conversion loop ──────────────────────────────────────────────────────
for major, minor in cn_states:
    env = load_routes_for_states([(major, minor)])
    route_keys = [k for k in env.SOL if k.startswith(f"{major}_{minor}")]

    state_dict = {}
    for rk in route_keys:
        try:
            form = convert_route(rk, env.SOL[rk], major, minor)
            state_dict[rk] = form
            n_free = form["n_free_vars"]
            det    = sum(1 for e in form["events"].values() if e["is_determined"])
            total  = len(form["events"])
            szero  = form["structural_zero_mults"]
            print(f"  {rk}: {n_free} free vars, {det}/{total} events determined"
                  + (f", N{szero}==0" if szero else ""))
        except Exception as ex:
            print(f"  {rk}: ERROR — {ex}")

    out_path = OUT_DIR / f"{major}_{minor}.sobj"
    with open(out_path, "wb") as f:
        pickle.dump(state_dict, f)
    print(f"  → saved {out_path.name}")

print(f"\nDone. {len(cn_states)} files written to {OUT_DIR}/")
