"""Sage-free unit tests for tau.tree_polytope's pure-arithmetic helpers.

These cover the affine / polytope-sampling logic that the browser mirrors, without
needing Sage or a loaded genome (which the heavier end-to-end build/verify exercises
separately via examples/build_tree_explorer_demo.py).
"""
import numpy as np

from tau.tree_polytope import (
    _grid_sample_polytope,
    _interval_1d,
    _affine_to_dict,
)


def _eval(c, free):
    return c["const"] + float(np.dot(c["coef"], free))


def test_affine_to_dict_roundtrip():
    d = _affine_to_dict(0.5, [1.0, -2.0])
    assert d == {"const": 0.5, "coef": [1.0, -2.0]}


def test_interval_1d_simple_box():
    # 0 <= x0 <= 3  and  0 <= x1 <= 5, as >=0 half-spaces
    cons = [
        {"const": 0.0, "coef": [1.0, 0.0]},   # x0 >= 0
        {"const": 3.0, "coef": [-1.0, 0.0]},  # 3 - x0 >= 0
        {"const": 0.0, "coef": [0.0, 1.0]},   # x1 >= 0
        {"const": 5.0, "coef": [0.0, -1.0]},  # 5 - x1 >= 0
    ]
    lo, hi = _interval_1d(cons, [0.0, 0.0], 0)
    assert abs(lo - 0.0) < 1e-9 and abs(hi - 3.0) < 1e-9
    lo, hi = _interval_1d(cons, [1.5, 0.0], 1)
    assert abs(lo - 0.0) < 1e-9 and abs(hi - 5.0) < 1e-9


def test_grid_sample_box_all_feasible():
    cons = [
        {"const": 0.0, "coef": [1.0, 0.0]},
        {"const": 3.0, "coef": [-1.0, 0.0]},
        {"const": 0.0, "coef": [0.0, 1.0]},
        {"const": 5.0, "coef": [0.0, -1.0]},
    ]
    pts = _grid_sample_polytope(2, cons, n_per=4, cap=64)
    assert len(pts) > 0
    for p in pts:
        assert all(_eval(c, p) >= -1e-9 for c in cons)
        assert 0 <= p[0] <= 3 and 0 <= p[1] <= 5


def test_grid_sample_coupled_triangle():
    # Triangle: x0 >= 0, x1 >= 0, x0 + x1 <= 1  (x1's upper bound couples to x0)
    cons = [
        {"const": 0.0, "coef": [1.0, 0.0]},    # x0 >= 0
        {"const": 0.0, "coef": [0.0, 1.0]},    # x1 >= 0
        {"const": 1.0, "coef": [-1.0, -1.0]},  # 1 - x0 - x1 >= 0
    ]
    pts = _grid_sample_polytope(2, cons, n_per=5, cap=100)
    assert len(pts) > 0
    for p in pts:
        assert all(_eval(c, p) >= -1e-9 for c in cons)
        assert p[0] + p[1] <= 1 + 1e-9


def test_grid_sample_zero_free():
    assert _grid_sample_polytope(0, []) == []
