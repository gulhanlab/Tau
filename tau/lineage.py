"""Forward-time lineage tracing for simulated segments — the ground truth a route label should encode.

WHY THIS EXISTS
---------------
`SimSegment.calculate_t_values` reconstructed t-values from the sequence of (major, minor) STATES, by
counting `sum(state - prev_state)` gains per event and clamping the index when later losses removed
copies. That loses the information the route is made of, and it is wrong in two ways:

  * a WGD on (1,1) counts 2 gains — one major and one minor — so if the minor allele is later lost the
    surviving (4,0) segment records a phantom gain, producing a spurious ZERO interval and a "1->3"
    jump that no doubling can produce (180 of 401 v8 (4,0) segments, 176 of them sitting on a WGD).
  * `assign_tags` then labelled the segment from `counts_diagram_As.txt`, which has NO minor==0 rows,
    so every LOH state fell through to "<M>_<m>.1" regardless of history. For a genuine
    (1,0)->WGD->(2,0)->WGD->(4,0) the true route is 4_0.2 with a zero third interval; labelling it
    4_0.1 emits the mutations at multiplicity 3 instead of multiplicity 2 — the DATA is wrong, not
    just the label (128 of 401 v8 (4,0) segments).

Tracing lineages forward fixes both at once: a lost copy is simply not in the tree, so it cannot
occupy a gain slot, and the surviving tree IS the route topology.

MODEL
-----
Each allele starts as one copy. Events act on living copies:

    wgd(t)                  every living copy splits in two            (a doubling, by definition)
    gain(t, allele, delta)  one living copy of `allele` gains `delta`  (focal amplification)
    loss(t, allele)         one living copy of `allele` dies

At the end the tree is pruned to copies that SURVIVE, single-child chains are collapsed (they are not
splits), and what remains is read off directly as:

    t_values      intervals between distinct split times
    events        (allele, parent_mult, split, time), the route_topology encoding
    route_class   every route whose topology is consistent with that tree  (STRUCTURAL truth)

Two levels of truth are recorded on purpose. `route_class` says what happened. Whether a method could
have RECOVERED it is a weaker question — co-temporal splits collapse multiplicities and can make
structurally different routes predict an identical N — so `observational_class()` is provided
separately as `equivalence_class` / `class_of`. Route selection should be scored against that class;
anything else penalises a method for failing to distinguish the indistinguishable.
"""
from __future__ import annotations

import gzip
import json
import os
from itertools import permutations

import numpy as np

_TOPOLOGY = None


def _load_topology(path=None):
    """route_topology.json.gz -> {route_key: ordered event list}. Cached."""
    global _TOPOLOGY
    if _TOPOLOGY is not None:
        return _TOPOLOGY
    path = path or os.path.join(os.path.dirname(__file__), "data", "route_topology.json.gz")
    with gzip.open(path, "rt") as fh:
        raw = json.load(fh)
    _TOPOLOGY = {
        k: sorted(v["events"], key=lambda e: e["col_index"]) for k, v in raw.items()
    }
    return _TOPOLOGY


class Copy:
    """One physical chromosome copy: born at `birth`, may split or die."""

    __slots__ = ("allele", "birth", "death", "parent", "children")

    def __init__(self, allele, birth=0.0, parent=None):
        self.allele = allele
        self.birth = float(birth)
        self.death = None          # set by loss(); None means it survives to sampling
        self.parent = parent
        self.children = []

    @property
    def alive(self):
        return self.death is None and not self.children

    def split(self, t):
        """Replace this copy by two children born at t. Returns the children."""
        a = Copy(self.allele, t, self)
        b = Copy(self.allele, t, self)
        self.children = [a, b]
        return a, b

    def n_surviving(self):
        if not self.children:
            return 0 if self.death is not None else 1
        return sum(c.n_surviving() for c in self.children)


class LineageTracker:
    """Forward-time lineage tracing for one segment, starting from (1,1)."""

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.roots = {"A": Copy("A", 0.0), "B": Copy("B", 0.0)}
        self.split_times = []      # (time, copy) in application order, for reproducibility

    # -- living copies ------------------------------------------------------
    def _living(self, allele=None):
        out = []
        stack = [r for a, r in self.roots.items() if allele is None or a == allele]
        while stack:
            c = stack.pop()
            if c.children:
                stack.extend(c.children)
            elif c.death is None:
                out.append(c)
        return out

    def counts(self):
        return {a: len(self._living(a)) for a in self.roots}

    def copy(self):
        """Deep copy of the tree.

        A simulated segment split in two by a later breakpoint shares its history up to that point and
        diverges after, so each half needs its OWN tree — sharing one would let an event applied to the
        left half appear in the right half's topology. The rng is deliberately shared, not cloned, so a
        run stays reproducible from a single stream given a fixed event order.
        """
        def rec(c, parent):
            n = Copy(c.allele, c.birth, parent)
            n.death = c.death
            n.children = [rec(k, n) for k in c.children]
            return n

        new = LineageTracker.__new__(LineageTracker)
        new.rng = self.rng
        new.roots = {a: rec(r, None) for a, r in self.roots.items()}
        new.split_times = list(self.split_times)
        return new

    # -- events -------------------------------------------------------------
    def wgd(self, t):
        """Every living copy splits in two. This is what makes a doubling a doubling."""
        for c in self._living():
            c.split(t)

    def gain(self, t, allele, delta=1):
        """One living copy of `allele` gains `delta` extra copies (focal amplification).

        Represented as `delta` binary splits at the SAME instant, which is how route topologies encode
        a multi-copy event — the zero-length intervals between them are real, not an artefact.
        """
        living = self._living(allele)
        if not living or delta < 1:
            return
        target = living[int(self.rng.integers(len(living)))]
        for _ in range(int(delta)):
            a, _b = target.split(t)
            target = a          # keep splitting the same lineage: one copy became delta+1

    def loss(self, t, allele):
        """One living copy of `allele` dies. Its lineage simply is not in the final tree."""
        living = self._living(allele)
        if not living:
            return
        living[int(self.rng.integers(len(living)))].death = float(t)

    # -- read the surviving tree -------------------------------------------
    def _pruned(self, allele):
        """Surviving sub-tree with single-child chains collapsed. -> root Copy or None."""
        def rec(c):
            kids = [k for k in (rec(x) for x in c.children) if k is not None]
            if not c.children:
                return None if c.death is not None else c
            if not kids:
                return None
            if len(kids) == 1:
                # one side died out, so this was not a real split: the survivor simply CONTINUES the
                # parent lineage and inherits its birth. (Keeping the child's own birth here was the
                # bug that pushed every event to the last split time.)
                kids[0].birth = c.birth
                return kids[0]
            node = Copy(c.allele, c.birth, None)      # a node's OWN birth, not its split time
            node.children = kids
            for k in kids:
                k.parent = node
            return node
        return rec(self.roots[allele])

    def final_state(self):
        """(major, minor) after all events, from surviving copies."""
        n = {a: (self._pruned(a).n_surviving() if self._pruned(a) else 0) for a in self.roots}
        hi, lo = max(n.values()), min(n.values())
        return hi, lo

    def topology(self):
        """-> (events, t_values). `events` matches route_topology's encoding, ordered by time."""
        maj_allele = max(self.roots, key=lambda a: (self._pruned(a).n_surviving()
                                                    if self._pruned(a) else 0))
        evs = []
        for allele in self.roots:
            root = self._pruned(allele)
            if root is None:
                continue
            label = "major" if allele == maj_allele else "minor"
            stack = [root]
            while stack:
                c = stack.pop()
                if not c.children:
                    continue
                mults = [k.n_surviving() for k in c.children]
                # the SPLIT happens when the children are born; c.birth is when c itself appeared
                evs.append({"allele": label, "parent_mult": c.n_surviving(),
                            "split": sorted(mults), "time": float(min(k.birth for k in c.children))})
                stack.extend(c.children)
        evs.sort(key=lambda e: e["time"])
        for i, e in enumerate(evs, 1):
            e["col_index"] = i
        times = [e["time"] for e in evs]
        t_vals = [times[0]] + [times[i] - times[i - 1] for i in range(1, len(times))] + \
                 [1.0 - times[-1]] if times else [1.0]
        return evs, [max(0.0, float(x)) for x in t_vals]

    # -- match to the route library ----------------------------------------
    def route_class(self):
        """Every route key whose TOPOLOGY is consistent with the traced tree (structural truth)."""
        evs, _ = self.topology()
        maj, mino = self.final_state()
        want = [(e["allele"], int(e["parent_mult"]), tuple(sorted(e["split"]))) for e in evs]
        wants = [want]
        if maj == mino:
            # BALANCED state: both alleles end with the same number of copies, so which one
            # topology() called "major" was decided by an arbitrary max() tie-break. The labels carry
            # no information here, and a route that lists the same splits with the labels exchanged
            # encodes the same history — so accept a match under either labelling. Without this every
            # balanced state whose tie-break disagreed with the library convention got NO route at all
            # (2,2 was the visible case; it is the commonest WGD state).
            sw = {"major": "minor", "minor": "major"}
            wants.append([(sw[a], m, s) for a, m, s in want])
        out = []
        for key, revs in _load_topology().items():
            if not key.startswith(f"{maj}_{mino}."):
                continue
            got = [(e["allele"], int(e["parent_mult"]), tuple(sorted(e["split"]))) for e in revs]
            if len(got) != len(want):
                continue
            if any(got == w or _tie_equivalent(w, got, evs) for w in wants):
                out.append(key)
        return sorted(out, key=lambda k: int(k.split(".")[1]))


def _tie_equivalent(want, got, evs):
    """True when `got` is a reordering of `want` that only permutes SIMULTANEOUS events.

    Events at the same instant have no intrinsic order, so any route listing them in a different
    order encodes the same history.
    """
    times = [e["time"] for e in evs]
    blocks, i = [], 0
    while i < len(times):
        j = i
        while j + 1 < len(times) and abs(times[j + 1] - times[i]) < 1e-9:
            j += 1
        blocks.append((i, j + 1))
        i = j + 1
    if all(b[1] - b[0] == 1 for b in blocks):
        return False
    for lo, hi in blocks:
        if sorted(want[lo:hi]) != sorted(got[lo:hi]):
            return False
    return True


def equivalence_class(t_vals, major, minor, matrices, tol=1e-12):
    """Routes indistinguishable for a history with THIS zero pattern.

    A zero-length interval contributes nothing to N = A.T @ t, so the corresponding matrix COLUMN is
    unconstrained: two routes that differ only in zeroed columns predict the same multiplicity vector
    for every timing sharing that pattern. Equivalence is therefore defined structurally — equality of
    the matrices restricted to the non-zero columns — rather than by comparing A.T @ t at one specific
    t, which can coincide by numerical accident and would not generalise.

    Returns the sorted list of route keys equivalent to the whole set (every key that shares the
    reduced matrix of at least one other), keyed by reduced matrix.
    """
    t = np.asarray(t_vals, float)
    keep = [i for i, x in enumerate(t) if abs(x) > tol]
    groups = {}
    for k, A in matrices.items():
        if not str(k).startswith(f"{major}_{minor}."):
            continue
        if A.shape[0] != len(t):
            continue
        red = np.asarray(A).T[:, keep]
        groups.setdefault(red.tobytes() + bytes(str(red.shape), "utf8"), []).append(str(k))
    return {kk: sorted(v, key=lambda x: int(x.split(".")[1])) for kk, v in groups.items()}


def class_of(route, t_vals, major, minor, matrices, tol=1e-12):
    """The equivalence class CONTAINING `route` under `equivalence_class`."""
    for members in equivalence_class(t_vals, major, minor, matrices, tol).values():
        if str(route) in members:
            return members
    return [str(route)]
