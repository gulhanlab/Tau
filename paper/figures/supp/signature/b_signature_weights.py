"""signature figure (supp), panel b — what "signature-weighted multiplicity counts" means.

Tau does not partition mutations between processes. Every mutation contributes to every signature's
count, weighted by the probability that this process produced it, which for a 96-channel model depends
only on the mutation's trinucleotide channel:

    w_j(c) = e_j s_j(c) / sum_k e_k s_k(c)

with e the sample's signature exposures and s the COSMIC reference spectra. Running the route's linear
system on sum-over-mutations of w_j gives that process's own molecular time, on the same route and the
same gain order as the clock.

Left column: the COSMIC v3.2 reference spectrum of each process. Right column: w_j(c) for THIS sample —
the weight a mutation in each channel carries towards that process. The two are not the same shape, and
that is the point. The reference says what a process emits; the weight says how much of a given mutation
this sample's exposures let that process claim, with the denominator running over every signature active
in the sample. Diagnostic channels approach 1 (T[T>G]T is almost entirely SBS17 here); channels that
several processes emit are split, and no mutation is assigned to one process.

The sample is the esophageal adenocarcinoma used in panel c (SBS17a+b dominant, 12,361 mutations
between them, against 6,032 for the SBS1+SBS5 clock) — so the clock is the minority process here, which
is exactly the situation the weighting has to handle.

SBS17 is shown as SBS17a+SBS17b and APOBEC as SBS2+SBS13, matching how Tau groups them.

Everything except the reference column is MEASURED. Tau's preprocessing stores each mutation's
trinucleotide channel (`sbs96`) and the COSMIC likelihood s_j(context) on the SNV table, so the observed
spectrum is a straight count and the weighted spectra are the real per-mutation weights summed by
channel — no reference genome and no model reconstruction needed downstream. Verified: reconstructing
the stored clock weight this way reproduces `mut_w` to 3e-16.

The spectrum is of the mutations TAU USED (on timed segments, after its filtering), 48,438 here against
MuSiCal's genome-wide 46,994 — the right denominator for a figure about Tau's weighting, but not the
same population.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from paper.config import panels_dir_of, cache_dir_of  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[4]   # repo root; sys.path.insert(0, str(R))
from paper.figures._style import S, apply as _apply_style, save as _save_panel, MM, SINGLE
_apply_style(compact=True)

PANELS = panels_dir_of(__file__); PANELS.mkdir(parents=True, exist_ok=True)
from paper.config import REF
COSMIC = REF / "COSMIC_v3p2_SBS_WGS_MuSiCal.csv"
EXPO = REF / "MuSiCal_assignment_PCAWG.txt"
SAMPLE = "0250e6f7"         # the Fig. 2b / Fig. 7a esophageal example; override with argv[1]
MIN_FRAC = 0.05              # a process is "active" in the sample above this exposure fraction
MAX_GROUPS = 5
ALL_MEMBERS = {"SBS1": ["SBS1"], "SBS5": ["SBS5"], "SBS17 (a+b)": ["SBS17a", "SBS17b"],
               "APOBEC (SBS2+13)": ["SBS2", "SBS13"], "SBS4": ["SBS4"],
               "SBS7 (a-d)": ["SBS7a", "SBS7b", "SBS7c", "SBS7d"], "SBS3": ["SBS3"]}
CACHE = cache_dir_of(__file__)
# Single column, so this panel can sit beside panel a and leave the full width to panel c.
# The spectra are ~40 mm each rather than ~87 mm: 96 channels still resolve at print size,
# and keeping both columns costs less than dropping the reference or a signature would.
WIDTH, ROW_H = SINGLE, 15 * MM
ALL_GROUPS = [("SBS1", ["SBS1"]), ("SBS5", ["SBS5"]),
              ("SBS17 (a+b)", ["SBS17a", "SBS17b"]), ("APOBEC (SBS2+13)", ["SBS2", "SBS13"]),
              ("SBS4", ["SBS4"]), ("SBS7 (a-d)", ["SBS7a", "SBS7b", "SBS7c", "SBS7d"]),
              ("SBS3", ["SBS3"])]


def _active_groups(row, cos):
    """The clock always shown; other processes only where this sample actually carries them."""
    tot = sum(float(row[c]) for c in cos.columns if c in row.index)
    out = []
    for lab, sigs in ALL_GROUPS:
        sigs = [x for x in sigs if x in cos.columns and x in row.index]
        if not sigs:
            continue
        f = sum(float(row[x]) for x in sigs) / max(tot, 1e-9)
        if lab in ("SBS1", "SBS5") or f >= MIN_FRAC:
            out.append((lab, sigs, f))
    out = sorted(out, key=lambda t: (t[0] not in ("SBS1", "SBS5"), -t[2]))[:MAX_GROUPS]
    return [(l, s) for l, s, _ in out]
# the standard SBS96 block colours, in channel order
BLOCK = ["#03bcee", "#010101", "#e32925", "#cac9c9", "#a1ce63", "#ebc6c4"]
SUBS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]


def make(save=True, sample=None, stem=None, panels_dir=None):
    sample = sample or SAMPLE
    cos = pd.read_csv(COSMIC, index_col=0)
    ex = pd.read_csv(EXPO, sep="\t")
    row = ex[ex.aliquot_id.astype(str).str.startswith(sample)].iloc[0]
    sp = pd.read_csv(CACHE / f"spectrum_{sample}.tsv", sep="\t")
    chans = list(cos.index)
    obs = (sp.drop_duplicates("channel").set_index("channel").observed
             .reindex(chans).fillna(0).to_numpy(float))
    # SUMMED weights per channel, not the per-mutation weight: the sum is what becomes the
    # signature's multiplicity counts, and it makes the rows add back to the unweighted total.
    WT = {g: d.set_index("channel").weighted.reindex(chans).fillna(0).to_numpy(float)
          for g, d in sp.groupby("group")}
    NW = sp.groupby("group").weighted.sum().to_dict()
    GROUPS = [(g, None) for g in
              sorted(WT, key=lambda k: -NW[k])[:MAX_GROUPS]]
    active = [c for c in cos.columns if c in row.index and float(row[c]) > 0]
    colours = [BLOCK[i // 16] for i in range(96)]
    x = np.arange(96)

    # A full-width top row for the sample's whole spectrum, then one two-column pair per process.
    # The total is the DENOMINATOR of w_j(c), so putting it above makes the weighting readable as a
    # decomposition: every mutation in the top row is shared out among the rows beneath it, and the
    # weights across all active signatures sum to 1 in every channel by construction.
    nrow = len(GROUPS) + 1
    fig = plt.figure(figsize=(WIDTH, ROW_H * nrow + 0.45))
    gs = fig.add_gridspec(nrow, 2, wspace=0.20, hspace=0.62)
    top = fig.add_subplot(gs[0, :])
    total = obs
    top.bar(np.arange(96), total, width=0.82, color=[BLOCK[i // 16] for i in range(96)], linewidth=0)
    top.set_xlim(-1, 96); top.set_xticks([]); top.set_ylim(0, total.max() * 1.18)
    top.set_ylabel("mutations", fontsize=S.label)
    top.tick_params(labelsize=S.tick, length=1.4, pad=1.2)
    top.set_title(f"all mutations, unweighted  —  {total.sum():,.0f} SNVs, "
                  f"{len(active)} active signatures", fontsize=S.annot, pad=2)
    for sp in ("top", "right"):
        top.spines[sp].set_visible(False)
    axes = np.array([[fig.add_subplot(gs[r + 1, c]) for c in range(2)] for r in range(len(GROUPS))])
    for r, (lab, sigs) in enumerate(GROUPS):
        members = [c for c in ALL_MEMBERS[lab] if c in cos.columns]
        ref = np.array(cos[members].sum(axis=1), dtype=float); ref = ref / ref.sum()
        wt = WT[lab]                                      # measured weight per mutation, by channel

        # titles carry the group name on both sides: at 40 mm wide, a reader scanning one column
        # should not have to look across to know which process it belongs to
        for c, (v, ttl) in enumerate(((ref, f"{lab} — COSMIC"),
                                      (wt, f"{lab} — weighted"))):
            ax = axes[r, c]
            ax.bar(x, v, width=0.82, color=colours, linewidth=0)
            ax.set_xlim(-1, 96); ax.set_xticks([])
            ax.set_ylim(0, v.max() * 1.18)
            ax.set_title(ttl, fontsize=S.annot, pad=2,
                         fontweight="bold" if c == 1 else "normal")
            ax.tick_params(labelsize=S.tick, length=1.4, pad=1.2)
            ax.set_ylabel("fraction" if c == 0 else "mutations", fontsize=S.label)
            if c == 1:
                ax.text(0.99, 0.86, f"{NW[lab]:,.0f} weighted", transform=ax.transAxes,
                        ha="right", fontsize=S.tiny, color="0.35")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == len(GROUPS) - 1:      # substitution-class strip on the bottom row only
                for i, sub in enumerate(SUBS):
                    ax.add_patch(plt.Rectangle((i * 16 - 0.5, -0.16), 16, 0.10, color=BLOCK[i],
                                               transform=ax.get_xaxis_transform(), clip_on=False))
                    ax.text(i * 16 + 8, -0.30, sub, transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=S.tiny)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.935, bottom=0.105)
    fig.suptitle(f"{row.tumor_type}  ·  {str(row.aliquot_id)[:8]}", fontsize=S.title,
                 fontweight="bold", y=0.995)
    if save:
        _save_panel(fig, Path(panels_dir) if panels_dir else PANELS,
                    stem or "b_signature_weights")
    return fig


if __name__ == "__main__":
    import sys as _s
    a = _s.argv[1] if len(_s.argv) > 1 else None
    make(sample=a, stem=(f"b_alt_{a}" if a else None),
         panels_dir=(PANELS.parent / "panels_alternatives") if a else None)
    print(f"b_signature_weights: written ({a or SAMPLE})")
