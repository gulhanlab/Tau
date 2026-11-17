"""Shared genome geometry and the loss colour ramp for the post-WGD loss panels (b and c).

Both panels need the same hg19 offsets and the same arm boundaries, and a difference between them would
be a silent inconsistency rather than a visible one — so they come from here.

BLUE is the ramp used throughout the loss analyses (dev/pcawg/FigWGDPGD/loss_4versions.py); loss is
drawn blue so it reads as a different quantity from the inferno gain landscapes rather than more of the
same.
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HG19 = {"1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276, "5": 180915260, "6": 171115067,
        "7": 159138663, "8": 146364022, "9": 141213431, "10": 135534747, "11": 135006516, "12": 133851895,
        "13": 115169878, "14": 107349540, "15": 102531392, "16": 90354753, "17": 81195210, "18": 78077248,
        "19": 59128983, "20": 63025520, "21": 48129895, "22": 51304566}
CHROMS = [str(i) for i in range(1, 23)]
OFF = {}
_c = 0
for _k in CHROMS:
    OFF[_k] = _c
    _c += HG19[_k]
GEN = _c
CMID = {k: (OFF[k] + HG19[k] / 2) / 1e6 for k in CHROMS}
BOUND = [OFF[k] / 1e6 for k in CHROMS] + [GEN / 1e6]

CEN = {"1": 125e6, "2": 93.3e6, "3": 91e6, "4": 50.4e6, "5": 48.4e6, "6": 61e6, "7": 59.9e6,
       "8": 45.6e6, "9": 49e6, "10": 40.2e6, "11": 53.7e6, "12": 35.8e6, "13": 17.9e6, "14": 17.6e6,
       "15": 19e6, "16": 36.6e6, "17": 24e6, "18": 17.2e6, "19": 26.5e6, "20": 27.5e6, "21": 13.2e6,
       "22": 14.7e6}
ACRO = {"13", "14", "15", "21", "22"}          # p-arm heterochromatic — q only
ARMS = [(c, a) for c in CHROMS for a in (("q",) if c in ACRO else ("p", "q"))]

BLUE = LinearSegmentedColormap.from_list(
    "kblue", ["#000010", "#10265e", "#2a5db0", "#4a9be0", "#cdebff"])


def arm_span(c, a):
    lo = OFF[c] if a == "p" else OFF[c] + CEN[c]
    hi = OFF[c] + (CEN[c] if a == "p" else HG19[c])
    return lo, hi


def loss_per_opportunity(z, t, collapse_time=True):
    """Per-locus loss-per-opportunity for one tumour type, with no-opportunity loci masked to NaN.

    Loss alone is not comparable along the genome: a locus that amplified more has more copies
    available to lose. Dividing by the WGD-linked gain density at the same locus gives a per-branch
    rate. Loci whose opportunity sits below the 25th percentile are masked rather than divided by a
    near-zero denominator.
    """
    L = z[f"loss__{t}"]
    opp = z[f"opp__{t}"]
    flo = np.percentile(opp[opp > 0], 25) if (opp > 0).any() else 1e-3
    ok = opp >= flo
    num = L.sum(0) if collapse_time else L
    r = np.where(ok, num / np.clip(opp, flo, None), np.nan) if collapse_time else \
        np.where(ok[None, :], num / np.clip(opp, flo, None)[None, :], np.nan)
    return r, ok


def arm_values(z, t):
    """Arm-level loss-per-opportunity for one type -> (values, arm labels). Arms with no
    surviving opportunity are dropped."""
    N_X = int(z["N_X"])
    xe = np.linspace(0, GEN, N_X + 1)
    L = z[f"loss__{t}"].sum(0)
    opp = z[f"opp__{t}"]
    flo = np.percentile(opp[opp > 0], 25) if (opp > 0).any() else 1e-3
    ok = opp >= flo
    vals, labs = [], []
    for c, a in ARMS:
        lo, hi = arm_span(c, a)
        sel = np.arange(max(0, np.searchsorted(xe, lo) - 1), min(N_X, np.searchsorted(xe, hi)))
        sel = sel[ok[sel]]
        if sel.size:
            vals.append(L[sel].sum() / opp[sel].sum())
            labs.append(f"{c}{a}")
    return np.array(vals), labs
