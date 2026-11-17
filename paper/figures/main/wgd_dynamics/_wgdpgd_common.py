"""Stub: the shared punctuated-gain helpers live in ../../_shared/wgdpgd_common.py; this binds the
directory THIS figure's panels are written to, so `from _wgdpgd_common import ...` works unchanged."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))
import wgdpgd_common as _c  # noqa: E402
from wgdpgd_common import *  # noqa: E402,F401,F403
from wgdpgd_common import _reps  # noqa: E402,F401

PANELS = Path(__file__).resolve().parent


def save_panel(fig, stem, save=True, exact=True, panels_dir=None):
    return _c.save_panel(fig, stem, save=save, exact=exact, panels_dir=panels_dir or PANELS)
