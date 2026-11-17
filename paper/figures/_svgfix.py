"""Make matplotlib SVGs readable by Illustrator (and other SVG parsers).

With `svg.fonttype: "none"` — which is what keeps label text editable rather than outlined — matplotlib
writes every text style using the CSS *shorthand* property:

    style="font: 700 5px 'Arial', 'Helvetica', 'DejaVu Sans', sans-serif; text-anchor: middle"

Illustrator's SVG parser does not implement the `font:` shorthand. It discards the entire declaration
and falls back to the SVG default font-size, **12**, so every label opens at 12 pt no matter what size
the figure was authored at. Rewriting the shorthand into longhand properties fixes it:

    style="font-family: 'Arial', ...; font-size: 5px; font-weight: 700; text-anchor: middle"

The rewrite is purely syntactic — identical rendering everywhere else — and idempotent, so it is safe
to run repeatedly over a directory of already-rendered panels.

Use it either per-save (see `paper/figures/_style.py::save`) or as a repair pass over existing
output:

    python paper/figures/_svgfix.py                 # repair every SVG under paper/
    python paper/figures/_svgfix.py path/to/dir ...  # or specific files/directories
"""
import re
import sys
from pathlib import Path

# font: [ <style> || <variant> || <weight> ] <size>px <family-list>
_SHORTHAND = re.compile(r"font:\s*((?:[A-Za-z]+|\d{3})\s+)*?([\d.]+)px\s+([^;\"]+)")
_STYLES = {"italic", "oblique", "normal"}
_VARIANTS = {"small-caps"}


def _expand(match):
    head, size, family = match.group(0), match.group(2), match.group(3).strip()
    # everything between "font:" and the size is some mix of style / variant / weight keywords
    pre = head[len("font:"):head.index(f"{size}px")].split()
    out = [f"font-family: {family}", f"font-size: {size}px"]
    for tok in pre:
        low = tok.lower()
        if low in _STYLES:
            out.append(f"font-style: {low}")
        elif low in _VARIANTS:
            out.append(f"font-variant: {low}")
        else:                                   # numeric weight, or bold / bolder / lighter
            out.append(f"font-weight: {low}")
    return "; ".join(out)


def longhand_fonts(path):
    """Rewrite `font:` shorthand to longhand in one SVG. Returns the number of declarations changed."""
    p = Path(path)
    txt = p.read_text()
    new, n = _SHORTHAND.subn(_expand, txt)
    if n:
        p.write_text(new)
    return n


def repair(targets):
    """Repair SVGs at the given files/directories (directories are walked). Returns (files, decls)."""
    svgs = []
    for t in targets:
        t = Path(t)
        svgs.extend(sorted(t.rglob("*.svg")) if t.is_dir() else [t])
    files = decls = 0
    for s in svgs:
        n = longhand_fonts(s)
        if n:
            files += 1
            decls += n
    return files, decls, len(svgs)


if __name__ == "__main__":
    roots = sys.argv[1:] or [Path(__file__).resolve().parents[1]]      # default: all of paper/
    files, decls, scanned = repair(roots)
    print(f"svgfix: repaired {files}/{scanned} SVGs ({decls} font declarations)")


# ───────────── glyphs Illustrator's Helvetica lacks → inline vector paths, keeping the rest as text ─────────────
import re as _re

# right arrow: 0.9 em wide shaft with a chevron head, centred on the x-height (y=350), stroke ~0.09 em
_GLYPH_PATHS = {
    "→": "M 60 305 L 650 305 L 500 155 L 570 90 L 900 350 L 570 610 L 500 545 L 650 395 L 60 395 Z",
}
_ADV = {"→": 980}                                      # advance width per glyph, em units
_TEXT_RE = _re.compile(r"<text([^>]*)>([^<]*)</text>")
_STYLE_SIZE = _re.compile(r"font-size:\s*([0-9.]+)px")
_FILL = _re.compile(r"fill:\s*(#[0-9a-fA-F]{6}|[a-z]+)")
_XY = _re.compile(r'\b(x|y)="([-0-9.]+)"')
_TRANSFORM = _re.compile(r'transform="([^"]*)"')
_ANCHOR = _re.compile(r"text-anchor:\s*(start|middle|end)")
_MEASURE_FIG = None


def _adv_px(s, size):
    """Advance of string `s` at `size` px, measured with matplotlib's text layout in Nimbus Sans
    (Helvetica-metric) at 72 dpi, so 1 px == 1 pt as in the SVG."""
    global _MEASURE_FIG
    if not s:
        return 0.0
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if _MEASURE_FIG is None:
            _MEASURE_FIG = plt.figure(figsize=(1, 1), dpi=72)
        t = _MEASURE_FIG.text(0, 0, s, fontsize=size, family="Nimbus Sans")
        _MEASURE_FIG.canvas.draw()
        w = t.get_window_extent(_MEASURE_FIG.canvas.get_renderer()).width
        t.remove()
        return float(w)
    except Exception:
        return 0.55 * size * len(s)


def _text_width_units(s, size):
    """Advance in px for `s` at `size` px: measured widths for text, 0.98 em per arrow."""
    parts = s.split("→")
    return sum(_adv_px(x, size) for x in parts) + (len(parts) - 1) * _ADV["→"] / 1000.0 * size


def arrows_to_paths(path):
    """Replace every U+2192 (→) inside text runs of the SVG with an inline <path> of the same size and
    colour, splitting the run so the surrounding words stay real text. Runs are assumed to be single
    line, positioned by x/y (+ optional transform), as matplotlib emits them."""
    t = path.read_text()

    def fix(m):
        attrs, txt = m.group(1), m.group(2)
        if "→" not in txt:
            return m.group(0)
        size = float(_STYLE_SIZE.search(attrs).group(1)) if _STYLE_SIZE.search(attrs) else 7.0
        fill = _FILL.search(attrs).group(1) if _FILL.search(attrs) else "#000000"
        xy = dict(_XY.findall(attrs)); x0, y0 = float(xy.get("x", 0)), float(xy.get("y", 0))
        anchor = _ANCHOR.search(attrs).group(1) if _ANCHOR.search(attrs) else "start"
        tr = _TRANSFORM.search(attrs); tr = tr.group(1) if tr else ""
        total = _text_width_units(txt, size)
        x = x0 - {"start": 0.0, "middle": total / 2, "end": total}[anchor]
        base_attrs = _re.sub(r'\b(x|y)="[-0-9.]+"', "", attrs)
        base_attrs = _ANCHOR.sub("text-anchor: start", base_attrs)
        out, cur = [], ""
        for ch in txt:
            if ch != "→":
                cur += ch; continue
            if cur:
                out.append(f'<text{base_attrs} x="{x:.3f}" y="{y0:.3f}">{cur}</text>'); x += _adv_px(cur, size); cur = ""
            sc = size / 1000.0
            path_el = f'<path d="{_GLYPH_PATHS["→"]}" fill="{fill}" transform="translate({x:.3f} {y0:.3f}) scale({sc:.5f} {-sc:.5f})"/>'
            out.append(f'<g transform="{tr}">{path_el}</g>' if tr else path_el)
            x += _ADV["→"] / 1000.0 * size
        if cur:
            out.append(f'<text{base_attrs} x="{x:.3f}" y="{y0:.3f}">{cur}</text>')
        return "".join(out)

    t2 = _TEXT_RE.sub(fix, t)
    if t2 != t:
        path.write_text(t2)
    return t2 != t
