"""Build paper/figures/export/ from manifest.py: one directory per figure, one file per panel named
<letter>_<name>.{svg,png,pdf}, plus README.md (the manifest as a table) and index.html (contact sheet).
Wiped and rebuilt on every run -- this is the tree to hand over for figure assembly.
Usage:  python paper/figures/export_panels.py [--check]
"""
import re, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import MAIN, SUPP, UNUSED  # noqa: E402
from _svgfix import longhand_fonts, arrows_to_paths  # noqa: E402

EXPORT = HERE / "export"
EXTS = ("svg", "pdf", "png")


FONT = "Helvetica"
_FAM = re.compile(r"(font(?:-family)?:\s*(?:(?:italic|oblique|normal|bold|[1-9]00)\s+)*(?:[0-9.]+px\s+)?)"
                  r"('[^']*'(?:,\s*'[^']*')*(?:,\s*sans-serif)?)")


def _helvetica_only(svg_path):
    """Rewrite every font-family list in the SVG to Helvetica alone. matplotlib writes the whole fallback
    stack (and names the fallback font it actually used for glyphs like the arrow), which makes
    Illustrator warn about Nimbus Sans / DejaVu Sans on import; Helvetica is what the paper uses."""
    t = svg_path.read_text()
    t2 = _FAM.sub(lambda m: m.group(1) + f"'{FONT}'", t)
    t2 = re.sub(r"font-family=\"[^\"]*\"", f'font-family="{FONT}"', t2)      # attribute form
    t2 = re.sub(r"font-family:\s*\"[^\"]*\"", f'font-family: "{FONT}"', t2)   # svglite (R) style form
    if t2 != t:
        svg_path.write_text(t2)


_SAFE = re.compile(r"[^\x20-\x7e\xa0-\xff\u2013\u2014\u2018\u2019\u201c\u201d\u2022\u2026\u00b7\u2212\u2265\u2264\u00b1\u00d7\u03b4\u0394\u2192]")
_WARN = re.compile(r"[\u2192\u03b4\u0394]")   # glyphs Helvetica may lack in Illustrator: report them


def _glyph_check(svg_path, rel):
    """Print any text glyph Illustrator's Helvetica may not have (the '->' arrow, Greek), so it can be
    replaced at source rather than silently dropped on import."""
    t = svg_path.read_text()
    txt = " ".join(re.findall(r">([^<]+)</text>", t) + re.findall(r">([^<]+)</tspan>", t))
    bad = sorted(set(_WARN.findall(txt)))
    if bad:
        print(f"   glyphs to check in {rel}: {' '.join(bad)}")


def _files(src_stem):
    base = HERE / src_stem
    return [base.with_suffix("." + e) for e in EXTS if base.with_suffix("." + e).exists()]


def run(check=False):
    if not check:
        if EXPORT.exists():
            shutil.rmtree(EXPORT)
        EXPORT.mkdir()
    rows, missing, no_svg, sheet = [], [], [], []
    for group, figs in (("main", MAIN), ("supp", SUPP)):
        for fig in figs:
            dest = EXPORT / group / fig["label"]
            if not check:
                dest.mkdir(parents=True)
            sheet.append(f'<h2 id="{group}-{fig["label"]}">{group}/{fig["label"]} — {fig.get("title", "")}</h2><div class="row">')
            for letter, src, note in fig["panels"]:
                if src is None:
                    missing.append((group, fig["label"], letter, note))
                    rows.append((group, fig["label"], letter, "—", "MISSING: " + note)); continue
                files = _files(src)
                if not files:
                    missing.append((group, fig["label"], letter, f"no rendered file at {src}.*"))
                    rows.append((group, fig["label"], letter, src, "MISSING: not rendered")); continue
                if not any(f.suffix == ".svg" for f in files):
                    no_svg.append((group, fig["label"], letter, src))
                rows.append((group, fig["label"], letter, src, note))
                # paper letter + descriptive stem; the SOURCE script's own letter prefix is dropped
                # (c_bursts_pile_at_wgd -> b_bursts_pile_at_wgd)
                stem = (f"{letter}_" if letter else "") + re.sub(r"^[a-z]_(?=[a-z])", "", Path(src).name)
                if not check:
                    for f in files:
                        shutil.copy2(f, dest / (stem + f.suffix))
                        if f.suffix == ".svg":
                            longhand_fonts(dest / (stem + f.suffix))     # Illustrator drops the size in `font: 700 6.5px …`
                            _helvetica_only(dest / (stem + f.suffix))
                            arrows_to_paths(dest / (stem + f.suffix))    # U+2192 -> vector path (Helvetica lacks it)
                            _glyph_check(dest / (stem + f.suffix), f"{group}/{fig['label']}/{stem}")
                rel = f"{group}/{fig['label']}/{stem}"
                if any(f.suffix == ".png" for f in files):
                    sheet.append(f'<figure><a href="{rel}.svg"><img src="{rel}.png" loading="lazy"></a>'
                                 f'<figcaption>{stem}</figcaption></figure>')
                else:
                    sheet.append(f'<figure class="pdf"><a href="{rel}{files[0].suffix}">{stem}{files[0].suffix}</a>'
                                 f'<figcaption>{stem} (no PNG)</figcaption></figure>')
            sheet.append("</div>")

    lines = ["# Paper figure panels — exported from `manifest.py`", "",
             "Derived tree: edit the panel script or `manifest.py` and re-run `export_panels.py`. "
             "Sources live next to their scripts in `paper/figures/main/<label>/` and `supp/<label>/`.", "",
             "| set | figure | panel | file (paper/figures/…) | note |", "|---|---|---|---|---|"]
    lines += [f"| {g} | {lab} | {let or '—'} | `{src}` | {note} |" for g, lab, let, src, note in rows]
    lines += ["", "## Missing — in the manuscript, no reproducible source here", ""]
    lines += [f"- **{g}/{lab}** panel **{let or '—'}**: {note}" for g, lab, let, note in missing] or ["- none"]
    if no_svg:
        lines += ["", "## Rendered but no SVG (only png/pdf)", ""]
        lines += [f"- {g}/{lab} {let}: `{src}`" for g, lab, let, src in no_svg]
    lines += ["", "## Unused — produced in the repo, not in the manuscript", ""]
    lines += [f"- `{u}`" for u in UNUSED]
    if not check:
        (EXPORT / "README.md").write_text("\n".join(lines) + "\n")

    toc = " · ".join(f'<a href="#{g}-{f["label"]}">{f["label"]}</a>' for g, figs in (("main", MAIN), ("supp", SUPP)) for f in figs)
    if not check:
      (EXPORT / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Tau paper panels</title>'
        '<style>body{font:13px system-ui;margin:20px;color:#222}h2{margin:28px 0 6px;border-bottom:1px solid #ddd}'
        '.row{display:flex;flex-wrap:wrap;gap:14px}figure{margin:0;width:300px}img{width:300px;border:1px solid #ccc}'
        'figcaption{font-family:monospace;font-size:11px;margin-top:3px}.pdf{padding:20px;border:1px dashed #aaa;text-align:center}'
        '.toc{line-height:1.9}</style><h1>Tau paper panels</h1>'
        f'<p class="toc">{toc}</p>' + "".join(sheet))
    n_ok = sum(1 for r in rows if not r[4].startswith("MISSING"))
    print(f"{'checked' if check else 'exported'} {n_ok} panels  |  {len(missing)} missing  |  {len(no_svg)} without svg  |  {len(UNUSED)} unused")
    for g, lab, let, note in missing:
        print(f"   MISSING  {g}/{lab} {let or '—'}: {note[:90]}")
    return rows, missing


if __name__ == "__main__":
    run(check="--check" in sys.argv)
