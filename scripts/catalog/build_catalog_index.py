#!/usr/bin/env python
"""Merge per-shard catalog metadata into catalog.json + a self-contained index.html.

The index is a single static page (metadata embedded inline so it works over file://):
a sortable, filterable table of every sample — tumor type, ploidy, WGD/PGD, events,
mutations, purity — each row linking to its interactive viewer.

Usage:
    PATH=".pixi/envs/default/bin:$PATH" python dev/pcawg/signatures/build_catalog_index.py \
        --out dev/pcawg/catalog
"""
import argparse, glob, json, os, datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dev/pcawg/catalog")
    ap.add_argument("--title", default="Tau — PCAWG sample catalog")
    args = ap.parse_args()

    rows, fails = [], []
    for p in sorted(glob.glob(f"{args.out}/_meta/shard_*.json")):
        with open(p) as f:
            d = json.load(f)
        rows.extend(d.get("rows", []))
        fails.extend(d.get("fails", []))
    rows.sort(key=lambda r: (r["tumor_type"], r["uuid"]))

    with open(f"{args.out}/catalog.json", "w") as f:
        json.dump(dict(rows=rows, fails=fails), f)

    tumor_types = sorted({r["tumor_type"] for r in rows})
    n_wgd = sum(1 for r in rows if r.get("wgd"))
    built = datetime.date.today().isoformat()
    data_js = json.dumps(rows)
    tt_opts = "".join(f'<option value="{t}">{t}</option>' for t in tumor_types)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{args.title}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;
   color:#1b2733;background:#f7f9fb}}
 header{{background:#0b6979;color:#fff;padding:14px 22px}}
 header h1{{margin:0;font-size:19px}} header .sub{{opacity:.85;font-size:13px;margin-top:3px}}
 .bar{{position:sticky;top:0;background:#fff;border-bottom:1px solid #dde3e8;padding:10px 22px;
   display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:5}}
 .bar input,.bar select{{font-size:13px;padding:5px 8px;border:1px solid #cbd3da;border-radius:5px}}
 .bar label{{font-size:12px;color:#566;display:flex;gap:5px;align-items:center}}
 #count{{font-size:12px;color:#667;margin-left:auto}}
 table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff}}
 th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid #eef1f4;white-space:nowrap}}
 th{{position:sticky;top:55px;background:#eef3f6;cursor:pointer;user-select:none}}
 th:hover{{background:#e2eaef}} tr:hover td{{background:#f3f8fa}}
 a.view{{color:#0b6979;font-weight:600;text-decoration:none}} a.view:hover{{text-decoration:underline}}
 .pill{{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;font-weight:600}}
 .wgd{{background:#0b69791a;color:#0b6979}} .pgd{{background:#cd9f2c26;color:#9a7410}}
 .muted{{color:#9aa6b0}}
</style></head><body>
<header><h1>{args.title}</h1>
 <div class="sub">{len(rows)} samples · {len(tumor_types)} tumour types · {n_wgd} WGD · built {built}</div></header>
<div class="bar">
 <input id="q" placeholder="search uuid…" size="22" oninput="render()">
 <select id="tt" onchange="render()"><option value="">all tumour types</option>{tt_opts}</select>
 <label><input type="checkbox" id="wgd" onchange="render()"> WGD only</label>
 <label><input type="checkbox" id="pgd" onchange="render()"> PGD only</label>
 <span id="count"></span>
</div>
<table><thead><tr id="hd"></tr></thead><tbody id="tb"></tbody></table>
<script>
const ROWS={data_js};
const COLS=[
 ["tumor_type","tumour type",r=>r.tumor_type],
 ["uuid","sample",r=>r.uuid],
 ["events","events",r=>r.n_events],
 ["wgdpgd","WGD / PGD",r=>(r.n_wgd*100+r.n_pgd)],
 ["ploidy","ploidy",r=>r.ploidy],
 ["fga","frac amp",r=>r.frac_genome_amplified],
 ["maxcn","max CN",r=>r.max_major_cn],
 ["nmut","mutations",r=>r.n_mutations],
 ["purity","purity",r=>r.purity],
 ["size","MB",r=>r.size_mb],
];
let sortKey="tumor_type", asc=true;
function fmt(r,key){{
 if(key=="tumor_type")return r.tumor_type;
 if(key=="uuid")return `<a class="view" href="${{r.html}}" target="_blank">${{r.uuid.slice(0,12)}}…</a>`;
 if(key=="events")return r.n_events;
 if(key=="wgdpgd"){{let s="";
   if(r.n_wgd)s+=`<span class="pill wgd">WGD${{r.n_wgd>1?'×'+r.n_wgd:''}} ${{r.wgd_times.join(', ')}}</span> `;
   if(r.n_pgd)s+=`<span class="pill pgd">PGD${{r.n_pgd>1?'×'+r.n_pgd:''}} ${{r.pgd_times.join(', ')}}</span>`;
   return s||'<span class="muted">—</span>';}}
 if(key=="ploidy")return r.ploidy??'<span class="muted">—</span>';
 if(key=="fga")return r.frac_genome_amplified!=null?r.frac_genome_amplified.toFixed(2):'<span class="muted">—</span>';
 if(key=="maxcn")return r.max_major_cn??'<span class="muted">—</span>';
 if(key=="nmut")return r.n_mutations!=null?r.n_mutations.toLocaleString():'<span class="muted">—</span>';
 if(key=="purity")return r.purity!=null?r.purity.toFixed(2):'<span class="muted">—</span>';
 if(key=="size")return r.size_mb;
 return "";
}}
function header(){{
 document.getElementById("hd").innerHTML=COLS.map(c=>
   `<th onclick="sortBy('${{c[0]}}')">${{c[1]}}${{sortKey==c[0]?(asc?' ▲':' ▼'):''}}</th>`).join("");
}}
function sortBy(k){{asc=(sortKey==k)?!asc:true;sortKey=k;render();}}
function render(){{
 const q=document.getElementById("q").value.toLowerCase();
 const tt=document.getElementById("tt").value;
 const wOnly=document.getElementById("wgd").checked, pOnly=document.getElementById("pgd").checked;
 let rs=ROWS.filter(r=>(!q||r.uuid.toLowerCase().includes(q))&&(!tt||r.tumor_type==tt)
   &&(!wOnly||r.wgd)&&(!pOnly||r.pgd));
 const get=COLS.find(c=>c[0]==sortKey)[2];
 rs.sort((a,b)=>{{let x=get(a),y=get(b);if(x==null)x=-Infinity;if(y==null)y=-Infinity;
   return (x<y?-1:x>y?1:0)*(asc?1:-1);}});
 document.getElementById("count").textContent=rs.length+" shown";
 document.getElementById("tb").innerHTML=rs.map(r=>
   "<tr>"+COLS.map(c=>`<td>${{fmt(r,c[0])}}</td>`).join("")+"</tr>").join("");
 header();
}}
header();render();
</script></body></html>"""
    idx = f"{args.out}/index.html"
    with open(idx, "w") as f:
        f.write(html)
    print(f"wrote {idx}  ({len(rows)} samples, {len(tumor_types)} tumour types, "
          f"{len(fails)} failures)")
    if fails:
        print("  first failures:", fails[:5])


if __name__ == "__main__":
    main()
