#!/usr/bin/env python3
"""Build the tree_polytope payload for the dc537fcf (4,2) segment, verify it,
and emit a self-contained interactive HTML demo."""
import gzip
import json
import pickle
import sys

GENOME = ("/n/data1/hms/dbmi/park/jbrew/Tau/Tau_github/dev/pcawg/tau_outputs/"
          "output_sbs1sbs5_soft_allpcawg/Stomach.AdenoCA/"
          "dc537fcf-d910-4c4b-8af9-e7da429f2633/"
          "dc537fcf-d910-4c4b-8af9-e7da429f2633.genome.pkl.gz")
SEG_ID = "13:61635570-71312996"   # (4,2) route 4_2.4, 2 free vars, 892 SNVs
OUT_HTML = "/n/data1/hms/dbmi/park/jbrew/Tau/Tau_github/examples/tree_explorer_demo.html"

from tau.tree_polytope import build_tree_polytope_payload, verify_payload
from tau.timing import load_routes_for_states

print("Loading genome...", flush=True)
with gzip.open(GENOME, "rb") as f:
    g = pickle.load(f)

print("Loading routes (Sage)...", flush=True)
env = load_routes_for_states([(4, 2)])

print("Building payload...", flush=True)
payload = build_tree_polytope_payload(g, SEG_ID, route_key="4_2.4", env=env, n_samples=16)

print("\n--- payload summary ---")
print("route_key:", payload["route_key"])
print("free_vars:", payload["free_vars"])
print("t_vars:", payload["t_vars"])
print("n events:", len(payload["events"]))
print("n event_time_maps:", len(payload["event_time_maps"]))
print("n nodes:", len(payload["nodes"]))
print("n sampled solutions:", len(payload["sampled_solutions"]))
print("init_free:", payload["init_free"])
print("seg:", payload["seg"])
print("events:")
for e in payload["events"]:
    print("  ", e)
print("interval_maps:")
for k, v in payload["interval_maps"].items():
    print("  ", k, v)
print("event_time_maps:")
for i, v in enumerate(payload["event_time_maps"]):
    print("  G%d" % (i + 1), v)
print("bounds constraints:")
for c in payload["constraints"]["bounds"]:
    print("  ", c)
print("nodes:")
for nd in payload["nodes"]:
    print("  ", nd)

print("\n--- verification ---")
metrics = verify_payload(payload, g, SEG_ID, env=env)

# also build a second, simpler-route payload to exercise default selection + 0-free-var path
print("\n--- edge case: default route selection (auto) ---")
p_auto = build_tree_polytope_payload(g, SEG_ID, env=env, n_samples=8)
print("auto-selected route:", p_auto["route_key"], "free vars:", p_auto["num_free_vars"])

# ---- write HTML ----
import os
os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
DATA_JSON = json.dumps(payload)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Tau — Tree Solution with Knobs (proof of mechanic)</title>
<style>
  body { font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
         margin: 0; padding: 24px; color: #1d2430; background: #f7f8fa; }
  h1 { font-size: 19px; margin: 0 0 4px; }
  .sub { color: #5a6473; font-size: 13px; margin-bottom: 18px; }
  .wrap { display: flex; gap: 28px; flex-wrap: wrap; align-items: flex-start; }
  .panel { background: #fff; border: 1px solid #e3e6eb; border-radius: 10px;
           padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,.04); }
  svg { display: block; }
  .knob { margin: 10px 0 16px; }
  .knob label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 4px; }
  .knob .row { display: flex; align-items: center; gap: 10px; }
  .knob input[type=range] { flex: 1; }
  .knob .val { font-variant-numeric: tabular-nums; font-size: 12px; color: #2d3340;
               min-width: 62px; text-align: right; }
  .knob .rng { font-size: 11px; color: #8a93a3; }
  table.t { border-collapse: collapse; font-size: 12px; margin-top: 6px; }
  table.t td, table.t th { padding: 3px 9px; border-bottom: 1px solid #eef0f3; text-align: right; }
  table.t th { color: #5a6473; font-weight: 600; }
  .feas { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px;
          font-weight: 600; }
  .feas.ok { background: #e3f5ea; color: #1f8b4c; }
  .feas.bad { background: #fdecea; color: #c0392b; }
  .meta { font-size: 12px; color: #5a6473; line-height: 1.6; }
  .legend { font-size: 12px; color: #444; margin-top: 8px; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; margin-right: 16px; }
  .legend .swatch { width: 26px; height: 0; border-top: 3px solid; }
  code { background: #eef0f3; padding: 1px 5px; border-radius: 4px; font-size: 11px; }
</style>
</head>
<body>
<h1>Tau — interactive tree solution with knobs</h1>
<div class="sub">Dial each free variable to move through the polytope of valid timing solutions; the allele tree updates live. All recomputation is plain arithmetic on affine coefficients extracted once in Sage.</div>

<div class="wrap">
  <div class="panel">
    <div id="tree"></div>
    <div class="legend">
      <span><span class="swatch" style="border-color:#0b6979"></span>major allele (solid)</span>
      <span><span class="swatch" style="border-color:#0b6979;border-top-style:dashed"></span>minor allele (dashed)</span>
    </div>
  </div>
  <div class="panel" style="min-width:300px;">
    <div id="knobs"></div>
    <div style="margin:8px 0 14px;">Feasible: <span id="feas" class="feas ok">yes</span></div>
    <table class="t" id="ttbl"><thead><tr><th>interval</th><th>t&nbsp;(fraction)</th></tr></thead><tbody></tbody></table>
    <div id="evtbl" style="margin-top:10px;"></div>
  </div>
  <div class="panel meta" style="max-width:320px;">
    <div id="segmeta"></div>
    <hr style="border:none;border-top:1px solid #eef0f3;margin:10px 0;"/>
    <div><b>How it works.</b> After substituting this segment's N-counts, each cumulative event time
    G<sub>j</sub> (tree-node height) is an <i>affine</i> map
    <code>const + &Sigma; coef&middot;free</code> of the free variables, and each knob's feasible
    range is the intersection of affine half-spaces. The faint grey lines are sampled polytope
    solutions (equally valid). Moving one knob re-clamps the others so you never leave the polytope.</div>
  </div>
</div>

<script>
const PAYLOAD = __DATA__;

// ---- affine helpers ----
function evalAffine(m, free){ let v=m.const; for(let k=0;k<free.length;k++) v+=m.coef[k]*free[k]; return v; }

// 1-D feasible interval for knob `ki` given the other knobs fixed at `free`.
// Each half-space  const + sum coef*free >= 0  becomes a bound on free[ki].
function feasibleInterval(ki, free){
  let lo=-Infinity, hi=Infinity;
  const cons = PAYLOAD.constraints.bounds.concat(PAYLOAD.constraints.interval_nonneg);
  for(const c of cons){
    // value if free[ki] were 0:
    let base=c.const; for(let k=0;k<free.length;k++){ if(k!==ki) base+=c.coef[k]*free[k]; }
    const a=c.coef[ki];                     // coefficient on this knob
    // a*x + base >= 0
    if(Math.abs(a)<1e-12){ continue; }      // no constraint on this knob
    if(a>0){ lo=Math.max(lo, -base/a); }
    else   { hi=Math.min(hi, -base/a); }
  }
  return [lo,hi];
}

function clampAll(free){
  // project each knob into its current feasible interval (one pass each)
  for(let k=0;k<free.length;k++){
    const [lo,hi]=feasibleInterval(k,free);
    if(isFinite(lo)) free[k]=Math.max(free[k],lo);
    if(isFinite(hi)) free[k]=Math.min(free[k],hi);
  }
  return free;
}

function isFeasible(free){
  const cons = PAYLOAD.constraints.bounds.concat(PAYLOAD.constraints.interval_nonneg);
  for(const c of cons){ if(evalAffine(c,free) < -1e-6) return false; }
  return true;
}

// ---- tree geometry ----
const W=300, H=380, PADX=40, PADT=24, PADB=30;
const TOP=PADT, BOT=H-PADB;                 // y for t=0 .. t=1
function yOf(t){ return BOT - t*(BOT-TOP); } // t=0 at bottom (early), t=1 at top (late)

// lay out node x positions: depth-first slots per allele, major on left, minor on right
function layout(){
  const nodes=PAYLOAD.nodes;
  // group by allele preserving payload order (already sort_nodes order)
  const maj=nodes.filter(n=>n.allele==='major');
  const min=nodes.filter(n=>n.allele==='minor');
  const order=[...maj, ...min];
  const xs={};
  order.forEach((n,i)=>{ xs[n.index]= PADX + (i+0.5)*((W-2*PADX)/order.length); });
  return xs;
}
const XS=layout();
const COL="#0b6979";

function nodeHeight(n, free){
  if(n.event_index<0) return 0.0;            // allele root
  let v=evalAffine(PAYLOAD.event_time_maps[n.event_index], free);
  return Math.max(0, Math.min(1, v));
}

function drawTree(free, sampled){
  const ns=PAYLOAD.nodes;
  let svg=`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  // axis
  svg+=`<line x1="${PADX-14}" y1="${TOP}" x2="${PADX-14}" y2="${BOT}" stroke="#c5cbd4"/>`;
  svg+=`<text x="${PADX-18}" y="${TOP+4}" font-size="10" fill="#8a93a3" text-anchor="end">late 1</text>`;
  svg+=`<text x="${PADX-18}" y="${BOT}" font-size="10" fill="#8a93a3" text-anchor="end">early 0</text>`;

  // faint sampled-solution overlay (vertical extents of each node height across samples)
  if(sampled){
    for(const fv of PAYLOAD.sampled_solutions){
      for(const n of ns){
        if(n.parent_index<0) continue;
        const x=XS[n.index], h=nodeHeight(n,fv);
        svg+=`<line x1="${x}" y1="${yOf(h)}" x2="${x}" y2="${yOf(1)}" stroke="${COL}" stroke-opacity="0.07" stroke-width="2"/>`;
      }
    }
  }

  // edges + vertical lines at current free assignment
  for(const n of ns){
    const x=XS[n.index];
    const h=nodeHeight(n,free);
    const dash = n.allele==='minor' ? 'stroke-dasharray="5 4"' : '';
    // vertical line from gain time up to t=1
    svg+=`<line x1="${x}" y1="${yOf(h)}" x2="${x}" y2="${yOf(1)}" stroke="${COL}" stroke-width="2.4" ${dash}/>`;
    // horizontal connector to parent at the gain time
    if(n.parent_index>=0){
      const px=XS[n.parent_index];
      svg+=`<line x1="${x}" y1="${yOf(h)}" x2="${px}" y2="${yOf(h)}" stroke="${COL}" stroke-width="2.4" ${dash}/>`;
    }
  }
  svg+=`</svg>`;
  document.getElementById('tree').innerHTML=svg;
}

// ---- UI ----
const free = PAYLOAD.init_free.slice();
clampAll(free);

function buildKnobs(){
  let html='';
  PAYLOAD.free_vars.forEach((name,k)=>{
    html+=`<div class="knob">
      <label>${name} <span class="rng" id="rng${k}"></span></label>
      <div class="row">
        <input type="range" id="kn${k}" min="0" max="1" step="0.0005"/>
        <span class="val" id="vv${k}"></span>
      </div></div>`;
  });
  if(PAYLOAD.free_vars.length===0){
    html='<div class="meta">This route is <b>fully determined</b> (0 free variables): a single static tree.</div>';
  }
  document.getElementById('knobs').innerHTML=html;
  PAYLOAD.free_vars.forEach((name,k)=>{
    document.getElementById('kn'+k).addEventListener('input',e=>onKnob(k, parseFloat(e.target.value)));
  });
}

function syncSlider(k){
  const [lo,hi]=feasibleInterval(k,free);
  const L=isFinite(lo)?lo:0, Hh=isFinite(hi)?hi:1;
  const sl=document.getElementById('kn'+k);
  sl.min=L; sl.max=Hh; sl.step=Math.max((Hh-L)/1000,1e-6); sl.value=free[k];
  document.getElementById('vv'+k).textContent=free[k].toFixed(4);
  document.getElementById('rng'+k).textContent=`[${L.toFixed(3)}, ${Hh.toFixed(3)}]`;
}

function onKnob(k, val){
  free[k]=val;
  // clamp the OTHER knobs into the new feasible region
  clampAll(free);
  refresh();
}

function refresh(){
  drawTree(free, true);
  PAYLOAD.free_vars.forEach((_,k)=>syncSlider(k));
  // interval table
  let rows='';
  PAYLOAD.t_vars.forEach(tn=>{
    const v=evalAffine(PAYLOAD.interval_maps[tn], free);
    rows+=`<tr><td>${tn}</td><td>${v.toFixed(4)}</td></tr>`;
  });
  document.querySelector('#ttbl tbody').innerHTML=rows;
  // event times
  let ev='<table class="t"><thead><tr><th>event</th><th>allele</th><th>time</th></tr></thead><tbody>';
  PAYLOAD.events.forEach((e,j)=>{
    const t=evalAffine(PAYLOAD.event_time_maps[j], free);
    ev+=`<tr><td>${e.split[0]}+${e.split[1]}&larr;${e.parent_mult}</td><td>${e.allele}</td><td>${t.toFixed(4)}</td></tr>`;
  });
  ev+='</tbody></table>';
  document.getElementById('evtbl').innerHTML=ev;
  // feasibility badge
  const ok=isFeasible(free);
  const badge=document.getElementById('feas');
  badge.textContent=ok?'yes':'no'; badge.className='feas '+(ok?'ok':'bad');
}

function segMeta(){
  const s=PAYLOAD.seg;
  document.getElementById('segmeta').innerHTML=
    `<b>Segment ${s.seg_id}</b><br/>chr${s.chrom}:${s.start.toLocaleString()}-${s.end.toLocaleString()}<br/>`+
    `CN = (${s.major_cn}, ${s.minor_cn})  &middot;  ${s.n_snvs} SNVs<br/>`+
    `route <code>${PAYLOAD.route_key}</code> &middot; ${PAYLOAD.num_free_vars} free variable(s)<br/>`+
    `N_used = [${s.N_used.map(x=>x.toFixed(1)).join(', ')}]  (total_sum=${s.total_sum.toFixed(2)})`;
}

segMeta();
buildKnobs();
refresh();
</script>
</body>
</html>
"""

HTML = HTML.replace("__DATA__", DATA_JSON)
with open(OUT_HTML, "w") as f:
    f.write(HTML)
print("\nwrote", OUT_HTML, os.path.getsize(OUT_HTML), "bytes")
# self-containment check
assert "__DATA__" not in HTML
assert '"interval_maps"' in HTML and '"event_time_maps"' in HTML and '"constraints"' in HTML
assert "cdn" not in HTML.lower() and "http://" not in HTML and "https://" not in HTML
print("HTML self-contained: OK (no external refs; affine + constraint data embedded)")
