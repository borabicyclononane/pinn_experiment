"""Build the interactive coordinate-check dashboard (self-contained HTML, Plotly from cdnjs).
Usage: python scripts/build_dashboard.py [group] [out.html]"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mupinn.analysis import aggregate_coord

groups = (sys.argv[1] if len(sys.argv) > 1 else "coordcheck").split(",")   # e.g. coordcheck,coordcheck_lowlr:lowlr,coord_depth1:depth1
out = sys.argv[2] if len(sys.argv) > 2 else "figs/dashboard.html"
agg = {}
for g in groups:
    name, _, suffix = g.partition(":")
    for (task, param, opt), d in aggregate_coord(name).items():
        agg[(task + ("@" + suffix if suffix else ""), param, opt)] = d
data = {}
for (task, param, opt), d in agg.items():
    key = f"{task}|{param}|{opt}"
    data[key] = {}
    for nq, dw in d.items():
        data[key][nq] = {str(w): {str(t): [round(m, 6) if m == m else None, round(s, 6)] for t, (m, s, n) in dt.items()} for w, dt in dw.items()}
tasks = sorted({k.split("|")[0] for k in data})
opts = sorted({k.split("|")[2] for k in data})
payload = json.dumps({"data": data, "tasks": tasks, "opts": opts}, separators=(",", ":"))

HTML = r"""<title>PINN μP Coordinate Checks</title>
<meta name="description" content="Interactive coordinate-check dashboard: per-node RMS statistics vs width for regression and PINN losses under SP and μP.">
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<style>
:root{color-scheme:light;--bg:#fcfcfb;--panel:#ffffff;--text:#0b0b0b;--text2:#52514e;--grid:#e6e5e1;--border:#dcdbd6;
 --sp:#eb6834;--mup:#2a78d6;--ntk:#1baf7a;--pos:#e34948;--neg:#2a78d6;--mid:#f0efec}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;--panel:#222221;--text:#ffffff;--text2:#c3c2b7;--grid:#333331;--border:#3a3a38;
 --sp:#d95926;--mup:#3987e5;--ntk:#199e70;--pos:#e66767;--neg:#3987e5;--mid:#383835}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--panel:#222221;--text:#ffffff;--text2:#c3c2b7;--grid:#333331;--border:#3a3a38;
 --sp:#d95926;--mup:#3987e5;--ntk:#199e70;--pos:#e66767;--neg:#3987e5;--mid:#383835}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;padding-inline:16px;padding-block:16px;min-width:0}
h1{font-size:20px;margin:0 0 4px} .sub{color:var(--text2);margin:0 0 12px;font-size:13px}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:12px;padding:10px 12px;border:1px solid var(--border);border-radius:8px;background:var(--panel)}
.controls label{display:flex;gap:6px;align-items:center;color:var(--text2);font-size:13px}
select,input[type=range]{font:inherit;color:var(--text);background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:3px 6px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:6px}
.card .t{font-size:13px;margin:2px 6px;color:var(--text)} .card .e{font-size:12px;color:var(--text2);margin:0 6px 4px}
.legend{display:flex;gap:16px;margin:8px 0 12px;font-size:13px;color:var(--text2)} .sw{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px;border-radius:2px}
.tabs{display:flex;gap:6px;margin-bottom:10px} .tabs button{font:inherit;padding:5px 12px;border:1px solid var(--border);border-radius:6px;background:var(--panel);color:var(--text);cursor:pointer}
.tabs button.on{background:var(--mup);color:#fff;border-color:var(--mup)}
#heat .card{min-height:320px}
.note{font-size:12px;color:var(--text2);margin-top:12px}
</style>
<h1>Coordinate checks: regression vs physics-informed losses, SP vs μP</h1>
<p class="sub">RMS of each node of the computation graph against width n, after t full-batch steps (mean ± s.e.m. over 3 seeds; MLP, 3 hidden layers, tanh). A flat line (exponent 0) is what μP predicts for every node; fitted exponents use n ≥ 256.</p>
<div class="tabs"><button id="tab-width" class="on">RMS vs width</button><button id="tab-exp">Exponent heatmap (node × step)</button><button id="tab-time">RMS vs step (per width)</button></div>
<div class="controls">
 <label>Task <select id="task"></select></label>
 <label>Optimizer <select id="opt"></select></label>
 <label>Quantity <select id="q"><option value="value">value</option><option value="delta" selected>delta (change since init)</option><option value="grad">grad of loss</option><option value="gradf">grad of mean f</option><option value="update">applied update (weights)</option></select></label>
 <label id="steplab">Step t <input id="step" type="range" min="0" max="8" value="1" style="width:120px"> <span id="stepv">1</span></label>
 <label>Nodes <select id="nodes"><option value="act">activations &amp; output</option><option value="deriv">derivative nodes (∂ₓh, f_x, …)</option><option value="w">weights</option><option value="all">all</option></select></label>
</div>
<div class="legend"><span><i class="sw" style="background:var(--sp)"></i>SP</span><span><i class="sw" style="background:var(--mup)"></i>μP</span><span id="leg2"></span></div>
<div id="grid" class="grid"></div>
<p class="note">Reference slopes drawn in grey: n⁰ (flat), n^{±1/2}, n^{±1}. Quantities: value = RMS of the node; delta = RMS of (node − node at init); grad = RMS of ∂loss/∂node; gradf = RMS of ∂(mean f)/∂node; update = RMS of the parameter update applied at this step. Learning rates: SGD 2⁻⁵, Adam 2⁻⁸ at every width (μP multipliers relative to base width 64); tasks tagged @lowlr use SGD 2⁻⁷ and Adam 2⁻¹⁰; tasks tagged @depth1 are one-hidden-layer networks (the mean-field setting) with widths up to 16384.</p>
<script>
const P = __PAYLOAD__;
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const ORDER = ["x","h1","h2","h3","f","dx_h1","dx_h2","dx_h3","f_x","dxx_h1","dxx_h2","dxx_h3","f_xx","f_xxx","f_xxxx","f_t","W1","W2","W3","Wout"];
const GROUPS = {act:["h1","h2","h3","f"], deriv:["dx_h1","dx_h2","dx_h3","f_x","dxx_h2","dxx_h3","f_xx","f_xxx","f_xxxx","f_t"], w:["W1","W2","W3","Wout"], all:ORDER};
const $ = id => document.getElementById(id);
let view = "width";
for (const t of P.tasks) $("task").add(new Option(t, t));
for (const o of P.opts) $("opt").add(new Option(o, o));
$("task").value = P.tasks.includes("ode2") ? "ode2" : P.tasks[0];
function series(key, nq) { const d = P.data[key]; return d && d[nq] ? d[nq] : null; }
function fitExp(ws, vals) { const pts = ws.map((w,i)=>[w,vals[i]]).filter(p=>p[0]>=256 && p[1]>0 && isFinite(p[1])); if (pts.length<2) return NaN;
  const x=pts.map(p=>Math.log2(p[0])), y=pts.map(p=>Math.log2(p[1])); const mx=x.reduce((a,b)=>a+b)/x.length, my=y.reduce((a,b)=>a+b)/y.length;
  let num=0, den=0; for (let i=0;i<x.length;i++){num+=(x[i]-mx)*(y[i]-my); den+=(x[i]-mx)**2;} return num/den; }
function layoutBase(title) { return {title:{text:title,font:{size:13,color:css('--text')}}, margin:{l:48,r:10,t:34,b:38}, height:260, paper_bgcolor:css('--panel'), plot_bgcolor:css('--panel'),
  font:{color:css('--text2'),size:11}, showlegend:false, xaxis:{gridcolor:css('--grid'),zeroline:false}, yaxis:{gridcolor:css('--grid'),zeroline:false}}; }
function nodesFor(sel) { const task=$("task").value, opt=$("opt").value, q=$("q").value; const d = P.data[`${task}|mup|${opt}`] || P.data[`${task}|sp|${opt}`] || {};
  return GROUPS[sel].filter(n => d[`${n}/${q}`]); }
function renderWidth() {
  const task=$("task").value, opt=$("opt").value, q=$("q").value, t=$("step").value; const grid=$("grid"); grid.innerHTML="";
  for (const node of nodesFor($("nodes").value)) {
    const card=document.createElement("div"); card.className="card"; const div=document.createElement("div"); card.appendChild(div); grid.appendChild(card);
    const traces=[]; let ttl=`${node} / ${q}`; const exps=[];
    for (const param of ["sp","mup"]) { const s=series(`${task}|${param}|${opt}`, `${node}/${q}`); if (!s) continue;
      const ws=Object.keys(s).map(Number).sort((a,b)=>a-b); const vals=ws.map(w=> (s[w][t]||[null])[0]); const errs=ws.map(w=>(s[w][t]||[0,0])[1]);
      const e=fitExp(ws, vals); exps.push(`${param==='sp'?'SP':'μP'}: n^${isFinite(e)?e.toFixed(2):'?'}`);
      traces.push({x:ws,y:vals,error_y:{type:'data',array:errs,visible:true,thickness:1},mode:'lines+markers',name:param==='sp'?'SP':'μP',line:{color:css(param==='sp'?'--sp':'--mup'),width:2},marker:{size:6},
        hovertemplate:`${param==='sp'?'SP':'μP'} n=%{x}: %{y:.3g}<extra></extra>`}); }
    // reference slopes anchored at the mup value at n=256 (or first)
    const ref=traces.find(tr=>tr.name==='μP')||traces[0]; if (ref) { const ws=ref.x; const i0=ws.indexOf(256)>=0?ws.indexOf(256):0; const y0=ref.y[i0];
      if (y0>0) for (const sl of [-1,-0.5,0,0.5,1]) traces.unshift({x:ws,y:ws.map(w=>y0*Math.pow(w/ws[i0],sl)),mode:'lines',line:{color:css('--grid'),width:1,dash:'dot'},hoverinfo:'skip'}); }
    const lay=layoutBase(ttl+'   ('+exps.join(', ')+')'); lay.xaxis.type='log'; lay.yaxis.type='log'; lay.xaxis.title={text:'width n',standoff:4}; lay.xaxis.tickvals=[64,256,1024,4096,16384]; lay.yaxis.exponentformat='power';
    Plotly.newPlot(div, traces, lay, {displayModeBar:false,responsive:true}); }
}
function renderExp() {
  const task=$("task").value, opt=$("opt").value, q=$("q").value; const grid=$("grid"); grid.innerHTML="";
  const nodes = nodesFor("all"); const steps=[0,1,2,3,4,5,6,7,8];
  for (const param of ["sp","mup"]) { const card=document.createElement("div"); card.className="card"; card.style.gridColumn="span 2"; const div=document.createElement("div"); card.appendChild(div); grid.appendChild(card);
    const z=nodes.map(node=>steps.map(t=>{ const s=series(`${task}|${param}|${opt}`, `${node}/${q}`); if(!s) return null; const ws=Object.keys(s).map(Number).sort((a,b)=>a-b); const e=fitExp(ws, ws.map(w=>(s[w][t]||[null])[0])); return isFinite(e)?e:null;}));
    const lay=layoutBase(`${param==='sp'?'SP':'μP'}: fitted width exponent of ${q} (0 = Θ(1))`); lay.height=Math.max(300, 18*nodes.length+90); lay.margin.l=70; lay.xaxis.title={text:'step t',standoff:4};
    Plotly.newPlot(div,[{z:z,x:steps,y:nodes,type:'heatmap',zmin:-1.5,zmax:1.5,colorscale:[[0,css('--neg')],[0.5,css('--mid')],[1,css('--pos')]],colorbar:{title:'exp',thickness:10,len:0.9},
      text:z.map(r=>r.map(v=>v==null?'':v.toFixed(2))),texttemplate:'%{text}',textfont:{size:10},hovertemplate:'%{y} t=%{x}: n^%{z:.2f}<extra></extra>',xgap:1,ygap:1}], lay, {displayModeBar:false,responsive:true}); }
}
function renderTime() {
  const task=$("task").value, opt=$("opt").value, q=$("q").value; const grid=$("grid"); grid.innerHTML="";
  const ramp=['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#1c5cab','#104281','#0d366b'];
  for (const node of nodesFor($("nodes").value)) for (const param of ["sp","mup"]) { const s=series(`${task}|${param}|${opt}`, `${node}/${q}`); if(!s) continue;
    const card=document.createElement("div"); card.className="card"; const div=document.createElement("div"); card.appendChild(div); grid.appendChild(card);
    const ws=Object.keys(s).map(Number).sort((a,b)=>a-b); const traces=ws.map((w,i)=>{ const ts=Object.keys(s[w]).map(Number).sort((a,b)=>a-b); return {x:ts,y:ts.map(t=>s[w][t][0]),mode:'lines+markers',name:'n='+w,line:{color:ramp[Math.round(i*(ramp.length-1)/Math.max(1,ws.length-1))],width:2},marker:{size:5},hovertemplate:`n=${w} t=%{x}: %{y:.3g}<extra></extra>`};});
    const lay=layoutBase(`${param==='sp'?'SP':'μP'} ${node} / ${q} (light→dark = narrow→wide)`); lay.yaxis.type='log'; lay.xaxis.title={text:'step t',standoff:4}; lay.showlegend=true; lay.legend={font:{size:9},x:1,xanchor:'right',y:0,yanchor:'bottom',bgcolor:'rgba(0,0,0,0)'};
    Plotly.newPlot(div,traces,lay,{displayModeBar:false,responsive:true}); }
}
function render(){ $("steplab").style.display = view==='width'?'flex':'none'; ({width:renderWidth, exp:renderExp, time:renderTime})[view](); }
for (const id of ["task","opt","q","nodes"]) $(id).addEventListener("change", render);
$("step").addEventListener("input", ()=>{ $("stepv").textContent=$("step").value; render(); });
for (const [id,v] of [["tab-width","width"],["tab-exp","exp"],["tab-time","time"]]) $(id).addEventListener("click", ()=>{ view=v; document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('on')); $(id).classList.add('on'); render(); });
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
render();
</script>
"""
open(out, "w").write(HTML.replace("__PAYLOAD__", payload))
print("wrote", out, f"{os.path.getsize(out)/1e6:.2f} MB", "keys:", len(data))
