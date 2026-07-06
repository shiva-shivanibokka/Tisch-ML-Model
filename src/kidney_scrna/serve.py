"""FastAPI serving layer for the kidney cell-type classifier.

    GET  /         -> interactive demo landing page (live expression readout)
    GET  /health   -> liveness/readiness probe
    GET  /model    -> metadata: model type, genes, class labels, metrics
    POST /predict  -> {gene: value, ...} -> predicted cell type + top-3 probabilities

Every prediction is logged as a structured JSON line.

Run locally:
    uvicorn kidney_scrna.serve:app --reload
"""
from __future__ import annotations
import json
import logging
import sys
import time
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config

logger = logging.getLogger("kidney_scrna.serve")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


def _log(event: str, **f: Any) -> None:
    logger.info(json.dumps({"event": event, **f}))


_BUNDLE: dict | None = None


def load_bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No model at {config.MODEL_PATH}. Run `python train.py` first.")
        _BUNDLE = joblib.load(config.MODEL_PATH)
        _log("model_loaded", model_type=_BUNDLE["model_type"], n_genes=len(_BUNDLE["genes"]))
    return _BUNDLE


def load_examples() -> dict:
    if config.EXAMPLES_PATH.exists():
        return json.loads(config.EXAMPLES_PATH.read_text())
    return {"samples": [], "genes": [], "classes": [], "stats": {},
            "per_class": {}, "metrics": {}, "model_type": ""}


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="gene -> expression value; all genes from /model required")


class TopClass(BaseModel):
    label: str
    prob: float


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    prediction: str
    confidence: float
    top3: list[TopClass]
    model_type: str


app = FastAPI(
    title="Kidney Cell-Type Classifier",
    description="Classifies human kidney cells into 10 types from scRNA-seq expression.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_available": config.MODEL_PATH.exists()}


@app.get("/model")
def model_info() -> dict:
    b = load_bundle()
    md = {}
    if config.METRICS_PATH.exists():
        md = json.loads(config.METRICS_PATH.read_text())
    return {"model_type": b["model_type"], "classes": b["classes"],
            "n_genes": len(b["genes"]), "genes": b["genes"], "metrics": md}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    b = load_bundle()
    genes = b["genes"]
    model = b["model"]
    missing = [g for g in genes if g not in req.features]
    if missing:
        raise HTTPException(422, f"Missing {len(missing)} gene(s), e.g. {missing[:5]}")
    row = [[float(req.features[g]) for g in genes]]
    t0 = time.time()
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    order = sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)
    top3 = [TopClass(label=classes[i], prob=round(float(proba[i]), 4)) for i in order[:3]]
    pred = top3[0]
    _log("prediction", prediction=pred.label, confidence=pred.prob,
         latency_ms=round((time.time() - t0) * 1000, 2))
    return PredictResponse(prediction=pred.label, confidence=pred.prob,
                           top3=top3, model_type=b["model_type"])


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Interactive demo landing page."""
    ex = load_examples()
    if not ex.get("model_type"):
        try:
            ex["model_type"] = load_bundle()["model_type"]
        except Exception:
            ex["model_type"] = ""
    return _LANDING_PAGE.replace("__DATA__", json.dumps(ex))


# --- Landing page ------------------------------------------------------------
# Self-contained page; `__DATA__` is replaced at request time with the demo JSON
# (genes, per-gene training stats, held-out samples, per-class F1, headline metrics).
_LANDING_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kidney Cell-Type Classifier - live demo</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='88'>&#128300;</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0A0E14; --panel:#111823; --panel2:#0D141D; --line:#1E2A38;
    --ink:#E7EEF6; --muted:#7C8CA0; --faint:#4C5B6E;
    --accent:#4FD1C5; --accent2:#63B3ED; --hi:#F6708A; --lo:#37C6E8;
    --good:#43D6A0; --bad:#F6708A;
    --sans:'IBM Plex Sans',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:
      radial-gradient(1200px 520px at 50% -220px,#12242200,#0A0E14 72%),var(--bg);
    color:var(--ink);font-family:var(--sans);line-height:1.55;-webkit-font-smoothing:antialiased;}
  .wrap{max-width:820px;margin:0 auto;padding:clamp(2rem,6vw,4.5rem) 1.25rem 4rem;}
  .eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.28em;
    text-transform:uppercase;color:var(--accent);margin:0 0 1rem;}
  h1{font-size:clamp(2rem,5.4vw,3.1rem);font-weight:600;line-height:1.05;
    letter-spacing:-.025em;margin:0 0 .9rem;}
  h1 .em{color:var(--accent);}
  .lede{color:var(--muted);font-size:1.06rem;max-width:58ch;margin:0 0 1.9rem;}
  .head{display:flex;align-items:center;gap:.5rem;margin:0 0 .55rem;}
  .k{font-family:var(--mono);font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;
    color:var(--muted);margin:0;}
  .q{width:18px;height:18px;border-radius:50%;border:1px solid var(--line);background:transparent;
    color:var(--muted);font-family:var(--mono);font-size:.72rem;line-height:16px;text-align:center;
    cursor:pointer;padding:0;flex:0 0 auto;}
  .q:hover{color:var(--ink);border-color:var(--faint);}
  .explain{display:none;margin:0 0 1rem;color:var(--muted);font-size:.86rem;
    border-left:2px solid var(--line);padding-left:.75rem;line-height:1.5;}
  .explain.open{display:block;}
  .stats{display:flex;flex-wrap:wrap;border:1px solid var(--line);border-radius:12px;
    overflow:hidden;margin-bottom:2.4rem;}
  .stat{flex:1 1 22%;min-width:120px;padding:.85rem 1.1rem;border-right:1px solid var(--line);}
  .stat:last-child{border-right:0;}
  .stat b{font-family:var(--mono);font-size:1.35rem;font-weight:500;display:block;letter-spacing:-.02em;}
  .stat .k{margin:.15rem 0 0;letter-spacing:.1em;font-size:.64rem;}
  .panel{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);
    border-radius:16px;padding:1.5rem;margin-bottom:1.5rem;}
  .chips{display:grid;grid-template-columns:1fr 1fr;gap:.55rem;}
  .chip{cursor:pointer;text-align:left;padding:.72rem .9rem;border-radius:10px;border:1px solid var(--line);
    background:#0e1620;color:var(--ink);font-family:var(--sans);font-size:.92rem;
    transition:border-color .15s,background .15s,transform .06s;}
  .chip:hover{border-color:var(--faint);background:#101c28;}
  .chip:active{transform:translateY(1px);}
  .chip.on{border-color:var(--accent);}
  .chip .dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:.5rem;
    vertical-align:middle;background:var(--accent);}
  .chip .lab{font-family:var(--mono);font-size:.68rem;color:var(--muted);display:block;
    margin-top:.15rem;letter-spacing:.03em;}
  .rand{margin-top:.6rem;width:100%;cursor:pointer;padding:.72rem;border-radius:10px;
    border:1px dashed var(--line);background:transparent;color:var(--muted);font-family:var(--mono);
    font-size:.82rem;letter-spacing:.03em;transition:.15s;}
  .rand:hover{color:var(--ink);border-color:var(--faint);background:#0e1620;}
  .readout{margin-top:1.4rem;opacity:0;max-height:0;overflow:hidden;transition:opacity .4s ease;}
  .readout.show{opacity:1;max-height:none;}
  .rhead{font-family:var(--mono);font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;
    color:var(--muted);display:flex;align-items:center;gap:.5rem;margin:.2rem 0 .6rem;}
  .heat{display:flex;gap:1px;height:46px;border-radius:4px;overflow:hidden;}
  .cell{flex:1 1 0;background:#16202c;opacity:0;transition:opacity .5s ease,background .5s ease;}
  .readout.show .cell{opacity:1;}
  .heatlabels{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.64rem;
    color:var(--muted);margin-top:.5rem;letter-spacing:.04em;}
  .swatch{display:inline-block;width:9px;height:9px;border-radius:2px;vertical-align:middle;margin:0 .25rem;}
  .verdict{display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;margin:1.4rem 0 .2rem;}
  .verdict .big{font-size:1.5rem;font-weight:600;letter-spacing:-.02em;color:var(--accent);}
  .verdict .sci{font-family:var(--mono);font-size:.8rem;color:var(--muted);}
  .match{font-family:var(--mono);font-size:.72rem;padding:.15rem .55rem;border-radius:20px;
    border:1px solid var(--line);}
  .match.ok{color:var(--good);border-color:rgba(67,214,160,.4);}
  .match.no{color:var(--bad);border-color:rgba(246,112,138,.4);}
  .bars{display:grid;gap:.5rem;margin-top:.4rem;}
  .brow{display:grid;grid-template-columns:11.5rem 1fr 3rem;gap:.7rem;align-items:center;
    font-family:var(--mono);font-size:.74rem;}
  .brow .gid{color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .brow.top .gid{color:var(--ink);}
  .btrack{position:relative;height:15px;background:#0e1620;border-radius:4px;border:1px solid var(--line);}
  .bfill{position:absolute;top:0;left:0;height:100%;border-radius:3px;background:var(--faint);
    transition:width .6s cubic-bezier(.2,.8,.2,1);}
  .brow.top .bfill{background:linear-gradient(90deg,var(--accent),var(--accent2));}
  .brow .pct{color:var(--muted);text-align:right;}
  .brow.top .pct{color:var(--ink);}
  .steps{display:grid;margin:.4rem 0 0;}
  .step{display:flex;gap:.9rem;padding:.78rem 0;border-top:1px solid var(--line);}
  .step .n{font-family:var(--mono);font-size:.75rem;color:var(--accent);min-width:2rem;
    padding-top:.15rem;letter-spacing:.05em;}
  .step p{margin:0;color:var(--muted);font-size:.95rem;}
  .step b{color:var(--ink);font-weight:600;}
  footer{border-top:1px solid var(--line);margin-top:2.2rem;padding-top:1.3rem;font-family:var(--mono);
    font-size:.8rem;color:var(--muted);display:flex;flex-wrap:wrap;gap:.4rem 1.1rem;align-items:center;}
  footer a{color:var(--muted);text-decoration:none;border-bottom:1px solid var(--line);}
  footer a:hover{color:var(--ink);}
  .hint{font-size:.8rem;color:var(--faint);margin:1rem 0 0;font-family:var(--mono);}
  @media(max-width:520px){.chips{grid-template-columns:1fr}.brow{grid-template-columns:8.5rem 1fr 2.6rem}}
  @media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head>
<body>
<div class="wrap">
  <p class="eyebrow">Single-cell classifier &middot; live demo</p>
  <h1>Name a kidney cell<br>from the genes it <span class="em">switches on</span>.</h1>
  <p class="lede">A model trained on 60,725 single cells &mdash; pooled from five human-kidney
     studies &mdash; names which of 10 cell types a cell is, from just 293 genes. Pick a real
     cell it never saw during training and watch it read the expression.</p>

  <div class="stats">
    <div class="stat"><b id="s-f1">&mdash;</b>
      <div class="head"><span class="k">Test F1</span><button class="q">?</button></div>
      <p class="explain">Weighted F1 on the held-out test set &mdash; balances precision and
         recall across all 10 cell types, so the majority class can't hide weak ones.</p></div>
    <div class="stat"><b id="s-auc">&mdash;</b>
      <div class="head"><span class="k">ROC-AUC</span><button class="q">?</button></div>
      <p class="explain">How well the model ranks the right cell type, averaged one-vs-rest.
         1.0 is flawless; 0.5 is a coin flip.</p></div>
    <div class="stat"><b id="s-genes">&mdash;</b>
      <div class="head"><span class="k">Genes used</span><button class="q">?</button></div>
      <p class="explain">The model reads only this many genes &mdash; selected from 2,358 &mdash;
         so each prediction is compact.</p></div>
    <div class="stat"><b id="s-types">&mdash;</b>
      <div class="head"><span class="k">Cell types</span><button class="q">?</button></div>
      <p class="explain">The 10 most abundant kidney cell types in the dataset, from tubular
         and vascular cells to immune populations.</p></div>
  </div>

  <div class="panel">
    <div class="head"><p class="k">Pick a cell</p><button class="q">?</button></div>
    <p class="explain">Real held-out cells the model never trained on &mdash; one per cell type.
       Click a labelled one, or draw a random cell &mdash; the model reads its 293 genes live and
       the result appears right below.</p>
    <div class="chips" id="chips"></div>
    <button class="rand" id="rand">&#9862; Draw a random held-out cell</button>

    <div class="readout" id="readout">
      <div class="rhead"><span>Expression signature</span><button class="q">?</button></div>
      <p class="explain">Each stripe is one of the 293 genes' expression in this cell,
         standardised against the training average. Cyan = under-expressed, red = over-expressed.</p>
      <div class="heat" id="heat"></div>
      <div class="heatlabels">
        <span><span class="swatch" style="background:var(--lo)"></span>under-expressed</span>
        <span>293-gene signature</span>
        <span>over-expressed<span class="swatch" style="background:var(--hi)"></span></span>
      </div>

      <div class="verdict" id="verdict"></div>
      <div class="rhead" style="margin-top:1rem"><span>Model confidence &middot; top 3</span><button class="q">?</button></div>
      <p class="explain">The three cell types the model considers most likely for this cell, with
         the probability it assigns to each.</p>
      <div class="bars" id="top3"></div>
      <p class="hint" id="hint"></p>
    </div>
  </div>

  <div class="panel">
    <div class="head"><p class="k">Per-class performance</p><button class="q">?</button></div>
    <p class="explain">Test-set F1 for each cell type. Distinct types (T cells, endothelium)
       score high; rare, closely-related tubule subtypes are hardest &mdash; as expected.</p>
    <div class="bars" id="perclass"></div>
  </div>

  <div class="head" style="margin-top:2rem"><p class="k">How it works</p><button class="q">?</button></div>
  <p class="explain">The pipeline behind every prediction, built to avoid the data leakage that
     inflates many gene-expression classifiers.</p>
  <div class="steps">
    <div class="step"><span class="n">01</span><p><b>Reduce</b> 2,358 genes to 293 with variance
       filters and recursive feature elimination &mdash; fit on training data only.</p></div>
    <div class="step"><span class="n">02</span><p><b>Balance &amp; tune</b> with SMOTE + undersampling
       applied <b>inside</b> each cross-validation fold &mdash; no resampling leakage.</p></div>
    <div class="step"><span class="n">03</span><p><b>Classify</b> with a tuned RBF support-vector
       machine, served here on Fly.io as a live API.</p></div>
  </div>

  <footer>
    <span id="f-model">model</span>
    <a href="/docs">API docs</a>
    <a href="/model">/model</a>
    <a href="/health">/health</a>
    <a href="https://github.com/shiva-shivanibokka/Tisch-ML-Model">GitHub</a>
  </footer>
</div>

<script>
const DATA = __DATA__;
const C_LO=[55,198,232], C_NEUT=[22,32,44], C_HI=[246,112,138];
function mix(a,b,t){return 'rgb('+a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')+')';}
function divColor(z){const L=1.6,c=Math.max(-L,Math.min(L,z)),t=(c+L)/(2*L);
  return t<0.5?mix(C_LO,C_NEUT,t*2):mix(C_NEUT,C_HI,(t-0.5)*2);}

const samples=DATA.samples||[], genes=DATA.genes||[], stats=DATA.stats||{};
const met=DATA.metrics||{};

// header stats
const g=id=>document.getElementById(id);
g('s-f1').textContent=met.weighted_f1!=null?met.weighted_f1.toFixed(3):'--';
g('s-auc').textContent=met.roc_auc!=null?met.roc_auc.toFixed(3):'--';
g('s-genes').textContent=genes.length||'--';
g('s-types').textContent=(DATA.classes||[]).length||'--';
g('f-model').textContent=(DATA.model_type||'model')+' - '+(DATA.n_test||'?')+' held-out cells';

// "?" explainers
document.querySelectorAll('.q').forEach(q=>q.addEventListener('click',()=>{
  const ex=q.closest('.head,.rhead').nextElementSibling;
  if(ex&&ex.classList.contains('explain'))ex.classList.toggle('open');
}));

// chips: first held-out cell of each class
const chips=g('chips'), seen={};
(DATA.classes||[]).forEach(cls=>{
  const s=samples.find(x=>x.label===cls); if(!s)return;
  const b=document.createElement('button'); b.className='chip';
  b.innerHTML='<span class="dot"></span>'+cls+'<span class="lab">held-out cell</span>';
  b.addEventListener('click',()=>predict(s,b)); chips.appendChild(b);
});
g('rand').addEventListener('click',()=>{ if(samples.length)predict(samples[Math.floor(Math.random()*samples.length)],null); });
if(!samples.length){chips.innerHTML='<p class="hint">No demo samples baked in. Run <code>python train.py</code>.</p>';}

// per-class F1 panel
const pc=DATA.per_class||{};
const rows=Object.keys(pc).map(k=>({name:k,f1:pc[k].f1,n:pc[k].support})).sort((a,b)=>b.f1-a.f1);
g('perclass').innerHTML=rows.map(r=>
  '<div class="brow"><span class="gid" title="'+r.name+' (n='+r.n+')">'+r.name+'</span>'+
  '<div class="btrack"><div class="bfill" style="width:'+(r.f1*100).toFixed(0)+'%;'+
  'background:linear-gradient(90deg,var(--accent),var(--accent2))"></div></div>'+
  '<span class="pct">'+r.f1.toFixed(2)+'</span></div>').join('');

function buildHeat(features){
  const heat=g('heat'); heat.innerHTML='';
  genes.forEach((gn,i)=>{
    const st=stats[gn]||{mean:0,std:1};
    const z=(features[gn]-st.mean)/(st.std||1);
    const c=document.createElement('div'); c.className='cell';
    c.title=gn+'  z '+z.toFixed(2);
    requestAnimationFrame(()=>{c.style.background=divColor(z);});
    heat.appendChild(c);
  });
}

let busy=false;
async function predict(sample,btn){
  if(busy)return; busy=true;
  document.querySelectorAll('.chip').forEach(c=>c.classList.remove('on'));
  if(btn)btn.classList.add('on');
  const ro=g('readout'); ro.classList.add('show');
  g('hint').textContent='reading expression...';
  buildHeat(sample.features);
  ro.scrollIntoView({behavior:'smooth',block:'nearest'});
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({features:sample.features})});
    const d=await r.json();
    const ok=d.prediction===sample.label;
    g('verdict').innerHTML='<span class="big">'+d.prediction+'</span>'+
      '<span class="match '+(ok?'ok':'no')+'">'+(ok?'✓ matches actual':'✗ actual: '+sample.label)+'</span>';
    g('top3').innerHTML=d.top3.map((t,i)=>{
      const pct=(t.prob*100).toFixed(1);
      return '<div class="brow'+(i===0?' top':'')+'"><span class="gid" title="'+t.label+'">'+t.label+'</span>'+
        '<div class="btrack"><div class="bfill" style="width:'+pct+'%"></div></div>'+
        '<span class="pct">'+pct+'%</span></div>';}).join('');
    g('hint').textContent='293 gene values -> live model -> prediction ('+(btn?'labelled cell':'random held-out cell')+')';
  }catch(e){ g('hint').textContent='Error: '+e; }
  busy=false;
}
</script>
</body></html>"""
