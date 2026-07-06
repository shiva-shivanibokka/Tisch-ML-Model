"""FastAPI serving layer for the kidney cell-type classifier."""
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
    return {"samples": [], "genes": [], "classes": []}


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
    try:
        b = load_bundle()
        model_line = (f"{b['model_type']} on {len(b['genes'])} selected gene probes "
                      f"({len(b['classes'])} cell types)")
    except Exception:
        model_line = "model not loaded - run `python train.py`"
    ex = load_examples()
    buttons = "\n".join(
        f"""<button class="sample" data-label="{s['label']}" data-features='{json.dumps(s['features'])}'>
              Kidney cell {i + 1} <span class="truth">(actual: {s['label']})</span></button>"""
        for i, s in enumerate(ex.get("samples", [])))
    return _PAGE.replace("{{MODEL_LINE}}", model_line).replace(
        "{{BUTTONS}}", buttons or "<p><i>No demo samples. Run <code>python train.py</code>.</i></p>")


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kidney Cell-Type Classifier</title>
<style>
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  max-width: 720px; margin: 5vh auto; padding: 0 1.2rem; line-height: 1.6; }
h1 { margin-bottom:.2rem; } .sub { opacity:.75; margin-top:0; }
code { background: rgba(128,128,128,.18); padding:.1rem .35rem; border-radius:4px; }
a { color:#2f81f7; }
.card { border:1px solid rgba(128,128,128,.3); border-radius:10px; padding:1rem 1.2rem; margin:1rem 0; }
button.sample { display:block; width:100%; text-align:left; cursor:pointer; margin:.5rem 0;
  padding:.7rem 1rem; border-radius:8px; font-size:1rem; border:1px solid rgba(128,128,128,.4);
  background: rgba(128,128,128,.08); }
button.sample:hover { background: rgba(47,129,247,.15); }
.truth { opacity:.6; font-size:.85em; }
#result { margin-top:1rem; padding:1.1rem 1.2rem; border-radius:10px; display:none;
  border:1px solid rgba(128,128,128,.4); }
.row { display:flex; align-items:center; gap:.6rem; margin:.3rem 0; }
.row .lab { width:230px; font-size:.9rem; } .bar { flex:1; height:10px; border-radius:5px;
  background: rgba(128,128,128,.25); overflow:hidden; } .bar>div { height:100%; background:#4c9be8; }
.ok { color:#3fb950; } .no { color:#e85c5c; }
</style></head><body>
<h1>&#129516; Kidney Cell-Type Classifier</h1>
<p class="sub">Predicts which of 10 human kidney cell types a single cell is, from its scRNA-seq gene expression.</p>
<div class="card"><b>Model:</b> {{MODEL_LINE}} &nbsp;&middot;&nbsp; <b>Status:</b> healthy</div>
<h3>&#9654; Try it - classify a real cell</h3>
<p class="sub">These are actual held-out cells the model never trained on. Click one to send its gene-expression values to the live model.</p>
{{BUTTONS}}
<div id="result"></div>
<h3>API</h3>
<ul><li><a href="/docs">/docs</a> - interactive API docs</li>
<li><a href="/health">/health</a> &middot; <a href="/model">/model</a> &middot; <code>POST /predict</code></li></ul>
<p class="sub">Source: <a href="https://github.com/shiva-shivanibokka/Tisch-ML-Model">github.com/shiva-shivanibokka/Tisch-ML-Model</a></p>
<script>
document.querySelectorAll('button.sample').forEach(function(btn){
  btn.addEventListener('click', async function(){
    const box=document.getElementById('result'); box.style.display='block'; box.innerHTML='Predicting...';
    try{
      const features=JSON.parse(btn.getAttribute('data-features'));
      const truth=btn.getAttribute('data-label');
      const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({features:features})});
      const d=await r.json(); const correct=d.prediction===truth;
      let rows=d.top3.map(function(t){const pct=(t.prob*100).toFixed(1);
        return '<div class="row"><div class="lab">'+t.label+'</div><div class="bar"><div style="width:'+pct+'%"></div></div><div>'+pct+'%</div></div>';}).join('');
      box.innerHTML='<b>Prediction: '+d.prediction+'</b> <span class="'+(correct?'ok':'no')+'">'+
        (correct?'&#9989; matches actual':'&#10060; actual: '+truth)+'</span>'+rows;
    }catch(e){ box.textContent='Error: '+e; }
  });
});
</script></body></html>"""
