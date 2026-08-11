"""Export the trained sklearn pipeline to ONNX, then prove the export is faithful.

Why this exists: the serving image runs on a serverless platform with a 250 MB
unpacked budget. scikit-learn drags in scipy, and the three of them together
measure 278 MB of Linux wheels before the model is even added — over budget with
no way to trim. onnxruntime + numpy is 145 MB and runs the identical model.

    python export_onnx.py

Writes artifacts/model.onnx (the graph) and artifacts/model_onnx.json (the gene
order, class labels and model name that serve.py needs and the graph does not
carry). Exits non-zero if the ONNX output disagrees with scikit-learn on the
held-out demo cells — a demo that silently predicts differently from the model
in the benchmarks is worse than no demo.
"""
from __future__ import annotations
import json
import sys

import joblib
import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

from kidney_scrna import config

# The label is decided by argmax over these, so a disagreement here is a
# disagreement in the prediction. 1e-4 is far tighter than the 4 decimal places
# the API actually returns, and comfortably above float32 round-off.
PROB_TOL = 1e-4


def main() -> int:
    bundle = joblib.load(config.MODEL_PATH)
    pipe, genes = bundle["model"], bundle["genes"]
    n = len(genes)

    # zipmap=False keeps the probability output a plain (1, n_classes) tensor
    # instead of a list-of-dicts, which onnxruntime returns as a Python object
    # and which serve.py would only have to unpack again.
    onx = to_onnx(
        pipe,
        initial_types=[("input", FloatTensorType([None, n]))],
        options={id(pipe.steps[-1][1]): {"zipmap": False}},
        target_opset=15,
    )
    config.MODEL_ONNX_PATH.write_bytes(onx.SerializeToString())

    meta = {
        "genes": genes,
        "classes": [str(c) for c in pipe.classes_],
        "model_type": bundle["model_type"],
    }
    config.MODEL_ONNX_META_PATH.write_text(json.dumps(meta))

    ok = verify(pipe, genes)
    size = config.MODEL_ONNX_PATH.stat().st_size / 1e6
    joblib_size = config.MODEL_PATH.stat().st_size / 1e6
    print(f"model.onnx {size:.1f} MB (from {joblib_size:.1f} MB joblib), {n} genes, "
          f"{len(meta['classes'])} classes")
    return 0 if ok else 1


def verify(pipe, genes: list[str]) -> bool:
    """Compare ONNX against scikit-learn on the demo cells, then on random input.

    The demo cells are what a visitor actually clicks, so they are the cases that
    must match. The random rows cover the rest of the input space the /predict
    endpoint accepts — a converter that is right on ten real cells and wrong in
    general would still be wrong in front of anyone who used the API.
    """
    import onnxruntime as ort

    rows: list[list[float]] = []
    if config.EXAMPLES_PATH.exists():
        ex = json.loads(config.EXAMPLES_PATH.read_text())
        for s in ex.get("samples", []):
            v = s.get("values") or s.get("features")
            if isinstance(v, dict):
                rows.append([float(v[g]) for g in genes])
            elif isinstance(v, list) and len(v) == len(genes):
                rows.append([float(x) for x in v])
    n_demo = len(rows)

    rng = np.random.default_rng(0)
    rows.extend(rng.normal(0, 2, size=(200, len(genes))).tolist())
    X = np.asarray(rows, dtype=np.float64)

    ref = pipe.predict_proba(X)
    sess = ort.InferenceSession(str(config.MODEL_ONNX_PATH),
                                providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": X.astype(np.float32)})
    got = np.asarray(out[1])

    delta = float(np.abs(ref - got).max())
    label_mismatch = int((ref.argmax(1) != got.argmax(1)).sum())
    print(f"verified {n_demo} demo cells + {len(rows) - n_demo} random rows: "
          f"max prob delta {delta:.2e}, label mismatches {label_mismatch}")
    if label_mismatch or delta > PROB_TOL:
        print(f"FAIL: ONNX disagrees with scikit-learn (tolerance {PROB_TOL})")
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
