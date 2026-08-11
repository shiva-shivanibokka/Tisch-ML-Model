"""Vercel serverless entrypoint.

Vercel's Python runtime looks for an ASGI application named `app`, so this file
does nothing but put src/ on the path and hand over the same FastAPI app the
Dockerfile serves. The deployment differs from the container in one respect: it
installs onnxruntime instead of scikit-learn (see api/requirements.txt), which
makes serve.py load artifacts/model.onnx. Same pipeline, same weights.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kidney_scrna.serve import app  # noqa: E402

__all__ = ["app"]
