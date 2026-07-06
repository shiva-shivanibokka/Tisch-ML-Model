FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install the package (serving deps only — no notebook/plot extras).
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir \
        "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2.6" "joblib>=1.3" \
        "scikit-learn>=1.3" "imbalanced-learn>=0.11" "numpy>=1.24" "scipy>=1.10" \
    && pip install --no-cache-dir -e . --no-deps

# Bake the trained model + demo samples into the image (small; no dataset).
COPY artifacts ./artifacts

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
CMD ["uvicorn", "kidney_scrna.serve:app", "--host", "0.0.0.0", "--port", "8080"]
