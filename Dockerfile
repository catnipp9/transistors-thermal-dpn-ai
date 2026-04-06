FROM python:3.11-slim

# System deps needed by opencv and ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source code
COPY api/       api/
COPY models/    models/

# Copy trained model checkpoints
COPY checkpoints/best_yolo_model.pt     checkpoints/best_yolo_model.pt
COPY checkpoints/best_sklearn_model.joblib checkpoints/best_sklearn_model.joblib
COPY checkpoints/best_fusion_model.joblib  checkpoints/best_fusion_model.joblib

# Expose API port
EXPOSE 7860

# Start FastAPI on port 7860 (required by Hugging Face Spaces)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
