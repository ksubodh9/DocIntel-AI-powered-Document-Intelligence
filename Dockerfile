FROM python:3.11-slim

WORKDIR /app

# Force Python stdout/stderr to flush immediately - critical for real-time Docker logs
ENV PYTHONUNBUFFERED=1

# System dependencies for PyMuPDF and pdfplumber
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    libfreetype6-dev \
    gcc \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
# - Embeddings run via fastembed/onnxruntime (no torch), so there's no multi-GB
#   CUDA download — the image is small and builds fast.
# - --retries / --timeout make the install resilient to slow/flaky downloads.
# Plain RUN (no BuildKit features) so it builds anywhere: HF Spaces, Render, local.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt

# Copy source
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/vectorstore

# Cap math-library threads. torch / sentence-transformers otherwise spawn one
# thread per CPU core, which can thrash on small hosts.
ENV OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

# Listen on $PORT when provided, else 7860 (Hugging Face Spaces' default port).
# Render overrides the start command with its own $PORT; docker-compose sets
# PORT=8000 explicitly. Shell form so ${PORT} is expanded at runtime.
EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}