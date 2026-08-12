# syntax=docker/dockerfile:1
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
# - A BuildKit cache mount (/root/.cache/pip) keeps downloaded wheels between
#   builds, so rebuilds don't re-download packages. (Needs the `# syntax` line
#   above and BuildKit, which is the default in modern Docker.)
# - Embeddings run via fastembed/onnxruntime (no torch), so there's no multi-GB
#   CUDA download — the image is small and builds fast.
# - --retries / --timeout make the install resilient to slow/flaky downloads.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --retries 5 --timeout 120 -r requirements.txt

# Pre-download the embedding model into the image at BUILD time, into a fixed
# cache dir. This runs after pip install (so fastembed is available) but before
# copying source, so the ~130 MB download is cached as its own layer and only
# re-runs when this line or the base deps change — not on every code edit.
# Baking it in means the container never downloads at runtime, so a restart /
# --reload can no longer interrupt a download and leave a half-written model
# (the "model_optimized.onnx File doesn't exist" failure). Same fastembed
# version builds and runs it, so the expected ONNX filename always matches.
ENV EMBEDDING_CACHE_DIR=/app/.fastembed_cache
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/.fastembed_cache')"

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

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]