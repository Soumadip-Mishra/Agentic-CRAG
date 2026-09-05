# ──── Stage 1: Builder ────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.docker.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.docker.txt

# ──── Stage 2: Runtime ────
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

# Ensure data directories exist even when data/ is omitted by .dockerignore
RUN mkdir -p /app/data /app/data/raw

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
