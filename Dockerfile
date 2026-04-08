# OpenEnv-FragileChain Dockerfile
#
# Multi-stage build for the pharmaceutical cold-chain environment.
# Compatible with Hugging Face Spaces deployment.
#
# Build:
#   docker build -t fragilechain .
#
# Run:
#   docker run -p 8000:8000 fragilechain
#
# With OpenAI baseline:
#   docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... fragilechain

FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer cache optimization)
COPY pyproject.toml ./
COPY server/requirements.txt ./server/

# Install dependencies (without the project itself first for better caching)
RUN uv pip install --system --no-cache \
    fastapi>=0.115.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.0.0 \
    requests>=2.31.0 \
    openai>=1.0.0

# Try to install openenv-core; graceful fallback if unavailable
RUN uv pip install --system --no-cache "openenv-core[core]>=0.2.1" || \
    echo "openenv-core not available, using fallback mode"

# ---- Runtime stage ----
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app

# Set environment
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

# Create output directories
RUN mkdir -p /app/outputs/logs /app/outputs/evals

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Expose port
EXPOSE 7860

# Default: run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
