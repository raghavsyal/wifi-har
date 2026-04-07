FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock README.md ./

# Install all dependencies
RUN uv sync --frozen --no-dev

# Copy source
COPY wifi_har/   ./wifi_har/
COPY server/     ./server/
COPY models.py   ./
COPY inference.py ./
COPY openenv.yaml ./
COPY README.md    ./

# HF Spaces runs as non-root user (uid 1000)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["uv", "run", "server"]
