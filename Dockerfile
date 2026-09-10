FROM python:3.12-slim

# System libs: libgl/glib for docling's image handling, curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
# scripts/serve.py and `uvicorn app.main:app` both import `app.*`, so the repo root
# must be importable inside the container.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app

# Torch from the CPU-only index (replaces PyPI for this call, so the multi-GB CUDA
# wheels are never fetched).
COPY pyproject.toml ./
RUN uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision
RUN uv pip install --system --no-cache -r pyproject.toml

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY seed/ ./seed/

EXPOSE 8000

CMD ["python", "scripts/serve.py"]

# NOTE: full image build + compose wiring is finalised in ticket #35.
