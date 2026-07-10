# CPU-only image for the car-price-vision demo API.
# Build from the repo root so we can COPY the src/ package in:
#   docker build -f serving/Dockerfile -t car-price-vision-demo .
#   docker run -p 8000:8000 car-price-vision-demo

FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow's image codecs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Install serving-only runtime deps first for better layer caching.
COPY serving/requirements.txt /app/serving/requirements.txt
RUN pip install --no-cache-dir -r /app/serving/requirements.txt

# Install the car_price_vision package itself (needed for
# data.transforms / models.multitask / interpret.gradcam imports in app.py).
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN pip install --no-cache-dir --no-deps -e .

COPY serving/app.py /app/serving/app.py
COPY serving/static /app/serving/static

# TODO(phase 4): COPY the exported model artifact(s) in, or mount them at
# runtime via a volume, e.g.:
#   COPY model.onnx /app/model.onnx
# Configure paths via env vars (see serving/app.py):
ENV CPV_ONNX_MODEL_PATH=/app/model.onnx
ENV CPV_TORCH_CHECKPOINT_PATH=/app/model.pt

EXPOSE 8000

# NOTE (deploy-time, not build-time): this container is published via a
# Cloudflare Tunnel, not by publishing host ports. It stays on a private
# Docker network alongside the `cloudflared` connector -- see
# serving/deploy/docker-compose.yml and serving/deploy/README.md.

WORKDIR /app/serving
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
