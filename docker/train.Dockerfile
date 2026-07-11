# Training image for car-price-vision.
#
# Base: official PyTorch 2.8 + CUDA 12.8 (cu128) runtime — verified to run on
# the NVIDIA RTX PRO 6000 Blackwell (compute capability sm_120) on the `rtx`
# box: torch.cuda.is_available() -> True, GPU matmul kernels launch cleanly.
# torch + torchvision come from the base image; we add the data / eval / export
# / notebook dependencies on top.
#
# Build (on the training box, from the repo root):
#   docker build -f docker/train.Dockerfile -t car-price-vision:train .
#
# Typical run (repo + data mounted, GPU on, detached training — see README):
#   docker run --rm --gpus all \
#     -v $PWD:/workspace -v /data/car-price-vision:/data/car-price-vision \
#     -e PYTHONPATH=/workspace/src car-price-vision:train \
#     python -m car_price_vision.train --config configs/default.yaml
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
        pandas \
        scikit-learn \
        matplotlib \
        pillow \
        tqdm \
        pyyaml \
        onnx \
        onnxruntime \
        jupyter \
        nbconvert \
        ipykernel

WORKDIR /workspace
