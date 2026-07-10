#!/usr/bin/env bash
# Phase 0: inventory the `rtx` training server before staging data/training.
# Prints a report of GPU, disk, docker, and python/torch environment.
# Every check is best-effort (`|| true`) so a missing tool never aborts
# the whole report -- we want as much info as possible in one pass.

set -euo pipefail

section() {
    echo ""
    echo "==================================================================="
    echo "== $1"
    echo "==================================================================="
}

section "Host"
hostname || true
uname -a || true
date || true

section "GPU (nvidia-smi)"
nvidia-smi || echo "nvidia-smi not available or no NVIDIA GPU present."

section "Disk usage (df -h)"
df -h || true

section "Docker version"
docker version || echo "docker not available."

section "Docker info (brief)"
docker info --format '{{json .}}' 2>/dev/null | head -c 2000 || true
echo ""

section "Python"
python3 --version || true
which python3 || true

section "Torch / CUDA"
python3 - <<'PYEOF' || true
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"  device {i}:", torch.cuda.get_device_name(i))
    print("cuda version (torch build):", torch.version.cuda)
except ImportError as e:
    print("torch not importable:", e)
PYEOF

section "Torchvision"
python3 -c "import torchvision; print('torchvision:', torchvision.__version__)" || echo "torchvision not importable."

section "Done"
echo "Inventory complete. TODO(phase 0): review disk space vs. DVM-CAR size (~1.45M images) before staging data."
