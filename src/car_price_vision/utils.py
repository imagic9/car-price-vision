"""Small shared utilities: seeding, config loading, device selection, logging."""

from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch RNGs for (best-effort) reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Args:
        prefer: optional explicit device string (e.g. "cuda:0", "cpu", "mps").
            If given and available, it is used as-is.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Config:
    """Thin typed wrapper around the nested dict loaded from YAML.

    Kept intentionally close to a dict (see `raw`) so new config keys added
    in configs/*.yaml do not require code changes here — access nested
    values via `cfg.raw["section"]["key"]` or the convenience accessors
    below for the keys we already rely on throughout the codebase.
    """

    raw: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def paths(self) -> dict[str, str]:
        return self.raw.get("paths", {})

    @property
    def data(self) -> dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def model(self) -> dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def loss(self) -> dict[str, Any]:
        return self.raw.get("loss", {})

    @property
    def train(self) -> dict[str, Any]:
        return self.raw.get("train", {})

    @property
    def eval(self) -> dict[str, Any]:
        return self.raw.get("eval", {})


def load_config(path: str | Path) -> Config:
    """Load a YAML config file (see configs/default.yaml) into a Config."""
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw or {})


class AverageMeter:
    """Tracks a running average of a scalar (e.g. loss, MAE) over steps."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __repr__(self) -> str:
        return f"{self.name}={self.avg:.4f}"


def setup_logging(log_dir: str | Path | None = None, name: str = "car_price_vision") -> logging.Logger:
    """Configure a logger that writes to stdout and, if given, a log file.

    Designed so train.py can run detached (e.g. nohup / tmux) with all
    progress persisted to disk, not just stdout.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "train.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
