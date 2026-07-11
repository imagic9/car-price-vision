"""Tests for the image transform pipelines in data/transforms.py.

No real DVM-CAR images needed: a small synthetic PIL image exercises the
full resize/crop/normalize pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from car_price_vision.data.transforms import eval_transforms, train_transforms

# ImageNet mean/std normalization maps [0, 1] pixel values to roughly this
# range; a very generous bound since RandomResizedCrop/ColorJitter can push
# values slightly further for extreme pixel values.
NORMALIZED_VALUE_BOUND = 6.0


@pytest.fixture
def synthetic_image() -> Image.Image:
    """A small synthetic RGB image with varied pixel content (not blank), so
    normalization isn't tested against a degenerate all-zero input.
    """
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_eval_transforms_output_shape_and_dtype(synthetic_image):
    transform = eval_transforms(224)
    out = transform(synthetic_image)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_eval_transforms_values_in_normalized_range(synthetic_image):
    transform = eval_transforms(224)
    out = transform(synthetic_image)

    assert out.min().item() > -NORMALIZED_VALUE_BOUND
    assert out.max().item() < NORMALIZED_VALUE_BOUND


def test_eval_transforms_is_deterministic(synthetic_image):
    transform = eval_transforms(224)
    out_a = transform(synthetic_image)
    out_b = transform(synthetic_image)
    torch.testing.assert_close(out_a, out_b)


def test_train_transforms_output_shape_and_dtype(synthetic_image):
    transform = train_transforms(224)
    out = transform(synthetic_image)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_train_transforms_respects_custom_img_size(synthetic_image):
    transform = train_transforms(128)
    out = transform(synthetic_image)
    assert out.shape == (3, 128, 128)
