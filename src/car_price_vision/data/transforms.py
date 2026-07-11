"""Image transform pipelines for training and evaluation.

Both pipelines resize to a square `img_size` and normalize with standard
ImageNet statistics (matching the pretrained torchvision backbones used in
models/backbone.py). Training adds mild augmentation; eval is deterministic.
"""

from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transforms(img_size: int = 224) -> T.Compose:
    """Augmented pipeline used during training.

    Random resized crop + horizontal flip + light color jitter. Flip/jitter
    are intentionally mild: car photos are mostly side/3-quarter shots and
    aggressive color jitter could wash out paint-color cues that are
    genuinely predictive of era/trim.
    """
    return T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def eval_transforms(img_size: int = 224) -> T.Compose:
    """Deterministic pipeline used for val/test/inference.

    Resize the short side then center-crop to `img_size`, matching standard
    torchvision ImageNet eval convention.
    """
    resize_size = int(img_size * 1.14)  # ~256 for 224, matches torchvision recipes
    return T.Compose(
        [
            T.Resize(resize_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
