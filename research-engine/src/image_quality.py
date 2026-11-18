"""Image-quality diagnostics (not model epistemic uncertainty).

The returned measurements are descriptive proxies only. They must not be
reported as calibrated probabilities or used as a contradiction score.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
from PIL import Image


def compute_image_quality_metrics(image: Image.Image) -> Dict[str, float | int]:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if image.width < 2 or image.height < 2:
        raise ValueError("image dimensions must be at least 2x2 pixels")

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    high_frequency_residual = cv2.absdiff(gray, smoothed)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    return {
        "width": int(image.width),
        "height": int(image.height),
        "mean_absolute_high_frequency_residual": float(np.mean(high_frequency_residual)),
        "laplacian_variance": float(np.var(laplacian)),
        "luminance_standard_deviation": float(np.std(gray)),
    }


def compute_patch_sharpness(image: Image.Image, grid: int = 7) -> List[float]:
    """Laplacian variance per CLIP ViT patch cell (default 7x7 = 49 patches).

    Used only to derive per-patch trust weights for partial optimal transport.
    This is a descriptive quality signal, not model uncertainty.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if grid < 1:
        raise ValueError("grid must be >= 1")
    side = 224
    rgb = np.asarray(image.convert("RGB").resize((side, side)), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cell = side // grid
    if cell < 1:
        raise ValueError("grid is too large for the resized patch canvas")
    sharpness: List[float] = []
    for gy in range(grid):
        for gx in range(grid):
            patch = gray[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell]
            laplacian = cv2.Laplacian(patch, cv2.CV_64F)
            sharpness.append(float(np.var(laplacian)))
    return sharpness