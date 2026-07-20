"""
anomaly_scorer.py

Turns per-patch anomaly distances into something a human can act on:
  1. An overall anomaly score for the image (0-100 scale)
  2. A heatmap showing exactly which regions triggered it, upsampled back
     to the original image resolution and overlaid as a translucent mask

This is the piece that makes the tool trustworthy rather than a black box:
you don't just get "ANOMALOUS", you see precisely where and can judge for
yourself whether it's a real defect or a false positive (e.g. a shadow,
a lighting artifact).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def scores_to_heatmap(
    patch_scores: torch.Tensor, grid_size: tuple[int, int], output_size: tuple[int, int]
) -> np.ndarray:
    """
    Reshape flat per-patch scores into the spatial grid, then upsample
    (bilinear) to the original image resolution so it can be overlaid.
    Returns a (H, W) float array normalized to [0, 1].
    """
    h, w = grid_size
    grid = patch_scores.reshape(1, 1, h, w)
    upsampled = F.interpolate(grid, size=output_size, mode="bilinear", align_corners=False)
    heatmap = upsampled.squeeze().numpy()

    # Normalize for display purposes only (the raw, un-normalized score is
    # what's used for the pass/fail anomaly decision — see AnomalyResult.raw_max_score)
    lo, hi = heatmap.min(), heatmap.max()
    if hi - lo > 1e-8:
        heatmap = (heatmap - lo) / (hi - lo)
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a red-scale heatmap over the original image."""
    image = image.convert("RGB")
    heatmap_img = (heatmap * 255).astype(np.uint8)

    # Simple red-channel heatmap (no matplotlib dependency needed)
    overlay = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
    overlay[..., 0] = heatmap_img  # red channel = anomaly intensity

    base = np.array(image.resize((heatmap.shape[1], heatmap.shape[0])))
    blended = (base * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return Image.fromarray(blended)


class AnomalyResult:
    def __init__(self, raw_scores: torch.Tensor, grid_size: tuple[int, int]):
        self.raw_scores = raw_scores
        self.grid_size = grid_size
        self.raw_max_score = float(raw_scores.max())
        self.raw_mean_score = float(raw_scores.mean())

    def heatmap(self, output_size: tuple[int, int]) -> np.ndarray:
        return scores_to_heatmap(self.raw_scores, self.grid_size, output_size)

    def normalized_score(self, calibration_max: float) -> float:
        """
        Scale raw_max_score to a 0-100 "anomaly score" using a calibration
        value (typically the max score observed across the reference images
        themselves, i.e. normal-vs-normal noise floor). Values consistently
        above 100 indicate patches further from normal than any reference
        image was from any other reference image.
        """
        if calibration_max <= 0:
            return 0.0
        return round(100 * self.raw_max_score / calibration_max, 1)
