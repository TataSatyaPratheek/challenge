"""
src/preprocessing.py
--------------------
Image preprocessing utilities.

This module contains functions for preparing images for segmentation, such
as thresholding, noise reduction, and exposure correction. Creating this
module aligns with the project structure described in the README.
"""
from __future__ import annotations

import logging
import cv2
import numpy as np
from . import config

log = logging.getLogger(__name__)


def get_binary_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Applies a robust binary threshold. Tries OTSU first, but falls back
    to adaptive thresholding if the result is nearly all black or white.
    """
    # 1. Simple binary threshold - try OTSU first
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. If OTSU fails (all black/white), try adaptive
    mean_val = otsu.mean()
    if mean_val < config.OTSU_FAIL_MEAN_LOW or mean_val > config.OTSU_FAIL_MEAN_HIGH:
        log.debug(f"OTSU thresholding failed (mean={mean_val:.2f}), falling back to adaptive.")
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, config.ADAPTIVE_THRESH_BLOCK_SIZE,
            config.ADAPTIVE_THRESH_C
        )
    else:
        binary = otsu
    return binary