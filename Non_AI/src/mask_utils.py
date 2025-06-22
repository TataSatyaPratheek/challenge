"""
src/mask_utils.py
-----------------
Light-weight utilities for post-processing and visualising instance masks.

All functions rely ONLY on OpenCV and NumPy – no ML libraries required.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Tuple

from . import config

# --------------------------------------------------------------------------- #
# 1. mask smoothing                                                           
# --------------------------------------------------------------------------- #
def smooth_mask(
    binary: np.ndarray,
    close_kernel: Tuple[int, int] = config.SMOOTH_MASK_CLOSE_KERNEL,
    blur_ksize: int = config.SMOOTH_MASK_BLUR_KSIZE,
    iterations: int = config.SMOOTH_MASK_ITERATIONS,
) -> np.ndarray:
    """
    Remove 1-pixel holes and staircase artefacts from a binary mask.

    Parameters
    ----------
    binary : np.ndarray
        8-bit mask (0 = background, 255 = foreground).
    close_kernel : (w, h)
        Elliptical kernel size for the closing operation.
    blur_ksize : int
        Kernel size for Gaussian blur (>=3 & odd); set 0 to skip blurring.
    iterations : int
        Morphology iterations (higher → stronger filling).

    Returns
    -------
    np.ndarray
        Smoothed binary mask (uint8: {0,255}).
    """
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    if blur_ksize and blur_ksize >= 3:
        closed = cv2.GaussianBlur(closed, (blur_ksize, blur_ksize), 0)
        # Re-binarise because Gaussian introduces grey pixels
        _, closed = cv2.threshold(closed, config.SMOOTH_MASK_THRESH, 255, cv2.THRESH_BINARY)

    return closed


# --------------------------------------------------------------------------- #
# 2. coloured overlay                                                          #
# --------------------------------------------------------------------------- #
def create_coloured_overlay(
    bgr: np.ndarray, markers: np.ndarray, alpha: float = config.OVERLAY_ALPHA
) -> np.ndarray:
    """
    Render each non-zero label in `markers` with a distinct pseudo-random colour
    and alpha-blend it over `bgr`.

    Parameters
    ----------
    bgr : np.ndarray
        Original BGR image.
    markers : np.ndarray
        int32 label matrix (0 = background, ≥1 = instance id).
    alpha : float
        Mask opacity in the final composite (0 = invisible, 1 = fully opaque).

    Returns
    -------
    np.ndarray
        BGR image with the coloured overlay.
    """
    overlay = np.zeros_like(bgr)

    # Golden-ratio hue stepping ⇒ visually distinct colours
    ids = [lbl for lbl in np.unique(markers) if lbl != 0]
    for i, lbl in enumerate(ids, 1):
        hue = (i * config.HUE_GOLDEN_RATIO_DEGREES) % 180  # 137° ≈ golden-ratio in HSV space
        colour = cv2.cvtColor(
            np.uint8([[[hue, config.OVERLAY_SATURATION, config.OVERLAY_VALUE]]]), cv2.COLOR_HSV2BGR
        )[0, 0].tolist()
        overlay[markers == lbl] = colour

    return cv2.addWeighted(bgr, 1.0, overlay, alpha, 0.0)


# --------------------------------------------------------------------------- #
# 3. helper: labels → single binary mask                                       #
# --------------------------------------------------------------------------- #
def labels_to_binary(markers: np.ndarray) -> np.ndarray:
    """
    Collapse an int-label matrix to a single 0/255 binary mask.

    Parameters
    ----------
    markers : np.ndarray
        Instance label matrix.

    Returns
    -------
    np.ndarray
        8-bit binary mask.
    """
    return np.where(markers > 0, 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 4. coloured mask from classified objects
# --------------------------------------------------------------------------- #
def create_coloured_mask(
    classified_objects: Dict, img_shape: tuple
) -> np.ndarray:
    """
    Render each accepted object from a dictionary in a unique colour.

    This creates a BGR image with coloured masks on a black background.

    Parameters
    ----------
    classified_objects : dict
        Dictionary of accepted objects, where each value has a "mask" key.
    img_shape : tuple
        The (height, width) of the output mask image.

    Returns
    -------
    np.ndarray
        BGR image with coloured masks.
    """
    canvas = np.zeros((*img_shape[:2], 3), dtype=np.uint8)
    if not classified_objects:
        return canvas

    for i, obj in enumerate(classified_objects.values()):
        hue = int((i * config.HUE_GOLDEN_RATIO_DEGREES) % 180)  # golden-ratio palette
        hsv = np.uint8([[[hue, config.MASK_SATURATION, config.MASK_VALUE]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        canvas[obj["mask"] > 0] = bgr.astype(np.uint8)

    return canvas
