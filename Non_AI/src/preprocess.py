from __future__ import annotations

import cv2
import numpy as np
from .config import (
    PREPROC_DARK_THRESH,
    PREPROC_BRIGHTEN_ALPHA,
    PREPROC_BRIGHTEN_BETA,
    PREPROC_BLUR_KERNEL_SIZE,
    PREPROC_BLUR_SIGMA,
    PREPROC_CANNY_LOW,
    PREPROC_CANNY_HIGH,
)


def stabilise(
    img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply minimal preprocessing to stabilise the image for segmentation.

    The goal is to preserve object boundaries at all costs. This involves a
    conditional brightness boost for very dark images, a minimal blur to
    suppress noise for Canny, and conservative Canny thresholds.
    Args:
        img: Input BGR image as a NumPy array.

    Returns:
        A tuple containing:
            - stabilised_bgr_img: Colour-balanced BGR image.
            - gray_img: Grayscale version of the stabilised image.
            - edges_img: Canny edges detected from the grayscale image.
    """
    # 1. Convert to grayscale and conditionally correct for severe darkness
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray_img.mean() < PREPROC_DARK_THRESH:
        # Boost contrast (alpha) and brightness (beta)
        gray_img = cv2.convertScaleAbs(
            gray_img, alpha=PREPROC_BRIGHTEN_ALPHA, beta=PREPROC_BRIGHTEN_BETA
        )
    # Create a 3-channel version for consumers that expect a BGR image
    stabilised_bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # 2. Apply a minimal blur to reduce noise for edge detection
    ksize = (PREPROC_BLUR_KERNEL_SIZE, PREPROC_BLUR_KERNEL_SIZE)
    blurred_img = cv2.GaussianBlur(gray_img, ksize, PREPROC_BLUR_SIGMA)

    # 3. Apply Canny edge detection with conservative, fixed thresholds
    edges_img = cv2.Canny(blurred_img, PREPROC_CANNY_LOW, PREPROC_CANNY_HIGH)

    return stabilised_bgr_img, gray_img, edges_img
