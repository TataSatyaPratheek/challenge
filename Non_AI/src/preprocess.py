from __future__ import annotations

import cv2
import numpy as np

from .config import AUTO_CANNY_HIGH, AUTO_CANNY_LOW, CLAHE_CLIP


def stabilise(
    img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply colour balancing and adaptive Canny edge detection.

    Args:
        img: Input BGR image as a NumPy array.

    Returns:
        A tuple containing:
            - stabilised_bgr_img: Colour-balanced BGR image.
            - gray_img: Grayscale version of the stabilised image.
            - edges_img: Canny edges detected from the grayscale image.
    """
    # Colour balancing using CLAHE in LAB space
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    l_channel_eq = clahe.apply(l_channel)

    stabilised_bgr_img = cv2.cvtColor(
        cv2.merge([l_channel_eq, a_channel, b_channel]), cv2.COLOR_LAB2BGR
    )

    # Convert to grayscale and blur
    gray_img = cv2.cvtColor(stabilised_bgr_img, cv2.COLOR_BGR2GRAY)
    # Consider making GaussianBlur kernel size (7,7) and sigma (1.5) configurable
    blurred_img = cv2.GaussianBlur(gray_img, (7, 7), 1.5)

    # Auto-tune Canny thresholds for dark frames
    canny_low_thresh, canny_high_thresh = AUTO_CANNY_LOW, AUTO_CANNY_HIGH
    # Consider making the mean intensity threshold (60) configurable
    if gray_img.mean() < 60:
        canny_low_thresh //= 2
        canny_high_thresh //= 2

    edges_img = cv2.Canny(blurred_img, canny_low_thresh, canny_high_thresh)

    return stabilised_bgr_img, gray_img, edges_img
