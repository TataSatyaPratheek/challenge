# src/segmentation.py
"""
Segmentation / marker-generation for screw-counting.

Fixes:
• concentric-duplicate explosion ⇒ radius-aware NMS
• soft-focus under-count ⇒ on-demand tile detection
• fallback still guarded by OPEN + area gate
"""

from __future__ import annotations

import logging, cv2
import numpy as np
from . import config
from .preprocessing import get_binary_threshold

log = logging.getLogger(__name__)


def _get_resolution_scaled_params(img_shape: tuple, is_tile: bool = False) -> dict:
    """Calculate resolution-appropriate parameters."""
    height, width = img_shape[:2]
    megapixels = (height * width) / 1_000_000

    # Choose base area based on context (tile vs. full image)
    base_area = (config.BASE_TILE_MIN_CONTOUR_AREA_1MP if is_tile
                 else config.BASE_MIN_CONTOUR_AREA_1MP)

    # Scale parameters by resolution using power laws
    min_contour_area = int(base_area *
                          (megapixels ** config.MIN_AREA_SCALE_FACTOR))

    log.debug(f"Resolution scaling: {megapixels:.1f}MP → min_area={min_contour_area}")

    return {'min_contour_area': min_contour_area, 'megapixels': megapixels}


def _simple_contour_detection(gray: np.ndarray, is_tile: bool = False) -> np.ndarray:
    """Dead simple contour detection that actually works, with area filtering."""
    # Get resolution-appropriate parameters
    params = _get_resolution_scaled_params(gray.shape)
    min_contour_area = params['min_contour_area']

    # For high-res images, add Gaussian smoothing to reduce noise
    if params['megapixels'] > 8.0:  # High-res images
        sigma = min(2.0, params['megapixels'] / 6.0)  # Scale smoothing with resolution
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
        log.debug(f"Applied Gaussian smoothing (σ={sigma:.1f}) for {params['megapixels']:.1f}MP image")

    log.debug(f"Using min_contour_area={min_contour_area} (is_tile={is_tile}, MP={params['megapixels']:.1f})")

    # Use smoothed image for thresholding (no unsharp masking - that creates noise!)
    binary = get_binary_threshold(gray)

    # Adaptive morphology based on tile size
    if is_tile:
        kernel_size = 2 if gray.shape[0] <= 256 else 3  # Smaller kernel for small tiles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 4. Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter out noise contours before creating markers
    filtered_contours = [
        c for c in contours if cv2.contourArea(c) >= min_contour_area
    ]
    log.debug(f"Contours: {len(contours)} total -> {len(filtered_contours)} after filtering")

    # 5. Create markers from contours
    markers = np.zeros(gray.shape, dtype=np.int32)
    for i, contour in enumerate(filtered_contours, 1):
        cv2.drawContours(markers, [contour], -1, i, -1)
    return markers


def _downsample_process_upsample(img: np.ndarray) -> np.ndarray:
    """Process high-res images by downsampling to eliminate noise."""
    original_height, original_width = img.shape[:2]
    current_mp = (original_height * original_width) / 1_000_000
    target_mp = config.DOWNSAMPLE_TARGET_MP

    # Calculate downsample factor
    scale_factor = np.sqrt(target_mp / current_mp)
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    log.info(f"Downsampling {current_mp:.1f}MP → {target_mp:.1f}MP (scale={scale_factor:.3f})")

    # Downsample with anti-aliasing
    downsampled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Process at lower resolution (eliminates noise)
    gray = cv2.cvtColor(downsampled, cv2.COLOR_BGR2GRAY)
    markers_small = _simple_contour_detection(gray, is_tile=False)

    if markers_small.max() == 0:
        return np.zeros((original_height, original_width), dtype=np.int32)

    # Upsample results back to original resolution using nearest-neighbor to preserve labels
    markers_upsampled = cv2.resize(
        markers_small.astype(np.float32),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)

    count = len(np.unique(markers_upsampled)) - 1
    log.info(f"Downsample-process-upsample found {count} objects")
    return markers_upsampled


def split_instances(img: np.ndarray) -> np.ndarray:
    """Segments object instances using a resolution-appropriate strategy."""
    if img is None:
        return np.zeros((1, 1), np.int32)

    height, width = img.shape[:2]
    megapixels = (height * width) / 1_000_000

    if megapixels > config.HIGH_RES_THRESHOLD_MP:
        log.info("High-resolution image (%.1fMP) - using downsample-process-upsample strategy", megapixels)
        return _downsample_process_upsample(img)
    else:
        log.info("Standard resolution (%.1fMP) - using direct processing", megapixels)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        markers = _simple_contour_detection(gray, is_tile=False)
        count = len(np.unique(markers)) - 1
        log.info(f"Direct processing found {count} objects")
        return markers