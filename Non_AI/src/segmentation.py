# src/segmentation.py
"""
Segmentation / marker-generation for screw-counting.

Fixes:
• concentric-duplicate explosion ⇒ radius-aware NMS
• soft-focus under-count ⇒ on-demand tile detection
• fallback still guarded by OPEN + area gate
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from .config import (
    get_adaptive_parameters,
    RELAXED_PARAM2,
    RELAXED_MIN_DIST,
    FALLBACK_MAX_CONTOURS,
    DEDUP_RADIUS_FACTOR,
)
from .tile_utils import generate_tiles, merge_tile_results

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# helper – draw & mask circles
# --------------------------------------------------------------------------- #
def _draw_circle_markers(shape, circles: List[Tuple[int, int, int]]) -> np.ndarray:
    m = np.zeros(shape, dtype=np.int32)
    for idx, (x, y, r) in enumerate(circles, 1):
        cv2.circle(m, (x, y), r, idx, -1)
    return m


def _mask_circles(gray: np.ndarray, circles: List[Tuple[int, int, int]]) -> np.ndarray:
    if not circles:
        return gray
    mask = np.ones_like(gray, np.uint8) * 255
    for x, y, r in circles:
        cv2.circle(mask, (x, y), int(r * 1.3), 0, -1)
    return cv2.bitwise_and(gray, mask)


# --------------------------------------------------------------------------- #
# radius-aware dedup
# --------------------------------------------------------------------------- #
def _dedup_circles(
    circles: List[Tuple[int, int, int]],
    min_dist: int,
    rad_tol_factor: float = DEDUP_RADIUS_FACTOR,
) -> List[Tuple[int, int, int]]:
    """
    Keep only one circle if centres are closer than *min_dist* AND radii differ
    by <= `rad_tol_factor` × smaller_radius –  removes concentric duplicates.
    """
    kept: list[Tuple[int, int, int]] = []
    # Smallest radius first (tends to hug the screw edge best)
    for x, y, r in sorted(circles, key=lambda c: c[2]):  # small radius first
        dup = False
        for kx, ky, kr in kept:
            if (x - kx) ** 2 + (y - ky) ** 2 <= min_dist**2:
                # consider same if radii differ less than or equal to tol-factor
                if abs(r - kr) <= rad_tol_factor * min(r, kr):
                    dup = True
                    break
        if not dup:
            kept.append((x, y, r))
    return kept


# --------------------------------------------------------------------------- #
# single-tile two-pass Hough
# --------------------------------------------------------------------------- #
def _two_stage_hough(gray: np.ndarray, params) -> List[Tuple[int, int, int]]:
    """Return list of unique (x,y,r) circles; may return empty list."""
    # strict
    strict = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (9, 9), 2),
        cv2.HOUGH_GRADIENT,
        dp=params["hough_dp"],
        minDist=params["hough_min_dist"],
        param1=params["hough_param1"],
        param2=params["hough_param2"],
        minRadius=params["hough_min_radius"],
        maxRadius=params["hough_max_radius"],
    )
    strict = (
        [tuple(map(int, c)) for c in np.round(strict[0, :]).astype(int)]
        if strict is not None
        else []
    )

    # relaxed on residual
    residual = _mask_circles(gray, strict)
    relaxed = cv2.HoughCircles(
        cv2.GaussianBlur(residual, (7, 7), 1),
        cv2.HOUGH_GRADIENT,
        dp=params["hough_dp"],
        minDist=RELAXED_MIN_DIST,
        param1=params["hough_param1"],
        param2=RELAXED_PARAM2,
        minRadius=params["hough_min_radius"],
        maxRadius=params["hough_max_radius"],
    )
    relaxed = (
        [tuple(map(int, c)) for c in np.round(relaxed[0, :]).astype(int)]
        if relaxed is not None
        else []
    )

    unique = _dedup_circles(
        strict + relaxed,
        min_dist=params["hough_min_dist"] // 2,
    )
    return unique


# --------------------------------------------------------------------------- #
# fallback (robust contour) – unchanged except for max-contours constant
# --------------------------------------------------------------------------- #
def _robust_fallback(gray: np.ndarray, params) -> np.ndarray:
    log.warning("HQ fallback engaged")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    )
    min_area = int(params["base_min_area"] * 0.2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if len(contours) > FALLBACK_MAX_CONTOURS:
        log.error("Fallback aborted: %d contours (> %d)", len(contours), FALLBACK_MAX_CONTOURS)
        return np.zeros(gray.shape, np.int32)

    markers = np.zeros(gray.shape, np.int32)
    for idx, c in enumerate(contours, 1):
        cv2.drawContours(markers, [c], -1, idx, -1)
    log.info("Fallback produced %d markers", len(contours))
    return markers


# --------------------------------------------------------------------------- #
# public entry
# --------------------------------------------------------------------------- #
def split_instances(img: np.ndarray) -> np.ndarray:
    if img is None:
        return np.zeros((1, 1), np.int32)

    params = get_adaptive_parameters(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if params["pipeline"] == "small_object_recovery":
        # small pipeline unchanged
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        bot = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        enhanced = cv2.add(top, bot)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2,
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        markers = np.zeros(gray.shape, np.int32)
        for i, c in enumerate(contours, 1):
            cv2.drawContours(markers, [c], -1, i, -1)
        log.info("Small pipeline: %d contours→markers", len(contours))
        return markers

    # ---------------- HQ branch ---------------- #
    circles = _two_stage_hough(gray, params)
    if 15 <= len(circles) <= 250:  # normal range
        log.info("Two-stage Hough produced %d circles", len(circles))
        return _draw_circle_markers(gray.shape, circles)

    # --- decide if we need tiling --------------------------------------------
    need_tiles = (len(circles) < 15) or (len(circles) > 250)
    if need_tiles and img.shape[0] * img.shape[1] > 3_000_000:
        log.info("Tiling engaged (circle count = %d)", len(circles))
        tile_dets = []
        for tile in generate_tiles(img, tile_size=1024, overlap=0.25):
            sub_gray = cv2.cvtColor(tile["tile"], cv2.COLOR_BGR2GRAY)
            sub_circles = _two_stage_hough(sub_gray, params)
            dets = [
                {"x": x, "y": y, "radius": r, "confidence": 1.0}
                for x, y, r in sub_circles
            ]
            tile_dets.append({"global_offset": tile["global_offset"], "detections": dets})

        from .tile_utils import merge_tile_results

        merged = merge_tile_results(tile_dets, nms_threshold=0.4)
        if merged:
            return _draw_circle_markers(gray.shape, [(d["x"], d["y"], d["radius"]) for d in merged])

    # final fallback
    return _robust_fallback(gray, params)
