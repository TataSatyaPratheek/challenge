"""
src/classify_mask.py
--------------------

Final-pass validator that turns raw `markers` (an int32 label map produced by
segmentation.py) into a dictionary of accepted objects. Key changes include:

1.  Adaptive area bounds now rely on true `cv2.contourArea`, never the rough
    pixel sum that exaggerates ragged masks.
2.  Enhanced classification: distinguishes between "screw", "nut", and "bolt"
    based on hole presence, inner/outer area ratios, and polygonal shape
    approximation (for nuts). Replaces the previous "washer" vs "screw" logic.
3.  Optional guard-rail: if segmentation has exploded into > 3000 labels the
    function bails out early (prevents runaway memory in counter.py).

Pure-OpenCV, no external ML.
"""

from __future__ import annotations

import logging
from typing import Dict

import cv2
import numpy as np

from .config import (
    get_adaptive_parameters,
    MIN_CIRC_HQ,
    MAX_CIRC_HQ,
    HOLE_RATIO_LOW,
    HOLE_RATIO_HIGH,
    HEX_VERT_TOL,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# adaptive area ladder
# --------------------------------------------------------------------------- #
def _get_adaptive_area_bounds(
    markers: np.ndarray, base_min_area: int, base_max_area: int
) -> tuple[int, int]:
    """
    Derive dynamic [min,max] area thresholds from the *true* contour areas
    of the first-pass markers.  Robust against ragged masks and outliers.
    """
    areas: list[float] = []

    for label in np.unique(markers):
        if label == 0:  # background
            continue

        mask = np.uint8(markers == label)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        areas.append(cv2.contourArea(max(contours, key=cv2.contourArea)))

    if len(areas) < 5:  # insufficient statistics
        return base_min_area, base_max_area

    median_area = float(np.median(areas))
    dyn_min = int(max(base_min_area * 0.5, median_area * 0.20))
    dyn_max = int(min(base_max_area * 1.5, median_area * 8.0))

    log.debug(
        "Adaptive area bounds: median=%.1f  →  [%d,%d] (base [%d,%d])",
        median_area,
        dyn_min,
        dyn_max,
        base_min_area,
        base_max_area,
    )
    return dyn_min, dyn_max


# --------------------------------------------------------------------------- #
# main routine
# --------------------------------------------------------------------------- #
def classify_objects(markers: np.ndarray) -> Dict[int, dict]:
    """
    Inspect every label in *markers*, apply adaptive area & circularity gates,
    decide washer vs screw.  Returns:

        { obj_id (int) :
            {
              "type"       : "screw" | "nut" | "bolt", # Reflects new classification types
              "mask"       : np.uint8 (1-ch binary mask),
              "area"       : float,
              "circularity": float
            }, ...
        }
    """
    results: Dict[int, dict] = {}

    if markers is None or markers.size == 0 or np.max(markers) == 0:
        return results

    params = get_adaptive_parameters(markers.shape)
    base_min_area, base_max_area = params["base_min_area"], params["base_max_area"]

    # dynamic circularity range --------------------------------------------------
    if params["pipeline"] == "high_quality_detection":
        # For HQ, params["min_circularity"] from get_adaptive_parameters is MIN_CIRC_HQ.
        min_circularity_filter = MIN_CIRC_HQ # Reverted: max(params["min_circularity"], MIN_CIRC_HQ) is redundant
        max_circularity_filter = MAX_CIRC_HQ
    else:  # small-object pipeline
        min_circularity_filter = params["min_circularity"]
        max_circularity_filter = 1.0  # Default upper bound for non-HQ

    # Bail-out guard-rail
    if np.unique(markers).size - 1 > 3000:
        log.error(">3000 marker labels – likely fallback explosion – aborting classification.")
        return results

    # Compute dynamic area ladder
    min_area, max_area = _get_adaptive_area_bounds(markers, base_min_area, base_max_area)

    log.info(
        "Classify: pipeline=%s area=[%d,%d] circ_filter=[%.2f, %.2f]",
        params["pipeline"],
        min_area,
        max_area,
        min_circularity_filter,
        max_circularity_filter,
    )
    # Consider adding HOLE_RATIO_LOW, HOLE_RATIO_HIGH, HEX_VERT_TOL to this log if verbose debug is needed.

    obj_id = 1
    for label in np.unique(markers):
        if label == 0:
            continue

        mask = np.uint8(markers == label)
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        # parent contour
        idx = np.argmax([cv2.contourArea(c) for c in contours])
        contour = contours[idx]
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(contour, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri + 1e-6)

        # --------------- hole & inner-area -----------------
        has_hole, inner_area = False, 0.0
        if hierarchy is not None and hierarchy[0][idx][2] != -1: # Check for a child contour (a hole)
            has_hole = True
            child_idx = hierarchy[0][idx][2] # Index of the first child contour
            inner_area = cv2.contourArea(contours[child_idx])

        # --------------- circularity gate -----------------
        # If the object has no hole, it must pass the circularity filter to be considered.
        # If it has a hole, this check is bypassed, as hole features will determine its type.
        if (
            not has_hole
            and not (min_circularity_filter <= circularity <= max_circularity_filter)
        ):
            continue

        # --------------- additional shape cues ------------
        # Approximate the contour to a polygon to check for hex-like shapes (nuts)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        hex_score = abs(len(approx) - 6)  # Lower score is more hex-like (0 or 1 for HEX_VERT_TOL=1)
        
        # Calculate the ratio of outer area to inner (hole) area
        ratio = area / inner_area if (has_hole and inner_area > 1) else 0.0

        # --------------- final type decision --------------
        if not has_hole:
            obj_type = "screw"  # Objects without holes are screws
        else:  # Object has a hole
            if hex_score <= HEX_VERT_TOL and HOLE_RATIO_LOW <= ratio <= HOLE_RATIO_HIGH:
                obj_type = "nut"   # Hex-like shape with a specific hole area ratio
            else:
                obj_type = "bolt"  # Other objects with holes (e.g., traditional washers, actual bolts)

        results[obj_id] = {
            "type": obj_type,
            "mask": mask,
            "area": area,
            "circularity": circularity,
        }
        obj_id += 1

    log.info("Classification kept %d objects.", len(results))
    return results


# --------------------------------------------------------------------------- #
# colour overlay for visual debug
# --------------------------------------------------------------------------- #
def create_colored_mask(classified_objects: dict, img_shape: tuple) -> np.ndarray:
    """Render each accepted object in a unique colour (HSV-golden-ratio palette)."""
    canvas = np.zeros((*img_shape[:2], 3), dtype=np.uint8)
    if not classified_objects:
        return canvas

    for i, (_, obj) in enumerate(classified_objects.items()):
        hue = int((i * 137) % 180)  # golden-ratio trick in HSV space
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        canvas[obj["mask"] > 0] = bgr.astype(np.uint8)

    return canvas
