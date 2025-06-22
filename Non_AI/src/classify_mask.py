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

from .mask_utils import smooth_mask
from .config import (
    get_adaptive_parameters,
    MIN_CIRC_HQ,
    MAX_CIRC_HQ,
    HOLE_RATIO_LOW,
    HOLE_RATIO_HIGH,
    HEX_VERT_TOL,
    BOLT_ASPECT_MIN,
    BOLT_MIN_AREA_FACTOR,
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
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            areas.append(cv2.contourArea(max(cnts, key=cv2.contourArea)))

    # use middle 60 % to kill extreme outliers
    if len(areas) < 5:
        return base_min_area, base_max_area

    areas.sort()
    k = int(len(areas) * 0.2)
    # Only trim if we have enough data points
    core = areas[k:-k] if len(areas) > 10 else areas
    if not core: # handle case where trimming results in empty list
        return base_min_area, base_max_area
    median_area = float(np.median(core))

    dyn_min = int(max(base_min_area * 0.5, median_area * 0.30))
    dyn_max = int(min(base_max_area * 2.0, median_area * 5.0))

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

        # Create mask for the current label
        mask = np.uint8(markers == label)

        # A rough pixel count to decide on smoothing. This is faster than finding
        # contours first. We apply light smoothing only to smaller objects, which
        # are more likely to be fragmented.
        pixel_count = np.count_nonzero(mask)
        if 0 < pixel_count < 500:
            mask = smooth_mask(mask, close_kernel=(3, 3), blur_ksize=0, iterations=1)

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

        # DEBUG: Log circularity values to diagnose filter issues
        log.debug(f"Object {label}: area={area:.1f}, circ={circularity:.3f}, hole={has_hole}")

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
        verts = len(approx)
        if 5 <= verts <= 7:
            # approx[i] is [[x,y]], so approx[i][0] is [x,y] for linalg.norm
            points = [pt[0] for pt in approx] 
            edges = [np.linalg.norm(points[i] - points[(i + 1) % verts])
                     for i in range(verts)]
            mean_edges = np.mean(edges)
            # Coefficient of variation for edge lengths. Lower is more regular.
            # Add small epsilon to prevent division by zero for very small/degenerate polygons.
            hex_score = np.std(edges) / (mean_edges + 1e-6) if mean_edges > 1e-9 else 1.0
        else:
            hex_score = 1.0 # Not 5-7 vertices, high score (less nut-like)

        # Calculate the ratio of inner (hole) area to outer area
        ratio = inner_area / area if (has_hole and area > 0) else 0.0

        # --------------- final type decision --------------
        if has_hole:
            # Nut: has hole, regular polygon, correct hole ratio
            if hex_score < 0.15 and HOLE_RATIO_LOW <= ratio <= HOLE_RATIO_HIGH:
                obj_type = "nut"
            else:
                obj_type = "bolt"
        else:
            # --------------- bolt detector for hole-less elongated parts ----------
            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            aspect = max(w, h) / (min(w, h) + 1e-6)
            if aspect > BOLT_ASPECT_MIN and area > min_area * BOLT_MIN_AREA_FACTOR:
                obj_type = "bolt"
            else:
                obj_type = "screw"

        results[obj_id] = {
            "type": obj_type,
            "mask": mask,
            "area": area,
            "circularity": circularity,
        }
        obj_id += 1

    log.info("Classification kept %d objects.", len(results))
    return results
