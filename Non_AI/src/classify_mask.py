"""
src/classify_mask.py
--------------------

Final-pass validator that turns raw `markers` (an int32 label map produced by
segmentation.py) into a dictionary of accepted objects. Key changes include:

1.  Adaptive area bounds now rely on true `cv2.contourArea`, never the rough
    pixel sum that exaggerates ragged masks.
2.  Enhanced classification: distinguishes between "screw", "nut", and "bolt"
    (placeholder logic). The main filtering is now based on adaptive area
    thresholds calculated from the image's object size distribution.
Pure-OpenCV, no external ML.
"""

from __future__ import annotations

import logging
from typing import Dict

import cv2
import numpy as np

from . import config
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# main routine
# --------------------------------------------------------------------------- #
def classify_objects(markers: np.ndarray) -> Dict[int, dict]:
    """
    Filters segmented objects based on fixed area bounds.

    This function validates objects from a raw marker map against fixed area
    thresholds from the configuration to remove noise and invalid detections.
    """
    if markers is None or markers.size == 0 or np.max(markers) == 0:
        return {}

    unique_labels = np.unique(markers)

    # Emergency brake for massive object counts to prevent memory/performance issues.
    if len(unique_labels) > config.CLASSIFY_HARD_LIMIT_OBJECTS:
        log.error(f"Too many objects ({len(unique_labels)}) - likely segmentation explosion. Aborting classification.")
        return {}

    # 1. Extract all potential objects and their properties in a single pass
    potential_objects = []
    all_areas = []
    for label in unique_labels[unique_labels != 0]:
        mask = np.uint8(markers == label)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        # Calculate properties for the largest contour found for this label
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

        potential_objects.append(
            {"mask": mask, "area": area, "label": label, "circularity": circularity}
        )
        all_areas.append(area)

    if not potential_objects:
        return {}

    # 2. Apply fixed area bounds for filtering.
    # As per new configuration, always use fixed bounds, dynamic percentiles are deprecated.
    min_area = config.FIXED_MIN_OBJECT_AREA
    max_area = config.FIXED_MAX_OBJECT_AREA
    log.info(
        f"Using fixed size bounds [{min_area:.0f}, {max_area:.0f}] for classification. "
        f"({len(all_areas)} potential objects before filtering)"
    )

    # 3. Filter objects and build the final dictionary
    results: Dict[int, dict] = {}
    obj_id = 1
    for obj in potential_objects:
        if min_area <= obj["area"] <= max_area:
            results[obj_id] = {
                "type": "object",
                "mask": obj["mask"],
                "area": obj["area"],
                "circularity": obj["circularity"],
            }
            obj_id += 1
        else:
            log.debug(
                f"Object with label {obj['label']} rejected: area={obj['area']:.0f} outside [{min_area:.0f}, {max_area:.0f}]"
            )

    log.info(
        "Classification kept %d of %d objects.", len(results), len(potential_objects)
    )
    return results
