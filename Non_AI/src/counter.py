"""
src/counter.py
==============

Coordinator for a *single* image:  runs segmentation  →  classification,
handles artefact I/O and enforces a global sanity veto so that pathological
fallback explosions do **not** clutter disk with multi-MB CSV/mask files.

Changes vs. earlier version
---------------------------
1.  If `total_objects > MAX_REASONABLE_COUNT` the result is considered
    invalid; the image is logged as an error and no artefacts are written.
2.  Writes an extra *.txt* alongside the mask that contains a one-line
    summary (useful for batch grepping).
3.  Returns **None** when vetoed so the CLI summary doesn’t count it.

Pure-OpenCV; no other dependencies.
"""

from __future__ import annotations

import csv
import numpy as np
import logging
import pathlib
from typing import Optional

import cv2

from .classify_mask import classify_objects
from .config import (
    CSV_DIR, MASK_DIR, MAX_REASONABLE_COUNT, MAX_MEAN_INTENSITY, MIN_MEAN_INTENSITY, OUT_DIR
)
from .mask_utils import (
    create_coloured_mask,
    create_coloured_overlay,
    labels_to_binary,
)
from .segmentation import split_instances

log = logging.getLogger(__name__)


def _write_csv(csv_path: pathlib.Path, objects: dict[int, dict]) -> None:
    """Dump id,type,area,circularity to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object_id", "type", "area", "circularity"])
        for i, (_, obj) in enumerate(objects.items(), 1):
            writer.writerow(
                [
                    i,
                    obj["type"],
                    int(obj["area"]),
                    f"{obj['circularity']:.3f}",
                ]
            )


def process_image(image_path: str | pathlib.Path) -> Optional[int]:
    """
    Full non-AI pipeline for one image.

    Returns:
        • int  – count of accepted objects
        • None – if the global sanity veto triggered or image failed to load
    """
    path = pathlib.Path(image_path)
    stem = path.stem
    log.info("Processing image: %s", path.name)

    raw_img = cv2.imread(str(path)) # Renamed for clarity, distinguishing from processed versions
    if raw_img is None:
        log.error("OpenCV failed to read image: %s", path)
        return None

    # Abort on extreme exposure conditions before more complex processing.
    # Note: raw_img.mean() calculates mean across all color channels if present.
    # For a more direct brightness measure, conversion to grayscale first could be considered.
    mean_intensity = raw_img.mean()
    if not (MIN_MEAN_INTENSITY <= mean_intensity <= MAX_MEAN_INTENSITY):
        log.error(
            "Frame exposure (mean intensity: %.1f) out of safe range [%d, %d] – skipped",
            mean_intensity,
            MIN_MEAN_INTENSITY,
            MAX_MEAN_INTENSITY,
        )
        return None

    # --- segmentation & classification ---------------------------------- #
    markers = split_instances(raw_img) # Pass the original (raw) image
    objects = classify_objects(markers)
    total = len(objects)
    log.info("Detected %d objects in %s", total, path.name)

    # --- sanity veto ----------------------------------------------------- #
    if total == 0 or total > MAX_REASONABLE_COUNT:
        log.error(
            "Global veto: unreasonable count (%d) for %s – artefacts NOT written",
            total,
            path.name,
        )
        return None

    # --- Reconstruct a final marker image from accepted objects ---
    final_markers = np.zeros(raw_img.shape[:2], dtype=np.int32)
    for i, obj in enumerate(objects.values(), 1):
        final_markers[obj["mask"] > 0] = i

    # --- write artefacts ------------------------------------------------- #
    # 1. Coloured mask (on black)
    mask_rgb = create_coloured_mask(objects, raw_img.shape)
    mask_file = MASK_DIR / f"{stem}_mask.png"
    cv2.imwrite(str(mask_file), mask_rgb)

    # 2. CSV with object properties
    csv_file = CSV_DIR / f"{stem}.csv"
    _write_csv(csv_file, objects)

    # 3. Text file with just the count
    txt_file = MASK_DIR / f"{stem}_count.txt"
    txt_file.write_text(f"{total}\n", encoding="utf-8")

    # 4. (New) Visual inspection overlay on original image
    overlay_img = create_coloured_overlay(raw_img, final_markers)
    overlay_file = OUT_DIR / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_file), overlay_img)

    # 5. (New) Simple binary mask of all accepted objects
    binary_mask = labels_to_binary(final_markers)
    binary_mask_file = MASK_DIR / f"{stem}_binary.png"
    cv2.imwrite(str(binary_mask_file), binary_mask)

    log.debug("Artefacts saved for %s", stem)
    return total
