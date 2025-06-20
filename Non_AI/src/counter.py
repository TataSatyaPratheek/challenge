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
import logging
import pathlib
from typing import Optional

import cv2

from .classify_mask import classify_objects, create_colored_mask
from .config import CSV_DIR, MASK_DIR, MAX_REASONABLE_COUNT
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

    img = cv2.imread(str(path))
    if img is None:
        log.error("OpenCV failed to read image: %s", path)
        return None

    # --- segmentation & classification ---------------------------------- #
    markers = split_instances(img)
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

    # --- write artefacts ------------------------------------------------- #
    mask_rgb = create_colored_mask(objects, img.shape)
    mask_file = MASK_DIR / f"{stem}_mask.png"
    csv_file = CSV_DIR / f"{stem}.csv"
    txt_file = MASK_DIR / f"{stem}_count.txt"

    cv2.imwrite(str(mask_file), mask_rgb)
    _write_csv(csv_file, objects)
    txt_file.write_text(f"{total}\n", encoding="utf-8")

    log.debug("Artefacts saved: %s  %s", mask_file.name, csv_file.name)
    return total
