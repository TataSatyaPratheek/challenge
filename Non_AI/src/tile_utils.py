# src/tile_utils.py
#
# Utility functions for fixed-square tiling and detection merging.
# Pure-NumPy implementation (no cv2 dependency) so it runs even on minimal
# headless environments.

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Tile generation
# --------------------------------------------------------------------------- #
def generate_tiles(
    img: np.ndarray, tile_size: int = 512, overlap: float = 0.3
) -> List[Dict]:
    """
    Slice *img* into overlapping square tiles.

    Returns a list of dictionaries:
        {
            "tile"         : ndarray(H, W, C),
            "global_offset": (x0, y0),   # top-left corner of tile in original img
            "tile_id"      : "x_y"       # string id for debugging
        }
    """
    assert 0.0 <= overlap < 1.0, "overlap must be in [0,1)"
    tiles: List[Dict] = []

    step = int(tile_size * (1.0 - overlap)) or 1
    H, W = img.shape[:2]

    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            x1, y1 = min(x0 + tile_size, W), min(y0 + tile_size, H)
            tile = img[y0:y1, x0:x1]
            tiles.append(
                {
                    "tile": tile,
                    "global_offset": (x0, y0),
                    "tile_id": f"{x0}_{y0}",
                }
            )
            # break inner loop if we reached right border
            if x1 == W:
                break
        # break outer loop if we reached bottom border
        if y1 == H:
            break
    return tiles


# --------------------------------------------------------------------------- #
# 2. Non-maximum suppression (greedy IoU)
# --------------------------------------------------------------------------- #
def _bbox_from_circle(x: int, y: int, r: int) -> Tuple[float, float, float, float]:
    return x - r, y - r, x + r, y + r


def _iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]):
    # Intersection-over-union of two axis-aligned boxes
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter)


def apply_nms(detections: List[Dict], nms_threshold: float = 0.5) -> List[Dict]:
    """
    Greedy IoU-based NMS.  Each detection is a dict with keys:
        { "x": int, "y": int, "radius": int, "confidence": float (optional) }
    """
    if not detections:
        return []

    # sort by confidence (default 1.0)
    detections = sorted(detections, key=lambda d: d.get("confidence", 1.0), reverse=True)

    keep: List[Dict] = []
    while detections:
        best = detections.pop(0)
        keep.append(best)

        b_best = _bbox_from_circle(best["x"], best["y"], best["radius"])

        remaining = []
        for det in detections:
            b_det = _bbox_from_circle(det["x"], det["y"], det["radius"])
            if _iou(b_best, b_det) <= nms_threshold:
                remaining.append(det)
        detections = remaining

    return keep


# --------------------------------------------------------------------------- #
# 3. Merge tile-local detections to global coordinates + NMS
# --------------------------------------------------------------------------- #
def merge_tile_results(
    tile_results: List[Dict], nms_threshold: float = 0.5
) -> List[Dict]:
    """
    Convert tile-relative detections to global coords and run NMS.

    Each *tile_results* element:
        {
            "global_offset": (x0, y0),
            "detections"   : [ {x, y, radius, confidence?}, ... ]
        }
    Returns a list with global x,y.
    """
    global_dets: List[Dict] = []
    for tr in tile_results:
        off_x, off_y = tr["global_offset"]
        for det in tr["detections"]:
            global_dets.append(
                {
                    "x": det["x"] + off_x,
                    "y": det["y"] + off_y,
                    "radius": det["radius"],
                    "confidence": det.get("confidence", 1.0),
                }
            )

    return apply_nms(global_dets, nms_threshold=nms_threshold)
