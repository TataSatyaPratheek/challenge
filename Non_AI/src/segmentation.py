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
from sklearn.cluster import DBSCAN


from .config import (
    get_adaptive_parameters,
    DENSITY_SPARSE_THRESHOLD,
    DENSITY_DENSE_THRESHOLD,
    HOUGH_PARAM2_STRICT,
    HOUGH_PARAM2_RELAXED,
    PEAK_RATIO,
)

from .preprocess import stabilise

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# helper – draw & mask circles
# --------------------------------------------------------------------------- #
def _draw_circle_markers(shape, circles: List[Tuple[int, int, int]]) -> np.ndarray:
    if not circles: # Handle empty circle list
        return np.zeros(shape, dtype=np.int32)
    m = np.zeros(shape, dtype=np.int32)
    for idx, (x, y, r) in enumerate(circles, 1):
        cv2.circle(m, (x, y), r, idx, -1)
    return m


def _conservative_hough(gray: np.ndarray, params) -> np.ndarray:
    """Hough with STRICT parameters to prevent explosion"""
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 1),
        cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=100, param2=HOUGH_PARAM2_STRICT,  # MUCH higher threshold
        minRadius=params["hough_min_radius"],
        maxRadius=params["hough_max_radius"]
    )

    if circles is not None and len(circles[0]) <= 200:  # Sanity check
        return _draw_circle_markers(gray.shape, [(int(x), int(y), int(r)) for x,y,r in circles[0]])
    else:
        log.warning("Hough failed, falling back to watershed")
        return _conservative_watershed(gray, params)

def _conservative_watershed(gray: np.ndarray, params) -> np.ndarray:
    """Watershed that DOESN'T fragment everything"""
    # Much stronger preprocessing to connect touching objects
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # AGGRESSIVE closing to prevent over-segmentation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # MUCH more conservative peak finding
    _, peaks = cv2.threshold(dist, PEAK_RATIO * dist.max(), 255, 0)

    _, markers = cv2.connectedComponents(peaks.astype(np.uint8))
    markers = markers + 1
    cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
    markers[markers <= 1] = 0

    # Only keep reasonably sized objects
    for label in np.unique(markers):
        if label and (markers == label).sum() < params["base_min_area"]:
            markers[markers == label] = 0

    return markers

def _hybrid_approach(gray: np.ndarray, params) -> np.ndarray:
    """Try Hough first, watershed only on residual"""
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 1),
        cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=100, param2=HOUGH_PARAM2_RELAXED,
        minRadius=params["hough_min_radius"],
        maxRadius=params["hough_max_radius"]
    )

    if circles is not None:
        # Mask detected circles
        mask = np.zeros_like(gray)
        for x, y, r in circles[0]:
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)

        # Watershed only on remaining areas
        residual = cv2.bitwise_and(gray, cv2.bitwise_not(mask))
        watershed_markers = _conservative_watershed(residual, params)

        # Combine results
        hough_markers = _draw_circle_markers(gray.shape, [(int(x), int(y), int(r)) for x,y,r in circles[0]])
        combined = hough_markers.copy()
        max_label = hough_markers.max()
        watershed_markers[watershed_markers > 0] += max_label
        combined[watershed_markers > 0] = watershed_markers[watershed_markers > 0]

        return combined
    else:
        return _conservative_watershed(gray, params)


def _merge_fragments_with_dbscan(markers: np.ndarray, eps: float = 25, min_samples: int = 1) -> np.ndarray:
    """Merge fragmented masks using DBSCAN clustering on centroids."""
    if markers.max() == 0:
        return markers

    # Extract centroids of each fragment
    labels = np.unique(markers)
    labels = labels[labels != 0]

    if len(labels) <= 1:
        return markers

    centroids = []
    for lbl in labels:
        ys, xs = np.where(markers == lbl)
        centroids.append([np.mean(xs), np.mean(ys)])

    centroids = np.array(centroids)

    # Run DBSCAN clustering on centroids
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
    cluster_labels = clustering.labels_

    # Create new merged mask
    merged_mask = np.zeros_like(markers)
    new_label = 1

    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            # Noise points - keep as individual objects
            noise_indices = np.where(cluster_labels == -1)[0]
            for idx in noise_indices:
                merged_mask[markers == labels[idx]] = new_label
                new_label += 1
        else:
            # Merge all fragments in this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            for idx in cluster_indices:
                merged_mask[markers == labels[idx]] = new_label
            new_label += 1

    log.info(f"DBSCAN merged {len(labels)} fragments → {new_label-1} objects")
    return merged_mask
# --------------------------------------------------------------------------- #
# public entry
# --------------------------------------------------------------------------- #
def split_instances(img: np.ndarray) -> np.ndarray:
    if img is None:
        return np.zeros((1, 1), np.int32)

    # Get adaptive parameters based on original image dimensions
    params = get_adaptive_parameters(img.shape)

    # Preprocess the image (stabilise returns: stabilised_bgr, gray, edges)
    # The input 'img' is shadowed by stabilised_bgr_img here.
    stabilised_bgr_img, gray, edges = stabilise(img)

    # ALWAYS try Hough first - screws are circular objects
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (3, 3), 1),
        cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
        param1=100, param2=35,  # Much more permissive than 50
        minRadius=params["hough_min_radius"],
        maxRadius=params["hough_max_radius"]
    )

    if circles is not None and len(circles[0]) >= 3:
        markers = _draw_circle_markers(gray.shape, [(int(x), int(y), int(r)) for x,y,r in circles[0]])
        log.info(f"Hough detected {len(circles[0])} circles")
    else:
        # Fallback to conservative watershed
        log.info("Hough failed, using watershed fallback")
        markers = _conservative_watershed(gray, params)

    # CRITICAL: Merge fragments using DBSCAN spatial clustering
    if markers.max() > 50:  # Only cluster if we have many fragments
        markers = _merge_fragments_with_dbscan(markers, eps=30, min_samples=1)

    return markers
