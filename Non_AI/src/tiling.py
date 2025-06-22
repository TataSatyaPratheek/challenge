"""
src/tiling.py
-------------
Handles large image processing via overlapping tiles with intelligent stitching.
This approach is crucial for maintaining performance and accuracy on high-resolution
images or images with a very high density of objects, preventing issues like
memory overload or segmentation failure.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from . import config
from .preprocessing import get_binary_threshold
log = logging.getLogger(__name__)


@dataclass
class TileInfo:
    """Metadata for a single tile, including its data and position in the original image."""

    x_start: int
    y_start: int
    x_end: int
    y_end: int
    tile_data: np.ndarray
    tile_id: int


def should_tile_image(
    img: np.ndarray, # Removed max_objects and min_area as they are no longer used
) -> bool:
    """
    Decides if an image requires tiling.

    Forces tiling for very large images, otherwise estimates object count
    using the same thresholding logic as the main segmentation pipeline.
    """
    height, width = img.shape[:2]
    total_pixels = height * width
    total_mp = total_pixels / 1_000_000

    # Force tiling for high-resolution images to prevent memory issues.
    if total_mp > config.HIGH_RES_THRESHOLD_MP:
        log.info(
            f"Large image ({total_mp:.1f}MP) - forcing tiled processing."
        )
        return True

    # With the removal of TILING_THRESHOLD_OBJECTS and related object estimation,
    # this function now only forces tiling for very large images.
    # For smaller images, it returns False.
    log.info(f"Image size: {img.shape[:2]} ({total_mp:.1f}MP) - not forcing tiled processing.")
    return False

def _estimate_object_scale(img: np.ndarray) -> str:
    """Estimate typical object size to choose appropriate tile strategy."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = get_binary_threshold(gray)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # This function is removed as its configuration parameters
    # (TILING_SCALE_MIN_AREA, TILING_SCALE_SMALL_THRESH, TILING_SCALE_MEDIUM_THRESH)
    # have been removed from config.py, indicating a move away from adaptive tile sizing.
    # If this function is still needed, new configuration constants or hardcoded values
    # would be required.
    raise NotImplementedError("Adaptive object scale estimation is deprecated and removed.")


def create_tiles(img: np.ndarray) -> List[TileInfo]:
    """
    Splits a large image into smaller, overlapping tiles with adaptive size
    based on the estimated scale of objects in the image.

    The overlap is essential to ensure objects on tile boundaries are fully
    captured in at least one tile, allowing for accurate reconstruction.
    """
    # Adaptive tile sizing based on object scale has been removed.
    # Using fixed tile size and overlap.
    # These values were previously configurable but are now removed from config.
    tile_size = 256  # Hardcoded, previously config.TILE_SIZE_LARGE_OBJECTS
    overlap = int(tile_size * 0.25) # Hardcoded 25% overlap, previously config.TILE_OVERLAP_RATIO
    log.info(f"Using fixed tile size ({tile_size}px) with {overlap}px overlap")

    overlap = int(tile_size * config.TILE_OVERLAP_RATIO)
    height, width = img.shape[:2]
    tiles = []
    tile_id = 0

    step = tile_size - overlap
    min_tile_dim = tile_size // 3  # More permissive for edge tiles

    for y in range(0, height, step):
        for x in range(0, width, step):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)

            # Skip tiles that are too small to be useful.
            if (x_end - x) < min_tile_dim or (y_end - y) < min_tile_dim:
                continue

            tile_data = img[y:y_end, x:x_end].copy()

            tiles.append(
                TileInfo(
                    x_start=x,
                    y_start=y,
                    x_end=x_end,
                    y_end=y_end,
                    tile_data=tile_data,
                    tile_id=tile_id,
                )
            )
            tile_id += 1

    log.info(
        f"Created {len(tiles)} tiles of size up to {tile_size}x{tile_size} with {overlap}px overlap"
    )
    return tiles


def _get_resolution_scaled_params(img_shape: tuple, is_tile: bool = False) -> dict:
    """Calculate resolution-appropriate parameters."""
    height, width = img_shape[:2]
    megapixels = (height * width) / 1_000_000

    # Only min_contour_area scaling remains. Emergency threshold scaling parameters
    # (BASE_EMERGENCY_THRESHOLD_1MP, EMERGENCY_THRESHOLD_SCALE_FACTOR) have been removed.
    min_contour_area = int(config.BASE_TILE_MIN_CONTOUR_AREA_1MP *
                           (megapixels ** config.MIN_AREA_SCALE_FACTOR))

    emergency_threshold = config.CLASSIFY_HARD_LIMIT_OBJECTS # Use a general hard limit from config

    log.debug(f"Resolution scaling: {megapixels:.1f}MP → min_area={min_contour_area}, "
              f"emergency_threshold={emergency_threshold}")

    return {
        'min_contour_area': min_contour_area,
        'emergency_threshold': emergency_threshold,
        'megapixels': megapixels
    }


def stitch_tile_results(
    tiles: List[TileInfo], tile_markers: List[np.ndarray], full_img_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Stitches segmentation results from individual tiles back into a single marker image.

    It intelligently merges objects that cross tile boundaries by checking for
    overlaps in the stitched canvas.
    """
    # The resolution-appropriate emergency threshold scaling parameters have been removed.
    # Using a general hard limit for the emergency brake.
    params = _get_resolution_scaled_params(full_img_shape, is_tile=True) # Still need megapixels from here for logging
    emergency_threshold = params['emergency_threshold']

    # Pre-process markers: slightly dilate objects touching tile boundaries.
    # This increases the chance of pixel overlap during stitching, helping to
    # correctly merge objects that were split across tiles.
    processed_markers = []
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary_width = 5  # pixels from edge

    for markers in tile_markers:
        if markers is None or markers.max() == 0:
            processed_markers.append(markers)
            continue

        expanded = markers.copy()
        h, w = markers.shape

        for label in np.unique(markers):
            if label == 0:
                continue

            obj_mask = markers == label

            # Check if the object mask touches any of the four tile boundaries.
            touches_boundary = (
                obj_mask[:boundary_width, :].any()
                or obj_mask[h - boundary_width :, :].any()
                or obj_mask[:, :boundary_width].any()
                or obj_mask[:, w - boundary_width :].any()
            )

            if touches_boundary:
                # Dilate the object mask and apply it ONLY to background pixels
                # to avoid overwriting adjacent objects within the same tile.
                dilated_mask = cv2.dilate(obj_mask.astype(np.uint8), dilation_kernel, iterations=1)
                expansion_pixels = (dilated_mask > 0) & (expanded == 0)
                expanded[expansion_pixels] = label

        processed_markers.append(expanded)

    full_markers = np.zeros(full_img_shape[:2], dtype=np.int32)
    global_label = 1
 
    # Emergency brake: Count total objects across all tiles. If it's excessively
    # high, it indicates a noise explosion. Abort to prevent hanging.
    total_raw_objects = sum(len(np.unique(markers)) - 1 for markers in processed_markers if markers is not None)
    if total_raw_objects > emergency_threshold:
        log.error(
            f"EMERGENCY BRAKE: {total_raw_objects} raw objects across tiles exceeds (fixed) "
            f"resolution-scaled limit of {emergency_threshold} ({params['megapixels']:.1f}MP image). "
            f"Returning empty markers to prevent hang."
        )
        return full_markers

    for tile, markers in zip(tiles, processed_markers):
        if markers is None or markers.max() == 0:
            continue

        tile_region = full_markers[tile.y_start : tile.y_end, tile.x_start : tile.x_end]

        # The markers from a tile should have the same dimensions as the tile itself.
        # A resize is unsafe and implies a mismatch in the processing pipeline.
        assert markers.shape == tile_region.shape, (
            f"Marker shape {markers.shape} must match tile region shape {tile_region.shape}"
        )

        # Process each object detected in the current tile
        for local_label in np.unique(markers):
            if local_label == 0:
                continue

            obj_mask = markers == local_label

            # Check for overlap with already placed objects in the full marker map
            # by looking at the pixels in the tile region corresponding to the new object.
            overlapping_pixels = tile_region[obj_mask]
            existing_labels, counts = np.unique(
                overlapping_pixels[overlapping_pixels > 0], return_counts=True
            )

            if existing_labels.size > 0:
                # Overlap detected. Merge with the existing object that has the largest overlap.
                best_label = existing_labels[np.argmax(counts)]
                tile_region[obj_mask] = best_label
            else:
                # No overlap. This is a new object. Assign a new global label.
                tile_region[obj_mask] = global_label
                global_label += 1

    log.info(
        f"Stitched {global_label - 1} objects from {len(tiles)} tiles into a single map."
    )
    return full_markers


def merge_nearby_objects(
    markers: np.ndarray, max_distance: float = 30.0
) -> np.ndarray:
    """
    Merges object fragments that are likely part of the same object but were
    segmented separately. Merging is based on centroid proximity.

    This is a post-processing step to clean up the final stitched mask. It uses
    an efficient graph-based approach to find connected components of nearby objects
    and vectorized NumPy operations for fast relabeling.
    """
    unique_labels = np.unique(markers)
    object_labels = unique_labels[unique_labels != 0]

    if len(object_labels) < 2:
        return markers  # Nothing to merge

    # Emergency brake: If there are too many objects even after stitching,
    # merging (which can be O(n^2)) can be very slow. Skip it.
    # Use the newly added TILING_MERGE_EMERGENCY_THRESHOLD from config.
    if len(object_labels) > config.TILING_MERGE_EMERGENCY_THRESHOLD:
        log.warning(
            f"Skipping merge step: object count ({len(object_labels)}) exceeds "
            f"threshold of {config.TILING_MERGE_EMERGENCY_THRESHOLD}."
        )
        return markers

    # 1. Calculate centroids for all objects
    centroids = {}
    # max_distance is now hardcoded as MERGE_DISTANCE_PIXELS was removed.
    max_distance = 25.0 # Hardcoded, previously config.MERGE_DISTANCE_PIXELS
    ys, xs = np.where(markers == label)
    if ys.size > 0:
        centroids[label] = (xs.mean(), ys.mean())

    if not centroids:
        return markers

    # 2. Build adjacency graph for objects within max_distance
    adj = collections.defaultdict(list)
    labels = list(centroids.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label1, label2 = labels[i], labels[j]
            x1, y1 = centroids[label1]
            x2, y2 = centroids[label2]

            dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if dist_sq <= max_distance**2:
                adj[label1].append(label2)
                adj[label2].append(label1)

    # 3. Find connected components (groups to merge) using BFS
    merge_map = {}
    visited = set()
    for label in labels:
        if label not in visited:
            component_root = label
            q = collections.deque([label])
            visited.add(label)
            component_nodes = []
            while q:
                curr = q.popleft()
                component_nodes.append(curr)
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            for node in component_nodes:
                merge_map[node] = component_root

    if not merge_map:
        return markers

    # 4. Apply the merges using a fast lookup table
    lookup = np.arange(markers.max() + 1, dtype=np.int32)
    for old_label, new_label in merge_map.items():
        lookup[old_label] = new_label
    merged_markers = lookup[markers]

    # 5. Renumber labels to be consecutive (1, 2, 3, ...) for cleanliness
    final_labels = np.unique(merged_markers)

    relabel_lookup = np.zeros(final_labels.max() + 1, dtype=np.int32)
    new_id = 1
    for old_id in final_labels:
        if old_id == 0:
            continue
        relabel_lookup[old_id] = new_id
        new_id += 1

    final_markers = relabel_lookup[merged_markers]

    log.info(f"Merged nearby objects: {len(labels)} initial -> {new_id - 1} final.")
    return final_markers
