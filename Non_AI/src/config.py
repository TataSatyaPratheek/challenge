"""
src/config.py  –  central configuration & logging

This file centralizes all tunable parameters, constants, and path
configurations for the object counting pipeline. It also handles the initial
setup of logging and output directories. Using a central config file makes it
easier to adjust the behavior of the pipeline without modifying the core logic.
"""

from __future__ import annotations

import logging
import os
import pathlib as _pl

# --------------------------------------------------------------------------- #
# I/O paths – project root is the parent of the “src” directory
# --------------------------------------------------------------------------- #
ROOT = _pl.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
MASK_DIR = OUT_DIR / "masks"
CSV_DIR = OUT_DIR / "csv"
LOG_DIR = OUT_DIR / "logs"

for d in (MASK_DIR, CSV_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- CLI / Evaluation ---------------------------------------------------- #
IMAGE_GLOB_PATTERN: str = "*.jpg"
GROUND_TRUTH_FILENAME: str = "ground_truth.json"
SUMMARY_FILENAME_PREFIX: str = "summary_"

# --------------------------------------------------------------------------- #
# General Processing Parameters
# --------------------------------------------------------------------------- #

# ---- Preprocessing & Sanity Checks --------------------------------------- #
MIN_MEAN_INTENSITY: int = 30
MAX_MEAN_INTENSITY: int = 230
ADAPTIVE_THRESH_BLOCK_SIZE: int = 31
ADAPTIVE_THRESH_C: int = 5
OTSU_FAIL_MEAN_LOW: int = 10
OTSU_FAIL_MEAN_HIGH: int = 245

# ---- Classification & Vetoes --------------------------------------------- #
CLASSIFY_HARD_LIMIT_OBJECTS: int = 5000  # Abort if segmentation produces more objects than this
MAX_REASONABLE_COUNT: int = 2000   # Global veto for unreasonable counts
# ---- Resolution-Aware Segmentation --------------------------------------- #
# Base values for 1MP images - will be scaled by resolution
BASE_MIN_CONTOUR_AREA_1MP: int = 20        # For full-image segmentation
BASE_TILE_MIN_CONTOUR_AREA_1MP: int = 30   # For tiled segmentation
# Scaling factors based on image resolution
MIN_AREA_SCALE_FACTOR: float = 0.8         # Scale min area by (MP^0.8)
MAX_AREA_SCALE_FACTOR: float = 1.2         # Scale max area by (MP^1.2)
# ---- Resolution-Aware Downsampling --------------------------------------- #
DOWNSAMPLE_TARGET_MP: float = 3.0        # Target resolution for downsampling high-res images.

# ---- High-Resolution Processing ------------------------------------------ #
HIGH_RES_THRESHOLD_MP: int = 6        # Above this, use downsample-process-upsample

# Fixed bounds for all image types (no more dynamic percentiles)
FIXED_MIN_OBJECT_AREA: int = 50       # Minimum object size (all resolutions)
FIXED_MAX_OBJECT_AREA: int = 15000    # Maximum object size (all resolutions)

# Fix missing constant that was breaking the merge step
TILING_MERGE_EMERGENCY_THRESHOLD: int = 1000  # Legacy constant for old code

# ---- Artefact Generation ------------------------------------------------- #
OVERLAY_ALPHA: float = 0.5
HUE_GOLDEN_RATIO_DEGREES: int = 137
OVERLAY_SATURATION: int = 200
OVERLAY_VALUE: int = 255
MASK_SATURATION: int = 255
MASK_VALUE: int = 255

# ---- Mask Smoothing Utilities -------------------------------------------- #
SMOOTH_MASK_CLOSE_KERNEL: tuple[int, int] = (3, 3)
SMOOTH_MASK_BLUR_KSIZE: int = 5
SMOOTH_MASK_ITERATIONS: int = 2
SMOOTH_MASK_THRESH: int = 127

# ---- Logging ------------------------------------------------------------- #
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


setup_logging()
