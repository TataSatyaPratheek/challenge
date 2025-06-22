"""
src/config.py  –  central configuration & logging

Changes in this “revamp”:

1.  Adds guard-rail and relaxed-Hough tunables
       • FALLBACK_MAX_CONTOURS  – abort fallback if contour count explodes
       • RELAXED_PARAM2         – lower accumulator for 2nd Hough sweep
       • RELAXED_MIN_DIST       – shorter minDist for 2nd sweep
2.  Returns those tunables as part of `get_adaptive_parameters`
3.  Uses pathlib to create the outputs/ tree on first import
"""

from __future__ import annotations

import logging
import os
import pathlib as _pl
from typing import Dict, Any

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

# --------------------------------------------------------------------------- #
# Constants / tunables
# --------------------------------------------------------------------------- #
DEDUP_RADIUS_FACTOR: float = 0.60   # consider same circle if |r1-r2| ≤ 0.6·min(r)

# ---- Input stabilisation ------------------------------------------------- #
CLAHE_CLIP          : float = 2.0   # contrast‐limited adaptive equalisation
AUTO_CANNY_LOW      : int   = 10    # will be halved if frame is dark
AUTO_CANNY_HIGH     : int   = 30

# ---- Preprocessing (stabilise) ------------------------------------------- #
PREPROC_DARK_THRESH      : int   = 50    # Mean intensity below which to apply correction for severely dark images
PREPROC_BRIGHTEN_ALPHA   : float = 1.2   # Contrast correction factor (1.0 = no change)
PREPROC_BRIGHTEN_BETA    : int   = 15    # Brightness correction value
PREPROC_BLUR_KERNEL_SIZE : int   = 3     # Gaussian blur kernel size (must be odd)
PREPROC_BLUR_SIGMA       : float = 0.5   # Minimal Gaussian blur sigma for Canny
PREPROC_CANNY_LOW        : int   = 30    # Conservative Canny edge detection lower threshold
PREPROC_CANNY_HIGH       : int   = 100   # Conservative Canny edge detection upper threshold
# ---- Shape classification (nuts, bolts, screws) ------------------------- #
# HOLE_RATIO is inner_area / outer_area for nuts
HOLE_RATIO_LOW      : float = 0.25
HOLE_RATIO_HIGH     : float = 0.40
# number of approxPolyDP vertices for a “hex-ish” outline
HEX_VERT_TOL: int = 1      # |verts-6| ≤ 1  ⇒ likely hex
BOLT_ASPECT_MIN: float = 2.5  # min aspect ratio (max(w,h)/min(w,h)) for an elongated object to be a bolt

# ---------------- circularity floors ---------------- #
# Typical pan-head screws in the provided dataset give circularity
# 0.12 ± 0.03, so use that as a universal minimum.
MIN_CIRC_HQ:    float = 0.1   # HQ pipeline (≥ 3 MP)
MIN_CIRC_SMALL: float = 0.1   # small-object pipeline (≤ 3 MP)

MAX_CIRC_HQ: float = 0.90   # upper bound (reject perfect rings of noise)

# Global veto used by counter.py
MAX_REASONABLE_COUNT: int = 400


# --- NEW: Density-based routing (objects per 1000x1000 pixel area) ---
DENSITY_SPARSE_THRESHOLD = 5    # ≤5 objects per 1000x1000 → Hough
DENSITY_DENSE_THRESHOLD  = 50   # ≥50 objects per 1000x1000 → watershed

# --- guard-rail & relaxed Hough (used by segmentation.py) ---
FALLBACK_MAX_CONTOURS: int = 2_000
RELAXED_PARAM2: int = 30        # accumulator threshold (~15 lower than strict)
RELAXED_MIN_DIST: int = 15      # pixels between circle centres in relaxed pass

# --- FIXED Hough parameters (no more 1500 circle explosions) ---
HOUGH_PARAM2_STRICT = 35   # More permissive - was missing 95% of screws
HOUGH_PARAM2_RELAXED = 25  # Even more permissive for fallback

# ---- Watershed separation ------------------------------------------------ #
PEAK_RATIO          : float = 0.85  # Much more conservative watershed
WAT_MIN_AREA_SMALL  : int   = 200    # min pix² for nut fragments
BOLT_MIN_AREA_FACTOR: float = 1.5 # Factor to distinguish larger bolts from smaller screws

# DBSCAN clustering for fragment merging
DBSCAN_EPS: float = 30.0     # Pixel distance for merging fragments
DBSCAN_MIN_SAMPLES: int = 1  # Allow single-fragment clusters


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


setup_logging()

# --------------------------------------------------------------------------- #
# Adaptive parameter provider
# --------------------------------------------------------------------------- #
def get_adaptive_parameters(img_shape: tuple[int, ...]) -> Dict[str, Any]:
    """
    Decide which pipeline to use and return all numeric knobs required by the
    rest of the code-base.

    Returns a dict that *always* contains:
        • pipeline                (str)
        • base_min_area / base_max_area
        • min_circularity
        • max_circularity_hq      (optional, only for HQ pipeline)
        • fallback_max_contours   (guard-rail)
        • relaxed_param2 / relaxed_min_dist (for stage-2 Hough)

    For the HQ pipeline extra Hough keys are included.
    """
    height, width = img_shape[:2]
    total_px = height * width

    # ---------- SMALL / LOW-RES  (≤ 3 MP) ----------
    if total_px <= 3_000_000:
        params: Dict[str, Any] = {
            "pipeline": "small_object_recovery",
            "base_min_area": 100,
            "base_max_area": 2_000,
            # ↓ new, taken from MIN_CIRC_SMALL
            "min_circularity": MIN_CIRC_SMALL,
            # Add missing Hough parameters for small object recovery
            "hough_dp": 1.5,
            "hough_min_dist": 40,        # Stricter hough distance
            "hough_param1": 70,          # Canny edge detector upper threshold
            "hough_param2": 55,          # Stricter accumulator threshold
            "hough_min_radius": 5,       # Min radius for small objects
            "hough_max_radius": 50,      # Max radius for small objects
        }

    # ---------- HIGH-QUALITY (> 3 MP) ----------
    else:
        params = {
            "pipeline": "high_quality_detection",
            "base_min_area": 400,
            "base_max_area": 8_000,
            # ↓ already 0.12 from edit above
            "min_circularity": MIN_CIRC_HQ,
            # strict Hough pass parameters
            "max_circularity_hq": MAX_CIRC_HQ,
            "hough_dp": 1.2,
            "hough_min_dist": 60,
            "hough_param1": 100,
            "hough_param2": 70,      # strict
            "hough_min_radius": 20,
            "hough_max_radius": 120,
        }

    # attach global tunables expected by segmentation.py
    params["fallback_max_contours"] = FALLBACK_MAX_CONTOURS
    params["relaxed_param2"] = RELAXED_PARAM2
    params["relaxed_min_dist"] = RELAXED_MIN_DIST

    return params
