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
WASHER_CIRCULARITY_THRESHOLD: float = 0.75  # higher → stricter washer decision
DEDUP_RADIUS_FACTOR: float = 0.60   # consider same circle if |r1-r2| ≤ 0.6·min(r)
# ---- new object-type / nut parameters ----------------------------------- #
HOLE_RATIO_LOW: float = 1.8    # outer / inner area ratio for nuts
HOLE_RATIO_HIGH: float = 3.5
# number of approxPolyDP vertices for a “hex-ish” outline
HEX_VERT_TOL: int = 1      # |verts-6| ≤ 1  ⇒ likely hex

# ---------------- extra HQ tunables ---------------- #
MIN_CIRC_HQ: float = 0.30   # lower bound for circularity on HQ images
MAX_CIRC_HQ: float = 0.90   # upper bound (reject perfect rings of noise)

# Global veto used by counter.py
MAX_REASONABLE_COUNT: int = 400


# --- guard-rail & relaxed Hough (used by segmentation.py) ---
FALLBACK_MAX_CONTOURS: int = 2_000
RELAXED_PARAM2: int = 30        # accumulator threshold (~15 lower than strict)
RELAXED_MIN_DIST: int = 15      # pixels between circle centres in relaxed pass

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
            "base_min_area": 15,
            "base_max_area": 1_000,
            "min_circularity": 0.25,
        }

    # ---------- HIGH-QUALITY (> 3 MP) ----------
    else:
        params = {
            "pipeline": "high_quality_detection",
            "base_min_area": 100,
            "base_max_area": 15_000,
            "min_circularity": MIN_CIRC_HQ,
            # strict Hough pass parameters
            "max_circularity_hq": MAX_CIRC_HQ,
            "hough_dp": 1.2,
            "hough_min_dist": 30,
            "hough_param1": 100,
            "hough_param2": 45,      # strict
            "hough_min_radius": 20,
            "hough_max_radius": 120,
        }

    # attach global tunables expected by segmentation.py
    params["fallback_max_contours"] = FALLBACK_MAX_CONTOURS
    params["relaxed_param2"] = RELAXED_PARAM2
    params["relaxed_min_dist"] = RELAXED_MIN_DIST

    return params
