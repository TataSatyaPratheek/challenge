# src/__init__.py
"""
Object counting package using OpenCV computer vision techniques.
"""

from . import config
from .counter import process_image
from .cli import main as run_cli

__all__ = ["config", "process_image", "run_cli"]

import logging
log = logging.getLogger(__name__)
log.info("OpenCV object counting package loaded")
