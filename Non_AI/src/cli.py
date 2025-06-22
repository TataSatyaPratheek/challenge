# src/cli.py
import argparse
import pathlib
import time
import json
import logging
from .counter import process_image
from .eval_counts import run_evaluation
from .config import (
    GROUND_TRUTH_FILENAME,
    IMAGE_GLOB_PATTERN,
    LOG_DIR,
    SUMMARY_FILENAME_PREFIX,
)

log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Count objects in images using OpenCV")
    parser.add_argument("--path", required=True, help="Image file or directory")
    args = parser.parse_args()
    
    input_path = pathlib.Path(args.path)
    
    # Collect image paths
    if input_path.is_dir():
        image_paths = sorted(input_path.glob(IMAGE_GLOB_PATTERN))
        log.info("Processing %d images from directory: %s", len(image_paths), input_path)
    elif input_path.is_file():
        image_paths = [input_path]
        log.info("Processing single image: %s", input_path)
    else:
        log.error("Path not found: %s", input_path)
        return
    
    if not image_paths:
        log.warning("No images matching '%s' found at: %s", IMAGE_GLOB_PATTERN, input_path)
        return
    
    # Process images
    results = {}
    start_time = time.time()
    
    for img_path in image_paths:
        try:
            count = process_image(img_path)
            if count is not None:
                results[img_path.name] = count
        except Exception as e:
            log.error("Error processing %s: %s", img_path.name, e, exc_info=True)
    
    # Save summary
    if results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = LOG_DIR / f"{SUMMARY_FILENAME_PREFIX}{timestamp}.json"
        summary_path.write_text(json.dumps(results, indent=2))
        
        total_objects = sum(results.values())
        elapsed = time.time() - start_time
        
        log.info("Summary: %d images, %d objects total, %.1fs elapsed", 
                len(results), total_objects, elapsed)
        log.info("Summary saved to: %s", summary_path)

        # --- auto-evaluation if a GT file is present ----
        gt_dir = input_path if input_path.is_dir() else input_path.parent
        gt_json_path = gt_dir / GROUND_TRUTH_FILENAME
        if gt_json_path.exists():
            log.info("Found ground_truth.json – running evaluation.")
            run_evaluation(gt_path=gt_json_path, pred_path=summary_path)

if __name__ == "__main__":
    main()
