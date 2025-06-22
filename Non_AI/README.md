# Object Counter and Classifier

This project processes images to detect, count, and classify objects like screws, nuts, and bolts. It features an adaptive pipeline that can handle images with high object density by automatically switching to a tiling-based approach for robust segmentation.

## Overview

The system takes an input image (or a directory of images), performs preprocessing, segments individual objects, classifies each object, and then outputs the results in various formats including a summary log, individual CSV files for each image, and visual mask images.

Key processing steps include:
1.  **Image Analysis**: The system first performs a quick check on image properties like brightness and object density.
2.  **Adaptive Segmentation**:
    - For images with low-to-moderate object density, it uses a direct contour detection method on the entire image.
    - For images with high object density (e.g., many small parts), it automatically switches to a **tiling strategy**. The image is split into overlapping tiles, each is processed independently, and the results are intelligently stitched back together. This prevents performance issues and improves accuracy in crowded scenes.
3.  **Classification**: After segmentation, each potential object is validated based on its area. The current implementation performs a general "object" classification, with infrastructure in place for more specific types (nuts, bolts, etc.).

## Features

- Process single JPG images or all JPG images within a specified directory.
- **Adaptive Tiling**: Automatically handles dense images to ensure robust performance.
- Generate individual CSV reports for each processed image, detailing each object's ID, type, area, and circularity.
- Save multiple visual artifacts for inspection:
    - Colored masks on a black background.
    - Colored overlays on the original image.
    - Simple binary masks.
- Produce a JSON summary log for each run, detailing the number of objects found in each image.
- Configurable processing parameters via `config.py`.
- Detailed logging for both Python and OpenCV operations.
- Built-in evaluation script to compare results against a ground truth file.

## Project Structure

The main components are:

- `cli.py`: Command-line interface for running the image processing pipeline.
- `counter.py`: Orchestrates the image processing steps for a single image.
- `segmentation.py`: Implements object instance segmentation.
- `preprocess.py`: Contains functions for image preparation like thresholding.
- `tiling.py`: Provides the core functions for creating, processing, and stitching image tiles.
- `classify_mask.py`: Classifies segmented objects and generates colored masks.
- `config.py`: Contains configuration parameters for paths, preprocessing, segmentation, classification, and logging.
- `eval_counts.py`: Compares pipeline output to a ground-truth JSON file.

**Output Directories (created automatically relative to the project root, one level above `src`):**

- `outputs/masks/`: Stores generated mask images (`<image_name>_mask.png`).
- `outputs/csv/`: Stores CSV files with details of detected objects for each image (`<image_name>.csv`).
- `outputs/logs/`: Stores JSON summary logs for each run (`run_YYYYMMDD_HHMMSS.json`).

## Prerequisites

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)

You can typically install these dependencies using pip:
```bash
pip install opencv-python numpy
```
It's recommended to use a virtual environment.

## Usage

To run the object counter and classifier, execute the `cli.py` script from the **project root directory** (the directory containing the `src` folder).

**Command:**
```bash
python -m src.cli --path <path_to_image.jpg | path_to_directory>
```

**Arguments:**

- `--path` (required):
    - Path to a single `.jpg` image file.
    - Path to a directory containing `.jpg` image files. The script will process all JPGs in this directory.

**Examples:**

1.  Process a single image:
    ```bash
    python -m src.cli --path /path/to/your/image.jpg
    ```

2.  Process all JPG images in a directory:
    ```bash
    python -m src.cli --path /path/to/your/images_directory/
    ```

## Configuration

Various parameters for the image processing pipeline can be adjusted in the `src/config.py` file. Key parameters include:

- **Preprocessing:**
    - `GAUSS_KERNEL`: Kernel size for Gaussian blur.
    - `THRESH_METHOD`: Thresholding method (currently "otsu").
- **Segmentation:**
    - `DIST_RATIO`: Ratio for distance transform thresholding to determine sure foreground.
    - `MIN_AREA`: Minimum area (in pixels) for an object to be considered valid.
- **Classification:**
    - `WASHER_CIRCULARITY_THRESHOLD`: Circularity value above which an object (if it also has a hole or meets this threshold) is classified as a washer.
- **Logging:**
    - `LOG_LEVEL`: Sets the logging verbosity (e.g., "INFO", "DEBUG"). Can also be set via the `LOG_LEVEL` environment variable.

## Output

Upon successful execution, the script will generate the following outputs in the `outputs` directory (relative to the project root):

1.  **Mask Images:**
    - Location: `outputs/masks/`
    - Filename: `<original_image_name>_mask.png`
    - Description: A PNG image where each detected and classified object is colored differently.

2.  **CSV Reports:**
    - Location: `outputs/csv/`
    - Filename: `<original_image_name>.csv`
    - Description: A CSV file for each processed image, containing:
        - `id`: A unique identifier for the object within the image.
        - `class`: The classified type of the object ("washer" or "screw").
        - `pixel_count`: The number of pixels belonging to the object.

3.  **Run Summary Log:**
    - Location: `outputs/logs/`
    - Filename: `run_YYYYMMDD_HHMMSS.json` (timestamped)
    - Description: A JSON file summarizing the run, mapping each processed image name to the number of objects found in it.

Console output will also provide information on the processing steps, errors, and a final summary including total images processed, objects found, and execution time.

## How It Works (Briefly)

1.  **Image Loading**: The input image is loaded.
2.  **Preprocessing (`preprocess.py`)**:
3.  **Segmentation (`segmentation.py` & `tiling.py`)**:
    - The system estimates object density to decide whether to use the standard or tiled approach.
    - **Standard Path**: The image is converted to grayscale, and contours are detected to create an initial marker map.
    - **Tiled Path**: The image is divided into overlapping tiles. Contour detection is run on each tile. The resulting markers are stitched into a full-size map, and object fragments across tile boundaries are merged.
4.  **Classification (`classify_mask.py`)**:
    - The raw markers from segmentation are processed.
    - Contours are extracted for each potential object to get an accurate area.
    - An adaptive area filter is applied, removing objects that are too small or too large relative to the main distribution of object sizes in the image.
    - A dictionary of accepted objects and their properties is created.
5.  **Output Generation (`counter.py`, `mask_utils.py`)**:
    - A final marker map is reconstructed from the accepted objects.
    - Various artifacts are saved: a colored mask, a colored overlay on the original image, a binary mask, and a text file with the final count.
    - A CSV file is written with details for each instance.
    - The total count of instances for the image is returned.
6.  **Reporting (`cli.py`)**:
    - A summary JSON file for the entire run is saved.
    - Overall statistics are logged to the console.