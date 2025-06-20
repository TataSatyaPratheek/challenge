# Object Counter and Classifier

This project processes images to detect, count, and classify objects, specifically distinguishing between "washers" and "screws" based on their geometric properties.

## Overview

The system takes an input image (or a directory of images), performs preprocessing, segments individual objects, classifies each object, and then outputs the results in various formats including a summary log, individual CSV files for each image, and visual mask images.

Key processing steps include:
1.  **Preprocessing**: Converts the image to grayscale, applies Gaussian blur, and then binarizes it using Otsu's thresholding.
2.  **Segmentation**: Uses Distance Transform and Watershed algorithm to separate touching or overlapping objects.
3.  **Classification**: Identifies objects as "washer" or "screw" based on contour circularity and the presence of holes.

## Features

- Process single JPG images or all JPG images within a specified directory.
- Classify detected objects into "washer" or "screw".
- Generate individual CSV reports for each processed image, detailing each object's ID, class, and pixel count.
- Save colored mask images for visual inspection of detected and classified objects.
- Produce a JSON summary log for each run, detailing the number of objects found in each image.
- Configurable processing parameters via `config.py`.
- Detailed logging for both Python and OpenCV operations.

## Project Structure

The main components are:

- `cli.py`: Command-line interface for running the image processing pipeline.
- `counter.py`: Orchestrates the image processing steps for a single image.
- `preprocess.py`: Handles image binarization.
- `segmentation.py`: Implements object instance segmentation.
- `classify_mask.py`: Classifies segmented objects and generates colored masks.
- `config.py`: Contains configuration parameters for paths, preprocessing, segmentation, classification, and logging.

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
    - The image is converted to grayscale.
    - Gaussian blur is applied to reduce noise.
    - Otsu's thresholding is used to create a binary mask.
3.  **Segmentation (`segmentation.py`)**:
    - The binary mask undergoes a distance transform.
    - The distance map is thresholded to find "sure foreground" regions.
    - The Watershed algorithm is applied to segment individual object instances.
4.  **Classification & Tagging (`classify_mask.py`)**:
    - For each segmented instance:
        - Contour area and perimeter are calculated.
        - Circularity is computed.
        - The presence of holes is checked.
        - Based on `MIN_AREA`, circularity (`WASHER_CIRCULARITY_THRESHOLD`), and hole presence, the object is classified as "washer" or "screw".
        - A dictionary of classified instances and their masks is created.
5.  **Output Generation (`counter.py`, `classify_mask.py`)**:
    - A colored mask image is generated where each instance has a unique color.
    - A CSV file is written with details for each instance.
    - The total count of instances for the image is returned.
6.  **Reporting (`cli.py`)**:
    - A summary JSON file for the entire run is saved.
    - Overall statistics are logged to the console.