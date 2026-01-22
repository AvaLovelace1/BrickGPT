# Objaverse & Mesh2Brick Evaluation Tools

This folder contains scripts to download 3D models from Objaverse, convert them into LEGO brick structures using `mesh2brick`, and visualize the results in a comparison gallery.

## Pipeline Workflow

### 1. Download Assets

Downloads a subset of Objaverse models specified in the CSV file.

- **Script:** `download_obj.py`
- **Input:** `objaversepp_small.csv`
- **Output:** Saves `.glb` files to `assets/`.
- **Usage:**
  ```bash
  uv run objaverse/download_obj.py
  ```

### 2. Convert to Bricks

Converts all `.glb` files in the `assets/` folder into LEGO structures.

- **Script:** `mesh2brick_objaverse.py`
- **Output:** Generates `.txt` (brick list) and `.ldr` (LDraw) files in `assets/`.
- **Usage:**
  ```bash
  uv run objaverse/mesh2brick_objaverse.py
  ```

### 3. Generate Gallery

Renders the LEGO models and creates an HTML page to compare the original 3D model vs. the LEGO conversion side-by-side.

- **Script:** `generate_gallery.py`
- **Output:**
  - Renders `.png` images in `assets/`.
  - Generates `gallery.html`.
- **Usage:**
  ```bash
  uv run objaverse/generate_gallery.py
  ```

## Directory Structure

- `assets/`: Stores all model files (`.glb`, `.ldr`, `.txt`, `.png`).
- `objaversepp_small.csv`: List of Objaverse UIDs to process (downloaded from Objaverse++).
