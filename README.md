# Mouse Video Annotation Tool

A powerful PyQt5-based desktop pipeline application to load `.avi` videos and automatically detect test subjects (such as mice) using multithreaded computer vision, while allowing seamless manual gap corrections.
Pixel densities and spatial tracks are logged rigorously to CSV logs for behavioral extraction.

## Requirements
Make sure you have a python environment (e.g., Anaconda). We recommend creating a new environment using **Python 3.10** (which has been tested and verified to work perfectly):

```bash
conda create -n rodencity python=3.10 -y
conda activate rodencity
pip install -r requirements.txt
```

## How to Run
```bash
python main.py
```

## Creating a Standalone Executable
If another lab member wants to run this tool without installing Python or Anaconda, you can compile it into a simple standalone `.exe` using `pyinstaller`.

1. Install PyInstaller into the environment:
```bash
pip install pyinstaller
```
2. Build the application executable file (this hides the debug console):
```bash
pyinstaller --onefile --windowed main.py --name rodencity
```
3. A `dist/` directory will automatically be created containing your executable executable/app!

## Analytical Pipeline & Usage

To ensure data integrity and prevent errors, the GUI layout enforces a strict 5-Step sequential order:

### Step 1: Video & Time Window
- **Load Video**: Select your `.mp4` or `.avi` testing video.
- **Set Window**: Use the slider to scrub past messy setup shots. Click `[ Set Start Time ]` and `[ Set End Time ]` to trim the video natively, restricting all algorithms from scanning noise data recorded prior to the actual session start. 

### Step 2: Environment Constraint
- **Calc Baseline**: Computes the static background median strictly across your inner time window.
- **Define 4-Point Arena**: Click explicitly on the 4 spatial corners of your inner-arena structure (to account for skewed camera angles!) to generate a constraining polygonal geometry.

### Step 3: Fast Mask Generation
- **Settings**: Adjust your subtraction threshold size or engage **Invert Detection** (if the software extracts environmental backgrounds instead of subjects).
- **Auto Mask ALL (Parallel)**: Subdivides the video across all available system multiprocessing cores to generate boundary mapping instantaneously. 

### Step 4: Manual Correction
- Navigate seamlessly employing the `A` and `D` rapid-review keyboard keys.
- Immediately paint and delete erroneous structural noise using `W` (Draw) and `E` (Erase) hotkeys.

### Step 5: Deliverables Output
- **Compile to Array (.npy)**: Generates a single massive 3D `uint8` Array of shape `(num_frames, height, width)` filled purely with `1`s (mouse) and `0`s (background). This file completely supersedes image-based logic, allowing you full programmatic flexibility to compute structural density and spatial spread over time downstream using pure Python or R scripts!
- **Render Labeled Video (.avi)**: Optionally export a visual check video. *(Note: Export stitching is an inherently sequential linear process; it does not utilize multiprocessing, but relies on heavily optimized single-core rendering).*

---

### Stimulus Tagging (Sidebar)
On the far right pane, establish timelines mapping when stimuli occur (e.g., light flashes or shocks):
1. Register `Mark START Here` and `Mark END Here` using the playhead.
2. Click `+ Add Stimulus to List`.
3. The system tracks absolute frame counts and duration intervals, generating a localized `stimulus_events.csv` structure that corresponds accurately alongside the primary `density_stats.csv`.

---

## Phase 2: NumPy Data Analysis & Execution Scripts

The `rodencity` layout inherently exports pure uncompressed computational binary structures (`_binary_masks.npy`). You can rapidly compile these outputs into tabular data constraints and graphs.

### Automated CSV processing
We provide a standalone mathematical extraction script that iterates over your binary files:
```bash
python scripts/process_video.py --npy myvideo_binary_masks.npy --stim myvideo_stimulus_events.csv
```
This engine structurally analyzes the physical pixel representations and parses them out into quantitative spreadsheets:
- `spatial_metrics_results.csv`: Extracts fundamental behavior on **every single continuous frame**. Triggers properties like `Area` density metrics, target tracking variables (`Centroid_X / Centroid_Y`), and geometric target density mappings (`Spread_Total`).
- `stimulus_correlations_results.csv`: Auto-crops isolated bounds across exact `stimulus` triggers isolated via the user.

### Instant Statistical Visualization Plots
Deploy the natively bundled chart render utility directly against your resulting csv sheets to isolate temporal anomalies without interacting with deep programming languages!
```bash
python scripts/visualize.py --metrics spatial_metrics_results.csv
```
**This automatically compiles:**
- `area_over_time.png`: Maps target contour "shrinking/enlargement" logic sequentially along timestamps (excellent for monitoring body tension or scatter responses).
- `motion_trajectory.png`: Visualizes structural X/Y tracking paths across the test arena (invaluable for continuous maze velocity calculations or exploration algorithms).
