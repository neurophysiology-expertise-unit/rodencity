# Rodencity

A PyQt5-based desktop suite for rodent video analysis. On launch, a tool picker lets you choose between two independent modules:

| Tool | Purpose |
|------|---------|
| **Analysis Pipeline** | Automated detection via background subtraction, arena masking, and density tracking |
| **Behavior Annotator** | Manually label behavioral events (seizure, grooming, rearing…) as time intervals — works over Samba/network shares via local caching |

## Requirements
Make sure you have a python environment manager like Anaconda or Miniconda.

### Environment Setup

1.  Create a new Conda environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
2.  Activate the newly created environment:
    ```bash
    conda activate density_heatmap_env
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

> [!WARNING]
> **"No Qt Platform Plugin" Error on Windows?**
> If your `.exe` crashes instantly with `no qt platform plugin could be initialized`, it is because the standard `opencv-python` comes bundled with its own Qt plugins which overwrite and conflict with `PyQt5` during the PyInstaller build sequence.
> **Fix:** Ensure you have uninstalled `opencv-python` and exclusively installed `opencv-python-headless` in your environment prior to running PyInstaller!

---

## Behavior Annotator

Designed for labeling behavioral events (e.g. seizures, grooming bouts) directly in videos stored on Samba/network shares, where random-seek playback would otherwise be too slow.

### How it works
When a user opens a video, the tool **copies it to a local temp directory** (`%TEMP%\rodencity_cache\` on Windows, `/tmp/rodencity_cache/` on Linux) before opening it. Subsequent opens of the same file (matched by filename + size) skip the copy. Playback then runs entirely from local storage at full speed.

### Workflow
1. **Load Video** — select any `.avi`, `.mp4`, `.mkv`, or `.mov` file. A progress bar tracks the local copy.
2. **Scrub** — use the slider, `A`/`D` or `←`/`→` arrow keys to navigate frames. Use the speed spinner to play back at 0.1× – 8× speed.
3. **Choose a label** — pick a preset (`seizure`, `grooming`, `rearing`, `freezing`, `exploration`, `other`) or type a custom label.
4. **Mark the interval** — navigate to the first frame of the event and press `S` (or **Mark Start**). Navigate to the last frame and press `E` (or **Mark End**).
5. **Add** — press `Enter` (or **+ Add Annotation**). The event is immediately appended to the list and saved.
6. **Remove** — select any row and click **- Remove Selected**.

### Keyboard shortcuts
| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `A` / `←` | Previous frame |
| `D` / `→` | Next frame |
| `S` | Mark start frame |
| `E` | Mark end frame |
| `Enter` | Add annotation |

### Output CSV
Annotations are saved automatically to `<video_name>_behavior_annotations.csv` **next to the original source video** (on the network share), so all lab members see the same file. Columns:

```
Label, Start_Frame, End_Frame, Start_Time_Sec, End_Time_Sec, Duration_Sec
```

### Notes
- The local cache is never automatically deleted — clear `rodencity_cache` manually if disk space is a concern.
- Works identically on Windows and Linux. No extra dependencies beyond what the existing environment already provides.

---

## Analysis Pipeline & Usage

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

- **Generate Standard Plots**:
  ```bash
  python scripts/visualize.py --metrics spatial_metrics_results.csv
  ```
  **This automatically compiles:**
  - `area_over_time.png`: Maps target contour "shrinking/enlargement" logic sequentially along timestamps (excellent for monitoring body tension or scatter responses).
  - `motion_trajectory.png`: Visualizes structural X/Y tracking paths across the test arena (invaluable for continuous maze velocity calculations or exploration algorithms).

- **Generate Density Heatmap**:
  To visualize the mean density across all stimulus events, use the `plot_density_heatmap.py` script. Make sure your `_density_stats.csv` and `_stimulus_events.csv` files are in the same directory.
  ```bash
  python plot_density_heatmap.py
  ```
  This will generate `density_heatmap.png`.
