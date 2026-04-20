# Mouse Video Annotation Tool

A powerful PyQt5-based desktop pipeline application to load `.avi` videos and automatically detect test subjects (such as mice) using multithreaded computer vision, while allowing seamless manual gap-interpolation corrections.
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
- **Interpolate Gap**: Geometrically transforms and extrapolates tracking geometry across untracked frames sandwiched by mapped ones! 

### Step 5: Render Engine
- **Export Final Labeled Video** compiles all constraints, temporal tags, masks, and boundaries spanning solely the analytical window into a structurally finalized `<vid>_labeled.avi` without mutating the underlying read-only asset.

---

### Stimulus Tagging (Sidebar)
On the far right pane, establish timelines mapping when stimuli occur (e.g., light flashes or shocks):
1. Register `Mark START Here` and `Mark END Here` using the playhead.
2. Click `+ Add Stimulus to List`.
3. The system tracks absolute frame counts and duration intervals, generating a localized `stimulus_events.csv` structure that corresponds accurately alongside the primary `density_stats.csv`.
