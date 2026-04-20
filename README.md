# Mouse Video Annotation Tool

A simple PyQt5-based desktop application to load `.avi` videos and manually annotate test subjects (such as mice) by drawing on the frames, or automatically detect them using computer vision!
Annotations are saved as individual mask images and summarized in a density CSV file.

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

## Usage
1. **Load Video**: Click `1. Load Video` and select your `.avi` file.
2. **Navigation**: Use the slider or press the `A` (`< Prev`) and `D` (`Next >`) keys to efficiently scrub across frames.
3. **Computer Vision Auto-Masking (Recommended)**:
   - Click `Calc Baseline (Median)` so the software can isolate the empty plexiglass floor.
   - Click `3. Define 4-Point Arena` and click exactly on the 4 corners of your physical testing container (tilted or straight) to draw a polygonal border so auto-detection ignores external shadows. 
   - Use `Auto Mask Current` or `Auto Mask ALL` to let OpenCV dynamically find your moving subjects and automatically draw borders around them! You can tweak the spinbox threshold value if it's too aggressive or weak.
4. **Manual Mode**:
   - You can also manually paint annotations. It automatically superimposes the previous drawing when entering a blank frame! If you skip multiple frames, use the `Interpolate Gap` button to instantly stretch missing labels between two manually traced points over time!
5. **Data Output**: Every time you draw or the computer auto-detects, masks are instantly autosaved to a `<videoname>_masks` folder alongside a continuous `density_stats.csv` recording exactly how much pixel density is tracked across frames.

## Publishing to GitHub
To put your project on GitHub:
1. Create an empty repository named `rodencity` under the `neurophysiology-expertise-unit` organization via GitHub's interface (do NOT initialize it with a README or .gitignore).
2. Run these commands from your local `rodencity` folder:

```bash
git remote add origin https://github.com/neurophysiology-expertise-unit/rodencity.git
git push -u origin main
```
