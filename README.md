# Mouse Video Annotation Tool

A simple PyQt5-based desktop application to load `.avi` videos and manually annotate test subjects (such as mice) by drawing on the frames. 
Annotations are saved as individual mask images and summarized in a density CSV file.

## Requirements
Make sure you have a python environment (e.g., Anaconda) and install the requirements:
```bash
pip install -r requirements.txt
```

## How to Run
```bash
python main.py
```

## Usage
1. Click **Load Video** and select your `.avi` file.
2. Use the **slider** at the bottom to navigate through frames.
3. Draw over the subjects using the mouse. You can change the brush size using the spin box. Select the **Erase Mode** checkbox to delete parts of the mask.
4. When you navigate to the next frame or release the mouse after drawing, the drawn mask is automatically saved to a `<videoname>_masks` folder (created next to your video).
5. In the same folder, a `density_stats.csv` file automatically records the density mean and standard deviation for the drawn annotations.

## Publishing to GitHub
To put your project on GitHub:
1. Create an empty repository on your GitHub account via github.com/new (do NOT initialize it with a README or .gitignore).
2. Run these commands from the `mouse-video-annotation` folder:

```bash
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin master
```
