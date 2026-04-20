import sys
import os
import argparse
import pandas as pd

# Enable relative root imports correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis.metrics import NumpyAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Process Rodencity Numpy Matrix Output Data")
    parser.add_argument("--npy", required=True, help="Path to your _binary_masks.npy database output")
    parser.add_argument("--stim", required=False, help="Path to your stimulus_events.csv mapping")
    parser.add_argument("--outdir", default=".", help="Output directory to place correlated spreadsheets")
    parser.add_argument("--fps", default=30, type=int, help="Frames per second of the video")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.npy):
        print(f"[Error] Numpy file not found: {args.npy}")
        sys.exit(1)
        
    print(f"Loading structural matrices from {args.npy}...")
    analyzer = NumpyAnalyzer(args.npy, fps=args.fps)
    
    print("Extracting geometric footprint (area, separation/spread, centroids)...")
    spatial_df = analyzer.extract_spatial_metrics()
    spatial_out = os.path.join(args.outdir, "spatial_metrics_results.csv")
    spatial_df.to_csv(spatial_out, index=False)
    print(f"-> Saved base metrics constraint to: {spatial_out}")
    
    if args.stim and os.path.exists(args.stim):
        print(f"Cross-correlating structural metrics against localized {args.stim} timelines...")
        event_df = analyzer.correlate_with_stimuli(spatial_df, args.stim)
        if event_df is not None:
            event_out = os.path.join(args.outdir, "stimulus_correlations_results.csv")
            event_df.to_csv(event_out, index=False)
            print(f"-> Saved isolated matrix events to: {event_out}")

if __name__ == "__main__":
    main()
