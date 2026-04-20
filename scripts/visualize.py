import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Plot Extracted Spatial Metrics from Rodencity Output")
    parser.add_argument("--metrics", required=True, help="Path to your targeted spatial_metrics_results.csv")
    args = parser.parse_args()
    
    if not os.path.exists(args.metrics):
        print(f"[Error] Could not locate metric data context boundary at {args.metrics}")
        sys.exit(1)
        
    print(f"Ingesting raw metrics from {args.metrics}...")
    df = pd.read_csv(args.metrics)
    
    if df.empty:
        print("[Error] Metrics dataset array appears unpopulated.")
        sys.exit(1)
        
    print("Rendering 'area_over_time.png' analytical trajectory...")
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time_Sec'], df['Area'], color='blue', alpha=0.7)
    plt.title('Subject Array Spatial Body Area Limits mapping over Recording Time')
    plt.xlabel('Time Environment (Seconds)')
    plt.ylabel('Volumetric Pixel Area Coverage')
    plt.grid(True)
    plt.savefig('area_over_time.png', dpi=300)
    plt.close()
    
    print("Rendering 'motion_trajectory.png' structural positioning grid...")
    plt.figure(figsize=(8, 8))
    plt.plot(df['Centroid_X'], df['Centroid_Y'], color='red', alpha=0.5, linewidth=1)
    
    plt.title('2D Tracked Locomotion Path Matrix')
    plt.xlabel('X Tracking Core')
    plt.ylabel('Y Tracking Core')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.savefig('motion_trajectory.png', dpi=300)
    plt.close()
    
    print("[Success] Output visualizations finalized structurally!")

if __name__ == "__main__":
    main()
