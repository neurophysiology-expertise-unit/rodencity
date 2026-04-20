import numpy as np
import pandas as pd
import os

class NumpyAnalyzer:
    def __init__(self, npy_path, fps=30):
        self.npy_path = npy_path
        self.fps = fps
        self.masks = np.load(npy_path)
        self.total_frames = self.masks.shape[0]

    def extract_spatial_metrics(self):
        """Calculates area, centroid, and distribution spread for all tracked frames."""
        results = []
        for i in range(self.total_frames):
            frame_mask = self.masks[i]
            y_coords, x_coords = np.nonzero(frame_mask)
            
            area = len(y_coords)
            
            if area > 0:
                cx = np.mean(x_coords)
                cy = np.mean(y_coords)
                spread_x = np.std(x_coords)
                spread_y = np.std(y_coords)
                overall_spread = np.sqrt(spread_x**2 + spread_y**2)
            else:
                cx, cy, spread_x, spread_y, overall_spread = (np.nan,)*5
                
            results.append({
                'Frame': i,
                'Time_Sec': i / self.fps,
                'Area': area,
                'Centroid_X': cx,
                'Centroid_Y': cy,
                'Spread_Total': overall_spread
            })
            
        return pd.DataFrame(results)

    def correlate_with_stimuli(self, metrics_df, stimulus_csv):
        """Cross-correlates the derived numpy metrics against isolated behavioral event tags."""
        if not os.path.exists(stimulus_csv):
            return None
            
        stimuli = pd.read_csv(stimulus_csv)
        event_stats = []
        
        for idx, row in stimuli.iterrows():
            start_f = int(row['Start'])
            end_f = int(row['End'])
            dur = int(row['Duration'])
            
            slice_df = metrics_df[(metrics_df['Frame'] >= start_f) & (metrics_df['Frame'] <= end_f)]
            
            if not slice_df.empty:
                event_stats.append({
                    'Event_ID': idx + 1,
                    'Start_Frame': start_f,
                    'End_Frame': end_f,
                    'Duration_Frames': dur,
                    'Avg_Area': slice_df['Area'].mean(),
                    'Peak_Spread': slice_df['Spread_Total'].max(),
                    'Avg_Spread': slice_df['Spread_Total'].mean(),
                    'Total_Movement_X': slice_df['Centroid_X'].max() - slice_df['Centroid_X'].min(),
                    'Total_Movement_Y': slice_df['Centroid_Y'].max() - slice_df['Centroid_Y'].min()
                })
                
        return pd.DataFrame(event_stats)
