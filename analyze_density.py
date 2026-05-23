import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = r"c:\Users\cagat\local\rodencity\data"
output_dir = r"c:\Users\cagat\local\rodencity\output"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
groups = ['pm', 'po']
animals = ['1', '2']

all_events = []
density_data = {}

for g in groups:
    for a in animals:
        animal_id = f"{g}_{a}"
        events_path = os.path.join(data_dir, f"{animal_id}_events.csv")
        density_path = os.path.join(data_dir, f"{animal_id}_density.csv")
        
        if not os.path.exists(events_path) or not os.path.exists(density_path):
            print(f"Missing files for {animal_id}")
            continue
            
        events = pd.read_csv(events_path)
        events['Type'] = events['Type'].str.strip() # Strip whitespaces
        events['Animal'] = animal_id
        events['Strain'] = g
        events = events.sort_values(by='Start')
        all_events.append(events)
        
        density = pd.read_csv(density_path)
        # Create mapping of Frame -> Mean_Density
        density_data[animal_id] = dict(zip(density['Frame'], density['Mean_Density']))

events_df = pd.concat(all_events, ignore_index=True)

# 2. Get unique stimulus types
stimulus_types = events_df['Type'].unique()
print(f"Unique stimulus types found: {stimulus_types}")

# 3. Process heatmaps to find global min/max
heatmap_data = {}
global_min = float('inf')
global_max = float('-inf')

for st in stimulus_types:
    st_events = events_df[events_df['Type'] == st].copy()
    max_duration = int(st_events['Duration'].max())
    
    heatmap_before = []
    heatmap_during = []
    y_labels = []
    
    for idx, row in st_events.iterrows():
        animal = row['Animal']
        start = int(row['Start'])
        duration = int(row['Duration'])
        dmap = density_data[animal]
        
        before = [dmap.get(f, np.nan) for f in range(start - max_duration, start)]
        during = [dmap.get(f, np.nan) if (f - start) < duration else np.nan for f in range(start, start + max_duration)]
        
        heatmap_before.append(before)
        heatmap_during.append(during)
        y_labels.append(f"{animal} (T={start})")

    heatmap_before = np.array(heatmap_before)
    heatmap_during = np.array(heatmap_during)
    
    # Ignore All-NaN slices when calculating min/max
    valid_mask = ~np.isnan(np.hstack([heatmap_before, heatmap_during]))
    if np.any(valid_mask):
        vmin = np.nanmin(np.hstack([heatmap_before, heatmap_during])[valid_mask])
        vmax = np.nanmax(np.hstack([heatmap_before, heatmap_during])[valid_mask])
        if vmin < global_min: global_min = vmin
        if vmax > global_max: global_max = vmax
        
    heatmap_data[st] = {
        'before': heatmap_before,
        'during': heatmap_during,
        'y_labels': y_labels,
        'max_duration': max_duration
    }

global_max = 0.1 # override max density due to outliers
print(f"Global Heatmap Colormap Limits: Min={global_min}, Max={global_max}")

# Plot heatmaps with consistent colormap
for st, data in heatmap_data.items():
    max_duration = data['max_duration']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'wspace': 0.1})
    
    num_ticks = 5
    xticks = np.linspace(0, max_duration - 1, num_ticks)
    
    sns.heatmap(data['before'], ax=axes[0], cmap='viridis', cbar=False, yticklabels=data['y_labels'], vmin=global_min, vmax=global_max)
    axes[0].set_title(f'{st} - Before Stimulus')
    axes[0].set_xlabel('Frames relative to start')
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels([str(int(x - max_duration)) for x in xticks], rotation=0)

    sns.heatmap(data['during'], ax=axes[1], cmap='viridis', cbar_kws={'label': 'Mean Density'}, yticklabels=False, vmin=global_min, vmax=global_max)
    axes[1].set_title(f'{st} - During Stimulus')
    axes[1].set_xlabel('Frames after start')
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels([str(int(x)) for x in xticks], rotation=0)

    plt.suptitle(f'Density Heatmap for Stimulus: {st}')
    plt.savefig(os.path.join(output_dir, f'heatmap_{st}.png'), bbox_inches='tight')
    plt.close()

# 4. Statistical Comparison
# Take first 2 trials of each type for each animal
first_2_trials = events_df.groupby(['Animal', 'Type']).head(2).copy()

# Calculate mean density for stimulus and baseline
def get_trial_mean_density(row):
    animal = row['Animal']
    start = int(row['Start'])
    duration = int(row['Duration'])
    dmap = density_data[animal]
    
    # Stimulus density
    stim_vals = [dmap.get(f, np.nan) for f in range(start, start + duration)]
    
    # Baseline density: use the same duration before the stimulus
    base_vals = [dmap.get(f, np.nan) for f in range(start - duration, start)]
    
    b_mean = np.nanmean(base_vals)
    b_std = np.nanstd(base_vals)
    s_mean = np.nanmean(stim_vals)
    z_score = (s_mean - b_mean) / b_std if b_std > 0 else np.nan
    
    return pd.Series({'Trial_Mean_Density': s_mean, 'Baseline_Mean_Density': b_mean, 'Z_Score': z_score})

first_2_trials[['Trial_Mean_Density', 'Baseline_Mean_Density', 'Z_Score']] = first_2_trials.apply(get_trial_mean_density, axis=1)

# Do NOT average the 2 trials for each animal, just use them directly
animal_means = first_2_trials.copy()

# Plot error bar plots for each stimulus
for st in stimulus_types:
    st_data = animal_means[animal_means['Type'] == st]
    if len(st_data) == 0:
        continue
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    order = ['pm', 'po']
    
    # Baseline Plot
    sns.pointplot(x='Strain', y='Baseline_Mean_Density', data=st_data, order=order, capsize=0.1, errorbar='se', join=False, color='black', ax=axes[0])
    sns.stripplot(x='Strain', y='Baseline_Mean_Density', data=st_data, order=order, color='blue', size=8, jitter=False, ax=axes[0], alpha=0.6)
    axes[0].set_title(f'Baseline (Before)')
    axes[0].set_ylabel('Mean Density (averaged over 2 trials)')
    axes[0].set_xlabel('Strain')
    
    # Stimulus Plot
    sns.pointplot(x='Strain', y='Trial_Mean_Density', data=st_data, order=order, capsize=0.1, errorbar='se', join=False, color='black', ax=axes[1])
    sns.stripplot(x='Strain', y='Trial_Mean_Density', data=st_data, order=order, color='red', size=8, jitter=False, ax=axes[1], alpha=0.6)
    axes[1].set_title(f'Stimulus (During)')
    axes[1].set_ylabel('') # Share y-axis
    axes[1].set_xlabel('Strain')
    
    plt.suptitle(f'Mean Density Comparison for Stimulus: {st}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{st}.png'), bbox_inches='tight')
    plt.close()

# 5. Summary Plots for Each Strain (All Stimulus Types, Before vs After)
long_df = pd.melt(animal_means, id_vars=['Strain', 'Animal', 'Type'], 
                  value_vars=['Baseline_Mean_Density', 'Trial_Mean_Density'], 
                  var_name='Period', value_name='Density')
long_df['Period'] = long_df['Period'].map({'Baseline_Mean_Density': 'Before', 'Trial_Mean_Density': 'After'})

for strain in ['pm', 'po']:
    strain_df = long_df[long_df['Strain'] == strain]
    if len(strain_df) == 0: continue
    
    plt.figure(figsize=(10, 6))
    
    # Enforce consistent order for stimulus types
    stim_order = ['DS', 'SS', 'EWD', 'EDD']
    
    # We use dodge=True so before and after don't overlap completely
    sns.pointplot(data=strain_df, x='Type', y='Density', hue='Period', 
                  capsize=0.1, errorbar='se', join=False, dodge=0.4, palette='dark', order=stim_order)
    sns.stripplot(data=strain_df, x='Type', y='Density', hue='Period', 
                  dodge=0.4, alpha=0.5, size=7, palette='pastel', zorder=0, order=stim_order)
    
    plt.title(f'Stimulus Response Comparison ({strain.upper()} Strain)')
    plt.ylabel('Mean Density')
    plt.xlabel('Stimulus Type')
    
    # Fix duplicate legend from pointplot and stripplot
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title='Period')
    
    plt.savefig(os.path.join(output_dir, f'summary_comparison_{strain}.png'), bbox_inches='tight')
    plt.close()

# 6. Summary Plots for Z-Scored Responses
for strain in ['pm', 'po']:
    strain_df = animal_means[animal_means['Strain'] == strain]
    if len(strain_df) == 0: continue
    
    plt.figure(figsize=(10, 6))
    
    # Enforce consistent order for stimulus types
    stim_order = ['DS', 'SS', 'EWD', 'EDD']
    
    sns.pointplot(data=strain_df, x='Type', y='Z_Score', 
                  capsize=0.1, errorbar='se', join=False, color='black', order=stim_order)
    sns.stripplot(data=strain_df, x='Type', y='Z_Score', 
                  alpha=0.5, size=7, color='red', zorder=0, order=stim_order)
    
    plt.title(f'Z-Scored Stimulus Response ({strain.upper()} Strain)')
    plt.ylabel('Z-Score (vs Baseline)')
    plt.xlabel('Stimulus Type')
    plt.axhline(0, color='gray', linestyle='--') # Add line for zero (baseline level)
    
    plt.savefig(os.path.join(output_dir, f'zscore_summary_{strain}.png'), bbox_inches='tight')
    plt.close()

print("Analysis complete. Check output directory: ", output_dir)
