import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import os


def bar_plot(csv_path, repeat, plot_type='line', save_dir='plots'):
    df = pd.read_csv(csv_path)

    # Remove DAPI
    df = df[df['marker'].str.upper() != 'DAPI']

    # Focus on specified repeat
    df = df[df['repeat'] == repeat]

    # Unique (repeat, condition) pairs
    groups = df.groupby(['repeat', 'condition'])

    # Regions
    regions = ['inner', 'mid', 'outer']
    x = np.arange(len(regions))

    # Define marker colors
    marker_colors = {
        'SOX2': '#00bcd4',
        'BRA': '#ffeb3b',
        'GATA3': '#9c27b0',
    }

    os.makedirs(save_dir, exist_ok=True)

    for (repeat, condition), group in groups:
        fig, ax = plt.subplots(figsize=(6,5))

        markers = group['marker'].tolist()
        n_markers = len(markers)
        bar_width = 0.25  # wider bars

        # For line plot smoothing
        x_smooth = np.linspace(x.min(), x.max(), 300)

        for i, (_, row) in enumerate(group.iterrows()):
            marker = row['marker']
            color = marker_colors.get(marker, 'gray')
            y = row[regions].values.astype(float)

            if plot_type == 'line':
                spline = make_interp_spline(x, y, k=2)
                y_smooth = spline(x_smooth)
                ax.plot(x_smooth, y_smooth, label=marker, color=color, linewidth=2)
                ax.scatter(x, y, color=color, edgecolor='k', zorder=5)

            elif plot_type == 'bar':
                # Shift bars by i * bar_width
                ax.bar(x + i * bar_width, y, width=bar_width, label=marker, color=color, edgecolor='black')

        if plot_type == 'bar':
            # Center x-ticks under group of bars
            ax.set_xticks(x + bar_width * (n_markers - 1) / 2)
        else:
            ax.set_xticks(x)

        ax.set_xticklabels([r.capitalize() for r in regions])
        ax.set_ylabel('Average Normalized Intensity')
        ax.set_xlabel('Region')
        ax.set_title(f'Marker Intensities - Repeat {repeat}, Condition {condition}')
        ax.legend(title='Marker')
        plt.tight_layout()
        plt.ylim(0, 1)

        output_path = os.path.join(save_dir, f'{repeat}/R{repeat}_intensities_plot_{condition}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")




def plot_marker_condition_overlap(csv_path, repeat, marker, save_dir='plots'):
    df = pd.read_csv(csv_path)

    # Remove DAPI
    df = df[df['marker'].str.upper() != 'DAPI']
    
    # Focus only on specified marker
    df = df[df['marker'].str.upper() == marker.upper()]

    # Focus on specified repeat
    df = df[df['repeat'] == repeat]
    
    if df.empty or len(df['condition'].unique()) < 2:
        print(f"Not enough data for marker '{marker}' with both conditions.")
        return

    regions = ['inner', 'mid', 'outer']
    x = np.arange(len(regions))
    x_smooth = np.linspace(x.min(), x.max(), 300)

    # Set colors
    condition_colors = {
        'WT': "#00da1d",
        'ND6': "#ff320e"
    }

    os.makedirs(save_dir, exist_ok=True)

    # Get average intensity per region for each condition
    avg_df = df.groupby('condition')[regions].mean()

    fig, ax = plt.subplots(figsize=(6,5))

    curves = {}
    for condition in ['WT', 'ND6']:
        if condition not in avg_df.index:
            continue

        y = avg_df.loc[condition].values.astype(float)
        spline = make_interp_spline(x, y, k=2)
        y_smooth = spline(x_smooth)
        curves[condition] = y_smooth

        ax.plot(x_smooth, y_smooth, label=condition, color=condition_colors[condition], linewidth=2)
        ax.scatter(x, y, color=condition_colors[condition], edgecolor='k', zorder=5)

    # Highlight overlap region
    y_min = np.minimum(curves['WT'], curves['ND6'])
    y_max = np.maximum(curves['WT'], curves['ND6'])

    ax.fill_between(x_smooth, y_min, where=(curves['WT'] > 0) & (curves['ND6'] > 0), 
                    interpolate=True, color="#ff8f0e", alpha=0.3, label='Overlap')

    # AUCs
    auc_wt = np.trapz(curves['WT'], x_smooth)
    auc_nd6 = np.trapz(curves['ND6'], x_smooth)
    auc_overlap = np.trapz(y_min, x_smooth)
    total_auc = auc_wt + auc_nd6

    overlap_percentage = 100 * (2 * auc_overlap) / total_auc
    overlap_text = f"Overlap: {overlap_percentage:.2f}%"

    # Add text to plot
    ax.text(0.95, 0.05, overlap_text,
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in regions])
    ax.set_ylabel('Average Normalized Intensity')
    ax.set_xlabel('Region')
    ax.set_title(f'{marker.upper()} Intensities (WT vs ND6)')
    ax.legend()
    plt.tight_layout()
    plt.ylim(0, 1)

    output_path = os.path.join(save_dir, f'{repeat}/R{repeat}_overlap_plot_{marker.upper()}.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot: {output_path}")
    print(f"AUC WT: {auc_wt:.3f}")
    print(f"AUC ND6: {auc_nd6:.3f}")
    print(f"Overlap AUC: {auc_overlap:.3f}")
    print(f"Overlap %: {overlap_percentage:.2f}%")



if __name__ == "__main__":

    meta_path = 'intensities/meta_intensities.csv'
    repeats = [1, 3]
    markers = ["SOX2", "BRA"]

    for repeat in repeats:
        bar_plot(meta_path, repeat, plot_type='line')

    for repeat in repeats:
        for marker in markers:
            plot_marker_condition_overlap(meta_path, repeat, marker)


