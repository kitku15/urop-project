from imports import *


def get_intensities(idx, image, center, diameter):

    raw_radii = diameter/2

    h, w = image.shape
    cx, cy = center[0]

    yy, xx = np.ogrid[:h, :w]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2

    mask = dist_sq <= raw_radii ** 2

    # Apply mask and compute mean intensity
    mean_intensity = image[mask].mean() if np.any(mask) else 0.0

    return mean_intensity


def get_all_intensities(img_boxes, coordinates, diameters):
    
    intensity_means = []

    for idx, (img, center, diameter) in enumerate(zip(img_boxes, coordinates, diameters)):
        mean_intensity = get_intensities(idx, img, center, diameter)
        intensity_means.append(mean_intensity)


    return intensity_means



def get_diameter(largest_region, binary_mask, img_box, idx):
    '''

    '''

    region = largest_region

    # CALCULATE CIRCULARITY 
    circularity = 4 * np.pi * region.area / (region.perimeter ** 2)

    # CALCULATE DIAMETER
    diameter = region.equivalent_diameter

    # Centroid and radius
    y, x = region.centroid
    r = diameter / 2

    # visualize dupa --------------------------
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # # LEFT: Binary mask with circle
    # ax1.imshow(binary_mask, cmap='gray')
    # ax1.set_title(f'{idx} - Binary Mask')
    # circle = plt.Circle((x, y), r, edgecolor='red', fill=False, linewidth=1.5)
    # ax1.add_patch(circle)
    # ax1.text(x, y, f'{diameter:.2f}', color='blue', fontsize=8, ha='center', va='center')
    # ax1.axis('off')

    # # RIGHT: Original image
    # ax2.imshow(img_box, cmap='gray' if img_box.ndim == 2 else None)
    # ax2.set_title(f'{idx} - Original Image')
    # ax2.axis('off')

    # # Save the figure
    # path = f'distribution/images/1/ND6_{idx}.png'
    # directory = os.path.dirname(path)
    # os.makedirs(directory, exist_ok=True)
    # plt.tight_layout()
    # plt.savefig(path)
    # plt.close()
    # -----------------------------------------

    return diameter, circularity


def get_all_diameter(largest_region_list, binary_mask_list, img_boxes):
    
    diameters = []
    circularities = []

    for idx, (larg_region, binary_mask, img_box) in enumerate(zip(largest_region_list, binary_mask_list, img_boxes)):
        diameter, circularity= get_diameter(larg_region, binary_mask, img_box, idx)
        diameters.append(diameter)
        circularities.append(circularity)


    return diameters, circularities


def save_histograms(csv_path):
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Define the features and corresponding filenames
    features = ['diameter', 'intensity', 'circularity']
    
    for feature in features:
        plt.figure()
        plt.hist(df[feature], bins=20, edgecolor='black')
        plt.title(f'Histogram of {feature.capitalize()}')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plot_path = csv_path.replace(".csv", "")

        plt.savefig(f'{plot_path}_{feature}_histogram.png')
        plt.close()


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_wt_mutant_overlap(marker, repeat, wt_csv, mutant_csv, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)

    # Load CSVs
    df_wt = pd.read_csv(wt_csv)
    df_mut = pd.read_csv(mutant_csv)

    features = ['intensity', 'normalized_intensity']
    condition_colors = {
        'WT': "#00da1d",
        'Mutant': "#ff320e"
    }

    for feature in features:
        data_wt = df_wt[feature].dropna().values
        data_mut = df_mut[feature].dropna().values

        # KDE estimation (smoother histogram)
        kde_wt = gaussian_kde(data_wt)
        kde_mut = gaussian_kde(data_mut)

        x_min = min(data_wt.min(), data_mut.min())
        x_max = max(data_wt.max(), data_mut.max())
        x_range = np.linspace(x_min, x_max, 1000)

        y_wt = kde_wt(x_range)
        y_mut = kde_mut(x_range)

        # Calculate AUCs
        auc_wt = np.trapz(y_wt, x_range)
        auc_mut = np.trapz(y_mut, x_range)
        y_overlap = np.minimum(y_wt, y_mut)
        auc_overlap = np.trapz(y_overlap, x_range)

        # Overlap % (normalized to average area)
        overlap_percentage = 100 * (2 * auc_overlap) / (auc_wt + auc_mut)

        # Plotting
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x_range, y_wt, label='WT', color=condition_colors['WT'], linewidth=2)
        ax.plot(x_range, y_mut, label='Mutant', color=condition_colors['Mutant'], linewidth=2)
        ax.fill_between(x_range, y_overlap, color='orange', alpha=0.3, label='Overlap')

        ax.set_title(f"Repeat {repeat} {feature.capitalize()} Distribution (WT vs Mutant)")
        ax.set_xlabel(f"{feature.capitalize()}")
        ax.set_ylabel("Frequency (Density)")
        ax.legend()

        ax.text(0.95, 0.95, f"Overlap: {overlap_percentage:.2f}%",
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

        plt.tight_layout()
        output_path = os.path.join(save_dir, f'{repeat}/R{repeat}_{marker}_overlap_{feature}.png')
        plt.savefig(output_path)
        plt.close()

        print(f"Saved plot: {output_path}")
        print(f"AUC WT: {auc_wt:.3f}, AUC Mutant: {auc_mut:.3f}, Overlap AUC: {auc_overlap:.3f}, Overlap %: {overlap_percentage:.2f}%")

def fix_marker_csv(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df[['Index', 'intensity']]  # Keep only these columns
    df.to_csv(output_path, index=False)
    print(f"Fixed CSV saved to: {output_path}")


def normalize_by_dapi(marker_csv, dapi_csv, output_path):
    # Read both CSVs
    marker_df = pd.read_csv(marker_csv)
    dapi_df = pd.read_csv(dapi_csv)

    # Check that the Index columns match
    if not marker_df['Index'].equals(dapi_df['Index']):
        raise ValueError("Indexes do not match between marker and DAPI CSVs.")

    # Add a new column with normalized values
    marker_df['normalized_intensity'] = marker_df['intensity'] / dapi_df['intensity']

    # Save to the same marker CSV or to a new file
    marker_df.to_csv(output_path, index=False)
    print(f"Normalized data added and saved to: {output_path}")



def plot_intensity_vs_diameter(repeat, csv1_path, csv2_path, marker, location, output_path):
    # Read the CSVs
    df_intensity = pd.read_csv(csv1_path)
    df_diameter = pd.read_csv(csv2_path)

    # Align the IDs: CSV1 starts at 1, CSV2 starts at 0
    df_intensity['aligned_index'] = df_intensity['ID'] - 1

    # Filter by marker
    df_marker = df_intensity[df_intensity['marker'] == marker]

    # Merge with diameter data on aligned index
    merged = df_marker.merge(df_diameter, left_on='aligned_index', right_on='Index')

    # Check if location exists
    if location not in ['inner', 'mid', 'outer']:
        raise ValueError("Location must be one of: 'inner', 'mid', 'outer'.")

    # Count number of points
    num_points = merged.shape[0]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(merged['diameter'], merged[location], color='blue', alpha=0.7)
    plt.xlabel('Diameter')
    plt.ylabel(f'{marker} Intensity in {location.capitalize()} Region')
    plt.title(f'Repeat {repeat} - {marker} Intensity vs Diameter ({location.capitalize()} Region)')
    plt.grid(True)

    # Add number of points as text in top-right corner
    plt.text(
        0.95, 0.95, 
        f'N = {num_points}', 
        ha='right', va='top', 
        transform=plt.gca().transAxes,
        fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray')
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to: {output_path}")


def combined_plot_intensity_vs_diameter(
    repeats_data,  # List of tuples: (repeat_label, csv1_path, csv2_path, marker)
    marker_loc_dict,  # e.g., {'SOX2': 'inner', 'BRA': 'mid', 'GATA3': 'outer'}
):
    plt.figure(figsize=(10, 7))

    for repeat, condition, csv1_path, csv2_path, marker in repeats_data:
        location = marker_loc_dict.get(marker)
        if location is None:
            print(f"Skipping marker {marker}: location not found in marker_loc_dict.")
            continue

        # Load CSVs
        df_intensity = pd.read_csv(csv1_path)
        df_diameter = pd.read_csv(csv2_path)
        df_intensity['aligned_index'] = df_intensity['ID'] - 1

        # Filter and merge
        df_marker = df_intensity[df_intensity['marker'] == marker]
        merged = df_marker.merge(df_diameter, left_on='aligned_index', right_on='Index')
        num_points = merged.shape[0]

        # Scatter plot for this group
        plt.scatter(
            merged['diameter'],
            merged[location],
            alpha=0.7,
            label=f'{condition}_{marker} (Repeat {repeat}, N={num_points})'
        )

    # Final plot settings
    plt.ylim(0, 1)
    plt.xlabel('Diameter')
    plt.ylabel('Marker Intensity (region-specific)')
    plt.title('Combined Intensity vs Diameter')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = f'plots/{repeat}/R{repeat}_{marker}_{location}_diameterDis.png'

    plt.savefig(output_path)
    plt.close()

    print(f"Combined plot saved to: {output_path}")
