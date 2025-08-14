import pandas as pd
import numpy as np
import os
from scipy.interpolate import make_interp_spline
from f_modelDetection import load_allowed_ids, load_boxes
from PIL import Image
import matplotlib.pyplot as plt


def score_gastruloid_similarity(directory, repeat, marker, condition):

    meta_individual = f"{directory}/{repeat}/intensities/meta_individual_{condition}.csv"
    meta_intensities = f"{directory}/meta_intensities.csv"

    output_csv_path=f'{directory}/{repeat}/{condition}_overlap_scores_{marker}.csv'

    # Load data
    interest_df = pd.read_csv(meta_individual) # contains intensity measurements for each individual gastruloid
    wt_df = pd.read_csv(meta_intensities) # contains AVERAGE intensity measurements for each repeat and condition 

    # Preprocess WT AVERAGE intensity data
    wt_df = wt_df[wt_df['marker'].str.upper() != 'DAPI']
    wt_df = wt_df[wt_df['marker'].str.upper() == marker.upper()]
    wt_df = wt_df[wt_df['repeat'] == repeat]
    wt_df = wt_df[wt_df['condition'] == "WT"]

    if wt_df.empty:
        print("No WT data found for given marker/repeat.")
        return

    # Preprocess gastruloid of interest individual data (called interest bcs it could be ND6 or WT)
    interest_df = interest_df[interest_df['marker'].str.upper() == marker.upper()]
    if interest_df.empty:
        print("No ND6 individual data found for given marker.")
        return

    # Regions and x positions
    regions = ['inner', 'mid', 'outer']  # or outer, mid, inner — just match your plotting logic
    x = np.arange(len(regions))
    x_smooth = np.linspace(x.min(), x.max(), 300)

    # Compute WT average curve
    wt_means = wt_df[regions].mean().values.astype(float)
    spline_wt = make_interp_spline(x, wt_means, k=2)
    wt_smooth = spline_wt(x_smooth)

    # Compute scores for each ND6 gastruloid
    results = []
    for _, row in interest_df.iterrows():
        nd6_vals = row[regions].values.astype(float)
        spline_nd6 = make_interp_spline(x, nd6_vals, k=2)
        nd6_smooth = spline_nd6(x_smooth)

        # Overlap calculation (same as plot logic)
        y_min = np.minimum(wt_smooth, nd6_smooth)
        auc_wt = np.trapz(wt_smooth, x_smooth)
        auc_nd6 = np.trapz(nd6_smooth, x_smooth)
        auc_overlap = np.trapz(y_min, x_smooth)
        total_auc = auc_wt + auc_nd6

        overlap_percentage = 100 * (2 * auc_overlap) / total_auc

        results.append({
            'ID': row['ID'],
            'marker': marker.upper(),
            'overlap_score': overlap_percentage
        })

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Saved scores to {output_csv_path}")
    return results_df, output_csv_path


def sort_overlap_scores(input_csv, output_csv=None):
    # Read CSV
    df = pd.read_csv(input_csv)
    
    # Sort by overlap_score descending
    df_sorted = df.sort_values(by='overlap_score', ascending=False)
    
    # Save or return
    if output_csv:
        df_sorted.to_csv(output_csv, index=False)
        print(f"Sorted CSV saved to {output_csv}")
    return df_sorted

def score_gastruloid(directory, repeats, markers, conditions):
    for repeat in repeats:
        for marker in markers:
            for condition in conditions:
                if marker != "DAPI": # filter out dapi
                    results, output_csv_path = score_gastruloid_similarity(directory, repeat, marker, condition)
                    results_sorted = results.sort_values(by='overlap_score', ascending=False)
                    results_sorted.to_csv(output_csv_path, index=False)

def final_score_gastruloid(directory, repeats, conditions):

    for repeat in repeats:
        for condition in conditions:
            # Paths to your CSVs
            gata3_csv = f"{directory}/{repeat}/{condition}_overlap_scores_GATA3.csv"
            sox2_csv = f"{directory}/{repeat}//{condition}_overlap_scores_SOX2.csv"
            bra_csv = f"{directory}/{repeat}//{condition}_overlap_scores_BRA.csv"

            # Load and rename overlap_score columns
            gata3_df = pd.read_csv(gata3_csv)[['ID', 'overlap_score']].rename(columns={'overlap_score': 'GATA3_score'})
            sox2_df = pd.read_csv(sox2_csv)[['ID', 'overlap_score']].rename(columns={'overlap_score': 'SOX2_score'})
            bra_df   = pd.read_csv(bra_csv)[['ID', 'overlap_score']].rename(columns={'overlap_score': 'BRA_score'})

            # Merge all on ID
            merged_df = gata3_df.merge(sox2_df, on='ID').merge(bra_df, on='ID')

            # Calculate final average score
            merged_df['final_score'] = merged_df[['GATA3_score', 'SOX2_score', 'BRA_score']].mean(axis=1)

            # rank based on final score 
            merged_df_sorted = merged_df.sort_values(by='final_score', ascending=False)

            # Save to CSV
            save_path = f"{directory}/{repeat}/{condition}_overlap_scores_compiled.csv"
            merged_df_sorted.to_csv(save_path, index=False)

            print(f"Saved {save_path}")

def hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)

def color_image(gray_img, color):
    """Convert a grayscale image to an RGB image with a specific color."""
    return np.stack([gray_img]*3, axis=-1) * color

def overlay_channels(gata3_img_path, sox2_img_path, bra_img_path, colors=None):
    if colors is None:
        colors = {
            'GATA3': "#FF00FF", 
            'SOX2': "#00FFFF",  
            'BRA': "#FFEE00"    
        }

    for key in colors:
        if isinstance(colors[key], str):
            colors[key] = hex_to_rgb01(colors[key])

    # Load grayscale images
    gata3 = np.array(Image.open(gata3_img_path).convert('L'), dtype=float)/255.0
    sox2  = np.array(Image.open(sox2_img_path).convert('L'), dtype=float)/255.0
    bra   = np.array(Image.open(bra_img_path).convert('L'), dtype=float)/255.0

    # Create colored images
    gata3_rgb = color_image(gata3, colors['GATA3'])
    sox2_rgb  = color_image(sox2, colors['SOX2'])
    bra_rgb   = color_image(bra, colors['BRA'])

    # Merge: pixel-wise max
    merge_rgb = np.maximum.reduce([gata3_rgb, sox2_rgb, bra_rgb])
    merge_rgb = np.clip(merge_rgb, 0, 1)

    return sox2_rgb, bra_rgb, gata3_rgb, merge_rgb

def plot_top_ids(condition, csv_path, images_dir, output_dir, top_n=5, selection='top'):
    df = pd.read_csv(csv_path)

    # Choose IDs
    if selection == 'top':
        selected = df.sort_values('final_score', ascending=False).head(top_n)
    elif selection == 'bottom':
        selected = df.sort_values('final_score', ascending=True).head(top_n)
    elif selection == 'middle':
        sorted_df = df.sort_values('final_score', ascending=False)
        mid_idx = len(sorted_df)//2
        start = max(0, mid_idx - top_n//2)
        selected = sorted_df.iloc[start:start+top_n]
    else:
        raise ValueError("selection must be 'top', 'bottom', or 'middle'")

    os.makedirs(output_dir, exist_ok=True)

    for _, row in selected.iterrows():
        ID = row['ID']
        # Build image paths
        gata3_img_path = f"{images_dir}/img_GATA3_{condition}/{ID:.0f}.tiff"
        sox2_img_path  = f"{images_dir}/img_SOX2_{condition}/{ID:.0f}.tiff"
        bra_img_path   = f"{images_dir}/img_BRA_{condition}/{ID:.0f}.tiff"

        sox2_rgb, bra_rgb, gata3_rgb, merge_rgb = overlay_channels(gata3_img_path, sox2_img_path, bra_img_path)

        # Create a single figure with 4 panels in a row
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        images = [sox2_rgb, bra_rgb, gata3_rgb, merge_rgb]
        titles = ['SOX2', 'BRA', 'GATA3', 'Merge']

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')

        plt.suptitle(f"ID: {ID}  |  Score: {row['final_score']:.2f}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        save_path = os.path.join(output_dir, f"{condition}_{selection}_{int(ID)}.png")
        plt.savefig(save_path)
        plt.close(fig)

def channels_plot_any(ID, directory, repeat, condition):

    images_dir = f"{directory}/{repeat}/boxes_tiff_selected"
    output_dir = f"{directory}/{repeat}/plots"
    
    gata3_img_path = f"{images_dir}/img_GATA3_{condition}/{ID:.0f}.tiff"
    sox2_img_path  = f"{images_dir}/img_SOX2_{condition}/{ID:.0f}.tiff"
    bra_img_path   = f"{images_dir}/img_BRA_{condition}/{ID:.0f}.tiff"

    sox2_rgb, bra_rgb, gata3_rgb, merge_rgb = overlay_channels(gata3_img_path, sox2_img_path, bra_img_path)

    # Create a single figure with 4 panels in a row
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    images = [sox2_rgb, bra_rgb, gata3_rgb, merge_rgb]
    titles = ['SOX2', 'BRA', 'GATA3', 'Merge']

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle(f"ID: {ID}, {condition}, R: {repeat}")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    save_path = os.path.join(output_dir, f"channelsplit_{int(ID)}_{condition}.png")
    plt.savefig(save_path)
    plt.close(fig)

def plot_gastruloid_scoring(directory, repeats, selection, conditions):
    for repeat in repeats:
        for condition in conditions:
            csv_path = f"{directory}/{repeat}/{condition}_overlap_scores_compiled.csv"
            images_dir = f"{directory}/{repeat}/boxes_tiff_selected"  # adjust to your actual image folder
            output_dir = f"{directory}/{repeat}/plots"
            plot_top_ids(condition, csv_path, images_dir, output_dir, top_n=5, selection=selection)

        

def DAPIIntensity_vs_score_scatterplot(directory, repeats, conditions):

    for repeat in repeats:
        for condition in conditions:

            scores_path = f"{directory}/{repeat}/{condition}_overlap_scores_compiled.csv"
            dapi_info_path = f"{directory}/{repeat}/distribution/{condition}_DAPI.csv"
            save_path = f"{directory}/{repeat}/plots/DAPIIntensity_vs_score_scatterplot_{condition}_.png"

            # Load CSVs
            scores_df = pd.read_csv(scores_path)  
            dapi_df = pd.read_csv(dapi_info_path)

            # Match IDs: ID 1 -> Index 0
            scores_df['DAPI_intensity'] = scores_df['ID'].apply(lambda x: dapi_df.loc[x-1, 'intensity'])

            # Scatter plot
            plt.figure(figsize=(6,5))
            plt.scatter(scores_df['final_score'], scores_df['DAPI_intensity'], color="#FF93BC", label='Data points')

            # Add trendline
            z = np.polyfit(scores_df['final_score'], scores_df['DAPI_intensity'], 1)  # linear fit
            p = np.poly1d(z)
            plt.plot(scores_df['final_score'], p(scores_df['final_score']), color="#7B2296", linestyle='--', label=f'Trendline (slope={z[0]:.2f})')

            plt.xlabel("Score")
            plt.ylabel("DAPI Intensity")
            plt.title(f"{condition} Final Score vs DAPI Intensity (R:{repeat})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

def DAPIIntensity_vs_score_scatterplot_combined(directory, repeats, conditions):
    """
    Plots DAPI intensity vs final score for each repeat.
    - Multiple conditions shown with different point colors.
    - One combined trendline for all conditions in the repeat.
    
    Args:
        directory (str): Base directory path.
        repeats (list): List of repeats to plot.
        conditions (list): List of conditions per repeat.
        point_colors (dict): Optional mapping {condition: color}.
        trendline_color (str): Color for the combined trendline.
    """

    point_colors = {
        "WT": "#FF93BC",
        "ND6": "#819EFF"
    }

    trendline_color="#7B2296"
    
    for repeat in repeats:
        plt.figure(figsize=(6,5))
        
        all_scores = []
        all_dapi = []

        for condition in conditions:
            scores_path = f"{directory}/{repeat}/{condition}_overlap_scores_compiled.csv"
            dapi_info_path = f"{directory}/{repeat}/distribution/{condition}_DAPI.csv"

            # Load CSVs
            scores_df = pd.read_csv(scores_path)  
            dapi_df = pd.read_csv(dapi_info_path)

            # Match IDs
            scores_df['DAPI_intensity'] = scores_df['ID'].apply(lambda x: dapi_df.loc[x-1, 'intensity'])

            # Store for combined trendline
            all_scores.extend(scores_df['final_score'])
            all_dapi.extend(scores_df['DAPI_intensity'])

            # Scatter plot for this condition
            plt.scatter(
                scores_df['final_score'], 
                scores_df['DAPI_intensity'], 
                color=point_colors.get(condition, None) if point_colors else None,
                label=condition
            )

        # Combined trendline
        z = np.polyfit(all_scores, all_dapi, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(all_scores), max(all_scores), 100)
        plt.plot(
            x_range, 
            p(x_range), 
            color=trendline_color, 
            linestyle='--', 
            linewidth=2,
            label=f"Combined trend (slope={z[0]:.2f})"
        )

        plt.xlabel("Gastruloid Score")
        plt.ylabel("DAPI Intensity")
        plt.title(f"Gastruloid Score vs DAPI Intensity (Repeat: {repeat})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        save_path = f"{directory}/{repeat}/plots/DAPIIntensity_vs_score_scatterplot_combined.png"
        plt.savefig(save_path)
        plt.close()

def DAPI_average_intensity(directory, repeats, conditions):
    """
    Calculates the average intensity from multiple CSV files and logs them into one output CSV.

    Parameters:
        directory (str): Base directory containing the data.
        repeats (list): List of repeat folder names.
        conditions (list): List of condition names.
        output_file (str): Path to save the output CSV.
    """
    results = []
    output_file=f"{directory}/meta_DAPIintensity.csv"

    for repeat in repeats:
        for condition in conditions:
            csv_path = f"{directory}/{repeat}/distribution/{condition}_DAPI.csv"
            
            df = pd.read_csv(csv_path)
            if 'intensity' not in df.columns:
                raise ValueError(f"CSV at {csv_path} must contain an 'intensity' column.")
            
            avg_intensity = df['intensity'].mean()
            results.append({"repeat": repeat, "condition": condition, "average_intensity": avg_intensity})

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Saved average intensities to {output_file}")

# Example usage
if __name__ == "__main__":


    csv_path = "nd6_overlap_scores_compiled.csv"
    images_dir = "CHIP_REPEATS_NEW/1/boxes_tiff_selected"  # adjust to your actual image folder
    plot_top_ids(csv_path, images_dir, top_n=5, selection='top')


def final_score_distribution(directory, repeats, conditions):
    # Define fixed bins from 0 to 100, step of 10
    bins = list(range(0, 101, 10))

    for repeat in repeats:
        for condition in conditions:
            csv_path = f"{directory}/{repeat}/{condition}_overlap_scores_compiled.csv"
            save_path = f"{directory}/{repeat}/plots/{condition}_reproducibility.png"

            if not os.path.exists(csv_path):
                print(f"Skipping missing file: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            bar_color = "#FF81CA"

            plt.hist(df['final_score'], bins=bins, color=bar_color, edgecolor='black')
            plt.xlabel('Gastruloid Score')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {condition} Gastruloid Scores (R: {repeat})')

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()



import pandas as pd
import matplotlib.pyplot as plt
import os

def final_score_distribution_combined(directory, repeats, conditions):
    """
    Plots combined histograms for all repeats per condition.
    Histograms are plotted in ascending order of mean final_score so that
    lower scoring repeats appear in front.
    """
    bins = list(range(0, 101, 10))  # fixed bins 0-100
    colors = ["#FF81CA", "#6EC1E4", "#FFD700"]  # custom colors

    for condition in conditions:
        plt.figure(figsize=(8, 6))

        # Load all repeats into a list with their mean final_score
        repeat_data = []
        for repeat in repeats:
            csv_path = f"{directory}/{repeat}/{condition}_overlap_scores_compiled.csv"
            if not os.path.exists(csv_path):
                print(f"Skipping missing file: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            mean_score = df['final_score'].mean()
            repeat_data.append((mean_score, repeat, df))

        # Sort repeats by mean_score ascending (lowest first)
        repeat_data.sort(key=lambda x: x[0])

        # Plot each repeat histogram in order
        for i, (_, repeat, df) in enumerate(repeat_data):
            plt.hist(
                df['final_score'],
                bins=bins,
                alpha=0.4,
                color=colors[i % len(colors)],
                edgecolor='black',
                label=f"Repeat {repeat}"
            )

        plt.xlabel('Gastruloid Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {condition} Gastruloid Scores (All Repeats)')
        plt.legend()

        save_path = f"{directory}/plots/{condition}_reproducibility_combined.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
