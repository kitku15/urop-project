from imports import *
from f_coordFinder import get_info
from f_modelDetection import load_allowed_ids, load_boxes
from f_validation import read_paths
from scipy.interpolate import make_interp_spline




def normalize_tiff(tiff):
    '''
    Normalizes a TIFF image based on percentile scaling to enhance contrast.

    Parameters:
        tiff (numpy.ndarray): The input TIFF image as a NumPy array, typically 2D or 3D.

    Returns:
        numpy.ndarray: The normalized image with intensity values scaled between 0 and 1 
        using the 1st and 99.8th percentiles.

    Notes:
        - This is useful for preparing images for intensity-based measurements or visualization.
        - Percentile normalization reduces the impact of extreme outlier values.
        - Normalization is applied along the spatial axes (axis=(0, 1)).
    '''

    img_norm = normalize(tiff, pmin=1, pmax=99.8, axis=(0, 1))


    return img_norm



def measure_blob_intensity_zones(idx, marker, image, mask, center, inner_r, mid_r, outer_r, GATA3_mask, print_info=False):
    """
    Measures intensity in three non-overlapping radial zones: inner, mid ring, outer ring.

    Parameters:
        marker (str): Name of the channel/marker (e.g., "DAPI", "MarkerX").
        image (2D ndarray): Grayscale cropped image.
        mask (2D ndarray): Binary mask (same size as image), where 1 indicates valid signal.
        center (tuple): (x, y) center of the circles.
        inner_r (float): Radius of inner circle.
        mid_r (float): Radius of middle circle.
        outer_r (float): Radius of outer circle.
        print_info (bool): If True, prints debug info.

    Returns:
        tuple: (inner_mean, mid_bin_mean, outer_bin_mean)
    """

    h, w = image.shape
    cx, cy = center[0]

    yy, xx = np.ogrid[:h, :w]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2

    # Define radial bin masks
    mask_inner = dist_sq <= inner_r**2
    mask_mid = (dist_sq <= mid_r**2) & (~mask_inner)
    mask_outer = (dist_sq <= outer_r**2) & (~(mask_inner | mask_mid))

    # Total pixel counts in each zone
    area_inner = np.count_nonzero(mask_inner)
    area_mid = np.count_nonzero(mask_mid)
    area_outer = np.count_nonzero(mask_outer)

    # If marker is not "DAPI", apply binary intensity mask
    if marker != "DAPI":
        if image.shape != mask.shape:
            raise ValueError(f"Image and mask must have the same shape, got {image.shape} and {mask.shape}")

        # Apply signal mask: only count values where signal exists
        signal_inner = image * ((mask > 0) & mask_inner)
        signal_mid = image * ((mask > 0) & mask_mid)
        signal_outer = image * ((mask > 0) & mask_outer)

        # If marker is "GATA3" we apply the additional filter we made to filter out noise in gastruloid center
        if marker == "GATA3":
            signal_inner = image * ((mask > 0) & mask_inner & (GATA3_mask > 0))
            signal_mid = image * ((mask > 0) & mask_mid & (GATA3_mask > 0))
            signal_outer = image * ((mask > 0) & mask_outer & (GATA3_mask > 0))

        # Sum signal only (not mean over signal area)
        sum_inner = np.sum(signal_inner)
        sum_mid = np.sum(signal_mid)
        sum_outer = np.sum(signal_outer)

        # Normalize to total area of radial bin
        mean_inner = sum_inner / area_inner if area_inner > 0 else 0
        mean_mid = sum_mid / area_mid if area_mid > 0 else 0
        mean_outer = sum_outer / area_outer if area_outer > 0 else 0
    else:
        # For DAPI: no masking — use raw pixel intensities
        mean_inner = np.sum(image[mask_inner]) / area_inner if area_inner > 0 else 0
        mean_mid = np.sum(image[mask_mid]) / area_mid if area_mid > 0 else 0
        mean_outer = np.sum(image[mask_outer]) / area_outer if area_outer > 0 else 0

    def visualize_zones(marker, image, mask, center, inner_r, mid_r, outer_r, output_path):
        h, w = image.shape
        cx, cy = center[0]

        yy, xx = np.ogrid[:h, :w]
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2

        # Define masks
        mask_inner = dist_sq <= inner_r**2
        mask_mid = (dist_sq <= mid_r**2) & (~mask_inner)
        mask_outer = (dist_sq <= outer_r**2) & (~(mask_inner | mask_mid))

        if marker != "DAPI":
            if image.shape != mask.shape:
                raise ValueError(f"Image and mask must have the same shape, got {image.shape} and {mask.shape}")
            mask_inner &= (mask > 0)
            mask_mid &= (mask > 0)
            mask_outer &= (mask > 0)

        # Create RGB image to show regions
        overlay = np.stack([image]*3, axis=-1).astype(np.float32)  # Grayscale to RGB

        overlay[mask_inner] = [255, 0, 0]    # Red for inner
        overlay[mask_mid] = [0, 255, 0]      # Green for mid
        overlay[mask_outer] = [0, 0, 255]    # Blue for outer

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Save with matplotlib (or cv2.imwrite if preferred)
        plt.imsave(output_path, overlay)

        print(f"Saved overlay to: {output_path}")
    
    # if marker == "GATA3":
    #     os.makedirs("intensities/visualize", exist_ok=True)
    #     output_path = f'intensities/visualize/1_WT_{marker}_{idx+1}.png'
    #     visualize_zones(marker, image, mask, center, inner_r, mid_r, outer_r, output_path)

    return mean_inner, mean_mid, mean_outer




def measure_all_blob_intensities_zones(marker, img_boxes, mask_boxes, coordinates, inner_radius, mid_radius, outer_radius, GATA3_masks):
    """
    Measures intensities for all blobs using constant inner, mid, and outer radii.

    Parameters:
        img_boxes (list of 2D ndarrays): Cropped grayscale images of blobs.
        coordinates (list of arrays): Each is a (1, 2) array for the blob center.
        inner_radius (float): Radius for the inner zone.
        mid_radius (float): Radius for the mid zone.
        outer_radius (float): Radius for the outer zone.

    Returns:
        tuple of 3 lists: (inner_means, mid_means, outer_means)
    """
    inner_means = []
    mid_means = []
    outer_means = []


    for idx, (img, mask, center, GATA3_mask) in enumerate(zip(img_boxes, mask_boxes, coordinates, GATA3_masks)):
        inner, mid, outer = measure_blob_intensity_zones(idx, marker, img, mask, center, inner_radius, mid_radius, outer_radius, GATA3_mask)
        inner_means.append(inner)
        mid_means.append(mid)
        outer_means.append(outer)

    return inner_means, mid_means, outer_means



def get_radius(csv_path, current_repeat, current_condition):

    df = pd.read_csv(csv_path)
    filtered_row = df[(df['repeat'] == current_repeat) & (df['condition'] == current_condition)]

    if not filtered_row.empty:
        outer_r = float(filtered_row['outer_r'].values[0])
        mid_r = float(filtered_row['mid_r'].values[0])
        inner_r = float(filtered_row['inner_r'].values[0])
        return outer_r, mid_r, inner_r

    raise ValueError(f"No radius data found for repeat={current_repeat}, condition={current_condition}")



def intensities_per_marker(directory, repeats, conditions, markers, adjusting_values):

    outer_r = adjusting_values["DAPI"]
    mid_r = adjusting_values["BRA"]
    inner_r = adjusting_values["SOX2"]
                    
    for repeat in repeats:
        for condition in conditions:
            # Load coordinates 
            coor_output_dir = f"{directory}/{repeat}/coordinates"
            coordinates_path = f"{coor_output_dir}/{condition}.npz"

            data = np.load(coordinates_path)
            coordinates = data["coords"]

            for marker in markers:
                print(f"------------Repeat: {repeat}, Condition: {condition}, marker: {marker}")
                    
                # get selected ids:
                selected_boxes_ids = load_allowed_ids(f'{directory}/{repeat}/selection/img_DAPI_{condition}.csv')
                selected_boxes_ids.sort()

                mask_boxes_path = f"{directory}/{repeat}/boxes_npz/mask_{marker}_{condition}.npz"
                image_boxes_path = f"{directory}/{repeat}/boxes_npz/img_{marker}_{condition}.npz"
                    
                img_boxes = load_boxes(image_boxes_path)
                mask_boxes = load_boxes(mask_boxes_path)

                # FILTER IMG_BOX to only contain selected ones
                filtered_img_boxes = [img_box for i, img_box in enumerate(img_boxes) if i+1 in selected_boxes_ids]
                filtered_mask_boxes = [mask_box for i, mask_box in enumerate(mask_boxes) if i+1 in selected_boxes_ids]

                # GET GATA 3 MASKS
                loaded_GATA3masks = np.load(f"{directory}/{repeat}/GATA3filter/masks/{condition}.npz")
                GATA3_masks = [loaded_GATA3masks[key] for key in loaded_GATA3masks]  

                # function to measure intensity and save 
                def measure_intensity_and_save(inner_radius, mid_radius, outer_radius):

                    # get intensities 
                    inner_means, mid_means, outer_means = measure_all_blob_intensities_zones(marker, filtered_img_boxes, filtered_mask_boxes, coordinates, inner_radius, mid_radius, outer_radius, GATA3_masks)

                    # Save
                    inner_output_path = f"{directory}/{repeat}/intensities/inner/{marker}_{condition}"
                    mid_output_path = f"{directory}/{repeat}/intensities/mid/{marker}_{condition}"
                    outer_output_path = f"{directory}/{repeat}/intensities/outer/{marker}_{condition}"
                                

                    inner_output_path_directory = os.path.dirname(inner_output_path)
                    os.makedirs(inner_output_path_directory, exist_ok=True)

                    mid_output_path_directory = os.path.dirname(mid_output_path)
                    os.makedirs(mid_output_path_directory, exist_ok=True)

                    outer_output_path_directory = os.path.dirname(outer_output_path)
                    os.makedirs(outer_output_path_directory, exist_ok=True)


                    np.save(inner_output_path, inner_means)
                    np.save(mid_output_path, mid_means)
                    np.save(outer_output_path, outer_means)
                        
                    # if marker == "DAPI": dupa
                    #     print(inner_means, mid_means, outer_means)

                    print("saved to:", inner_output_path, mid_output_path, outer_output_path)

                # run function above
                measure_intensity_and_save(inner_r, mid_r, outer_r)
            



def meta_intensities_save(directory, repeat, condition, marker, outer, mid, inner):

    new_row = [repeat, condition, marker, outer, mid, inner]

    csv_file = f'{directory}/meta_intensities.csv'

    # Check if file exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(['repeat', 'condition', 'marker', 'outer', 'mid', 'inner'])  

        writer.writerow(new_row)

def meta_intensities_save_individual(directory, ID, repeat, condition, marker, outer, mid, inner):

    new_row = [ID, marker, outer, mid, inner]

    csv_file = f'{directory}/{repeat}/intensities/meta_individual_{condition}.csv'

    # Check if file exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(['ID', 'marker', 'outer', 'mid', 'inner'])  

        writer.writerow(new_row)


def normalize_markers(directory, repeats, conditions, markers):
    levels = ["outer", "mid", "inner"]

    for repeat in repeats:
        for condition in conditions:
            for marker in markers:
                outer = 0
                mid = 0
                inner = 0
                individually_normalized_outer = []
                individually_normalized_mid = []
                individually_normalized_inner = []
            
                for level in levels:
                    if marker != 'DAPI': 
                        print("------------------------Now processing:")
                        print("repeat:", repeat)
                        print("condition:", repeat)
                        print("level:", level)
                        print("marker:", marker)

                        # get paths
                        marker_path = f"{directory}/{repeat}/intensities/{level}/{marker}_{condition}.npy"
                        DAPI_path = f"{directory}/{repeat}/intensities/{level}/DAPI_{condition}.npy"

                        # load intensities 
                        print(f"loading intensities for {marker} and DAPI")
                        intensity_m = np.load(marker_path, allow_pickle=True)
                        intensity_DAPI = np.load(DAPI_path, allow_pickle=True)


                        for i in range(len(intensity_m)):
                            m = intensity_m[i]
                            d = intensity_DAPI[i]
                            
                            print(f"{i}-----------------------")
                            print("marker", m)
                            print("DAPI norm", d)

                            n = m/d

                            print("normalized", n)

                            if level == "outer":
                                individually_normalized_outer.append(n)
                            elif level == "mid":
                                individually_normalized_mid.append(n)
                            else:
                                individually_normalized_inner.append(n)
                        

                        # normalize 
                        if level == "outer":
                            normalized_manually = np.nanmean(individually_normalized_outer)
                            outer = normalized_manually
                        elif level == "mid":
                            normalized_manually = np.nanmean(individually_normalized_mid)
                            mid = normalized_manually
                        else:
                            normalized_manually = np.nanmean(individually_normalized_inner)
                            inner = normalized_manually


                        print(f"normalized manually for {level}:", normalized_manually)


                print(f"saving to csv for {marker}")
                meta_intensities_save(directory, repeat, condition, marker, outer, mid, inner)

                for i in range(1, len(individually_normalized_outer)+1):
                    meta_intensities_save_individual(directory, i, repeat, condition, marker, individually_normalized_outer[i-1], individually_normalized_mid[i-1], individually_normalized_inner[i-1])


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

        output_path = os.path.join(save_dir, f'intensities_plot_{condition}.png')
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

    output_path = os.path.join(save_dir, f'overlap_plot_{marker.upper()}.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot: {output_path}")
    print(f"AUC WT: {auc_wt:.3f}")
    print(f"AUC ND6: {auc_nd6:.3f}")
    print(f"Overlap AUC: {auc_overlap:.3f}")
    print(f"Overlap %: {overlap_percentage:.2f}%")


def variance_of_laplacian(image):
    """Compute the Laplacian of the image and return the variance."""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def detect_blurriness_in_parts(image, grid_size, blur_threshold):

    # Ensure grayscale
    if len(image.shape) == 2:
        gray = image
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    h, w = gray.shape
    num_rows, num_cols = grid_size
    cell_h, cell_w = h // num_rows, w // num_cols
    mask = np.zeros((h, w), dtype=np.uint8)

    for row in range(num_rows):
        for col in range(num_cols):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell = gray[y1:y2, x1:x2]

            score = variance_of_laplacian(cell)
            is_sharp = score >= blur_threshold
            if is_sharp:
                mask[y1:y2, x1:x2] = 1  # sharp = 1

    # Convert mask to 3-channel so we can stack it with original image
    mask_rgb = (mask * 255).astype(np.uint8)  # 0 or 255
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_GRAY2BGR)

    # Normalize original grayscale image
    image_bgr_float = image_bgr.astype(np.float32)
    image_bgr_float /= image_bgr_float.max()

    # Convert mask to float 0–1
    mask_rgb_float = mask_rgb.astype(np.float32) / 255.0

    # Stack them
    combined_float = np.hstack((image_bgr_float, mask_rgb_float))

    return combined_float, mask


def make_GATA3_filter(directory, repeats, conditions):
    marker = "GATA3"
    for repeat in repeats:
        for condition in conditions:
            print(f"------------Repeat: {repeat}, Condition: {condition}, marker: {marker}")

            # output directort for images (to visualize the mask)
            img_output_dir = f"{directory}/{repeat}/GATA3filter/images_{condition}"
            os.makedirs(img_output_dir, exist_ok=True)

            # get selected boxes of GATA3
            selected_boxes_ids = load_allowed_ids(f'{directory}/{repeat}/selection/img_DAPI_{condition}.csv')
            selected_boxes_ids.sort()

            image_boxes_path = f"{directory}/{repeat}/boxes_npz/img_{marker}_{condition}.npz"  
            img_boxes = load_boxes(image_boxes_path)
            filtered_img_boxes_with_index = [(i, img_box) for i, img_box in enumerate(img_boxes) if i+1 in selected_boxes_ids]

            masks = []
            all_box_bluriness = []
            for i, image in filtered_img_boxes_with_index:
                # calculate the total blurriness of the image
                box_blurriness = variance_of_laplacian(image)
                all_box_bluriness.append(box_blurriness)

            bluriness_average = sum(all_box_bluriness) / len(all_box_bluriness)
            blur_threshold = bluriness_average * 0.75

            for i, image in filtered_img_boxes_with_index:
                # detect blurriness and make mask
                grid_size = (30, 30)
                output_image, mask = detect_blurriness_in_parts(image, grid_size, blur_threshold)

                # save image (for visual validation side by side with image)
                savepath = os.path.join(img_output_dir, f"{i+1}.png")
                plt.imsave(savepath, output_image, cmap='gray')

                masks.append(mask)


            # save GATA3 masks
            mask_output_dir =  f"{directory}/{repeat}/GATA3filter/masks"
            os.makedirs(mask_output_dir, exist_ok=True)

            masks_path = f"{mask_output_dir}/{condition}.npz"
            np.savez_compressed(masks_path, *masks)
            print(f"Saved GATA3 filter mask for Repeat: {repeat}, Condition: {condition}")

            
            print(f"Bluriness for this set is {bluriness_average}")