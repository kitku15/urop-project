from imports import *
from f_coordFinder import get_info
from f_modelDetection import load_allowed_ids, load_boxes
from f_validation import read_paths



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



def measure_blob_intensity_zones(idx, marker, image, mask, center, inner_r, mid_r, outer_r, print_info=False):
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

    
    # if marker == "SOX2":
    #     print("SOX2------------------")
    #     print("inner", mean_inner)
    #     print("mid", mean_mid)
    #     print("outer", mean_outer)
    

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




def measure_all_blob_intensities_zones(marker, img_boxes, mask_boxes, coordinates, inner_radius, mid_radius, outer_radius):
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

    for idx, (img, mask, center) in enumerate(zip(img_boxes, mask_boxes, coordinates)):
        inner, mid, outer = measure_blob_intensity_zones(idx, marker, img, mask, center, inner_radius, mid_radius, outer_radius)
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



def intensities_per_marker(repeat, condition, marker, coordinates, outer_r, mid_r, inner_r):

    for path in read_paths():
        parts = path.split("/")
        if int(parts[1]) == repeat:
            if marker in path:
                if condition in path:
                    # path example: blobs_npz/1/GATA3_WT.npz
                    print("------------------------Now processing:", path)
                    
                    repeat, condition, _ = get_info(path)

                    # get selected ids:
                    selected_boxes_ids = load_allowed_ids(f'selection/{repeat}/img_DAPI_{condition}.csv')
                    selected_boxes_ids.sort()
                    # print("box ids:", selected_boxes_ids)
                    # print("length", len(selected_boxes_ids))

                    # load blobs output path 
                    # data = np.load(path, allow_pickle=True)

                    mask_boxes_path = f"boxes_npz/{repeat}/mask_{marker}_{condition}.npz"
                    image_boxes_path = f"boxes_npz/{repeat}/img_{marker}_{condition}.npz"
                    
                    img_boxes = load_boxes(image_boxes_path)
                    mask_boxes = load_boxes(mask_boxes_path)


                    # # Extracting img_boxes from data
                    # img_boxes = data['img_boxes']
                    # mask_boxes = data['mask_boxes']
                    

                    # print("BEFORE FILTERING:------------")
                    # print("img boxes len", len(img_boxes))
                    # print("mask boxes len", len(mask_boxes))
                    # print("coordinates len", len(coordinates))

                    # FILTER IMG_BOX to only contain selected ones
                    filtered_img_boxes = [img_box for i, img_box in enumerate(img_boxes) if i in selected_boxes_ids]
                    filtered_mask_boxes = [mask_box for i, mask_box in enumerate(mask_boxes) if i in selected_boxes_ids]


                    # print("AFTER FILTERING:------------")
                    # print("img boxes len", len(filtered_img_boxes))
                    # print("mask boxes len", len(filtered_mask_boxes))
                    # print("coordinates len", len(coordinates))


                        # function to measure intensity and save 
                    def measure_intensity_and_save(inner_radius, mid_radius, outer_radius):

                        # get intensities 
                        inner_means, mid_means, outer_means = measure_all_blob_intensities_zones(marker, filtered_img_boxes, filtered_mask_boxes, coordinates, inner_radius, mid_radius, outer_radius)

                        # Save
                        inner_output_path = f"intensities/{repeat}/inner/{marker}_{condition}"
                        mid_output_path = f"intensities/{repeat}/mid/{marker}_{condition}"
                        outer_output_path = f"intensities/{repeat}/outer/{marker}_{condition}"
                                

                        inner_output_path_directory = os.path.dirname(inner_output_path)
                        os.makedirs(inner_output_path_directory, exist_ok=True)

                        mid_output_path_directory = os.path.dirname(mid_output_path)
                        os.makedirs(mid_output_path_directory, exist_ok=True)

                        outer_output_path_directory = os.path.dirname(outer_output_path)
                        os.makedirs(outer_output_path_directory, exist_ok=True)


                        np.save(inner_output_path, inner_means)
                        np.save(mid_output_path, mid_means)
                        np.save(outer_output_path, outer_means)
                        
                        if marker == "DAPI":
                            print(inner_means, mid_means, outer_means)

                        print("saved to:", inner_output_path, mid_output_path, outer_output_path)

                    # run function above
                    measure_intensity_and_save(inner_r, mid_r, outer_r)
            



def meta_intensities_save(repeat, condition, marker, outer, mid, inner):

    new_row = [repeat, condition, marker, outer, mid, inner]

    csv_file = 'intensities/meta_intensities.csv'

    # Check if file exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(['repeat', 'condition', 'marker', 'outer', 'mid', 'inner'])  

        writer.writerow(new_row)

def meta_intensities_save_individual(ID, repeat, condition, marker, outer, mid, inner):

    new_row = [ID, marker, outer, mid, inner]

    csv_file = f'intensities/meta_individual_{repeat}_{condition}.csv'

    # Check if file exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(['ID', 'marker', 'outer', 'mid', 'inner'])  

        writer.writerow(new_row)

