from imports import *
from skimage import io, color, measure, exposure
import seaborn as sns
from f_modelDetection import load_boxes, load_allowed_ids
from f_validation import load_DAPI

def outside_intensity(image, coordinates_rescaled, radii_rescaled):
    '''
    Calculate average and total intensity outside the blob area over the full image,
    with intensities min-max normalized to [0, 1].

    Parameters:
        image (numpy.ndarray): Original image.
        coordinates_rescaled (numpy.ndarray): Array of shape (1, 2) with (x, y) coordinates of the blob center.
        radii_rescaled (numpy.ndarray): Array containing radius of the blob.

    Returns:
        tuple: (average_normalized_intensity, total_normalized_intensity, normalized_intensities_outside_blob)
    '''
    
    height, width = image.shape[:2]
    x_center, y_center = coordinates_rescaled
    radius = radii_rescaled

    Y, X = np.ogrid[:height, :width]
    dist_sq = (X - x_center)**2 + (Y - y_center)**2
    mask = dist_sq > radius**2  # True outside blob

    intensities_outside_blob = image[mask]

    if intensities_outside_blob.size > 0:
        # Min-max normalize intensities
        min_val = intensities_outside_blob.min()
        max_val = intensities_outside_blob.max()

        if max_val > min_val:
            normalized = (intensities_outside_blob - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(intensities_outside_blob)

        avg_intensity = normalized.mean()
        total_intensity = normalized.sum()

        # before normalization
        non_min_values = intensities_outside_blob[intensities_outside_blob > min_val]
        if non_min_values.size > 0:
            avg_intensity_raw = non_min_values.mean()
        else:
            avg_intensity_raw = intensities_outside_blob.mean()


    else:
        avg_intensity = np.nan
        total_intensity = np.nan
        normalized = np.array([])
        avg_intensity_raw = np.nan

    return avg_intensity, total_intensity, normalized, avg_intensity_raw

def get_binary_mask(area, sigma):

    # If RGBA, drop alpha and convert to grayscale
    if area.ndim == 3 and area.shape[2] == 4:
            area = area[:, :, :3]  # Drop alpha
            area = color.rgb2gray(area)

    # normalize before applying Gaussian
    normalized = (area - area.min()) / (area.max() - area.min())

    # Apply Gaussian blur
    blurred = gaussian(normalized, sigma=sigma)

    def regions_touch_border(binary_mask):
        '''
        function that, for a threshold, generates binary mask and finds regions touching border.

        '''
        labeled = measure.label(binary_mask)
        regions = measure.regionprops(labeled)
        height, width = binary_mask.shape

        if not regions:
            return False  # no regions at all
        
        # Find largest region by area
        largest_region = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = largest_region.bbox
        
        height, width = binary_mask.shape
        # Check if bounding box touches border
        if minr == 0 or minc == 0 or maxr == height or maxc == width:
            return True
        return False

    def find_best_slope_tol(image, slope_tol_range, sigma=2, spike_fraction=0.2):
        '''
        Iterate over a range of slope_tol, find threshold, make binary mask, test border touching
        '''
        best_slope_tol = None
        best_threshold = None
        
        # Normalize image as in your processing
        min_intensity = image.min()
        max_intensity = image.max()
        norm_image = (image - min_intensity) / (max_intensity - min_intensity)
        blurred = gaussian(norm_image, sigma=sigma)
        
        for slope_tol in slope_tol_range:
            threshold = find_curve_base_threshold(image, slope_tol=slope_tol, spike_fraction=spike_fraction)
            
            binary_mask = blurred > threshold
            if not regions_touch_border(binary_mask):
                best_slope_tol = slope_tol
                best_threshold = threshold
            else:
                # regions touch border - slope_tol too big probably, skip
                pass
        
        if best_slope_tol is None:
            # print("No slope_tol found without border-touching regions")
            # fallback - smallest slope_tol's threshold
            best_threshold = find_curve_base_threshold(image, slope_tol=slope_tol_range[0], spike_fraction=spike_fraction)
        
        return best_slope_tol, best_threshold
    
    def find_curve_base_threshold(image, slope_tol, spike_fraction=0.2):
        
        smooth_sigma = 2
        min_intensity = image.min()

        hist, bin_edges = np.histogram(image.ravel(), bins=400, range=(min_intensity, 1))
        
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=smooth_sigma)
        deriv = np.gradient(hist_smooth)


        max_count = hist_smooth.max()

        # Find index where histogram falls below a fraction of the max count
        spike_end_idx = 0
        for i, count in enumerate(hist_smooth):
            if count < max_count * spike_fraction:
                spike_end_idx = i
                break
        
        # Start searching after the spike ends
        start_idx = spike_end_idx if spike_end_idx > 0 else 10


        for idx in range(start_idx, len(deriv)):
            # Find first index where slope magnitude is below tolerance (flattening)
            if abs(deriv[idx]) < slope_tol:
                return bin_edges[idx]
        
        # Fallback if no flattening found
        # print("no flattening found")
        return 0.1
    
    def multi_stage_slope_tol_search(image, sigma, spike_fraction):

        max_slope = 500

        # Stage 0: Wide search with big steps (50)---------------------------------------------
        stage0_vals = np.arange(1, max_slope+1, 50)
        best_slope, best_thresh = find_best_slope_tol(image, stage0_vals, sigma=sigma, spike_fraction=spike_fraction)
        
        if best_slope is None:
            # print("No suitable slope_tol found in stage 1, falling back to default threshold")
            return 10, 0.2  # or any reasonable default slope_tol and threshold

        # Stage 1: Wide search with big steps (5)---------------------------------------------
        low = max(1, best_slope - 50)
        high = min(max_slope, best_slope + 50)
        stage1_vals = np.arange(low, high + 5, 5)
        best_slope, best_thresh = find_best_slope_tol(image, stage1_vals, sigma=sigma, spike_fraction=spike_fraction)
        
        if best_slope is None:
            # print("No suitable slope_tol found in stage 1, falling back to default threshold")
            return 10, 0.2  # or any reasonable default slope_tol and threshold
        
        # Stage 2: Narrow search +/- 5 around best from stage 1, steps of 1-------------------
        low = max(1, best_slope - 5)
        high = min(max_slope, best_slope + 5)
        stage2_vals = np.arange(low, high + 1, 1)

        best_slope, best_thresh = find_best_slope_tol(image, stage2_vals, sigma=sigma, spike_fraction=spike_fraction)
        
        if best_slope is None:
            # print("No suitable slope_tol found in stage 2, falling back to default threshold")
            return 10, 0.2
        
        # Stage 3: Narrow search +/- 1 around best from stage 2, steps of 0.2----------------------
        low = max(1, best_slope - 1)
        high = min(max_slope, best_slope + 1)
        stage3_vals = np.arange(low, high + 0.2, 0.2)
        best_slope, best_thresh = find_best_slope_tol(image, stage3_vals, sigma=sigma, spike_fraction=spike_fraction)
        
        if best_slope is None:
            # print("No suitable slope_tol found in stage 3, falling back to default threshold")
            return 10, 0.2
        
        # Stage 4: Narrow search +/- 0.5 around best from stage 2, steps of 0.05----------------------
        low = max(1, best_slope - 0.5)
        high = min(max_slope, best_slope + 0.5)
        stage4_vals = np.arange(low, high + 0.05, 0.05)
        best_slope, best_thresh = find_best_slope_tol(image, stage4_vals, sigma=sigma, spike_fraction=spike_fraction)
        
        if best_slope is None:
            # print("No suitable slope_tol found in stage 4, falling back to default threshold")
            return 10, 0.2
        
        return best_slope, best_thresh

    # get the best slope tol value (maximum with no regions touching border)
    best_slope, binary_thresh = multi_stage_slope_tol_search(blurred, sigma=2, spike_fraction=0.3)


    # print(f"Best slope_tol: {best_slope:.3f}, threshold: {binary_thresh:.3f}")


    # Convert to binary mask
    binary_mask = blurred > binary_thresh # can adjust

    labeled = measure.label(binary_mask) # Label connected components
    regions = measure.regionprops(labeled) # Measure properties
    largest_region = max(regions, key=lambda r: r.area)

    return largest_region, binary_mask, blurred, binary_thresh


def get_coordinates(img_boxes, selection_csv):

    coordinates = []
    coordinates_ids = []
    largest_region_list = []
    binary_mask_list = []

    # get the selected boxes and sort the list 
    selected_boxes_ids = load_allowed_ids(selection_csv)
    selected_boxes_ids.sort()

    # Get coordinates 
    print("Making Binary masks----------------------")
    # Filter only selected boxes for processing
    box_args = [(i, img_box) for i, img_box in enumerate(img_boxes) if i+1 in selected_boxes_ids]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_binarymask, box_args))  # preserves original order of box_args

    # Now unpack results in order
    for idx, coordinate, largest_region, binary_mask in results:
        coordinates.append(coordinate)
        coordinates_ids.append(idx)
        largest_region_list.append(largest_region)
        binary_mask_list.append(binary_mask)

    print("Saved all Binary masks----------------------")
    return coordinates, coordinates_ids, largest_region_list, binary_mask_list

def process_binarymask(args):
    i, img_box = args
    img_box = img_as_float(img_box)
    largest_region, binary_mask, _, _ = get_binary_mask(img_box, 10)
    coordinate = largest_region.centroid
    return (i+1, coordinate, largest_region, binary_mask)



def get_info(path):
    
    # get info from path 
    stripped = os.path.splitext(path)[0]
    info = stripped.split("/")
    repeat = info[1]
    info_2 = info[2].split("_")
    marker = info_2[0]
    condition = info_2[1]

    # print info obtained
    print("repeat:", repeat)
    print("condition:", condition)
    print("marker:", marker)
        
    return repeat, condition, marker

def adjust_binaryMask(directory, repeats, conditions):
    '''
    Adjust for Binary mask threshold based on DAPI. This is a pain but we have to do it until I find a better alternative
    '''
            
    for repeat in repeats:
        for condition in conditions:
                
            image_boxes_path = f"{directory}/{repeat}/boxes_npz/img_DAPI_{condition}.npz"
            img_boxes = load_boxes(image_boxes_path)

            # selected boxes csv 
            selection_output_dir = f"{directory}/{repeat}/selection"
            selection_csv = f"{selection_output_dir}/img_DAPI_{condition}.csv"
            # get selected ids:
            selected_boxes_ids = load_allowed_ids(selection_csv)
            selected_boxes_ids.sort()


            # Save binary mask images next to raw image
            for i, img_box in enumerate(img_boxes):
                if i+1 in selected_boxes_ids:
                    # print(f"Starting {i+1} for Binary thresh adjustment----------------")

                                                
                    img_box = img_as_float(img_box)

                    # Get the binary maks with smoothing value set below
                    smoothing = 10
                    largest_region, binary_mask, blurred, binary_thresh = get_binary_mask(img_box, smoothing)

                        
                    # Visualize binary mask ---------------------------------
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                    # Show original image (img_box)
                    axes[0].imshow(img_box, cmap='gray')
                    axes[0].set_title('Original Image')

                    # Mark centroid
                    cy, cx = largest_region.centroid  # centroid is (row, col) = (y, x)
                    axes[0].scatter(cx, cy, s=50, edgecolor='red', facecolor='none', linewidth=2, label='Centroid')

                    axes[0].axis('off')
                    axes[0].legend(loc='upper right')

                    # Show binary mask (mask is boolean, convert to int for visualization)
                    axes[1].imshow(binary_mask, cmap='gray')
                    axes[1].set_title(f'Binary Mask {binary_thresh:.3f}')
                    axes[1].axis('off')

                    # Plot histogram on third subplot
                    axes[2].hist(blurred.ravel(), bins=256, range=(0, 1), color='gray')
                    axes[2].axvline(x=binary_thresh, color='red', linestyle='--', label=f'Threshold = {binary_thresh:.3f}')
                    axes[2].set_title("Pixel Intensity Histogram")
                    axes[2].set_xlabel("Pixel Intensity")
                    axes[2].set_ylabel("Number of Pixels")
                    axes[2].legend()

                    # Make output directory if it doesn't exist
                    output_dir = f'{directory}/{repeat}/binary_adj/{condition}'
                    os.makedirs(output_dir, exist_ok=True)

                    # Save figure with a meaningful name (e.g., box index)
                    filename = f'{i+1}.png'
                    filepath = os.path.join(output_dir, filename)
                    plt.savefig(filepath, bbox_inches='tight')
                    plt.close(fig)  # Close the figure to save memory
                    print(f"saved {i+1} for Binary thresh adjustment")

                    #----------------------------------------------------------------

                    return largest_region, binary_mask, blurred, binary_thresh

    

def run_R2(directory, repeats, conditions, adjusting_values, adjusting=False):


    for repeat in repeats:
        for condition in conditions:

            print(f"Repeat: {repeat}, Condition: {condition}")

            outer_radius = adjusting_values["DAPI"]
            mid_radius = adjusting_values["BRA"]
            inner_radius = adjusting_values["SOX2"]

            # LOADING DAPI---------------------------------------------------------------------------------------------
            # DAPI is used to adjust for coordinates. 
            image_boxes_path = f"{directory}/{repeat}/boxes_npz/img_DAPI_{condition}.npz"
            img_boxes = load_boxes(image_boxes_path)

            # selected boxes csv 
            selection_output_dir = f"{directory}/{repeat}/selection"
            selection_csv = f"{selection_output_dir}/img_DAPI_{condition}.csv"
            # get selected ids:
            selected_boxes_ids = load_allowed_ids(selection_csv)
            selected_boxes_ids.sort()

            # STEPS TO GET REFINED COORDINATES-----------------------------------------------------------------------
    
            # Refine coordinates 
            coordinates, coordinates_ids, largest_region_list, binary_mask_list = get_coordinates(img_boxes, selection_csv)

            # print what we have to reconfirm, all should be the same length 
            print("All these below should be the same length:-----------------------")
            print("coor len", len(coordinates))
            print("coor id len", len(coordinates_ids))

            # FIX EMPTY ARRAYS BY REFILLING (IDK WHY THIS HAPPENS)----------------------------------------------------
            outer_fill_value = np.array([outer_radius])
            mid_fill_value = np.array([mid_radius])
            inner_fill_value = np.array([inner_radius])

            # --------------------------------------------------------------------------------------------------------

            # convert coordinates (in tuple format) into numpy array format
            converted_coordinates = [np.array([[float(y), float(x)]]) for x, y in coordinates]


            def adjust(marker_adjusting, outer_radius, mid_radius, inner_radius):

                marker_boxes_path = f"{directory}/{repeat}/boxes_npz/img_{marker_adjusting}_{condition}.npz"
                marker_boxes = load_boxes(marker_boxes_path)


                # Get coordinates 
                counter = 0
                for i, marker_box in enumerate(marker_boxes):
                    if i+1 in selected_boxes_ids:

                        region = largest_region_list[counter]
                        
                        marker_box = img_as_float(marker_box)
                    
                        # Visualize binary mask ---------------------------------
                        fig, ax = plt.subplots(figsize=(8, 4))

                        # Use ax[0] for the original image and annotations
                        ax.imshow(marker_box, cmap='gray')
                        ax.set_title(f"{marker_adjusting}_{outer_radius:.1f}_{mid_radius:.1f}_{inner_radius:.1f}")

                        cy, cx = region.centroid
                        ax.scatter(cx, cy, s=50, edgecolors='red', facecolors='none', linewidth=2, label='Centroid')

                        # make circles
                        outer_circle = patches.Circle((cx, cy), radius=outer_radius, edgecolor='#9c27b0', facecolor='none', linewidth=2)
                        mid_circle = patches.Circle((cx, cy), radius=mid_radius, edgecolor='#ffeb3b', facecolor='none', linewidth=2)
                        inner_circle = patches.Circle((cx, cy), radius=inner_radius, edgecolor='#00bcd4', facecolor='none', linewidth=2)
                        
                        ax.add_patch(outer_circle)
                        ax.add_patch(mid_circle)
                        ax.add_patch(inner_circle)

                        ax.axis('off')

                        output_dir = f'{directory}/{repeat}/adjusting/{condition}_{marker_adjusting}'
                        os.makedirs(output_dir, exist_ok=True)

                        filename = f'{i+1}.png'
                        filepath = os.path.join(output_dir, filename)
                        plt.savefig(filepath, bbox_inches='tight')
                        plt.close(fig)
                        #----------------------------------------------------------------
                        counter = counter + 1


                print(f"Finished adjusting for {marker_adjusting}")


            if adjusting:
                for key, _ in adjusting_values.items():
                    adjust(key, outer_radius, mid_radius, inner_radius)
            
            # save coordinates
            coor_output_dir = f"{directory}/{repeat}/coordinates"
            os.makedirs(coor_output_dir, exist_ok=True)

            coordinates_path = f"{coor_output_dir}/{condition}.npz"
            np.savez(coordinates_path, coords=converted_coordinates)

            # save region (gastruloid area)
            regions_output_dir = f"{directory}/{repeat}/regions"
            os.makedirs(regions_output_dir, exist_ok=True)

            regions_path = f"{regions_output_dir}/{condition}.npz"
            np.savez(regions_path, regions=largest_region_list)
            
            # save binary mask 
            binarymasks_output_dir = f"{directory}/{repeat}/binarymasks"
            os.makedirs(binarymasks_output_dir, exist_ok=True)

            binarymasks_path = f"{binarymasks_output_dir}/{condition}.npz"
            np.savez(binarymasks_path, binarymasks=np.array(binary_mask_list, dtype=object))


            
