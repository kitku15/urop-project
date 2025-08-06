from imports import *
from skimage import io, color, measure, exposure, img_as_ubyte
import seaborn as sns
from f_modelDetection import load_boxes, load_allowed_ids
from f_validation import load_DAPI, read_paths, slider_visual

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
        avg_intensity = np.nan
        total_intensity = np.nan
        normalized = np.array([])

    return avg_intensity, total_intensity, normalized, avg_intensity_raw

def get_binary_mask(area, sigma, blurred_thresh):
    '''
    preprocessing for regions props 
    
    '''

    # If RGBA, drop alpha and convert to grayscale
    if area.ndim == 3 and area.shape[2] == 4:
            area = area[:, :, :3]  # Drop alpha
            area = color.rgb2gray(area)

    # normalize before applying Gaussian
    normalized = (area - area.min()) / (area.max() - area.min())

    # Apply Gaussian blur
    blurred = gaussian(normalized, sigma=sigma)

    # Convert to binary mask
    binary_mask = blurred > blurred_thresh # can adjust

    labeled = measure.label(binary_mask) # Label connected components
    regions = measure.regionprops(labeled) # Measure properties
    largest_region = max(regions, key=lambda r: r.area)

    return largest_region, binary_mask


def get_coordinates(img_boxes, selection_csv, DAPI_coordinates, outer_radius_array):

    coordinates = []
    coordinates_ids = []

    # get the selected boxes and sort the list 
    selected_boxes_ids = load_allowed_ids(selection_csv)
    selected_boxes_ids.sort()

    # print(len(img_boxes))
    # print(len(selected_boxes_ids))
    # print(len(DAPI_coordinates))
    # print(DAPI_coordinates)
    # print(len(outer_radius_array))

    # print(DAPI_coordinates)
    print(selected_boxes_ids)

    default_coord = np.array([[212., 268.]])
    default_rad = np.array([198.56180832])


    

    # Get coordinates 
    counter = 0
    for i, img_box in enumerate(img_boxes):
        if i+1 in selected_boxes_ids:
            
            # Detect if the array is empty
            if DAPI_coordinates[counter].size == 0:
                print(f"[WARNING] Empty coordinate at index {i+1}, using default")
                DAPI_coordinates[counter] = default_coord
            if outer_radius_array[counter].size == 0:
                print(f"[WARNING] Empty Radii at index {i+1}, using default")
                outer_radius_array[counter] = default_rad

            # print(i+1)
            # print(DAPI_coordinates[counter])
            # print(DAPI_coordinates[counter][0])
            # print(outer_radius_array[counter])
            # print(outer_radius_array[counter][0])
            
            
            # use average outside intensity to set simple binary threshold 
            _, _, _, avg_intensity_raw = outside_intensity(img_box, DAPI_coordinates[counter][0], outer_radius_array[counter][0])
           
            thresh_int = 2400
            if avg_intensity_raw > thresh_int:
                binary_thresh = 0.25
            else:
                binary_thresh = 0.15
    
            counter = counter + 1
            img_box = img_as_float(img_box)

            # use binary mask to get centroid coordinates which are much more accurate than blob_log coordinates
            largest_region, binary_mask = get_binary_mask(img_box, 10, binary_thresh)
           
            coordinate = largest_region.centroid
            coordinates.append(coordinate)
            coordinates_ids.append(i+1)

    return coordinates, coordinates_ids, largest_region



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



def adjust(marker, condition, coordinates, outer_radius_array, mid_radius_array, inner_radius_array):
    for path in read_paths():
        if marker in path:
            if condition in path:
                # path example: blobs_npz/1/GATA3_WT.npz
                print("------------------------Now processing:", path)
                
                # get info from path:
                repeat, condition, _ = get_info(path)

                mask_boxes_path = f"boxes_npz/{repeat}/mask_{marker}_{condition}.npz"
                image_boxes_path = f"boxes_npz/{repeat}/img_{marker}_{condition}.npz"
                
                img_boxes = load_boxes(image_boxes_path)
                mask_boxes = load_boxes(mask_boxes_path)

                # get selected ids:
                selected_boxes_ids = load_allowed_ids(f'selection/{repeat}/img_DAPI_{condition}.csv')
                selected_boxes_ids.sort()
                print("box ids:", selected_boxes_ids)
                print("length", len(selected_boxes_ids))

                # open slider visual to visualize it
                print(f"opening slider visual to adjust for {marker}...")
                slider_visual(selected_boxes_ids,
                        img_boxes_list=[img_boxes],
                        mask_boxes_list=[mask_boxes],
                        all_coordinates_list=[coordinates],  
                        outer_radius_list=[outer_radius_array],
                        mid_radius_list=[mid_radius_array],
                        inner_radius_list=[inner_radius_array],
                        labels=[f"{marker}"],
                        print_info=False
                    )

def run_R2(repeat, marker, condition, adjusting=False):

    # LOADING ---------------------------------------------------------------------------------------------
    # image boxes
    image_boxes_path = f"boxes_npz/{repeat}/img_{marker}_{condition}.npz"
    img_boxes = load_boxes(image_boxes_path)

    # selected boxes csv 
    selection_output_dir = f"selection/{repeat}"
    selection_csv = f"{selection_output_dir}/img_DAPI_{condition}.csv"
    # -----------------------------------------------------------------------------------------------------


    # STEPS TO GET REFINED COORDINATES-----------------------------------------------------------------------
    # Get Raw Coordinates and Radius from DAPI model detection
    DAPI_coordinates, _, _, outer_radius_array, mid_radius_array, inner_radius_array, outer_radius, mid_radius, inner_radius = load_DAPI(condition)

    # Refine coordinates 
    coordinates, coordinates_ids, largest_region = get_coordinates(img_boxes, selection_csv, DAPI_coordinates, outer_radius_array)

    # print what we have to reconfirm, all should be the same length 
    print("coor len", len(coordinates))
    print("coor id len", len(coordinates_ids))
    print("DAPI coor len", len(DAPI_coordinates))
    print("radius array len", len(outer_radius_array))
    # -------------------------------------------------------------------------------------------------------


    # FIX EMPTY ARRAYS BY REFILLING (IDK WHY THIS HAPPENS)----------------------------------------------------
    outer_fill_value = np.array([outer_radius])
    mid_fill_value = np.array([mid_radius])
    inner_fill_value = np.array([inner_radius])

    # Replace empty arrays 
    cleaned_outer_radius_array = [x if x.size != 0 else outer_fill_value for x in outer_radius_array]
    cleaned_mid_radius_array = [x if x.size != 0 else mid_fill_value for x in mid_radius_array]
    cleaned_inner_radius_array = [x if x.size != 0 else inner_fill_value for x in inner_radius_array]
    # --------------------------------------------------------------------------------------------------------

    # convert coordinates (in tuple format) into numpy array format
    converted_coordinates = [np.array([[float(y), float(x)]]) for x, y in coordinates]

    if adjusting:
        adjust("DAPI", condition, converted_coordinates, cleaned_outer_radius_array, cleaned_mid_radius_array, cleaned_inner_radius_array)
        # adjust("SOX2", converted_coordinates, cleaned_outer_radius_array, cleaned_mid_radius_array, cleaned_inner_radius_array)
        # adjust("BRA", converted_coordinates, cleaned_outer_radius_array, cleaned_mid_radius_array, cleaned_inner_radius_array)
        adjust("GATA3", condition, converted_coordinates, cleaned_outer_radius_array, cleaned_mid_radius_array, cleaned_inner_radius_array)
    
    return converted_coordinates, outer_radius, mid_radius, inner_radius, largest_region







# def circularity_filter(area, c_thresh, box_id, filter_no):
#     '''
#     Apply circularity filter on small image: if blob not circular enough --> filter out!
#     '''
#     regions, binary_mask = area
#     circularity_pass = False
#     circular_blobs = []

#     # Filter by circularity
#     for region in regions:

#         if region.perimeter == 0: # to prevent division by zero 
#             continue

#         circularity = 4 * np.pi * region.area / (region.perimeter ** 2)

#         # SIZE FILTERS 
#         min_diameter = 50
#         diameter = region.equivalent_diameter

#         if circularity > c_thresh and min_diameter <= diameter:
#             circular_blobs.append((region, circularity))

#     # visualize dupa --------------------------
#     fig, ax = plt.subplots()
#     ax.imshow(binary_mask, cmap='gray')
#     ax.set_title(f'{len(circular_blobs)} Circular Blobs Detected')

#     if not circular_blobs:
#         ax.set_title('No Circular Blobs Detected')
#     else:
#         ax.set_title(f'{len(circular_blobs)} Circular Blob(s) Detected')
#         for region in circular_blobs:
#             y, x = region.centroid
#             diameter = region.equivalent_diameter
#             r = diameter / 2.0
#             # Draw the circle
#             circle = plt.Circle((x, y), r, edgecolor='red', fill=False, linewidth=1.5)
#             ax.add_patch(circle)
#             # Label the diameter on the image
#             ax.text(x, y, f'{circularity:.3f}', color='blue', fontsize=8, ha='center', va='center')
#     plt.axis('off')
#     plt.savefig(f'Circularity_Filter_output_{box_id}_filter{filter_no}')
#     plt.close()
#     # -----------------------------------------

#     if circular_blobs:
#         # Find the blob with the largest area
#         largest_blob, best_circularity = max(circular_blobs, key=lambda x: x[0].area)
#         if best_circularity > c_thresh:
#             return True, best_circularity
#         else: 
#             return False, 0
#     else:
#         return False, 0