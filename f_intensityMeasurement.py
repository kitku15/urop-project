from imports import *
from r2_Radii_adjustment import read_paths, load_DAPI



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



def measure_blob_intensity(image, center, radius, print_info=False):
    """
    Measures mean intensity inside a circular mask in a single image.

    Parameters:
        image (2D ndarray): Grayscale cropped image.
        center (tuple): (x, y) center of the circle in pixel coordinates.
        radius (float): Radius of the circular mask.

    Returns:
        float: Mean intensity inside the circle, or None if center/radius invalid.
    """
    
    if print_info:
        print("center shape:", center.shape)
        print("center:",center)
        print("radius shape:",radius.shape)
        print("radius:",radius)


    if center is None or center.shape != (1, 2):
        return None

    h, w = image.shape
    cx, cy = center[0]
    r = radius

    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx)**2 + (yy - cy)**2 <= r**2

    values = image[mask]
    return np.mean(values) if values.size > 0 else None


def measure_all_blob_intensities(img_boxes, all_coordinates, radius):
    """
    Measures intensities for all blobs using a single radius value.

    Parameters:
        img_boxes (list of 2D ndarrays): Cropped grayscale images of blobs.
        all_coordinates (list of arrays): Each is a (1, 2) array for center.
        radius (float): Radius to use for all intensity measurements.

    Returns:
        list of float or None: Mean intensities for each blob.
    """
    intensities = [
        measure_blob_intensity(image, center, radius)
        for image, center in zip(img_boxes, all_coordinates)
    ]
    return intensities


def get_radius(csv_path, current_repeat, current_condition):

    df = pd.read_csv(csv_path)
    filtered_row = df[(df['repeat'] == current_repeat) & (df['condition'] == current_condition)]

    if not filtered_row.empty:
        outer_r = float(filtered_row['outer_r'].values[0])
        mid_r = float(filtered_row['mid_r'].values[0])
        inner_r = float(filtered_row['inner_r'].values[0])
        return outer_r, mid_r, inner_r

    raise ValueError(f"No radius data found for repeat={current_repeat}, condition={current_condition}")



def intensities_per_marker(marker, DAPI_coordinates, outer_r, mid_r, inner_r):

    for path in read_paths():
        if marker in path:
            # path example: blobs_npz/1/GATA3_WT.npz
            print("------------------------Now processing:", path)
            
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

            # load blobs output path 
            data = np.load(path, allow_pickle=True)

            # Extracting img_boxes from data
            img_boxes = data['img_boxes']

            # functino to measure intensity and save 
            def measure_intensity_and_save(level, radius):

                # get intensities 
                intensities = measure_all_blob_intensities(img_boxes, DAPI_coordinates, radius)

                # Save
                output_path = f"intensities/{repeat}/{level}/{marker}_{condition}"
                directory = os.path.dirname(output_path)
                os.makedirs(directory, exist_ok=True)
                np.save(output_path, intensities)

                print("saved to:", output_path)

            # run function above
            measure_intensity_and_save("outer", outer_r)
            measure_intensity_and_save("mid", mid_r)
            measure_intensity_and_save("inner", inner_r)
            

            
def safe_normalize(intensities, reference):
    """
    Replace None with 0, convert to arrays, then divide element‐wise.
    
    Parameters:
        intensities (list of floats or None)
        reference   (list of floats or single float or None)
    
    Returns:
        np.ndarray: result of intensities / reference, with None→0.
    """
    # If reference is a single value, broadcast it to the length of intensities
    if not isinstance(reference, (list, tuple, np.ndarray)):
        reference = [reference] * len(intensities)

    # Replace None with 0
    clean_i = [0.0 if v is None else v for v in intensities]
    clean_r = [0.0 if v is None else v for v in reference]

    arr_i = np.array(clean_i, dtype=float)
    arr_r = np.array(clean_r, dtype=float)

    # Perform element-wise division (wherever reference is zero, result will be inf or nan)
    return arr_i / arr_r

def donut_areas(outer_radius, mid_radius, inner_radius):
    outer_donut_area = math.pi * (outer_radius**2 - mid_radius**2)
    middle_donut_area = math.pi * (mid_radius**2 - inner_radius**2)
    inner_circle_area = math.pi * (inner_radius**2)

    return outer_donut_area, middle_donut_area, inner_circle_area



def meta_intensities_save(repeat, condition, marker, outer, mid, inner):

    new_row = [repeat, condition, marker, outer, mid, inner]

    csv_file = 'meta_intensities.csv'

    # Check if file exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(['repeat', 'condition', 'marker', 'outer', 'mid', 'inner'])  

        writer.writerow(new_row)

def safe_subtract(intensities, reference):
    """
    Replace None with 0, convert to arrays, then subtract element‐wise.
    
    Parameters:
        intensities (list of floats or None)
        reference   (list of floats or single float or None)
    
    Returns:
        np.ndarray: result of intensities - reference, with None→0.
    """
    # If reference is a single value, broadcast it to the length of intensities
    if not isinstance(reference, (list, tuple, np.ndarray)):
        reference = [reference] * len(intensities)

    # Replace None with 0
    clean_i = [0.0 if v is None else v for v in intensities]
    clean_r = [0.0 if v is None else v for v in reference]

    arr_i = np.array(clean_i, dtype=float)
    arr_r = np.array(clean_r, dtype=float)

    # Perform element-wise subtraction
    return arr_i - arr_r

def load_levels(marker, current_repeat, current_condition):
    
    inner = f"intensities/{current_repeat}/inner/norm_{marker}_{current_condition}.npy"
    mid = f"intensities/{current_repeat}/mid/norm_{marker}_{current_condition}.npy"
    outer = f"intensities/{current_repeat}/outer/norm_{marker}_{current_condition}.npy"

    intensity_inner = np.load(inner, allow_pickle=True)
    intensity_mid = np.load(mid, allow_pickle=True)
    intensity_outer = np.load(outer, allow_pickle=True)

    inner_bin = intensity_inner
    mid_bin = safe_subtract(intensity_mid, intensity_inner)
    outer_bin = safe_subtract(intensity_outer, intensity_mid)

    return inner_bin, mid_bin, outer_bin