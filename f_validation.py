from imports import *

def view_tiff(*layers):
    '''
    Displays multiple image-related layers in Napari, such as images, masks, and points.

    Parameters:
        *layers (tuple): A variable number of (name, data) tuples.
            - If `name` starts with "img", `data` is assumed to be an image array and will be added as an image layer.
            - If `name` starts with "mask", `data` is assumed to be a label/mask array and will be added as a labels layer.
            - If `name` starts with "points", `data` should be a tuple (coordinates, radii), where:
                - `coordinates` is an array of point locations.
                - `radii` is an array of radii for each point (used to set the size).

    Notes:
        - TIFF images should already be loaded using `tifffile.imread()` before being passed to this function.
        - The function automatically launches the Napari viewer and displays the layers with appropriate visual settings.
    '''
    viewer = napari.Viewer()

    for name, layer in layers:
        if name.startswith("img"):
            viewer.add_image(layer, name=name, colormap="gray_r")

        if name.startswith("mask"):
            labels = layer.astype(np.uint16)
            viewer.add_labels(labels, name=name, opacity=0.4)

        if name.startswith("points"):
            coordinates = layer[0]
            radii = layer[1]
            color = layer[2]
            viewer.add_points(coordinates, size=radii * 2, face_color=color, name='Blobs',  opacity=0.4,)

    napari.run()


def plot_radii_distribution(radii, output_filename):
    '''
    Plots and saves a histogram showing the distribution of blob radii.

    Parameters:
        radii (numpy.ndarray or list): Array or list of radii values (floats or ints) to be plotted.

    Returns:
        None

    Notes:
        - The histogram is saved as 'radii_distribution.png' in the current working directory.
        - Uses 30 bins, sky blue fill, and black edges for visual clarity.
        - Automatically closes the figure after saving to free memory.
    '''
    print("plotting radii distribution...")
    
    radii = np.concatenate([r for r in radii if r.size > 0])

    plt.figure(figsize=(8, 5))
    plt.hist(radii, bins=30, color='skyblue', edgecolor='black')

    plt.title('Distribution of Blob Radii')
    plt.xlabel('Radius')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(output_filename)
    plt.close()


def pad_to_shape(arr, target_shape):
    '''
    Pads a 2D array with zeros to match a target shape, centering the original array.

    Parameters:
        arr (numpy.ndarray): 2D input array to be padded.
        target_shape (tuple): Desired shape as (height, width).

    Returns:
        tuple:
            - padded (numpy.ndarray): The zero-padded array of shape `target_shape`.
            - (pad_left, pad_top) (tuple): The number of pixels the original array was shifted
              in the x and y directions, respectively, due to centering.

    Notes:
        - Padding is symmetric when possible; extra pixel goes to the bottom/right if dimensions are odd.
        - Assumes `arr.shape` is smaller than or equal to `target_shape` in both dimensions.
    '''
    pad_height = target_shape[0] - arr.shape[0]
    pad_width = target_shape[1] - arr.shape[1]

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded, (pad_left, pad_top)  # return x and y shifts


def view_single_box(box_no, img_boxes, mask_boxes, all_coordinates, whole_radii, big_radii, med_radii, small_radii):
    '''
    Displays a single box from image and mask arrays in Napari, along with detected blob points.

    Parameters:
        box_no (int): Index of the box to view.
        img_boxes (list): List of image box arrays (e.g. DAPI channel).
        mask_boxes (list): List of corresponding mask box arrays.
        all_coordinates (list): List of NumPy arrays containing (x, y) coordinates of blobs per box.
        all_radii (list): List of NumPy arrays containing blob radii per box.

    Returns:
        None

    Notes:
        - Transposes the image and mask for correct orientation before display.
        - The `view_tiff()` function is used to open the Napari viewer with layers:
            - Image (transposed)
            - Mask (transposed)
            - Blob points with radii visualized as blue circles
    '''
    print(f"Visualizing box {box_no}")


    layers = [
    (f"img_DAPI_box{box_no}_T", img_boxes[box_no].T),
    (f"mask_grid_box{box_no}_T", mask_boxes[box_no].T),
    (f"points_whole_radii{box_no}", [all_coordinates[box_no], whole_radii[box_no], "red"]),
    (f"points_big_radii{box_no}", [all_coordinates[box_no], big_radii[box_no], "yellow"]),
    (f"points_med_radii{box_no}", [all_coordinates[box_no], med_radii[box_no], "green"]),
    (f"points_small_radii{box_no}", [all_coordinates[box_no], small_radii[box_no], "cyan"]),
    ]

    view_tiff(*layers)

def check_list_size(all_coordinates, all_radii, img_boxes, mask_boxes):
    '''
    Verifies that all related lists (coordinates, radii, image boxes, mask boxes) are of equal length.

    Parameters:
        all_coordinates (list): List of blob coordinate arrays.
        all_radii (list): List of blob radii arrays.
        img_boxes (list): List of image box arrays.
        mask_boxes (list): List of mask box arrays.

    Returns:
        None

    Side Effects:
        - Prints a confirmation message if all lists have equal length.
        - Prints a warning ("uh oh") if a size mismatch is detected.

    Notes:
        - Useful for validating data consistency before running batch operations.
    '''

    coord_len = len(all_coordinates)
    radii_len = len(all_radii)
    img_box_len = len(img_boxes)
    mask_box_len = len(mask_boxes)

    if coord_len == radii_len == img_box_len == mask_box_len:
        print("Coordinates, Radii, and all boxes are of same length! proceed..")
    else:
        print("uh oh")

import numpy as np
import napari


def pad_to_shape(arr, target_shape):
    """Pad a 2D numpy array to a target shape (height, width)."""
    pad_h = target_shape[0] - arr.shape[0]
    pad_w = target_shape[1] - arr.shape[1]
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    return padded, (-pad_right, -pad_bottom)  # x, y shift

def load_allowed_ids(csv_path):
    allowed_ids = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2 and row[1].strip().lower() == 'yes':
                allowed_ids.append(int(row[0]))
    return allowed_ids

def slider_visual(selected_boxes_ids,
    img_boxes_list,
    mask_boxes_list,
    all_coordinates_list,
    outer_radius_list,
    mid_radius_list,
    inner_radius_list,
    labels=None,
    print_info=False
):
    """
    Visualize multiple sets of image/mask boxes and their coordinate-based blobs using napari.

    Parameters:
    - img_boxes_list: list of lists of np.ndarrays (each is a set of image boxes)
    - mask_boxes_list: list of lists of np.ndarrays (each is a set of mask boxes)
    - all_coordinates_list: list of lists of (N, 2) arrays with coordinates per image
    - outer_radius_list, mid_radius_list, inner_radius_list: list of radius lists per set
    - labels: Optional list of strings for naming each image/mask group
    - print_info: Print debug info
    """

    viewer = napari.Viewer()

    if labels is None:
        labels = [f"Set {i+1}" for i in range(len(img_boxes_list))]

    for set_idx, (img_boxes, mask_boxes, coordinates, outer_r, mid_r, inner_r, label) in enumerate(
        zip(img_boxes_list, mask_boxes_list, all_coordinates_list, outer_radius_list, mid_radius_list, inner_radius_list, labels)
    ):
        if print_info:
            print(f"\nProcessing {label}")

        # Transpose all boxes
        img_boxes = [box.T for box in img_boxes]
        mask_boxes = [box.T for box in mask_boxes]

        # Get max shape
        max_h = max(box.shape[0] for box in img_boxes)
        max_w = max(box.shape[1] for box in img_boxes)
        target_shape = (max_h, max_w)

        if print_info:
            print(f"{label}: Padding to {target_shape}")

        img_boxes_padded = []
        mask_boxes_padded = []
        adjusted_coords = []
        adjusted_outer = []
        adjusted_mid = []
        adjusted_inner = []

        # print("Mapping of Number on Napari to Image no.---------------")

        for id in range(len(selected_boxes_ids)):
            # print(f"{id}: {selected_boxes_ids[id]}")
                  
            padded_img, (shift_x, shift_y) = pad_to_shape(img_boxes[selected_boxes_ids[id]-1], target_shape)
            padded_mask, _ = pad_to_shape(mask_boxes[selected_boxes_ids[id]-1], target_shape)

            img_boxes_padded.append(padded_img)
            mask_boxes_padded.append(padded_mask)

            coords = coordinates[id]
            if len(coords) > 0:
                adj_coords = coords + np.array([shift_x, shift_y])
            else:
                adj_coords = coords  # empty

            adjusted_coords.append(adj_coords)
            adjusted_outer.append(outer_r[id])
            adjusted_mid.append(mid_r[id])
            adjusted_inner.append(inner_r[id])

            if print_info:
                print(f"  Box {id}: shift=({shift_y}, {shift_x}) coords adjusted")

        img_stack = np.stack(img_boxes_padded)
        mask_stack = np.stack(mask_boxes_padded).astype(np.uint8)

        viewer.add_image(img_stack, name=f'{label} - Images', colormap='gray')
        viewer.add_labels(mask_stack, name=f'{label} - Masks', opacity=0.4)

        def add_blobs(coords_list, radii_list, color, name):
            points = []
            sizes = []

            for t, (coords, radii) in enumerate(zip(coords_list, radii_list)):

                print(coords, radii)
                
                if coords.ndim == 1 and coords.shape[0] == 2:
                    coords = coords[np.newaxis, :]

                if radii.ndim == 0:
                    radii = np.array([radii])

                if len(coords) != len(radii):
                    print(f"!!! MISMATCH at time {t}: coords={len(coords)}, radii={len(radii)}")
                else:
                    print(f"OK at time {t}: coords={len(coords)}, radii={len(radii)}")

                if len(coords) > 0:
                    time_coords = np.column_stack((np.full(len(coords), t), coords))
                    points.append(time_coords)
                    sizes.append(radii * 2)  # Use diameter
            if points:
                all_points = np.concatenate(points, axis=0)
                all_sizes = np.concatenate(sizes, axis=0)
            else:
                all_points = np.empty((0, 3))
                all_sizes = np.empty((0,))

            print(">>> FINAL total points:", all_points.shape)
            print(">>> FINAL total sizes:", all_sizes.shape)


            viewer.add_points(
                all_points,
                size=all_sizes,
                name=f'{label} - {name}',
                face_color=color,
                border_width=0.0,
                opacity=0.2
            )

        print(len(adjusted_coords))
        print(len(adjusted_outer))


        add_blobs(adjusted_coords, adjusted_outer, 'purple', 'Outer Blobs')
        add_blobs(adjusted_coords, adjusted_mid, 'yellow', 'Mid Blobs')
        add_blobs(adjusted_coords, adjusted_inner, 'cyan', 'Inner Blobs')

    napari.run()


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




def plot_all_markers(marker_intensities, output):
    """
    Plot a grouped bar chart showing intensities for inner/mid/outer levels,
    with one bar per marker in each level group.
    """
    labels = ['inner', 'mid', 'outer']
    colors = ['#00bcd4', '#ffeb3b', '#9c27b0']  # can now be reused per marker if needed

    markers = list(marker_intensities.keys())
    n_markers = len(markers)
    n_levels = 3  # inner, mid, outer

    # Preprocess mean and std per marker
    means = {marker: [] for marker in markers}
    stds = {marker: [] for marker in markers}

    for marker in markers:
        bins = marker_intensities[marker]
        means[marker] = [np.nanmean(b) for b in bins]
        stds[marker] = [np.nanstd(b) for b in bins]

    # Plotting: x-axis is levels, each with multiple bars (one per marker)
    x = np.arange(n_levels)  # positions for inner, mid, outer
    width = 0.8 / n_markers  # shrink bar width to fit all markers

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, marker in enumerate(markers):
        offset = (i - n_markers / 2) * width + width / 2
        ax.bar(x + offset,
               means[marker],
               width,
               yerr=stds[marker],
               capsize=5,
               label=marker,
               alpha=0.8,
               edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average Normalized Intensity")
    ax.set_title("Intensity per Region Level (Grouped by Marker)")
    ax.legend(title="Marker")
    ax.grid(axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output)
    plt.close()

    print("Combined plot saved to:", output)


def load_DAPI(condition, printing=True):
    # open dapi npz
    for path in read_paths():
        if "DAPI" in path:
            if condition in path:
                DAPI_data = np.load(path, allow_pickle=True)
                # get info from path 
                stripped = os.path.splitext(path)[0]
                info = stripped.split("/")
                repeat = info[1]

                info_2 = info[2].split("_")
                marker = info_2[0]
                condition = info_2[1]

    # get DAPI info
    DAPI_coordinates = DAPI_data['all_coordinates']
    DAPI_radii = DAPI_data['all_radii']
    DAPI_mask_boxes = DAPI_data['mask_boxes']
    DAPI_img_boxes = DAPI_data['img_boxes']

    radii = np.concatenate([r for r in DAPI_radii if r.size > 0])
    max_radii = np.max(radii)
    if printing:
        print("biggest radius in DAPI list is:", max_radii)

    # Define the value to fill with (once already adjusted)
    outer_radius = 198.5618083164127
    mid_radius = 133.5618083164127
    inner_radius = 93.56180831641271

    # keep track of radius per repeat and condition 
    new_row = [repeat,condition,outer_radius,mid_radius,inner_radius]
    with open('radius.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_row)

    if printing:
        print("outer radius set to:", outer_radius)
        print("mid radius set to:", mid_radius)
        print("inner radius set to:", inner_radius)

    outer_radius_array = set_radius(DAPI_radii, outer_radius)
    mid_radius_array = set_radius(DAPI_radii, mid_radius)
    inner_radius_array = set_radius(DAPI_radii, inner_radius)

    return DAPI_coordinates, DAPI_mask_boxes, DAPI_img_boxes, outer_radius_array, mid_radius_array, inner_radius_array, outer_radius, mid_radius, inner_radius

def read_paths(filename="blob_output_paths.txt"):
    with open(filename, "r") as f:
        for line in f:
            path = line.strip()  # Remove trailing newline and any extra spaces
            if path:  # Skip empty lines if any
                yield path


def set_radius(all_radii, radius):
    all_radii = [np.array([radius]) if r.size > 0 else r for r in all_radii]
    return all_radii


