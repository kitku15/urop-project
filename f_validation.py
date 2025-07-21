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


def slider_visual(mask_boxes, img_boxes, all_coordinates, all_radii, print_info = False):

    print("Creating slider...")

    # transpose 
    for i in range(len(mask_boxes)):
        mask_boxes[i] = mask_boxes[i].T

    for i in range(len(img_boxes)):
        img_boxes[i] = img_boxes[i].T

    # print first to check
    if print_info:
        print("\n Inspecting Coordinates:")
        for i, item in enumerate(all_coordinates, start=1):
            print(f"ID {i}: coord = {item}")

        print("Image Boxes Shapes and IDs:")
        for i, item in enumerate(img_boxes, start=1):
            print(f"ID {i}: shape = {item.shape}")

        print("\nMask Boxes Shapes and IDs:")
        for i, item in enumerate(mask_boxes, start=1):
            print(f"ID {i}: shape = {item.shape}")


    # Step 1: Get max shape
    max_h = max(box.shape[0] for box in img_boxes)
    max_w = max(box.shape[1] for box in img_boxes)
    target_shape = (max_h, max_w)

    if print_info:
        print(f"Padding all boxes to shape: {target_shape}")

    # Step 4: Pad boxes and adjust coordinates
    img_boxes_padded = []
    mask_boxes_padded = []
    adjusted_coordinates = []
    adjusted_radii = []

    for i in range(len(img_boxes)):
        padded_img, (shift_x, shift_y) = pad_to_shape(img_boxes[i], target_shape)
        padded_mask, _ = pad_to_shape(mask_boxes[i], target_shape)

        img_boxes_padded.append(padded_img)
        mask_boxes_padded.append(padded_mask)

        coords = all_coordinates[i]
        rads = all_radii[i]

        if len(coords) > 0:
            adjusted_coords = coords + np.array([shift_x, shift_y])
        else:
            adjusted_coords = coords  # empty

        adjusted_coordinates.append(adjusted_coords)
        adjusted_radii.append(rads)

        if print_info:
            print(f"Box {i}: shift = ({shift_y}, {shift_x})")
            print(f" - First coord before: {coords[0] if len(coords) else 'None'}")
            print(f" - After adjust: {adjusted_coords[0] if len(coords) else 'None'}")



    # Step 3: Now you can stack
    img_stack = np.stack(img_boxes_padded)
    mask_stack = np.stack(mask_boxes_padded)

    # Initialize viewer with time axis as slider
    viewer = napari.Viewer()

    # Add image and mask with time axis (0th axis = "frame")
    viewer.add_image(img_stack, name='Image Boxes', colormap='gray')

    mask_stack = mask_stack.astype(np.uint8)  # or np.int32 if needed
    viewer.add_labels(mask_stack, name='Mask Boxes', opacity=0.7)


    # Add points: per-frame blobs
    # Napari wants (T, N, 2) shape for time-aware points
    points = []
    sizes = []

    for t, (coords, radii) in enumerate(zip(adjusted_coordinates, adjusted_radii)):  # ← use adjusted
        if len(coords) > 0:
            time_coords = np.column_stack((np.full(len(coords), t), coords))
            points.append(time_coords)
            sizes.append(radii * 2)  # Diameter, for visual sizing
        else:
            continue  # No blobs in this frame

    # Finalize arrays for Napari
    if points:
        all_points = np.concatenate(points, axis=0)  # shape: (total_blobs, 3)
        all_sizes = np.concatenate(sizes, axis=0)    # shape: (total_blobs,)
    else:
        all_points = np.empty((0, 3))
        all_sizes = np.empty((0,))

    viewer.add_points(all_points, size=all_sizes, name='Blobs', face_color='blue', opacity=0.5)


    napari.run()


def plot_intensities(intensities, output):

    # Filter out None values
    valid_intensities = [val for val in intensities if val is not None]

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(valid_intensities, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Blob Intensities")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
