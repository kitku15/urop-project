from imports import *
from f_preprocessing import reload_cropped


def draw_grid(img):
    '''
    Divides an input image into a 26x26 grid and assigns a unique integer label to 
    each grid cell.

    Parameters:
        img (numpy.ndarray): The input image as a 2D (grayscale) or 3D (color) NumPy array.
    Returns:
        numpy.ndarray: A 2D NumPy array (grid_mask) of the same height and width as the input 
        image, where each element contains a unique label corresponding to its grid cell 
        (ranging from 1 to 676).
    
    Notes:
        - The image is divided evenly into 26 rows and 26 columns.
        - If the image dimensions are not perfectly divisible by 26, the last row and column 
        will absorb the remainder to ensure full coverage.
    '''
    
    image_shape_yx = img.shape

    # Define grid dimensions
    grid_rows, grid_cols = 26, 26
    
    # Define the height and width of each box
    box_height = image_shape_yx[0] // grid_rows
    box_width = image_shape_yx[1] // grid_cols

    # Create coordinate grids
    grid_mask = np.zeros(image_shape_yx, dtype=np.uint16)

    label = 1  # Start labeling from 1

    for i in range(grid_rows): 
        for j in range(grid_cols):
            # Define box boundaries
            start_y = i * box_height
            end_y = (i + 1) * box_height if i < grid_rows - 1 else image_shape_yx[0]

            start_x = j * box_width
            end_x = (j + 1) * box_width if j < grid_cols - 1 else image_shape_yx[1]

            # Label the region
            grid_mask[start_y:end_y, start_x:end_x] = label
            label += 1

    print(f"\nMade grid")

    return grid_mask


def save_blobs(filename, blobs, coordinates, radii):
    '''
    Saves blob detection results to a compressed `.npz` file.

    Parameters:
        filename (str): Path to the output `.npz` file (should end with `.npz`).
        blobs (numpy.ndarray): Array containing blob detection results (e.g., combined data or metadata).
        coordinates (numpy.ndarray): Array of (y, x) or (z, y, x) positions of detected blobs.
        radii (numpy.ndarray): Array of radii corresponding to each blob.

    Returns:
        None

    Notes:
        - The data can be loaded later using `np.load(filename)`.
        - Useful for saving intermediate or final results of blob detection workflows.
    '''
    np.savez(filename, blobs=blobs, coordinates=coordinates, radii=radii)



def load_blobs(filename):
    '''
    Loads blob detection data from a `.npz` file saved using `save_blobs`.

    Parameters:
        filename (str): Path to the `.npz` file containing blob data.

    Returns:
        tuple:
            - blob (numpy.ndarray): The original blob data array.
            - coordinates (numpy.ndarray): Array of blob positions (e.g., (y, x) or (z, y, x)).
            - radii (numpy.ndarray): Array of radii corresponding to each blob.

    Notes:
        - The `.npz` file must have been saved using the `save_blobs` function format.
        - `allow_pickle=True` is used to support loading any object arrays, if present.
    '''
    data = np.load(filename, allow_pickle=True)
    blob = data['blobs']
    coordinates = data['coordinates']
    radii = data['radii']
    
    return blob, coordinates, radii


def detect_blobs(tiff, downscale_factor=0.25, sigma=2, min_sigma=20, max_sigma=50, exclude_border=10):
    '''
    Detects blobs in a TIFF image using the Laplacian of Gaussian (LoG) method after downscaling.

    Parameters:
        tiff (numpy.ndarray): Input image as a NumPy array.
        downscale_factor (float, optional): Factor by which to downscale the image before detection 
            (default is 0.25, i.e., image is resized to 25% of original size).

    Returns:
        tuple:
            - blob (numpy.ndarray): Array of shape (1, 3) containing the largest detected blob's (y, x, sigma).
              Empty array if no blobs detected.
            - coordinates_rescaled (numpy.ndarray): Array of shape (1, 2) with (x, y) coordinates of the largest blob,
              rescaled to original image size.
            - radii_rescaled (numpy.ndarray): Array containing the radius of the largest blob, rescaled to original size.

    Notes:
        - The input image is downscaled and smoothed with a Gaussian filter before blob detection.
        - The largest blob by radius (sigma) is selected from all detected blobs.
        - Coordinates are converted from (y, x) to (x, y) and scaled back to the original image dimensions.
        - If no blobs are detected, returns empty arrays.
    '''

    downsampled = rescale(tiff, downscale_factor, anti_aliasing=True)  # 4x smaller

    # #invert
    # tiff = invert(tiff)
    
    # Convert to float and normalize
    tiff = img_as_float(downsampled)


    # smooth
    smoothed = gaussian(tiff, sigma)

    # Detect blobs
    blobs = blob_log(image=smoothed, min_sigma=min_sigma, max_sigma=max_sigma, exclude_border=exclude_border)

    if blobs.shape[0] == 0:
        return np.empty((0, 3)), np.empty((0, 2)), np.array([])
    
    # Select the blob with the largest radius (sigma)
    largest_blob_idx = np.argmax(blobs[:, 2])
    blob = blobs[largest_blob_idx].reshape(1, -1)  # shape (1,3)

    # blobs: Nx3 array of (y, x, sigma)
    coordinates = blob[:, :2][:, ::-1]  # convert (y,x) to (x,y)
    radii = blob[:, 2] * np.sqrt(2)

    # Rescale coordinates and radii to match original image
    scale = 1 / downscale_factor
    coordinates_rescaled = coordinates * scale
    radii_rescaled = radii * scale

    return blob, coordinates_rescaled, radii_rescaled


def analyse_by_grid(tiff, grid_mask, marker, condition, repeat, downscale_factor=0.25, mask=True, rescale_switch=True):
    '''
    Analyzes an image by dividing it into grid regions and extracting each region in parallel,
    optionally downscaling before processing and upscaling the results afterward.

    Parameters:
        tiff (numpy.ndarray): The input TIFF image as a NumPy array (2D or 3D).
        grid_mask (numpy.ndarray): A 2D mask with labeled grid regions (same spatial size as `tiff`).
        downscale_factor (float, optional): Factor by which to downscale the image and grid for faster processing (default: 0.25).
        mask (bool, optional): If True, indicates the input is a mask; otherwise treated as an image (default: True).
        rescale_switch (bool, optional): If True, downscales before processing and upscales the cropped boxes afterward (default: True).

    Returns:
        list: A list of cropped (and optionally upscaled) NumPy arrays, one per grid region.

    Side Effects:
        - Saves the resulting list of image/mask patches as a compressed `.npz` file named:
          `"rescale_mask_boxes.npz"`, `"NOrescale_img_boxes.npz"`, etc., based on the options used.

    Notes:
        - Uses multithreading to accelerate both cropping and upscaling steps.
        - Each grid cell is processed individually using the `crop_and_save()` function.
        - If `rescale_switch` is enabled, nearest-neighbor interpolation is used for resizing `grid_mask`.
        - `upscale_box()` must return data compatible with `.npz` saving (e.g., NumPy arrays).
    '''

    # settings for rescale 
    if rescale_switch:
        tiff_small = rescale(tiff, downscale_factor, anti_aliasing=True)
        grid_mask_small = rescale(grid_mask.astype(float), downscale_factor, order=0, anti_aliasing=False).astype(np.uint16)
    else:
        tiff_small = tiff
        grid_mask_small = grid_mask.astype(np.uint16)

    unique_boxes = np.unique(grid_mask_small)

    # make and set all directory paths 
    unique_boxes = np.unique(grid_mask_small)
    npz_directory = f"boxes_npz/{repeat}"
    tiff_directory =f"boxes_tiff/{repeat}"
    os.makedirs(npz_directory, exist_ok=True)
    os.makedirs(tiff_directory, exist_ok=True)
    heading_npz = f"{npz_directory}/mask_{marker}_{condition}" if mask else f"{npz_directory}/img_{marker}_{condition}"
    heading_tiff = f"{tiff_directory}/mask_{marker}_{condition}" if mask else f"{tiff_directory}/img_{marker}_{condition}"

    # set scale status, not really used but Ill keep this for now
    scale_status = "rescale" if rescale_switch else "NOrescale"

    # Crop and save tiffs individually (slow but good for manual visual checking)
    args_list = [(box_id, tiff_small, grid_mask_small, heading_tiff) for box_id in unique_boxes]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        cropped_boxes = list(executor.map(crop_and_save, args_list))

    # Parallel upscale only if rescaling was done
    if rescale_switch:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(upscale_box, cropped, downscale_factor) for cropped in cropped_boxes]
            upscaled_boxes = [future.result() for future in concurrent.futures.as_completed(futures)]
    else:
        upscaled_boxes = cropped_boxes

    # save npz file
    save_path = f"{heading_npz}_{scale_status}_boxes.npz"
    np.savez_compressed(save_path, *upscaled_boxes)
    print(f"Saved {len(upscaled_boxes)} boxes to {save_path}")
    print(f"Boxes done")
    return upscaled_boxes


def crop_and_save(args):
    '''
    Crops a specific region from a downscaled image based on a grid mask and saves it as a PNG.

    Parameters:
        args (tuple): A tuple containing:
            - box_id (int): The unique label of the grid cell to crop.
            - tiff_small (numpy.ndarray): The downscaled image or mask.
            - grid_mask_small (numpy.ndarray): The downscaled grid mask with labeled regions.
            - heading (str): A string indicating the data type ('img' or 'mask'), used for folder naming.

    Returns:
        numpy.ndarray: The cropped image/mask corresponding to the grid cell.

    Side Effects:
        - Saves the cropped region as a PNG image in the path: `boxes/{heading}/{box_id}.png`
        - Prints confirmation when the box is saved.

    Notes:
        - Assumes the output directory `boxes/{heading}/` already exists.
        - The image is saved as 8-bit PNG after scaling to [0, 255].
    '''
    box_id, tiff_small, grid_mask_small, heading = args
    box_mask = grid_mask_small == box_id
    ys, xs = np.where(box_mask)
    ymin, ymax = ys.min(), ys.max() + 1
    xmin, xmax = xs.min(), xs.max() + 1
    cropped_small = tiff_small[ymin:ymax, xmin:xmax]

    os.makedirs(f"boxes_tiffs/{heading}", exist_ok=True)

    plt.imsave(f"boxes_tiffs/{heading}/{box_id}.tiff", cropped_small, cmap='gray') 
    print(f"added Box {box_id}")
    return cropped_small


def upscale_box(cropped, downscale_factor):
    '''
    Upscales a cropped image or mask back to its original resolution using the specified downscale factor.

    Parameters:
        cropped (numpy.ndarray): The downscaled cropped image or mask.
        downscale_factor (float): The factor by which the original image was downscaled (e.g., 0.25 for 4× downscaling).

    Returns:
        numpy.ndarray: The upscaled image or mask, resized to approximate original dimensions.

    Notes:
        - Uses `skimage.transform.resize` with anti-aliasing enabled for smooth interpolation.
        - Output dimensions are calculated as (height / factor, width / factor).
        - If precise alignment with original resolution is critical, consider rounding or padding post-resize.
    '''
    target_shape = (int(cropped.shape[0] / downscale_factor), int(cropped.shape[1] / downscale_factor))
    return resize(cropped, target_shape, anti_aliasing=True)

def load_boxes(path):
    '''
    Loads all arrays from a compressed `.npz` file and returns them as a list.

    Parameters:
        path (str): Path to the `.npz` file containing saved image or mask boxes.

    Returns:
        list: A list of NumPy arrays, one for each saved box.

    Notes:
        - The file must be saved using `np.savez` or `np.savez_compressed`.
        - Uses `allow_pickle=True` to support loading object arrays if present.
        - Preserves the original order of keys as stored in the `.npz` file.
    '''
    data = np.load(path, allow_pickle=True)
    return [data[key] for key in data]


def process_box(args):
    '''
    Detects blobs in a single image box, saves the results, and returns coordinates and radii.

    Parameters:
        args (tuple): A tuple containing:
            - box_id (int): The unique identifier for the box (used in the output filename).
            - box (numpy.ndarray): The image data for the box to process.

    Returns:
        tuple:
            - coordinates (numpy.ndarray): Array of detected (x, y) coordinates. Shape: (N, 2).
            - radii (numpy.ndarray): Array of corresponding blob radii. Shape: (N,).

    Side Effects:
        - Saves the blob detection result to `Blobs/{box_id}.npz` using `save_blobs()`.
        - Prints status messages depending on the number of blobs detected:
            ✓ for exactly one blob,
            X for none or more than one blob.

    Notes:
        - Uses the `detect_blobs()` function for blob detection.
        - Ensures returned arrays have consistent shape using `np.atleast_2d` and `np.ravel`.
    '''
    box_id, box = args
    blob, coordinates, radii = detect_blobs(box)

    # removed saving blobs seperately as I just wanto save all radii and coordinates at the end     

    if len(blob) == 1:
        print(f"Saved blob for box {box_id}")
    elif len(blob) == 0:
        print(f"X No blobs detected in box {box_id}")
    else:
        print(f"X Unexpected blob count in box {box_id} (count: {len(blob)})")

    coordinates = np.atleast_2d(coordinates)
    radii = np.ravel(radii)
    return coordinates, radii

def detect_blob_in_all_boxes(mask_boxes):
    '''
    Detects blobs in a list of image or mask boxes in parallel and collects their coordinates and radii.

    Parameters:
        mask_boxes (list): A list of 2D NumPy arrays, each representing an individual box (typically cropped from a mask or image).

    Returns:
        tuple:
            - all_coordinates (list): A list of NumPy arrays, each containing (x, y) coordinates of blobs detected in the corresponding box.
            - all_radii (list): A list of NumPy arrays, each containing the radii of detected blobs in the corresponding box.

    Notes:
        - Uses `ThreadPoolExecutor` to parallelize processing with `process_box()`.
        - Box IDs are assigned starting from 1 and used to save blob results as `Blobs/{box_id}.npz`.
        - Preserves the original order of boxes in the returned lists.
        - Each box is expected to contain at most one detectable blob. Multiple or no blobs will trigger warning messages from `process_box()`.
    '''
    all_coordinates = []
    all_radii = []

    box_args = [(i + 1, box) for i, box in enumerate(mask_boxes)]  # (box_id, box)

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_box, box_args)  # preserves order

    for coords, rads in results:
        all_coordinates.append(coords)
        all_radii.append(rads)

    return all_coordinates, all_radii

def set_radius(all_radii, radius):
    all_radii = [np.array([radius]) if r.size > 0 else r for r in all_radii]
    return all_radii


def make_grid_and_split_all(markers, conditions, repeat_no, output_list=False):
    '''
    Assumes ALL cropped images are already made and only need to be reloaded 
    List of images (index according to marker where 1:DAPI, 2:SOX2, 3:BRA, 4:GATA3) KEEP THE SAME ALWAYS
        
        img_DAPI, img_SOX2, img_BRA, img_GATA3 = images

        mask_DAPI, mask_SOX2, mask_BRA, mask_GATA3 = masks

    '''
    mask_boxes_list = []
    img_boxes_list = []

    for condition in conditions:
        for repeat in repeat_no:

            # reload cropped images that are already made 
            images, masks = reload_cropped(repeat, condition)

            for i, marker in enumerate(markers):
                
                # draw grid
                grid_mask = draw_grid(images[i])
                print(f"finished making grid for repeat: {repeat}, condition: {condition}, marker: {marker}")

                # make mask boxes per repeat and save 
                print(f"------------------------Starting mask boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")
                mask_boxes = analyse_by_grid(masks[i], grid_mask, marker, condition, repeat, mask=True, rescale_switch = False)
                print(len(mask_boxes), "should be 676")
                print(f"------------------------finished mask boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")

                # make image boxes per repeat and save
                print(f"------------------------Starting image boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")
                img_boxes = analyse_by_grid(images[i], grid_mask, marker, condition, repeat, mask=False, rescale_switch = False)
                print(len(img_boxes), "should be 676")
                print(f"------------------------finished image boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")

                mask_boxes_list.append(mask_boxes)
                img_boxes_list.append(img_boxes)

    if output_list:
        return mask_boxes_list, img_boxes_list


def detect_blob_all(markers, conditions, repeat_no):
    blob_output_paths = []

    for repeat in repeat_no:
        for condition in conditions:
            for marker in markers:
        
                # getting directories
                mask_boxes_path = f"boxes_npz/{repeat}/mask_{marker}_{condition}_NOrescale_boxes.npz"
                image_boxes_path = f"boxes_npz/{repeat}/img_{marker}_{condition}_NOrescale_boxes.npz"

                blobs_output_directory = f"blobs_npz/{repeat}"
                blobs_output_path = f"{blobs_output_directory}/{marker}_{condition}.npz"
                os.makedirs(blobs_output_directory, exist_ok=True)
                
                # A. RELOAD: IMAGE, MASK BOXES 
                print(f"------------------------Starting Load mask boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")
                mask_boxes = load_boxes(mask_boxes_path) 
                print(f"------------------------Finished Load mask boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")

                print(f"------------------------Starting Load images boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")
                img_boxes = load_boxes(image_boxes_path)
                print(f"------------------------Finished Load images boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")

    
                # B. DETECT BLOB IN EACH MASK BOX AND SAVE
                print(f"------------------------Starting Blob detection for repeat: {repeat}, condition: {condition}, marker: {marker}")
                all_coordinates, all_radii = detect_blob_in_all_boxes(mask_boxes)
                print(f"------------------------Finished Blob detection for repeat: {repeat}, condition: {condition}, marker: {marker}")


                # save all outputs for easy visualization and further analysis
                print(f"------------------------Saving Blob detection outputs for repeat: {repeat}, condition: {condition}, marker: {marker} in {blobs_output_path}")
                
                print(f"------------------------{blobs_output_path} info:")
                print("mask_boxes:", type(mask_boxes), len(mask_boxes), type(mask_boxes[0]))
                print("img_boxes:",type(img_boxes), len(img_boxes), type(img_boxes[0]))
                print("all_coordinates:",type(all_coordinates), len(all_coordinates), type(all_coordinates[0]))
                print("all_radii:",type(all_radii), len(all_radii), type(all_radii[0]))
                print(f"------------------------")
                
                np.savez(blobs_output_path,
                        mask_boxes=np.array(mask_boxes, dtype=object),
                        img_boxes=np.array(img_boxes, dtype=object),
                        all_coordinates=np.array(all_coordinates, dtype=object),
                        all_radii=np.array(all_radii, dtype=object))

                blob_output_paths.append(blobs_output_path)

    with open("blob_output_paths.txt", "a") as f:
        for path in blob_output_paths:
            f.write(path)
    print("Finshed putting all blob output paths to blob_output_paths.txt")