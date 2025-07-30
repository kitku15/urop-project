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

def small_box(image, coordinates_rescaled, radii_rescaled):
    '''
    get small box around identified blob 
    '''
    # get image info
    height, width = image.shape[:2]
    x_center, y_center = coordinates_rescaled[0]
    radius = radii_rescaled[0]

    # Define bounding box with a bit of space
    space = -10
    x_min = max(int(x_center - radius), 0) - space
    x_max = min(int(x_center + radius), width) + space
    y_min = max(int(y_center - radius), 0) - space
    y_max = min(int(y_center + radius), height) + space

    # Extract region from image
    small_image = image[y_min:y_max, x_min:x_max]

    return small_image
    
def regions_props_preprocessing(area, sigma, blurred_thresh):
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

    # dupa visualize blurred--------------------------
    # # Flatten the blurred image to 1D for distribution
    # blurred_flat = blurred.ravel()

    # # Plot histogram and KDE (kernel density estimate)
    # plt.figure(figsize=(8, 5))
    # sns.histplot(blurred_flat, bins=50, kde=True, color='blue')
    # plt.title("Distribution of Blurred Pixel Values")
    # plt.xlabel("Pixel Intensity")
    # plt.ylabel("Frequency")
    # plt.grid(True)
    # plt.savefig("Blurred.png")
    #-----------------------------------

    # Convert to binary mask
    binary_mask = blurred > blurred_thresh # can adjust

    labeled = measure.label(binary_mask) # Label connected components
    regions = measure.regionprops(labeled) # Measure properties

    return regions, binary_mask

def circularity_filter(area, c_thresh, box_id, filter_no):
    '''
    Apply circularity filter on small image: if blob not circular enough --> filter out!
    '''
    regions, binary_mask = area
    circularity_pass = False
    circular_blobs = []
    c_thresh = 0

    # Filter by circularity
    for region in regions:

        if region.perimeter == 0: # to prevent division by zero 
            continue

        circularity = 4 * np.pi * region.area / (region.perimeter ** 2)

        # SIZE FILTERS 
        min_diameter = 50
        diameter = region.equivalent_diameter

        if circularity > c_thresh and min_diameter <= diameter:
            circular_blobs.append((region, circularity))

    # visualize dupa --------------------------
    # fig, ax = plt.subplots()
    # ax.imshow(binary_mask, cmap='gray')
    # ax.set_title(f'{len(circular_blobs)} Circular Blobs Detected')

    # if not circular_blobs:
    #     ax.set_title('No Circular Blobs Detected')
    # else:
    #     ax.set_title(f'{len(circular_blobs)} Circular Blob(s) Detected')
    #     for region in circular_blobs:
    #         y, x = region.centroid
    #         diameter = region.equivalent_diameter
    #         r = diameter / 2.0
    #         # Draw the circle
    #         circle = plt.Circle((x, y), r, edgecolor='red', fill=False, linewidth=1.5)
    #         ax.add_patch(circle)
    #         # Label the diameter on the image
    #         ax.text(x, y, f'{circularity:.3f}', color='blue', fontsize=8, ha='center', va='center')
    # plt.axis('off')
    # plt.savefig(f'Circularity_Filter_output_{box_id}_filter{filter_no}')
    # plt.close()
    # -----------------------------------------

    if circular_blobs:
        # Find the blob with the largest area
        largest_blob, best_circularity = max(circular_blobs, key=lambda x: x[0].area)
        return True, best_circularity
    else:
        return False, 0
    
def border_filter(area):
    '''
    Apply border filtering: if white area touches border --> filter out!
    '''
    regions, binary_mask = area
    area_height, area_width = binary_mask.shape   # get height and width
    border_blob_regions = []
    for region in regions:
        # Get bounding box: (min_row, min_col, max_row, max_col)
        minr, minc, maxr, maxc = region.bbox
        # Check if it touches image border
        if minr <= 0 or minc <= 0 or maxr >= area_height or maxc >= area_width:
            border_blob_regions.append(region)
            
    if border_blob_regions:  
        border_pass = False
        return border_pass              
    else:
        border_pass = True
        return border_pass

def filter_1(box_id, img_box, F1_LVThresh, coordinates_rescaled, radii_rescaled):
    '''
    Blurry-ness filter using Laplacian Variance
    
    '''      
    small_img = small_box(img_box, coordinates_rescaled, radii_rescaled)

    def is_blurry(image):
        # Only convert to grayscale if image has more than one channel
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()

        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        return laplacian_var
    
    # laplacian_var = is_blurry(img_box) # on whole image
    laplacian_var = is_blurry(small_img) # on small image 

    thresh = F1_LVThresh

    if laplacian_var > thresh:
        filter_1_pass = True
        # print({box_id}, "Is a clear Image. Passed filter 1. Small Lap Value:", laplacian_var)

    else:
        filter_1_pass = False
        # print({box_id}, "Is a blurry Image. Failed filter 1, throwing it out.. Small Lap Value:", laplacian_var)
        
    return filter_1_pass, laplacian_var

def filter_2(box_id, img_box, sigma, c_thresh, blurred, coordinates_rescaled, radii_rescaled):
    '''
    Circularity and Border Filter 

    '''

    # Apply contrast 
    img_box = img_as_float(img_box)
    # img_con = exposure.adjust_sigmoid(img_box, gain=20, cutoff=0.2)


    # Handle grayscale images
    is_gray = img_box.ndim == 2
    
    # trying to visualize contrast effect dupa -------------------------
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(img_box, cmap='gray' if is_gray else None)
    # axes[0].set_title("Original")
    # axes[0].axis('off')
    # axes[1].imshow(img_con, cmap='gray' if is_gray else None)
    # axes[1].set_title("Contrast Adjusted")
    # axes[1].axis('off')
    # plt.tight_layout()
    # plt.savefig(f'Contrast_effect_output_{box_id}.png')
    # plt.close()
    # -----------------------------------------------------------------

    # images: divide image into whole and small area and preprocess
    whole = regions_props_preprocessing(img_box, sigma, blurred)
    # small = regions_props_preprocessing(small_box(img_con, coordinates_rescaled, radii_rescaled), sigma)

    # Apply filters
    filter_no = 2
    circularity_pass, circularity = circularity_filter(whole, c_thresh, box_id, filter_no)
    # print({box_id}, f'Circularity is: {circularity:.3f}')
    
    border_pass  = border_filter(whole)

    # print details of filtering
    # if not circularity_pass:
        # print({box_id}, "Failed Filter 2 Circularity.. throwing it out")
    # if not border_pass:
        # print({box_id}, "Failed Filter 2 Border.. throwing it out")

    if circularity_pass and border_pass:
        filter_2_pass = True
        # print({box_id}, "Passed Filter 2, Included in analysis!")
    else:
        filter_2_pass = False

    return filter_2_pass, circularity



def detect_blobs(box_id, tiff, img, F1_LVThresh, F2_sigma, F2_binaryThresh, F2_circThresh, downscale_factor=0.25, sigma=2, min_sigma=20, max_sigma=50, exclude_border=45):
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

    # open csv 
    df = pd.read_csv("backup/meta.csv")

    # Find the row by box_id
    row = df.loc[df['box_id'] == box_id]

    # Get the status
    status = row['status'].values[0] # status is either 'v' or 'x'


    # downscale
    downsampled = rescale(tiff, downscale_factor, anti_aliasing=True) 
    
    # Convert to float and normalize
    tiff_float = img_as_float(downsampled)

    # smooth
    smoothed = gaussian(tiff_float, sigma)

    # Detect blobs
    blobs = blob_log(image=smoothed, min_sigma=min_sigma, max_sigma=max_sigma, exclude_border=exclude_border)

    if blobs.shape[0] == 0:

         # if true positive and classified as negative: False negative
        if status == 'v': 
            return np.empty((0, 3)), np.empty((0, 2)), np.array([]), [box_id, 0, 0, "x", 0, 1]

        # if true negative and classified as negative: True negative
        if status == 'x': 
            return np.empty((0, 3)), np.empty((0, 2)), np.array([]), [box_id, 0, 0, "x", 0, 0]

        
    
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

    # Filter 1 to filter out blurry images
    filter_1_pass, laplacian_var = filter_1(box_id, img, F1_LVThresh, coordinates_rescaled, radii_rescaled)

    # CSV--------------------------------

  

    # FP_list = [45, 46, 47, 48, 49, 50, 54, 55, 57, 69, 72, 75, 76, 86, 94, 133, 142, 162, 215, 
    #            216, 242, 243, 255, 256, 309, 310, 311, 312, 335, 338, 380, 412, 490, 506, 515, 
    #            539, 538, 547, 548, 550, 552, 558, 573, 579, 580, 581, 582, 583, 598, 599, 601, 
    #            603, 606, 625, 653, 664, 665, 666, 668]

    # Filter 2 on images that passed bluriness filter and are considered clear
    if filter_1_pass:
        # if box_id in FP_list:
        #     print({box_id}, "Wrongfully included through filter 1")

        # reassign parameters to the thing taken by filter 2 (should be made neat later)
        c_sigma = F2_sigma
        blurred = F2_binaryThresh
        c_thresh = F2_circThresh

        filter_2_pass, circularity = filter_2(box_id, img, c_sigma, c_thresh, blurred, coordinates_rescaled, radii_rescaled)
        
    else: 
        # if box_id in FP_list:
        #     print({box_id}, "Correctly filtered out by filter 1")

        # if true positive and classified as negative: False negative
        if status == 'v': 
            return np.empty((0, 3)), np.empty((0, 2)), np.array([]), [box_id, laplacian_var, 0, "x", 0, 1]
            

        # if true negative and classified as negative: True negative
        if status == 'x': 
            return np.empty((0, 3)), np.empty((0, 2)), np.array([]), [box_id, laplacian_var, 0, "x", 0, 0]



    # if it passes filter 2, we keep it.
    if filter_2_pass:

        # if true positive and classified as positive: True positive
        if status == 'v': 
            return blob, coordinates_rescaled, radii_rescaled, [box_id, laplacian_var, circularity, "v", 0, 0]

        
        # if true negative and classified as positive: False positive
        if status == 'x':
            return blob, coordinates_rescaled, radii_rescaled, [box_id, laplacian_var, circularity, "v", 1, 0]

    
    else: 
        # if true positive and classified as negative: False negative
        if status == 'v': 
            return np.empty((0, 3)), np.empty((0, 2)), np.array([]), [box_id, laplacian_var, circularity, "x", 0, 1]

        # if true negative and classified as negative: True negative
        if status == 'x': 
            return np.empty((0, 3)), np.empty((0, 2)), np.array([]), [box_id, laplacian_var, circularity, "x", 0, 0]
            


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
    # print(f"Saved {len(upscaled_boxes)} boxes to {save_path}")
    # print(f"Boxes done")
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

    os.makedirs(f"{heading}", exist_ok=True)

    plt.imsave(f"{heading}/{box_id}.tiff", cropped_small, cmap='gray') 
    # print(f"added Box {box_id}")
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


def process_box(box_args, img_box_args, F1_LVThresh, F2_sigma, F2_binaryThresh, F2_circThresh):
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

    box_id, mask_box = box_args
    _, img_box = img_box_args

    blob, coordinates, radii, csv_array = detect_blobs(box_id, mask_box, img_box, F1_LVThresh, F2_sigma, F2_binaryThresh, F2_circThresh)

    # dupa check

    # if box_id == 380:
    #     blob, coordinates, radii = detect_blobs(box_id, mask_box, img_box)
    # else:
    #     return None, None, None
    

    if len(blob) == 1:
        # print(f"Saved blob for box {box_id}")
        model_found = True
    else: 
        # print(f"X No blobs detected in box {box_id}")
        model_found = False


    coordinates = np.atleast_2d(coordinates)
    radii = np.ravel(radii)
    
    return coordinates, radii, model_found, csv_array

def detect_blob_in_all_boxes(mask_boxes, img_boxes, F1_LVThresh, F2_sigma, F2_binaryThresh, F2_circThresh):
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
    model_count = 0


    box_args = [(i + 1, box) for i, box in enumerate(mask_boxes)]  # (box_id, box)
    img_box_args = [(i + 1, box) for i, box in enumerate(img_boxes)]  # (box_id, box)

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_box, box_args, img_box_args, repeat(F1_LVThresh), repeat(F2_sigma), repeat(F2_binaryThresh), repeat(F2_circThresh))  # preserves order

        # append to csv 
        df = pd.read_csv('meta.csv', na_values=['', 'nan'])
        df['classification'] = df['classification'].astype(str)


        for coords, rads, model_found, csv_array in results:
            all_coordinates.append(coords)
            all_radii.append(rads)
            if model_found:
                model_count = model_count + 1

            # print(csv_array)
            box_id, lv, circ, classification, fp, fn = csv_array
            # print(box_id, lv, circ, classification, fp, fn)

        
            df.loc[df['box_id'] == box_id, ['LV', 'circularity', 'classification', 'FP_score', 'FN_score']] =  [float(lv), float(circ), str(classification), float(fp), float(fn)]

        df.to_csv(f"metas/meta_{F1_LVThresh}_{F2_sigma}_{F2_binaryThresh}_{F2_circThresh}.csv", index=False)

        


    return all_coordinates, all_radii, model_count

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


def detect_blob_all(markers, conditions, repeat_no, F1_LVThresh, F2_sigma, F2_binaryThresh, F2_circThresh):
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
                # print(f"------------------------Starting Load mask boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")
                mask_boxes = load_boxes(mask_boxes_path) 
                # print(f"------------------------Finished Load mask boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")

                # print(f"------------------------Starting Load images boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")
                img_boxes = load_boxes(image_boxes_path)
                # print(f"------------------------Finished Load images boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")

    
                # B. DETECT BLOB IN EACH MASK BOX AND SAVE
                # print(f"------------------------Starting Blob detection for repeat: {repeat}, condition: {condition}, marker: {marker}")
                all_coordinates, all_radii, model_count = detect_blob_in_all_boxes(mask_boxes, img_boxes, F1_LVThresh, F2_sigma, F2_binaryThresh, F2_circThresh)
                # print(f"Models detected: {model_count}/676")
                # print(f"------------------------Finished Blob detection for repeat: {repeat}, condition: {condition}, marker: {marker}")


                # save all outputs for easy visualization and further analysis
                # print(f"------------------------Saving Blob detection outputs for repeat: {repeat}, condition: {condition}, marker: {marker} in {blobs_output_path}")
                
                # print(f"------------------------{blobs_output_path} info:")
                # print("mask_boxes:", type(mask_boxes), len(mask_boxes), type(mask_boxes[0]))
                # print("img_boxes:",type(img_boxes), len(img_boxes), type(img_boxes[0]))
                # print("all_coordinates:",type(all_coordinates), len(all_coordinates), type(all_coordinates[0]))
                # print("all_radii:",type(all_radii), len(all_radii), type(all_radii[0]))
                # print(f"------------------------")
                
                np.savez(blobs_output_path,
                        mask_boxes=np.array(mask_boxes, dtype=object),
                        img_boxes=np.array(img_boxes, dtype=object),
                        all_coordinates=np.array(all_coordinates, dtype=object),
                        all_radii=np.array(all_radii, dtype=object))

                blob_output_paths.append(blobs_output_path)

    with open("blob_output_paths.txt", "a") as f:
        for path in blob_output_paths:
            f.write(path + "\n")
    # print("Finshed putting all blob output paths to blob_output_paths.txt")