from imports import *
from f_validation import view_tiff


def czi_to_tiffs(folder_path):
    """
    takes folder containing all czi files, 
    converts them into tiffs split by 4 channels. 
    puts them in the same folder 
    
    """
    czi_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.czi'):  
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                czi_paths.append(file_path)


    for czi_path in czi_paths:
        try:
            img = czifile.imread(czi_path)
            czi = CziFile(czi_path)

            print(f"Original CZI shape: {img.shape}")
            print("Axes :", czi.dims)

            img_squeezed = img.squeeze() # Remove singleton dimensions
            print(f"Squeezed shape: {img_squeezed.shape}")

            # Map channels to folders
            channel_folders = {
                0: 'DAPI',
                1: 'SOX2',
                2: 'GATA3',
                3: 'BRA'
            }
            
            num_channels = img_squeezed.shape[0]    
            print("num channels", num_channels)
            
            for channel_no in range(num_channels):
                channel_data = img_squeezed[channel_no]  # (Z, Y, X)
                base_path, _ = os.path.splitext(czi_path)
                output_path = f"{base_path}_{channel_folders[channel_no]}.tiff"
                tifffile.imwrite(output_path, channel_data)
                print(f"Saved TIFF: {output_path}")


        except Exception as e:
            print(f"Error processing {czi_path}: {e}")

def crop_tiff(tiff_path, ymin, ymax, xmin, xmax, rotate_angle):
    '''
    Crops a TIFF image to a specified rectangular region and returns the result.

    Parameters:
        tiff_path (str): Path to the TIFF file to be cropped.
        ymin (int): Starting Y-coordinate (row) of the crop.
        ymax (int): Ending Y-coordinate (row) of the crop (exclusive).
        xmin (int): Starting X-coordinate (column) of the crop.
        xmax (int): Ending X-coordinate (column) of the crop (exclusive).
        rotate_angle (int): How much to rotate the image if needed 

    Returns:
        numpy.ndarray: The cropped image region as a NumPy array.

    Notes:
        - Automatically squeezes singleton dimensions from the image (e.g., shape (1, Y, X) → (Y, X)).
        - Supports cropping 2D or 3D TIFF images (only spatial dimensions Y and X are cropped).
    '''

    img = tifffile.imread(tiff_path)
    img = np.squeeze(img)
    # original_dimensions = image_squeezed.shape
    # print("original dimensions:", original_dimensions)
    print(f"\nProcessing {tiff_path}")
    # rotateing if needed
    if rotate_angle != 0:
        print(f"rotating this much: {rotate_angle}")
        img = rotate(img, angle=rotate_angle, reshape=True)

    # cropping
    print("Cropping...")
    img = img[ymin:ymax, xmin:xmax]

    print(f"Finished cropping {tiff_path}")

    return img


def crop_all_tiffs_in_repeat(directory, repeat, condition, markers, ymin, ymax, xmin, xmax, rotate_angle, visualize=False, mask=False):
    
    full_directory = f"{directory}\REPEAT{repeat}"
    output_folder = "CROPPED"
    

    img_dict = {}
    for marker in markers:
        if mask:
            img = crop_tiff(f"{full_directory}/{marker}_{condition}_mask.tiff", ymin, ymax, xmin, xmax, rotate_angle)
            img_dict[f"mask_{marker}"] = img
        else:
            img = crop_tiff(f"{full_directory}/{condition}_{marker}.tiff", ymin, ymax, xmin, xmax, rotate_angle)
            img_dict[f"img_{marker}"] = img

        output_path = f'{output_folder}/{repeat}/{condition}'
        os.makedirs(output_path, exist_ok=True)
        
        if mask:
            tifffile.imwrite(f'{output_path}/{marker}_mask.tiff', img)
        else:
            tifffile.imwrite(f'{output_path}/{marker}.tiff', img)

    
    if visualize:
        layers = []
        for marker in markers:
            layers.append((f"img{repeat}_{marker}", img_dict[f'img_{marker}']))
        view_tiff(*layers)

    


def get_crop_coordinates(csv_path, repeat, condition):
    """
    Fetches cropping coordinates for a given repeat and condition from a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file.
        repeat (int): Repeat number (e.g., 1, 2, 3).
        condition (str): Experimental condition (e.g., "WT", "ND6").

    Returns:
        list: [ymin, ymax, xmin, xmax]

    Raises:
        ValueError: If the combination of repeat and condition is not found.
    """
    df = pd.read_csv(csv_path)

    # Filter based on repeat and condition
    row = df[(df['repeat'] == repeat) & (df['condition'] == condition)]

    if row.empty:
        raise ValueError(f"No entry found for repeat {repeat} and condition '{condition}'.")

    # Return as list
    return row.iloc[0][['ymin', 'ymax', 'xmin', 'xmax', 'rotate_angle']].tolist()

def reload_cropped(repeat, condition):
    images = []
    masks = []
    # load DAPI
    img_DAPI = tifffile.imread(f'CROPPED/{repeat}/{condition}/DAPI.tiff')
    images.append(img_DAPI)
    mask_DAPI = tifffile.imread(f'CROPPED/{repeat}/{condition}/DAPI_mask.tiff')
    masks.append(mask_DAPI)

    # load SOX2
    img_SOX2 = tifffile.imread(f'CROPPED/{repeat}/{condition}/SOX2.tiff')
    images.append(img_SOX2)
    mask_SOX2 = tifffile.imread(f'CROPPED/{repeat}/{condition}/SOX2_mask.tiff')
    masks.append(mask_SOX2)

    # load BRA
    img_BRA = tifffile.imread(f'CROPPED/{repeat}/{condition}/BRA.tiff')
    images.append(img_BRA)    
    mask_BRA = tifffile.imread(f'CROPPED/{repeat}/{condition}/BRA_mask.tiff')
    masks.append(mask_BRA)

    # load GATA3
    img_GATA3 = tifffile.imread(f'CROPPED/{repeat}/{condition}/GATA3.tiff')
    images.append(img_GATA3)    
    mask_GATA3 = tifffile.imread(f'CROPPED/{repeat}/{condition}/GATA3_mask.tiff')
    masks.append(mask_GATA3)

    return images, masks

def crop_all(directory, cropping_csv_path, conditions, repeat_no, markers):

    for condition in conditions:
        for i in repeat_no:
            ymin, ymax, xmin, xmax, rotate_angle = get_crop_coordinates(cropping_csv_path, repeat=i, condition=condition)
            crop_all_tiffs_in_repeat(directory=directory, repeat=i, condition=condition, markers=markers, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax, rotate_angle=rotate_angle, visualize=False, mask=False)
            crop_all_tiffs_in_repeat(directory=directory, repeat=i, condition=condition, markers=markers, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax, rotate_angle=rotate_angle, visualize=False, mask=True)
