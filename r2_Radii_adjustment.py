from f_intensityMeasurement import *
from f_modelDetection import *
from f_preprocessing import *
from f_validation import *
import csv
from pathlib import Path


# blob_output_paths_filtered05.txt
# blob_output_paths.txt

def read_paths(filename="blob_output_paths_5300.txt"):
    with open(filename, "r") as f:
        for line in f:
            path = line.strip()  # Remove trailing newline and any extra spaces
            if path:  # Skip empty lines if any
                yield path

def load_DAPI():
    # open dapi npz
    for path in read_paths():
        if "DAPI" in path:
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
    print("biggest radius in DAPI list is:", max_radii)

    outer_radius = max_radii + 10
    mid_radius = max_radii - 55
    inner_radius = max_radii - 95

    # keep track of radius per repeat and condition 
    new_row = [repeat,condition,outer_radius,mid_radius,inner_radius]
    with open('radius.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_row)

    print("outer radius set to:", outer_radius)
    print("mid radius set to:", mid_radius)
    print("inner radius set to:", inner_radius)

    outer_radius_array = set_radius(DAPI_radii, outer_radius)
    mid_radius_array = set_radius(DAPI_radii, mid_radius)
    inner_radius_array = set_radius(DAPI_radii, inner_radius)

    return DAPI_coordinates, DAPI_mask_boxes, DAPI_img_boxes, outer_radius_array, mid_radius_array, inner_radius_array


def adjust(marker, DAPI_coordinates, outer_radius_array, mid_radius_array, inner_radius_array):
    
    # getting image and mask bozes from boxes_npz instead of blobs_npz
    repeat = 1
    directory = Path(f"boxes_npz/{repeat}")
    npz_files = list(directory.glob("*.npz"))

    # split based on mask / image
    image_files = [f for f in npz_files if f.name.startswith("img_")]
    mask_files = [f for f in npz_files if f.name.startswith("mask_")]

    # Create a dict to find masks by their key (e.g. BRA, DAPI)
    def extract_key(filename):
        # Example: img_BRA_WT_NOrescale_boxes.npz -> BRA
        parts = filename.split("_")
        return parts[1]  

    # Map mask keys to mask file paths
    mask_dict = {extract_key(f.name): f for f in mask_files}

    # Pair images with masks
    pairs = []
    for img_file in image_files:
        key = extract_key(img_file.name)
        mask_file = mask_dict.get(key)
        if mask_file:
            pairs.append((img_file, mask_file))
    
    
    for img, mask in pairs:
        if marker in str(img):
            print(f"Image: {img}  <-->  Mask: {mask}")
            # boxes_npz\1\mask_SOX2_WT_NOrescale_boxes.npz
            print("------------------------Now processing:", img)
            
            # get info from image path 
            stripped = os.path.splitext(img)[0]
            print(stripped)

            parts = os.path.normpath(stripped).split(os.sep)
            print(parts)
            
            repeat = parts[1]
            info_2 = parts[2].split("_")
            marker = info_2[1]
            condition = info_2[2]

            # print info obtained
            print("repeat:", repeat)
            print("condition:", condition)
            print("marker:", marker)

            # load image and mask boxes
            img_boxes = np.load(img, allow_pickle=True)
            mask_boxes = np.load(mask, allow_pickle=True)

            img_opened = [img_boxes[key] for key in img_boxes.files]
            mask_opened = [mask_boxes[key] for key in mask_boxes.files]

            # open slider visual to visualize it
            print(f"opening slider visual to adjust for {marker}...")
            slider_visual(
                    img_boxes_list=[img_opened],
                    mask_boxes_list=[mask_opened],
                    all_coordinates_list=[DAPI_coordinates],  
                    outer_radius_list=[outer_radius_array],
                    mid_radius_list=[mid_radius_array],
                    inner_radius_list=[inner_radius_array],
                    labels=[f"{marker}"],
                    print_info=False
                )

if __name__ == "__main__":
    DAPI_coordinates, DAPI_mask_boxes, DAPI_img_boxes, outer_radius_array, mid_radius_array, inner_radius_array = load_DAPI()

    adjust("BRA", DAPI_coordinates, outer_radius_array, mid_radius_array, inner_radius_array)
    # adjust("GATA3", DAPI_coordinates, outer_radius_array, mid_radius_array, inner_radius_array)

    # adjust("SOX2", DAPI_coordinates, outer_radius_array, mid_radius_array, inner_radius_array)







#---------------------------------------------------------------------------#

# 7. View what we have..
# functions I can use to view:
    # view_single_box
    # plot_radii_distribution
    # slider_visual



