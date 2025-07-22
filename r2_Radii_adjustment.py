from f_intensityMeasurement import *
from f_modelDetection import *
from f_preprocessing import *
from f_validation import *
import csv



def read_paths(filename="blob_output_paths.txt"):
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

            # Extracting lists from data
            mask_boxes = data['mask_boxes']
            img_boxes = data['img_boxes']

            # open slider visual to visualize it
            print(f"opening slider visual to adjust for {marker}...")
            slider_visual(
                    img_boxes_list=[img_boxes],
                    mask_boxes_list=[mask_boxes],
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
    adjust("GATA3", DAPI_coordinates, outer_radius_array, mid_radius_array, inner_radius_array)

    # adjust("SOX2", DAPI_coordinates, outer_radius_array, mid_radius_array, inner_radius_array)







#---------------------------------------------------------------------------#

# 7. View what we have..
# functions I can use to view:
    # view_single_box
    # plot_radii_distribution
    # slider_visual



