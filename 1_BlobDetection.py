from f_intensityMeasurement import *
from f_modelDetection import *
from f_preprocessing import *
from f_validation import *

# START TIMER
start_time = time.time() 

# CONSTANT INFO:
# list of markers (index according to marker where 1:DAPI, 2:SOX2, 3:BRA, 4:GATA3)
# I currently only have mask for REPEAT 1, WT 
markers = ["DAPI", "SOX2", "BRA", "GATA3"]
conditions = ["WT"]
repeat_no = [1]
directory = "CHIP_REPEATS"
cropping_csv_path = "cropping.csv"
blob_output_paths = "blob_output_paths.txt"

# PREPROCESSING 1: SPLIT CZI INTO TIFFS
# czi_to_tiffs('CHIP_REPEATS')

# PREPROCESSING 2: CROP IMAGE AND MASK, SAVE
# crop_all(directory, cropping_csv_path, conditions, repeat_no, markers)
# print("finished cropping all!")

# BLOB DETECTION 1. RELOAD CROPPED IMAGES, MAKE GRID MASK, SPLIT MASK AND IMAGE INTO BOXES BASED ON GRID
# make_grid_and_split_all(markers, conditions, repeat_no)

# BLOB DETECTION 2: BLOB DETECTION IN EACH MASK BOX AND SAVE
detect_blob_all(markers, conditions, repeat_no)


#---------------------------------------------------------------------------#
# function needs to be changed to accommodate the new format: blob_output_paths
# # B. RELOAD: BLOBS
# all_coordinates, all_radii = reload_all_blobs("Blobs")

# How to Load Later
# data = np.load('blobs_output_path.npz', allow_pickle=True)
# mask_boxes = data['mask_boxes']
#---------------------------------------------------------------------------#

# END TIMER
end_time = time.time()
print(f"\nScript completed in {end_time - start_time:.2f} seconds.")

# 5. Do a Final Re-check of list lengths
# check_list_size(all_coordinates, all_radii, img_boxes, mask_boxes)


# 7. View what we have..
# functions I can use to view:
    # view_single_box
    # plot_radii_distribution
    # slider_visual





