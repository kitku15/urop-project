from f_intensityMeasurement import *
from f_modelDetection import *
from f_preprocessing import *
from f_validation import *

# START TIMER
start_time = time.time() 

# CONSTANT INFO:
# list of markers (index according to marker where 1:DAPI, 2:SOX2, 3:BRA, 4:GATA3)
markers = ["DAPI", "SOX2", "BRA", "GATA3"]
conditions = ["WT", "ND6"]
repeat_no = [1]
directory = "CHIP_REPEATS"
cropping_csv_path = "cropping.csv"
blob_output_paths = "blob_output_paths.txt"


# PREPROCESSING: SPLIT CZI INTO TIFFS
# czi_to_tiffs('CHIP_REPEATS')

# 1. CROP IMAGE AND MASK, SAVE
crop_all(directory, cropping_csv_path, conditions, repeat_no, markers)


# 2. RELOAD CROPPED IMAGES, MAKE GRID MASK, SPLIT MASK AND IMAGE INTO BOXES BASED ON GRID
make_grid_and_split_all(markers, conditions, repeat_no)

# 3. BLOB DETECTION IN EACH MASK BOX AND SAVE
detect_blob_all(markers, conditions, repeat_no)


#---------------------------------------------------------------------------#
# function needs to be changed to accommodate the new format: blob_output_paths
# # B. RELOAD: BLOBS
# all_coordinates, all_radii = reload_all_blobs("Blobs")
#---------------------------------------------------------------------------#

# END TIMER
end_time = time.time()
print(f"\nScript completed in {end_time - start_time:.2f} seconds.")

# 5. Do a Final Re-check of list lengths
# check_list_size(all_coordinates, all_radii, img_boxes, mask_boxes)

# 6. change all radius to be the same value (200)
# whole_radii = set_radius(all_radii, 200)
# # 6A. make circles! can adjust value later
# big_radii = set_radius(all_radii, 150)
# med_radii = set_radius(all_radii, 100)
# small_radii = set_radius(all_radii, 50)

# 7. View what we have..

view_single_box(500, img_boxes, mask_boxes, all_coordinates, all_radii)
# plot_radii_distribution(all_radii, "radii_distribution_box")
# slider_visual(mask_boxes, img_boxes, all_coordinates, whole_radii, print_info = False)

# 8. measure intensity within the blob (whole intensity)
# whole_intensities = measure_all_blob_intensities(img_boxes, all_coordinates, whole_radii)
# big_intensities = measure_all_blob_intensities(img_boxes, all_coordinates, big_radii) # GATA3
# med_intensities = measure_all_blob_intensities(img_boxes, all_coordinates, med_radii) # BRA 
# small_intensities = measure_all_blob_intensities(img_boxes, all_coordinates, small_radii) # SPX2

# print(whole_intensities.shape)


# plot_intensities(whole_intensities, "whole_intensities_distribution")
# plot_intensities(big_intensities, "big_intensities_distribution")
# plot_intensities(med_intensities, "med_intensities_distribution")
# plot_intensities(small_intensities, "small_intensities_distribution")


# 8A. measure intensity within smaller circle 
# small_intensities = measure_all_blob_intensities(img_boxes, all_coordinates, small_radii)
# plot_intensities(small_intensities, "intensities_distribution_small")
# view_single_box(500, img_boxes, mask_boxes, all_coordinates, whole_radii, big_radii, med_radii, small_radii)
