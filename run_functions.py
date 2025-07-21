from f_intensityMeasurement import *
from f_modelDetection import *
from f_preprocessing import *
from f_validation import *

# START TIMER
start_time = time.time() 

# CONSTANT INFO:
markers = ["DAPI", "SOX2", "BRA", "GATA3"]
conditions = ["WT", "ND6"]
repeat_no = 1
directory = "CHIP_REPEATS"
cropping_csv_path = "cropping.csv"

# # DIRECT OUTPUT FILE comment this out if preffered in terminal
# output_file = open('output.txt', 'w')
# sys.stdout = output_file

# PREPROCESSING: SPLIT CZI INTO TIFFS
# czi_to_tiffs('CHIP_REPEATS')

# 1. CROP IMAGE AND MASK 
# images = None
# masks = None

# for condition in conditions:
#     for i in range(1,repeat_no + 1):
#         ymin, ymax, xmin, xmax, rotate_angle = get_crop_coordinates(cropping_csv_path, repeat=i, condition=condition)
#         images = crop_all_tiffs_in_repeat(directory=directory, repeat=i, condition=condition, markers=markers, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax, rotate_angle=rotate_angle, visualize=False, mask=False)
#         masks = crop_all_tiffs_in_repeat(directory=directory, repeat=i, condition=condition, markers=markers, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax, rotate_angle=rotate_angle, visualize=False, mask=True)

# 1A. RELOAD CROPPED IMAGES
#---------------------------------------------------------------------------#
# ONLY USE THIS PART WHEN RELOADING FILES THAT ALREADY EXIST 
images, masks = reload_cropped(1, "WT")
img_DAPI, img_SOX2, img_BRA, img_GATA3 = images
mask_DAPI, mask_SOX2, mask_BRA, mask_GATA3 = masks
#---------------------------------------------------------------------------#

# looking at marker: SOX2

# 2. MAKE GRID MASK
grid_mask = draw_grid(img_SOX2)

# 3. BLOB DETECTION 
#---------------------------------------------------------------------------#
# ONLY USE THIS PART WHEN RELOADING FILES THAT ALREADY EXIST 

# # A. RELOAD: IMAGE, MASK BOXES 
# mask_boxes = load_boxes("NOrescale_mask_boxes.npz") # add rescale / NOrescale accordingly
# img_boxes = load_boxes("NOrescale_img_boxes.npz")
# # B. RELOAD: BLOBS
# all_coordinates, all_radii = reload_all_blobs("Blobs")
#---------------------------------------------------------------------------#

#---------------------------------------------------------------------------#
# ONLY USE THIS PART WHEN DETECTING BLOBS FROM SCRATCH / SAVING FILES

# A. SPLIT MASK INTO BOXES BASED ON GRID
mask_boxes = analyse_by_grid(mask_SOX2, grid_mask, "SOX2", downscale_factor=0.25, mask=True, rescale_switch = False)
print(len(mask_boxes)) # should be 676

# B. SPLIT IMAGE INTO BOXES BASED ON GRID
img_boxes = analyse_by_grid(img_SOX2, grid_mask, "SOX2", downscale_factor=0.25, mask=False, rescale_switch = False)
print(len(img_boxes)) # should be 676

# C. DETECT BLOB IN EACH MASK BOX AND SAVE
all_coordinates, all_radii = detect_blob_in_all_boxes(mask_boxes)

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
