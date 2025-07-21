from f_intensityMeasurement import *
from f_modelDetection import *
from f_preprocessing import *
from f_validation import *



#---------------------------------------------------------------------------#
# function needs to be changed to accommodate the new format: blob_output_paths
# # B. RELOAD: BLOBS
# all_coordinates, all_radii = reload_all_blobs("Blobs")

# How to Load Later
# data = np.load('blobs_output_path.npz', allow_pickle=True)
# mask_boxes = data['mask_boxes']
#---------------------------------------------------------------------------#


# 6. change all radius to be the same value (200)
# whole_radii = set_radius(all_radii, 200)
# # 6A. make circles! can adjust value later
# big_radii = set_radius(all_radii, 150)
# med_radii = set_radius(all_radii, 100)
# small_radii = set_radius(all_radii, 50)


# 8. measure intensity within the blob (whole intensity)
# whole_intensities = measure_all_blob_intensities(img_boxes, all_coordinates, whole_radii)
# big_intensities = measure_all_blob_intensities(img_boxes, all_coordinates, big_radii) # GATA3
# med_intensities = measure_all_blob_intensities(img_boxes, all_coordinates, med_radii) # BRA 
# small_intensities = measure_all_blob_intensities(img_boxes, all_coordinates, small_radii) # SPX2


# plot_intensities(whole_intensities, "whole_intensities_distribution")
# plot_intensities(big_intensities, "big_intensities_distribution")
# plot_intensities(med_intensities, "med_intensities_distribution")
# plot_intensities(small_intensities, "small_intensities_distribution")


# 8A. measure intensity within smaller circle 
# small_intensities = measure_all_blob_intensities(img_boxes, all_coordinates, small_radii)
# plot_intensities(small_intensities, "intensities_distribution_small")
# view_single_box(500, img_boxes, mask_boxes, all_coordinates, whole_radii, big_radii, med_radii, small_radii)
