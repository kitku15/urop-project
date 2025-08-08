
from f_coordFinder import run_R2
from f_modelDetection import load_boxes, load_allowed_ids
from f_distributions import *
import os
import csv
import time



# This part will get the distribution of the Model's 
# - Diameter (based on DAPI)
# - All marker including DAPI Intensities
# - Circularity (not priority)

# START TIMER
start_time = time.time() 

# BOOLS
bLOAD = False
bGETDISTRIBUTIONS = False
bNORMALIZE = False
bOVERLAPDENSITYPLOT = False
bPLOTDISTRIBUTION_INTvsDIA = True

# INFO
repeat = 3
condition = "ND6"
DAPI_marker = "DAPI"

# SET RADIUS ACCORDING TO R2
outer_radius = 0
mid_radius = 0
inner_radius = 0


markers = ["SOX2", "BRA", "GATA3"]

for marker in markers:

    # GET COORDINATES AND MODEL REGION
    if bLOAD:
        print(f"------------------------Starting Load for repeat: {repeat}, condition: {condition}")

        coordinates, _, _, _, largest_region_list, binary_mask_list = run_R2(repeat, "DAPI", condition, outer_radius, mid_radius, inner_radius)

        # GET BOXES 
        image_boxes_path = f"boxes_npz/{repeat}/img_{marker}_{condition}.npz"

        img_boxes = load_boxes(image_boxes_path)

        # GET SELECTED IDS 
        selected_boxes_ids = load_allowed_ids(f'selection/{repeat}/img_DAPI_{condition}.csv')
        selected_boxes_ids.sort()
        
        # FILTER IMG_BOX to only contain selected ones
        filtered_img_boxes = [img_box for i, img_box in enumerate(img_boxes) if i+1 in selected_boxes_ids]

        # SET SAVING DIRECTORIES 
        intensity_means_output_path = f"distribution/{repeat}/{condition}_{marker}.csv"
        directory = os.path.dirname(intensity_means_output_path)
        os.makedirs(directory, exist_ok=True)

        print(f"------------------------Finished Load for repeat: {repeat}, condition: {condition}")


    # GET DISTRIBUTIONS OF DIAMETER, CIRCULARITY AND INTENSITIES
    if bGETDISTRIBUTIONS:
        # CALCULATE DIMAETER AND CIRCULARITY OF MODEL, USE TO GET INTENSITY 
        diameters, circularities = get_all_diameter(largest_region_list, binary_mask_list, filtered_img_boxes)
        intensity_means = get_all_intensities(filtered_img_boxes, coordinates, diameters)

        # SAVE TO CSV
        with open(intensity_means_output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'diameter', 'intensity', 'circularity']) 
            for i in range(len(diameters)):
                writer.writerow([i, diameters[i], intensity_means[i], circularities[i]])

        # PLOT DATA ON CSV AND SAVE 
        save_histograms(intensity_means_output_path)

# NORMALIZE MARKER INTENSITIES    
if bNORMALIZE:
    for marker in markers:

        # remove the useless diameter and circularity collumns (this is only relevat for DAPI)
        fix_marker_csv(intensity_means_output_path, intensity_means_output_path)
        
        # normalize intensity values by DAPI 
        normalize_by_dapi(intensity_means_output_path, f'distribution/{repeat}/{condition}.csv', intensity_means_output_path)

# PLOT OVERLAP GRAPH    
if bOVERLAPDENSITYPLOT:

    # FOR DAPI (not normalized)
    plot_wt_mutant_overlap(repeat, f'distribution/{repeat}/WT.csv', f'distribution/{repeat}/ND6.csv')

    # FOR ALL OTHER MARKERS (normalized)
    for marker in markers:
        plot_wt_mutant_overlap(marker, repeat, f'distribution/{repeat}/WT_{marker}.csv', f'distribution/{repeat}/ND6_{marker}.csv')

# PLOT DISTRIBUTION (SCATTER) PLOT WHERE Y AXIS IS INTENSITY IN A BIN, X AXIS IS DIAMETER
if bPLOTDISTRIBUTION_INTvsDIA:
    marker_loc_dict = {'SOX2': 'inner', 'BRA': 'mid', 'GATA3': 'outer'}
    
    # for marker, location in marker_loc_dict.items():
    #     plot_intensity_vs_diameter(
    #         repeat=repeat,
    #         csv1_path=f'intensities/meta_individual_{repeat}_{condition}.csv',
    #         csv2_path=f'distribution/{repeat}/{condition}.csv',
    #         marker=marker,
    #         location=location,
    #         output_path=f'plots/{repeat}/R{repeat}_{condition}_{marker}_{location}_diameterDis.png'
    #     )
    
    for marker, _ in marker_loc_dict.items():
        repeats_data = [
            (repeat, 'WT', f'intensities/meta_individual_{repeat}_WT.csv', f'distribution/{repeat}/WT.csv', marker),
            (repeat, 'ND6', f'intensities/meta_individual_{repeat}_ND6.csv', f'distribution/{repeat}/ND6.csv', marker),
        ]

        combined_plot_intensity_vs_diameter(
            repeats_data,
            marker_loc_dict
        )



# END TIMER
end_time = time.time()
print(f"\nScript completed in {end_time - start_time:.2f} seconds.")