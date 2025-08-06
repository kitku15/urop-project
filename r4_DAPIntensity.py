
from f_coordFinder import run_R2
from f_modelDetection import load_boxes, load_allowed_ids
from f_distributions import get_all_diameter, get_all_intensities, save_histograms, plot_wt_mutant_overlap
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
bOVERLAPDENSITYPLOT = False

# INFO
repeat = 1
condition = "ND6"
marker = "DAPI"

# GET COORDINATES AND MODEL REGION
if bLOAD:
    print(f"------------------------Starting Load for repeat: {repeat}, condition: {condition}")

    coordinates, _, _, _, largest_region_list, binary_mask_list = run_R2(repeat, marker, condition)

    # GET RAW RADIUS READINGS 
    image_boxes_path = f"boxes_npz/{repeat}/img_{marker}_{condition}.npz"

    img_boxes = load_boxes(image_boxes_path)

    # GET SELECTED IDS 
    selected_boxes_ids = load_allowed_ids(f'selection/{repeat}/img_DAPI_{condition}.csv')
    selected_boxes_ids.sort()
    
    # FILTER IMG_BOX to only contain selected ones
    filtered_img_boxes = [img_box for i, img_box in enumerate(img_boxes) if i+1 in selected_boxes_ids]

    # SET SAVING DIRECTORIES 
    intensity_means_output_path = f"distribution/{repeat}/{condition}.csv"
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

# PLOT OVERLAP GRAPH    
if bOVERLAPDENSITYPLOT:
    plot_wt_mutant_overlap('distribution/1/WT.csv', 'distribution/1/ND6.csv')



# END TIMER
end_time = time.time()
print(f"\nScript completed in {end_time - start_time:.2f} seconds.")