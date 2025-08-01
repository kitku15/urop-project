from f_intensityMeasurement import *
from f_modelDetection import *
from f_preprocessing import *
from f_validation import *
from meta_analysis import calculate_fp_fn


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

# bools to trigger what to run
bPREPROCESSING_1 = False
bPREPROCESSING_2 = False
bBLOB_DECETION_1 = False
bBLOB_DECETION_2 = True

# PREPROCESSING 1: SPLIT CZI INTO TIFFS
if bPREPROCESSING_1:
    czi_to_tiffs('CHIP_REPEATS')

# PREPROCESSING 2: CROP IMAGE AND MASK, SAVE
if bPREPROCESSING_2:
    crop_all(directory, cropping_csv_path, conditions, repeat_no, markers)
    print("finished cropping all!")

# BLOB DETECTION 1. RELOAD CROPPED IMAGES, MAKE GRID MASK, SPLIT MASK AND IMAGE INTO BOXES BASED ON GRID
if bBLOB_DECETION_1:
    make_grid_and_split_all(markers, conditions, repeat_no)

# BLOB DETECTION 2: BLOB DETECTION IN DAPI MASK BOX AND SAVE
if bBLOB_DECETION_2:

    # Define the parameters
    F1_LVThresh = 400000
    F2_sigma = 10
    F2_binaryThresh = 0.06
    F2_circThresh = 0.7

    # Iterate over every possible combination
    print("Laplaican Value Thresh:", F1_LVThresh)
    print("Circularity Smoothness:", F2_sigma)
    print("Binary Mask Thresh:", F2_binaryThresh)
    print("Circularity Thresh:", F2_circThresh)


    detect_blob_all(["DAPI"], ["WT"], [1], F1_LVThresh, F2_sigma, F2_binaryThresh, F2_circThresh)
    
    print("Finished Running Detection!")

    meta_path = f"metas/meta_{F1_LVThresh}_{F2_sigma}_{F2_binaryThresh}_{F2_circThresh}.csv"
    meta_analysis_path = "meta_analysis.csv"

    total_fp, total_fn = calculate_fp_fn(meta_path)
    score = 100 - ((total_fp+total_fn)/676 * 100)

    # The row you want to append (as a list)
    new_row = [F1_LVThresh, F2_sigma, F2_binaryThresh, F2_circThresh, total_fp, total_fn, score]

    # Open the CSV file in append mode ('a')
    with open(meta_analysis_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(new_row)
    
    print("Saved Score to:", meta_analysis_path)



# END TIMER
end_time = time.time()
print(f"\nScript completed in {end_time - start_time:.2f} seconds.")

# 5. Do a Final Re-check of list lengths
# check_list_size(all_coordinates, all_radii, img_boxes, mask_boxes)





