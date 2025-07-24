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
    # lst = (np.arange(3000, 6000, 1000)).tolist()
    # for threshold in lst:
    #     print("starting threshold:", threshold)
    #     detect_blob_all(markers=["DAPI"], conditions=["WT"], repeat_no=[1], threshold=threshold)
    
    # threshold below is for small normalized box 
    # blw 3967 is blob
    # btwn 3967 and 4451
    # abov 4451 is blob
    detect_blob_all(markers=["DAPI"], conditions=["WT"], repeat_no=[1], threshold=5300)


# END TIMER
end_time = time.time()
print(f"\nScript completed in {end_time - start_time:.2f} seconds.")

# 5. Do a Final Re-check of list lengths
# check_list_size(all_coordinates, all_radii, img_boxes, mask_boxes)





