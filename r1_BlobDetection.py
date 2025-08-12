from f_intensityMeasurement import *
from f_modelDetection import *
from f_preprocessing import *
from f_validation import *

# START TIMER
start_time = time.time() 

# CONSTANT INFO:
# list of markers (index according to marker where 1:DAPI, 2:SOX2, 3:BRA, 4:GATA3)
markers = ["DAPI", "SOX2", "BRA", "GATA3"]

wt = "WT"
mutant = "ND6"
conditions = [wt]

channel_folders = {
        0: 'DAPI',
        1: 'SOX2',
        2: 'GATA3',
        3: 'BRA'
    }

repeats = [1]

directory = "CHIP_REPEATS"
cropping_csv_path = "cropping.csv"
blob_output_paths = "blob_output_paths.txt"

# bools to trigger what to run
bPREPROCESSING_1 = False
bPREPROCESSING_2 = False
bBLOB_DECETION_1 = True
bBLOB_DECETION_2 = False

# PREPROCESSING 1: SPLIT CZI INTO TIFFS
if bPREPROCESSING_1:
    # CONVERT RAW CZI FILE INTO A TIFF (channels still intact)
    convert_czi_to_tiff(directory)
    # NOW USER WILL USE IMAGEJ TO MANUALLY CROP AND ADJUST ANGLES (will make tutorial for this)
    # USER WILL ALSO MAKE MASKS FOR EACH CHANNEL AND SAVE THEM ACCORDING TO A SPECIFIC NAMING SYSTEM

    # SPLIT TIFFS INTO THE 4 CHANNELS WHICH IS SET BY CHANNEL_FOLDERS DICTIONARY ABOVE
    split_tiff_into_channels(directory, repeats, conditions, channel_folders)

# PREPROCESSING 2: CROP IMAGE AND MASK, SAVE
if bPREPROCESSING_2:

    # COMPILE ALL COORDINATES SEPERATELY SAVED FROM FIJI INTO THE MAIN CROPPING CSV
    for repeat in repeats:
        for condition in conditions:
            compile_crop_coordinates(repeat, condition, directory, cropping_csv_path)
    
    # AFTER THIS STEP USER WILL HAVE TO MANUALLY ADD ANGLES SET IN FIJI (IF ANY ARE SET)
    crop_masks(directory, cropping_csv_path, conditions, repeats, markers)
    print("finished cropping all masks!")

# BLOB DETECTION 1. RELOAD CROPPED IMAGES, MAKE GRID MASK, SPLIT MASK AND IMAGE INTO BOXES BASED ON GRID
if bBLOB_DECETION_1:
    make_grid_and_split_all(markers, conditions, repeats)

# BLOB DETECTION 2: BLOB DETECTION IN DAPI MASK BOX AND SAVE
if bBLOB_DECETION_2:
    detect_blob_all(markers=["DAPI"], conditions=["ND6"], repeat_no=[3])


# END TIMER
end_time = time.time()
print(f"\nScript completed in {end_time - start_time:.2f} seconds.")

# 5. Do a Final Re-check of list lengths
# check_list_size(all_coordinates, all_radii, img_boxes, mask_boxes)





