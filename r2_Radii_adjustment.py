from f_intensityMeasurement import *
from f_modelDetection import *
from f_preprocessing import *
from f_validation import *
from f_coordFinder import run_R2, adjust_binaryMask


# this is for adjustment
start_time = time.time() 


adjusting_values = {"DAPI": 100, # change these numbers (theyre just an example)
                     "BRA": 70,
                     "SOX2": 50 }



directory = "CHIP_REPEATS_NEW"
repeats = [1]
conditions = ["WT"]
markers = ["DAPI", "SOX2", "BRA", "GATA3"] # change this to your markers 


# make_grid_and_split_all(directory, markers, conditions, repeats)

# adjust_binaryMask(directory, repeats, conditions)
_, _, _, _, _, _ = run_R2(directory, repeats, conditions, adjusting_values, adjusting=True)

# END TIMER
end_time = time.time()
print(f"\nScript completed in {end_time - start_time:.2f} seconds.")

