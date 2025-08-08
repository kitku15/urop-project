from f_intensityMeasurement import *
from f_modelDetection import *
from f_preprocessing import *
from f_validation import *
from f_coordFinder import run_R2


# this is for adjustment
start_time = time.time() 


current_repeat = 3
current_condition = "WT"


# TO ADJUST 
outer_radius = 0
mid_radius = 0
inner_radius = 0


_, _, _, _, _, _ = run_R2(current_repeat, "DAPI", current_condition, outer_radius, mid_radius, inner_radius, adjusting=True)

# END TIMER
end_time = time.time()
print(f"\nScript completed in {end_time - start_time:.2f} seconds.")

