
from f_coordFinder import run_R2, get_coordinates
from f_modelDetection import load_boxes


# This part will get the distribution of the Model's 
# - Radius (based on DAPI)
# - All marker including DAPI Intensities
# - Circularity (not priority)


# bools to trigger what to run
bLOAD = True



repeat = 1
condition = "WT"
marker = "DAPI"

# GET COORDINATES AND MODEL REGION
if bLOAD:
    coordinates, _, _, _, largest_region = run_R2(1, "DAPI", "WT")

# GET RAW RADIUS READINGS 
mask_boxes_path = f"boxes_npz/{repeat}/mask_{marker}_{condition}.npz"
image_boxes_path = f"boxes_npz/{repeat}/img_{marker}_{condition}.npz"

print(f"------------------------Starting Load images boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")
img_boxes = load_boxes(image_boxes_path)
print(f"------------------------Finished Load images boxes for repeat: {repeat}, condition: {condition}, marker: {marker}")

# CALCULATE INTENSITY WITHIN MODEL REGION 




# CALCULATE CIRCULARITY AND RADIUS 