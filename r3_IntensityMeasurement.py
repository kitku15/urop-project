from f_intensityMeasurement import *
from f_modelDetection import *
from f_preprocessing import *
from f_validation import *

current_repeat = 1
current_condition = 'WT'
csv_path = 'radius.csv'
markers = ["DAPI", "SOX2", "BRA", "GATA3"]
levels = ["outer", "mid", "inner"]

# bools to trigger what to run
bGETAREA = False
bGETINTENSITIES = False
bNORMALIZE = False
bPLOT2 = True # combined plot

# Load DAPI coordinates, et set outer, mid and inner radius 
DAPI_coordinates, _, _, _, _, _ = load_DAPI()
outer_r, mid_r, inner_r = get_radius(csv_path, current_repeat, current_condition)

# get areas of each bin to normalize for plotting (is this necessary?)
if bGETAREA:
    outer_donut_area, middle_donut_area, inner_circle_area = donut_areas(outer_r, mid_r, inner_r)
    print("outer area:", outer_donut_area)
    print("middle area:", middle_donut_area)
    print("inner area:", inner_circle_area)


# get raw intensities for each marker and save it to the intensity folder 
if bGETINTENSITIES:
    for marker in markers:
        intensities_per_marker(marker, DAPI_coordinates, outer_r, mid_r, inner_r)


# normalize raw intensities by DAPI intensity and save 
if bNORMALIZE:
    for level in levels:
        for marker in markers:
            if marker != 'DAPI': 
                print("------------------------Now processing:")
                print("repeat:", current_repeat)
                print("condition:", current_condition)
                print("level:", level)
                print("marker:", marker)

                # get paths
                marker_path = f"intensities/{current_repeat}/{level}/{marker}_{current_condition}.npy"
                DAPI_path = f"intensities/{current_repeat}/{level}/DAPI_{current_condition}.npy"

                # load intensities 
                print(f"loading intensities for {marker} and DAPI")
                intensity_m = np.load(marker_path, allow_pickle=True)
                intensity_DAPI = np.load(DAPI_path, allow_pickle=True)

                # normalize intensity
                print(f"Normalizing intensities for {marker}")
                n_intensity = safe_normalize(intensity_m, intensity_DAPI)

                # save result
                n_intensity_path = f"intensities/{current_repeat}/{level}/norm_{marker}_{current_condition}.npy"
                directory = os.path.dirname(n_intensity_path)
                os.makedirs(directory, exist_ok=True)
                np.save(n_intensity_path, n_intensity)
                print("------------------------Results saved to:", n_intensity_path)


# optional: plot

    
if bPLOT2:
    marker_intensities = {}  # Key: marker, Value: [inner_bin, mid_bin, outer_bin]

    for marker in markers:
        if marker != 'DAPI': 
            
            inner_bin, mid_bin, outer_bin = load_levels(marker, current_repeat, current_condition)

            marker_intensities[marker] = [inner_bin, mid_bin, outer_bin]

    output = f"intensities/{current_repeat}/{current_condition}_all_markers.png"
    plot_all_markers(marker_intensities, output, norm_by_area=False)



# To do: save to csv 
    for marker in markers:
        if marker != 'DAPI':
            inner_bin, mid_bin, outer_bin = load_levels(marker, current_repeat, current_condition)

            avg_inner = np.nanmean(inner_bin)
            avg_mid = np.nanmean(mid_bin)
            avg_outer = np.nanmean(outer_bin)
            
            meta_intensities_save(current_repeat, current_condition, marker, avg_outer, avg_mid, avg_inner)