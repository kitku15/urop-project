from f_intensityMeasurement import *
from f_modelDetection import *
from f_preprocessing import *
from f_validation import *
from f_coordFinder import run_R2

current_repeat = 1
current_condition = "ND6"
csv_path = 'radius.csv'
markers = ["DAPI", "SOX2", "BRA", "GATA3"]
levels = ["outer", "mid", "inner"]

# bools to trigger what to run
bLOAD = True
bGETINTENSITIES = True
bNORMALIZE = True

# GET COORDINATES AND ALL 3 RADIUS AFTER ADJUSTMENT 
if bLOAD:
    coordinates, outer_radius, mid_radius, inner_radius, _ = run_R2(1, "DAPI", current_condition)

# get raw intensities for each marker and save it to the intensity folder 
if bGETINTENSITIES:
    for marker in markers:
        intensities_per_marker(current_condition, marker, coordinates, outer_radius, mid_radius, inner_radius)


# normalize raw intensities by DAPI intensity and save 
if bNORMALIZE:
    for marker in markers:
        outer = 0
        mid = 0
        inner = 0
        individually_normalized_outer = []
        individually_normalized_mid = []
        individually_normalized_inner = []
    
        for level in levels:
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


                for i in range(len(intensity_m)):
                    m = intensity_m[i]
                    d = intensity_DAPI[i]

                    # if d == 0:
                    #     d = m
                    
                    print(f"{i}-----------------------")
                    print("marker", m)
                    print("DAPI norm", d)

                    n = m/d

                    print("normalized", n)


                    if level == "outer":
                        individually_normalized_outer.append(n)
                    elif level == "mid":
                        individually_normalized_mid.append(n)
                    else:
                        individually_normalized_inner.append(n)
                   
                    # print("-----------------------")



                # the average of all the models 
                avg_intensity_m = np.nanmean(intensity_m)
                avg_intensity_DAPI = np.nanmean(intensity_DAPI)

                # normalize (manually?)
                if level == "outer":
                    normalized_manually = np.nanmean(individually_normalized_outer)
                    outer = normalized_manually
                elif level == "mid":
                    normalized_manually = np.nanmean(individually_normalized_mid)
                    mid = normalized_manually
                else:
                    normalized_manually = np.nanmean(individually_normalized_inner)
                    inner = normalized_manually

                print("Manual mean ratio:", avg_intensity_m / avg_intensity_DAPI)
                print("Mean of ratios:", np.nanmean(intensity_m / intensity_DAPI))


                print(f"normalized manually for {level}:", normalized_manually)


        print(f"saving to csv for {marker}")
        meta_intensities_save(current_repeat, current_condition, marker, outer, mid, inner)

        for i in range(1, len(individually_normalized_outer)+1):
            meta_intensities_save_individual(i, current_repeat, current_condition, marker, individually_normalized_outer[i-1], individually_normalized_mid[i-1], individually_normalized_inner[i-1])


