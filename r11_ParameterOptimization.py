from f_intensityMeasurement import *
from f_modelDetection import *
from f_preprocessing import *
from f_validation import *
import numpy as np
import os
import csv
from itertools import product
from multiprocessing import Pool, cpu_count
from threading import Lock
import time

# START TIMER
start_time = time.time() 

# constant stuff
markers = ["DAPI", "SOX2", "BRA", "GATA3"]
conditions = ["WT"]
repeat_no = [1]
directory = "CHIP_REPEATS"
cropping_csv_path = "cropping.csv"
blob_output_paths = "blob_output_paths.txt"

# Thread-safe file writing
csv_lock = Lock()
output_file = "coarse_grid_results.csv"
meta_dir = "metas"


def get_fp_fn_from_meta(filepath):
    total_fp = 0.0
    total_fn = 0.0
    try:
        with open(filepath, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                total_fp += float(row['FP_score'])
                total_fn += float(row['FN_score'])
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return total_fp, total_fn

def run_and_log(params):
    lv, sigma, b_thresh, c_thresh = params

    print(f"Running Detection for: LV={lv}, Sigma={sigma}, BinaryThresh={b_thresh}, CircThresh={c_thresh}")
    
    # Run the detection
    detect_blob_all(
        markers=["DAPI"],
        conditions=["WT"],
        repeat_no=[1],
        F1_LVThresh=lv,
        F2_sigma=sigma,
        F2_binaryThresh=b_thresh,
        F2_circThresh=c_thresh
    )

    # Construct expected meta file path
    meta_path = f"metas/meta_{lv}_{sigma}_{b_thresh}_{c_thresh}.csv"
    

    # Wait for the file to be written (optional: add timeout if needed)
    wait_time = 0
    while not os.path.exists(meta_path) and wait_time < 200:
        time.sleep(1)
        wait_time += 1

    # Read FP and FN from the meta file
    if os.path.exists(meta_path):
        fp, fn = get_fp_fn_from_meta(meta_path)
        row = [lv, sigma, b_thresh, c_thresh, fp, fn]
    else:
        print(f"Meta file not found: {meta_path}")
        row = [lv, sigma, b_thresh, c_thresh, None, None]

    # Save results to central CSV
    with csv_lock:
        write_header = not os.path.exists(output_file)
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["LV", "Sigma", "BinaryThresh", "CircThresh", "FP", "FN"])
            writer.writerow(row)


if __name__ == '__main__':

    # combination of parameters to try 
    F1_LVThresh = np.arange(300000, 500001, 50000)
    F2_sigma = np.arange(10, 31, 4)
    F2_binaryThresh = np.round(np.arange(0.050, 0.1001, 0.01), 3)
    F2_circThresh = np.round(np.arange(0.3, 0.91, 0.2), 2)

    param_combos = list(product(F1_LVThresh, F2_sigma, F2_binaryThresh, F2_circThresh))

    print(f"Total combinations to run: {len(param_combos)}")
    os.makedirs(meta_dir, exist_ok=True)  # Ensure meta folder exists

    with Pool(processes=cpu_count()) as pool:
        pool.map(run_and_log, param_combos)

    # END TIMER
    end_time = time.time()
    print(f"\nScript completed in {end_time - start_time:.2f} seconds.")