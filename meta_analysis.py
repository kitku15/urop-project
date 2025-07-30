import csv
import os

def calculate_fp_fn(csv_file_path):
    total_fp = 0.0
    total_fn = 0.0

    with open(csv_file_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                total_fp += float(row['FP_score'])
                total_fn += float(row['FN_score'])
            except ValueError as e:
                print(f"Skipping row due to error: {e}, row: {row}")

    score = 100 - ((total_fp+total_fn)/676 * 100)
    print(csv_file_path, score)

    return total_fp, total_fn

if __name__ == '__main__':
    directory = 'metas'
    paths = os.listdir(directory)

    for path in paths:
        calculate_fp_fn(f'{directory}/{path}') 
