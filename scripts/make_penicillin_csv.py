import json
import csv

def main():
    with open('tests/data/penicillin.json', 'r') as f:
        data = json.load(f)
        
    X = data['inputs']['X']
    Zt = data['inputs']['Zt']
    y = data['inputs']['y']
    
    # Zt is a 30x144 matrix (24 plates + 6 samples)
    # The first 24 rows represent plates (a, b, c, ... x).
    # The last 6 rows represent samples (A, B, C, D, E, F).
    # 
    # For each observation (column j in Zt), we find which row i in 0..23 is 1 -> that's the plate.
    # We find which row i in 24..29 is 1 -> that's the sample.
    
    plates = [chr(ord('a') + i) for i in range(24)]
    samples = [chr(ord('A') + i) for i in range(6)]
    
    n_obs = len(y)
    
    with open('tests/data/penicillin.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['diameter', 'plate', 'sample'])
        
        for j in range(n_obs):
            # Find plate
            plate_idx = next(i for i in range(24) if Zt[i][j] == 1)
            plate_name = plates[plate_idx]
            
            # Find sample
            sample_idx = next(i for i in range(24, 30) if Zt[i][j] == 1) - 24
            sample_name = samples[sample_idx]
            
            writer.writerow([y[j], plate_name, sample_name])
            
    print("Created tests/data/penicillin.csv")

if __name__ == '__main__':
    main()
