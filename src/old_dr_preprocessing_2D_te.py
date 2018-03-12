import os
import numpy as np
import csv

def get_current_radii(proportion, original_radii):
    return (round(original_radii[0] * proportion, 3), round(original_radii[1] * proportion, 3)) 
    
def get_current_centers(phc_id, prop, original_centers):
    current_proportion = 2.5 - (phc_id * 0.3)
    current_centers = original_centers * current_proportion
        
    # exclusive for proportion 0.48, which is a test crystal
    if (phc_id > 8) and (prop < 0.5):
        current_proportion = 2.5 - (-1 * 0.06)
        current_centers = original_centers * current_proportion # decay proportion for TE-PC with p = 0.48
    if (phc_id > 9) and (prop > 1.3):
        current_proportion = 2.5 - ((8 * 0.3) + 0.06) 
        current_centers = original_centers * current_proportion # decay proportion for TE-PC with p = 1.32
    
    return current_centers
        
def get_current_block_dims(phc_id, initial_air_b_len, initial_air_s_len):
    current_air_b_len = initial_air_b_len + (phc_id * 0.0156) # rectangle biggest length
    current_air_s_len = initial_air_s_len - (phc_id * 0.05) # rectangle smallest length
    
    if phc_id == 9: # pc with proportion equals to 0.48
        current_air_b_len = initial_air_b_len - 0.00312
        current_air_s_len = initial_air_s_len + 0.01
        
#    print(round(current_air_b_len, 4), round(current_air_s_len, 4))
    # blocks at the middle of squircle's edges: top, bottom, left, right
    lateral_block_b_len = current_air_s_len 
    lateral_block_s_len = (1 - current_air_b_len) / 2
    
#     if (lateral_block_b_len <= 0) or (lateral_block_s_len <= 0):
#         lateral_block_b_len = 0.
#         lateral_block_s_len = 0.
    
    return (lateral_block_b_len, lateral_block_s_len)

prefix = "/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/square/"
folder = prefix + "te/16_interpolated_points/"
dr_files = []
dataset = []

orginal_radii = np.array([0.467, 0.34])
original_centers = np.array([0.113, 0.0887])
initial_air_block_dims = np.array([0.79, 0.5])

dr_files += [dr_file for dr_file in os.listdir(folder) if dr_file.endswith('.dat')]
dr_files.sort()
phc_id = 0

for f_name in dr_files:
    # get the proportion value from file name
    filename, file_extension = os.path.splitext(f_name)
    proportion_idx = filename.index('p')
    proportion_str = filename[proportion_idx + 1:]
    print("proportion: ", proportion_str)
    # geometric parameters
    proportional_radii = get_current_radii(float(proportion_str), orginal_radii) 
    print(proportional_radii)
    proportional_centers = get_current_centers(phc_id, float(proportion_str), original_centers)
    print(proportional_centers)
    proportional_block_dims = get_current_block_dims(phc_id, initial_air_block_dims[0], initial_air_block_dims[1])
    print(proportional_block_dims)
    print("-------------")

    # read file content
    f = open(folder + f_name, 'r')
    fc = f.readlines()
    band_data = fc[1:len(fc)]
      
    for band_line in band_data:
        band_line = band_line[0:len(band_line) - 1]
        list_band_line = band_line.split(',')
        # k vector and respective magnitude
        k_vec = np.array(list_band_line[2:6], dtype=np.float)
        # mode vector
        mode = list_band_line[6:9]
        # squared mode vector
        # mode = np.array(mode, dtype=np.float) ** 2
        # create current data pattern
        pattern = np.concatenate([proportional_radii, proportional_centers, proportional_block_dims, k_vec, mode])
        dataset.append(pattern)
          
    phc_id += 1
          
# vra = len(dataset)
      
with open(folder + 'dr_te_pc_dataset.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(dataset)