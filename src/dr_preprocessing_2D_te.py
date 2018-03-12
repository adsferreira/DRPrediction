import os
import numpy as np
import csv
    
def get_current_bigger_center(prop, original_center):
    return prop * original_center
            
prefix = "/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/square/"
folder = prefix + "te/tests_new_db/16_interpolated_points/"
dr_files = []
dataset = []

original_bigger_center = 0.113

dr_files += [dr_file for dr_file in os.listdir(folder) if dr_file.endswith('.dat')]
dr_files.sort()

for f_name in dr_files:
    # get the proportion value from file name
    filename, file_extension = os.path.splitext(f_name)
    proportion_idx = filename.index('p')
    proportion_str = filename[proportion_idx + 1:]
    print("proportion: ", proportion_str)
    proportional_center = get_current_bigger_center(float(proportion_str), original_bigger_center)
    print(proportional_center)
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
        param = np.append(proportional_center, k_vec)
        pattern = np.concatenate([param, mode])
        dataset.append(pattern)
                 
# vra = len(dataset)
      
with open(folder + 'dr_te_pc_dataset.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(dataset)