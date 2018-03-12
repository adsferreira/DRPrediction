import os
import numpy as np
import csv

prefix = "/home/adriano/Projects/ANNDispersionRelation/ann_training/second_case_study/"
folder = prefix + "diamond2/16_interpolated_k_points/"
dr_files = []
dataset = []
orig_radius = 0.1

dr_files += [dr_file for dr_file in os.listdir(folder) if dr_file.endswith('.dat')]
dr_files.sort()

for f_name in dr_files:
    filename, file_extension = os.path.splitext(f_name)
    proportion_idx = filename.index('p')
    proportion_str = filename[proportion_idx + 1:]
    proportional_radius = orig_radius * float(proportion_str)
    
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
        mode = np.array(mode, dtype=np.float)
        # pattern for data set
        param = np.append(proportional_radius, k_vec)
        # create current data pattern
        pattern = np.concatenate([param, mode])
        dataset.append(pattern)
        
with open(folder + 'dr_diamond_pc_dataset.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(dataset)