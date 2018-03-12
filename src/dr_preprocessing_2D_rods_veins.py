import os
import numpy as np
import csv

prefix = "/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/"
folder = prefix + "square/complete/tm/16_interpolated_points/"
dr_files = []
dataset = []
# parameters are rod's radius and vein's width, respectively
orginal_param = np.array([0.280, 0.042])

dr_files += [dr_file for dr_file in os.listdir(folder) if dr_file.endswith('.dat')]
dr_files.sort()

for f_name in dr_files:
    # get the proportion value from file name
    filename, file_extension = os.path.splitext(f_name)
    proportion_idx = filename.index('p', 6)
    proportion_str = filename[proportion_idx + 2:]
    proportional_param = orginal_param * float(proportion_str)
    print(proportion_str, proportional_param)
    
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
        mode = np.array(list_band_line[6:10], dtype=np.float)
        # squared mode vector is used in the original paper
#         mode = np.array(mode, dtype=np.float) ** 2
        # create current data pattern
        pattern = np.concatenate([proportional_param, k_vec, mode])
        dataset.append(pattern)
         
vra = len(dataset)
      
with open(folder + 'dr_tm_rods_veins_pc_dataset.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(dataset)