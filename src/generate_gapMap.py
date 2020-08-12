import numpy as np
import csv
import DataSet as ds

def load_patterns(ds_path, ds_file):
    # load pattern data
    dataSet = ds.DataSet(ds_path)
    dataSet.read_csv_file(ds_file)  
    #print(len(dataSet.all_patterns[192:,:]))
    return dataSet

def set_patterns_test(dataSet, nr_target_output_attrs, nr_training_samples):
    dataSet.split_inputs_and_targets(nr_target_output_attrs)
    dataSet.create_patterns_set_for_testing2(nr_training_samples)
    #print(dataSet.targetsTesting)
    #dataSet.create_random_testing_set(2)
    
def calculate_pbg(bands):
    modes = bands.T
    #print(modes)
    modes_min_freqs = np.amin(modes, axis = 1)
    modes_max_freqs = np.amax(modes, axis = 1)
    modes_min_freqs_ids = np.argmax(modes, axis = 1)
    modes_max_freqs_ids = np.argmin(modes, axis = 1)
    nr_modes = modes.shape[0]
    lw_freq_id = []
    up_freq_id = []
    lw_freq = []
    up_freq = []
    nr_of_pbgs = 0
    
    for i in range(0, nr_modes - 1):
        if (modes_min_freqs[i + 1] - modes_max_freqs[i]) > 0:
            abs_pbg = modes_min_freqs[i + 1] - modes_max_freqs[i]
            lower_freq = modes_max_freqs[i]
            mid_gap_freq = lower_freq + (abs_pbg / 2)
            frac_gap_size = (abs_pbg / mid_gap_freq) * 100    
             
            nr_of_pbgs += 1 
            print('--------------------')
            print('PBG number: ', nr_of_pbgs)                
            print('lower frequency id', modes_min_freqs_ids[i])
            print('upper frequency id', modes_max_freqs_ids[i + 1])
            print('lower frequency', round(modes_max_freqs[i], 4))       
            print('upper frequency', round(modes_min_freqs[i + 1], 4))
            print('absolute pbg:', round(abs_pbg, 4))
            print('central frequency', round(mid_gap_freq, 4))
            print('fraction gap size', round(frac_gap_size, 2))
            
            lw_freq_id.append(modes_min_freqs_ids[i])
            up_freq_id.append(modes_max_freqs_ids[i + 1])
            lw_freq.append(modes_max_freqs[i])
            up_freq.append(modes_min_freqs[i + 1])
              
    return(lw_freq_id, up_freq_id, lw_freq, up_freq)

if __name__ == '__main__':
    
    prefix_fc = '/home/adriano/Projects/ANNDispersionRelation/ann_training/3d/fcc/'
    prefix_sc = '/home/adriano/Projects/ANNDispersionRelation/ann_training/3d/sc/'
    prefix_sq = '/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/square/'
    prefix_tr = '/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/triangular/'
    phcs = ['diamond3', 'spheres_cylindrical_veins', 'dielectric_veins', 'square_rods_veins', 'tri_diel_rods']
    phc = phcs[3]#[:-1]
    polarization = 'te'
    
    ds_path = prefix_sq + phc + '/with_material/' + polarization + '/ds/16_interpolated_points/'
    #ds_path = prefix + phcs[0] + '/with_material' + polarization + '/ds/16_interpolated_points/'
    ds_file = phc + '_' + polarization + '_ds.csv'
    lw_up_frequencies_file = ds_path + 'freqs_pbgs_e_11_56_teste.csv'
        
    nr_target_output_attrs = 2
    nr_training_samples = 2548
    output = []
   
    ds = load_patterns(ds_path, ds_file)
    set_patterns_test(ds, nr_target_output_attrs, nr_training_samples)
    
    nr_training_phcs = 7
    nr_interpolated_points = 16
    nr_corner_points = 4
    nr_k_points = (nr_corner_points - 1) * nr_interpolated_points + nr_corner_points
    pbgs_list = [[float(0), float(0)] for i in range(nr_training_phcs)]
    step = 7
            
    # calculate band gaps of testing photonic crystals
    for i in range(nr_training_phcs):
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print("test PhC: ", i + 1)
        print("--------------------------------------------------")
        print("MPB PBG:")
        band_struct_iid = i * step * nr_k_points
        band_struct_fid = i * step * nr_k_points + nr_k_points
        mpb_lw_freq_id, mpb_up_freq_id, mpb_lw_freq, mpb_up_freq = calculate_pbg(ds.targets[band_struct_iid:band_struct_fid])
        
        # if there is pbg in the current PhC, get it from the vectors mpb_lw_freq and mpb_up_freq
        # else, set to zero the lower and upper frequencies
        if (len(mpb_lw_freq) > 0):
            pbgs_list[i][0] = mpb_lw_freq[0]
            pbgs_list[i][1] = mpb_up_freq[0]
        else:
            pbgs_list[i][0] = 0.
            pbgs_list[i][1] = 0.
    
    with open(lw_up_frequencies_file, 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        wr.writerows(pbgs_list)
        