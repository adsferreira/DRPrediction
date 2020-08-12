import DataSet as ds
import csv

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

if __name__ == '__main__':
    prefix_fc = '/home/adriano/Projects/ANNDispersionRelation/ann_training/3d/fcc/'
    prefix_sc = '/home/adriano/Projects/ANNDispersionRelation/ann_training/3d/sc/'
    prefix_sq = '/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/square/'
    prefix_tr = '/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/triangular/'
    phcs = ['diamond3', 'spheres_cylindrical_veins', 'dielectric_veins', 'square_rods_veins', 'tri_diel_rods', 'diamond', 'sc5']
    phc = phcs[6]#[:-1]
    polarization = ''
    
    ds_path = prefix_sc + phc + polarization + '/ds/16_interpolated_points/'
    #ds_path = prefix + phcs[0] + '/with_material' + polarization + '/ds/16_interpolated_points/'
    ds_file = phc + '_' + polarization + 'pc_ds.csv'
            
    nr_target_output_attrs = 6
    nr_training_samples = 621
    output = []
   
    ds = load_patterns(ds_path, ds_file)
    set_patterns_test(ds, nr_target_output_attrs, nr_training_samples)
    
    
    with open('original_3D_SC_HolSph_PhC_DR.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        wr.writerows(ds.targets[345:414])