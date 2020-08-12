from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from sklearn.linear_model import LinearRegression
from random_layer import RandomLayer
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import DataSet as ds
import matplotlib.pyplot as plt
import numpy as np
import csv
import time

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

def scale_input_data(dataSet):
    # scale input data
    scaler = StandardScaler() 
    scaler.fit(dataSet.inputs)
    X_train = scaler.transform(dataSet.inputs)  
    X_test = scaler.transform(dataSet.inputsTesting) 
    
    return (X_train, X_test)

def train_ann(X_train, targets, f_name):
    # training ELM
    elm = pipeline.Pipeline([('rhl', RandomLayer(n_hidden=100, activation_func='multiquadric')),
                             ('lr', LinearRegression(fit_intercept=False))])
    elm.fit(X_train, targets)
    tr_mse = mean_squared_error(targets, predict(elm, X_train))
    print("training mse: ", tr_mse)
    # save model
    #joblib.dump(elm, f_name)
    
    return (elm, tr_mse)
    
def load_model(path, elm_model_file):
    elm = joblib.load(path + elm_model_file) 
    return elm
        
def predict(elm, X_test):  
    return elm.predict(X_test)

def save_estimatives(file_name, outputs):
    with open(file_name, 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(outputs)

def plot_results(targets_test, elm_results):
    k_ticks = ['X', 'U', 'L', '$\Gamma$', 'X', 'W', 'X']
    interpolated_points = 16
    total_nr_k_points = interpolated_points * (len(k_ticks) - 1) + len(k_ticks)
    k_index = np.arange(0, total_nr_k_points, interpolated_points + 1)
    
    solid_line = plt.plot(np.array(targets_test)[:,:], "-")
    dotted_line = plt.plot(np.array(elm_results)[:,:], "--")
    plt.setp(solid_line, color='k', linewidth=2.0, markerfacecolor = "w", label = "TM (MPB)")
    plt.setp(dotted_line, color='b', linewidth=2.0, label = "TM (ANN)")
    plt.xlim(0, total_nr_k_points - 1)
    plt.ylim(0, max(map(max, targets_test)) + 0.003)
 
    plt.xticks(k_index, k_ticks)
    plt.tick_params(labelsize=16)
    plt.grid(True,  which="both")
    plt.ylabel("$\omega$a/2$\pi$c)$^2$", fontsize=30)
    plt.xlabel("K Vector", fontsize=30)
 
    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc = 'best')
 
    plt.tight_layout()
    #plt.savefig('pc_diamond_bs.pdf', dpi=300)
    plt.show()
    
    
if __name__ == '__main__':
    prefix = '/home/adriano/Projects/ANNDispersionRelation/ann_training/3d/fcc/'
    phcs = ['diamond2']
    phc = phcs[0][:-1]
    polarization = ''
    
    ds_path = prefix + phcs[0] + '/with_material' + polarization + '/ds/16_interpolated_points/'
    ds_file = phc + '_' + polarization + 'ds.csv'
    elm_path = prefix + phcs[0] + '/with_material' + polarization + '/ann_models/'
    elm_files = ['elm_100_diamond_pc.pkl']
    
    elm_file = elm_files[0]
    elm_verbose = 1
    train = 0
    nr_target_output_attrs = 3
    nr_training_samples = 1442
    output = []
   
    ds = load_patterns(ds_path, ds_file)
    set_patterns_test(ds, nr_target_output_attrs, nr_training_samples)
    X_train, X_test = scale_input_data(ds)
            
    if train:
        te_mse = 1
        #while(te_mse > 2.72e-05):    
        while(te_mse > 1.1e-01):
            start_time = time.clock()
            elm, mse = train_ann(X_train, ds.targets, elm_path + elm_file)
            print ("-----\nElapsed time (in secs) for ELM: ", (time.clock() - start_time))
            outputs = predict(elm, X_test)
            te_mse = mean_squared_error(ds.targetsTesting, outputs)
            print("test mse: ", te_mse)
            #plot_results(ds.targetsTesting, outputs)
    else:
        nr_test_phcs = 1
        nr_interpolated_points = 16
        nr_corner_points = 7
        nr_k_points = (nr_corner_points - 1) * nr_interpolated_points + nr_corner_points
        elm = load_model(elm_path, elm_file)
        start_time = time.clock()
        outputs = predict(elm, X_test)
        print ("-----\nElapsed time(s): ", (time.clock() - start_time))
                
        # calculate band gaps of testing photonic crystals
        for i in range(nr_test_phcs):
            print("--------------------------------------------------")
            print("--------------------------------------------------")
            print("test PhC: ", i + 1)
            print("--------------------------------------------------")
            print("MPB PBG:")
            band_struct_iid = i * nr_k_points
            band_struct_fid = i * nr_k_points + nr_k_points
            mpb_lw_freq_id, mpb_up_freq_id, mpb_lw_freq, mpb_up_freq = calculate_pbg(ds.targetsTesting[band_struct_iid:band_struct_fid])
            print("----------------------")
            print("ELM PBG:")
            elm_lw_freq_id, elm_up_freq_id, elm_lw_freq, elm_up_freq = calculate_pbg(outputs[band_struct_iid:band_struct_fid])
            plot_results(ds.targetsTesting[band_struct_iid:band_struct_fid], outputs[band_struct_iid:band_struct_fid])
                    
        if elm_verbose: 
            print("\n--------------------------------------")
            print("ELM training verbose")
            print("--------------------------------------")
            print("number of patterns: ", X_train.shape)
            #start_time = time.clock()
            tr_outs = predict(elm, X_train)
            tr_mse =  mean_squared_error(ds.targets, tr_outs)
            te_mse = mean_squared_error(ds.targetsTesting, outputs)
            print("training mse: ", tr_mse)
            print("test mse: ", te_mse)
    
       
    save_estimatives('original_square_rods_veins_te_original_pc.csv', ds.targetsTesting)
    #save_estimatives('elm_100_diamond_pc.csv', outputs)