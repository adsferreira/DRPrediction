from sklearn import pipeline
from sklearn.linear_model import LinearRegression
from random_layer import RandomLayer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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
    # test1 = dataSet.inputs[20]
    # delete kz column
    dataSet.inputs = np.delete(dataSet.inputs, 8, 1)
    #dataSet.targets = np.delete(dataSet.targets, 5, 1)
    # test2 = dataSet.inputs[20]
    
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
    # training MLP
#     n_neurons = 16
#     mlp = MLPRegressor(hidden_layer_sizes=(n_neurons, n_neurons, n_neurons), activation='tanh', solver='lbfgs', tol=1e-12) #, alpha=1e-01
#     mlp.fit(X_train, targets)
#     tr_mse = mean_squared_error(targets, predict(mlp, X_train))
#     print("n iterations: ", mlp.n_iter_)
#     print("training mse: ", tr_mse)
#     # save model
#     joblib.dump(mlp, f_name)
#     
#     return (mlp, tr_mse)
    elm = pipeline.Pipeline([('rhl', RandomLayer(n_hidden=256, activation_func='multiquadric', alpha=0.5, rbf_width=0.3)),
                             ('lr', LinearRegression(fit_intercept=True))])
    elm.fit(X_train, targets)
    tr_mse = mean_squared_error(targets, predict(elm, X_train))
    print("training mse: ", tr_mse)
    # save model
    joblib.dump(elm, f_name)
    
    return (elm, tr_mse)
    
def load_model(path, mlp_model_file):
    mlp = joblib.load(path + mlp_model_file) 
    return mlp
        
def predict(mlp, X_test):  
    return mlp.predict(X_test)

def save_estimatives(file_name, outputs):
    with open(file_name, 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(outputs)

def plot_results(targets_test, mlp_results):
    k_ticks = ['$\Gamma$', 'M', 'K', '$\Gamma$']
    interpolated_points = 16
    total_nr_k_points = interpolated_points * (len(k_ticks) - 1) + len(k_ticks)
    k_index = np.arange(0, total_nr_k_points, interpolated_points + 1)
    
    solid_line = plt.plot(np.array(targets_test)[:,:], "-")
    dotted_line = plt.plot(np.array(mlp_results)[:,:], "--")
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
    
def concat_ds(ds1, ds2):
    return np.concatenate((ds1, ds2), axis = 0)

def euclid_dist(Ab):
    distances = []
                
    k = 1   
                            
    while k < (len(Ab) / 52):      
        #Ab_0 = Ab[0]
        if k == 25:
            Ab_k = Ab[k * 52] 
        
        sum_square = np.sum((Ab[0] - Ab[k * 52]) ** 2)                                 
        # compute euclidian distance    
        eucl_dist = np.sqrt(sum_square)
        distances.append(eucl_dist)
        print("-------\nEuclidean distance between cell " + str(0) + " e " + str(k) + ": " + str(eucl_dist) + "\n-------")
          
        k = k + 1
            
    print('distances: ', distances)
    max_dist = np.max(np.array(distances))
            
    return max_dist
    
    
if __name__ == '__main__':
    prefix = '/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/square/'
    phcs = ['two_diel_rods']
    phc = phcs[0]
    polarization = 'tm'
    
    ds_path = prefix + phc + '/non_linear/no_material/' + polarization + '/ds/16_interpolated_points/'
    ds_file = phc + '_' + polarization + '_parent_0_beta_50.csv'
    mlp_path = prefix + phc + '/non_linear/no_material/' + polarization + '/ann_model/'
    mlp_files = ['elm_256_' + phc + '_pc_0_beta_50_last.pkl']
    
    mlp_file = mlp_files[0]
    elm_verbose = 1
    nr_target_output_attrs = 5
    nr_training_samples = 11960
    output = []
   
    ds = load_patterns(ds_path, ds_file)
    set_patterns_test(ds, nr_target_output_attrs, nr_training_samples)
    
    
#     vara1 = ds.inputs[:52,:6]
#     vara2 = ds.inputsTesting[:,:6]
#     
#     s_vara1 = vara1.shape
#     s_vara2 = vara2.shape
#     
#     geo_inputs = concat_ds(ds.inputs[:52,:6], ds.inputsTesting[:,:6])
#     
#     s_geo_inputs = geo_inputs.shape
#     
#     max_dist = euclid_dist(geo_inputs)
    
    X_train, X_test = scale_input_data(ds)
    
    train = 0
            
    if train:
        te_mse = 1
        #while(te_mse > 2.72e-05):    
        while(te_mse > 5e-06):
            start_time = time.clock()
            mlp, mse = train_ann(X_train, ds.targets, mlp_path + mlp_file)
            print ("-----\nElapsed time (in secs) for MLP: ", (time.clock() - start_time))
            outputs = predict(mlp, X_test)
            te_mse = mean_squared_error(ds.targetsTesting, outputs)
            print("test mse: ", te_mse)
            plot_results(ds.targetsTesting[1248:1300], outputs[1248:1300])
    else:
        nr_test_phcs = 25
        nr_interpolated_points = 16
        nr_corner_points = 4
        nr_k_points = (nr_corner_points - 1) * nr_interpolated_points + nr_corner_points
        elm = load_model(mlp_path, mlp_file)
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
            print("MLP PBG:")
            elm_lw_freq_id, mlp_up_freq_id, mlp_lw_freq, mlp_up_freq = calculate_pbg(outputs[band_struct_iid:band_struct_fid])
            
            if i == 24:
                phc_test = ds.inputsTesting[band_struct_iid]
                print("\ninput test: ", phc_test)
                #plot_results(ds.targetsTesting[band_struct_iid:band_struct_fid], outputs[band_struct_iid:band_struct_fid])
                #save_estimatives(ds_path + 'original_two_diel_rods_tm.csv', ds.targets[0:52])
                #save_estimatives(ds_path + 'test_two_diel_rods_tm.csv', ds.targetsTesting[band_struct_iid:band_struct_fid])
                #save_estimatives(ds_path + 'elm_256_two_diel_rods_tm.csv', outputs[band_struct_iid:band_struct_fid])
                    
        if elm_verbose:
            #strc = [coef.shape[0] for coef in elm.coefs_] 
            print("\n--------------------------------------")
            print("ELM training verbose")
            print("--------------------------------------")
            print("number of patterns: ", X_train.shape)
            #print("architecture: ", strc)
            #print("number of output neurons: ", elm.n_outputs_)
            #print("output neurons' tranfer function: ", elm.out_activation_)
            #print("nr of iterations: ", elm.n_iter_)
            #start_time = time.clock()
            tr_outs = predict(elm, X_train)
            tr_mse =  mean_squared_error(ds.targets, tr_outs)
            te_mse = mean_squared_error(ds.targetsTesting, outputs)
            print("training mse: ", tr_mse)
            print("test mse: ", te_mse)
    
       
    