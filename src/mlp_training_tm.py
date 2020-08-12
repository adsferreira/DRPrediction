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
            print('lower frequency', round(modes_max_freqs[i], 3))       
            print('upper frequency', round(modes_min_freqs[i + 1], 3))
            print('absolute pbg:', round(abs_pbg, 3))
            print('central frequency', round(mid_gap_freq, 3))
            print('fraction gap size', round(frac_gap_size, 1))
            
            lw_freq_id.append(modes_min_freqs_ids[i])
            up_freq_id.append(modes_max_freqs_ids[i + 1])
            lw_freq.append(modes_max_freqs[i])
            up_freq.append(modes_min_freqs[i + 1])
    
    return(lw_freq_id, up_freq_id, lw_freq, up_freq)

def load_patterns():
    # load pattern data
    dataSet = ds.DataSet('/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/square/tm/16_interpolated_points/')
    dataSet.read_csv_file('dr_tm_pc_ds.csv')  
    #print(len(dataSet.all_patterns[192:,:]))

    return dataSet

def set_patterns_test(dataSet):
    dataSet.split_inputs_and_targets(3)
    dataSet.create_patterns_set_for_testing2(468)
    #print(dataSet.targetsTesting)
    #dataSet.create_random_testing_set(2)

def scale_input_data(dataSet):
    # scale input data
    scaler = StandardScaler() 
    scaler.fit(dataSet.inputs)
    X_train = scaler.transform(dataSet.inputs)  
    X_test = scaler.transform(dataSet.inputsTesting)
    
    return (X_train, X_test)

def train_ann(X_train, targets):
    # training MLP
    mlp = MLPRegressor(hidden_layer_sizes=(30,30,30), activation='tanh', solver='lbfgs', max_iter=1000, tol=1e-20)
    mlp.fit(X_train, targets)
    tr_mse = mean_squared_error(targets, predict(mlp, X_train))
    #print("n iterations: ", mlp.n_iter_)
    #print("training mse: ", tr_mse)
    # save model
    joblib.dump(mlp, 'models/6_12_17/mlp/tm_pc/mlp_tm_pc_prediction.pkl')
    
    return (mlp, tr_mse)

def get_loss_vc_epoch(X_train, targets):
    #mse = []
    mlp = MLPRegressor(hidden_layer_sizes=(10,10), activation='tanh', solver='adam', tol=1e-8, max_iter=360, warm_start=False)
#     for _ in range(1, 360):
    mlp.fit(X_train, targets)
#         output = mlp.predict(X_train)
#         mse.append(mean_squared_error(targets, output))
    
    print('final mse: ', mlp.loss_curve_[len(mlp.loss_curve_) - 1])    
    return mlp, mlp.loss_curve_

def load_model():
    mlp = joblib.load('models/2017/6_12_17/mlp/tm_pc/best_mlp_23_23_23_tm_pc.pkl') 
    return mlp

def predict(mlp, X_test):
    mlp_results = mlp.predict(X_test)
    #print(dataSet.targetsTesting)
    #print("\n")
    # ds.targetsTesting = ds.targetsTesting * 100
    #print(mlp_results)
    return mlp_results

def save_estimatives(outputs):
    with open('original_test_tm_pc_outs.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(outputs)
        
def save_loss_vs_epoch(loss_vec):
    with open('/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/square/tm/16_interpolated_points/loss_vec.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(loss_vec)

def plot_results(targets_test, mlp_results):
    k_index = np.arange(0, 52, 17)
    k_ticks = ['$\Gamma$', 'X', 'M', '$\Gamma$']
    solid_line = plt.plot(targets_test, "-")
    dotted_line = plt.plot(mlp_results, "--")
    plt.setp(solid_line, color='k', linewidth=2.0, markerfacecolor = "w", label = "TM (MPB)")
    plt.setp(dotted_line, color='b', linewidth=2.0, label = "TM (ANN)")
    plt.xlim(0, 51)
    plt.ylim(0, 1.0)
 
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
    #plt.savefig('pc_sc52_bs.pdf', dpi=300)
    plt.show()
    
if __name__ == '__main__':
    ds = load_patterns()
    set_patterns_test(ds)
    X_train, X_test = scale_input_data(ds)
    
    train = 0
    mlp_verbose = 0
    
    if train:
        te_mse = 1
        #mlp, tr_mse = train_ann(X_train, ds.targets)
        #mlp, mse_vec = get_loss_vc_epoch(X_train, ds.targets)
        #vra = type(mse_vec)
        #save_loss_vs_epoch(mse_vec)
        #plt.plot(mse_vec)
        #plt.show()
    else:
        nr_test_phcs = 2
        nr_interpolated_points = 16
        nr_corner_points = 4
        nr_k_points = (nr_corner_points - 1) * nr_interpolated_points + nr_corner_points
        mlp = mlp = load_model()
        start_time = time.clock()
        outputs = predict(mlp, X_test)
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
            #plot_results(ds.targetsTesting[band_struct_iid:band_struct_fid], outputs[band_struct_iid:band_struct_fid])
                    
        if mlp_verbose:
            strc = [coef.shape[0] for coef in mlp.coefs_] 
            print("\n--------------------------------------")
            print("MLP training verbose")
            print("--------------------------------------")
            print("number of patterns: ", X_train.shape)
            print("architecture: ", strc)
            print("number of output neurons: ", mlp.n_outputs_)
            #print("output neurons' tranfer function: ", mlp.out_activation_)
            print("nr of iterations: ", mlp.n_iter_)
            #start_time = time.clock()
            tr_outs = predict(mlp, X_train)
            tr_mse =  mean_squared_error(ds.targets, tr_outs)
            te_mse = mean_squared_error(ds.targetsTesting[:52], outputs[:52])
            print("training mse: ", tr_mse)
            print("test mse: ", te_mse)
        #mlp = load_model()
        #    strc = [coef.shape for coef in mlp.coefs_] 
        #    print("architecture: ", strc)
        #    print("nr of iterations: ", mlp.n_iter_)
        #outputs = predict(mlp, X_test)
        #     tr_mse = mean_squared_error(ds.targets, predict(mlp, X_train))
        #te_mse1 = mean_squared_error(ds.targetsTesting[0:52], outputs[0:52])
        #te_mse2 = mean_squared_error(ds.targetsTesting[52:], outputs[52:])
    
        #print("first test mse: ", te_mse1)
        #print("second test mse: ", te_mse2)
        #print("mean mse: ", (te_mse1 + te_mse2)/2)
#    save_estimatives(ds.targetsTesting)
#     max_freq_pc_1 = max(map(max, ds.targetsTesting[0:52]))
#     max_freq_pc_2 = max(map(max, ds.targetsTesting[52:]))
#    plot_results(ds.targetsTesting[52:], outputs[52:])
#    plot_results(ds.targetsTesting[0:52], outputs[0:52])