from random_layer import RandomLayer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import pipeline
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
    dataSet = ds.DataSet('/home/adriano/Projects/ANNDispersionRelation/ann_training/3d/fcc/diamond2/no_material/16_interpolated_points/')
    dataSet.read_csv_file('dr_diamond_pc_dataset.csv')  
    #print(len(dataSet.all_patterns[192:,:]))
    return dataSet

def set_patterns_test(dataSet):
    dataSet.split_inputs_and_targets(3)
    dataSet.create_patterns_set_for_testing2(927)
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
    elm = pipeline.Pipeline([('rhl', RandomLayer(n_hidden=200, activation_func='multiquadric', alpha=0.69)),
                           ('lr', LinearRegression(fit_intercept=False))])

    #elmr = GenELMRegressor( hidden_layer = rl )
    #elmr = ELMRegressor(n_hidden=98,random_state=0, alpha=0.8)
    elm.fit(X_train, targets)
    tr_mse = mean_squared_error(targets, predict(elm, X_train))
    print("training mse: ", tr_mse)
    # save model
    joblib.dump(elm, 'models/2018/23_06_18/elm/diel_diamond/elm_XX_diel_diamond.pkl')
    
    return (elm, tr_mse)

def load_model():
    #return joblib.load('models/2018/23_06_18/elm/diel_diamond/elm_200_alpha=0_68_multi_diel_diamond5.pkl') 
    return joblib.load('models/2017/6_12_17/elm/diamond_pc/elm_88_diamond_pc.pkl')
        
def predict(elm, X_test):  
    return elm.predict(X_test)

def save_estimatives(outputs):
    with open('test_elm_diamond_pc_outs.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(outputs)

def plot_results(targets_test, mlp_results):
    k_ticks = ['X', 'U', 'L', '$\Gamma$', 'X', 'W', 'X']
    interpolated_points = 16
    total_nr_k_points = interpolated_points * (len(k_ticks) - 1) + len(k_ticks)
    k_index = np.arange(0, total_nr_k_points, interpolated_points + 1)
    
    solid_line = plt.plot(targets_test, "-")
    dotted_line = plt.plot(mlp_results, "--")
    plt.setp(solid_line, color='k', linewidth=2.0, markerfacecolor = "w", label = "TM (MPB)")
    plt.setp(dotted_line, color='b', linewidth=2.0, label = "TM (ANN)")
    plt.xlim(0, total_nr_k_points - 1)
    plt.ylim(0, max(map(max, targets_test)) + 0.007)
 
    plt.xticks(k_index, k_ticks)
    plt.tick_params(labelsize=16)
    plt.grid(True,  which="both")
    plt.ylabel("$\omega$a/2$\pi$c)$^2$", fontsize=30)
    plt.xlabel("K Vector", fontsize=30)
 
    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc = 1)
 
    plt.tight_layout()
    #plt.savefig('pc_diamond_bs.pdf', dpi=300)
    plt.show()
    
    
if __name__ == '__main__':
    train = 0
    elm_verbose = 0
    output = []
    ds = load_patterns()
    set_patterns_test(ds)
    X_train, X_test = scale_input_data(ds)
            
    if train:
        te_mse = 1
        while(te_mse > 1.62e-04):    
            elm, mse = train_ann(X_train, ds.targets)
            outputs = predict(elm, X_test)
            te_mse = mean_squared_error(ds.targetsTesting, outputs)
            print("test mse: ", te_mse)
    else:
        nr_test_phcs = 2
        nr_interpolated_points = 16
        nr_corner_points = 7
        nr_k_points = (nr_corner_points - 1) * nr_interpolated_points + nr_corner_points
        elm = load_model()
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
#         tr_mse = mean_squared_error(ds.targets, predict(elm, X_train))
#         te_mse = mean_squared_error(ds.targetsTesting, outputs)
#         print("training mse: ", tr_mse)
#         print("test mse: ", te_mse)
        
    #save_estimatives(outputs)
    #plot_results(ds.targetsTesting[0:103], outputs[0:103]) 
    #plot_results(ds.targetsTesting[103:], outputs[103:])
