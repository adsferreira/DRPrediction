from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from random_layer import RandomLayer
import DataSet as ds
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_patterns():
    # load pattern data
    dataSet = ds.DataSet('/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/square/tm/16_interpolated_points/')
    dataSet.read_csv_file('dr_tm_pc_dataset.csv')  
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
    elm = pipeline.Pipeline([('rhl', RandomLayer(n_hidden = 80, activation_func='multiquadric', alpha=0.65)),
                           ('lr', LinearRegression(fit_intercept=False))])
    elm.fit(X_train, targets)
    tr_elm = mean_squared_error(targets, predict(elm, X_train))
    #print("training mse: ", tr_elm)
    # save model
    joblib.dump(elm, 'models/27_02_18/tm_pc/elm_XX_tm_pc.pkl')
    
    return (elm, tr_elm)

def load_model():
    elm = joblib.load('models/6_12_17/elm/tm_pc/elm_80_tm_pc.pkl') 
    return elm

def predict(elm, X_test):
    elm_results = elm.predict(X_test)
    #print(dataSet.targetsTesting)
    #print("\n")
    # ds.targetsTesting = ds.targetsTesting * 100
    #print(mlp_results)
    return elm_results

def save_estimatives(outputs):
    with open('test_elm_tm_pc_outs.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(outputs)

def plot_results(targets_test, outputs):
    k_index = np.arange(0, 52, 17)
    k_ticks = ['$\Gamma$', 'X', 'M', '$\Gamma$']
    solid_line = plt.plot(targets_test, "-")
    dotted_line = plt.plot(outputs, "--")
    plt.setp(solid_line, color='k', linewidth=2.0, markerfacecolor = "w", label = "TM (MPB)")
    plt.setp(dotted_line, color='b', linewidth=2.0, label = "TM (ANN)")
    plt.xlim(0, 51)
    plt.ylim(0, max(outputs[0]) + 0.02)
 
    plt.xticks(k_index, k_ticks)
    plt.tick_params(labelsize=16)
    plt.grid(True,  which="both")
    plt.ylabel("Normalized Frequency ($\omega$a/2$\pi$c)$^2$", fontsize=20)
    plt.xlabel("Wave Vector", fontsize=20)
 
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
#     elm, mse = train_ann(X_train, ds.targets)
    elm = load_model()
    outputs = predict(elm, X_test)
#     
#     total_k_points = 4
#     nr_pcs = 2
#     ii = 0
# 
#     for i in range(nr_pcs):
#         ie = (i + 1) * total_k_points
#         mpb_bands = ds.targetsTesting[ii:ie]
#         elm_bands = outputs[ii:ie]
#         ii = (i + 1) * total_k_points
#         
#         org_modes = np.array(mpb_bands).T
#         min_org_vec = np.amin(org_modes, axis=1)
#         max_org_vec = np.amax(org_modes, axis=1)
#         nr_modes = org_modes.shape[0]    
#         
#         elm_modes = np.array(elm_bands).T
#         min_elm_vec = np.amin(elm_modes, axis=1)
#         max_elm_vec = np.amax(elm_modes, axis=1)
#         
#         low_freqs_diff = min_org_vec[1:] - min_elm_vec[1:]
#         hig_freqs_diff = max_org_vec[:-1] - max_elm_vec[:-1]
#         
#         print("PC nr: ", i + 1)
#         
#         for j in range(0, nr_modes - 1):
#             if (min_org_vec[j + 1] - max_org_vec[j]) > 0:
#                 height_org = min_org_vec[j + 1] - max_org_vec[j]
#                 origin_y = max_org_vec[j]
#                 mid_gap_org_freq = origin_y + (height_org / 2)
#                 org_frac_gap_size = (height_org / mid_gap_org_freq) * 100 
#                 
#                 height_elm = min_elm_vec[j + 1] - max_elm_vec[j]
#                 origin_y = max_elm_vec[j]
#                 mid_gap_elm_freq = origin_y + (height_elm / 2)
#                 elm_frac_gap_size = (height_elm / mid_gap_elm_freq) * 100 
#                                 
#                 print("PBG nr ", j + 1)
#                 print("low frequency: ", min_org_vec[j + 1], "- upper frequency: ", max_org_vec[j])
#                 print("absolute gap: ", height_org)
#                 print("mid gap ratio ", mid_gap_org_freq)
#                 print("fractional gap: ", org_frac_gap_size)
#                 print("--")
#                 print("elm low frequency: ", min_elm_vec[j + 1], "- elm upper frequency: ", max_elm_vec[j])
#                 print("elm absolute gap: ", height_elm)
#                 print("elm mid gap ratio ", mid_gap_elm_freq)
#                 print("elm fractional gap: ", elm_frac_gap_size)
#                 print("-----------")
#                 
#         
#         print("-----------------------------------------------------")        
#                 
#         
# #     tr_mse = mean_squared_error(ds.targets, predict(elm, X_train))
# #     te_mse = mean_squared_error(ds.targetsTesting, outputs)    
# #     print("training mse: ", tr_mse)
# #     print("test mse: ", te_mse)
# #     save_estimatives(outputs)
    plot_results(ds.targetsTesting[52:], outputs[52:])
    plot_results(ds.targetsTesting[0:52], outputs[0:52])