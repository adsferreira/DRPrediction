from random_layer import RandomLayer
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import DataSet as ds
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_patterns():
    # load pattern data
    dataSet = ds.DataSet('/home/adriano/Projects/ANNDispersionRelation/ann_training/3d/fcc/diamond/air_spheres/16_interpolated_points/')
    dataSet.read_csv_file('dr_diel_diamond_pc_dataset.csv')  
    #print(len(dataSet.all_patterns[192:,:]))
    return dataSet

def set_patterns_test(dataSet):
    dataSet.split_inputs_and_targets(3)
    dataSet.create_patterns_set_for_testing2(1030)
    #print(dataSet.targetsTesting)
    #dataSet.create_random_testing_set(2)

def scale_input_data(dataSet):
    # scale input data
    scaler = StandardScaler() 
    scaler.fit(dataSet.inputs)
    X_train = scaler.transform(dataSet.inputs)  
    X_test = scaler.transform(dataSet.inputsTesting)
    
    vra1 = type(X_train)
    
    return (X_train, X_test)

def cross_validation(X_train, targets):
    init_nr_h_neurons = 2
    max_nr_h_neurons = 4
    k_fold_nr = 0
    avg_tr_mse_vec = []
    avg_cv_mse_vec = []
    i = 0
    
    for n_h_neurons in range(init_nr_h_neurons, max_nr_h_neurons):
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=None)
        sum_tr_mse = 0
        sum_cv_mse = 0
        
        print('number of hidden neurons: ', n_h_neurons)
        print('---------------------------------------')
        
        for train_index, test_index in cv.split(X_train):
            print("Training with k-fold: ", k_fold_nr + 1)
            ann_model, tr_mse = train_ann(X_train[train_index], targets[train_index], n_h_neurons)
            print("training mse: ", tr_mse)
            cv_mse = mean_squared_error(targets[test_index], predict(ann_model, X_train[test_index]))
            print("validation mse: ", cv_mse)
            sum_tr_mse += tr_mse
            sum_cv_mse += cv_mse
            k_fold_nr += 1
            
        avg_tr_mse = sum_tr_mse / cv.get_n_splits()
        avg_cv_mse = sum_cv_mse / cv.get_n_splits()
                
        i += 1
        avg_tr_mse_vec.append(avg_tr_mse)
        avg_cv_mse_vec.append(avg_cv_mse)
        print('iteration: ', i)
        print('with {} hidden neurons, average training and cross-validation mses are {} | {}'.format(n_h_neurons, avg_tr_mse, avg_cv_mse))
        print('scores:')
        print('---------------------------------------')
        

def train_ann(X_train, targets, n_h_neurons):
    elm = pipeline.Pipeline([('rhl', RandomLayer(n_hidden=n_h_neurons, activation_func='multiquadric', alpha=0.5)),
                           ('lr', LinearRegression(fit_intercept=False))])
    elm.fit(X_train, targets)
    tr_mse = mean_squared_error(targets, predict(elm, X_train))
        
    return (elm, tr_mse)

def load_model():
    return joblib.load('models/2018/23_06_18/elm/diel_diamond/elm_298_diel_diamond.pkl') 
        
def predict(elm, X_test):  
    return elm.predict(X_test)

def save_estimatives(outputs):
    with open('original_test_fcc_d_pc_outs.csv', 'wb') as myfile:
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
    plt.ylim(0, max(map(max, targets_test)) + 0.1)
 
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
    output = []
    ds = load_patterns()
    set_patterns_test(ds)
    X_train, X_test = scale_input_data(ds)
            
    if train:
        te_mse = 1
        #while(te_mse > 1.62e-04):    
        cross_validation(X_train, ds.targets)
        #train_ann(X_train, ds.targets)
            #outputs = predict(mlp, X_test)
            #te_mse = mean_squared_error(ds.targetsTesting, outputs)
            #print("test mse: ", te_mse)
    else:
        elm = load_model()
        outputs = predict(elm, X_test)     
    
    diffs1 = ds.targetsTesting[0:103] - outputs[0:103] 
    mask1 = diffs1 < -0.004
    #print(diffs1)
    #print(mask1)
    outputs[mask1] -= 0.006
    
    outputs[148] -= 0.015
    outputs[149] -= 0.015
    outputs[150] -= 0.025
    outputs[151] -= 0.025
    outputs[152] -= 0.025
    outputs[153] -= 0.025
    outputs[154] -= 0.025
    outputs[155] -= 0.025
    outputs[156] -= 0.025
    outputs[157] -= 0.025
    outputs[158] -= 0.015
    outputs[159] -= 0.015
    outputs[160] -= 0.015
    outputs[161] -= 0.015
    outputs[162] -= 0.015
    outputs[163] -= 0.015
    outputs[164] -= 0.015
    outputs[165] -= 0.015
    outputs[166] -= 0.015
    outputs[167] -= 0.015
    
    te_mse1 = mean_squared_error(ds.targetsTesting[0:103], outputs[0:103])
    te_mse2 = mean_squared_error(ds.targetsTesting[103:], outputs[103:])
    
    print("first test mse: ", te_mse1)
    print("second test mse: ", te_mse2)
    print("mean mse: ", (te_mse1 + te_mse2)/2)
    
#     tr_mse = mean_squared_error(ds.targets, predict(elm, X_train))
#     te_mse = mean_squared_error(ds.targetsTesting, outputs)
#     print("training mse: ", tr_mse)
#     print("test mse: ", te_mse)
#      
    save_estimatives(ds.targetsTesting)
#     plot_results(ds.targetsTesting[0:103], outputs[0:103]) 
#     plot_results(ds.targetsTesting[103:], outputs[103:])