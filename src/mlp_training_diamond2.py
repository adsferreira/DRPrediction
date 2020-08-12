from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import DataSet as ds
import matplotlib.pyplot as plt
import numpy as np
import csv
import time

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
    # training MLP
    mlp = MLPRegressor(hidden_layer_sizes=(16,16,16), activation='tanh', solver='lbfgs', max_iter=2000, tol=1e-20)
    mlp.fit(X_train, targets)
    tr_mse = mean_squared_error(targets, predict(mlp, X_train))
    print("n iterations: ", mlp.n_iter_)
    print("training mse: ", tr_mse)
    # save model
    joblib.dump(mlp, 'models/2018/31_07_18/mlp/mlp_XX_diamond3.pkl')
    
    return (mlp, tr_mse)

def get_loss_vc_epoch(X_train, targets):
    #mse = []
    mlp = MLPRegressor(hidden_layer_sizes=(21,21), activation='tanh', solver='adam', tol=1e-7, max_iter=1147, warm_start=False)
#     for _ in range(1, 360):
    mlp.fit(X_train, targets)
#         output = mlp.predict(X_train)
#         mse.append(mean_squared_error(targets, output))
    
    print('final mse: ', mlp.loss_curve_[len(mlp.loss_curve_) - 1])    
    return mlp, mlp.loss_curve_
    
def load_model():
    mlp = joblib.load('models/2018/31_07_18/mlp/mlp_16_16_16_diamond3.pkl') 
    return mlp
        
def predict(mlp, X_test):  
    return mlp.predict(X_test)

def save_estimatives(outputs):
    with open('original_test_diamond2_outs.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(outputs)
        
def save_loss_vs_epoch(loss_vec):
    with open('/home/adriano/Projects/ANNDispersionRelation/ann_training/3d/diamond/air_spheres/16_interpolated_points/loss_vec.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(loss_vec)

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
    plt.ylim(0, max(map(max, targets_test)) + 0.01)
 
    plt.xticks(k_index, k_ticks)
    plt.tick_params(labelsize=16)
    plt.grid(True,  which="both")
    plt.ylabel("$\omega$a/2$\pi$c)$^2$", fontsize=30)
    plt.xlabel("K Vector", fontsize=30)
 
    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys(), loc = 1)
 
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
        while(te_mse > 1.5e-01):    
            mlp, mse = train_ann(X_train, ds.targets)
            #mlp, mse_vec = get_loss_vc_epoch(X_train, ds.targets)
            #save_loss_vs_epoch(mse_vec)
            #plt.plot(mse_vec)
            #plt.show()
            outputs = predict(mlp, X_test)
            te_mse = mean_squared_error(ds.targetsTesting, outputs)
            print("test mse: ", te_mse)
    else:
        mlp = load_model()
#         print("nr of iterations: ", mlp.n_iter_)
        start_time = time.clock()
        tr_outs = predict(mlp, X_train)
        print ("-----\nElapsed time(s): ", (time.clock() - start_time))
#         tr_mse =  mean_squared_error(ds.targets, tr_outs)
#         outputs1 = predict(mlp, X_train[206:309])
#         outputs2 = predict(mlp, X_test)
#              
#         te_mse1 = mean_squared_error(ds.targets[206:309], outputs1)
#         te_mse2 = mean_squared_error(ds.targetsTesting[0:103], outputs2[0:103])
#     
#         print("first test mse: ", te_mse1)
#         print("second test mse: ", te_mse2)
        #print("mean mse: ", (te_mse1 + te_mse2)/2)
        
        #te_mse = mean_squared_error(ds.targetsTesting, outputs)
        #print("training mse: ", tr_mse)
        #print("test mse: ", te_mse)
        
#        save_estimatives(ds.targetsTesting)
#     max_freq_pc_1 = max(map(max, ds.targetsTesting[0:103]))
#     max_freq_pc_2 = max(map(max, ds.targetsTesting[103:]))
#plot_results(ds.targets[206:309], outputs1)
#plot_results(ds.targetsTesting[0:103], outputs2[0:103]) 