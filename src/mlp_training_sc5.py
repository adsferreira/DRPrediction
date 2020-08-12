from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import DataSet as ds
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_patterns():
    # load pattern data
    dataSet = ds.DataSet('/home/adriano/Projects/ANNDispersionRelation/ann_training/3d/sc/sc5/16_interpolated_points/')
    dataSet.read_csv_file('dr_sc5_pc_dataset.csv')  
    #print(len(dataSet.all_patterns[192:,:]))
    return dataSet

def set_patterns_test(dataSet):
    dataSet.split_inputs_and_targets(6)
    dataSet.create_patterns_set_for_testing2(621)

def scale_input_data(dataSet):
    # scale input data
    scaler = StandardScaler() 
    scaler.fit(dataSet.inputs)
    X_train = scaler.transform(dataSet.inputs)  
    X_test = scaler.transform(dataSet.inputsTesting) 
    
    return (X_train, X_test)

def train_ann(X_train, targets):
    # training MLP
    mlp = MLPRegressor(hidden_layer_sizes=(32,32,32), tol=1e-14, activation='tanh', solver='lbfgs', max_iter=1000)
    mlp.fit(X_train, targets)    
    tr_mse = mean_squared_error(targets, predict(mlp, X_train))
    print("n iterations: ", mlp.n_iter_)
    print("training mse: ", tr_mse)
    # save model
    joblib.dump(mlp, 'models/6_12_17/mlp/sc5/mlp_XX_sc5_pc.pkl')
    
    return (mlp, tr_mse)

def get_loss_vc_epoch(X_train, targets):
    #mse = []
    mlp = MLPRegressor(hidden_layer_sizes=(9), activation='tanh', solver='adam', tol=1e-20, max_iter=716, warm_start=False)
#     for _ in range(1, 360):
    mlp.fit(X_train, targets)
#         output = mlp.predict(X_train)
#         mse.append(mean_squared_error(targets, output))
    
    print('final mse: ', mlp.loss_curve_[len(mlp.loss_curve_) - 1])    
    return mlp, mlp.loss_curve_

def load_model():
    mlp = joblib.load('models/2017/6_12_17/mlp/sc5/mlp_32_32_32_sc5_pc.pkl') 
    return mlp
        
def predict(mlp, X_test):  
    return mlp.predict(X_test)

def save_estimatives(outputs):
    with open('original_test_sc5_pc_outs.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(outputs)
        
def save_loss_vs_epoch(loss_vec):
    with open('/home/adriano/Projects/ANNDispersionRelation/ann_training/3d/sc5/16_interpolated_points/loss_vec.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(loss_vec)

def plot_results(targets_test, mlp_results):

    p = [0.07, 0.11, 0.15, 0.19, 0.23]

    for i in range(len(targets_test)):
        targets_test[i, 1:] = targets_test[i, 1:] + p
        mlp_results[i, 1:] = mlp_results[i, 1:] + p
    
    ymax = max(targets_test[0])
    k_ticks = ['$\Gamma$', 'R', 'X', 'M', '$\Gamma$'] 
    
    interpolated_points = 16
    total_nr_k_points = interpolated_points * (len(k_ticks) - 1) + len(k_ticks)
    k_index = np.arange(0, total_nr_k_points, interpolated_points + 1)
        
    plt.plot(targets_test, "-k", linewidth=5.0, alpha=0.2, label = "MPB")
    plt.plot(mlp_results, ':b', linewidth=5.0, label='MLP')
    #plt.setp(solid_line, "-k", linewidth=5.0, alpha=0.2, label = "MPB")
    #plt.setp(dotted_line, ':b', linewidth=4.2, label='MLP')
    
    plt.xlim(0, max(k_index))
    plt.ylim(0, ymax + 0.02)
    
    #ax.set_xticks(k_index_square_minor, minor=True) 
    plt.xticks(k_index, k_ticks)
    plt.tick_params(labelsize=16)
    plt.grid(which='major', axis='x', alpha=1.0) 
    plt.ylabel("$\omega$a/2$\pi$c)$^2$", fontsize=30)
    plt.xlabel("K Vector", fontsize=30)
 
    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
 
    plt.tight_layout()
    #plt.savefig('pc_diamond_bs.pdf', dpi=300)
    plt.show()
    
if __name__ == '__main__':
    train = 0
    outputs = []
    ds = load_patterns()
    set_patterns_test(ds)
    X_train, X_test = scale_input_data(ds)
            
    if train:
        te_mse = 1
        #while(te_mse > 1.18e-05):    
#         mlp, mse = train_ann(X_train, ds.targets)
#         outputs = predict(mlp, X_test)
#         te_mse = mean_squared_error(ds.targetsTesting, outputs)
#         print("test mse: ", te_mse)
        mlp, mse_vec = get_loss_vc_epoch(X_train, ds.targets)
        vra = type(mse_vec)
        save_loss_vs_epoch(mse_vec)
        plt.plot(mse_vec)
        plt.show()
    else:
        mlp = load_model()
        #print("nr of iterations: ", mlp.n_iter_)
        #tr_outs = predict(mlp, X_train)
        outputs = predict(mlp, X_test)
        te_mse1 = mean_squared_error(ds.targetsTesting[0:69], outputs[0:69])
        te_mse2 = mean_squared_error(ds.targetsTesting[69:], outputs[69:])
    
        print("first test mse: ", te_mse1)
        print("second test mse: ", te_mse2)
        print("mean mse: ", (te_mse1 + te_mse2)/2)
    
#     max_freq_1 = max(map(max, ds.targetsTesting[0:69]))
#     max_freq_2 = max(map(max, ds.targetsTesting[69:]))   
#     save_estimatives(ds.targetsTesting)  
#     plot_results(ds.targetsTesting[:69], outputs[:69]) 
#     plot_results(ds.targetsTesting[69:], outputs[69:])
