from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import DataSet as ds
import matplotlib.pyplot as plt
import numpy as np
import csv
import sklearn

def load_patterns():
    # load pattern data
    dataSet = ds.DataSet('/home/adriano/Projects/ANNDispersionRelation/ann_training/2d/square/te/tests_new_db/16_interpolated_points/')
    dataSet.read_csv_file('dr_te_pc_dataset.csv')  
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
    mlp = MLPRegressor(hidden_layer_sizes=(25,25,25), activation='tanh', solver='lbfgs', max_iter=1000, tol=1e-11)
    mlp.fit(X_train, targets)
    tr_mse = mean_squared_error(targets, predict(mlp, X_train))
    #print("n iterations: ", mlp.n_iter_)
    #print("training mse: ", tr_mse)
    # save model
    joblib.dump(mlp, 'models/12_01_18/mlp/te_pc/mlp_XX_XX_XX_te_pc.pkl')
    
    return (mlp, tr_mse)
    
def load_model():
    mlp = joblib.load('models/12_01_18/mlp/te_pc/mlp_25_25_25_te_pc.pkl') 
    return mlp
    
def predict(mlp, X_test):
    mlp_results = mlp.predict(X_test)
    #print(dataSet.targetsTesting)
    #print("\n")
    # ds.targetsTesting = ds.targetsTesting * 100
    #print(mlp_results)
    return mlp_results

def save_estimatives(outputs):
    with open('tests/test_mlp_te_pc_outs.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(outputs)


def plot_results(targets_test, mlp_results):
    k_index = np.arange(0, 52, 17)
    k_ticks = ['$\Gamma$', 'X', 'M', '$\Gamma$']
    solid_line = plt.plot(targets_test, "-")
    dotted_line = plt.plot(mlp_results, "--")
    plt.setp(solid_line, color='k', linewidth=2.0, markerfacecolor = "w", label = "TM (MPB)")
    plt.setp(dotted_line, color='b', linewidth=2.0, label = "TM (ANN)")
    plt.xlim(0, 51)
    plt.ylim(0, max(targets_test[0]) + 0.03)
 
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
    #print('The scikit-learn version is {}.'.format(sklearn.__version__))
    ds = load_patterns()
    set_patterns_test(ds)
    X_train, X_test = scale_input_data(ds)
    mlp, tr_mse = train_ann(X_train, ds.targets)
    #mlp = load_model()
    #strc = [coef.shape for coef in mlp.coefs_] 
    #print("architecture: ", strc)
    #print("nr of iterations: ", mlp.n_iter_)
    #tr_mse =  mean_squared_error(ds.targets, predict(mlp, X_train))
    #outputs = predict(mlp, X_test)
#     te_mse = mean_squared_error(ds.targetsTesting, outputs)
#     tr_outs = predict(mlp, X_train)
#     print("training mse: ", tr_mse)
#     print("test mse: ", te_mse)
#   save_estimatives(outputs)
#     plot_results(ds.targetsTesting[52:], outputs[52:])
#     plot_results(ds.targetsTesting[0:52], outputs[0:52])
