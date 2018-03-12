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

# training ELM
def train_ann(X_train, targets):
    elm = pipeline.Pipeline([('rhl', RandomLayer(n_hidden=85, activation_func='tanh', alpha=0.65)),
                            ('lr', LinearRegression(fit_intercept=False))])
    elm.fit(X_train, targets)
    tr_elm = mean_squared_error(targets, predict(elm, X_train))
    #print("training mse: ", tr_elm)
    # save model
    joblib.dump(elm, 'models/6_12_17/elm/te_pc/elm_XX_te_pc.pkl')
    
    return (elm, tr_elm)
    
def load_model():
    elm = joblib.load('models/6_12_17/elm/te_pc/elm_85_te_pc.pkl') 
    return elm

def predict(elm, X_test):
    return elm.predict(X_test)

def save_estimatives(outputs):
    with open('test_elm_te_pc_outs.csv', 'wb') as myfile:
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
    plt.ylim(0, max(targets_test[0]) + 0.03)
 
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
    output = []
    train = 1
    ds = load_patterns()
    set_patterns_test(ds)
    X_train, X_test = scale_input_data(ds)
        
    if train:
        #te_mse = 1
        #while(te_mse > 2e-6):    
        elm, mse = train_ann(X_train, ds.targets)
        #    outputs = predict(elm, X_test)
        #    te_mse = mean_squared_error(ds.targetsTesting, outputs)
        #    print("test mse: ", te_mse)
            #save_estimatives(outputs)
    else:
        elm = load_model()
        outputs = predict(elm, X_test)
#         tr_mse = mean_squared_error(ds.targets, predict(elm, X_train))
#         te_mse = mean_squared_error(ds.targetsTesting, outputs)
#         print("training mse: ", tr_mse)
#         print("test mse: ", te_mse)
#         
#     save_estimatives(outputs)    
#     plot_results(ds.targetsTesting[52:], outputs[52:])
#     plot_results(ds.targetsTesting[0:52], outputs[0:52])
